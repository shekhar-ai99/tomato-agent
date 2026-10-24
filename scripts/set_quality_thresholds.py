#!/usr/bin/env python3
"""
Stage 2 of the workflow (Section 4.6): quality features, degraded test conditions, quality thresholds.

1. For every eval set (pv_val, pv_test, pd_calib, pd_test) and every capture condition
   (clean + 5 degradations x levels x copies, Section 4.3) compute the capture-quality features.
   -> artifacts/quality_features/<set>.npz   (keys: paths, labels, F__<condition>)
2. Choose the quality agent's thresholds on validation photos only (PlantVillage val + the PlantDoc
   calibration hold-out by default; labels are not used), with a per-signal false-flag cap.
   -> artifacts/quality_thresholds.json
3. Report how well the thresholds detect each degradation on every set.
   -> artifacts/quality_detection.csv
Optional: --examples N saves N example photos per condition to artifacts/degraded_examples/ for the paper figure.

Usage:
  python scripts/set_quality_thresholds.py --pv-root PlantVillage-Dataset --pd-root PlantDoc-Dataset
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tomato_agent.classes import C2I
from tomato_agent.degrade import DEGRADATIONS, conditions, load_condition
from tomato_agent.io_utils import EVAL_SETS, add_data_args, eval_set, write_csv, write_json
from tomato_agent.quality import FEATURES, WORK_SIZE, compute_features, detection_report, select_thresholds


def _features_job(job):
    full, rel, conds = job
    out = []
    for name, deg, lv, cp in conds:
        f = compute_features(load_condition(full, rel, deg, lv, cp))
        out.append([f[k] for k in FEATURES])
    return out


def features_for_set(args, set_name, conds):
    out_path = Path(args.out) / "quality_features" / f"{set_name}.npz"
    names = [c[0] for c in conds]
    if out_path.exists() and not args.force:
        z = np.load(out_path, allow_pickle=True)
        if all(f"F__{n}" in z.files for n in names):
            print(f"[skip] {set_name}: features cached")
            return {n: z[f"F__{n}"] for n in names}, z
    root, rows = eval_set(args, set_name)
    jobs = [(str(root / r["path"]), r["path"], conds) for r in rows]
    print(f"{set_name}: {len(rows)} images x {len(conds)} conditions")
    if args.workers > 1:
        with Pool(args.workers) as pool:
            res = pool.map(_features_job, jobs, chunksize=16)
    else:
        res = [_features_job(j) for j in jobs]
    F = np.asarray(res, dtype=np.float64).reshape(len(rows), len(conds), len(FEATURES))
    arrays = {f"F__{n}": F[:, i, :] for i, n in enumerate(names)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, paths=np.array([r["path"] for r in rows]),
                        labels=np.array([C2I[r["label"]] for r in rows]), features=np.array(FEATURES), **arrays)
    return {n: arrays[f"F__{n}"] for n in names}, None


def save_examples(args, conds):
    from PIL import Image
    root, rows = eval_set(args, "pd_test")
    out = Path(args.out) / "degraded_examples"
    out.mkdir(parents=True, exist_ok=True)
    for r in rows[:args.examples]:
        stem = Path(r["path"]).stem[:40]
        for name, deg, lv, cp in conds:
            im = load_condition(root / r["path"], r["path"], deg, lv, cp)
            im.thumbnail((384, 384), Image.BILINEAR)
            im.save(out / f"{stem}__{name}.jpg", quality=90)
    print(f"Saved examples to {out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--sets", nargs="+", default=list(EVAL_SETS))
    ap.add_argument("--levels", nargs="+", type=int, default=[2], help="1 mild, 2 moderate, 3 severe")
    ap.add_argument("--copies", type=int, default=2, help="independent degraded copies (copy 1+ = pessimistic retakes)")
    ap.add_argument("--max-fpr", type=float, default=0.03, help="max share of clean photos one signal may flag, per set")
    ap.add_argument("--select-on", nargs="+", default=["pv_val", "pd_calib"],
                    help="sets whose photos (not labels) choose the thresholds; test sets are never used")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--examples", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    conds = conditions(tuple(args.levels), args.copies)
    feats = {s: features_for_set(args, s, conds)[0] for s in args.sets}

    missing = [s for s in args.select_on if s not in feats]
    if missing:
        sys.exit(f"--select-on sets {missing} must also be in --sets")
    sel = [feats[s] for s in args.select_on]
    degraded = {d: np.concatenate([v[n] for v in sel for n, dd, lv, cp in conds if dd == d and cp == 0])
                for d in DEGRADATIONS}
    thresholds, report = select_thresholds([v["none"] for v in sel], degraded, args.max_fpr)
    write_json(Path(args.out) / "quality_thresholds.json", {
        "thresholds": thresholds, "selection": report, "selected_on": args.select_on, "max_fpr": args.max_fpr,
        "levels": args.levels, "copies": args.copies, "work_size": WORK_SIZE, "features": FEATURES})
    print(f"\nQuality thresholds (chosen on {', '.join(args.select_on)}; images only, no labels):")
    for r in report:
        print(f"  {r['threshold']:16s} {r['value']:>10.4f}   TPR({r['tpr_on']}) = {r['tpr']:.3f}   "
              f"max FPR(clean) = {r['fpr_clean_max']:.3f}")

    det = []
    for s, f in feats.items():
        by_deg = {d: np.concatenate([f[n] for n, dd, lv, cp in conds if dd == d]) for d in DEGRADATIONS}
        for row in detection_report(f["none"], by_deg, thresholds):
            det.append({"set": s, **row})
    write_csv(Path(args.out) / "quality_detection.csv", det)
    print("\nFlag rate per condition (none = clean photos; lower is better there, higher is better elsewhere):")
    for r in det:
        print(f"  {r['set']:9s} {r['degradation']:14s} flagged {r['flag_rate']:.3f}")
    if args.examples:
        save_examples(args, conds)
    print(f"\nWrote {args.out}/quality_thresholds.json and quality_detection.csv")


if __name__ == "__main__":
    main()
