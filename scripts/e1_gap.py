#!/usr/bin/env python3
"""
Stage 5 / E1 (RQ1): the laboratory-to-field accuracy gap of the static models.

Reads metrics.json and saved logits of every run. Writes:
  results/e1_gap_per_run.csv     one row per backbone x seed
  results/e1_gap.csv / .md       mean +- SD over seeds per backbone (paper Table 5.1)
  results/e1_per_class_recall.csv
  results/confusion_<backbone>_pd_test.csv   summed over seeds (paper Figure 2)

Gate (Section 4.6): if the mean field drop is smaller than --min-gap, print a warning to reframe the
paper around calibration and deferral rather than domain shift.

Usage:  python scripts/e1_gap.py --runs runs
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tomato_agent.classes import CLASSES
from tomato_agent.io_utils import run_dirs, write_csv


def ms(xs):
    xs = np.asarray(xs, float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--min-gap", type=float, default=0.10)
    args = ap.parse_args()
    out = Path(args.results)

    per_run, recall_rows = [], []
    conf = defaultdict(lambda: np.zeros((len(CLASSES), len(CLASSES)), int))
    for run in run_dirs(args.runs):
        m = json.loads((run / "metrics.json").read_text())
        bb, seed = m["backbone"], m["seed"]
        per_run.append({"backbone": bb, "seed": seed, "pv_test_acc": m["pv_test"]["accuracy"],
                        "pd_test_acc": m["pd_test"]["accuracy"], "gap": m["field_gap_accuracy"],
                        "pv_test_macro_f1": m["pv_test"]["macro_f1"], "pd_test_macro_f1": m["pd_test"]["macro_f1"]})
        for c, r in m["pd_test"]["per_class_recall"].items():
            recall_rows.append({"backbone": bb, "seed": seed, "set": "pd_test", "class": c, "recall": r})
        for c, r in m["pv_test"]["per_class_recall"].items():
            recall_rows.append({"backbone": bb, "seed": seed, "set": "pv_test", "class": c, "recall": r})
        z = np.load(run / "logits_pd_test.npz", allow_pickle=True)
        np.add.at(conf[bb], (z["labels"].astype(int), z["logits"].argmax(1)), 1)
    if not per_run:
        sys.exit("No finished runs (metrics.json) found")

    write_csv(out / "e1_gap_per_run.csv", per_run)
    write_csv(out / "e1_per_class_recall.csv", recall_rows)
    table, md = [], ["| Backbone | PlantVillage accuracy | PlantDoc accuracy | Gap | PlantDoc macro-F1 |",
                     "|---|---|---|---|---|"]
    for bb in sorted({r["backbone"] for r in per_run}):
        rs = [r for r in per_run if r["backbone"] == bb]
        row = {"backbone": bb, "n_seeds": len(rs)}
        for k in ("pv_test_acc", "pd_test_acc", "gap", "pd_test_macro_f1"):
            row[f"{k}_mean"], row[f"{k}_sd"] = ms([r[k] for r in rs])
        table.append(row)
        f = lambda k: f"{row[k + '_mean'] * 100:.1f} ± {row[k + '_sd'] * 100:.1f}"
        md.append(f"| {bb} | {f('pv_test_acc')} | {f('pd_test_acc')} | {f('gap')} | {f('pd_test_macro_f1')} |")
        cm = conf[bb]
        write_csv(out / f"confusion_{bb}_pd_test.csv",
                  [{"true": CLASSES[i], **{CLASSES[j]: int(cm[i, j]) for j in range(len(CLASSES))}} for i in range(len(CLASSES))])
    write_csv(out / "e1_gap.csv", table)
    (out / "e1_gap.md").write_text("Accuracy in %, mean ± SD over seeds.\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    worst = defaultdict(list)
    for r in recall_rows:
        if r["set"] == "pd_test":
            worst[r["class"]].append(r["recall"])
    print("\nPlantDoc recall per class (mean over all runs), hardest first:")
    for c, v in sorted(worst.items(), key=lambda kv: np.mean(kv[1])):
        print(f"  {c:24s} {np.mean(v):.3f}")

    gap = np.mean([r["gap"] for r in per_run])
    if gap < args.min_gap:
        print(f"\nGATE: mean field gap {gap:.3f} < {args.min_gap}. Reframe the paper around calibration and "
              "deferral rather than domain shift before continuing (Section 4.6, stage 5).")
    else:
        print(f"\nGATE passed: mean field gap {gap:.3f} >= {args.min_gap}.")


if __name__ == "__main__":
    main()
