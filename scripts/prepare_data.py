#!/usr/bin/env python3
"""
Data preparation for the tomato crop-monitoring paper (Section 4.6, stages 1 and 2 of the workflow).

Step 1  manifests : label mapping, leaf-grouped PlantVillage split, PlantDoc calibration/test split.
Step 2  dedup     : perceptual-hash removal of training images that are near-copies of val/test images.

Expected folder layout (full clones of the two public repositories):
  <pv-root>/raw/color/Tomato___<class>/*.JPG
  <pv-root>/leaf_grouping/filtered_leafmaps/Tomato___<class>.csv
  <pv-root>/ ... github.com/spMohanty/PlantVillage-Dataset
  <pd-root>/train/<PlantDoc class>/*.jpg ,  <pd-root>/test/<PlantDoc class>/*.jpg
  <pd-root>/ ... github.com/pratikkayal/PlantDoc-Dataset

Usage:
  python prepare_data.py manifests --pv-root PlantVillage-Dataset --pd-root PlantDoc-Dataset --out manifests
  python prepare_data.py dedup --pv-root PlantVillage-Dataset --manifest manifests/pv_manifest.csv --out manifests

Dependencies: Python 3.9+. `dedup` also needs: pip install pillow numpy imagehash

Design (matches the paper):
  * 8 classes shared by PlantVillage and PlantDoc; target spot and spider mites are excluded.
  * PlantVillage images of the same physical leaf (4 per leaf) always land in the same split.
  * Images with no leaf ID (unmapped, including all mosaic virus) are training-only.
  * PlantDoc is never used for training: 20% per class is held out as a calibration split, the rest is the test split.
  * Splits are frozen by one seed and reused for every model seed.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

CLASSES = [
    "bacterial_spot", "early_blight", "late_blight", "leaf_mold",
    "septoria_leaf_spot", "yellow_leaf_curl_virus", "mosaic_virus", "healthy",
]

# PlantVillage folder -> canonical label (excluded: Target_Spot, Spider_mites)
PV_FOLDERS = {
    "Tomato___Bacterial_spot": "bacterial_spot",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl_virus",
    "Tomato___Tomato_mosaic_virus": "mosaic_virus",
    "Tomato___healthy": "healthy",
}

# PlantDoc folder -> canonical label.
# TO CONFIRM against sample images or the PlantDoc paper: "Tomato leaf" = healthy,
# "yellow virus" = yellow leaf curl virus, "mold leaf" = leaf mold.
PD_FOLDERS = {
    "Tomato leaf bacterial spot": "bacterial_spot",
    "Tomato Early blight leaf": "early_blight",
    "Tomato leaf late blight": "late_blight",
    "Tomato mold leaf": "leaf_mold",
    "Tomato Septoria leaf spot": "septoria_leaf_spot",
    "Tomato leaf yellow virus": "yellow_leaf_curl_virus",
    "Tomato leaf mosaic virus": "mosaic_virus",
    "Tomato leaf": "healthy",
}

IMG_EXT = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------------- helpers
def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_leaf_map(path):
    """filename (original PlantVillage name) -> leaf number, from the repo's filtered leaf maps."""
    out = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        next(rd, None)  # header
        for row in rd:
            if len(row) >= 2 and row[0].strip():
                out[row[0].strip()] = row[1].strip()
    return out


# ----------------------------------------------------------------------------- step 1
def build_plantvillage(root, seed, val_frac, test_frac):
    rows = []
    for folder, label in PV_FOLDERS.items():
        d = root / "raw" / "color" / folder
        if not d.is_dir():
            sys.exit(f"Missing folder: {d}")
        lmap = read_leaf_map(root / "leaf_grouping" / "filtered_leafmaps" / f"{folder}.csv")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            # raw file names look like "<uuid>___<original name>"; the leaf maps use the original name
            original = p.name.split("___", 1)[-1].strip()
            leaf = lmap.get(original)
            rows.append({
                "path": str(p.relative_to(root)),
                "label": label,
                "leaf_id": f"{label}:{leaf}" if leaf else "",
                "source": "mapped" if leaf else "unmapped",
            })

    # leaf-level split: all images of one leaf share a split
    rng = random.Random(seed)
    leaves_by_label = defaultdict(set)
    for r in rows:
        if r["leaf_id"]:
            leaves_by_label[r["label"]].add(r["leaf_id"])
    leaf_split = {}
    for label in CLASSES:
        leaves = sorted(leaves_by_label.get(label, ()))
        rng.shuffle(leaves)
        n = len(leaves)
        n_test, n_val = round(n * test_frac), round(n * val_frac)
        for i, leaf in enumerate(leaves):
            leaf_split[leaf] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    for r in rows:
        r["split"] = leaf_split.get(r["leaf_id"], "train")  # unmapped images are training-only
    return rows


def build_plantdoc(root, seed, calib_frac):
    rows = []
    for sub in ("train", "test"):
        for folder, label in PD_FOLDERS.items():
            d = root / sub / folder
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    rows.append({"path": str(p.relative_to(root)), "label": label, "origin": sub})
    if not rows:
        sys.exit(f"No PlantDoc tomato images found under {root}")

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for i, r in enumerate(rows):
        by_label[r["label"]].append(i)
    for label in CLASSES:
        idx = by_label.get(label, [])
        rng.shuffle(idx)
        n_cal = round(len(idx) * calib_frac)
        for k, i in enumerate(idx):
            rows[i]["split"] = "calib" if k < n_cal else "test"
    return rows


def split_report(pv_rows, pd_rows):
    counts = Counter()
    for r in pv_rows:
        counts[("plantvillage", r["split"], r["label"])] += 1
    for r in pd_rows:
        counts[("plantdoc", r["split"], r["label"])] += 1
    rows = [{"dataset": d, "split": s, "label": l, "count": c} for (d, s, l), c in sorted(counts.items())]

    print(f"\n{'dataset':13s} {'split':6s} " + " ".join(f"{c[:9]:>9s}" for c in CLASSES) + "   total")
    for d, splits in (("plantvillage", ("train", "val", "test")), ("plantdoc", ("calib", "test"))):
        for s in splits:
            vals = [counts[(d, s, c)] for c in CLASSES]
            print(f"{d:13s} {s:6s} " + " ".join(f"{v:9d}" for v in vals) + f"   {sum(vals)}")
    for c in CLASSES:  # classes that cannot be evaluated in-domain
        for s in ("val", "test"):
            if counts[("plantvillage", s, c)] == 0:
                print(f"WARNING: no PlantVillage {s} images for '{c}' (no leaf map): it is training-only in-domain.")
    return rows


def cmd_manifests(a):
    out = Path(a.out)
    pv = build_plantvillage(Path(a.pv_root), a.seed, a.val_frac, a.test_frac)
    pd = build_plantdoc(Path(a.pd_root), a.seed, a.calib_frac)
    write_csv(out / "pv_manifest.csv", pv, ["path", "label", "leaf_id", "source", "split"])
    write_csv(out / "pd_manifest.csv", pd, ["path", "label", "origin", "split"])
    report = split_report(pv, pd)
    write_csv(out / "split_report.csv", report, ["dataset", "split", "label", "count"])
    meta = {
        "created": date.today().isoformat(), "seed": a.seed, "val_frac": a.val_frac,
        "test_frac": a.test_frac, "plantdoc_calib_frac": a.calib_frac, "classes": CLASSES,
        "plantvillage_label_map": PV_FOLDERS, "plantdoc_label_map": PD_FOLDERS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote manifests to {out}/ (pv_manifest.csv, pd_manifest.csv, split_report.csv, meta.json)")


# ----------------------------------------------------------------------------- step 2
def hamming_min(train_hashes, eval_hashes):
    """For each train hash, the minimum Hamming distance to any eval hash (64-bit hashes)."""
    import numpy as np
    ev = np.asarray(eval_hashes, dtype=np.uint64)
    best, arg = [], []
    for h in train_hashes:
        x = (ev ^ np.uint64(h)).view(np.uint8).reshape(-1, 8)
        d = np.unpackbits(x, axis=1).sum(axis=1)
        j = int(d.argmin())
        best.append(int(d[j]))
        arg.append(j)
    return best, arg


def compute_hashes(root, paths, cache_path):
    from PIL import Image
    import imagehash
    cache = {}
    if cache_path.exists():
        cache = {r["path"]: r["hash"] for r in read_csv(cache_path)}
    todo = [p for p in paths if p not in cache]
    for k, p in enumerate(todo, 1):
        with Image.open(root / p) as im:
            cache[p] = str(imagehash.phash(im.convert("RGB"), hash_size=8))  # 64-bit hash
        if k % 1000 == 0:
            print(f"  hashed {k}/{len(todo)}")
    write_csv(cache_path, [{"path": p, "hash": h} for p, h in cache.items()], ["path", "hash"])
    return {p: int(cache[p], 16) for p in paths}


def cmd_dedup(a):
    root, out = Path(a.pv_root), Path(a.out)
    rows = read_csv(a.manifest)
    hashes = compute_hashes(root, [r["path"] for r in rows], out / "pv_phash_cache.csv")

    eval_rows = [r for r in rows if r["split"] in ("val", "test")]
    train_rows = [r for r in rows if r["split"] == "train"]
    dist, nearest = hamming_min([hashes[r["path"]] for r in train_rows],
                                [hashes[r["path"]] for r in eval_rows])

    removed = []
    for r, d, j in zip(train_rows, dist, nearest):
        if d <= a.threshold:
            r["split"] = "dropped_dup"
            removed.append({"path": r["path"], "label": r["label"], "source": r["source"],
                            "distance": d, "nearest_eval": eval_rows[j]["path"]})
    write_csv(out / "pv_manifest_dedup.csv", rows, ["path", "label", "leaf_id", "source", "split"])
    write_csv(out / "dedup_removed.csv", removed, ["path", "label", "source", "distance", "nearest_eval"])

    src = Counter(r["source"] for r in removed)
    print(f"Removed {len(removed)} training images within Hamming distance <= {a.threshold} of a val/test image "
          f"(mapped: {src['mapped']}, unmapped: {src['unmapped']}).")
    print("Report these counts in the paper. Inspect a few pairs in dedup_removed.csv before trusting the threshold.")


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifests", help="build label-mapped, leaf-grouped split manifests")
    m.add_argument("--pv-root", required=True)
    m.add_argument("--pd-root", required=True)
    m.add_argument("--out", default="manifests")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--val-frac", type=float, default=0.15)
    m.add_argument("--test-frac", type=float, default=0.15)
    m.add_argument("--calib-frac", type=float, default=0.20, help="PlantDoc share held out for recalibration")
    m.set_defaults(func=cmd_manifests)

    d = sub.add_parser("dedup", help="drop training images that are near-copies of val/test images")
    d.add_argument("--pv-root", required=True)
    d.add_argument("--manifest", default="manifests/pv_manifest.csv")
    d.add_argument("--out", default="manifests")
    d.add_argument("--threshold", type=int, default=6, help="max Hamming distance (of 64 bits) counted as a duplicate")
    d.set_defaults(func=cmd_dedup)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
