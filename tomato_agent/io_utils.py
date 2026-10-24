"""Small file helpers shared by the scripts."""
import csv
import json
from pathlib import Path


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def manifest_rows(path, split):
    """Rows of one split from a prepare_data.py manifest, in file order (the order logits are saved in)."""
    return [r for r in read_csv(path) if r["split"] == split]


# eval sets used throughout: name -> (which manifest, which split, which image root)
EVAL_SETS = {
    "pv_val": ("pv", "val"),
    "pv_test": ("pv", "test"),
    "pd_calib": ("pd", "calib"),
    "pd_test": ("pd", "test"),
}


def run_dirs(runs_root, backbones=None, seeds=None):
    """All <runs>/<backbone>/seed<k>/ folders that have a trained best.pt, sorted."""
    out = []
    for bb_dir in sorted(Path(runs_root).iterdir()) if Path(runs_root).exists() else []:
        if not bb_dir.is_dir() or (backbones and bb_dir.name not in backbones):
            continue
        for sd in sorted(bb_dir.glob("seed*")):
            k = int(sd.name[4:])
            if seeds is not None and k not in seeds:
                continue
            if (sd / "best.pt").exists():
                out.append(sd)
    return out


def add_data_args(ap):
    ap.add_argument("--pv-root", required=True, help="PlantVillage-Dataset folder")
    ap.add_argument("--pd-root", required=True, help="PlantDoc-Dataset folder")
    ap.add_argument("--pv-manifest", default="manifests/pv_manifest_dedup.csv")
    ap.add_argument("--pd-manifest", default="manifests/pd_manifest.csv")


def eval_set(args, name):
    """(image root, manifest rows) of one eval set, in the same order train_diagnosis.py uses."""
    which, split = EVAL_SETS[name]
    root = args.pv_root if which == "pv" else args.pd_root
    manifest = args.pv_manifest if which == "pv" else args.pd_manifest
    rows = manifest_rows(manifest, split)
    if getattr(args, "limit", 0):
        rows = rows[:args.limit]
    return Path(root), rows
