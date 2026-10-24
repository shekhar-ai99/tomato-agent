#!/usr/bin/env python3
"""
Stage 3b: logits of every trained model on every capture condition (clean + degraded copies).

Images are degraded once per condition and fed to all models in the same batch, so the cost of
degradation is paid only once. Each model gets its own normalisation (e.g. ViT uses mean/std 0.5).

Output per run: runs/<backbone>/seed<k>/cond_<set>.npz   keys: paths, labels, logits__<condition>

Usage:
  python scripts/predict_conditions.py --pv-root PlantVillage-Dataset --pd-root PlantDoc-Dataset --runs runs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from tomato_agent.classes import C2I
from tomato_agent.degrade import conditions, load_condition
from tomato_agent.io_utils import EVAL_SETS, add_data_args, eval_set, read_json, run_dirs
from tomato_agent.models import LoadedRun


class ConditionDataset(Dataset):
    def __init__(self, root, rows, cond, size):
        self.root, self.rows, self.cond = root, rows, cond
        self.tf = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        _, deg, lv, cp = self.cond
        return self.tf(load_condition(self.root / r["path"], r["path"], deg, lv, cp))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--backbones", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--sets", nargs="+", default=list(EVAL_SETS))
    ap.add_argument("--quality", default="artifacts/quality_thresholds.json",
                    help="levels/copies are read from here so both stages use the same conditions")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    q = read_json(args.quality) if Path(args.quality).exists() else {"levels": [2], "copies": 2}
    conds = conditions(tuple(q["levels"]), q["copies"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = run_dirs(args.runs, args.backbones, args.seeds)
    if not dirs:
        sys.exit(f"No trained runs (best.pt) under {args.runs}/")
    runs = [LoadedRun(d, device) for d in dirs]
    sizes = sorted({r.img_size for r in runs})
    norms = {id(r): (torch.tensor(r.data_cfg["mean"], device=device).view(1, 3, 1, 1),
                     torch.tensor(r.data_cfg["std"], device=device).view(1, 3, 1, 1)) for r in runs}
    print(f"device {device}; {len(runs)} runs; {len(conds)} conditions per image")

    for set_name in args.sets:
        todo = [r for r in runs if args.force or not (r.run_dir / f"cond_{set_name}.npz").exists()]
        if not todo:
            print(f"[skip] {set_name}: all runs done")
            continue
        root, rows = eval_set(args, set_name)
        labels = np.array([C2I[r["label"]] for r in rows])
        store = {id(r): {} for r in todo}
        for cond in conds:
            for size in sizes:
                group = [r for r in todo if r.img_size == size]
                if not group:
                    continue
                loader = DataLoader(ConditionDataset(root, rows, cond, size), batch_size=args.batch_size,
                                    num_workers=args.workers, shuffle=False)
                outs = {id(r): [] for r in group}
                with torch.no_grad():
                    for x in loader:
                        x = x.to(device)
                        for r in group:
                            m, s = norms[id(r)]
                            outs[id(r)].append(r.model((x - m) / s).float().cpu().numpy())
                for r in group:
                    store[id(r)][f"logits__{cond[0]}"] = np.concatenate(outs[id(r)])
            acc = {f"{r.backbone}/s{r.seed}": float((store[id(r)][f'logits__{cond[0]}'].argmax(1) == labels).mean())
                   for r in todo[:3]}
            print(f"  {set_name:9s} {cond[0]:24s} " + "  ".join(f"{k} acc={v:.3f}" for k, v in acc.items()))
        for r in todo:
            np.savez_compressed(r.run_dir / f"cond_{set_name}.npz", paths=np.array([x["path"] for x in rows]),
                                labels=labels, **store[id(r)])
        print(f"{set_name}: wrote cond_{set_name}.npz for {len(todo)} runs")


if __name__ == "__main__":
    main()
