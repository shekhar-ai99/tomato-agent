#!/usr/bin/env python3
"""
Stage 3 of the workflow (Section 4.6): train the diagnosis agent's classifier.

Trains each backbone on the PlantVillage train split (8 shared tomato classes) for several seeds,
selects the best epoch by validation loss, then evaluates on PlantVillage val/test and on PlantDoc
calib/test. Nothing from PlantDoc is used for training or model selection.

Inputs (from prepare_data.py):
  manifests/pv_manifest_dedup.csv   (or pv_manifest.csv)   -> columns: path,label,leaf_id,source,split
  manifests/pd_manifest.csv                                   -> columns: path,label,origin,split

Outputs, per backbone and seed, in <out>/<backbone>/seed<k>/ :
  config.json, history.csv, best.pt, metrics.json,
  logits_pv_val.npz, logits_pv_test.npz, logits_pd_calib.npz, logits_pd_test.npz
The saved logits feed stage 4 (temperature scaling), stage 6 (confidence gating) and the baselines.

Usage:
  pip install torch torchvision timm pillow numpy
  python train_diagnosis.py --pv-root PlantVillage-Dataset --pd-root PlantDoc-Dataset \
      --pv-manifest manifests/pv_manifest_dedup.csv --pd-manifest manifests/pd_manifest.csv \
      --backbones efficientnet_b0 resnet50 vit_small_patch16_224 --seeds 0 1 2 3 4 --out runs

Runs that already have metrics.json are skipped, so an interrupted job can be restarted.
Default hyperparameters are proposals: edit them, and report what you actually used.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASSES = [
    "bacterial_spot", "early_blight", "late_blight", "leaf_mold",
    "septoria_leaf_spot", "yellow_leaf_curl_virus", "mosaic_virus", "healthy",
]
C2I = {c: i for i, c in enumerate(CLASSES)}

# proposed per-backbone learning rates for full fine-tuning with AdamW
DEFAULT_LR = {"efficientnet_b0": 3e-4, "resnet50": 3e-4, "vit_small_patch16_224": 1e-4}


# ----------------------------------------------------------------------------- data
def read_rows(path, splits):
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] in splits]


class ManifestDataset(Dataset):
    def __init__(self, root, rows, tf):
        self.root, self.rows, self.tf = Path(root), rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(self.root / r["path"]) as im:
            x = self.tf(im.convert("RGB"))
        return x, C2I[r["label"]]


def make_transforms(cfg, size):
    mean, std = cfg["mean"], cfg["std"]
    # Direct square resize (no centre crop) so off-centre leaves in field photos are not cut off.
    train_tf = T.Compose([
        T.RandomResizedCrop((size, size), scale=(0.5, 1.0), ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])
    return train_tf, eval_tf


# ----------------------------------------------------------------------------- metrics
def accuracy(logits, labels):
    return float((logits.argmax(1) == labels).mean())


def macro_f1_and_recall(logits, labels):
    """Macro-F1 over classes that have at least one true sample, plus per-class recall."""
    pred = logits.argmax(1)
    f1s, recall = [], {}
    for k, c in enumerate(CLASSES):
        support = int((labels == k).sum())
        if support == 0:
            continue
        tp = int(((pred == k) & (labels == k)).sum())
        fp = int(((pred == k) & (labels != k)).sum())
        fn = support - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        recall[c] = round(rec, 4)
    return float(np.mean(f1s)), recall


def summarize(logits, labels):
    f1, rec = macro_f1_and_recall(logits, labels)
    return {"n": int(len(labels)), "accuracy": accuracy(logits, labels), "macro_f1": f1,
            "n_classes_with_support": len(rec), "per_class_recall": rec}


# ----------------------------------------------------------------------------- train / eval
def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outs, labs = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).float().cpu())
        labs.append(y)
    return torch.cat(outs).numpy(), torch.cat(labs).numpy()


def class_weights(rows, mode):
    if mode == "none":
        return None
    counts = np.array([sum(r["label"] == c for r in rows) for c in CLASSES], dtype=float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train_one(args, backbone, seed, device):
    run_dir = Path(args.out) / backbone / f"seed{seed}"
    if (run_dir / "metrics.json").exists() and not args.force:
        print(f"[skip] {backbone} seed{seed}: metrics.json exists")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, args.deterministic)

    kwargs = {"img_size": args.img_size} if backbone.startswith("vit") else {}
    model = timm.create_model(backbone, pretrained=bool(args.pretrained), num_classes=len(CLASSES), **kwargs)
    cfg = timm.data.resolve_data_config({}, model=model)
    train_tf, eval_tf = make_transforms(cfg, args.img_size)
    model.to(device)

    pv_train = read_rows(args.pv_manifest, {"train"})
    pv_val = read_rows(args.pv_manifest, {"val"})
    pv_test = read_rows(args.pv_manifest, {"test"})
    pd_calib = read_rows(args.pd_manifest, {"calib"})
    pd_test = read_rows(args.pd_manifest, {"test"})
    if args.limit:  # quick smoke test only
        pv_train, pv_val, pv_test = pv_train[:args.limit], pv_val[:args.limit], pv_test[:args.limit]
        pd_calib, pd_test = pd_calib[:args.limit], pd_test[:args.limit]

    g = torch.Generator()
    g.manual_seed(seed)
    mk = lambda root, rows, tf, shuffle: DataLoader(
        ManifestDataset(root, rows, tf), batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, generator=g if shuffle else None, drop_last=False,
        pin_memory=device.type == "cuda")
    train_loader = mk(args.pv_root, pv_train, train_tf, True)
    loaders = {"pv_val": mk(args.pv_root, pv_val, eval_tf, False), "pv_test": mk(args.pv_root, pv_test, eval_tf, False),
               "pd_calib": mk(args.pd_root, pd_calib, eval_tf, False), "pd_test": mk(args.pd_root, pd_test, eval_tf, False)}

    lr = args.lr or DEFAULT_LR.get(backbone, 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total, warm = args.epochs * steps_per_epoch, steps_per_epoch  # one warm-up epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    crit = nn.CrossEntropyLoss(weight=class_weights(pv_train, args.class_weights))
    if crit.weight is not None:
        crit.weight = crit.weight.to(device)

    cfg_out = {**vars(args), "backbone": backbone, "seed": seed, "lr_used": lr,
               "n_train": len(pv_train), "n_val": len(pv_val)}
    (run_dir / "config.json").write_text(json.dumps(cfg_out, indent=2, default=str))

    best_loss, best_epoch, bad, history = float("inf"), -1, 0, []
    val_crit = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and args.amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        vl, vy = predict(model, loaders["pv_val"], device)
        val_loss = float(val_crit(torch.from_numpy(vl), torch.from_numpy(vy)))
        row = {"epoch": epoch, "train_loss": run_loss / max(n, 1), "val_loss": val_loss,
               "val_acc": accuracy(vl, vy), "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"{backbone} seed{seed} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    metrics = {"backbone": backbone, "seed": seed, "best_epoch": best_epoch, "best_val_loss": best_loss}
    rows_by = {"pv_val": pv_val, "pv_test": pv_test, "pd_calib": pd_calib, "pd_test": pd_test}
    for name, loader in loaders.items():
        logits, labels = predict(model, loader, device)
        np.savez_compressed(run_dir / f"logits_{name}.npz", logits=logits, labels=labels,
                            paths=np.array([r["path"] for r in rows_by[name]]))
        metrics[name] = summarize(logits, labels)
    metrics["field_gap_accuracy"] = metrics["pv_test"]["accuracy"] - metrics["pd_test"]["accuracy"]
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"{backbone} seed{seed}: PV test acc {metrics['pv_test']['accuracy']:.4f} | "
          f"PlantDoc test acc {metrics['pd_test']['accuracy']:.4f} | gap {metrics['field_gap_accuracy']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pv-root", required=True)
    ap.add_argument("--pd-root", required=True)
    ap.add_argument("--pv-manifest", default="manifests/pv_manifest_dedup.csv")
    ap.add_argument("--pd-manifest", default="manifests/pd_manifest.csv")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--backbones", nargs="+", default=["efficientnet_b0", "resnet50", "vit_small_patch16_224"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5, help="early stopping on validation loss")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="override the per-backbone default")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--class-weights", choices=["none", "sqrt"], default="sqrt",
                    help="sqrt = inverse-square-root class frequency (PlantVillage is imbalanced)")
    ap.add_argument("--pretrained", type=int, default=1, help="1 = ImageNet weights (needs internet once)")
    ap.add_argument("--amp", action="store_true", help="mixed precision on GPU")
    ap.add_argument("--deterministic", action="store_true", help="slower, but more repeatable on GPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows per split (smoke test)")
    ap.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    for backbone in args.backbones:
        for seed in args.seeds:
            train_one(args, backbone, seed, device)


if __name__ == "__main__":
    main()
