#!/usr/bin/env python3
"""
Tiny SYNTHETIC stand-ins for PlantVillage and PlantDoc, in exactly the folder layout prepare_data.py
expects. Only for smoke-testing the 8 stages end to end without downloading anything.
Numbers produced from this data mean nothing.

  python scripts/make_synthetic_data.py --out synthetic --per-class 40
"""
import argparse
import csv
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PV = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
      "Tomato___Septoria_leaf_spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
      "Tomato___healthy", "Tomato___Target_Spot", "Tomato___Spider_mites Two-spotted_spider_mite"]
PD = ["Tomato leaf bacterial spot", "Tomato Early blight leaf", "Tomato leaf late blight", "Tomato mold leaf",
      "Tomato Septoria leaf spot", "Tomato leaf yellow virus", "Tomato leaf mosaic virus", "Tomato leaf"]
# per class: (spot colour, number of spots, spot radius, leaf tint)
STYLE = [((40, 30, 20), 25, 4, (70, 140, 50)), ((110, 70, 30), 6, 14, (90, 140, 50)),
         ((60, 60, 40), 3, 30, (60, 110, 50)), ((200, 200, 90), 8, 12, (80, 150, 60)),
         ((150, 150, 150), 30, 5, (70, 140, 50)), ((220, 210, 60), 0, 0, (170, 180, 60)),
         ((40, 120, 40), 12, 16, (110, 170, 80)), ((0, 0, 0), 0, 0, (60, 150, 60)),
         ((90, 60, 40), 10, 9, (70, 140, 50)), ((230, 230, 200), 40, 2, (100, 150, 60))]


def leaf_image(rng, k, size, field):
    w, h = size
    if field:  # cluttered background: soil, sky, other leaves
        bg = rng.integers(40, 200, (h // 8 + 1, w // 8 + 1, 3)).astype(np.uint8)
        img = Image.fromarray(bg).resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
    else:
        g = int(rng.integers(150, 190))
        img = Image.new("RGB", (w, h), (g, g, g - 5))
    d = ImageDraw.Draw(img)
    spot, n, r, tint = STYLE[k]
    jitter = lambda c: tuple(int(np.clip(v + rng.integers(-15, 16), 0, 255)) for v in c)
    cx, cy = w * rng.uniform(0.4, 0.6), h * rng.uniform(0.4, 0.6)
    rx, ry = w * rng.uniform(0.25, 0.38), h * rng.uniform(0.18, 0.3)
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=jitter(tint))
    d.line([cx - rx, cy, cx + rx, cy], fill=jitter((40, 90, 30)), width=max(1, w // 120))
    for _ in range(n if not field else max(0, n + int(rng.integers(-n // 2 - 1, 2)))):
        a, t = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0, 0.8))
        x, y = cx + rx * t * np.cos(a), cy + ry * t * np.sin(a)
        rr = r * (w / 256) * rng.uniform(0.6, 1.3)
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=jitter(spot))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(img).astype(np.int16) + rng.normal(0, 4 if not field else 9, (h, w, 3)).astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--per-class", type=int, default=40, help="PlantVillage images per class (multiple of 4)")
    ap.add_argument("--pd-per-class", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pv, pd = Path(a.out) / "PlantVillage-Dataset", Path(a.out) / "PlantDoc-Dataset"
    for k, folder in enumerate(PV):
        d = pv / "raw" / "color" / folder
        d.mkdir(parents=True, exist_ok=True)
        lm = []
        for i in range(a.per_class):
            orig = f"{uuid.UUID(int=int(rng.integers(0, 2**63))).hex[:8]}___RS_Leaf {i}.JPG"
            leaf_image(rng, k, (256, 256), False).save(d / f"{uuid.uuid4()}___{orig}", quality=92)
            lm.append((orig, i // 4))
        if "mosaic" not in folder:  # the real dataset has no leaf map for mosaic virus
            m = pv / "leaf_grouping" / "filtered_leafmaps"
            m.mkdir(parents=True, exist_ok=True)
            with open(m / f"{folder}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["file", "leaf"])
                # like the real maps, only part of the images have a leaf id
                w.writerows([row for j, row in enumerate(lm) if j < int(len(lm) * 0.8)])
    for k, folder in enumerate(PD):
        for sub, n in (("train", a.pd_per_class - 3), ("test", 3)):
            d = pd / sub / folder
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                size = (int(rng.integers(400, 900)), int(rng.integers(300, 700)))
                leaf_image(rng, k, size, True).save(d / f"{sub}_{k}_{i}.jpg", quality=88)
    print(f"Synthetic data written to {a.out}/ (PlantVillage-Dataset, PlantDoc-Dataset)")


if __name__ == "__main__":
    main()
