"""Synthetic capture degradations (paper Section 4.3).

Every degraded copy is deterministic: it depends only on (image path, degradation, level, copy).
copy 0 is the first shot; copy 1, 2, ... are independently degraded retakes (pessimistic bound).
The optimistic retake is the clean original.

Photos are first shrunk so the longer side is at most 512 px (never enlarged), so blur strength is
comparable across PlantVillage (256 px) and PlantDoc (often several thousand px).
"""
import hashlib

import numpy as np
from PIL import Image, ImageFilter

DEGRADATIONS = ["gaussian_blur", "motion_blur", "low_light", "over_exposure", "occlusion"]
LEVELS = {1: "mild", 2: "moderate", 3: "severe"}
MAX_SIDE = 512

_GAUSS_SIGMA = {1: 1.2, 2: 2.4, 3: 4.0}          # in pixels at 256 px short side
_MOTION_LEN = {1: 7, 2: 13, 3: 21}               # in pixels at 256 px short side
_DARK_GAIN = {1: 0.45, 2: 0.28, 3: 0.16}
_BRIGHT_GAIN = {1: 1.6, 2: 2.1, 3: 2.8}
_OCCLUDED_AREA = {1: 0.25, 2: 0.45, 3: 0.65}


def condition_name(deg, level, copy):
    return "none" if deg == "none" else f"{deg}_L{level}_c{copy}"


def _rng(key):
    h = hashlib.md5(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _shrink(img):
    img = img.convert("RGB")
    w, h = img.size
    s = MAX_SIDE / max(w, h)
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return img


def _motion_kernel(length, angle_deg):
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    t = np.deg2rad(angle_deg)
    for i in np.linspace(-c, c, length * 3):
        x, y = int(round(c + i * np.cos(t))), int(round(c + i * np.sin(t)))
        k[y, x] = 1
    return k / k.sum()


def _convolve_rgb(a, k):
    """Same-size 2D convolution per channel with reflect padding."""
    from scipy.ndimage import convolve
    return np.stack([convolve(a[..., ch], k, mode="reflect") for ch in range(3)], axis=-1)


def apply(img, deg, level=2, key=""):
    """Return a degraded PIL copy of img. `key` makes the randomness reproducible."""
    if deg == "none":
        return img.convert("RGB")
    rng = _rng(f"{key}|{deg}|{level}")
    img = _shrink(img)
    scale = min(img.size) / 256.0
    if deg == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=_GAUSS_SIGMA[level] * scale))
    a = np.asarray(img, dtype=np.float32)
    if deg == "motion_blur":
        length = max(3, int(round(_MOTION_LEN[level] * scale)) | 1)
        out = _convolve_rgb(a, _motion_kernel(length, float(rng.uniform(0, 180))))
    elif deg == "low_light":
        gain = _DARK_GAIN[level] * float(rng.uniform(0.85, 1.15))
        out = a * gain + rng.normal(0, 2.0 + 2.0 * level, a.shape)  # sensor noise in the dark
    elif deg == "over_exposure":
        gain = _BRIGHT_GAIN[level] * float(rng.uniform(0.9, 1.1))
        out = a * gain + 20.0 * level
    elif deg == "occlusion":
        out = a.copy()
        H, W = a.shape[:2]
        target, covered = _OCCLUDED_AREA[level], np.zeros((H, W), bool)
        for _ in range(12):
            if covered.mean() >= target:
                break
            ph = int(H * rng.uniform(0.25, 0.6))
            pw = int(W * rng.uniform(0.25, 0.6))
            y0, x0 = int(rng.integers(0, H - ph + 1)), int(rng.integers(0, W - pw + 1))
            grey = float(rng.uniform(30, 200))  # a neutral object: thumb shadow, paper, wall
            out[y0:y0 + ph, x0:x0 + pw] = grey + rng.normal(0, 4, (ph, pw, 1))
            covered[y0:y0 + ph, x0:x0 + pw] = True
    else:
        raise ValueError(f"unknown degradation {deg}")
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def conditions(levels=(2,), copies=2, degradations=DEGRADATIONS):
    """All (name, deg, level, copy) conditions evaluated in the study, 'none' first."""
    out = [("none", "none", 0, 0)]
    for d in degradations:
        for lv in levels:
            for c in range(copies):
                out.append((condition_name(d, lv, c), d, lv, c))
    return out


def load_condition(full_path, rel_key, deg, level, copy):
    """Open an image and apply one condition. rel_key (the manifest path) seeds the randomness,
    so the same copy is produced on any machine."""
    with Image.open(full_path) as im:
        im = im.convert("RGB")
        return apply(im, deg, level, key=f"{rel_key}|c{copy}") if deg != "none" else im.copy()
