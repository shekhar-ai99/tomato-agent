"""Capture-quality agent (paper Section 3.2).

Cheap, explainable signals computed on a 256x256 working copy of the photo:

  sharpness    variance of the 4-neighbour Laplacian of luminance (low = blurry)
  lum_mean     mean luminance in [0, 1]           (low = dark, high = over-exposed)
  dark_frac    share of pixels with luminance < 16
  bright_frac  share of pixels with luminance > 240
  leaf_frac    share of leaf-coloured pixels (HSV hue 35..170 deg = yellow to green, sat >= 0.25, val >= 0.12;
               beige soil and grey backgrounds do not count)
               (low = leaf not visible, covered by a finger, or too far away)

The exact same arithmetic is implemented in the Android app (QualityAgent.kt). Keep the two in
sync; tests/test_quality_parity.py writes shared/quality_parity_fixture.json, which the app's unit
test reads.
"""
from dataclasses import dataclass, field

import numpy as np

WORK_SIZE = 256
FEATURES = ["sharpness", "lum_mean", "dark_frac", "bright_frac", "leaf_frac"]
DARK_LEVEL, BRIGHT_LEVEL = 16, 240
REASON_PRIORITY = ["no_leaf", "blurry", "too_dark", "too_bright"]

# Fallback thresholds, used only until set_quality_thresholds.py has been run.
DEFAULT_THRESHOLDS = {
    "min_sharpness": 60.0,
    "min_lum_mean": 0.18,
    "max_lum_mean": 0.85,
    "max_dark_frac": 0.60,
    "max_bright_frac": 0.35,
    "min_leaf_frac": 0.12,
}


def to_work_array(img):
    """PIL image (or HxWx3 uint8 array) -> 256x256x3 uint8 array (bilinear resize)."""
    from PIL import Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    img = img.convert("RGB")
    if img.size != (WORK_SIZE, WORK_SIZE):
        img = img.resize((WORK_SIZE, WORK_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def luminance(rgb):
    """Integer-weighted luma, as float: (299 R + 587 G + 114 B) / 1000."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    return (299.0 * r + 587.0 * g + 114.0 * b) / 1000.0


def plant_mask(rgb):
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    v = mx / 255.0
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0.0)
    safe = np.where(d > 0, d, 1)
    h = np.where(mx == r, ((g - b) / safe) % 6.0,
                 np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) * 60.0
    h = np.where(d > 0, h, 0.0)
    return (d > 0) & (h >= 35.0) & (h <= 170.0) & (s >= 0.25) & (v >= 0.12)


def features_from_array(rgb):
    """Quality features of an already 256x256 (or any size) uint8 RGB array. No resizing here."""
    y = luminance(rgb)
    lap = y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:] - 4.0 * y[1:-1, 1:-1]
    return {
        "sharpness": float(lap.var()),
        "lum_mean": float(y.mean() / 255.0),
        "dark_frac": float((y < DARK_LEVEL).mean()),
        "bright_frac": float((y > BRIGHT_LEVEL).mean()),
        "leaf_frac": float(plant_mask(rgb).mean()),
    }


def compute_features(img):
    return features_from_array(to_work_array(img))


def reasons_from_features(f, t):
    """List of quality problems, most important first. Empty list = photo is usable."""
    out = []
    if f["leaf_frac"] < t["min_leaf_frac"]:
        out.append("no_leaf")
    if f["sharpness"] < t["min_sharpness"]:
        out.append("blurry")
    if f["lum_mean"] < t["min_lum_mean"] or f["dark_frac"] > t["max_dark_frac"]:
        out.append("too_dark")
    if f["lum_mean"] > t["max_lum_mean"] or f["bright_frac"] > t["max_bright_frac"]:
        out.append("too_bright")
    return out


def flags_from_matrix(F, t):
    """Vectorised: F is (N, 5) in FEATURES order -> boolean (N,) 'photo flagged'."""
    F = np.asarray(F, dtype=np.float64)
    s, lm, dk, br, lf = (F[:, i] for i in range(5))
    return ((lf < t["min_leaf_frac"]) | (s < t["min_sharpness"]) |
            (lm < t["min_lum_mean"]) | (dk > t["max_dark_frac"]) |
            (lm > t["max_lum_mean"]) | (br > t["max_bright_frac"]))


@dataclass
class QualityResult:
    ok: bool
    reasons: list
    features: dict = field(default_factory=dict)

    @property
    def main_reason(self):
        return self.reasons[0] if self.reasons else None


class CaptureQualityAgent:
    """Rejects unusable photos and says why, so the farmer can fix it on the retake."""

    def __init__(self, thresholds=None):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    @classmethod
    def from_file(cls, path):
        from .io_utils import read_json
        return cls(read_json(path)["thresholds"])

    def assess(self, img):
        f = compute_features(img)
        reasons = reasons_from_features(f, self.thresholds)
        return QualityResult(ok=not reasons, reasons=reasons, features=f)


# ----------------------------------------------------------------------------- threshold selection
# signal -> (feature column, direction, the degradations it is meant to catch)
SIGNALS = {
    "min_sharpness": ("sharpness", "low", ["gaussian_blur", "motion_blur"]),
    "min_lum_mean": ("lum_mean", "low", ["low_light"]),
    "max_dark_frac": ("dark_frac", "high", ["low_light"]),
    "max_lum_mean": ("lum_mean", "high", ["over_exposure"]),
    "max_bright_frac": ("bright_frac", "high", ["over_exposure"]),
    "min_leaf_frac": ("leaf_frac", "low", ["occlusion"]),
}


# Sanity limits so a very uniform validation set (PlantVillage's plain backgrounds) cannot produce
# thresholds that flag every outdoor photo with a bit of sky or shadow.
BOUNDS = {
    "min_sharpness": (None, None),
    "min_lum_mean": (0.05, 0.30),
    "max_lum_mean": (0.70, 0.95),
    "max_dark_frac": (0.25, 0.90),
    "max_bright_frac": (0.15, 0.90),
    "min_leaf_frac": (0.02, 0.30),
}


def select_thresholds(clean_sets, degraded, max_fpr=0.03, bounds=BOUNDS):
    """Pick one threshold per signal on validation photos (no labels needed).

    clean_sets: list of (N_i, 5) feature arrays of clean validation photos (e.g. PlantVillage val and
                the PlantDoc calibration hold-out); the false-flag rate is kept <= max_fpr on EACH.
    degraded:   dict degradation -> (M, 5) features of degraded copies
    For each signal, choose the cut maximising TPR(matching degradations) - max_i FPR_i(clean),
    subject to every FPR_i <= max_fpr, then clamp it to `bounds`.
    Returns (thresholds, report rows).
    """
    if isinstance(clean_sets, np.ndarray):
        clean_sets = [clean_sets]
    clean_sets = [np.asarray(c, dtype=np.float64) for c in clean_sets]
    pooled = np.concatenate(clean_sets)
    out, report = {}, []
    for name, (feat, direction, degs) in SIGNALS.items():
        j = FEATURES.index(feat)
        cs = [c[:, j] for c in clean_sets]
        pos = np.concatenate([np.asarray(degraded[d])[:, j] for d in degs if d in degraded] or [np.empty(0)])
        qs = np.linspace(0.0, max_fpr, 31)
        cands = np.quantile(pooled[:, j], qs) if direction == "low" else np.quantile(pooled[:, j], 1.0 - qs)
        rate = (lambda v, th: float((v < th).mean())) if direction == "low" else (lambda v, th: float((v > th).mean()))
        lo, hi = bounds.get(name, (None, None))
        clamp = lambda th: float(np.clip(th, lo if lo is not None else -np.inf, hi if hi is not None else np.inf))
        best = None
        for th in sorted({clamp(t) for t in cands}, reverse=(direction == "high")):
            fpr = max(rate(c, th) for c in cs)
            if fpr > max_fpr + 1e-9:
                continue
            tpr = rate(pos, th) if len(pos) else 0.0
            if best is None or tpr - fpr > best[0] + 1e-12:
                best = (tpr - fpr, th, tpr, fpr)
        if best is None:  # nothing satisfies the FPR cap within bounds: take the most lenient bound
            th = clamp(-np.inf if direction == "low" else np.inf)
            best = (0.0, th, rate(pos, th) if len(pos) else 0.0, max(rate(c, th) for c in cs))
        out[name] = best[1]
        report.append({"threshold": name, "value": round(best[1], 6), "feature": feat,
                       "tpr_on": "+".join(degs), "tpr": round(best[2], 4), "fpr_clean_max": round(best[3], 4)})
    return out, report


def detection_report(F_clean, F_by_deg, t):
    """Precision/recall of 'flagged' against 'is degraded', per degradation (paper Table 5.2 support)."""
    fc = flags_from_matrix(F_clean, t)
    rows = [{"degradation": "none", "flag_rate": round(float(fc.mean()), 4)}]
    for d, F in F_by_deg.items():
        fd = flags_from_matrix(F, t)
        tp, fp = int(fd.sum()), int(fc.sum())
        rows.append({"degradation": d, "flag_rate": round(float(fd.mean()), 4),
                     "recall": round(float(fd.mean()), 4),
                     "precision_vs_clean": round(tp / max(tp + fp, 1), 4)})
    return rows
