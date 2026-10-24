import numpy as np
from PIL import Image, ImageDraw

from tomato_agent import degrade
from tomato_agent.quality import (DEFAULT_THRESHOLDS, CaptureQualityAgent, compute_features, features_from_array,
                                  select_thresholds)


def leaf(seed=0):
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (400, 300), (150, 140, 120))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 50, 340, 250], fill=(70, 140, 50))
    for _ in range(40):
        x, y = rng.integers(90, 310), rng.integers(80, 220)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(90, 60, 30))
    return im


def test_features_basic():
    flat = np.full((256, 256, 3), 128, np.uint8)
    f = features_from_array(flat)
    assert f["sharpness"] == 0 and f["leaf_frac"] == 0 and abs(f["lum_mean"] - 128 / 255) < 1e-9
    g = compute_features(leaf())
    assert 0.3 < g["leaf_frac"] < 0.8 and g["sharpness"] > 10


def test_each_degradation_moves_its_signal():
    im = leaf()
    base = compute_features(im)
    f = {d: compute_features(degrade.apply(im, d, 2, key="k")) for d in degrade.DEGRADATIONS}
    assert f["gaussian_blur"]["sharpness"] < base["sharpness"] * 0.5
    assert f["motion_blur"]["sharpness"] < base["sharpness"] * 0.5
    assert f["low_light"]["lum_mean"] < base["lum_mean"] * 0.5
    assert f["over_exposure"]["lum_mean"] > base["lum_mean"]
    assert f["occlusion"]["leaf_frac"] < base["leaf_frac"]


def test_degradation_is_deterministic_and_copies_differ():
    im = leaf()
    a = np.asarray(degrade.apply(im, "occlusion", 2, key="x|c0"))
    b = np.asarray(degrade.apply(im, "occlusion", 2, key="x|c0"))
    c = np.asarray(degrade.apply(im, "occlusion", 2, key="x|c1"))
    assert (a == b).all() and not (a == c).all()


def test_agent_flags_dark_photo():
    agent = CaptureQualityAgent(DEFAULT_THRESHOLDS)
    r = agent.assess(degrade.apply(leaf(), "low_light", 3, key="k"))
    assert not r.ok and "too_dark" in r.reasons


def test_threshold_selection_respects_fpr_cap():
    rng = np.random.default_rng(0)
    clean = np.column_stack([rng.normal(300, 50, 500), rng.uniform(.35, .6, 500), rng.uniform(0, .05, 500),
                             rng.uniform(0, .05, 500), rng.uniform(.3, .6, 500)])
    blur = clean.copy()
    blur[:, 0] = rng.normal(40, 10, 500)
    th, _ = select_thresholds([clean], {"gaussian_blur": blur, "motion_blur": blur}, max_fpr=0.03)
    assert (clean[:, 0] < th["min_sharpness"]).mean() <= 0.03
    assert (blur[:, 0] < th["min_sharpness"]).mean() > 0.95
