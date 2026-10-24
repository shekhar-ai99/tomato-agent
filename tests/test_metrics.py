import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_metrics import K, auc_binary, average_precision, confusion, core_metrics, roc_points  # noqa: E402


def test_auc_known_values():
    assert auc_binary([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75   # textbook example
    assert auc_binary([0, 1], [0.2, 0.9]) == 1.0
    assert auc_binary([1, 0], [0.2, 0.9]) == 0.0
    assert auc_binary([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5     # all ties
    assert np.isnan(auc_binary([1, 1, 1], [0.1, 0.2, 0.3]))          # no negatives


def test_roc_points_match_auc():
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 300), rng.random(300)
    fpr, tpr = roc_points(y, s, max_points=10 ** 9)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    assert abs(trapz(tpr, fpr) - auc_binary(y, s)) < 1e-9
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1


def test_average_precision_perfect_and_worst():
    assert average_precision([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert average_precision([0, 0, 1], [0.9, 0.8, 0.1]) < 0.5


def test_core_metrics_perfect_and_uniform():
    labels = np.arange(K).repeat(3)
    perfect = np.eye(K)[labels]
    m = core_metrics(perfect, labels)[0]
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["auc_macro_ovr"] == 1 and m["brier"] == 0 and m["ece"] == 0
    uniform = np.full((len(labels), K), 1 / K)
    u = core_metrics(uniform, labels)[0]
    assert abs(u["auc_macro_ovr"] - 0.5) < 1e-9 and abs(u["brier"] - (1 - 1 / K)) < 1e-9
    cm = confusion(labels, labels)
    assert cm.trace() == len(labels)


def test_absent_class_is_left_out_of_macro_average():
    labels = np.array([0, 0, 1, 1, 2, 2])          # classes 3..7 never occur, as mosaic_virus in PlantVillage
    probs = np.eye(K)[labels]
    m = core_metrics(probs, labels)[0]
    assert m["macro_f1"] == 1 and m["auc_macro_ovr"] == 1
