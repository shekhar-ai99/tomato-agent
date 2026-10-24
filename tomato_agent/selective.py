"""Selective-prediction metrics: risk-coverage curve, AURC, accuracy at fixed coverage (paper Section 4.5)."""
import math

import numpy as np


def risk_coverage(score, correct):
    """Sort cases by score (high = most trusted). Returns coverage[k], risk[k] for k = 1..N answered."""
    score, correct = np.asarray(score, dtype=np.float64), np.asarray(correct, dtype=float)
    order = np.argsort(-score, kind="stable")
    c = correct[order]
    n = np.arange(1, len(c) + 1)
    return n / len(c), 1.0 - np.cumsum(c) / n


def aurc(score, correct):
    """Area under the risk-coverage curve (lower is better), Geifman & El-Yaniv style mean over k."""
    _, risk = risk_coverage(score, correct)
    return float(risk.mean()) if len(risk) else float("nan")


def selective_accuracy(score, correct, coverage):
    """Accuracy on the ceil(coverage * N) highest-scoring cases."""
    score, correct = np.asarray(score, dtype=np.float64), np.asarray(correct, dtype=float)
    if len(score) == 0:
        return float("nan")
    k = max(1, math.ceil(coverage * len(score)))
    order = np.argsort(-score, kind="stable")[:k]
    return float(correct[order].mean())


def curve_on_grid(score, correct, grid=None):
    """Risk at a fixed coverage grid, so curves from different seeds can be averaged."""
    grid = np.linspace(0.05, 1.0, 96) if grid is None else grid
    return grid, np.array([1.0 - selective_accuracy(score, correct, g) for g in grid])
