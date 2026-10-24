"""Shared machinery for stages 6-7 (E2-E5): turning saved logits + quality features into
multi-shot cases, and evaluating baselines and the gated pipeline on them.

Inputs per run folder (runs/<backbone>/seed<k>/):
  cond_<set>.npz        logits for every capture condition (scripts/predict_conditions.py)
  calibration.json      T_pv and T_pd (scripts/calibrate.py)
Inputs shared by all runs:
  artifacts/quality_features/<set>.npz   quality features per condition (scripts/set_quality_thresholds.py)
  artifacts/quality_thresholds.json
"""
from pathlib import Path

import numpy as np

from .calibration import ece, softmax
from .degrade import DEGRADATIONS, condition_name
from .gate import Costs, GateConfig, outcome_metrics, proposed_score, select_thresholds, simulate
from .io_utils import read_json
from .quality import flags_from_matrix
from .selective import aurc, curve_on_grid, selective_accuracy

COVERAGES = (0.9, 0.8, 0.7)
MODES = ("opt", "pess")  # optimistic retake = clean original; pessimistic = independently re-degraded copy


def parse_costs(specs, c_ret):
    out = []
    for s in specs:
        a, b = s.split(":")
        out.append(Costs(float(a), float(b), c_ret))
    return out


class SetData:
    """Calibrated probabilities and quality flags of one run on one eval set, for every condition."""

    def __init__(self, run_dir, set_name, quality_dir, thresholds, temperature="pv"):
        z = np.load(Path(run_dir) / f"cond_{set_name}.npz", allow_pickle=True)
        q = np.load(Path(quality_dir) / f"{set_name}.npz", allow_pickle=True)
        if list(z["paths"]) != list(q["paths"]):
            raise ValueError(f"{run_dir} {set_name}: logits and quality features are not in the same order")
        cal = read_json(Path(run_dir) / "calibration.json") if (Path(run_dir) / "calibration.json").exists() else {}
        self.T = 1.0 if temperature == "none" else float(cal.get(f"T_{temperature}", 1.0))
        self.labels = z["labels"].astype(int)
        self.paths = z["paths"]
        self.conds = [k[len("logits__"):] for k in z.files if k.startswith("logits__")]
        self.logits = {c: z[f"logits__{c}"] for c in self.conds}
        self.probs = {c: softmax(self.logits[c], self.T) for c in self.conds}
        self.raw_probs = {c: softmax(self.logits[c], 1.0) for c in self.conds}
        self.qflag = {c: flags_from_matrix(q[f"F__{c}"], thresholds) for c in self.conds if f"F__{c}" in q.files}
        self.n = len(self.labels)


def scenario_conditions(data, scenario, level, mode, R, set_name, mix_seed=1234):
    """Per-case condition names for shots 0..R.

    scenario: 'none', a degradation name, or 'mixed' (each case gets one of none + the 5
    degradations at random, the same draw for every model).
    """
    N = data.n
    if scenario == "mixed":
        rng = np.random.default_rng(mix_seed + sum(map(ord, set_name)))
        choice = rng.integers(0, len(DEGRADATIONS) + 1, N)
        degs = np.array(["none"] + DEGRADATIONS, dtype=object)[choice]
    else:
        degs = np.array([scenario] * N, dtype=object)
    shots = []
    for s in range(R + 1):
        names = []
        for d in degs:
            if d == "none":
                names.append("none")
            elif s == 0:
                names.append(condition_name(d, level, 0))
            elif mode == "opt":
                names.append("none")
            else:
                c = s
                while condition_name(d, level, c) not in data.probs and c > 0:
                    c -= 1  # fewer stored copies than retakes: reuse the last independent copy
                names.append(condition_name(d, level, c))
        shots.append(np.array(names, dtype=object))
    return shots, degs


def gather(data, names, which="probs"):
    src = getattr(data, which)
    out = np.empty((data.n, src["none"].shape[1]))
    for c in np.unique(names):
        m = names == c
        out[m] = src[c][m]
    return out


def gather_flags(data, names):
    out = np.zeros(data.n, bool)
    for c in np.unique(names):
        m = names == c
        out[m] = data.qflag[c][m]
    return out


def build_shots(data, scenario, level, mode, R, set_name, which="probs"):
    names, degs = scenario_conditions(data, scenario, level, mode, R, set_name)
    return [gather(data, n, which) for n in names], [gather_flags(data, n) for n in names], degs


def selective_row(score, correct, probs_for_ece=None):
    row = {f"acc@{int(c * 100)}": selective_accuracy(score, correct, c) for c in COVERAGES}
    row["aurc"] = aurc(score, correct)
    if probs_for_ece is not None:
        row["ece"] = ece(*probs_for_ece)
    return row


def choose_gate(data, costs, set_name, level, R=1, use_quality=True, mode="pess"):
    P, Q, _ = build_shots(data, "mixed", level, mode, R, set_name)
    return select_thresholds(P, Q, data.labels, costs, R, use_quality)


def evaluate_proposed(data, cfg, scenario, level, mode, set_name, costs):
    P, Q, _ = build_shots(data, scenario, level, mode, cfg.retake_budget, set_name)
    sim = simulate(P, Q, data.labels, cfg)
    sim_all = simulate(P, Q, data.labels, cfg, answer_all=True)
    row = outcome_metrics(sim, costs)
    row["acc_all"] = float(sim_all["correct"].mean())
    final_probs = np.stack(P)[sim["final_shot"], np.arange(data.n)]
    row.update(selective_row(proposed_score(sim), sim["correct"], (final_probs, data.labels)))
    return row, proposed_score(sim), sim["correct"]


def evaluate_threshold_only(data, cfg, scenario, level, set_name, costs, which="probs"):
    """B3 (calibrated) and, with which='raw_probs', the uncalibrated static model B1."""
    P, Q, _ = build_shots(data, scenario, level, "pess", 0, set_name, which)
    p0 = P[0]
    correct = p0.argmax(1) == data.labels
    row = {"acc_all": float(correct.mean())}
    row.update(selective_row(p0.max(1), correct, (p0, data.labels)))
    if cfg is not None:
        row.update(outcome_metrics(simulate(P, Q, data.labels, cfg), costs))
    return row, p0.max(1), correct


def evaluate_ensemble(datas, scenario, level, set_name):
    """B2: mean of the calibrated probabilities of several backbones (same seed), first photo only."""
    probs = []
    for d in datas:
        P, _, _ = build_shots(d, scenario, level, "pess", 0, set_name)
        probs.append(P[0])
    p = np.mean(probs, axis=0)
    labels = datas[0].labels
    correct = p.argmax(1) == labels
    row = {"acc_all": float(correct.mean())}
    row.update(selective_row(p.max(1), correct, (p, labels)))
    return row, p.max(1), correct


def quality_only(data, scenario, level, mode, R, set_name):
    """RQ2: quality check + retake, no confidence gate, always answers (paper Table 5.2)."""
    P, Q, degs = build_shots(data, scenario, level, mode, R, set_name)
    static = float((P[0].argmax(1) == data.labels).mean())
    sim = simulate(P, Q, data.labels, GateConfig(0.0, 0.0, R, True), answer_all=True)
    return {"static_acc": static, "quality_retake_acc": float(sim["correct"].mean()),
            "retake_rate": float((sim["retakes"] > 0).mean()), "flag_rate_first_photo": float(Q[0].mean())}


def curve_rows(score, correct, meta):
    grid, risk = curve_on_grid(score, correct)
    return [{**meta, "coverage": round(float(g), 4), "risk": float(r)} for g, r in zip(grid, risk)]
