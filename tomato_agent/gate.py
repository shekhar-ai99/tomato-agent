"""Verification agent: the confidence-gated accept / retake / defer policy (paper Section 3.4).

One step of the policy, for the current photo of a case:
  1. if the quality agent flags the photo and retakes remain           -> RETAKE (with the quality reason)
  2. else if confidence >= tau_accept                                    -> ACCEPT
  3. else if confidence < tau_defer, or no retakes remain                -> DEFER to the expert
  4. else (tau_defer <= confidence < tau_accept, retakes remain)         -> RETAKE (medium confidence)

Thresholds are chosen on validation data by minimising
  cost = C_err * P(accepted and wrong) + C_def * P(deferred) + C_ret * E[retakes per case].
The Android app implements the same `decide` (Gate.kt).
"""
from dataclasses import asdict, dataclass

import numpy as np

ACCEPT, RETAKE, DEFER = "accept", "retake", "defer"


@dataclass
class GateConfig:
    tau_accept: float = 0.90
    tau_defer: float = 0.50
    retake_budget: int = 1
    use_quality: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class Costs:
    c_err: float = 10.0
    c_def: float = 1.0
    c_ret: float = 0.1

    @property
    def label(self):
        return f"{self.c_err:g}:{self.c_def:g}"


def decide(confidence, quality_ok, retakes_left, cfg):
    """One policy step. Returns (action, reason)."""
    if cfg.use_quality and not quality_ok and retakes_left > 0:
        return RETAKE, "quality"
    if confidence >= cfg.tau_accept:
        return ACCEPT, "confident"
    if confidence < cfg.tau_defer:
        return DEFER, "low_confidence"
    if retakes_left <= 0:
        return DEFER, "retakes_exhausted"
    return RETAKE, "medium_confidence"


def simulate(probs_seq, qflag_seq, labels, cfg, answer_all=False):
    """Run the policy on a batch of cases.

    probs_seq: list of (N, K) calibrated probabilities, one per shot (shot 0 = first photo,
               shot s = s-th retake). Needs at least cfg.retake_budget + 1 entries.
    qflag_seq: list of (N,) bool, True = quality agent flags that shot.
    answer_all: never defer (used for 'accuracy over all cases' and for RQ2's quality-only rows):
               the final shot's prediction is always returned.
    """
    labels = np.asarray(labels, dtype=int)
    N, R = len(labels), int(cfg.retake_budget)
    state = np.zeros(N, np.int8)            # 0 active, 1 accepted, 2 deferred
    final = np.full(N, -1)
    retakes = np.zeros(N, np.int16)
    reason = np.empty(N, dtype=object)
    for s in range(R + 1):
        active = state == 0
        if not active.any():
            break
        p = probs_seq[min(s, len(probs_seq) - 1)]
        conf = p.max(1)
        q = np.asarray(qflag_seq[min(s, len(qflag_seq) - 1)], bool) if cfg.use_quality else np.zeros(N, bool)
        left = R - s
        r_quality = active & q & (left > 0)
        rest = active & ~r_quality
        if answer_all:
            acc = rest & ((conf >= cfg.tau_accept) | (left <= 0))
            dfr = np.zeros(N, bool)
        else:
            acc = rest & (conf >= cfg.tau_accept)
            dfr = rest & ~acc & ((conf < cfg.tau_defer) | (left <= 0))
        r_conf = rest & ~acc & ~dfr
        state[acc], state[dfr] = 1, 2
        final[acc | dfr] = s
        reason[acc] = "confident"
        reason[dfr & (conf < cfg.tau_defer)] = "low_confidence"
        reason[dfr & (conf >= cfg.tau_defer)] = "retakes_exhausted"
        retakes[r_quality | r_conf] += 1
    final[final < 0] = R
    P = np.stack([probs_seq[min(s, len(probs_seq) - 1)] for s in range(R + 1)])  # (R+1, N, K)
    pf = P[final, np.arange(N)]
    qf = np.stack([np.asarray(qflag_seq[min(s, len(qflag_seq) - 1)], bool) for s in range(R + 1)])[final, np.arange(N)]
    pred, conf = pf.argmax(1), pf.max(1)
    return {"accepted": state == 1, "deferred": state == 2, "pred": pred, "conf": conf,
            "correct": pred == labels, "retakes": retakes, "final_shot": final,
            "final_flagged": qf & cfg.use_quality, "reason": reason}


def outcome_metrics(sim, costs=None):
    acc, dfr, corr = sim["accepted"], sim["deferred"], sim["correct"]
    N = len(acc)
    n_acc = int(acc.sum())
    out = {
        "n": N,
        "coverage": float(acc.mean()),
        "deferral_rate": float(dfr.mean()),
        "retake_rate": float((sim["retakes"] > 0).mean()),
        "retakes_per_case": float(sim["retakes"].mean()),
        "selective_accuracy": float(corr[acc].mean()) if n_acc else float("nan"),
        "auto_error_rate": float((acc & ~corr).mean()),
        "system_accuracy_oracle": float(((acc & corr) | dfr).mean()),  # expert = ground truth
        "final_accuracy_all": float(corr.mean()),
        "retake_rescued": float(((sim["retakes"] > 0) & acc & corr).mean()),
    }
    if costs is not None:
        out["expected_cost"] = (costs.c_err * out["auto_error_rate"] + costs.c_def * out["deferral_rate"]
                       + costs.c_ret * out["retakes_per_case"])
    return out


def proposed_score(sim):
    """Ranking score for the gated pipeline's risk-coverage curve: final calibrated confidence,
    with photos still flagged by the quality agent ranked last."""
    return sim["conf"] - 1.0 * sim["final_flagged"]


def threshold_grid(n=40):
    return np.unique(np.round(np.concatenate([np.linspace(0.30, 0.95, n), [0.97, 0.98, 0.99, 0.995, 0.999]]), 4))


def select_thresholds(probs_seq, qflag_seq, labels, costs, retake_budget=1, use_quality=True, grid=None):
    """Grid search for (tau_accept, tau_defer) minimising expected cost on validation data."""
    grid = threshold_grid() if grid is None else grid
    best = None
    for ta in grid:
        for td in grid[grid <= ta]:
            cfg = GateConfig(float(ta), float(td), retake_budget, use_quality)
            m = outcome_metrics(simulate(probs_seq, qflag_seq, labels, cfg), costs)
            key = (round(m["expected_cost"], 9), -m["coverage"], m["retakes_per_case"])
            if best is None or key < best[0]:
                best = (key, cfg, m)
    return best[1], best[2]
