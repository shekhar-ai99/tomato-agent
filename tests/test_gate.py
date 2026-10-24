import numpy as np
import pytest

from tomato_agent.gate import ACCEPT, DEFER, RETAKE, Costs, GateConfig, decide, outcome_metrics, select_thresholds, simulate

CFG = GateConfig(tau_accept=0.8, tau_defer=0.4, retake_budget=1)


@pytest.mark.parametrize("conf,q_ok,left,expected", [
    (0.95, True, 1, ACCEPT), (0.95, False, 1, RETAKE), (0.95, False, 0, ACCEPT),
    (0.30, True, 1, DEFER), (0.60, True, 1, RETAKE), (0.60, True, 0, DEFER), (0.30, False, 1, RETAKE),
])
def test_decide_table(conf, q_ok, left, expected):
    assert decide(conf, q_ok, left, CFG)[0] == expected


def loop_reference(probs_seq, q_seq, labels, cfg):
    """Case-by-case use of decide(): the definition simulate() must match."""
    acc, dfr, pred, retakes = [], [], [], []
    for i in range(len(labels)):
        used = 0
        for s in range(cfg.retake_budget + 1):
            p = probs_seq[s][i]
            a, _ = decide(p.max(), not q_seq[s][i], cfg.retake_budget - used, cfg)
            if a == RETAKE:
                used += 1
                continue
            break
        acc.append(a == ACCEPT)
        dfr.append(a == DEFER)
        pred.append(p.argmax())
        retakes.append(used)
    return np.array(acc), np.array(dfr), np.array(pred), np.array(retakes)


@pytest.mark.parametrize("R", [0, 1, 2])
@pytest.mark.parametrize("use_quality", [True, False])
def test_simulate_matches_decide(R, use_quality):
    rng = np.random.default_rng(R)
    N, K = 400, 8
    probs = [rng.dirichlet(np.ones(K) * 0.5, N) for _ in range(R + 1)]
    q = [rng.random(N) < 0.3 for _ in range(R + 1)]
    labels = rng.integers(0, K, N)
    cfg = GateConfig(0.7, 0.35, R, use_quality)
    sim = simulate(probs, q, labels, cfg)
    acc, dfr, pred, ret = loop_reference(probs, q if use_quality else [np.zeros(N, bool)] * (R + 1), labels, cfg)
    assert (sim["accepted"] == acc).all()
    assert (sim["deferred"] == dfr).all()
    assert (sim["pred"] == pred).all()
    assert (sim["retakes"] == ret).all()
    assert (sim["accepted"] | sim["deferred"]).all()


def test_answer_all_never_defers():
    rng = np.random.default_rng(0)
    probs = [rng.dirichlet(np.ones(8), 100) for _ in range(2)]
    sim = simulate(probs, [np.zeros(100, bool)] * 2, rng.integers(0, 8, 100), CFG, answer_all=True)
    assert not sim["deferred"].any() and sim["accepted"].all()


def test_selection_prefers_deferral_when_errors_are_expensive():
    rng = np.random.default_rng(1)
    N, K = 500, 8
    labels = rng.integers(0, K, N)
    p = rng.dirichlet(np.ones(K), N)
    p[np.arange(N), labels] += rng.random(N)  # confidence weakly related to correctness
    p /= p.sum(1, keepdims=True)
    q = [np.zeros(N, bool)] * 2
    cheap, _ = select_thresholds([p, p], q, labels, Costs(1.5, 1.0, 0.1))
    dear, _ = select_thresholds([p, p], q, labels, Costs(50, 1.0, 0.1))
    m_cheap = outcome_metrics(simulate([p, p], q, labels, cheap))
    m_dear = outcome_metrics(simulate([p, p], q, labels, dear))
    assert m_dear["deferral_rate"] >= m_cheap["deferral_rate"]
    assert m_dear["auto_error_rate"] <= m_cheap["auto_error_rate"]
