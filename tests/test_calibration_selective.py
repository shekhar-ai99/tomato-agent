import numpy as np

from tomato_agent.calibration import ece, fit_temperature, softmax
from tomato_agent.selective import aurc, risk_coverage, selective_accuracy


def test_temperature_recovers_known_value():
    rng = np.random.default_rng(0)
    N, K, T_true = 20000, 8, 2.5
    z = rng.normal(0, 3, (N, K))
    y = np.array([rng.choice(K, p=p) for p in softmax(z, T_true)])
    assert abs(fit_temperature(z, y) - T_true) < 0.15
    assert ece(softmax(z, T_true), y) < 0.02


def test_selective_metrics():
    correct = np.array([1, 1, 1, 0, 0], float)
    good, bad = np.array([5, 4, 3, 2, 1.]), np.array([1, 2, 3, 4, 5.])
    assert selective_accuracy(good, correct, 0.6) == 1.0
    assert selective_accuracy(bad, correct, 0.4) == 0.0
    assert aurc(good, correct) < aurc(bad, correct)
    cov, risk = risk_coverage(good, correct)
    assert cov[-1] == 1.0 and abs(risk[-1] - 0.4) < 1e-12
