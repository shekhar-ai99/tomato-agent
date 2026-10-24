"""Temperature scaling and calibration metrics (paper Sections 3.3 and 4.5)."""
import numpy as np


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / float(T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, labels, T=1.0):
    p = softmax(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(logits, labels, lo=0.05, hi=20.0):
    """T minimising validation NLL (Guo et al., 2017). 1-D bounded search over log T."""
    from scipy.optimize import minimize_scalar
    logits, labels = np.asarray(logits), np.asarray(labels, dtype=int)
    if len(labels) == 0:
        return 1.0
    res = minimize_scalar(lambda lt: nll(logits, labels, np.exp(lt)),
                          bounds=(np.log(lo), np.log(hi)), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(res.x))


def ece(probs, labels, n_bins=15):
    """Expected calibration error with equal-width confidence bins (top-label)."""
    probs, labels = np.asarray(probs), np.asarray(labels, dtype=int)
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def brier(probs, labels):
    probs, labels = np.asarray(probs), np.asarray(labels, dtype=int)
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def reliability_bins(probs, labels, n_bins=15):
    """Per-bin (mean confidence, accuracy, count) for reliability diagrams (paper Figure 4)."""
    probs, labels = np.asarray(probs), np.asarray(labels, dtype=int)
    conf, correct = probs.max(1), (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "count": int(m.sum()),
                     "conf": float(conf[m].mean()) if m.any() else np.nan,
                     "acc": float(correct[m].mean()) if m.any() else np.nan})
    return rows


def entropy(probs):
    p = np.clip(np.asarray(probs), 1e-12, 1)
    return -(p * np.log(p)).sum(1)


def calib_summary(logits, labels, T):
    p0, p1 = softmax(logits, 1.0), softmax(logits, T)
    return {"n": int(len(labels)), "nll_before": nll(logits, labels, 1.0), "nll_after": nll(logits, labels, T),
            "ece_before": ece(p0, labels), "ece_after": ece(p1, labels),
            "brier_before": brier(p0, labels), "brier_after": brier(p1, labels)}
