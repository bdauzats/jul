"""Temperature scaling and calibration metrics (numpy)."""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """T minimizing validation NLL of softmax(logits / T): coarse log-grid, then a local refinement."""
    grid = np.exp(np.linspace(np.log(0.02), np.log(50), 200))
    best = min(grid, key=lambda t: nll(logits / t, y))
    fine = np.exp(np.linspace(np.log(best) - 0.05, np.log(best) + 0.05, 101))
    return float(min(fine, key=lambda t: nll(logits / t, y)))


def reliability(probs: np.ndarray, y: np.ndarray, bins: int = 10):
    conf = probs.max(-1)
    correct = probs.argmax(-1) == y
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return rows


def metrics(logits: np.ndarray, y: np.ndarray, thresholds=(0.8, 0.9, 0.95)) -> dict:
    probs = softmax(logits)
    conf = probs.max(-1)
    correct = probs.argmax(-1) == y
    ece = sum(n / len(y) * abs(acc - c) for _, _, n, c, acc in reliability(probs, y, bins=15))
    onehot = np.eye(probs.shape[1])[y]
    out = {
        "acc": float(correct.mean()),
        "conf": float(conf.mean()),
        "nll": nll(logits, y),
        "ece": float(ece),
        "brier": float(((probs - onehot) ** 2).sum(-1).mean()),
    }
    # "Act if confident, escalate otherwise": share of inputs handled automatically, and their accuracy.
    for t in thresholds:
        m = conf >= t
        out[f"cover@{t}"] = float(m.mean())
        out[f"acc@{t}"] = float(correct[m].mean()) if m.any() else float("nan")
    return out


def fit_temperature_bias(logits: np.ndarray, y: np.ndarray, l2: float = 0.01, steps: int = 3000,
                         lr: float = 0.05) -> tuple[float, np.ndarray]:
    """Fit softmax(logits / T + b): a temperature plus one bias per option (Adam on validation NLL).

    Temperature fixes over/under-confidence; the bias fixes a systematic preference for some options
    (e.g. a zero-shot model that leans towards "no"), which also moves the decision threshold.
    """
    n, k = logits.shape
    onehot = np.eye(k)[y]
    s, b = 1.0 / fit_temperature(logits, y), np.zeros(k)
    m, v = np.zeros(k + 1), np.zeros(k + 1)
    for t in range(1, steps + 1):
        g_z = (softmax(logits * s + b) - onehot) / n
        g = np.concatenate([[(g_z * logits).sum()], g_z.sum(0) + l2 * b])
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        step = lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        s, b = max(s - step[0], 1e-3), b - step[1:]
    return 1.0 / s, b - b.mean()
