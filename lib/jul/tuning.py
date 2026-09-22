"""Optional per-task head, trained on the vectors the method already computes.

The LLM is never modified: the head is one linear layer over the frozen vectors, it trains in a few
seconds and costs well under a millisecond at inference. Measured (JOURNAL §6 and §9 ter): MiniCPM5-2B
on dair-ai/emotion goes from 0.549 zero-shot to 0.633 with 2000 training examples. The gain for
Qwen3.5-9B has never been measured, and its zero-shot is already much higher, so nothing is promised.

Safety net: the head is judged by stratified cross-validation, so every labeled example is predicted
by a head that never saw it. If the head does not beat the zero-shot method there, it is not
activated and the report says so. A single held-out fifth was too noisy to decide on: at 50 examples
it judged on ten, and measurably refused heads worth +12 points while accepting one worth -0.5
(JOURNAL §9 nonies).

A head only knows the options it was trained on, and only the preset whose vectors it saw: changing
either means tuning again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import fit_temperature, softmax

#: Below this, only calibration is fitted. A flat floor was wrong: measured on six model x dataset
#: curves (JOURNAL §9 nonies), 50 examples blocked a head worth +12 points on Qwen3.5-9B / AG News,
#: while 200 examples were not enough for Banking77's 72 options. What matters is examples per option,
#: plus enough total for the cross-validation to mean anything. Above this floor the gate decides.
def min_examples(n_options: int) -> int:
    return max(20, 3 * n_options)


@dataclass
class TuningReport:
    """What `client.autotune` gives back: what was fitted, and whether it was good enough to be used."""

    question: str
    n_examples: int
    n_evaluated: int
    n_options: int
    zero_shot_accuracy: float
    head_accuracy: float | None
    activated: bool
    reason: str
    temperature: float | None = None

    def __str__(self) -> str:
        head = "not trained" if self.head_accuracy is None else f"{self.head_accuracy:.3f}"
        state = "ACTIVE" if self.activated else "not used"
        return (f"question {self.question!r}: {self.n_examples} labeled examples, {self.n_options} options\n"
                f"  judged on            : {self.n_evaluated} (cross-validated)\n"
                f"  zero-shot accuracy   : {self.zero_shot_accuracy:.3f}\n"
                f"  tuned head accuracy  : {head}\n"
                f"  result               : {state} ({self.reason})")

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def stratified_folds(y: np.ndarray, n_folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Fold indices keeping each option's share, so every example gets an out-of-fold prediction."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for option in np.unique(y):
        idx = np.flatnonzero(y == option)
        rng.shuffle(idx)
        for position, i in enumerate(idx):
            folds[position % n_folds].append(int(i))
    return [np.array(sorted(f), dtype=int) for f in folds if f]


def stratified_split(y: np.ndarray, fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Indices (rest, held out), keeping each option's share and at least one example per option."""
    rng = np.random.default_rng(seed)
    held: list[int] = []
    for k in np.unique(y):
        idx = np.flatnonzero(y == k)
        rng.shuffle(idx)
        n = int(round(fraction * len(idx)))
        held += list(idx[: min(max(n, 1), len(idx) - 1)] if len(idx) > 1 else [])
    if not held:                      # every option had a single example
        held = [int(np.flatnonzero(y == np.unique(y)[0])[0])]
    held_set = set(held)
    rest = np.array([i for i in range(len(y)) if i not in held_set], dtype=int)
    return rest, np.array(sorted(held), dtype=int)


def _fit_linear(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, k: int,
                l2: float = 1e-2, epochs: int = 400, lr: float = 0.05, patience: int = 30,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression, Adam on cross-entropy, early stopping on validation NLL."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    W = rng.normal(0, 0.01, (d, k))
    b = np.zeros(k)
    onehot = np.eye(k)[y]
    mW, vW, mb, vb = (np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b))
    best, best_params, bad = np.inf, (W.copy(), b.copy()), 0
    for t in range(1, epochs + 1):
        g = (softmax(x @ W + b) - onehot) / n
        gW, gb = x.T @ g + l2 * W, g.sum(0)
        mW, vW = 0.9 * mW + 0.1 * gW, 0.999 * vW + 0.001 * gW * gW
        mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb * gb
        W -= lr * (mW / (1 - 0.9 ** t)) / (np.sqrt(vW / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
        p = softmax(xv @ W + b)
        val = float(-np.log(p[np.arange(len(yv)), yv] + 1e-12).mean())
        if val < best - 1e-5:
            best, best_params, bad = val, (W.copy(), b.copy()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_params


def train(features: np.ndarray, y: np.ndarray, zero_shot_scores: np.ndarray, option_keys: list[str],
          question: str, preset_name: str, seed: int = 0) -> tuple[dict | None, TuningReport]:
    """Fit a head on `features`, judge it against the zero-shot scores on held-out examples.

    features: (N, d) vectors produced by the preset's formulations, concatenated.
    zero_shot_scores: (N, K) cosine scores the method would have produced on its own.
    Returns (head payload or None, report).
    """
    features, y = np.asarray(features, dtype=np.float64), np.asarray(y)
    k = len(option_keys)
    zs_accuracy = float((zero_shot_scores.argmax(1) == y).mean())

    floor = min_examples(k)
    if len(features) < floor:
        return None, TuningReport(question, len(y), len(y), k, zs_accuracy, None, False,
                                  f"fewer than {floor} examples for {k} options: calibration only")
    if len(np.unique(y)) < k:
        return None, TuningReport(question, len(y), len(y), k, zs_accuracy, None, False,
                                  "some options have no training example")

    def fit(train_idx: np.ndarray, seed_offset: int = 0):
        inner, stop = stratified_split(y[train_idx], 0.2, seed + 1 + seed_offset)
        tr, st = train_idx[inner], train_idx[stop]
        mean, std = features[tr].mean(0), features[tr].std(0) + 1e-3
        norm = lambda a: (a - mean) / std
        W, b = _fit_linear(norm(features[tr]), y[tr], norm(features[st]), y[st], k, seed=seed)
        return W, b, mean, std, st

    # Judge by cross-validation: every example is predicted by a head that never saw it.
    smallest = int(np.bincount(y, minlength=k).min())
    folds = stratified_folds(y, max(2, min(5, smallest)), seed)
    out_of_fold = np.zeros(len(y), dtype=int)
    for f, fold in enumerate(folds):
        rest = np.setdiff1d(np.arange(len(y)), fold)
        if len(np.unique(y[rest])) < k:
            continue
        W, b, mean, std, _ = fit(rest, seed_offset=f)
        out_of_fold[fold] = (((features[fold] - mean) / std) @ W + b).argmax(1)
    head_accuracy = float((out_of_fold == y).mean())

    W, b, mean, std, stop_idx = fit(np.arange(len(y)))
    temperature = fit_temperature(((features[stop_idx] - mean) / std) @ W + b, y[stop_idx])

    activated = head_accuracy > zs_accuracy
    reason = (f"beats zero-shot in {len(folds)}-fold cross-validation (+{head_accuracy - zs_accuracy:.3f})"
              if activated else
              f"does not beat zero-shot in {len(folds)}-fold cross-validation "
              f"({head_accuracy - zs_accuracy:+.3f})")
    payload = {
        "W": W, "b": b, "mean": mean, "std": std,
        "temperature": np.array([temperature]),
        "meta": {"question": question, "preset": preset_name, "options": option_keys,
                 "n_examples": int(len(y)), "n_folds": len(folds),
                 "zero_shot_accuracy": zs_accuracy, "head_accuracy": head_accuracy,
                 "activated": bool(activated)},
    }
    report = TuningReport(question, len(y), len(y), k, zs_accuracy, head_accuracy, activated, reason,
                          temperature)
    return (payload if activated else None), report


def apply(payload: dict, features: np.ndarray) -> np.ndarray:
    """Probabilities from a trained head, for one example (d,) or a batch (N, d)."""
    x = (np.asarray(features, dtype=np.float64) - payload["mean"]) / payload["std"]
    logits = (x @ payload["W"] + payload["b"]) / float(np.asarray(payload["temperature"]).ravel()[0])
    if logits.ndim == 1:
        return softmax(logits[None])[0]
    return softmax(logits)
