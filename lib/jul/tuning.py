"""Optional per-task head, trained on the vectors the method already computes.

The LLM is never modified: the head is one linear layer over the frozen vectors, it trains in a few
seconds and costs well under a millisecond at inference. Measured: MiniCPM5-2B
on dair-ai/emotion goes from 0.549 zero-shot to 0.633 with 2000 training examples. The gain for
Qwen3.5-9B has never been measured, and its zero-shot is already much higher, so nothing is promised.

Safety net: the head is judged by stratified cross-validation, so every labeled example is predicted
by a head that never saw it. If the head does not beat the zero-shot method there, it is not
activated and the report says so. A single held-out fifth was too noisy to decide on: at 50 examples
it judged on ten, and measurably refused heads worth +12 points while accepting one worth -0.5.

A head only knows the options it was trained on, and only the preset whose vectors it saw: changing
either means tuning again.

What the head reads is chosen with `features`:
- "vector" (the default): the vectors of the preset's formulations, as above;
- "lexical": TF-IDF of the text alone (jul/lexical.py), no model vectors;
- "hybrid": both, in one head. The vectors are standardized and scaled by a weight chosen with the
  regularization, so neither part drowns the other.
The last two train with scikit-learn (`pip install 'jul[tune]'`) and run with numpy alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import fit_temperature, softmax

FEATURES = ("vector", "lexical", "hybrid")

#: Below this, only calibration is fitted. A flat floor was wrong: measured on six model x dataset
#: curves, 50 examples blocked a head worth +12 points on Qwen3.5-9B / AG News,
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
    features: str = "vector"

    def __str__(self) -> str:
        head = "not trained" if self.head_accuracy is None else f"{self.head_accuracy:.3f}"
        state = "ACTIVE" if self.activated else "not used"
        return (f"question {self.question!r}: {self.n_examples} labeled examples, {self.n_options} options, "
                f"{self.features} features\n"
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
          question: str, preset_name: str, seed: int = 0, texts: list[str] | None = None,
          mode: str = "vector") -> tuple[dict | None, TuningReport]:
    """Fit a head, judge it against the zero-shot scores on held-out examples.

    features: (N, d) vectors produced by the preset's formulations, concatenated.
    zero_shot_scores: (N, K) cosine scores the method would have produced on its own.
    texts: the N texts, needed by the "lexical" and "hybrid" modes.
    Returns (head payload or None, report).
    """
    if mode not in FEATURES:
        raise ValueError(f"features must be one of {', '.join(FEATURES)} (got {mode!r})")
    if mode != "vector" and texts is None:
        raise ValueError(f"{mode!r} features need the texts")
    features, y = np.asarray(features, dtype=np.float64), np.asarray(y)
    k = len(option_keys)
    zs_accuracy = float((zero_shot_scores.argmax(1) == y).mean())

    floor = min_examples(k)
    if len(features) < floor:
        return None, TuningReport(question, len(y), len(y), k, zs_accuracy, None, False,
                                  f"fewer than {floor} examples for {k} options: calibration only", features=mode)
    if len(np.unique(y)) < k:
        return None, TuningReport(question, len(y), len(y), k, zs_accuracy, None, False,
                                  "some options have no training example", features=mode)
    if mode != "vector":
        return _train_sparse(features, list(texts), y, zs_accuracy, option_keys, question, preset_name, seed, mode)

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
        "mode": np.array("vector"), "W": W, "b": b, "mean": mean, "std": std,
        "temperature": np.array([temperature]),
        "meta": {"question": question, "preset": preset_name, "options": option_keys,
                 "n_examples": int(len(y)), "n_folds": len(folds),
                 "zero_shot_accuracy": zs_accuracy, "head_accuracy": head_accuracy,
                 "activated": bool(activated)},
    }
    report = TuningReport(question, len(y), len(y), k, zs_accuracy, head_accuracy, activated, reason,
                          temperature)
    return (payload if activated else None), report


#: Grids searched on an inner split before the cross-validation: the inverse regularization C, and
#: for "hybrid" the weight of the standardized vectors next to the L2-normalized TF-IDF rows. The
#: jul-lambda runs chose w 0.02-0.1 and C 1-30 on emotion and Banking77.
GRID_C = (1.0, 3.0, 10.0, 30.0)
GRID_W = (0.02, 0.05, 0.1, 0.3)


def _logistic():
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError("lexical and hybrid heads train with scikit-learn: pip install 'jul[tune]'") from exc
    return LogisticRegression


def _full_logits(logits: np.ndarray) -> np.ndarray:
    """scikit-learn gives one column for two classes: softmax([0, z]) is its sigmoid."""
    return np.stack([np.zeros_like(logits), logits], 1) if logits.ndim == 1 else logits


def _train_sparse(features, texts, y, zs_accuracy, option_keys, question, preset_name, seed, mode):
    import scipy.sparse as sp

    from .lexical import Tfidf
    LogisticRegression = _logistic()
    k = len(option_keys)

    def design(tfidf, idx, mean, std, w):
        rows = tfidf.transform([texts[i] for i in idx]).scipy()
        if mode == "lexical":
            return rows
        return sp.hstack([sp.csr_matrix((features[idx] - mean) / std * w), rows]).tocsr()

    def fit(idx, w, c):
        tfidf = Tfidf.fit([texts[i] for i in idx])
        mean, std = features[idx].mean(0), features[idx].std(0) + 1e-3
        clf = LogisticRegression(C=c, max_iter=3000).fit(design(tfidf, idx, mean, std, w), y[idx])
        return clf, tfidf, mean, std

    rest, held = stratified_split(y, 0.2, seed + 1)
    best = None
    for w in (GRID_W if mode == "hybrid" else (0.0,)):
        for c in GRID_C:
            clf, tfidf, mean, std = fit(rest, w, c)
            acc = float((clf.predict(design(tfidf, held, mean, std, w)) == y[held]).mean())
            if best is None or acc > best[0]:
                best = (acc, w, c)
    _, w, c = best

    smallest = int(np.bincount(y, minlength=k).min())
    folds = stratified_folds(y, max(2, min(5, smallest)), seed)
    oof = np.zeros((len(y), k))
    for fold in folds:
        rest_f = np.setdiff1d(np.arange(len(y)), fold)
        if len(np.unique(y[rest_f])) < k:
            continue
        clf, tfidf, mean, std = fit(rest_f, w, c)
        oof[fold] = _full_logits(clf.decision_function(design(tfidf, fold, mean, std, w)))
    head_accuracy = float((oof.argmax(1) == y).mean())
    # every example was predicted by a head that never saw it: calibrate on those logits
    temperature = fit_temperature(oof, y)

    clf, tfidf, mean, std = fit(np.arange(len(y)), w, c)
    coef = clf.coef_ if clf.coef_.shape[0] == k else np.concatenate([np.zeros_like(clf.coef_), clf.coef_])
    bias = clf.intercept_ if len(clf.intercept_) == k else np.concatenate([[0.0], clf.intercept_])
    d = features.shape[1] if mode == "hybrid" else 0
    activated = head_accuracy > zs_accuracy
    reason = (f"beats zero-shot in {len(folds)}-fold cross-validation (+{head_accuracy - zs_accuracy:.3f})"
              if activated else
              f"does not beat zero-shot in {len(folds)}-fold cross-validation ({head_accuracy - zs_accuracy:+.3f})")
    payload = {
        "mode": np.array(mode), "b": bias, "temperature": np.array([temperature]),
        "W_lex": coef[:, d:].T.astype(np.float32), **tfidf.to_arrays(),
        "meta": {"question": question, "preset": preset_name, "options": option_keys, "features": mode,
                 "C": c, "vector_weight": w, "n_examples": int(len(y)), "n_folds": len(folds),
                 "zero_shot_accuracy": zs_accuracy, "head_accuracy": head_accuracy,
                 "activated": bool(activated)},
    }
    if mode == "hybrid":
        # the head saw (x - mean) / std * w: fold w into the weights, apply() reads (x - mean) / std
        payload.update({"W": coef[:, :d].T * w, "mean": mean, "std": std})
    report = TuningReport(question, len(y), len(y), k, zs_accuracy, head_accuracy, activated, reason,
                          temperature, features=mode)
    return (payload if activated else None), report


def apply(payload: dict, features: np.ndarray, texts: str | list[str] | None = None) -> np.ndarray:
    """Probabilities from a trained head, for one example (d,) or a batch (N, d).

    A "lexical" or "hybrid" head also reads the text(s): one string, or a list for a batch.
    """
    mode = str(payload.get("mode", "vector"))
    single = np.asarray(features).ndim == 1
    logits = np.asarray(payload["b"], dtype=np.float64)
    if mode in ("vector", "hybrid"):
        x = (np.atleast_2d(np.asarray(features, dtype=np.float64)) - payload["mean"]) / payload["std"]
        logits = logits + x @ payload["W"]
    if mode in ("lexical", "hybrid"):
        if texts is None:
            raise ValueError(f"a {mode!r} head needs the text")
        tfidf = payload.get("_tfidf")
        if tfidf is None:
            from .lexical import Tfidf
            tfidf = payload["_tfidf"] = Tfidf.from_arrays(payload)
        logits = logits + tfidf.transform([texts] if isinstance(texts, str) else list(texts)).dot(payload["W_lex"])
    logits = np.atleast_2d(logits) / float(np.asarray(payload["temperature"]).ravel()[0])
    probabilities = softmax(logits)
    return probabilities[0] if single else probabilities
