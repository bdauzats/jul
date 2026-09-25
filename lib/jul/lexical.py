"""Lexical features for the tuned heads: TF-IDF over words and character n-grams, numpy only.

What the vectors of a model miss, the words often carry: on dair-ai/emotion, a head on Harrier 0.6B
vectors alone scored 0.734 and TF-IDF alone 0.751, while one head over both scored 0.797 (jul-lambda,
2026-09-25). This module turns texts into the sparse TF-IDF rows such a head reads.

Two analyzers, each L2-normalized on its own and then concatenated (scikit-learn's
FeatureUnion of two TfidfVectorizer): word 1-2 grams, and character 2-5 grams inside word
boundaries (`char_wb`). Sublinear term frequency, smoothed idf. Fitting keeps the `max_features`
most frequent terms of each analyzer. The vocabulary and idf serialize to plain arrays, so a head
that uses them saves and loads with the rest of a context and needs no scikit-learn at inference.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

_WORD = re.compile(r"(?u)\b\w\w+\b")
_SPACE = re.compile(r"\s\s+")


def _word_terms(text: str) -> list[str]:
    tokens = _WORD.findall(text.lower())
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def _char_terms(text: str, lo: int = 2, hi: int = 5) -> list[str]:
    terms = []
    for word in _SPACE.sub(" ", text.lower()).split():
        padded = f" {word} "
        for n in range(lo, hi + 1):
            if n > len(padded):
                break
            terms += [padded[i:i + n] for i in range(len(padded) - n + 1)]
    return terms


ANALYZERS = {"word": _word_terms, "char": _char_terms}


@dataclass
class Rows:
    """Sparse rows in CSR form: row i holds `indices[indptr[i]:indptr[i+1]]` with those `data`."""

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    n_features: int

    def __len__(self) -> int:
        return len(self.indptr) - 1

    def scipy(self):
        from scipy.sparse import csr_matrix
        return csr_matrix((self.data, self.indices, self.indptr), shape=(len(self), self.n_features))

    def dot(self, weights: np.ndarray) -> np.ndarray:
        """(n_rows, k) = rows @ weights, for weights (n_features, k)."""
        out = np.zeros((len(self), weights.shape[1]))
        for i in range(len(self)):
            a, b = self.indptr[i], self.indptr[i + 1]
            if b > a:
                out[i] = self.data[a:b] @ weights[self.indices[a:b]]
        return out


@dataclass
class Tfidf:
    """Fitted vocabulary and idf of each analyzer, in the order of their concatenation."""

    vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    idf: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def fit(cls, texts: list[str], max_features: int = 50_000) -> "Tfidf":
        model = cls()
        n = len(texts)
        for name, analyzer in ANALYZERS.items():
            df = Counter()
            for text in texts:
                df.update(set(analyzer(text)))
            kept = sorted(df, key=lambda t: (-df[t], t))[:max_features]
            kept.sort()
            model.vocab[name] = {t: i for i, t in enumerate(kept)}
            model.idf[name] = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in kept])
        return model

    @property
    def n_features(self) -> int:
        return sum(len(v) for v in self.vocab.values())

    def transform(self, texts: list[str]) -> Rows:
        data, indices, indptr, nnz = [], [], [0], 0
        for text in texts:
            offset = 0
            for name, analyzer in ANALYZERS.items():
                vocab, idf = self.vocab[name], self.idf[name]
                counts = Counter(t for t in analyzer(text) if t in vocab)
                if counts:
                    idx = np.array([vocab[t] for t in counts])
                    val = (1 + np.log(np.array(list(counts.values()), dtype=np.float64))) * idf[idx]
                    val /= np.linalg.norm(val)
                    order = np.argsort(idx)
                    indices.append(idx[order] + offset)
                    data.append(val[order])
                    nnz += len(idx)
                offset += len(vocab)
            indptr.append(nnz)
        return Rows(np.concatenate(data) if data else np.zeros(0),
                    np.concatenate(indices) if indices else np.zeros(0, dtype=int),
                    np.array(indptr), self.n_features)

    # --- serialization, as arrays for np.savez ----------------------------------------------------

    def to_arrays(self, prefix: str = "lex_") -> dict[str, np.ndarray]:
        out = {}
        for name in ANALYZERS:
            terms = sorted(self.vocab[name], key=self.vocab[name].get)
            out[f"{prefix}{name}_terms"] = np.array(terms, dtype=str)
            out[f"{prefix}{name}_idf"] = self.idf[name]
        return out

    @classmethod
    def from_arrays(cls, arrays: dict, prefix: str = "lex_") -> "Tfidf":
        model = cls()
        for name in ANALYZERS:
            terms = [str(t) for t in arrays[f"{prefix}{name}_terms"]]
            model.vocab[name] = {t: i for i, t in enumerate(terms)}
            model.idf[name] = np.asarray(arrays[f"{prefix}{name}_idf"], dtype=np.float64)
        return model
