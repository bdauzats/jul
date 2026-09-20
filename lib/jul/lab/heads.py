"""Trainable heads on top of frozen backbone features (MLX).

- LinearProbe (approach B): one layer's hidden state -> K fixed labels.
- HybridHead (approach B+C): query and label are both encoded by the backbone, projected into a
  shared space and compared by scaled cosine. Labels are inputs, not weights, so new labels can be
  added without retraining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


class Normalizer:
    """Per-(layer, dim) standardization fitted on training queries. Tames LLM outlier dimensions."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mx.array(mean, dtype=mx.float32)
        self.std = mx.array(std, dtype=mx.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        x = x.astype(np.float32)
        return cls(x.mean(0), x.std(0) + 1e-3)

    def __call__(self, x) -> mx.array:
        return (mx.array(x).astype(mx.float32) - self.mean) / self.std


class LinearProbe(nn.Module):
    def __init__(self, d: int, k: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, k)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out(self.drop(x))


class HybridHead(nn.Module):
    def __init__(self, n_layers: int, d: int, dim: int = 256, hidden: int = 1024, dropout: float = 0.1,
                 tied: bool = True):
        super().__init__()
        self.tied = tied
        self.q_mix = mx.zeros((n_layers,))
        self.l_mix = mx.zeros((n_layers,))
        self.l_shift = mx.zeros((d,))  # aligns the label-prompt distribution with the query one
        self.drop = nn.Dropout(dropout)
        self.q_proj = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, dim))
        if not tied:
            self.l_proj = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, dim))
            self.l_proj.update(self.q_proj.parameters())
        self.log_scale = mx.array(math.log(10.0))

    @staticmethod
    def _mix(w: mx.array, x: mx.array) -> mx.array:
        return (mx.softmax(w)[:, None] * x).sum(-2)

    def embed_queries(self, xq: mx.array) -> mx.array:
        z = self.q_proj(self.drop(self._mix(self.q_mix, xq)))
        return z / mx.linalg.norm(z, axis=-1, keepdims=True)

    def embed_labels(self, xl: mx.array) -> mx.array:
        proj = self.q_proj if self.tied else self.l_proj
        z = proj(self._mix(self.l_mix, xl) + self.l_shift)
        return z / mx.linalg.norm(z, axis=-1, keepdims=True)

    def __call__(self, xq: mx.array, xl: mx.array) -> mx.array:
        return mx.exp(self.log_scale) * self.embed_queries(xq) @ self.embed_labels(xl).T


@dataclass
class TrainConfig:
    epochs: int = 60
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 8
    seed: int = 0


def _fit(model: nn.Module, forward, x_train, y_train, x_val, y_val, cfg: TrainConfig, verbose=False):
    """Minibatch AdamW on cross-entropy with early stopping on validation NLL. Returns best val NLL."""
    mx.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)

    def loss_fn(m, xb, yb):
        return nn.losses.cross_entropy(forward(m, xb), yb, reduction="mean")

    step = nn.value_and_grad(model, loss_fn)
    y_train_mx, y_val_mx = mx.array(y_train), mx.array(y_val)
    best, best_params, bad = float("inf"), None, 0
    for epoch in range(cfg.epochs):
        model.train(True)
        order = rng.permutation(len(y_train))
        for s in range(0, len(order), cfg.batch):
            idx = mx.array(order[s:s + cfg.batch])
            loss, grads = step(model, x_train[idx], y_train_mx[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
        model.train(False)
        val = float(loss_fn(model, x_val, y_val_mx))
        if verbose:
            print(f"    epoch {epoch:3d}  val_nll={val:.4f}")
        if val < best - 1e-4:
            best, best_params, bad = val, dict(tree_flatten(model.parameters())), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_weights(list(best_params.items()))
    model.train(False)
    return best


def train_probe(x_train, y_train, x_val, y_val, k: int, cfg: TrainConfig = TrainConfig()):
    """x_*: (N, d) already-normalized mx arrays."""
    model = LinearProbe(x_train.shape[-1], k)
    _fit(model, lambda m, xb: m(xb), x_train, y_train, x_val, y_val, cfg)
    return model


def train_hybrid(x_train, y_train, x_val, y_val, label_feats, cfg: TrainConfig = TrainConfig(), **head_kw):
    """x_*: (N, L, d) normalized queries; label_feats: (K, L, d) normalized labels (training labels only)."""
    model = HybridHead(x_train.shape[1], x_train.shape[-1], **head_kw)
    _fit(model, lambda m, xb: m(xb, label_feats), x_train, y_train, x_val, y_val, cfg)
    return model
