"""Generic head for the vector method: a small low-rank correction of the text and label vectors.

    score(text, label) = cos(f_text(t), f_label(l)) / tau,   f(v) = v + up(down(v))

`up` starts at zero, so an untrained head is exactly the plain vector method. It is trained across many
datasets (one label set per batch) and judged on datasets it never saw, so that it learns to align a
text with the right option in general, not the labels of one task.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


def unit(v: mx.array) -> mx.array:
    return v / mx.linalg.norm(v, axis=-1, keepdims=True)


class GenericHead(nn.Module):
    def __init__(self, d: int, rank: int = 64, dropout: float = 0.1, tau: float = 0.0456):
        super().__init__()
        self.rank = rank
        self.t_down, self.t_up = nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False)
        self.l_down, self.l_up = nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False)
        self.t_up.weight = mx.zeros_like(self.t_up.weight)
        self.l_up.weight = mx.zeros_like(self.l_up.weight)
        self.drop = nn.Dropout(dropout)
        self.log_tau = mx.array(math.log(tau))

    def f_text(self, v: mx.array) -> mx.array:
        v = unit(v)
        return unit(v + self.t_up(self.drop(self.t_down(v))))

    def f_label(self, v: mx.array) -> mx.array:
        v = unit(v)
        return unit(v + self.l_up(self.l_down(v)))

    def __call__(self, texts: mx.array, labels: mx.array) -> mx.array:
        """texts (B, d), labels (K, d), both already centered -> logits (B, K)."""
        return self.f_text(texts) @ self.f_label(labels).T / mx.exp(self.log_tau)

    def save(self, path: Path, meta: dict):
        path.mkdir(parents=True, exist_ok=True)
        self.save_weights(str(path / "head.safetensors"))
        (path / "config.json").write_text(json.dumps({"rank": self.rank, "d": self.t_down.weight.shape[1], **meta}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GenericHead":
        cfg = json.loads((path / "config.json").read_text())
        head = cls(cfg["d"], cfg["rank"])
        head.load_weights(str(path / "head.safetensors"))
        head.train(False)
        return head


def accuracy(head: GenericHead, tasks: list[dict]) -> list[float]:
    """tasks: [{"X": (N, d), "L": (K, d), "y": (N,)}] with centered vectors."""
    head.train(False)
    return [float((np.array(head(mx.array(t["X"]), mx.array(t["L"]))).argmax(1) == t["y"]).mean()) for t in tasks]


def train(train_tasks: list[dict], val_tasks: list[dict], rank=64, dropout=0.1, lr=1e-3, weight_decay=0.05,
          epochs=30, batch=32, patience=5, seed=0, eval_every: int | None = None, log=print) -> tuple[GenericHead, list[dict]]:
    """Batches never mix datasets (each has its own label set). Keeps the checkpoint with the best mean
    validation accuracy on held-out datasets; checkpoint 0 is the untrained head (plain vector method).
    Validation runs every `eval_every` steps (default: once per epoch); patience counts validations."""
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    d = train_tasks[0]["X"].shape[1]
    head = GenericHead(d, rank, dropout)
    opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(m, x, labels, y):
        return nn.losses.cross_entropy(m(x, labels), y, reduction="mean")

    step = nn.value_and_grad(head, loss_fn)
    tasks = [{"X": mx.array(t["X"]), "L": mx.array(t["L"]), "y": mx.array(t["y"])} for t in train_tasks]
    history = [{"step": 0, "val": accuracy(head, val_tasks)}]
    best = (np.mean(history[0]["val"]), 0, dict(tree_flatten(head.parameters())))
    log(f"    step    0  val {np.mean(history[0]['val']):.3f}  (untrained = plain vector method)")
    bad, n_step, losses = 0, 0, []
    for epoch in range(1, epochs + 1):
        batches = [(ti, idx) for ti, t in enumerate(tasks)
                   for idx in np.array_split(rng.permutation(t["X"].shape[0]), max(1, t["X"].shape[0] // batch))]
        rng.shuffle(batches)
        every = eval_every or len(batches)
        for ti, idx in batches:
            head.train(True)
            t, i = tasks[ti], mx.array(idx)
            loss, grads = step(head, t["X"][i], t["L"], t["y"][i])
            opt.update(head, grads)
            mx.eval(head.parameters(), opt.state)
            losses.append(float(loss))
            n_step += 1
            if n_step % every:
                continue
            val = accuracy(head, val_tasks)
            history.append({"step": n_step, "epoch": epoch, "train_loss": float(np.mean(losses)), "val": val})
            log(f"    step {n_step:4d}  train loss {np.mean(losses):.3f}  val {np.mean(val):.3f}  " + " ".join(f"{a:.2f}" for a in val))
            losses = []
            if np.mean(val) > best[0] + 1e-4:
                best, bad = (np.mean(val), n_step, dict(tree_flatten(head.parameters()))), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if bad >= patience:
            break
    head.load_weights(list(best[2].items()))
    head.train(False)
    log(f"    kept step {best[1]} (val {best[0]:.3f})")
    return head, history
