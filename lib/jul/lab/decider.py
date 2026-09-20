"""Inference API: text in, calibrated probability per label out. One (possibly truncated) forward pass."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..backbone import Backbone
from .heads import HybridHead, LinearProbe, Normalizer
from .primitives import FORMATTERS
from .task import Label, Prompts, Task


def _probs(logits: mx.array, labels: list[str], T: float) -> dict[str, float]:
    p = np.array(mx.softmax(logits / T))
    return dict(sorted(zip(labels, p.tolist()), key=lambda kv: -kv[1]))


class LogitDecider:
    """Approach A: read the probability of each option letter. No training."""

    def __init__(self, backbone: Backbone, task: Task, T: float = 1.0):
        self.prompts = Prompts(backbone, task)
        if self.prompts.options is None:
            raise ValueError("Approach A supports at most 26 labels")
        self.ids = mx.array(self.prompts.letter_ids)
        self.labels = task.label_names
        self.T = T

    def decide(self, text: str) -> dict[str, float]:
        _, logits = self.prompts.options.run(text, logits=True)
        return _probs(logits[self.ids], self.labels, self.T)


class Decider:
    """Approaches B and B+C, loaded from a run directory produced by `quickfast evaluate`."""

    def __init__(self, run_dir: str | Path, backbone: Backbone | None = None):
        run_dir = Path(run_dir)
        self.cfg = json.loads((run_dir / "config.json").read_text())
        self.task = Task.load(run_dir.parent / "task.json")
        self.backbone = backbone or Backbone(self.cfg["model"])
        self.prompts = Prompts(self.backbone, self.task)
        self.template = self.prompts.plain if self.cfg["prompt"] == "plain" else self.prompts.options
        n = np.load(run_dir / "norm.npz")
        self.norm = Normalizer(n["mean"], n["std"])
        self.layers = self.cfg["layers"]
        self.T = self.cfg["T"]
        self.labels = list(self.cfg["labels"])
        if self.cfg["kind"] == "hybrid":
            self.head = HybridHead(self.cfg["n_layers"], self.cfg["d"], tied=self.cfg["tied"])
            self.head.load_weights(str(run_dir / "head.safetensors"))
            by_name = {l.name: l for l in self.task.labels}
            self.label_emb = self._embed_labels([by_name[n] for n in self.labels])
        else:
            n_out = len(self.labels)
            self.head = LinearProbe(self.cfg.get("d", self.norm.mean.shape[-1]), n_out)
            self.head.load_weights(str(run_dir / "head.safetensors"))
        self.head.train(False)

    def _embed_labels(self, labels: list[Label]) -> mx.array:
        feats = self.prompts.label_features(labels, self.layers)
        return self.head.embed_labels(self.norm(feats))

    def add_label(self, name: str, description: str = ""):
        """Hybrid only: a new decision option, usable immediately without retraining."""
        if self.cfg["kind"] != "hybrid":
            raise ValueError("Only hybrid heads accept new labels")
        self.label_emb = mx.concatenate([self.label_emb, self._embed_labels([Label(name, description)])])
        self.labels.append(name)

    def answer(self, text: str) -> dict:
        """Typed output in TypeSafe's shape, according to the task type (choice, noul or score)."""
        probs = self.decide(text)
        p = np.array([probs[l] for l in self.labels])
        if self.task.type == "score":
            by_name = {l.name: l for l in self.task.labels}
            return FORMATTERS["score"]([by_name[l].description or l for l in self.labels], p)
        return FORMATTERS[self.task.type](self.labels, p)

    def decide(self, text: str) -> dict[str, float]:
        h, _ = self.template.run(text, layers=self.layers)
        x = self.norm(mx.stack([h[l] for l in self.layers]))
        if self.cfg["kind"] == "hybrid":
            logits = mx.exp(self.head.log_scale) * self.head.embed_queries(x[None]) @ self.label_emb.T
        else:
            logits = self.head(x[None, 0])
        return _probs(logits[0], self.labels, self.T)
