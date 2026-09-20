"""Extract and cache backbone features for a task's splits, so heads can be trained in seconds."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..backbone import Backbone
from .task import Prompts, Task

SPLITS = ("train", "val", "test")


def feature_dir(root: str | Path, task: Task, model: str) -> Path:
    return Path(root) / task.name / model.replace("/", "__")


def extract(backbone: Backbone, task: Task, out_dir: Path, limit: int | None = None, log_every: int = 250):
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = Prompts(backbone, task)
    layers = backbone.layer_indices()

    label_feats = prompts.label_features(task.labels, layers)
    np.save(out_dir / "labels.npy", np.array(label_feats.astype(mx.float16)))

    meta = {"model": backbone.name, "repo": backbone.repo, "n_layers": backbone.n_layers, "layers": layers,
            "labels": task.label_names, "has_options": prompts.options is not None, "splits": {}}
    for split in SPLITS:
        texts, ys = task.load_split(split)
        if limit:
            texts, ys = texts[:limit], ys[:limit]
        n, n_l, d = len(texts), len(layers), label_feats.shape[-1]
        q_plain = np.zeros((n, n_l, d), np.float16)
        q_opts = np.zeros((n, n_l, d), np.float16) if prompts.options else None
        letter_logits = np.zeros((n, len(task.labels)), np.float32) if prompts.options else None
        letter_mass = np.zeros(n, np.float32) if prompts.options else None
        t0 = time.perf_counter()
        for i, text in enumerate(texts):
            h, _ = prompts.plain.run(text, layers=layers)
            q_plain[i] = np.array(mx.stack([h[l] for l in layers]).astype(mx.float16))
            if prompts.options:
                h, logits = prompts.options.run(text, layers=layers, logits=True)
                q_opts[i] = np.array(mx.stack([h[l] for l in layers]).astype(mx.float16))
                sel = logits[mx.array(prompts.letter_ids)]
                letter_logits[i] = np.array(sel)
                letter_mass[i] = float(mx.exp(mx.logsumexp(sel) - mx.logsumexp(logits)))
            if log_every and (i + 1) % log_every == 0:
                rate = (i + 1) / (time.perf_counter() - t0)
                print(f"  [{backbone.name}] {split} {i + 1}/{n}  ({rate:.1f} ex/s)", flush=True)
        arrays = {"y": np.array(ys, np.int64), "q_plain": q_plain}
        if prompts.options:
            arrays.update(q_opts=q_opts, letter_logits=letter_logits, letter_mass=letter_mass)
        np.savez(out_dir / f"{split}.npz", **arrays)
        meta["splits"][split] = {"n": n, "seconds": round(time.perf_counter() - t0, 1)}
        print(f"  [{backbone.name}] {split}: {n} examples in {meta['splits'][split]['seconds']}s", flush=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load(out_dir: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]], np.ndarray]:
    meta = json.loads((out_dir / "meta.json").read_text())
    splits = {s: dict(np.load(out_dir / f"{s}.npz")) for s in SPLITS}
    labels = np.load(out_dir / "labels.npy")
    return meta, splits, labels
