"""MLX backend (Apple Silicon), through mlx-lm."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

from ..backbone import Backbone


class _StopForward(Exception):
    pass


class _Tap:
    """Wraps a transformer block to record its features and optionally halt the forward."""

    def __init__(self, block, idx: int, backbone: "MLXBackbone"):
        self.block = block
        self.idx = idx
        self.bb = backbone

    def __call__(self, *args, **kwargs):
        h = self.block(*args, **kwargs)
        if self.idx in self.bb._want:
            start, end = self.bb._pool
            # [last token ; mean over the input tokens]
            self.bb._captured[self.idx] = mx.concatenate([h[:, -1, :], h[:, start:end, :].mean(1)], axis=-1)
        if self.bb._stop_at == self.idx:
            raise _StopForward
        return h

    def __getattr__(self, name):
        return getattr(self.block, name)


@dataclass
class _Prefix:
    n: int
    cache: list | None = None      # attention KV cache: trimmed back to the prefix after each query
    snapshot: list | None = None   # recurrent state: copied, each query starts from the copy


class MLXBackbone(Backbone):
    backend = "mlx"

    def __init__(self, name: str, backend: str | None = None):
        super().__init__(name)
        self.model, self.tokenizer = load(self.repo)
        layers = self.model.layers
        for i, block in enumerate(layers):
            layers[i] = _Tap(block, i, self)
        self.n_layers = len(layers)
        self._want: set[int] = set()
        self._stop_at: int | None = None
        self._captured: dict[int, mx.array] = {}
        self._pool = (0, 1)
        lm = getattr(self.model, "language_model", self.model)  # multimodal wrappers (e.g. Qwen3.5) nest the text model
        self._inner = lm.model
        self._lm_head = lm.lm_head if hasattr(lm, "lm_head") else self._inner.embed_tokens.as_linear

    def forward(self, tokens, layers=(), logits=False, pool=None, prefix: _Prefix | None = None, cache=None):
        """`cache` (an mlx-lm prompt cache, advanced in place) is kept for the dev scripts."""
        if prefix is None:
            return self._run(tokens, cache, layers, logits, pool)
        if prefix.snapshot is not None:
            return self._run(tokens, self._restored(prefix.snapshot), layers, logits, pool)
        try:
            return self._run(tokens, prefix.cache, layers, logits, pool)
        finally:
            for c in prefix.cache:
                if c.offset > prefix.n:
                    c.trim(c.offset - prefix.n)

    def cache_prefix(self, tokens) -> _Prefix:
        cache = make_prompt_cache(self.model)
        self._run(tokens, cache, logits=True)
        mx.eval([c.state for c in cache])
        if can_trim_prompt_cache(cache):
            return _Prefix(len(tokens), cache=cache)
        # Recurrent state (e.g. Gated DeltaNet in Qwen3.5) cannot be trimmed: keep a copy of the state
        # after the prefix. Layers replace their state arrays rather than writing into them, so the
        # copy is never modified.
        return _Prefix(len(tokens), snapshot=[tuple(c.state) for c in cache])

    def _restored(self, snapshot):
        cache = make_prompt_cache(self.model)
        for c, state in zip(cache, snapshot):
            c.state = list(state)
        return cache

    def _run(self, tokens, cache=None, layers=(), logits=False, pool=None):
        self._want = set(layers)
        self._pool = pool or (0, len(tokens))
        self._stop_at = None if logits else (max(layers) if layers else None)
        self._captured = {}
        x = mx.array(tokens)[None]
        out = None
        try:
            h = self._inner(x, cache=cache)
            if logits:
                out = self._lm_head(h[:, -1:, :])[0, 0].astype(mx.float32)
        except _StopForward:
            pass
        captured = {k: v[0].astype(mx.float32) for k, v in self._captured.items()}
        mx.eval(list(captured.values()) + ([out] if out is not None else []))
        return ({k: np.array(v) for k, v in captured.items()},
                np.array(out) if out is not None else None)
