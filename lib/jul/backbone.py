"""MLX backbone: one forward pass, tap hidden states at chosen layers, reuse a cached prompt prefix.

The instruction part of every prompt is identical across calls, so its KV cache is computed once
and each query only pays for its own tokens. When only intermediate layers are needed, the forward
stops right after the deepest one (the remaining layers are never computed).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

MODELS = {
    "minicpm5-2b": "openbmb/MiniCPM5-2B-MLX",
    "qwen3-0.6b": "mlx-community/Qwen3-0.6B-4bit",
    "qwen3-1.7b": "mlx-community/Qwen3-1.7B-4bit",
    "qwen3.5-9b": "mlx-community/Qwen3.5-9B-4bit",
}

# Layers tapped during extraction, as fractions of the model depth.
DEFAULT_LAYER_FRACTIONS = (0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class _StopForward(Exception):
    pass


class _Tap:
    """Wraps a transformer block to record its features and optionally halt the forward."""

    def __init__(self, block, idx: int, backbone: "Backbone"):
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


class Backbone:
    def __init__(self, name: str):
        self.name = name
        self.repo = MODELS.get(name, name)
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

    def layer_indices(self, fractions=DEFAULT_LAYER_FRACTIONS) -> list[int]:
        return sorted({max(0, min(self.n_layers - 1, round(f * self.n_layers) - 1)) for f in fractions})

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def forward(self, tokens: list[int], cache=None, layers=(), logits=False, pool: tuple[int, int] | None = None):
        """Run tokens through the model.

        Returns ({layer: (2d,) features}, last-token logits or None). Features are the last-token hidden
        state concatenated with the mean hidden state over positions pool=(start, end) of `tokens`.
        """
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
        captured = {k: v[0] for k, v in self._captured.items()}
        mx.eval(list(captured.values()) + ([out] if out is not None else []))
        return captured, out


@dataclass
class PromptTemplate:
    """A chat prompt split around the user input: a cached prefix and a per-query suffix."""

    backbone: Backbone
    prefix_text: str
    suffix_text: str
    use_prefix_cache: bool = True

    SENTINEL = "⁣QF_INPUT⁣"

    @classmethod
    def from_user_message(cls, backbone: Backbone, message: str, **kw) -> "PromptTemplate":
        """`message` must contain {input}; it is rendered with the model's chat template."""
        rendered = backbone.tokenizer.apply_chat_template(
            [{"role": "user", "content": message.replace("{input}", cls.SENTINEL)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix, suffix = rendered.split(cls.SENTINEL)
        return cls(backbone, prefix, suffix, **kw)

    def __post_init__(self):
        self.prefix_tokens = self.backbone.encode(self.prefix_text)
        self._n_suffix = len(self.backbone.encode(self.suffix_text))
        self._cache = None
        self._snapshot = None
        if self.use_prefix_cache and self.prefix_tokens:
            cache = make_prompt_cache(self.backbone.model)
            self.backbone.forward(self.prefix_tokens, cache=cache, logits=True)
            mx.eval([c.state for c in cache])
            if can_trim_prompt_cache(cache):
                self._cache = cache  # attention KV cache: trimmed back to the prefix after each query
            else:
                # Recurrent state (e.g. Gated DeltaNet in Qwen3.5) cannot be trimmed: keep a copy of the
                # state after the prefix and start each query from it. Layers replace their state
                # arrays rather than writing into them, so the copy is never modified.
                self._snapshot = [tuple(c.state) for c in cache]

    def _restored_cache(self):
        cache = make_prompt_cache(self.backbone.model)
        for c, state in zip(cache, self._snapshot):
            c.state = list(state)
        return cache

    def run(self, text: str, layers=(), logits=False):
        query = self.backbone.encode(text + self.suffix_text)
        n_input = max(1, len(query) - self._n_suffix)
        if self._snapshot is not None:
            return self.backbone.forward(query, cache=self._restored_cache(), layers=layers, logits=logits, pool=(0, n_input))
        if self._cache is None:
            p = len(self.prefix_tokens)
            return self.backbone.forward(self.prefix_tokens + query, layers=layers, logits=logits, pool=(p, p + n_input))
        try:
            return self.backbone.forward(query, cache=self._cache, layers=layers, logits=logits, pool=(0, n_input))
        finally:
            n = len(self.prefix_tokens)
            for c in self._cache:
                if c.offset > n:
                    c.trim(c.offset - n)


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t
