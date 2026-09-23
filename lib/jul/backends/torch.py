"""PyTorch backend, through transformers: CUDA, CPU, or MPS.

Same mechanics as the MLX backend: forward hooks on the decoder layers capture the features and
halt the forward after the deepest layer needed; the prompt prefix is cached once per template.
Device and dtype default to `JUL_DEVICE` / cuda > mps > cpu, and `JUL_DTYPE` / bfloat16 (float16 on
GPUs older than Ampere, float32 on CPU).

`forward_batch` runs several queries in one forward. They are padded on the right and no attention
mask is passed: under the causal mask a real token never sees the padding after it, so its hidden
state is the one a forward of that query alone would give.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

from ..backbone import Backbone

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class _StopForward(Exception):
    pass


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device_type: str, capability: tuple[int, int] | None = None) -> torch.dtype:
    """bfloat16, except on CPU (float32) and on CUDA GPUs before Ampere (sm_80): Turing and Volta
    (T4, V100) have no bfloat16 tensor cores, and float16 runs there at full speed."""
    if device_type == "cpu":
        return torch.float32
    if device_type == "cuda" and capability is not None and capability < (8, 0):
        return torch.float16
    return torch.bfloat16


@dataclass
class _Prefix:
    n: int
    cache: object
    #: A plain attention KV cache is cropped back to the prefix after each query. Anything else
    #: (recurrent state, sliding window) cannot be rewound: each query runs on a copy.
    croppable: bool

    def rewind(self) -> None:
        """Back to the prefix, layer by layer: a query that stopped early never reached the layers
        above its stop, and those must keep their prefix whole."""
        for layer in self.cache.layers:
            if layer.is_initialized and layer.keys.shape[-2] > self.n:
                layer.keys = layer.keys[..., : self.n, :]
                layer.values = layer.values[..., : self.n, :]


def _croppable(cache) -> bool:
    return type(cache) is DynamicCache and all(type(layer) is DynamicLayer for layer in cache.layers)


class TorchBackbone(Backbone):
    backend = "torch"

    def __init__(self, name: str, backend: str | None = None, device: str | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__(name)
        self.device = torch.device(device or os.environ.get("JUL_DEVICE") or default_device())
        if dtype is None and os.environ.get("JUL_DTYPE"):
            if os.environ["JUL_DTYPE"] not in DTYPES:
                raise ValueError(f"JUL_DTYPE={os.environ['JUL_DTYPE']!r}: expected one of {', '.join(DTYPES)}")
            dtype = DTYPES[os.environ["JUL_DTYPE"]]
        if dtype is None:
            capability = torch.cuda.get_device_capability(self.device) if self.device.type == "cuda" else None
            dtype = default_dtype(self.device.type, capability)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.model = AutoModelForCausalLM.from_pretrained(self.repo, dtype=dtype).to(self.device).eval()
        self._decoder = self.model.get_decoder()
        self._lm_head = self.model.get_output_embeddings()
        layers = self._decoder.layers
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._hook(i))
        self.n_layers = len(layers)
        pad = self.tokenizer.pad_token_id
        self._pad = pad if pad is not None else (self.tokenizer.eos_token_id or 0)
        self._want: set[int] = set()
        self._stop_at: int | None = None
        self._captured: dict[int, torch.Tensor] = {}
        self._last: torch.Tensor | None = None   # (B,) index of each row's last real token
        self._pool: torch.Tensor | None = None   # (B, T, 1) bool mask of each row's pooled positions

    def _hook(self, idx: int):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if idx in self._want:
                rows = torch.arange(h.shape[0], device=h.device)
                # masked_fill, not a product: a padding position that overflows float16 (inf) would
                # give inf * 0 = nan and spoil the row's mean
                mean = h.masked_fill(~self._pool, 0).sum(1, dtype=torch.float32) / self._pool.sum(1)
                # [last token ; mean over the input tokens]
                self._captured[idx] = torch.cat([h[rows, self._last].float(), mean], dim=-1)
            if self._stop_at == idx:
                raise _StopForward
        return hook

    @torch.inference_mode()
    def forward(self, tokens, layers=(), logits=False, pool=None, prefix: _Prefix | None = None):
        if prefix is None:
            return self._first(self._run([tokens], None, layers, logits, [pool]))
        if not prefix.croppable:
            return self._first(self._run([tokens], copy.deepcopy(prefix.cache), layers, logits, [pool]))
        try:
            return self._first(self._run([tokens], prefix.cache, layers, logits, [pool]))
        finally:
            prefix.rewind()

    @torch.inference_mode()
    def forward_batch(self, queries, layers=(), pools=None, prefix: _Prefix | None = None):
        pools = pools or [None] * len(queries)
        if len(queries) == 1 or (prefix is not None and not prefix.croppable):
            # a recurrent state is not repeated over a batch: one copy per query, as in `forward`
            return super().forward_batch(queries, layers, pools, prefix)
        cache = None
        if prefix is not None:
            cache = copy.deepcopy(prefix.cache)
            cache.batch_repeat_interleave(len(queries))
        captured, _, _ = self._run(queries, cache, layers, False, pools)
        return [{k: v[i] for k, v in captured.items()} for i in range(len(queries))]

    @torch.inference_mode()
    def cache_prefix(self, tokens) -> _Prefix:
        _, _, cache = self._run([tokens], None, logits=True, use_cache=True)
        return _Prefix(len(tokens), cache, _croppable(cache))

    @torch.inference_mode()
    def last_hidden(self, tokens, prefix: _Prefix | None = None):
        self._want, self._stop_at = set(), None
        cache = None if prefix is None else (prefix.cache if prefix.croppable else copy.deepcopy(prefix.cache))
        try:
            ids = torch.tensor([tokens], device=self.device)
            h = self._decoder(input_ids=ids, past_key_values=cache, use_cache=cache is not None).last_hidden_state
            return h[0].float().cpu().numpy()
        finally:
            if prefix is not None and prefix.croppable:
                prefix.rewind()

    @staticmethod
    def _first(result):
        captured, logits, _ = result
        return {k: v[0] for k, v in captured.items()}, (logits[0] if logits is not None else None)

    def _run(self, seqs, cache=None, layers=(), logits=False, pools=None, use_cache=None):
        """Right-padded batch of token lists. Returns ({layer: (B, 2d)}, (B, vocab) or None, cache)."""
        lengths = [len(s) for s in seqs]
        ids = torch.full((len(seqs), max(lengths)), self._pad, dtype=torch.long)
        pool = torch.zeros(len(seqs), max(lengths), 1, dtype=torch.bool)
        for i, (s, p) in enumerate(zip(seqs, pools or [None] * len(seqs))):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            start, end = p or (0, len(s))
            pool[i, start:end] = True
        self._want = set(layers)
        self._stop_at = None if logits else (max(layers) if layers else None)
        self._captured = {}
        self._last = torch.tensor([n - 1 for n in lengths], device=self.device)
        self._pool = pool.to(self.device)
        out = None
        try:
            result = self._decoder(input_ids=ids.to(self.device), past_key_values=cache,
                                   use_cache=cache is not None if use_cache is None else use_cache)
            cache = result.past_key_values
            if logits:
                rows = torch.arange(len(seqs), device=self.device)
                out = self._lm_head(result.last_hidden_state[rows, self._last]).float()
        except _StopForward:
            pass
        captured = {k: v.cpu().numpy() for k, v in self._captured.items()}
        for k, v in captured.items():
            if not np.isfinite(v).all():
                raise FloatingPointError(f"non-finite hidden state at layer {k} in {self.model.dtype}: "
                                         "the model overflows this dtype, set JUL_DTYPE=float32")
        return captured, (out.cpu().numpy() if out is not None else None), cache
