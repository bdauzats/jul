"""PyTorch backend, through transformers: CUDA, CPU, or MPS.

Same mechanics as the MLX backend: forward hooks on the decoder layers capture the features and
halt the forward after the deepest layer needed; the prompt prefix is cached once per template.
Device and dtype default to `JUL_DEVICE` / cuda > mps > cpu, and bfloat16 (float32 on CPU).
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


class _StopForward(Exception):
    pass


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class _Prefix:
    n: int
    cache: object
    #: A plain attention KV cache is cropped back to the prefix after each query. Anything else
    #: (recurrent state, sliding window) cannot be rewound: each query runs on a copy.
    croppable: bool


def _croppable(cache) -> bool:
    return type(cache) is DynamicCache and all(type(layer) is DynamicLayer for layer in cache.layers)


class TorchBackbone(Backbone):
    backend = "torch"

    def __init__(self, name: str, backend: str | None = None, device: str | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__(name)
        self.device = torch.device(device or os.environ.get("JUL_DEVICE") or default_device())
        if dtype is None:
            dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.model = AutoModelForCausalLM.from_pretrained(self.repo, dtype=dtype).to(self.device).eval()
        self._decoder = self.model.get_decoder()
        self._lm_head = self.model.get_output_embeddings()
        layers = self._decoder.layers
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._hook(i))
        self.n_layers = len(layers)
        self._want: set[int] = set()
        self._stop_at: int | None = None
        self._captured: dict[int, torch.Tensor] = {}
        self._pool = (0, 1)

    def _hook(self, idx: int):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if idx in self._want:
                start, end = self._pool
                # [last token ; mean over the input tokens]
                self._captured[idx] = torch.cat([h[0, -1].float(), h[0, start:end].float().mean(0)])
            if self._stop_at == idx:
                raise _StopForward
        return hook

    @torch.inference_mode()
    def forward(self, tokens, layers=(), logits=False, pool=None, prefix: _Prefix | None = None):
        if prefix is None:
            return self._run(tokens, None, layers, logits, pool)[:2]
        if not prefix.croppable:
            return self._run(tokens, copy.deepcopy(prefix.cache), layers, logits, pool)[:2]
        try:
            return self._run(tokens, prefix.cache, layers, logits, pool)[:2]
        finally:
            extra = prefix.cache.get_seq_length() - prefix.n
            if extra > 0:
                prefix.cache.crop(-extra)

    @torch.inference_mode()
    def cache_prefix(self, tokens) -> _Prefix:
        _, _, cache = self._run(tokens, None, logits=True, use_cache=True)
        return _Prefix(len(tokens), cache, _croppable(cache))

    def _run(self, tokens, cache=None, layers=(), logits=False, pool=None, use_cache=None):
        self._want = set(layers)
        self._pool = pool or (0, len(tokens))
        self._stop_at = None if logits else (max(layers) if layers else None)
        self._captured = {}
        ids = torch.tensor([tokens], device=self.device)
        out = None
        try:
            result = self._decoder(input_ids=ids, past_key_values=cache,
                                   use_cache=cache is not None if use_cache is None else use_cache)
            cache = result.past_key_values
            if logits:
                out = self._lm_head(result.last_hidden_state[:, -1])[0].float()
        except _StopForward:
            pass
        captured = {k: v.cpu().numpy() for k, v in self._captured.items()}
        return captured, (out.cpu().numpy() if out is not None else None), cache
