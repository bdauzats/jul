"""ONNX Runtime backend: CPU inference without torch, for small deployments (AWS Lambda, containers).

The model directory is written by `python -m jul.backends.onnx_export` (see there): a graph that
takes `input_ids` (batch, seq) and the KV cache of a prefix, and returns the raw output of a fixed
set of decoder layers and the extended cache; the tokenizer; and `jul_onnx.json`. Only those layers
can be read. `cache_prefix` runs a prompt prefix once and keeps its cache: a query then runs its own
tokens only, the cache repeated over the rows of a batch. An export without a cache (older jul) still
works: its prefix is kept as tokens and run again before each query.

The tokenizer is read with `tokenizers` alone (see `Tokenizer`): transformers weighs ~110 MB, too much
for a 250 MB Lambda package.

What the graph cannot do, compared with the torch backend: no logits and no final norm, so the
letters reading and the pointer method are not available.

An encoder export (jul_onnx.json "architecture": "encoder", see jul/encoder.py) takes `input_ids`
and `attention_mask` and has no cache: the prefix runs again with each query, and a row is capped
at the model's positions.

`JUL_ONNX_MODEL` replaces the graph named in jul_onnx.json: a local path, or `s3://bucket/key`, read
into memory with boto3 (no local copy: a Lambda with SnapStart has 512 MB of /tmp). The graph must
then be a single file, as the int8 export is.

Batches are padded on the right with no attention mask, as in the torch backend: under the causal
mask a real token never sees the padding after it.

Memory: the graph returns every requested layer for every position, and every row carries the
prefix (as cache or as tokens). A batch is therefore split into runs of at most
`JUL_ONNX_BATCH_TOKENS` (rows x (prefix + longest row), default 2048), and the CPU memory arena is
off: with a new shape at each run it only grows.
The exported attention is not memory-efficient: it builds the full (heads, T, T) score matrix, ~20 GB
at 18k tokens. A row is therefore capped at `JUL_ONNX_MAX_TOKENS` (default 2048): past it, the end of
the input is cut, and the prompt prefix and suffix are kept, so the last token read is still the one
the prompt ends on.
"""

from __future__ import annotations

import atexit
import gc
import json
import os
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tokenizers

from .. import encoder
from ..backbone import Backbone

META = "jul_onnx.json"
#: Live backbones. Their sessions are released before the interpreter shuts down: an ONNX Runtime
#: session destroyed during static destruction (next to torch, on macOS) can abort the process with
#: "recursive_mutex lock failed" after everything has run.
_LIVE: "weakref.WeakSet[ONNXBackbone]" = weakref.WeakSet()


@atexit.register
def _release_sessions() -> None:
    for backbone in list(_LIVE):
        backbone.session = None
    gc.collect()
BATCH_TOKENS = 2048
MAX_TOKENS = 2048


def _graph(location: str) -> str | bytes:
    """A path for InferenceSession, or the bytes of an s3:// object."""
    if not location.startswith("s3://"):
        return location
    import io

    import boto3
    from boto3.s3.transfer import TransferConfig
    bucket, _, key = location.removeprefix("s3://").partition("/")
    buffer = io.BytesIO()
    boto3.client("s3").download_fileobj(bucket, key, buffer, Config=TransferConfig(max_concurrency=16))
    return buffer.getvalue()


def _token(value) -> str | None:
    """A special token as tokenizer_config.json stores it: a string or a serialized AddedToken."""
    return value.get("content") if isinstance(value, dict) else value


class Tokenizer:
    """What jul reads of a Hugging Face tokenizer (`encode`, `decode`, the pad and eos ids, the chat
    template), on the `tokenizers` library and jinja2, from tokenizer.json and tokenizer_config.json."""

    def __init__(self, root: Path):
        self._tok = tokenizers.Tokenizer.from_file(str(root / "tokenizer.json"))
        config_path = root / "tokenizer_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        template = root / "chat_template.jinja"
        self.chat_template = template.read_text() if template.exists() else config.get("chat_template")
        self.bos_token, self.eos_token, self.pad_token = (_token(config.get(k)) for k in ("bos_token", "eos_token", "pad_token"))
        self.eos_token_id = self._tok.token_to_id(self.eos_token) if self.eos_token else None
        self.pad_token_id = self._tok.token_to_id(self.pad_token) if self.pad_token else None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, tokenize: bool = True, add_generation_prompt: bool = False, **kwargs):
        """Rendered as transformers does: a sandboxed jinja2 environment, trimmed blocks."""
        if not self.chat_template:
            raise ValueError("This tokenizer has no chat template")
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message):
            raise TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, indent=None, **kw: json.dumps(x, ensure_ascii=False, indent=indent)
        env.globals["raise_exception"] = raise_exception
        text = env.from_string(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token or "", eos_token=self.eos_token or "", **kwargs)
        return self.encode(text, add_special_tokens=False) if tokenize else text


@dataclass
class _Prefix:
    tokens: list[int]
    #: the prefix's cache, one (1, heads, len(tokens), head dim) array per cache input of the graph;
    #: None for an export without a cache, whose prefix runs again before each query
    cache: list[np.ndarray] | None


class ONNXBackbone(Backbone):
    backend = "onnx"

    def __init__(self, name: str, backend: str | None = None, providers: list[str] | None = None):
        super().__init__(name)
        root = self.model_dir
        meta = json.loads((root / META).read_text())
        self.tokenizer = Tokenizer(root)
        self.n_layers = meta["n_layers"]
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        if os.environ.get("JUL_ONNX_THREADS"):
            options.intra_op_num_threads = int(os.environ["JUL_ONNX_THREADS"])
        self.session = ort.InferenceSession(_graph(os.environ.get("JUL_ONNX_MODEL") or str(root / meta["file"])),
                                            options, providers=providers or ["CPUExecutionProvider"])
        _LIVE.add(self)
        self._outputs = {int(o.name.removeprefix("layer_")): o.name for o in self.session.get_outputs()
                         if o.name.startswith("layer_")}
        self._kv = meta.get("kv_cache")
        self._past = [i.name for i in self.session.get_inputs() if i.name.startswith("past_")]
        self._presents = [n.replace("past_", "present_", 1) for n in self._past]
        pad = self.tokenizer.pad_token_id
        self._pad = pad if pad is not None else (self.tokenizer.eos_token_id or 0)
        self.architecture = meta.get("architecture", "decoder")
        if self.architecture == "encoder":
            self.text_prefix = meta.get("text_prefix", "")
            self._head, self._tail = encoder.special_tokens(self.tokenizer.encode)
            self.max_tokens = meta["max_positions"] - len(self._head) - len(self._tail)

    @property
    def layers(self) -> list[int]:
        """The layers this export can read."""
        return sorted(self._outputs)

    def forward(self, tokens, layers=(), logits=False, pool=None, prefix=None):
        if logits:
            raise NotImplementedError("The onnx backend exports hidden states only, no logits: "
                                      "the letters reading needs the mlx or torch backend")
        return self.forward_batch([tokens], layers, [pool], prefix)[0], None

    def forward_batch(self, queries, layers=(), pools=None, prefix=None):
        if self.architecture == "encoder":
            return self._encode(queries, list(layers), pools, prefix)
        p = list(prefix.tokens) if prefix is not None else []
        cache = prefix.cache if prefix is not None else None
        # with a cache the prefix is not part of the rows; without one it is prepended to each
        shift = 0 if cache is not None else len(p)
        limit = int(os.environ.get("JUL_ONNX_MAX_TOKENS") or MAX_TOKENS)
        seqs, spans = [], []
        for q, pool in zip(queries, pools or [None] * len(queries)):
            q = list(q)
            start, end = pool or (0, len(q))   # positions in the query, after the prefix
            over = len(p) + len(q) - limit
            if over > 0:
                cut = min(over, end - start - 1)   # the input keeps at least one token
                warnings.warn(f"onnx backend: a prompt of {len(p) + len(q)} tokens is over "
                              f"JUL_ONNX_MAX_TOKENS={limit}: its input is cut by {cut} tokens", stacklevel=3)
                q = q[: end - cut] + q[end:]
                end -= cut
            seqs.append(q if cache is not None else p + q)
            spans.append((shift + start, shift + end))
        budget = int(os.environ.get("JUL_ONNX_BATCH_TOKENS") or BATCH_TOKENS)
        cached = len(p) if cache is not None else 0
        out, group = [], []
        for i in range(len(seqs)):
            longest = cached + max([len(seqs[j]) for j in group] + [len(seqs[i])])
            if group and (len(group) + 1) * longest > budget:
                out += self._run([seqs[j] for j in group], [spans[j] for j in group], layers, cache)
                group = []
            group.append(i)
        if group:
            out += self._run([seqs[j] for j in group], [spans[j] for j in group], layers, cache)
        return out

    def cache_prefix(self, tokens) -> _Prefix:
        """Runs the prefix once and keeps its KV cache (an export without one keeps the tokens)."""
        tokens = list(tokens)
        if not self._kv:   # an encoder, or a decoder exported without a cache
            return _Prefix(tokens, None)
        feed = {"input_ids": np.array([tokens], dtype=np.int64), **self._empty_cache(1)}
        return _Prefix(tokens, self.session.run(self._presents, feed))

    def last_hidden(self, tokens, prefix=None):
        raise NotImplementedError("The onnx backend exports hidden states only, without the final "
                                  "norm: the pointer method needs the mlx or torch backend")

    def _encode(self, queries, layers, pools, prefix) -> list[dict[int, np.ndarray]]:
        """An encoder's features: rows grouped under the token budget, prefix prepended to each."""
        self._check(layers)
        if not layers:
            return [{} for _ in queries]
        limit = min(self.max_tokens, int(os.environ.get("JUL_ONNX_MAX_TOKENS") or MAX_TOKENS))
        fitted = [encoder.fit(list(prefix.tokens) if prefix else [], list(q), p, limit)
                  for q, p in zip(queries, pools or [None] * len(queries))]
        budget = int(os.environ.get("JUL_ONNX_BATCH_TOKENS") or BATCH_TOKENS)
        out, group = [], []

        def run():
            ids, mask, spans = encoder.batch([fitted[j][0] for j in group], [fitted[j][1] for j in group],
                                             self._head, self._tail, self._pad)
            hidden = self.session.run([self._outputs[l] for l in layers], {"input_ids": ids, "attention_mask": mask})
            return encoder.features(dict(zip(layers, hidden)), mask, spans)

        for i in range(len(fitted)):
            longest = max(len(fitted[j][0]) for j in group + [i])
            if group and (len(group) + 1) * longest > budget:
                out += run()
                group = []
            group.append(i)
        return out + run()

    def _check(self, layers) -> None:
        missing = sorted(set(layers) - set(self._outputs))
        if missing:
            raise ValueError(f"Layers {missing} are not in this export (it has {self.layers}): "
                             "export it again with them")

    def _empty_cache(self, rows: int) -> dict[str, np.ndarray]:
        shape = (rows, self._kv["heads"], 0, self._kv["head_dim"]) if self._kv else None
        return {name: np.zeros(shape, dtype=np.float32) for name in self._past}

    def _run(self, seqs, pools, layers, cache=None) -> list[dict[int, np.ndarray]]:
        """Right-padded batch after `cache` (the prefix's, repeated over the rows; None: an empty one).
        Returns, per row, {layer: (2d,) [last token ; mean over its pool]}."""
        self._check(layers)
        layers = list(layers)
        if not layers:
            return [{} for _ in seqs]
        lengths = [len(s) for s in seqs]
        ids = np.full((len(seqs), max(lengths)), self._pad, dtype=np.int64)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s
        feed = {"input_ids": ids}
        if self._past:
            # one row reads the prefix's cache as it is; a batch repeats it over its rows
            feed.update({name: (c if len(seqs) == 1 else np.repeat(c, len(seqs), axis=0))
                         for name, c in zip(self._past, cache)}
                        if cache is not None else self._empty_cache(len(seqs)))
        hidden = self.session.run([self._outputs[l] for l in layers], feed)
        out = []
        for i, (n, (start, end)) in enumerate(zip(lengths, pools)):
            row = {}
            for layer, h in zip(layers, hidden):
                h = h[i].astype(np.float32, copy=False)
                row[layer] = np.concatenate([h[n - 1], h[start:end].mean(0)])
                if not np.isfinite(row[layer]).all():
                    raise FloatingPointError(f"non-finite hidden state at layer {layer}")
            out.append(row)
        return out
