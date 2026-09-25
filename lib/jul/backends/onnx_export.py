"""Export a Hugging Face decoder to the model directory the onnx backend reads.

    python -m jul.backends.onnx_export microsoft/harrier-oss-v1-0.6b models/harrier-0.6b-onnx
    python -m jul.backends.onnx_export models/harrier-0.6b-onnx models/harrier-0.6b-onnx-int8 --int8

Needs torch, transformers, onnx and onnxscript (`pip install -e ".[onnx-export]"`); the backend
itself only needs onnxruntime and transformers.

The graph takes `input_ids` (batch, seq), no attention mask, and the KV cache of the tokens before
them: `past_key_<j>` / `past_value_<j>` (batch, kv heads, past, head dim) for every kept layer j, with
`past` possibly 0. It returns `layer_<i>`: the raw output of decoder layer i for the new tokens (what
the torch backend's hooks read, before the final norm), as float32 (batch, seq, hidden), and
`present_key_<j>` / `present_value_<j>`, the cache extended with them. The onnx backend runs a
prompt prefix once with an empty cache and keeps its presents: each query then pays for its own
tokens only, as with the torch backend. The layers above the deepest exported one are dropped from
the graph, and so are their caches. By default the export holds every layer `jul models add` may
choose (the upper half of the stack); once a preset is fitted, `--layers` can narrow it.

The directory also holds the tokenizer, config.json and jul_onnx.json (source repo, layers, cache
shape).

An encoder (BERT, XLM-R, e5..., see jul/encoder.py) exports as a graph of `input_ids` and
`attention_mask` (batch, seq) to the same `layer_<i>` outputs, without a cache; jul_onnx.json then
says "architecture": "encoder", with the model's positions and its input convention (e5: "query: ").

`--int8` on an export directory writes a copy with 8-bit weights: the matmuls become MatMulNBits
(blocks of 32, symmetric) computed with int8 activations (accuracy_level 4); the embedding table stays
float32. About 45 % of the size, one file under protobuf's 2 GB limit, so it can be loaded from bytes.
Its vectors are within a cosine of 0.9995 of float32 on an M4 and on AWS Graviton2 alike. Dynamic
quantization (QInt8 MatMulInteger) was tried first: fine on the M4, but on Graviton2 (no i8mm) ONNX
Runtime 1.30 returned vectors at a cosine of ~0.85 from float32, which flips answers.

The embedding table of a multilingual encoder is most of its weights (e5-small: 96M of 118M
parameters, 384 MB of 470). An encoder's is therefore quantized too, at 4 bits
(GatherBlockQuantized; ONNX Runtime has no 8-bit one): e5-small goes from 470 MB to 86 MB, at a
cosine of 0.9997 from float32. `--embedding-bits` sets it for any model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..encoder import MODEL_TYPES, text_prefix
from .onnx import META


class _Taps(torch.nn.Module):
    def __init__(self, model, layers: list[int]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.n_cache = max(layers) + 1
        # hidden_states[i + 1] is the output of layer i; the last one would be normed otherwise
        model.norm = torch.nn.Identity()
        model.layers = model.layers[: self.n_cache]

    def forward(self, input_ids, *past):
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for j in range(self.n_cache):
            cache.update(past[2 * j], past[2 * j + 1], j)
        out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
        presents = [t for layer in out.past_key_values.layers for t in (layer.keys, layer.values)]
        return (*[out.hidden_states[i + 1] for i in self.layers], *presents)


class _EncoderTaps(torch.nn.Module):
    def __init__(self, model, layers: list[int]):
        super().__init__()
        self.model = model
        self.layers = layers
        model.encoder.layer = model.encoder.layer[: max(layers) + 1]
        if hasattr(model, "pooler"):
            model.pooler = None

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return tuple(out.hidden_states[i + 1] for i in self.layers)


def cache_names(n_cache: int, kind: str = "past") -> list[str]:
    return [f"{kind}_{kv}_{j}" for j in range(n_cache) for kv in ("key", "value")]


def default_layers(n_layers: int) -> list[int]:
    from ..calibrate import LAYER_RANGE
    return list(range(max(0, round(LAYER_RANGE[0] * n_layers) - 1), n_layers))


def export(repo: str, out: Path, layers: list[int] | None = None) -> Path:
    config = AutoConfig.from_pretrained(repo)
    n_layers = config.num_hidden_layers
    layers = sorted(set(layers or default_layers(n_layers)))
    if layers[-1] >= n_layers:
        raise ValueError(f"{repo} has {n_layers} layers: cannot export layer {layers[-1]}")
    model = AutoModel.from_pretrained(repo, dtype=torch.float32).eval()
    out.mkdir(parents=True, exist_ok=True)
    if config.model_type in MODEL_TYPES:
        return _export_encoder(repo, model, config, out, layers)
    taps = _Taps(model, layers).eval()
    kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    # example sizes above 1: torch.export specializes a dimension of size 0 or 1
    ids = (torch.arange(10).reshape(2, 5) * 7919) % config.vocab_size
    past = [torch.randn(2, kv_heads, 4, head_dim) for _ in range(2 * taps.n_cache)]
    batch = torch.export.Dim("batch")
    seq = torch.export.Dim("seq", min=1, max=config.max_position_embeddings)
    past_len = torch.export.Dim("past", min=0, max=config.max_position_embeddings)
    torch.onnx.export(taps, (ids, *past), out / "model.onnx", dynamo=True, external_data=True,
                      input_names=["input_ids", *cache_names(taps.n_cache)],
                      output_names=[f"layer_{i}" for i in layers] + cache_names(taps.n_cache, "present"),
                      dynamic_shapes={"input_ids": {0: batch, 1: seq},
                                      "past": tuple({0: batch, 2: past_len} for _ in past)})
    AutoTokenizer.from_pretrained(repo).save_pretrained(out)
    config.save_pretrained(out)
    (out / META).write_text(json.dumps({"source": repo, "file": "model.onnx", "n_layers": n_layers,
                                        "layers": layers, "dtype": "float32",
                                        "kv_cache": {"layers": taps.n_cache, "heads": kv_heads,
                                                     "head_dim": head_dim}}, indent=2) + "\n")
    return out


def _export_encoder(repo: str, model, config, out: Path, layers: list[int]) -> Path:
    taps = _EncoderTaps(model, layers).eval()
    ids = (torch.arange(10).reshape(2, 5) * 7919) % config.vocab_size + 5
    mask = torch.ones_like(ids)
    mask[1, 3:] = 0
    batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq", min=1, max=config.max_position_embeddings)
    torch.onnx.export(taps, (ids, mask), out / "model.onnx", dynamo=True, external_data=True,
                      input_names=["input_ids", "attention_mask"], output_names=[f"layer_{i}" for i in layers],
                      dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}})
    tokenizer = AutoTokenizer.from_pretrained(repo)
    tokenizer.save_pretrained(out)
    config.save_pretrained(out)
    positions = min(tokenizer.model_max_length, config.max_position_embeddings)
    (out / META).write_text(json.dumps({"source": repo, "file": "model.onnx", "n_layers": config.num_hidden_layers,
                                        "layers": layers, "dtype": "float32", "architecture": "encoder",
                                        "max_positions": positions, "text_prefix": text_prefix(repo)},
                                       indent=2) + "\n")
    return out


def quantize_int8(src: Path, out: Path, embedding_bits: int | None = None) -> Path:
    """An int8 copy of the export in `src` (see the module docstring). `embedding_bits=4` also
    quantizes the embedding table (4-bit blocks of 32); by default an encoder's is, a decoder's not."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    meta = json.loads((src / META).read_text())
    if embedding_bits is None and meta.get("architecture") == "encoder":
        embedding_bits = 4
    out.mkdir(parents=True, exist_ok=True)
    model = onnx.load(src / meta["file"])
    steps = ([(embedding_bits, ("Gather",))] if embedding_bits else []) + [(8, ("MatMul",))]
    for bits, ops in steps:
        quantizer = MatMulNBitsQuantizer(model, bits=bits, block_size=32, is_symmetric=True, accuracy_level=4,
                                         op_types_to_quantize=ops, quant_axes=(("MatMul", 0), ("Gather", 1)))
        quantizer.process()
        model = quantizer.model.model
    onnx.save(model, out / "model.onnx")
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("model.onnx") and f.name != META:
            shutil.copy2(f, out / f.name)
    dtype = "8-bit weights, int8 compute" + (f", {embedding_bits}-bit embedding" if embedding_bits else "")
    (out / META).write_text(json.dumps({**meta, "file": "model.onnx", "dtype": dtype}, indent=2) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m jul.backends.onnx_export", description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", help="Hugging Face repo or local directory of the model (with --int8: an export)")
    ap.add_argument("out", type=Path, help="model directory to write")
    ap.add_argument("--layers", type=int, nargs="+", help="default: the upper half of the stack")
    ap.add_argument("--int8", action="store_true", help="quantize the export in `repo` instead")
    ap.add_argument("--embedding-bits", type=int, choices=[4], help="with --int8: quantize the embedding "
                    "table too (default: an encoder's)")
    a = ap.parse_args()
    print(f"wrote {quantize_int8(Path(a.repo), a.out, a.embedding_bits) if a.int8 else export(a.repo, a.out, a.layers)}")


if __name__ == "__main__":
    main()
