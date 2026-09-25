"""The onnx backend reads the vectors of the torch backend, from a graph exported by jul itself.

The model is a tiny random Qwen3 (4 layers, width 64) built on the fly: the export and the
comparison take seconds, and only the tokenizer is downloaded. `JUL_TEST_TOKENIZER` overrides it.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from jul.backbone import Backbone, PromptTemplate
from jul.presets import ONE_WORD, Formulation, Preset, load_preset, repo_fields, save_preset

HAS_EXPORT = all(importlib.util.find_spec(m) for m in ("torch", "onnx", "onnxscript", "onnxruntime"))
TOKENIZER = os.environ.get("JUL_TEST_TOKENIZER", "Qwen/Qwen3-0.6B")
TEXTS = ["I was charged twice for my subscription", "the app crashes on export",
         "do you offer annual plans?", "Je voudrais annuler ma commande"]
needs_export = pytest.mark.skipif(not HAS_EXPORT, reason="needs torch, onnx, onnxscript, onnxruntime")


def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def test_an_onnx_preset_keeps_its_repo_through_json(tmp_path):
    preset = Preset(name="tiny", **repo_fields("onnx", "/models/tiny-onnx"),
                    formulations=(Formulation("one_word", ONE_WORD, 3),), tau=0.05,
                    latency_ms="~10", quality="", backend="onnx")
    loaded = load_preset(save_preset(preset, tmp_path))
    assert loaded == preset
    assert loaded.repos == {"onnx": "/models/tiny-onnx"}


@needs_export
def test_the_export_holds_the_upper_half_of_the_stack(pair):
    _, onnx = pair
    assert onnx.n_layers == 4
    assert onnx.layers == [1, 2, 3]


@needs_export
@pytest.mark.parametrize("use_prefix_cache", [True, False])
def test_onnx_reads_the_vectors_of_torch(pair, use_prefix_cache):
    """Single queries and a right-padded batch, both halves of the features, at every exported layer.
    Without a KV cache the onnx backend runs the prefix again: the vectors must not change."""
    torch_bb, onnx_bb = pair
    prefix, suffix = ONE_WORD.split("{state}")
    ref = PromptTemplate(torch_bb, prefix, suffix)
    got = PromptTemplate(onnx_bb, prefix, suffix, use_prefix_cache=use_prefix_cache)
    layers = onnx_bb.layers
    batched = got.run_batch(TEXTS, layers=layers)
    for text, b in zip(TEXTS, batched):
        expected, _ = ref.run(text, layers=layers)
        single, _ = got.run(text, layers=layers)
        for layer in layers:
            assert cosine(single[layer], expected[layer]) > 0.9999, (text, layer)
            assert cosine(b[layer], expected[layer]) > 0.9999, (text, layer)


@needs_export
def test_onnx_passes_the_calibration_checks(pair):
    """`jul models add` runs these before fitting: prefix and plain prompts agree, calls are independent."""
    from jul.calibrate import check
    _, onnx = pair
    assert check(onnx, onnx.n_layers - 1) == []


@needs_export
def test_onnx_names_what_it_cannot_read(pair):
    _, onnx = pair
    tokens = onnx.encode(TEXTS[0])
    with pytest.raises(ValueError, match=r"Layers \[0\] are not in this export"):
        onnx.forward(tokens, layers=[0])
    with pytest.raises(NotImplementedError, match="letters reading"):
        onnx.forward(tokens, layers=[3], logits=True)
    with pytest.raises(NotImplementedError, match="pointer method"):
        onnx.last_hidden(tokens)


@needs_export
def test_onnx_cuts_a_long_input_and_keeps_the_prompt_around_it(pair, monkeypatch):
    """Past JUL_ONNX_MAX_TOKENS the end of the input goes; the prefix and suffix stay, so the vector
    is the one of the cut text in the same prompt."""
    _, onnx = pair
    prefix, suffix = ONE_WORD.split("{state}")
    template = PromptTemplate(onnx, prefix, suffix)
    long_text = " ".join(TEXTS * 20)
    n_prompt = len(template.prefix_tokens) + len(onnx.encode(long_text + suffix))
    limit = n_prompt - 30
    monkeypatch.setenv("JUL_ONNX_MAX_TOKENS", str(limit))
    with pytest.warns(UserWarning, match="is cut by 30 tokens"):
        cut, _ = template.run(long_text, layers=[3])
    kept = onnx.tokenizer.decode(onnx.encode(long_text)[:-30])
    monkeypatch.delenv("JUL_ONNX_MAX_TOKENS")
    expected, _ = template.run(kept, layers=[3])
    assert cosine(cut[3], expected[3]) > 0.9999


@needs_export
def test_the_onnx_tokenizer_reads_what_transformers_reads(pair):
    """The onnx backend tokenizes without transformers: same ids, same chat template rendering."""
    from transformers import AutoTokenizer
    _, onnx = pair
    ref = AutoTokenizer.from_pretrained(onnx.repo)
    texts = TEXTS + ["  spaces  and\nnewlines\t", "émoji 🙂 et accents", "<|im_end|> inside"]
    for text in texts:
        assert onnx.tokenizer.encode(text, add_special_tokens=False) == ref.encode(text, add_special_tokens=False)
    assert (onnx.tokenizer.pad_token_id, onnx.tokenizer.eos_token_id) == (ref.pad_token_id, ref.eos_token_id)
    messages = [{"role": "user", "content": "Which team? Input: " + TEXTS[0]}]
    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert onnx.tokenizer.apply_chat_template(messages, **kw) == ref.apply_chat_template(messages, **kw)


@needs_export
def test_the_8_bit_export_reads_nearly_the_float_vectors(pair, tmp_path):
    from jul.backends.onnx_export import quantize_int8
    torch_bb, onnx_bb = pair
    w8 = Backbone(str(quantize_int8(Path(onnx_bb.repo), tmp_path / "w8")), "onnx")
    prefix, suffix = ONE_WORD.split("{state}")
    for text in TEXTS:
        expected, _ = PromptTemplate(torch_bb, prefix, suffix).run(text, layers=[3])
        got, _ = PromptTemplate(w8, prefix, suffix).run(text, layers=[3])
        assert cosine(got[3], expected[3]) > 0.99, text


@needs_export
def test_the_prefix_runs_once_and_its_cache_serves_every_query(pair, monkeypatch):
    """With a cache, a query runs its own tokens only, and reads what the whole prompt run at once reads."""
    _, onnx = pair
    prefix, suffix = ONE_WORD.split("{state}")
    template = PromptTemplate(onnx, prefix, suffix)
    assert template._prefix.cache is not None and template._prefix.tokens == template.prefix_tokens
    sizes = []
    real_run = onnx.session.run
    monkeypatch.setattr(onnx.session, "run", lambda names, feed: sizes.append(feed["input_ids"].shape) or real_run(names, feed))
    cached = template.run_batch(TEXTS, layers=[3])
    assert all(width < len(template.prefix_tokens) + 12 for _, width in sizes)
    monkeypatch.setattr(onnx.session, "run", real_run)
    plain = PromptTemplate(onnx, prefix, suffix, use_prefix_cache=False).run_batch(TEXTS, layers=[3])
    for a, b in zip(cached, plain):
        assert cosine(a[3], b[3]) > 0.9999
