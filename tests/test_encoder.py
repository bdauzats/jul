"""Encoders as backbones (jul/encoder.py): torch and onnx read the same vectors, as the model was trained
to be read, and the rest of jul (calibration checks, autotune, pack) runs on them unchanged.

The model is a tiny random XLM-R (4 layers, width 32) built on the fly, see conftest.py.
"""

import importlib.util

import numpy as np
import pytest

from jul.backbone import Backbone, PromptTemplate
from jul.encoder import templates

HAS_EXPORT = all(importlib.util.find_spec(m) for m in ("torch", "onnx", "onnxscript", "onnxruntime"))
pytestmark = pytest.mark.skipif(not HAS_EXPORT, reason="needs torch, onnx, onnxscript, onnxruntime")
TEXTS = ["I was charged twice for my subscription", "the app crashes on export",
         "do you offer annual plans?", "Je voudrais annuler ma commande"]


def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


@pytest.fixture(scope="module")
def encoders(tiny_encoder):
    hf, onnx = tiny_encoder
    return Backbone(str(hf), "torch"), Backbone(str(onnx), "onnx")


def test_an_encoder_is_recognized_on_both_backends(encoders):
    for bb in encoders:
        assert bb.architecture == "encoder" and bb.n_layers == 4
        assert bb.templates["one_word"] == "{state}"
    assert encoders[1].layers == [1, 2, 3]


def test_the_vector_is_the_mean_the_model_was_trained_on(encoders, tiny_encoder):
    """First half: the masked mean over [CLS] prompt [SEP], sentence-transformers' pooling."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    hf, _ = tiny_encoder
    tok, model = AutoTokenizer.from_pretrained(hf), AutoModel.from_pretrained(hf).eval()
    enc = tok(["query: " + t for t in TEXTS], padding=True, return_tensors="pt")
    with torch.inference_mode():
        h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    expected = ((h * m).sum(1) / m.sum(1)).numpy()
    for bb in encoders:
        rows = PromptTemplate(bb, "query: ", "").run_batch(TEXTS, layers=[3])
        for row, e in zip(rows, expected):
            assert cosine(row[3][:32], e) > 0.9999


@pytest.mark.parametrize("use_prefix_cache", [True, False])
def test_onnx_reads_the_vectors_of_torch(encoders, use_prefix_cache):
    torch_bb, onnx_bb = encoders
    ref = PromptTemplate(torch_bb, "query: ", " thanks")
    got = PromptTemplate(onnx_bb, "query: ", " thanks", use_prefix_cache=use_prefix_cache)
    batched = got.run_batch(TEXTS, layers=onnx_bb.layers)
    for text, b in zip(TEXTS, batched):
        expected, _ = ref.run(text, layers=onnx_bb.layers)
        single, _ = got.run(text, layers=onnx_bb.layers)
        for layer in onnx_bb.layers:
            assert cosine(single[layer], expected[layer]) > 0.9999, (text, layer)
            assert cosine(b[layer], expected[layer]) > 0.9999, (text, layer)


def test_an_encoder_passes_the_calibration_checks(encoders):
    from jul.calibrate import check
    for bb in encoders:
        assert check(bb, bb.n_layers - 1) == []


def test_a_long_input_is_cut_to_the_positions_and_keeps_its_suffix(encoders):
    _, onnx_bb = encoders
    template = PromptTemplate(onnx_bb, "query: ", " thanks")
    with pytest.warns(UserWarning, match="is cut by"):
        row, _ = template.run("word " * 200, layers=[3])
    assert np.isfinite(row[3]).all()
    long_prefix = PromptTemplate(onnx_bb, "query: " + "option, " * 100, " thanks")
    with pytest.warns(UserWarning, match="its prefix by"):
        row, _ = long_prefix.run("my card was stolen", layers=[3])
    assert np.isfinite(row[3]).all()


def test_the_question_template_of_an_encoder_drops_its_options():
    from jul.presets import question_template
    t = templates("query: ")
    assert question_template(t["question_options"]) == t["question"] == "query: {instructions}: {state}"


def test_the_8_bit_export_reads_nearly_the_float_vectors(encoders, tiny_encoder, tmp_path):
    from jul.backends.onnx_export import quantize_int8
    _, onnx = tiny_encoder
    q = Backbone(str(quantize_int8(onnx, tmp_path / "w8")), "onnx")
    a = PromptTemplate(encoders[1], "query: ", "").run_batch(TEXTS, layers=[3])
    b = PromptTemplate(q, "query: ", "").run_batch(TEXTS, layers=[3])
    assert min(cosine(x[3], y[3]) for x, y in zip(a, b)) > 0.99


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="needs scikit-learn (jul[tune])")
def test_a_bundle_packed_on_an_encoder_answers_what_its_client_answers(tiny_encoder, tmp_path, monkeypatch):
    import jul.presets
    from jul import Bundle, Choice, Context, Noul, TypeSafeClient, pack
    from jul.presets import Formulation, Preset, repo_fields, save_preset
    monkeypatch.setattr(jul.presets, "PRESET_HOME", tmp_path / "presets")
    _, onnx = tiny_encoder
    t = templates("query: ")
    save_preset(Preset(name="micro", **repo_fields("onnx", str(onnx)), backend="onnx", tau=0.05,
                       formulations=(Formulation("one_word", t["one_word"], 3),
                                     Formulation("question_options", t["question_options"], 2)),
                       latency_ms="?", quality="test"), tmp_path / "presets")
    client = TypeSafeClient(model="micro", backend="onnx", context_home=tmp_path / "contexts")
    questions = {"team": Choice(instructions="Which team?", criteria={"billing": "payments, refunds",
                                                                      "tech": "bugs, crashes", "sales": "plans, pricing"}),
                 "urgent": Noul(instructions="Is it urgent?")}
    words = {"billing": "refund invoice charged", "tech": "crash error bug", "sales": "plan price upgrade"}
    labeled = [(f"{words[l].split()[i % 3]} please, ticket {i}", {"team": l, "urgent": i % 2 == 0})
               for i, l in enumerate(["billing", "tech", "sales"] * 10)]
    ctx = Context(name="tickets")
    client.autotune(ctx, questions, labeled, features="hybrid", save=False,
                    formulations={"team": ["one_word"], "urgent": ["question"]})
    bundle = Bundle.load(pack(client, questions, tmp_path / "bundle", context=ctx))
    for state in ["refund my invoice now", "the app crashes", "which plan is cheaper?"]:
        want = client.system_one(state=state, questions=questions, context=ctx).answers
        got = bundle.system_one(state).answers
        assert got["team"].choice == want["team"].choice
        assert np.allclose(list(got["team"].probabilities.values()), list(want["team"].probabilities.values()),
                           atol=1e-3)
        assert abs(got["urgent"].noul - want["urgent"].noul) < 1e-3
    client.close()
