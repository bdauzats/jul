"""`jul pack`: a bundle answers exactly what the client that packed it answers, state as the only input."""

import importlib.util

import numpy as np
import pytest

from jul import Choice, Context, Noul, TypeSafeClient
from jul.bundle import Bundle, pack
from jul.presets import ONE_WORD, QUESTION_OPTIONS, Formulation, Preset, repo_fields, save_preset

HAS_EXPORT = all(importlib.util.find_spec(m) for m in ("torch", "onnx", "onnxscript", "onnxruntime"))
pytestmark = pytest.mark.skipif(not HAS_EXPORT, reason="needs torch, onnx, onnxscript, onnxruntime")
QUESTIONS = {
    "team": Choice(instructions="Which team should handle this ticket?",
                   criteria={"billing": "payments, invoices, refunds", "tech": "bugs, errors, crashes",
                             "sales": "plans, pricing, upgrades"}),
    "urgent": Noul(instructions="Is it urgent?"),
}
STATES = ["I was charged twice for my subscription", "the app crashes on export", "do you offer annual plans?",
          "refund my invoice now", "error when I log in", "which plan is cheaper?"]


@pytest.fixture
def client(tiny_models, tmp_path, monkeypatch):
    """A client on a preset fitted for the tiny onnx export, stored in a temporary jul home."""
    import jul.presets
    monkeypatch.setattr(jul.presets, "PRESET_HOME", tmp_path / "presets")
    _, onnx_dir = tiny_models
    save_preset(Preset(name="tiny", **repo_fields("onnx", str(onnx_dir)),
                       formulations=(Formulation("one_word", ONE_WORD, 3),
                                     Formulation("question_options", QUESTION_OPTIONS, 2)),
                       tau=0.05, latency_ms="?", quality="test", center="options", backend="onnx"),
                tmp_path / "presets")
    c = TypeSafeClient(model="tiny", backend="onnx", context_home=tmp_path / "contexts")
    yield c
    c.close()


def probabilities(response):
    return {name: np.array(list(a.probabilities.values())) if hasattr(a, "probabilities") else np.array([a.noul])
            for name, a in response.answers.items()}


def same_answers(bundle, client, context=None):
    for state in STATES:
        want = probabilities(client.system_one(state=state, questions=QUESTIONS, context=context))
        got = probabilities(bundle.system_one(state))
        for name in QUESTIONS:
            assert np.allclose(got[name], want[name], atol=1e-3), (state, name)


def test_a_bundle_answers_what_its_client_answers(client, tmp_path):
    bundle = Bundle.load(pack(client, QUESTIONS, tmp_path / "bundle"))
    assert bundle.question_names == ["team", "urgent"]
    same_answers(bundle, client)


def test_a_shared_prompt_is_read_once_per_state(client, tmp_path):
    bundle = Bundle.load(pack(client, QUESTIONS, tmp_path / "bundle"))
    # the "one word" prompt serves both questions: 1 shared + 1 per question
    assert len(bundle.prompts) == 3 and len(bundle._reads) == 3


def test_a_batch_answers_like_single_calls(client, tmp_path):
    bundle = Bundle.load(pack(client, QUESTIONS, tmp_path / "bundle"))
    batch = bundle.system_one_batch(STATES)
    for state, response in zip(STATES, batch):
        single = probabilities(bundle.system_one(state))
        for name, p in probabilities(response).items():
            assert np.allclose(p, single[name], atol=1e-3)


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="needs scikit-learn (jul[tune])")
def test_a_bundle_carries_the_context_heads_and_calibration(client, tmp_path):
    ctx = Context(name="tickets")
    labels = ["billing", "tech", "sales"] * 10
    words = {"billing": "refund invoice charged", "tech": "crash error bug", "sales": "plan price upgrade"}
    labeled = [(f"{words[l].split()[i % 3]} please, ticket {i}", {"team": l, "urgent": i % 2 == 0})
               for i, l in enumerate(labels)]
    client.autotune(ctx, QUESTIONS, labeled, features="hybrid", save=False)
    bundle = Bundle.load(pack(client, QUESTIONS, tmp_path / "bundle", context=ctx))
    assert any(q.head is not None or q.calibration is not None for q in bundle.questions)
    same_answers(bundle, client, context=ctx)


def test_loading_on_another_backend_warns(client, tmp_path, tiny_models):
    path = pack(client, QUESTIONS, tmp_path / "bundle")
    hf, _ = tiny_models
    with pytest.warns(UserWarning, match="packed on onnx"):
        Bundle.load(path, backend="torch", model=str(hf))


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="needs scikit-learn (jul[tune])")
def test_per_question_formulations_follow_the_head_into_the_bundle(client, tmp_path):
    """A head trained on chosen formulations is answered, and packed, with exactly those."""
    ctx = Context(name="tickets")
    labels = ["billing", "tech", "sales"] * 10
    words = {"billing": "refund invoice charged", "tech": "crash error bug", "sales": "plan price upgrade"}
    labeled = [(f"{words[l].split()[i % 3]} please, ticket {i}", {"team": l, "urgent": i % 2 == 0})
               for i, l in enumerate(labels)]
    client.autotune(ctx, QUESTIONS, labeled, features="hybrid", save=False,
                    formulations={"team": ["question"], "urgent": ["one_word"]})
    heads = {h["meta"]["question"]: h["meta"]["formulations"] for h in ctx.heads.values()}
    assert heads.get("team", ["question"]) == ["question"] and heads.get("urgent", ["one_word"]) == ["one_word"]
    bundle = Bundle.load(pack(client, QUESTIONS, tmp_path / "bundle", context=ctx))
    passes = {q.name: len(q.passes) for q in bundle.questions}
    if ctx.heads:
        assert min(passes.values()) == 1
    same_answers(bundle, client, context=ctx)
