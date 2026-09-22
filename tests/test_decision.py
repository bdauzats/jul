"""The pointer method (jul/decision.py) against Kev's own code, which trained minicpm5-2b-decision.

fixtures/decision_kev.json was written by Kev (torch, fp32, merged-free LoRA): for three requests, the
prefix and branch token ids its encoder produces and the probabilities its model gives (calibrated,
T = 1.954). The fast tests check that jul encodes every request token for token like Kev; the slow one
that the MLX 4-bit model gives the same answers.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from jul.decision import DecisionSpec, PointerReader, render
from jul.types import Choice, Noul, NoulCriteria, Score, options_of

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "decision_kev.json").read_text())
MODELS = Path(os.environ.get("JUL_DECISION_MODELS", Path(__file__).resolve().parents[2] / "models"))


def question(q: dict):
    if q["type"] == "choice":
        c = q["criteria"]
        return Choice(q["instructions"], {k: v or "" for k, v in c.items()} if any(c.values()) else list(c))
    if q["type"] == "noul":
        c = q.get("criteria") or {}
        return Noul(q["instructions"], NoulCriteria(true=c.get("true", ""), false=c.get("false", "")) if c else None)
    return Score(q["instructions"], q["criteria"])


def kind(q: dict) -> str:
    return q["type"]


@pytest.fixture(scope="module")
def reader():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("openbmb/MiniCPM5-2B", local_files_only=True)
    except Exception:
        pytest.skip("the MiniCPM5-2B tokenizer is not in the local Hugging Face cache")

    class Backbone:
        name, tokenizer = "minicpm5-2b-decision", tok

    return PointerReader(Backbone(), DecisionSpec.load(FIXTURES, FIXTURES / "decision_minicpm5-2b.json"))


def test_objects_are_rendered_as_labeled_fields():
    assert render({"subject": "Hi", "items": ["a", "b"]}) == "subject: Hi\nitems:\n  - a\n  - b"


@pytest.mark.parametrize("case", CASES, ids=lambda c: "+".join(c["request"]["questions"]))
def test_requests_are_encoded_token_for_token_like_kev(reader, case):
    req = case["request"]
    assert reader.encode_state(req["state"]) == case["prefix"]
    for q, want in zip(req["questions"].values(), case["branches"]):
        texts, _ = reader.option_texts(kind(q), options_of(question(q)))
        branch, q_idx, opt_idx = reader.encode_question(q["instructions"], texts)
        assert (branch, q_idx, opt_idx) == (want["ids"], want["decide"], want["opts"])


def test_noul_options_follow_the_trained_order_and_map_back(reader):
    options = options_of(Noul("Late?", NoulCriteria(true="the parcel is late", false="anything else")))
    texts, index = reader.option_texts("noul", options)
    assert texts == ["no: anything else", "yes: the parcel is late"]
    assert [options[i].key for i in index] == ["false", "true"]
    # jul's default descriptions are not shown to the model, which was trained without them
    texts, _ = reader.option_texts("noul", options_of(Noul("Late?")))
    assert texts == ["no", "yes"]


@pytest.mark.slow
@pytest.mark.parametrize("weights, tolerance", [("mlx-bf16", 0.03), ("mlx-4bit", 0.3)])
def test_mlx_answers_match_kev(tmp_path, monkeypatch, weights, tolerance):
    """bf16 checks the port itself; 4-bit moves probabilities more (up to ~0.3 measured) but not the answer."""
    model_dir = MODELS / f"minicpm5-2b-decision-{weights}"
    if not (model_dir / "decision.json").exists():
        pytest.skip(f"{model_dir} is not on disk")
    from jul import TypeSafeClient
    from jul.presets import pointer_preset, save_preset
    monkeypatch.setattr("jul.presets.PRESET_HOME", tmp_path)
    save_preset(pointer_preset("decision-test", str(model_dir), "mlx"), home=tmp_path)
    client = TypeSafeClient(model="decision-test", backend="mlx")
    for case in CASES:
        qs = {name: question(q) for name, q in case["request"]["questions"].items()}
        answers = client.system_one(state=case["request"]["state"], questions=qs).answers
        for name, want in case["probabilities"].items():
            a = answers[name]
            got = a.probabilities if hasattr(a, "probabilities") else {"true": a.noul, "false": 1 - a.noul}
            keys = list(want)
            assert np.argmax([got[k] for k in keys]) == np.argmax([want[k] for k in keys]), name
            assert max(abs(got[k] - want[k]) for k in keys) < tolerance, (name, got, want)
    client.close()


def test_mlx_gets_the_rope_base_that_transformers_5_moved(tmp_path):
    """Without it mlx-lm silently uses 10000 instead of the model's base, and answers are wrong."""
    pytest.importorskip("mlx")
    from jul.backends.mlx import _rope_fix
    (tmp_path / "config.json").write_text(json.dumps({"rope_parameters": {"rope_theta": 5000000}}))
    assert _rope_fix(str(tmp_path)) == {"rope_theta": 5000000}
    (tmp_path / "config.json").write_text(json.dumps({"rope_theta": 10000, "rope_parameters": {"rope_theta": 10000}}))
    assert _rope_fix(str(tmp_path)) is None


def test_a_decision_model_is_recognised_by_its_spec_file(tmp_path):
    """Without this, `jul models add` would run the vector calibration on a decision model."""
    from jul.decision import spec_source
    assert spec_source(str(tmp_path)) is None
    (tmp_path / "decision.json").write_text("{}")
    assert spec_source(str(tmp_path)) == str(tmp_path.resolve())
