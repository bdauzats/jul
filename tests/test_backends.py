"""The backend is an implementation detail: same interface, same vectors, and nothing learned on one
backend (centers, heads, calibrations) is silently reused by another."""

import importlib.util
import os

import numpy as np
import pytest

from jul.backbone import PromptTemplate, model_key, repo_for, resolve_backend
from jul.context import Context
from jul.presets import ONE_WORD, resolve

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_MLX = importlib.util.find_spec("mlx") is not None
TEXTS = ["I was charged twice for my subscription", "the app crashes on export",
         "do you offer annual plans?", "Je voudrais annuler ma commande"]


def test_the_backend_comes_from_the_argument_then_the_environment(monkeypatch):
    monkeypatch.setenv("JUL_BACKEND", "torch")
    assert resolve_backend() == "torch"
    assert resolve_backend("mlx") == "mlx"
    with pytest.raises(ValueError):
        resolve_backend("onnx")


@pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed")
def test_mlx_is_the_default_on_apple_silicon(monkeypatch):
    monkeypatch.delenv("JUL_BACKEND", raising=False)
    assert resolve_backend() == "mlx"


def test_each_backend_has_its_own_repo():
    assert repo_for("minicpm5-2b", "mlx") == "openbmb/MiniCPM5-2B-MLX"
    assert repo_for("minicpm5-2b", "torch") == "openbmb/MiniCPM5-2B"
    assert repo_for("some/repo", "torch") == "some/repo"
    assert resolve("minicpm5-2b").repos == {"mlx": "openbmb/MiniCPM5-2B-MLX", "torch": "openbmb/MiniCPM5-2B"}


def test_what_mlx_learned_is_not_reused_by_torch():
    """MLX keeps the bare name, so contexts saved before backends existed stay valid."""
    preset = resolve("minicpm5-2b")
    f = preset.formulations[0]
    assert model_key(preset.name, "mlx") == preset.name
    ctx = Context()
    ctx.set_center(model_key(preset.name, "mlx"), f, np.ones(3))
    assert np.allclose(ctx.center_for(preset, f), 1)
    assert ctx.center_for(model_key(preset.name, "torch"), f) is None


def test_a_backend_without_its_own_generic_center_falls_back_to_the_mlx_one():
    preset = resolve("qwen3.5-9b")
    f = preset.formulations[0]
    assert np.array_equal(preset.generic_center(f, "torch"), preset.generic_center(f))


# --- the torch backend on its own -----------------------------------------------------------------
#
# These need torch but NOT MLX, so CI (Linux, no Apple Silicon) runs them on every push. The model is
# small on purpose: `JUL_TEST_MODEL` overrides it, the default is ~1.2 GB in bf16.

def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


SMALL = os.environ.get("JUL_TEST_MODEL", "qwen3-0.6b")


@pytest.fixture(scope="module")
def small_torch():
    from jul.backbone import Backbone
    return Backbone(SMALL, "torch")


def vectors(backbone, layer, **kw):
    prefix, suffix = ONE_WORD.split("{state}")
    template = PromptTemplate(backbone, prefix, suffix, **kw)
    out = []
    for text in TEXTS:
        h, _ = template.run(text, layers=[layer])
        out.append(h[layer][: h[layer].shape[0] // 2])
    return np.stack(out)


@pytest.mark.slow
@pytest.mark.torch
@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_the_torch_prefix_cache_is_a_pure_speed_up(small_torch):
    """The cached prefix must be a pure optimisation, at every depth.

    `cache_prefix` runs the whole stack, while a query stops at its layer: the crop that follows must
    leave the prefix intact for the layers above the stop as well. Checked from a quarter of the depth
    to the last layer, because only the last one was exercised before.
    """
    for layer in sorted({max(0, round(f * small_torch.n_layers) - 1) for f in (0.25, 0.5, 0.75, 1.0)}):
        cached = vectors(small_torch, layer)
        plain = vectors(small_torch, layer, use_prefix_cache=False)
        assert min(cosine(x, y) for x, y in zip(cached, plain)) > 0.999, f"layer {layer}"
        # the cache is back to the prefix: a second identical call reads the same vectors
        assert np.allclose(vectors(small_torch, layer), cached, atol=1e-3), f"layer {layer}"


@pytest.mark.slow
@pytest.mark.torch
@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_the_torch_prefix_cache_does_not_drift_over_many_calls(small_torch):
    """Queries of varying length through one template: state must not accumulate."""
    layer = small_torch.n_layers // 2
    prefix, suffix = ONE_WORD.split("{state}")
    template = PromptTemplate(small_torch, prefix, suffix)
    first, _ = template.run(TEXTS[0], layers=[layer])
    for i in range(20):
        template.run(TEXTS[i % len(TEXTS)] + " " + "x " * (i % 7), layers=[layer])
    last, _ = template.run(TEXTS[0], layers=[layer])
    assert np.allclose(first[layer], last[layer], atol=1e-3)


@pytest.mark.slow
@pytest.mark.torch
@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_a_model_passes_the_calibration_checks_on_torch(small_torch):
    """`jul models add` refuses to fit a model that fails these: they are the backend's contract."""
    from jul.calibrate import check
    assert check(small_torch, small_torch.n_layers - 1) == []


# --- torch against MLX, on the same model ---------------------------------------------------------
#
# Apple Silicon only: both frameworks must be installed and the weights are the 2B ones.

@pytest.fixture(scope="module")
def pair():
    """The same bf16 weights on both sides: the preset's MLX repo is 4-bit, which alone moves the
    vectors to a cosine of ~0.95 with the bf16 ones."""
    from jul.backbone import Backbone
    return Backbone("openbmb/MiniCPM5-2B", "mlx"), Backbone("minicpm5-2b", "torch")


@pytest.mark.slow
@pytest.mark.torch
@pytest.mark.skipif(not (HAS_TORCH and HAS_MLX), reason="needs both backends")
def test_torch_reads_the_same_vectors_as_mlx(pair):
    mlx, torch = pair
    assert mlx.n_layers == torch.n_layers
    for layer in resolve("minicpm5-2b").layers:
        a, b = vectors(mlx, layer), vectors(torch, layer)
        assert min(cosine(x, y) for x, y in zip(a, b)) > 0.999


@pytest.mark.slow
@pytest.mark.torch
@pytest.mark.skipif(not (HAS_TORCH and HAS_MLX), reason="needs both backends")
def test_torch_and_mlx_agree_on_the_answer(pair):
    from jul import Choice
    from jul.engine import Engine
    from jul.types import options_of
    question = Choice(instructions="Which team should handle this ticket?",
                      criteria={"billing": "payments, invoices", "technical": "bugs, errors",
                                "sales": "plans, pricing, upgrades"})
    options = options_of(question)
    results = []
    for backbone in pair:
        engine = Engine(resolve("minicpm5-2b"), backbone=backbone)
        compiled = engine.compile("choice", question.instructions, options)
        results.append(np.stack([engine.vector_probabilities(compiled, t)[0] for t in TEXTS[:3]]))
    assert np.array_equal(results[0].argmax(1), results[1].argmax(1))
    # bf16 kernels differ between the frameworks, and tau ~0.04 amplifies it: measured up to 0.03 on
    # a close call (0.565 / 0.595), 1e-6 on a clear one.
    assert np.abs(results[0] - results[1]).max() < 0.05
