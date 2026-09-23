"""The prompt-prefix cache must be a pure speed-up: same answer, and never corrupted by a call."""

import numpy as np
import pytest

from jul import Choice
from jul.backbone import Backbone, PromptTemplate
from jul.engine import Engine
from jul.presets import resolve

PREFIX = 'This text: "'
SUFFIX = '" means in one word: "'


@pytest.fixture(scope="module")
def backbone():
    return Backbone("minicpm5-2b")


def vector(template, backbone, text, layer=39):
    h, _ = template.run(text, layers=[layer])
    return h[layer][: h[layer].shape[0] // 2]


def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


@pytest.mark.slow
@pytest.mark.parametrize("text", ["I was charged twice for my subscription",
                                  "the app crashes on export",
                                  "do you offer annual plans?"])
def test_the_cached_prefix_points_the_same_way_as_no_cache(backbone, text):
    """4-bit kernels differ slightly between a full prefill and a cached one, on a few outlier
    dimensions. Only the direction is read (cosine), and it is preserved to ~2e-4."""
    cached = PromptTemplate(backbone, PREFIX, SUFFIX, use_prefix_cache=True)
    plain = PromptTemplate(backbone, PREFIX, SUFFIX, use_prefix_cache=False)
    assert cosine(vector(cached, backbone, text), vector(plain, backbone, text)) > 0.999


@pytest.mark.slow
def test_the_cache_does_not_change_the_answer(backbone):
    from jul.types import options_of
    question = Choice(instructions="Which team should handle this ticket?",
                      criteria={"billing": "payments, invoices", "technical": "bugs, errors"})
    options = options_of(question)
    engine = Engine(resolve("minicpm5-2b"), backbone=backbone)
    compiled = engine.compile("choice", question.instructions, options)
    probabilities, _, _ = engine.vector_probabilities(compiled, "I was charged twice")

    for p in compiled.passes:                 # same prompts, without any prefix cache
        p.template = PromptTemplate(backbone, p.template.prefix_text, p.template.suffix_text,
                                    use_prefix_cache=False)
    uncached, _, _ = engine.vector_probabilities(compiled, "I was charged twice")
    assert np.argmax(probabilities) == np.argmax(uncached)
    assert np.allclose(probabilities, uncached, atol=0.02)


@pytest.mark.slow
def test_a_call_does_not_disturb_the_next_one(backbone):
    """The cache is trimmed (or the state copied) back to the prefix after every query."""
    template = PromptTemplate(backbone, PREFIX, SUFFIX)
    first = vector(template, backbone, "the app crashes on export")
    vector(template, backbone, "a completely different and much longer sentence about invoices")
    again = vector(template, backbone, "the app crashes on export")
    assert np.allclose(first, again, atol=1e-3)


@pytest.mark.slow
def test_option_vectors_are_computed_once_per_question(backbone):
    engine = Engine(resolve("minicpm5-2b"), backbone=backbone)
    question = Choice(instructions="Which team?", criteria={"billing": "payments", "tech": "bugs"})
    from jul.types import options_of
    options = options_of(question)
    first = engine.compile("choice", question.instructions, options)
    second = engine.compile("choice", question.instructions, options)
    assert first is second
    assert all(np.shares_memory(a.centered_options, b.centered_options)
               for a, b in zip(first.passes, second.passes))


@pytest.mark.slow
def test_the_one_word_vector_is_shared_by_the_questions_of_a_call(client):
    """Two questions cost less than twice one question, because the shared pass is computed once."""
    one = client.system_one(state="I was charged twice",
                            questions={"a": Choice(instructions="Which team?",
                                                   criteria={"billing": "payments", "tech": "bugs"})})
    two = client.system_one(state="I was charged twice",
                            questions={"a": Choice(instructions="Which team?",
                                                   criteria={"billing": "payments", "tech": "bugs"}),
                                       "b": Choice(instructions="Which urgency?",
                                                   criteria={"low": "can wait", "high": "urgent"})})
    assert two.usage.input_tokens < 2 * one.usage.input_tokens


@pytest.mark.slow
def test_repeating_a_call_gives_the_same_answer(client):
    question = {"team": Choice(instructions="Which team should handle this ticket?",
                               criteria={"billing": "payments, invoices", "technical": "bugs, errors"})}
    first = client.system_one(state="I was charged twice", questions=question)
    second = client.system_one(state="I was charged twice", questions=question)
    assert first.choices["team"].probabilities == second.choices["team"].probabilities
    assert first.request_id != second.request_id       # a fresh id per call


@pytest.mark.slow
@pytest.mark.parametrize("use_prefix_cache", [True, False])
def test_an_mlx_batch_reads_the_vectors_of_single_queries(backbone, use_prefix_cache):
    """Both halves of the features, at a middle layer and the last one, for texts of different
    lengths. 4-bit kernels round a batch slightly differently from a single row (~2e-4, the same order
    as a cached prefix against a full prefill), padded or not."""
    texts = ["I was charged twice for my subscription", "the app crashes on export",
             "do you offer annual plans?", "Je voudrais annuler ma commande"]
    template = PromptTemplate(backbone, PREFIX, SUFFIX, use_prefix_cache=use_prefix_cache)
    layers = [backbone.n_layers // 2, backbone.n_layers - 1]
    batched = template.run_batch(texts, layers=layers)
    for text, h in zip(texts, batched):
        single, _ = template.run(text, layers=layers)
        for layer in layers:
            assert cosine(h[layer], single[layer]) > 0.999, (text, layer)
    # the batch ran on its own cache: the template's prefix is untouched
    again, _ = template.run(texts[0], layers=layers)
    assert cosine(again[layers[1]], batched[0][layers[1]]) > 0.999


@pytest.mark.slow
def test_the_padding_of_an_mlx_batch_is_never_read(backbone, monkeypatch):
    """Right padding under the causal mask alone: whatever tokens fill the padding, the real tokens
    give exactly the same features."""
    template = PromptTemplate(backbone, PREFIX, SUFFIX, use_prefix_cache=False)
    texts = ["the app crashes on export", "I was charged twice for my subscription, and nobody answers"]
    layers = [backbone.n_layers // 2, backbone.n_layers - 1]
    reference = template.run_batch(texts, layers=layers)[0]
    for pad in (777, 1234):
        monkeypatch.setattr(backbone, "_pad", pad)
        h = template.run_batch(texts, layers=layers)[0]
        assert all(np.array_equal(h[layer], reference[layer]) for layer in layers), pad
