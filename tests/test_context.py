"""Context compilation and its disk cache."""

import numpy as np
import pytest

from jul import Context
from jul.context import question_digest, resolve_context
from jul.presets import resolve
from jul.types import Option

#: These tests exercise the plumbing with a handful of examples; the "too few examples"
#: warning is checked on its own below.
few = pytest.mark.filterwarnings("ignore:Context has")


@few
def test_saving_and_loading_keeps_description_examples_and_centers(home):
    preset = resolve("minicpm5-2b")
    formulation = preset.formulations[0]
    context = Context(description="Support tickets.", examples=["one", "two"], name="tickets")
    context.set_center(preset, formulation, np.arange(4, dtype=np.float32))
    context.save(home=home)

    loaded = Context.load("tickets", home=home)
    assert loaded.description == "Support tickets."
    assert loaded.examples == ["one", "two"]
    assert np.allclose(loaded.center_for(preset, formulation), np.arange(4))


@few
def test_a_center_belongs_to_one_preset_and_one_formulation(home):
    minicpm, qwen = resolve("minicpm5-2b"), resolve("qwen3.5-9b")
    context = Context(name="t", examples=["a"])
    context.set_center(minicpm, minicpm.formulations[0], np.ones(3))
    assert context.center_for(qwen, qwen.formulations[0]) is None
    assert context.center_for(minicpm, minicpm.formulations[1]) is None


def test_listing_and_deleting(home):
    Context(name="a", description="x").save(home=home)
    Context(name="b", description="y").save(home=home)
    assert Context.list_saved(home) == ["a", "b"]
    assert Context.delete("a", home) is True
    assert Context.list_saved(home) == ["b"]
    assert Context.delete("a", home) is False


@few
def test_cache_key_changes_with_what_reaches_the_prompt():
    base = Context(description="tickets", examples=["a", "b"], name="t")
    assert base.cache_key() == Context(description="tickets", examples=["a", "b"], name="t").cache_key()
    assert base.cache_key() != Context(description="tickets", examples=["a"], name="t").cache_key()
    # The description is off by default, so changing it changes nothing in the prompts.
    assert base.cache_key() == Context(description="other", examples=["a", "b"], name="t").cache_key()
    # Once it is switched on, it does reach the prompts.
    on = Context(description="tickets", examples=["a", "b"], name="t", use_description=True)
    assert on.cache_key() != base.cache_key()
    assert on.cache_key() != Context(description="other", examples=["a", "b"], name="t",
                                     use_description=True).cache_key()


@few
def test_the_description_is_off_by_default():
    """Measured harmful on average on both presets (JOURNAL §9 decies)."""
    assert Context(description="some data").use_description is False


@few
def test_whether_the_description_is_used_survives_a_save(home):
    Context(description="d", name="on", use_description=True).save(home=home)
    Context(description="d", name="off").save(home=home)
    assert Context.load("on", home=home).use_description is True
    assert Context.load("off", home=home).use_description is False


def test_resolve_accepts_an_object_a_saved_name_or_nothing(home):
    Context(name="saved", description="d").save(home=home)
    assert resolve_context(None) is None
    assert resolve_context("saved", home).description == "d"
    obj = Context(description="inline")
    assert resolve_context(obj) is obj
    with pytest.raises(TypeError):
        resolve_context(42)


def test_loading_an_unknown_context_says_so(home):
    with pytest.raises(FileNotFoundError):
        Context.load("missing", home=home)


def test_a_question_digest_is_tied_to_the_preset_and_the_options():
    options = [Option("a", "x"), Option("b", "y")]
    base = question_digest("minicpm5-2b", "choice", "q", options)
    assert base == question_digest("minicpm5-2b", "choice", "q", options)
    assert base != question_digest("qwen3.5-9b", "choice", "q", options)
    assert base != question_digest("minicpm5-2b", "choice", "other question", options)
    assert base != question_digest("minicpm5-2b", "choice", "q", options + [Option("c", "z")])


def test_a_context_needs_a_name_to_be_saved(home):
    with pytest.raises(ValueError):
        Context(description="x").save(home=home)


def test_too_few_examples_warns_because_the_center_is_noise():
    with pytest.warns(UserWarning, match="mostly noise"):
        Context(examples=["one", "two", "three"])


def test_enough_examples_or_none_at_all_stay_quiet(recwarn):
    Context(examples=[f"example {i}" for i in range(60)])
    Context(description="no examples, no center")
    assert [w for w in recwarn if "mostly noise" in str(w.message)] == []
