"""Engine pieces that need no model."""

import numpy as np

from jul.engine import normalize, short_names, softmax
from jul.presets import PRESETS, one_word_preset, resolve


def test_short_names_drops_the_words_every_option_shares():
    labels = ["This example is about sports", "This example is about health"]
    assert short_names(labels) == ["sports", "health"]


def test_short_names_leaves_already_short_options_alone():
    assert short_names(["billing", "technical"]) == ["billing", "technical"]


def test_short_names_never_empties_an_option():
    assert all(short_names(["a b", "a b"]))


def test_softmax_is_a_distribution_and_shift_invariant():
    z = np.array([1.0, 2.0, 3.0])
    assert np.isclose(softmax(z).sum(), 1)
    assert np.allclose(softmax(z), softmax(z + 100))


def test_normalize_gives_unit_vectors():
    assert np.isclose(np.linalg.norm(normalize(np.array([3.0, 4.0]))), 1)


def test_presets_are_aliased_and_unknown_names_are_refused():
    assert resolve("fast").name == "minicpm5-2b"
    assert resolve("accurate").name == "wemm-4b-4bit"
    assert resolve(None).name == "wemm-4b-4bit"
    try:
        resolve("gpt-9")
    except ValueError as e:
        assert "Unknown model" in str(e)
    else:
        raise AssertionError("an unknown preset should be refused")


def test_every_preset_carries_a_layer_and_a_temperature():
    for preset in PRESETS.values():
        assert preset.tau > 0
        assert preset.formulations and all(f.layer > 0 for f in preset.formulations)
        assert preset.center in {"generic", "options", "none"}


def test_the_one_word_variant_keeps_a_single_formulation():
    preset = one_word_preset("minicpm5-2b")
    assert len(preset.formulations) == 1
    assert preset.tau != resolve("minicpm5-2b").tau     # its own fitted temperature


def test_minicpm_ships_its_generic_center():
    preset = resolve("minicpm5-2b")
    center = preset.generic_center(preset.formulations[0])
    assert center is not None and center.ndim == 1


def test_softmax_of_a_single_option_is_certain():
    assert softmax(np.array([0.3])).tolist() == [1.0]


def test_softmax_refuses_a_degenerate_vector_rather_than_returning_nan():
    import pytest
    with pytest.raises(ValueError, match="degenerate"):
        softmax(np.array([np.nan, 0.2]))


def test_normalize_of_a_zero_vector_is_nan_without_a_numpy_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.isnan(out[0]).all() and out[1].tolist() == [0.6, 0.8]
