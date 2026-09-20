"""The per-task head, its safety net and the example-count tiers."""

import numpy as np
import pytest

from jul import tuning


def dataset(n=400, d=32, k=3, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (k, d))
    y = rng.integers(0, k, n)
    return centers[y] + rng.normal(0, noise, (n, d)), y, rng


def test_a_head_that_beats_zero_shot_is_activated():
    x, y, rng = dataset()
    blind = rng.normal(0, 1, (len(y), 3))          # a zero-shot that knows nothing
    head, report = tuning.train(x, y, blind, ["a", "b", "c"], "q", "minicpm5-2b")
    assert head is not None and report.activated
    assert report.head_accuracy > report.zero_shot_accuracy


def test_a_head_that_does_not_beat_zero_shot_is_refused():
    x, y, _ = dataset()
    perfect = np.eye(3)[y] * 10                     # a zero-shot that is always right
    head, report = tuning.train(x, y, perfect, ["a", "b", "c"], "q", "minicpm5-2b")
    assert head is None and not report.activated
    assert "does not beat zero-shot" in report.reason


def test_below_the_floor_only_calibration_is_fitted():
    x, y, rng = dataset(n=15)
    head, report = tuning.train(x, y, rng.normal(0, 1, (15, 3)), ["a", "b", "c"], "q", "minicpm5-2b")
    assert head is None and report.head_accuracy is None
    assert str(tuning.min_examples(3)) in report.reason


def test_the_floor_scales_with_the_number_of_options():
    """Measured: 200 examples were not enough for Banking77's 72 options (JOURNAL §9 nonies)."""
    assert tuning.min_examples(2) == tuning.min_examples(4) == 20
    assert tuning.min_examples(72) == 216


def test_the_gate_judges_every_example_out_of_fold():
    x, y, rng = dataset()
    head, report = tuning.train(x, y, rng.normal(0, 1, (len(y), 3)), ["a", "b", "c"], "q", "m")
    assert report.n_evaluated == len(y)          # all of them, not a fifth
    assert "cross-validation" in report.reason
    assert head["meta"]["n_folds"] >= 2


def test_an_option_with_no_training_example_stops_the_head():
    x, y, rng = dataset(n=120, k=3)
    y = np.where(y == 2, 0, y)                      # option 2 disappears
    head, report = tuning.train(x, y, rng.normal(0, 1, (len(y), 3)), ["a", "b", "c"], "q", "m")
    assert head is None and not report.activated


def test_head_probabilities_are_a_distribution():
    x, y, rng = dataset()
    head, _ = tuning.train(x, y, rng.normal(0, 1, (len(y), 3)), ["a", "b", "c"], "q", "m")
    one = tuning.apply(head, x[0])
    batch = tuning.apply(head, x[:5])
    assert one.shape == (3,) and np.isclose(one.sum(), 1)
    assert batch.shape == (5, 3) and np.allclose(batch.sum(1), 1)


def test_the_holdout_keeps_every_option_and_is_disjoint():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    rest, held = tuning.stratified_split(y, 0.25, seed=0)
    assert set(rest) & set(held) == set()
    assert len(rest) + len(held) == len(y)
    assert set(y[held]) == {0, 1, 2}


def test_the_head_records_what_it_was_trained_on():
    x, y, rng = dataset()
    head, _ = tuning.train(x, y, rng.normal(0, 1, (len(y), 3)), ["a", "b", "c"], "topic", "minicpm5-2b")
    assert head["meta"]["preset"] == "minicpm5-2b"
    assert head["meta"]["options"] == ["a", "b", "c"]
    assert head["meta"]["question"] == "topic"


def test_the_report_reads_as_a_sentence():
    x, y, rng = dataset()
    _, report = tuning.train(x, y, rng.normal(0, 1, (len(y), 3)), ["a", "b", "c"], "topic", "m")
    text = str(report)
    assert "zero-shot accuracy" in text and "topic" in text
