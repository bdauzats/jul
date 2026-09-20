"""Turning probabilities into the three answer types."""

import numpy as np
import pytest

from jul.client import _answer_index, _format, _kind_of
from jul.types import Choice, Noul, Option, Score


def options(*pairs):
    return [Option(k, d) for k, d in pairs]


def test_choice_reports_the_most_likely_option():
    answer = _format("choice", None, options(("a", "x"), ("b", "y")), np.array([0.3, 0.7]))
    assert answer.choice == "b" and answer.confidence == 0.7
    assert answer.probabilities == {"a": 0.3, "b": 0.7}


def test_noul_is_the_probability_of_true():
    answer = _format("noul", None, options(("true", ""), ("false", "")), np.array([0.25, 0.75]))
    assert answer.noul == 0.25


def test_score_is_the_mean_level_and_carries_its_legend():
    answer = _format("score", None, options(("0", "Calm"), ("1", "Mad"), ("2", "Furious")),
                     np.array([0.0, 0.0, 1.0]))
    assert answer.score == 2.0
    assert answer.confidence == 1.0        # all the mass on one level
    assert answer.legend == {"0": "Calm", "1": "Mad", "2": "Furious"}


def test_score_confidence_is_lowest_when_split_between_the_ends():
    answer = _format("score", None, options(("0", "Calm"), ("1", "Mad"), ("2", "Furious")),
                     np.array([0.5, 0.0, 0.5]))
    assert answer.confidence == 0.0


def test_kind_is_read_from_the_question_type():
    assert _kind_of(Choice()) == "choice" and _kind_of(Noul()) == "noul" and _kind_of(Score()) == "score"
    with pytest.raises(TypeError):
        _kind_of("not a question")


def test_labeled_answers_accept_keys_booleans_and_level_indices():
    assert _answer_index("choice", "billing", {"billing": 0, "tech": 1}) == 0
    assert _answer_index("noul", True, {"true": 0, "false": 1}) == 0
    assert _answer_index("noul", "false", {"true": 0, "false": 1}) == 1
    assert _answer_index("score", 2, {"0": 0, "1": 1, "2": 2}) == 2
    with pytest.raises(ValueError):
        _answer_index("choice", "nope", {"billing": 0})
