"""Question and answer shapes: the part that must stay identical to the Jev SDK."""

import json

import pytest

from jul import Choice, Noul, NoulCriteria, Score, SystemOneResponse, Usage
from jul.types import ChoiceAnswer, NoulAnswer, ScoreAnswer, options_of, serialize_state


def test_choice_options_keep_order_and_keys():
    options = options_of(Choice(instructions="q", criteria={"billing": "payments", "tech": "bugs"}))
    assert [o.key for o in options] == ["billing", "tech"]
    assert [o.text for o in options] == ["payments", "bugs"]


def test_choice_from_a_list_uses_the_keys_as_text():
    options = options_of(Choice(instructions="q", criteria=["sports", "health"]))
    assert [(o.key, o.text) for o in options] == [("sports", "sports"), ("health", "health")]


def test_noul_is_always_true_then_false():
    options = options_of(Noul(instructions="q", criteria=NoulCriteria(true="it is a bug", false="it is not")))
    assert [o.key for o in options] == ["true", "false"]
    assert options[0].text == "it is a bug"


def test_noul_defaults_and_yes_no_aliases():
    assert [o.text for o in options_of(Noul(instructions="q"))] == ["Yes.", "No."]
    assert options_of(Noul(instructions="q", criteria={"yes": "sure"}))[0].text == "sure"


def test_score_levels_are_indexed_from_zero():
    options = options_of(Score(instructions="q", criteria=["Calm", "Angry", "Furious"]))
    assert [o.key for o in options] == ["0", "1", "2"]


@pytest.mark.parametrize("question", [Score(criteria=["only"])])
def test_a_question_needs_at_least_two_options(question):
    with pytest.raises(ValueError):
        options_of(question)


def test_choice_accepts_single_option():
    """TypeSafe accepts single-option Choice (e.g., CLICK actions); JuL should too."""
    options = options_of(Choice(criteria={"click": "click the element"}))
    assert len(options) == 1
    assert options[0].key == "click"


def test_state_is_serialized_once_and_strings_pass_through():
    assert serialize_state("plain text") == "plain text"
    assert json.loads(serialize_state({"ticket": "charged twice"})) == {"ticket": "charged twice"}


def test_response_splits_answers_by_type():
    response = SystemOneResponse(
        answers={"team": ChoiceAnswer("billing", {"billing": 0.9, "tech": 0.1}, 0.9),
                 "is_bug": NoulAnswer(0.04),
                 "anger": ScoreAnswer(0.6, {"0": "Calm"}, {"0": 1.0}, 0.8)},
        model="minicpm5-2b", usage=Usage(input_tokens=42), request_id="abc")
    assert list(response.choices) == ["team"]
    assert list(response.nouls) == ["is_bug"]
    assert list(response.scores) == ["anger"]
    assert response.answers["team"].choice == "billing"


def test_response_serializes_like_the_jev_api():
    response = SystemOneResponse(answers={"team": ChoiceAnswer("billing", {"billing": 1.0}, 1.0)},
                                 model="m", usage=Usage(input_tokens=7), request_id="abc")
    payload = json.loads(json.dumps(response.as_dict()))
    assert payload["request_id"] == "abc" and payload["model"] == "m"
    assert payload["usage"] == {"input_tokens": 7, "output_tokens": 0, "total_tokens": 7}
    assert payload["answers"]["team"]["choice"] == "billing"


def test_usage_reports_no_generated_tokens():
    assert Usage(input_tokens=10).total_tokens == 10
