"""Typed questions and answers, with the same names and fields as the TypeSafe (Jev) Python SDK.

A script written against `typesafe_sdk` runs unchanged by swapping its import for `jul`, so the
shapes here are fixed by that SDK rather than chosen: `Choice`/`Noul`/`Score` in, a
`SystemOneResponse` out whose `choices`, `nouls` and `scores` are keyed by the question name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "Choice", "Noul", "NoulCriteria", "Score", "Question",
    "ChoiceAnswer", "NoulAnswer", "ScoreAnswer", "Usage", "SystemOneResponse",
    "Option", "options_of", "serialize_state",
]


# --- questions ---------------------------------------------------------------------------------

@dataclass
class Choice:
    """Pick one option.

    `criteria` maps an option key to its description (`{"billing": "payments, invoices"}`), or is a
    plain list of keys. The description is what the model actually compares against, so options must
    describe themselves; the key is only the identifier you get back in `choice`.
    """

    instructions: str = ""
    criteria: Mapping[str, str] | Iterable[str] = field(default_factory=dict)


@dataclass
class NoulCriteria:
    """What "true" and "false" mean for a `Noul`."""

    true: str = ""
    false: str = ""


@dataclass
class Noul:
    """A yes/no question. The answer is the probability of "yes", in [0, 1]."""

    instructions: str = ""
    criteria: NoulCriteria | Mapping[str, str] | None = None


@dataclass
class Score:
    """A graded question. `criteria` are the level descriptions, lowest first."""

    instructions: str = ""
    criteria: Iterable[str] = field(default_factory=list)


Question = Choice | Noul | Score


# --- answers -----------------------------------------------------------------------------------

@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float

    def as_dict(self) -> dict:
        return {"type": "choice", "choice": self.choice, "probabilities": self.probabilities,
                "confidence": self.confidence}


@dataclass
class NoulAnswer:
    noul: float

    def as_dict(self) -> dict:
        return {"type": "noul", "noul": self.noul}


@dataclass
class ScoreAnswer:
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float

    def as_dict(self) -> dict:
        return {"type": "score", "score": self.score, "legend": self.legend,
                "probabilities": self.probabilities, "confidence": self.confidence}


Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer


@dataclass
class Usage:
    """Tokens fed to the local model. Nothing is generated, so `output_tokens` is always 0."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens}


@dataclass
class SystemOneResponse:
    answers: dict[str, Answer]
    model: str
    usage: Usage
    request_id: str

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: v for k, v in self.answers.items() if isinstance(v, ChoiceAnswer)}

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: v for k, v in self.answers.items() if isinstance(v, NoulAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: v for k, v in self.answers.items() if isinstance(v, ScoreAnswer)}

    def as_dict(self) -> dict:
        """The same shape as the Jev HTTP response, ready for `json.dumps`."""
        return {"request_id": self.request_id, "model": self.model, "usage": self.usage.as_dict(),
                "answers": {k: v.as_dict() for k, v in self.answers.items()}}


# --- options, shared by the engine and the tuning heads ------------------------------------------

@dataclass(frozen=True)
class Option:
    """One answerable option: `key` identifies it, `text` is what the model is shown."""

    key: str
    description: str = ""

    @property
    def text(self) -> str:
        return self.description or self.key


NOUL_KEYS = ("true", "false")
NOUL_DEFAULTS = {"true": "Yes.", "false": "No."}


def options_of(question: Question) -> list[Option]:
    """The options of a question, in the order their probabilities are reported."""
    if isinstance(question, Noul):
        c = question.criteria
        if isinstance(c, NoulCriteria):
            given = {"true": c.true, "false": c.false}
        else:
            given = dict(c or {})
            given = {"true": given.get("true", given.get("yes", "")),
                     "false": given.get("false", given.get("no", ""))}
        return [Option(k, given.get(k) or NOUL_DEFAULTS[k]) for k in NOUL_KEYS]
    if isinstance(question, Score):
        levels = list(question.criteria)
        if len(levels) < 2:
            raise ValueError("A Score needs at least two levels, lowest first")
        return [Option(str(i), d) for i, d in enumerate(levels)]
    criteria = question.criteria
    if isinstance(criteria, Mapping):
        options = [Option(str(k), str(v)) for k, v in criteria.items()]
    else:
        options = [Option(str(c)) for c in criteria]
    if len(options) < 2:
        raise ValueError("A Choice needs at least two options")
    return options


def serialize_state(state: Any) -> str:
    """One text per call: strings pass through, anything else becomes JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=1, default=str)
