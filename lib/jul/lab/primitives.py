"""Jev-style typed questions on a local backbone: Choice, Noul and Score.

Each call is one forward pass. Two ways of reading the answer:

  vector   (default for choice) the input and every option are encoded with the same template
           ('This text: "..." means in one word: "'); the answer is the option whose vector is the
           closest (cosine, after subtracting a center vector). No letters are ever produced.
  letters  (noul, score) the options are listed in the prompt and the answer is read from the logits of
           the answer tokens — letters (A = yes, B = no for noul), level digits for score, numbers
           beyond 26 options. The prompt prefix is cached per question.

The vector settings (layer, template, temperature, generic center) were tuned on BTZSC datasets that
the Jev benchmark does not use (scripts/dev_vectors.py, scripts/dev_two_pass.py). There, the vector
method beat letters clearly, and a second letter pass over the top-k options only made things worse.

Outputs follow TypeSafe's shapes (https://docs.typesafe.ai/primitives):
  choice -> {"choice", "probabilities", "confidence"}
  noul   -> {"noul"}                       probability of "yes"
  score  -> {"score", "legend", "probabilities", "confidence"}

Zero-shot, a small model is overconfident and can lean towards some options (MiniCPM5-2B leans to "no"
on yes/no questions while still ranking inputs well). `calibrate` fixes both from a few labeled examples.

`confidence` is defined so that it can be thresholded:
  choice: probability of the chosen option (calibrated once `calibrate` has been run for the question)
  score:  1 - std / max_std of the level distribution (1 = all mass on one level, 0 = split on both ends)
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..backbone import Backbone, PromptTemplate
from ..calibration import fit_temperature_bias
from .task import LETTERS, Label, Task, answer_token_ids

DEFAULT_MODEL = "minicpm5-2b"
VECTOR_TEMPLATE = 'This text: "{x}" means in one word: "'
VECTOR_LAYER = {"minicpm5-2b": 39}  # other models: ~93% of the depth
VECTOR_TAU = 0.0456  # softmax(cosine / tau), fitted on pooled dev NLL
ASSETS = Path(__file__).resolve().parent.parent / "assets"
NOUL_CRITERIA = {"yes": "", "no": ""}


# --- output formatting, shared with trained heads (decider.Decider.answer) ---

def format_choice(labels: list[str], probs: np.ndarray) -> dict:
    i = int(np.argmax(probs))
    return {"choice": labels[i], "probabilities": _rounded(labels, probs), "confidence": round(float(probs[i]), 4)}


def format_noul(labels: list[str], probs: np.ndarray) -> dict:
    return {"noul": round(float(probs[labels.index("yes")]), 4)}


def format_score(legend: list[str], probs: np.ndarray) -> dict:
    levels = np.arange(len(probs))
    mean = float((levels * probs).sum())
    std = math.sqrt(max(0.0, float((probs * (levels - mean) ** 2).sum())))
    max_std = (len(probs) - 1) / 2
    return {
        "score": round(mean, 4),
        "legend": {str(i): d for i, d in enumerate(legend)},
        "probabilities": _rounded([str(i) for i in levels], probs),
        "confidence": round(1 - std / max_std, 4) if max_std else 1.0,
    }


FORMATTERS = {"choice": format_choice, "noul": format_noul, "score": format_score}


def _rounded(keys, probs) -> dict[str, float]:
    return {k: round(float(p), 4) for k, p in zip(keys, probs)}


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


# --- zero-shot engine ---

class QuickFast:
    """Typed decisions from one forward pass of a local model.

    >>> qf = QuickFast()
    >>> qf.choice("Route this ticket", {"billing": "payments, invoices", "technical": "bugs, errors"},
    ...           "I was charged twice this month")
    {'choice': 'billing', 'probabilities': {...}, 'confidence': 0.93}
    """

    def __init__(self, model: str = DEFAULT_MODEL, backbone: Backbone | None = None, max_cached_questions: int = 64):
        self.backbone = backbone or Backbone(model)
        self._questions: OrderedDict[tuple, tuple[PromptTemplate, np.ndarray]] = OrderedDict()
        self._max_cached = max_cached_questions
        self.calibration: dict[tuple, tuple[float, np.ndarray]] = {}  # question -> (temperature, bias)
        n = self.backbone.n_layers
        self.vector_layer = VECTOR_LAYER.get(self.backbone.name, round(0.93 * n) - 1)
        center = ASSETS / f"{self.backbone.name}.one_word.center.npy"
        self._generic_center = np.load(center) if center.exists() else None  # None: center on the labels
        self._label_vectors: dict[tuple, np.ndarray] = {}
        self._centers: dict[tuple, np.ndarray] = {}  # options -> center from the task's own unlabeled texts

    # Public API -----------------------------------------------------------------------------

    def choice(self, instructions: str, criteria: dict[str, str] | list[str], input, temperature: float | None = None,
               method: str = "vector") -> dict:
        """method="vector" ignores `instructions`: options must describe themselves (name and/or description)."""
        labels = _question_labels("choice", criteria)
        probs = self._probs("choice", instructions, labels, input, temperature, method)
        return format_choice([l.name for l in labels], probs)

    def fit_center(self, criteria: dict[str, str] | list[str], texts: list) -> None:
        """Vector method: center on the task's own unlabeled inputs (~200 is enough) instead of the generic center."""
        labels = _question_labels("choice", criteria)
        self._centers[_options_key(labels)] = np.mean([self._vector(_as_text(t)) for t in texts], axis=0)

    def noul(self, instructions: str, input, criteria: dict[str, str] | None = None, temperature: float | None = None) -> dict:
        """criteria may describe what "yes" and "no" mean: {"yes": "...", "no": "..."}."""
        labels = _question_labels("noul", criteria)
        return format_noul([l.name for l in labels], self._probs("noul", instructions, labels, input, temperature))

    def score(self, instructions: str, criteria: list[str], input, temperature: float | None = None) -> dict:
        """criteria: ordered level descriptions, lowest first."""
        labels = _question_labels("score", criteria)
        return format_score(list(criteria), self._probs("score", instructions, labels, input, temperature))

    def calibrate(self, kind: str, instructions: str, criteria, examples: list[tuple[object, str | int | bool]],
                  method: str | None = None):
        """Fit this question's temperature and per-option bias on labeled (input, answer) pairs.

        Later calls to the same question use them. Returns (temperature, bias).

        Answers: the option name for choice, True/False for noul, the level index for score.
        """
        method = method or _default_method(kind)
        labels = _question_labels(kind, criteria)
        names = [l.name for l in labels]
        logits = np.stack([self._logits(kind, instructions, labels, x, method) for x, _ in examples])
        y = np.array([names.index(_answer_key(kind, a)) for _, a in examples])
        key = _key(kind, instructions, labels, method)
        self.calibration[key] = fit_temperature_bias(logits, y)
        return self.calibration[key]

    # Internals ------------------------------------------------------------------------------

    def _template(self, kind: str, instructions: str, labels: list[Label]):
        key = _key(kind, instructions, labels)
        if key in self._questions:
            self._questions.move_to_end(key)
            return self._questions[key]
        answers, message = _prompt(kind, instructions, labels)
        template = PromptTemplate.from_user_message(self.backbone, message)
        entry = (template, np.array(answer_token_ids(self.backbone, answers)))
        self._questions[key] = entry
        if len(self._questions) > self._max_cached:
            self._questions.popitem(last=False)
        return entry

    def _vector(self, text: str) -> np.ndarray:
        """Last-token hidden state at the vector layer; the forward stops right after that layer."""
        layer = self.vector_layer
        h, _ = self.backbone.forward(self.backbone.encode(VECTOR_TEMPLATE.replace("{x}", text)), layers=[layer])
        return h[layer][: h[layer].shape[0] // 2]

    def _vector_logits(self, labels: list[Label], input) -> np.ndarray:
        key = _options_key(labels)
        if key not in self._label_vectors:
            self._label_vectors[key] = np.stack([self._vector(_label_text(l)) for l in labels])
        L = self._label_vectors[key]
        c = self._centers.get(key, self._generic_center if self._generic_center is not None else L.mean(0))
        x, L = self._vector(_as_text(input)) - c, L - c
        return (L / np.linalg.norm(L, axis=-1, keepdims=True)) @ (x / np.linalg.norm(x)) / VECTOR_TAU

    def _logits(self, kind, instructions, labels, input, method="letters") -> np.ndarray:
        if method == "vector":
            return self._vector_logits(labels, input)
        if len(labels) > 100:
            raise ValueError("At most 100 options per question")
        template, ids = self._template(kind, instructions, labels)
        _, logits = template.run(_as_text(input), logits=True)
        return logits[ids]

    def _probs(self, kind, instructions, labels, input, temperature, method="letters") -> np.ndarray:
        T, bias = self.calibration.get(_key(kind, instructions, labels, method), (1.0, 0.0))
        return _softmax(self._logits(kind, instructions, labels, input, method) / (temperature or T) + bias)


def _prompt(kind: str, instructions: str, labels: list[Label]) -> tuple[list[str], str]:
    """(answer tokens, user message with {input}) for a question."""
    if kind == "score" and len(labels) <= 10:
        answers = [str(i) for i in range(len(labels))]
        hint = "Answer with the number of the level only."
    elif len(labels) > len(LETTERS):
        answers = [str(i + 1) for i in range(len(labels))]  # answer_token_ids checks they are single tokens
        hint = "Answer with the number of the option only."
    else:
        answers = list(LETTERS[: len(labels)])
        hint = "Answer with the letter of the option only."
    task = Task(name=kind, instruction=instructions, labels=labels, input_name="Input", answer_hint=hint,
                options_name="Levels" if kind == "score" else "Options")
    return answers, task.options_message(markers=answers)


def _question_labels(kind: str, criteria) -> list[Label]:
    if kind == "noul":
        return [Label(k, v) for k, v in {**NOUL_CRITERIA, **(criteria or {})}.items()]
    if kind == "score":
        return [Label(str(i), d) for i, d in enumerate(criteria)]
    if isinstance(criteria, dict):
        return [Label(k, v) for k, v in criteria.items()]
    return [Label(c) for c in criteria]


def _key(kind, instructions, labels, method="letters") -> tuple:
    return kind, instructions, _options_key(labels), method


def _options_key(labels) -> tuple:
    return tuple((l.name, l.description) for l in labels)


def _label_text(label: Label) -> str:
    return label.name + (f": {label.description}" if label.description else "")


def _default_method(kind: str) -> str:
    return "vector" if kind == "choice" else "letters"


def _answer_key(kind, answer) -> str:
    if kind == "noul":
        return "yes" if answer else "no"
    return str(answer)


def _as_text(input) -> str:
    return input if isinstance(input, str) else json.dumps(input, ensure_ascii=False, indent=1)
