"""The public API: `TypeSafeClient`, drop-in for the TypeSafe (Jev) Python SDK, plus context and tuning.

    from jul import TypeSafeClient, Choice
    client = TypeSafeClient(model="minicpm5-2b")
    response = client.system_one(state={"ticket": "charged twice"},
                                 questions={"team": Choice(instructions="Which team?",
                                                           criteria={"billing": "...", "tech": "..."})})
    response.choices["team"].choice

Everything runs locally: no API key, no network. The model is loaded on first use and only one is
held in memory at a time, since Qwen3.5-9B alone weighs about 5.5 GB.
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import tuning
from .calibration import fit_temperature_bias
from .backbone import model_key, resolve_backend
from .context import Context, question_digest, resolve_context
from .engine import Engine, softmax
from .presets import Preset, one_word_preset, resolve
from .types import (Choice, ChoiceAnswer, Noul, NoulAnswer, Option, Question, Score, ScoreAnswer,
                    SystemOneResponse, Usage, options_of, serialize_state)

#: How each question type is read by default. Vectors everywhere, measured (JOURNAL §9 septies):
#: on 480 class-balanced yes/no examples, vectors beat letters on accuracy (0.771 vs 0.692), ranking
#: (AUC 0.938 vs 0.904) and calibration (ECE 0.148 vs 0.238); on the ordinal hand set, 7/9 vs 6/9 with
#: letters unable to reach the lowest level at all. `method="letters"` keeps the older reading.
DEFAULT_METHOD = {"choice": "vector", "noul": "vector", "score": "vector"}

_KIND = {Choice: "choice", Noul: "noul", Score: "score"}


class TypeSafeClient:
    """Local, typed decisions. Accepts the Jev SDK's constructor arguments and ignores the remote ones."""

    def __init__(self, model: str | None = None, context: Context | str | None = None,
                 method: str | None = None, one_word_only: bool = False, backend: str | None = None,
                 context_home: Path | None = None, api_key: str | None = None, base_url: str | None = None,
                 timeout: float | None = None, max_retries: int | None = None, **_ignored: Any):
        self._one_word_only = one_word_only
        self._backend = resolve_backend(backend)
        self._preset = self._resolve_preset(model)
        self._engine: Engine | None = None
        self._context_home = context_home
        self.context = resolve_context(context, context_home)
        self.method = method

    # --- model handling -----------------------------------------------------------------------

    def _resolve_preset(self, model: str | None) -> Preset:
        return (one_word_preset(model, self._backend) if self._one_word_only
                else resolve(model, self._backend))

    def _engine_for(self, model: str | None) -> Engine:
        preset = self._resolve_preset(model) if model else self._preset
        if self._engine is None or self._engine.preset.name != preset.name:
            self._engine = None  # drop the previous model before loading another
            self._engine = Engine(preset, backend=self._backend)
        self._engine.preset = preset
        self._preset = preset
        return self._engine

    @property
    def model(self) -> str:
        return self._preset.name

    def close(self) -> None:
        """Release the model. The Jev SDK closes an HTTP session here."""
        self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- the call -----------------------------------------------------------------------------

    def system_one(self, state: Any, questions: Mapping[str, Question], context: Context | str | None = None,
                   model: str | None = None, method: str | None = None, **_ignored: Any) -> SystemOneResponse:
        """Answer every question about one state, in a single pass per formulation.

        `_ignored` swallows the Jev arguments that mean nothing locally (`response_model`, `retry`,
        `extra_body`, ...) so existing code keeps running.
        """
        if not questions:
            raise ValueError("system_one needs at least one question")
        ctx = resolve_context(context, self._context_home) if context is not None else self.context
        engine = self._engine_for(model)
        text = serialize_state(state)
        shared: dict[int, np.ndarray] = {}
        answers, tokens = {}, 0

        for name, question in questions.items():
            kind = _kind_of(question)
            how = method or self.method or DEFAULT_METHOD[kind]
            options = options_of(question)
            probabilities, spent = self._answer_probabilities(engine, kind, how, question, options, text,
                                                              ctx, shared)
            tokens += spent
            answers[name] = _format(kind, question, options, probabilities)

        return SystemOneResponse(answers=answers, model=self._preset.name, usage=Usage(input_tokens=tokens),
                                 request_id=str(uuid.uuid4()))

    def _answer_probabilities(self, engine: Engine, kind: str, how: str, question: Question,
                              options: list[Option], text: str, ctx: Context | None,
                              shared: dict) -> tuple[np.ndarray, int]:
        # A tuned head is trained on vector features, so it pins the reading to vectors whatever
        # `method` says; otherwise a head trained by `autotune` would be silently ignored.
        head = self._head(ctx, kind, question, options)
        if how == "letters" and head is None:
            letters = engine.compile_letters(kind, question.instructions, options)
            logits, tokens = engine.letter_logits(letters, text)
            return self._calibrated(ctx, kind, question, options, logits), tokens

        compiled = engine.compile(kind, question.instructions, options, ctx)
        scores, features, tokens = engine.read(compiled, text, shared)
        if head is not None:
            return tuning.apply(head, features), tokens
        return self._calibrated(ctx, kind, question, options, scores / self._preset.tau), tokens

    def _digest(self, kind: str, question: Question, options: list[Option]) -> str:
        return question_digest(model_key(self._preset.name, self._backend), kind, question.instructions, options)

    def _head(self, ctx: Context | None, kind, question, options) -> dict | None:
        return ctx.heads.get(self._digest(kind, question, options)) if ctx else None

    def _calibrated(self, ctx: Context | None, kind, question, options, logits: np.ndarray) -> np.ndarray:
        if ctx:
            fitted = ctx.calibration.get(self._digest(kind, question, options))
            if fitted:
                temperature, bias = fitted
                return softmax(logits / temperature + np.asarray(bias))
        return softmax(logits)

    # --- tuning -------------------------------------------------------------------------------

    def autotune(self, context: Context | str, questions: Mapping[str, Question], labeled: list,
             model: str | None = None, save: bool = True) -> dict[str, tuning.TuningReport]:
        """Fit a per-task head (and a calibration) from labeled examples, and store it in a context.

        `labeled` is a list of `(state, {question_name: answer})`. An answer is the option key for a
        Choice, True/False for a Noul, the level index for a Score. Returns one report per question;
        a head that does not beat the zero-shot method on held-out examples is not activated.
        """
        ctx = resolve_context(context, self._context_home)
        if ctx is None:
            raise ValueError("autotune needs a Context or the name of one")
        if isinstance(context, str) and ctx.name is None:
            ctx.name = context
        engine = self._engine_for(model)
        states = [serialize_state(s) for s, _ in labeled]
        reports: dict[str, tuning.TuningReport] = {}

        for name, question in questions.items():
            kind = _kind_of(question)
            options = options_of(question)
            keys = [o.key for o in options]
            index = {k: i for i, k in enumerate(keys)}
            rows = [(i, _answer_index(kind, a[name], index)) for i, (_, a) in enumerate(labeled) if name in a]
            if len(rows) < 2:
                raise ValueError(f"question {name!r} has fewer than 2 labeled examples")

            compiled = engine.compile(kind, question.instructions, options, ctx)
            scores, features = [], []
            for i, _ in rows:
                s, f, _ = engine.read(compiled, states[i], None)
                scores.append(s)
                features.append(f)
            scores, features = np.stack(scores), np.stack(features)
            y = np.array([label for _, label in rows])

            digest = self._digest(kind, question, options)
            head, report = tuning.train(features, y, scores, keys, name, self._preset.name)
            ctx.calibration[digest] = fit_temperature_bias(scores / self._preset.tau, y)
            if head is not None:
                ctx.heads[digest] = head
            else:
                ctx.heads.pop(digest, None)
            reports[name] = report

        if save and ctx.name:
            ctx.save(home=self._context_home)
        return reports


class AsyncTypeSafeClient(TypeSafeClient):
    """The same API, awaitable. MLX work runs in a worker thread so the event loop keeps turning."""

    async def system_one(self, *args: Any, **kwargs: Any) -> SystemOneResponse:  # type: ignore[override]
        import asyncio
        return await asyncio.to_thread(TypeSafeClient.system_one, self, *args, **kwargs)

    async def autotune(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        import asyncio
        return await asyncio.to_thread(TypeSafeClient.autotune, self, *args, **kwargs)

    async def close(self) -> None:  # type: ignore[override]
        TypeSafeClient.close(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


# --- helpers ---------------------------------------------------------------------------------

def _kind_of(question: Question) -> str:
    kind = _KIND.get(type(question))
    if kind is None:
        raise TypeError(f"Questions must be Choice, Noul or Score (got {type(question).__name__})")
    return kind


def _answer_index(kind: str, answer: Any, index: Mapping[str, int]) -> int:
    if kind == "noul":
        if isinstance(answer, str):
            answer = answer.strip().lower() in {"true", "yes", "1"}
        return index["true"] if answer else index["false"]
    key = str(answer)
    if key not in index:
        raise ValueError(f"Answer {answer!r} is not one of the options {list(index)}")
    return index[key]


def _format(kind: str, question: Question, options: list[Option], probabilities: np.ndarray):
    keys = [o.key for o in options]
    probs = {k: round(float(p), 4) for k, p in zip(keys, probabilities)}
    if kind == "noul":
        return NoulAnswer(noul=probs["true"])
    if kind == "score":
        levels = np.arange(len(options))
        mean = float((levels * probabilities).sum())
        std = math.sqrt(max(0.0, float((probabilities * (levels - mean) ** 2).sum())))
        max_std = (len(options) - 1) / 2
        return ScoreAnswer(score=round(mean, 4),
                           legend={o.key: o.description for o in options},
                           probabilities=probs,
                           confidence=round(1 - std / max_std, 4) if max_std else 1.0)
    best = int(np.argmax(probabilities))
    return ChoiceAnswer(choice=keys[best], probabilities=probs,
                        confidence=round(float(probabilities[best]), 4))
