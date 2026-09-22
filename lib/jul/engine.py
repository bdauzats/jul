"""The vector method: read one hidden state, compare it to the options, never generate a token.

For each formulation of the preset, the state and every option are encoded by the same prompt; the
answer is the option whose vector is closest (cosine, after subtracting a center). The scores of the
formulations are averaged, then turned into probabilities by `softmax(cosine / tau)`.

Everything that does not depend on the state is computed once and cached: the prompt prefix (as a KV
cache inside `PromptTemplate`), the option vectors, and the center. A call therefore pays for its own
tokens only. The "one word" formulation does not mention the question, so its vector is computed once
per state and shared by every question of the call.

`letters` is the older reading used for `Noul` and `Score`: the options are listed in the prompt and
the answer is read from the logits of the answer tokens. Neither reading has been validated for those
two types yet (see PLAN.md, etape 0).
"""

from __future__ import annotations

import string
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .backbone import MODELS, Backbone, PromptTemplate
from .presets import Formulation, Preset
from .types import Option

LETTERS = string.ascii_uppercase


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def short_names(texts: list[str]) -> list[str]:
    """Drop the words shared by every option at the start and end.

    'This example is about sports' / '... about health' -> 'sports' / 'health'. Keeps the options
    listing short, which matters when there are 72 of them.
    """
    words = [t.split() for t in texts]
    pre = 0
    while all(len(w) > pre + 1 for w in words) and len({w[pre] for w in words}) == 1:
        pre += 1
    suf = 0
    while all(len(w) > pre + suf + 1 for w in words) and len({w[-1 - suf] for w in words}) == 1:
        suf += 1
    return [" ".join(w[pre:len(w) - suf]).strip(" .:") for w in words]


@dataclass
class Pass:
    """One formulation compiled for one question: its prompt, its option vectors and its center."""

    formulation: Formulation
    template: PromptTemplate
    centered_options: np.ndarray  # (K, d), centered and L2-normalized
    center: np.ndarray            # (d,)
    prompt_tokens: int            # tokens of the cached prefix, counted once per call

    def scores(self, vector: np.ndarray) -> np.ndarray:
        return self.centered_options @ normalize(vector - self.center)


@dataclass
class CompiledQuestion:
    options: list[Option]
    passes: list[Pass]
    shared_one_word: bool = False
    letters: "LetterPass | None" = None


@dataclass
class LetterPass:
    template: PromptTemplate
    answer_ids: np.ndarray
    prompt_tokens: int


class Engine:
    """Runs a preset's vector method on a backbone, caching everything that is state-independent."""

    def __init__(self, preset: Preset, backbone: Backbone | None = None, max_cached_questions: int = 64,
                 backend: str | None = None):
        self.preset = preset
        if backbone is None:
            MODELS[preset.name] = preset.repos  # presets are the source of truth for repos
            backbone = Backbone(preset.name, backend)
        self.backbone = backbone
        self.pointer = None
        if preset.method == "pointer":
            from .decision import DecisionSpec, PointerReader
            self.pointer = PointerReader(backbone, DecisionSpec.load(backbone.model_dir))
        self._questions: OrderedDict[tuple, CompiledQuestion] = OrderedDict()
        self._max_cached = max_cached_questions
        self._one_word_templates: dict[str, PromptTemplate] = {}

    # --- prompts ------------------------------------------------------------------------------

    def _template(self, text: str, description: str | None) -> PromptTemplate:
        """`text` contains {state}; the part before it is the cached prefix."""
        if description:
            text = f"Context: {description}\n" + text
        prefix, suffix = text.split("{state}")
        return PromptTemplate(self.backbone, prefix, suffix)

    def _one_word_template(self, formulation: Formulation, description: str | None) -> PromptTemplate:
        key = description or ""
        if key not in self._one_word_templates:
            self._one_word_templates[key] = self._template(formulation.template, description)
        return self._one_word_templates[key]

    def _render(self, formulation: Formulation, instructions: str, options: list[Option]) -> str:
        if "{instructions}" not in formulation.template:
            return formulation.template
        listing = ", ".join(short_names([o.text for o in options]))
        return formulation.template.replace("{instructions}", instructions).replace("{options}", listing)

    # --- vectors ------------------------------------------------------------------------------

    def vector(self, template: PromptTemplate, layer: int, text: str) -> np.ndarray:
        """Hidden state just before the first generated word. The forward stops after `layer`."""
        h, _ = template.run(text, layers=[layer])
        return h[layer][: h[layer].shape[0] // 2]

    def _center(self, template: PromptTemplate, formulation: Formulation, options_matrix: np.ndarray,
                context) -> np.ndarray:
        texts = getattr(context, "examples", None) if context is not None else None
        if texts:
            cached = context.center_for(self.backbone.key, formulation)
            if cached is None:
                cached = np.mean([self.vector(template, formulation.layer, t) for t in texts], axis=0)
                context.set_center(self.backbone.key, formulation, cached)
            return cached
        if self.preset.center == "generic":
            generic = self.preset.generic_center(formulation, self.backbone.backend)
            if generic is not None:
                return generic
        if self.preset.center == "none":
            return np.zeros(options_matrix.shape[-1], dtype=options_matrix.dtype)
        return options_matrix.mean(0)

    def compile(self, kind: str, instructions: str, options: list[Option], context=None) -> CompiledQuestion:
        key = (kind, instructions, tuple((o.key, o.description) for o in options),
               getattr(context, "cache_key", lambda: None)() if context is not None else None)
        if key in self._questions:
            self._questions.move_to_end(key)
            return self._questions[key]

        passes = []
        for f in self.preset.formulations:
            shared = "{instructions}" not in f.template
            template = (self._one_word_template(f, _description(context)) if shared
                        else self._template(self._render(f, instructions, options), _description(context)))
            L = np.stack([self.vector(template, f.layer, o.text) for o in options])
            center = self._center(template, f, L, context)
            passes.append(Pass(f, template, normalize(L - center), center, len(template.prefix_tokens)))

        compiled = CompiledQuestion(options=options, passes=passes,
                                    shared_one_word=any("{instructions}" not in f.template
                                                        for f in self.preset.formulations))
        self._questions[key] = compiled
        if len(self._questions) > self._max_cached:
            self._questions.popitem(last=False)
        return compiled

    # --- reading the answer -------------------------------------------------------------------

    def read(self, compiled: CompiledQuestion, state: str,
             shared: dict[int, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        """(mean cosine score per option, concatenated centered vectors, tokens spent).

        The centered vectors are what a tuned head is trained on, so training and inference read the
        state in exactly the same way. `shared` carries the "one word" vector across the questions of
        a call, since that formulation does not mention the question.
        """
        scores, features, tokens = [], [], 0
        for p in compiled.passes:
            is_shared = "{instructions}" not in p.formulation.template
            if is_shared and shared is not None and p.formulation.layer in shared:
                vector = shared[p.formulation.layer]
            else:
                vector = self.vector(p.template, p.formulation.layer, state)
                tokens += p.prompt_tokens + _state_tokens(p.template, self.backbone, state)
                if is_shared and shared is not None:
                    shared[p.formulation.layer] = vector
            scores.append(p.scores(vector))
            features.append(normalize(vector - p.center))
        return np.mean(scores, axis=0), np.concatenate(features), tokens

    def vector_probabilities(self, compiled: CompiledQuestion, state: str, tau: float | None = None,
                             shared=None) -> tuple[np.ndarray, np.ndarray, int]:
        """(probabilities, raw cosine scores, tokens)."""
        scores, _, tokens = self.read(compiled, state, shared)
        return softmax(scores / (tau or self.preset.tau)), scores, tokens

    # --- the letters reading, for Noul and Score ----------------------------------------------

    def compile_letters(self, kind: str, instructions: str, options: list[Option]) -> LetterPass:
        markers = ([str(i) for i in range(len(options))] if kind == "score" and len(options) <= 10
                   else list(LETTERS[: len(options)]))
        if len(options) > len(markers):
            raise ValueError(f"The letters reading supports at most {len(LETTERS)} options")
        hint = ("Answer with the number of the level only." if kind == "score"
                else "Answer with the letter of the option only.")
        listing = "\n".join(f"{m}) {o.text}" for m, o in zip(markers, options))
        name = "Levels" if kind == "score" else "Options"
        message = f"{instructions}\n\n{name}:\n{listing}\n\n{hint}\n\n" + "Input: {input}"
        template = PromptTemplate.from_user_message(self.backbone, message)
        ids = []
        for m in markers:
            toks = self.backbone.encode(m)
            if len(toks) != 1:
                raise ValueError(f"Answer marker {m!r} is not a single token for {self.backbone.name}")
            ids.append(toks[0])
        if len(set(ids)) != len(ids):
            raise ValueError("Answer markers collide on this tokenizer")
        return LetterPass(template, np.array(ids), len(template.prefix_tokens))

    def letter_logits(self, letters: LetterPass, state: str) -> tuple[np.ndarray, int]:
        _, logits = letters.template.run(state, logits=True)
        tokens = letters.prompt_tokens + _state_tokens(letters.template, self.backbone, state)
        return logits[letters.answer_ids], tokens


def _description(context) -> str | None:
    if context is None:
        return None
    return context.description if getattr(context, "use_description", False) else None


def _state_tokens(template: PromptTemplate, backbone: Backbone, state: str) -> int:
    return len(backbone.encode(state + template.suffix_text))
