"""Synthetic labeled data, written by an LLM from a few real examples, for a later `autotune(...)`.

This module only writes a dataset. It never tunes and never judges one: the texts come out cleaner and
more typical than real ones, so keep real labeled examples aside to check what a head trained on them
is worth.

Two steps:
  1. understand: from the seed texts, the writer describes the source once (who writes, form, length,
     language, tone). Every generation prompt reuses that description.
  2. generate: for each question and each option, the writer produces new texts in batches, shown the
     full option list (to aim at what sets the option apart), the seeds labeled with that option and
     the texts it already wrote (to keep varying). Duplicates and copies of seeds are dropped.

The writer is any `str -> str` function. `mlx_writer(repo)` wraps a local generative model through
mlx-lm; there is no default, since jul's own presets read models rather than make them write.
Measured: the writer's quality decides everything (MiniCPM5-2B described texts instead of writing them;
a 9B instruct model wrote realistic ones).
"""

from __future__ import annotations

import random
import re
import sys
from typing import Callable, Mapping

from .types import Question, options_of, serialize_state

Writer = Callable[[str], str]

UNDERSTAND_PROMPT = """Here are real texts from a data source:

{examples}

Describe this source in 3 to 5 short lines: who writes these texts, to whom, in what form, typical length, language, register and tone, recurring quirks (typos, abbreviations, formatting). Describe the style only, not the topics. No preamble."""

GENERATE_PROMPT = """{instructions}
The possible answers are:
{listing}

{style}Write {k} new, different texts for which the right answer is clearly "{key}" and none of the others. Write the texts themselves, exactly as their author would write them, never a description of a text. Vary the situation, wording, length and tone.
{seeds}{avoid}One text per line, no numbering, no quotes, nothing else."""


def synthesize(questions: Mapping[str, Question], seeds: list[tuple[str, dict]], per_option: int,
               write: Writer, batch: int = 8, max_seeds: int = 5, seed: int = 0,
               on_option: Callable[[list[dict]], None] | None = None) -> list[dict]:
    """Write `per_option` texts for every option of every question.

    seeds: `(state, answers)` pairs; `answers` may be empty (the text then only informs the style).
    Returns rows `{"state", "answers": {question: option key}, "synthetic": True}`, the format read by
    `jul autotune --labeled`. `on_option` receives each option's rows as soon as they are written.
    """
    rng = random.Random(seed)
    texts = [serialize_state(s) for s, _ in seeds]
    style = describe_source(texts, write, rng) if texts else ""
    seen = {_norm(t) for t in texts}
    rows: list[dict] = []

    for name, question in questions.items():
        options = options_of(question)
        listing = "\n".join(f"- {o.key}: {o.description}" if o.description else f"- {o.key}"
                            for o in options)
        for option in options:
            examples = [t for t, (_, a) in zip(texts, seeds) if name in a and _key(a[name]) == option.key]
            written: list[str] = []
            for _ in range(3 * (per_option // batch + 1)):          # bounded: stop if it keeps repeating
                if len(written) >= per_option:
                    break
                prompt = GENERATE_PROMPT.format(
                    instructions=question.instructions or "Choose the answer that fits the text.",
                    listing=listing, key=option.key, k=min(batch, per_option - len(written)),
                    style=f"The texts come from this source:\n{style}\n\n" if style else "",
                    seeds=_block("Real examples with this answer:", rng.sample(examples, min(max_seeds, len(examples)))),
                    avoid=_block("Already written, do not repeat them:", written[-batch:]))
                for text in parse_lines(write(prompt)):
                    if _norm(text) not in seen and len(written) < per_option:
                        seen.add(_norm(text))
                        written.append(text)
            if len(written) < per_option:
                print(f"  {name}={option.key}: only {len(written)} distinct texts", file=sys.stderr)
            new = [{"state": t, "answers": {name: option.key}, "synthetic": True} for t in written]
            rows += new
            if on_option:
                on_option(new)
    return rows


def describe_source(texts: list[str], write: Writer, rng: random.Random, n: int = 12) -> str:
    sample = rng.sample(texts, min(n, len(texts)))
    return write(UNDERSTAND_PROMPT.format(examples="\n".join(f"- {t}" for t in sample))).strip()


def parse_lines(output: str) -> list[str]:
    """One text per line; drops numbering, bullets, quotes and lines too short to be a text."""
    out = []
    for line in output.split("</think>")[-1].splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip().strip("\"'“”").strip()
        if len(line) > 8:
            out.append(line)
    return out


def mlx_writer(model: str, max_tokens: int = 1024, temperature: float = 0.9) -> Writer:
    """A local writer through mlx-lm. Sampling, not greedy: batches must not repeat each other."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    from .presets import resolve

    try:
        repo = resolve(model).repo or model
    except ValueError:
        repo = model                        # any mlx-lm repo
    lm, tok = load(repo)
    sampler = make_sampler(temp=temperature, top_p=0.95)

    def write(prompt: str) -> str:
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        return generate(lm, tok, chat, max_tokens=max_tokens, sampler=sampler, verbose=False)
    return write


def _block(title: str, items: list[str]) -> str:
    return f"{title}\n" + "\n".join(f"- {t}" for t in items) + "\n\n" if items else ""


def _key(answer) -> str:
    if isinstance(answer, bool):
        return "true" if answer else "false"
    return str(answer).strip().lower() if str(answer).strip().lower() in {"true", "false"} else str(answer)


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()
