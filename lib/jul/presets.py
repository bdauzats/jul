"""Tuned settings per model. Adding a model means adding a preset, not touching the code.

Every number here was fitted on the dev datasets (yahootopics, empathetic, massive,
financialphrasebank) with `scripts/dev_fit_tau.py`, never on the Jev benchmark. See docs/JOURNAL.md
for the measurements behind each value.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"

ONE_WORD = 'This text: "{state}" means in one word: "'
QUESTION_OPTIONS = '{instructions}\nPossible answers: {options}.\nText: "{state}"\nIn one word, the answer is: "'


@dataclass(frozen=True)
class Formulation:
    """One prompt formulation, read at one layer. `name` also names its generic-center asset."""

    name: str
    template: str
    layer: int


@dataclass(frozen=True)
class Preset:
    name: str
    repo: str                      # MLX repo
    formulations: tuple[Formulation, ...]
    tau: float
    latency_ms: str
    quality: str
    #: Fallback center when no context supplies task texts. "options" = mean of the option vectors,
    #: "generic" = the asset fitted on varied texts, "none" = no centering.
    center: str = "options"
    notes: str = ""
    torch_repo: str | None = None  # transformers repo; None: no torch backend for this preset

    @property
    def layers(self) -> list[int]:
        return sorted({f.layer for f in self.formulations})

    @property
    def repos(self) -> dict[str, str]:
        return {"mlx": self.repo, **({"torch": self.torch_repo} if self.torch_repo else {})}

    def generic_center(self, formulation: Formulation, backend: str = "mlx") -> np.ndarray | None:
        """The asset fitted with this backend's weights, else the MLX one."""
        names = ([f"{self.name}.{backend}.{formulation.name}.center.npy"] if backend != "mlx" else [])
        for path in [ASSETS / n for n in names + [f"{self.name}.{formulation.name}.center.npy"]]:
            if path.exists():
                return np.load(path)
        return None


PRESETS: dict[str, Preset] = {
    # Layer 39 / 40 and both temperatures fitted on the dev sets (JOURNAL §9 quater; the combination
    # is stable over layers 38-41, 0.565-0.578). Fitting the combination's own tau lowered mean dev
    # ECE from 0.179 to 0.155.
    "minicpm5-2b": Preset(
        name="minicpm5-2b",
        repo="openbmb/MiniCPM5-2B-MLX",
        torch_repo="openbmb/MiniCPM5-2B",
        formulations=(Formulation("one_word", ONE_WORD, 39),
                      Formulation("question_options", QUESTION_OPTIONS, 40)),
        tau=0.04133,
        latency_ms="~65",
        quality="0.617 zero-shot on the Jev bench, above GLiNER (0.583); 0.757 with a tuned head",
        center="generic",
        notes="Centering measured on the dev sets (JOURNAL §9 octies): generic 0.560 > options 0.520 "
              "> none 0.500, and the task center of a Context(examples=...) is best at 0.585. The "
              "asset covers 'one_word'; 'question_options' falls back to the mean of the option "
              "vectors, which measured 0.535 overall.",
    ),
    # JOURNAL §9 quinquies: layers 28 / 31, taus 0.0315 / 0.0483 fitted on the dev sets.
    "qwen3.5-9b": Preset(
        name="qwen3.5-9b",
        repo="mlx-community/Qwen3.5-9B-4bit",
        torch_repo="Qwen/Qwen3.5-9B",
        formulations=(Formulation("one_word", ONE_WORD, 31),
                      Formulation("question_options", QUESTION_OPTIONS, 31)),
        tau=0.04827,
        latency_ms="~220-275",
        quality="0.660 zero-shot on the Jev bench (Jev: 0.753); 0.747 with a context, 0.770 tuned",
        center="options",
        notes="Centering measured on the dev sets (JOURNAL §9 octies): options 0.685 = task 0.685 = "
              "generic 0.675, all within noise, while no centering costs 4.5 points. Options is kept "
              "because it is free. A generic center ships anyway for center='generic'. The Jev bench "
              "was run with a task center: pass a Context with `examples` to reproduce it.",
    ),
}

#: Single-formulation variants, kept because they are what the 'one word' rows of the bench measured.
ONE_WORD_ONLY: dict[str, tuple[int, float]] = {"minicpm5-2b": (39, 0.04554), "qwen3.5-9b": (28, 0.03148)}

ALIASES = {"fast": "minicpm5-2b", "accurate": "qwen3.5-9b"}
DEFAULT_MODEL = "minicpm5-2b"


def resolve(name: str | None) -> Preset:
    name = name or DEFAULT_MODEL
    key = ALIASES.get(name, name)
    if key not in PRESETS:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(PRESETS)} "
                         f"(aliases: {', '.join(ALIASES)})")
    return PRESETS[key]


def one_word_preset(name: str | None = None) -> Preset:
    """The cheaper single-pass variant of a preset: one formulation, its own temperature."""
    p = resolve(name)
    layer, tau = ONE_WORD_ONLY[p.name]
    return dataclasses.replace(p, formulations=(Formulation("one_word", ONE_WORD, layer),), tau=tau)
