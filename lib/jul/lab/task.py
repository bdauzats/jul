"""Task definition (instruction + typed labels) and the prompts built from it."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from pathlib import Path

from ..backbone import Backbone, PromptTemplate

LETTERS = string.ascii_uppercase


@dataclass
class Label:
    name: str
    description: str = ""


@dataclass
class Task:
    name: str
    instruction: str
    labels: list[Label]
    input_name: str = "Text"
    options_name: str = "Options"
    answer_hint: str = "Answer with the letter of the option only."
    label_prompt: str = "Category: {name}\n{description}"
    type: str = "choice"  # choice | noul | score (score: labels ordered lowest first)
    data_dir: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Task":
        d = json.loads(Path(path).read_text())
        d["labels"] = [Label(**l) if isinstance(l, dict) else Label(l) for l in d["labels"]]
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known}, extra={k: v for k, v in d.items() if k not in known})

    @property
    def label_names(self) -> list[str]:
        return [l.name for l in self.labels]

    def plain_message(self) -> str:
        return f"{self.instruction}\n\n{self.input_name}: {{input}}"

    def options_message(self, labels: list[Label] | None = None, markers: list[str] | None = None) -> str:
        """Options listed as `marker) name — description`; the answer is read on the marker tokens."""
        labels = labels or self.labels
        markers = markers or list(LETTERS[: len(labels)])
        opts = "\n".join(f"{m}) {_option_text(m, l)}" for m, l in zip(markers, labels))
        return f"{self.instruction}\n\n{self.options_name}:\n{opts}\n\n{self.answer_hint}\n\n{self.input_name}: {{input}}"

    def label_message(self, label: Label) -> str:
        return self.label_prompt.format(name=label.name, description=label.description).strip()

    def load_split(self, split: str, data_dir: str | Path | None = None) -> tuple[list[str], list[int]]:
        path = Path(data_dir or self.data_dir) / f"{split}.jsonl"
        index = {n: i for i, n in enumerate(self.label_names)}
        texts, ys = [], []
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                texts.append(row["text"])
                ys.append(index[row["label"]])
        return texts, ys


class Prompts:
    """The three prompts a task needs on a given backbone."""

    def __init__(self, backbone: Backbone, task: Task, use_prefix_cache: bool = True):
        self.backbone = backbone
        self.task = task
        self.plain = PromptTemplate.from_user_message(backbone, task.plain_message(), use_prefix_cache=use_prefix_cache)
        self.options = None
        self.letter_ids = None
        if len(task.labels) <= len(LETTERS):
            self.options = PromptTemplate.from_user_message(
                backbone, task.options_message(), use_prefix_cache=use_prefix_cache
            )
            self.letter_ids = letter_token_ids(backbone, len(task.labels))

    def label_features(self, labels: list[Label], layers: list[int]):
        """(K, n_layers, d) features of each label, read at the assistant-turn position."""
        import mlx.core as mx

        out = []
        for label in labels:
            t = PromptTemplate.from_user_message(self.backbone, "{input}", use_prefix_cache=False)
            h, _ = t.run(self.task.label_message(label), layers=layers)
            out.append(mx.stack([h[l] for l in layers]))
        return mx.stack(out)


def _option_text(marker: str, label: Label) -> str:
    if label.name == marker:  # e.g. score levels, whose name is their number
        return label.description
    return label.name + (f" — {label.description}" if label.description else "")


def answer_token_ids(backbone: Backbone, answers: list[str]) -> list[int]:
    """Token id of each possible answer; each must be a single, distinct token."""
    ids = []
    for a in answers:
        toks = backbone.encode(a)
        if len(toks) != 1:
            raise ValueError(f"Answer {a!r} is not a single token for {backbone.name}: {toks}")
        ids.append(toks[0])
    if len(set(ids)) != len(ids):
        raise ValueError("Answer tokens collide")
    return ids


def letter_token_ids(backbone: Backbone, k: int) -> list[int]:
    return answer_token_ids(backbone, list(LETTERS[:k]))
