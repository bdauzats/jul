"""Context: what the data to sort looks like. Jev has no equivalent.

A context acts at three levels, from the lightest to the strongest:

  description   one sentence, prepended to every formulation as `Context: ...`. It sits in the cached
                prefix, so it costs nothing per call. **Off by default**: measured on both presets and
                four dev datasets, it failed the plan's criterion (JOURNAL §9 decies) -- MiniCPM gained
                1.3 points on average with one dataset losing 4, and Qwen lost on all four, 5.2 on
                average. Turn it on for your own data with `use_description=True`, and measure.
  examples      unlabeled texts of the task. Their mean vector becomes the center, replacing the
                model's generic one. Measured as clearly useful on topics (AG News 0.66 -> 0.75) and
                mildly harmful on fine-grained tasks (Banking77 0.55 -> 0.53).
  labeled       labeled examples. They fit a temperature and a per-option bias, and can train a
                per-task head (see tuning.py).

Compiled contexts are cached on disk under ~/.jul/contexts/<name>/ so the CLI and the library reuse
them without recomputing anything.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .home import JUL_HOME

CONTEXT_HOME = JUL_HOME / "contexts"

#: Measured (JOURNAL §9 decies): 10 examples already capture most of the benefit (+3 points for
#: MiniCPM, +1.7 for Qwen over the preset's own center), 50 is the best of the sizes tried, and 200
#: adds nothing. Below 10 the center is noise -- by hand, 5 examples turned a 0.98 "billing" into a
#: wrong "technical". A warning, not a refusal.
MIN_EXAMPLES_FOR_CENTER = 10


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


@dataclass
class Context:
    """Describes the data a question will be asked about.

    >>> tickets = Context(description="Support tickets of an online bank.",
    ...                   examples=open("tickets.txt").read().splitlines())
    """

    description: str = ""
    examples: list[str] = field(default_factory=list)
    labeled: list = field(default_factory=list)
    name: str | None = None
    #: The description reaches the prompt only when this is on. Off by default: it measured harmful
    #: on average (JOURNAL §9 decies). `examples` are unaffected and stay on.
    use_description: bool = False

    #: (preset, formulation) -> center vector, filled by the engine and persisted.
    centers: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    #: question digest -> (temperature, bias), fitted from `labeled`.
    calibration: dict[str, tuple] = field(default_factory=dict, repr=False)
    #: question digest -> trained head payload, see tuning.py.
    heads: dict[str, dict] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.examples = list(self.examples or [])
        if 0 < len(self.examples) < MIN_EXAMPLES_FOR_CENTER:
            warnings.warn(
                f"Context has {len(self.examples)} examples; a task center needs at least "
                f"{MIN_EXAMPLES_FOR_CENTER} to help, and about 50 works best. With fewer it is mostly "
                f"noise and can make answers worse than no context at all.", stacklevel=2)
        if self.description and not isinstance(self.description, str):
            raise TypeError("description must be a string")

    # --- identity -----------------------------------------------------------------------------

    def cache_key(self) -> str:
        """Identifies the context for the engine's per-question caches."""
        return _digest(self.name or "", self.description if self.use_description else "",
                       str(len(self.examples)), *self.examples[:8])

    # --- centers ------------------------------------------------------------------------------

    @staticmethod
    def _center_key(model, formulation) -> str:
        """`model`: a Preset, or a backbone's `key` (the preset name, tagged with a non-MLX backend)."""
        return f"{getattr(model, 'name', model)}.{formulation.name}.{formulation.layer}"

    def center_for(self, model, formulation) -> np.ndarray | None:
        return self.centers.get(self._center_key(model, formulation))

    def set_center(self, model, formulation, vector: np.ndarray) -> None:
        self.centers[self._center_key(model, formulation)] = vector

    # --- disk cache ---------------------------------------------------------------------------

    @staticmethod
    def path(name: str, home: Path | None = None) -> Path:
        return (home or CONTEXT_HOME) / name

    def save(self, name: str | None = None, home: Path | None = None) -> Path:
        name = name or self.name
        if not name:
            raise ValueError("A context needs a name to be saved")
        self.name = name
        out = self.path(name, home)
        (out / "centers").mkdir(parents=True, exist_ok=True)
        for key, vector in self.centers.items():
            np.save(out / "centers" / f"{key}.npy", vector)
        if self.heads:
            (out / "heads").mkdir(parents=True, exist_ok=True)
            for key, head in self.heads.items():
                np.savez(out / "heads" / f"{key}.npz",
                         **{k: np.asarray(v) for k, v in head.items() if k != "meta" and not k.startswith("_")})
                (out / "heads" / f"{key}.json").write_text(json.dumps(head["meta"], indent=2))
        (out / "meta.json").write_text(json.dumps({
            "name": name,
            "description": self.description,
            "use_description": self.use_description,
            "n_examples": len(self.examples),
            "calibration": {k: [float(t), list(map(float, b))] for k, (t, b) in self.calibration.items()},
        }, indent=2))
        (out / "examples.txt").write_text("\n".join(e.replace("\n", " ") for e in self.examples))
        return out

    @classmethod
    def load(cls, name: str, home: Path | None = None) -> "Context":
        out = cls.path(name, home)
        if not (out / "meta.json").exists():
            raise FileNotFoundError(f"No context named {name!r} under {out.parent}")
        meta = json.loads((out / "meta.json").read_text())
        examples = (out / "examples.txt").read_text().splitlines() if (out / "examples.txt").exists() else []
        ctx = cls(description=meta.get("description", ""), examples=examples, name=name,
                  use_description=meta.get("use_description", False))
        for path in sorted((out / "centers").glob("*.npy")) if (out / "centers").exists() else []:
            ctx.centers[path.stem] = np.load(path)
        for path in sorted((out / "heads").glob("*.npz")) if (out / "heads").exists() else []:
            payload = {k: v for k, v in np.load(path).items()}
            meta_path = path.with_suffix(".json")
            payload["meta"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            ctx.heads[path.stem] = payload
        ctx.calibration = {k: (float(t), np.array(b)) for k, (t, b) in meta.get("calibration", {}).items()}
        return ctx

    @staticmethod
    def list_saved(home: Path | None = None) -> list[str]:
        root = home or CONTEXT_HOME
        return sorted(p.name for p in root.glob("*") if (p / "meta.json").exists()) if root.exists() else []

    @staticmethod
    def delete(name: str, home: Path | None = None) -> bool:
        out = Context.path(name, home)
        if out.exists():
            shutil.rmtree(out)
            return True
        return False


def question_digest(preset_name: str, kind: str, instructions: str, options) -> str:
    """Identifies a question for the heads and calibration stored in a context.

    Includes the preset: a head trained on one model's vectors means nothing for another.
    """
    return _digest(preset_name, kind, instructions, *[f"{o.key}\x1f{o.description}" for o in options])


def resolve_context(context, home: Path | None = None) -> Context | None:
    """Accepts a Context, the name of a saved one, or None."""
    if context is None or isinstance(context, Context):
        return context
    if isinstance(context, str):
        return Context.load(context, home)
    raise TypeError(f"context must be a Context, a saved context name, or None (got {type(context).__name__})")
