"""Fit a preset for any model, on any backend, in one command: `jul models add <name> --repo <repo>`.

The same protocol that produced the built-in presets (docs/JOURNAL.md §9 quater to §9 octies), run
end to end. Every choice is made on the dev datasets (yahootopics, empathetic, massive,
financialphrasebank), never on the Jev benchmark, which stays a separate measurement.

1. Checks: the model loads, the prefix cache leaves the vectors unchanged, a call does not disturb
   the next one, and the letters reading (Noul, Score) has single-token markers.
2. Extraction: one pass per text reads every candidate layer (the upper half of the model), for the
   two formulations, on the dev texts, their labels, their unlabeled texts, and the generic texts.
3. Choice, offline from those vectors: the layers and the center by dev accuracy, averaged over the
   neighbouring layers (the maximum of hundreds of combinations would be optimistic), then tau by
   pooled NLL with the task center, as `scripts/dev_fit_tau.py` does.
4. Output: `<name>@<backend>.json` and its generic center in ~/.jul/presets.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .backbone import MODELS, Backbone, PromptTemplate, resolve_backend
from .calibration import metrics
from .engine import LETTERS, short_names
from .presets import (ONE_WORD, PRESET_HOME, PRESETS, QUESTION_OPTIONS, Formulation, Preset,
                      center_asset_name, save_preset)

DEV = ("yahootopics", "empathetic", "massive", "financialphrasebank")
GENERIC = ("amazonpolarity", "appreviews", "biasframes_intent", "biasframes_offensive", "biasframes_sex",
           "capsotu", "imdb", "manifesto", "rottentomatoes", "trueteacher", "wikitoxic_insult",
           "wikitoxic_obscene", "wikitoxic_threat", "wikitoxic_toxicaggregated", "yelpreviews")
QUESTION = "Which single label best describes the input text?"
DATA_HOME = Path.home() / ".jul" / "calibration-data"
#: The BTZSC revision pinned by the Jev benchmark protocol.
BTZSC = ("btzsc/btzsc", "fef2a2ac62b69c58670047dddf045c53d7c3cb5e")
#: Candidate layers, as fractions of the depth. Every measured optimum sat in the upper half.
LAYER_RANGE = (0.5, 1.0)
#: Center strategies a preset can use without task data, in order of preference on a tie.
CENTERS = ("options", "generic", "none")
TAU_GRID = np.exp(np.linspace(np.log(0.003), np.log(1.0), 300))


class CalibrationError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(msg, flush=True)


# --- data -----------------------------------------------------------------------------------------

@dataclass
class DevSet:
    name: str
    texts: list[str]
    y: np.ndarray
    labels: list[str]
    unlabeled: list[str]


def load_data(root: Path, n_dev: int, n_generic: int) -> tuple[list[DevSet], list[str]]:
    """`root/dev/<set>.jsonl` (+ `.unlabeled.jsonl`) and `root/generic/*.jsonl`, as prepared by
    `scripts/prepare_dev.py` / `prepare_generic.py` or by `fetch_data`."""
    dev = []
    for name in DEV:
        rows = [json.loads(line) for line in open(root / "dev" / f"{name}.jsonl")][:n_dev]
        unlabeled = [json.loads(line)["text"] for line in open(root / "dev" / f"{name}.unlabeled.jsonl")][:n_dev]
        dev.append(DevSet(name, [r["text"] for r in rows], np.array([r["target_index"] for r in rows]),
                          rows[0]["labels"], unlabeled))
    files = sorted((root / "generic").glob("*.jsonl"))
    per_file = max(1, n_generic // len(files))
    generic = [json.loads(line)["text"] for f in files for line in list(open(f))[:per_file]][:n_generic]
    return dev, generic


def data_ready(root: Path) -> bool:
    return (all((root / "dev" / f"{n}.jsonl").exists() for n in DEV)
            and any((root / "generic").glob("*.jsonl")))


def fetch_data(root: Path = DATA_HOME, n_dev: int = 200, n_generic: int = 20, seed: int = 7) -> Path:
    """Download the calibration sets from BTZSC (needs `datasets`: pip install 'jul[calibrate]').

    Balanced over the classes, like the prepare_* scripts; the rows themselves differ from theirs.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise CalibrationError("Fetching the calibration data needs `datasets`: "
                               "pip install 'jul[calibrate]', or pass --data <dir>") from exc
    repo, revision = BTZSC

    def balanced(name: str, n: int) -> tuple[list[dict], list[str]]:
        rows = load_dataset(repo, name=name, split="test", revision=revision)
        texts, flags = [str(t) for t in rows["text"]], [int(v) for v in rows["labels"]]
        k = next((i for i, t in enumerate(texts) if t != texts[0]), len(texts))
        labels = [str(rows[i]["hypothesis"]) for i in range(k)]
        by_class = defaultdict(list)
        for s in range(len(texts) // k):
            values = flags[s * k:(s + 1) * k]
            if sum(values) == 1:
                by_class[values.index(1)].append(s)
        rng = random.Random(seed)
        queues = [rng.sample(v, len(v)) for _, v in sorted(by_class.items())]
        picked = []
        while any(queues) and len(picked) < n:
            for q in queues:
                if q and len(picked) < n:
                    s = q.pop()
                    picked.append({"text": texts[s * k], "target_index": flags[s * k:(s + 1) * k].index(1),
                                   "labels": labels})
        return picked, labels

    (root / "dev").mkdir(parents=True, exist_ok=True)
    (root / "generic").mkdir(parents=True, exist_ok=True)
    for name in DEV:
        _log(f"  fetching {name}")
        picked, _ = balanced(name, 2 * n_dev)
        labeled, unlabeled = picked[:n_dev], picked[n_dev:]
        with open(root / "dev" / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in labeled)
        with open(root / "dev" / f"{name}.unlabeled.jsonl", "w") as f:
            f.writelines(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n" for r in unlabeled)
    for name in GENERIC:
        _log(f"  fetching {name}")
        picked, _ = balanced(name, n_generic)
        with open(root / "generic" / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in picked)
    return root


# --- extraction -----------------------------------------------------------------------------------

def _template(backbone: Backbone, text: str, **kw) -> PromptTemplate:
    prefix, suffix = text.split("{state}")
    return PromptTemplate(backbone, prefix, suffix, **kw)


def _vectors(template: PromptTemplate, layers: list[int], texts: list[str]) -> dict[int, np.ndarray]:
    """{layer: (n, d)} last-token vectors, all layers read in one pass per text."""
    out = {l: [] for l in layers}
    for text in texts:
        h, _ = template.run(text, layers=layers)
        for l in layers:
            out[l].append(h[l][: h[l].shape[0] // 2])
    return {l: np.stack(v) for l, v in out.items()}


def _question_options(labels: list[str]) -> str:
    return QUESTION_OPTIONS.replace("{instructions}", QUESTION).replace("{options}", ", ".join(short_names(labels)))


def check(backbone: Backbone, layer: int) -> list[str]:
    """Raises when the vector method cannot run on this model; returns warnings otherwise."""
    texts = ["I was charged twice for my subscription", "the app crashes on export", "do you offer annual plans?"]
    cached = _vectors(_template(backbone, ONE_WORD), [layer], texts)[layer]
    plain = _vectors(_template(backbone, ONE_WORD, use_prefix_cache=False), [layer], texts)[layer]
    cos = min(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for a, b in zip(cached, plain))
    if cos < 0.999:
        raise CalibrationError(f"The prefix cache changes the vectors (cosine {cos:.4f} < 0.999): "
                               f"this architecture is not supported by the {backbone.backend} backend yet")
    template = _template(backbone, ONE_WORD)
    first = _vectors(template, [layer], texts[:1])[layer]
    _vectors(template, [layer], ["a much longer sentence about invoices, refunds and the billing team"])
    if not np.allclose(first, _vectors(template, [layer], texts[:1])[layer], atol=1e-3):
        raise CalibrationError("A call changes the next one: the prefix cache is not restored")

    warnings = []
    if not getattr(backbone.tokenizer, "chat_template", None):
        warnings.append("no chat template: the letters reading (Noul, Score) will not work")
    for markers, what in ((list(LETTERS), "letters"), ([str(i) for i in range(10)], "digits")):
        ids = [backbone.encode(m) for m in markers]
        if any(len(t) != 1 for t in ids) or len({t[0] for t in ids}) != len(ids):
            warnings.append(f"the {what} are not distinct single tokens: the letters reading may fail")
    return warnings


# --- choice ---------------------------------------------------------------------------------------

def _nrm(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _cosines(X: np.ndarray, L: np.ndarray, center: np.ndarray) -> np.ndarray:
    return _nrm(X - center) @ _nrm(L - center).T


def smooth(grid: np.ndarray) -> np.ndarray:
    """Mean over each cell and its neighbours (1D or 2D): favours a stable plateau over a lucky peak."""
    padded = np.pad(grid.astype(float), 1, constant_values=np.nan)
    windows = np.stack([np.roll(np.roll(padded, di, 0), dj, 1) if grid.ndim == 2 else np.roll(padded, di)
                        for di in (-1, 0, 1) for dj in ((-1, 0, 1) if grid.ndim == 2 else (0,))])
    out = np.nanmean(windows, axis=0)
    return out[1:-1, 1:-1] if grid.ndim == 2 else out[1:-1]


def fit_tau(scores: list[np.ndarray], ys: list[np.ndarray]) -> float:
    def nll(tau):
        total = 0.0
        for s, y in zip(scores, ys):
            z = s / tau
            z = z - z.max(1, keepdims=True)
            total -= float((z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).sum())
        return total
    return float(min(TAU_GRID, key=nll))


def _center_of(strategy: str, pack: dict, generic: np.ndarray | None) -> np.ndarray:
    """The center a preset uses without task data; 'generic' only exists for 'one word'."""
    if strategy == "none":
        return np.zeros(pack["L"].shape[-1], dtype=pack["L"].dtype)
    if strategy == "generic" and generic is not None:
        return generic
    return pack["L"].mean(0)


def _scores(pack: dict, key: str, l: int, center: str, generic: dict[int, np.ndarray]) -> np.ndarray:
    return _cosines(pack[key]["X"][l], pack[key]["L"][l],
                    _center_of(center, {"L": pack[key]["L"][l]}, generic[l] if key == "ow" else None))


def choose(packs: list[dict], generic: dict[int, np.ndarray], layers: list[int]) -> dict:
    """Pick the layers, the center and tau from the extracted vectors.

    `packs`: per dev set, {"y", "ow": {"X", "L", "U"}, "qo": {...}}, each array {layer: (n, d)}.

    Each center gets its best layers (smoothed accuracy). Centering has helped every model measured
    (JOURNAL §9 octies), so "none" is only kept when it beats the best center by more than one
    standard error: with hundreds of combinations searched, a smaller lead is noise. tau is fitted
    with the chosen center, since that is what a call without task data computes.
    """
    n = len(layers)
    ys = [p["y"] for p in packs]
    n_total = sum(len(y) for y in ys)
    acc = {c: np.zeros((n, n)) for c in CENTERS}      # combination, [one_word layer, qo layer]
    acc_ow = {c: np.zeros(n) for c in CENTERS}        # one word alone
    for pack in packs:
        y = pack["y"]
        for c in CENTERS:
            ow = [_scores(pack, "ow", l, c, generic) for l in layers]
            qo = [_scores(pack, "qo", l, c, generic) for l in layers]
            for i in range(n):
                acc_ow[c][i] += (ow[i].argmax(1) == y).mean() / len(packs)
                for j in range(n):
                    acc[c][i, j] += ((ow[i] + qo[j]).argmax(1) == y).mean() / len(packs)

    best = {c: float(smooth(acc[c]).max()) for c in CENTERS}
    centered = max((c for c in CENTERS if c != "none"), key=lambda c: (best[c], -CENTERS.index(c)))
    stderr = np.sqrt(best[centered] * (1 - best[centered]) / n_total)
    center = "none" if best["none"] > best[centered] + stderr else centered
    i, j = np.unravel_index(np.argmax(smooth(acc[center])), (n, n))
    k = int(np.argmax(smooth(acc_ow[center])))
    l_ow, l_qo, l_one = layers[i], layers[j], layers[k]

    combo = [(_scores(p, "ow", l_ow, center, generic) + _scores(p, "qo", l_qo, center, generic)) / 2
             for p in packs]
    one = [_scores(p, "ow", l_one, center, generic) for p in packs]
    tau, tau_one = fit_tau(combo, ys), fit_tau(one, ys)

    def task(pack, key, l):
        return _cosines(pack[key]["X"][l], pack[key]["L"][l], pack[key]["U"][l].mean(0))

    combo_task = [(task(p, "ow", l_ow) + task(p, "qo", l_qo)) / 2 for p in packs]
    accuracy = float(np.mean([(s.argmax(1) == y).mean() for s, y in zip(combo, ys)]))
    return {
        "layers": {"one_word": l_ow, "question_options": l_qo, "one_word_only": l_one},
        "center": center, "tau": tau, "tau_one_word": tau_one,
        "tau_task_center": fit_tau(combo_task, ys),
        "dev_accuracy": accuracy,
        "dev_accuracy_stderr": float(np.sqrt(accuracy * (1 - accuracy) / n_total)),
        "dev_accuracy_task_center": float(np.mean([(s.argmax(1) == y).mean() for s, y in zip(combo_task, ys)])),
        "dev_ece": float(np.mean([metrics(s / tau, y)["ece"] for s, y in zip(combo, ys)])),
        "dev_accuracy_by_center": {c: round(best[c], 4) for c in CENTERS},
        "dev_accuracy_by_set": {p["name"]: float((s.argmax(1) == p["y"]).mean()) for p, s in zip(packs, combo)},
        "n_dev": n_total,
    }


def extract(backbone: Backbone, dev: list[DevSet], generic_texts: list[str],
            layers: list[int]) -> tuple[list[dict], dict[int, np.ndarray]]:
    """(packs for `choose`, generic "one word" center per layer)."""
    t0 = time.perf_counter()
    one_word = _template(backbone, ONE_WORD)
    generic = _vectors(one_word, layers, generic_texts)
    generic_center = {l: generic[l].mean(0) for l in layers}
    packs = []
    for d in dev:
        qo = _template(backbone, _question_options(d.labels))
        packs.append({"name": d.name, "y": d.y, **{
            key: {"X": _vectors(t, layers, d.texts), "L": _vectors(t, layers, d.labels),
                  "U": _vectors(t, layers, d.unlabeled)}
            for key, t in (("ow", one_word), ("qo", qo))}})
        _log(f"  {d.name}: done ({time.perf_counter() - t0:.0f}s)")
    return packs, generic_center


# --- the command ----------------------------------------------------------------------------------

def calibrate(name: str, repo: str | None = None, backend: str | None = None, data: Path | None = None,
              n_dev: int = 50, n_generic: int = 200, home: Path | None = None) -> Preset:
    backend = resolve_backend(backend)
    if repo is None:
        known = PRESETS[name].repos if name in PRESETS else MODELS.get(name, {})
        if backend not in known:
            raise CalibrationError(f"No known {backend} repo for {name!r}: pass --repo")
        repo = known[backend]
    MODELS[name] = {**MODELS.get(name, {}), backend: repo}
    t0 = time.perf_counter()

    root = Path(data) if data else DATA_HOME
    if not data_ready(root):
        if data:
            raise CalibrationError(f"No calibration data in {root} (expects dev/ and generic/)")
        _log(f"Fetching the calibration data into {root} (once)")
        fetch_data(root)
    dev, generic_texts = load_data(root, n_dev, n_generic)

    _log(f"Loading {repo} ({backend})")
    backbone = Backbone(name, backend)
    lo, hi = LAYER_RANGE
    layers = list(range(max(0, round(lo * backbone.n_layers) - 1), backbone.n_layers))
    _log(f"  {backbone.n_layers} layers, candidates {layers[0]}-{layers[-1]}")

    _log("1/4 checks")
    warnings = check(backbone, layers[-1])
    for w in warnings:
        _log(f"  warning: {w}")

    _log(f"2/4 extraction: {len(dev)} dev sets x {n_dev} examples, {len(generic_texts)} generic texts")
    packs, generic_center = extract(backbone, dev, generic_texts, layers)

    _log("3/4 choice")
    report = choose(packs, generic_center, layers)
    lay = report["layers"]
    _log(f"  layers {lay['one_word']} / {lay['question_options']}, center {report['center']}, "
         f"tau {report['tau']:.4f}; dev accuracy {report['dev_accuracy']:.3f} "
         f"± {report['dev_accuracy_stderr']:.3f}, ECE {report['dev_ece']:.3f}")

    _log("4/4 latency")
    latency = _latency(backbone, dev, lay["one_word"], lay["question_options"])

    report.update({"backend": backend, "repo": repo, "n_layers": backbone.n_layers,
                   "candidate_layers": [layers[0], layers[-1]], "n_generic": len(generic_texts),
                   "latency_ms_p50": latency, "warnings": warnings, "date": date.today().isoformat(),
                   "seconds": round(time.perf_counter() - t0)})
    home = home or PRESET_HOME
    home.mkdir(parents=True, exist_ok=True)
    np.save(home / center_asset_name(name, backend, "one_word"),
            generic_center[lay["one_word"]].astype(np.float32))
    preset = Preset(
        name=name, repo=repo if backend == "mlx" else "", torch_repo=repo if backend == "torch" else None,
        formulations=(Formulation("one_word", ONE_WORD, lay["one_word"]),
                      Formulation("question_options", QUESTION_OPTIONS, lay["question_options"])),
        tau=round(report["tau"], 5), center=report["center"],
        one_word=(lay["one_word_only"], round(report["tau_one_word"], 5)),
        latency_ms=f"~{latency:.0f}",
        quality=(f"dev accuracy {report['dev_accuracy']:.3f} ± {report['dev_accuracy_stderr']:.3f} "
                 f"(n={report['n_dev']}); not measured on the Jev bench"),
        notes=f"Fitted by `jul models add` on {report['date']}.",
        backend=backend, asset_dir=home, calibration=report)
    path = save_preset(preset, home)
    _log(f"Saved {path} ({report['seconds']}s)")
    return preset


def _latency(backbone: Backbone, dev: list[DevSet], l_ow: int, l_qo: int, per_set: int = 5) -> float:
    """p50 of one decision (both formulations, prefixes cached, each stopping at its layer), over
    texts of every dev set: their lengths range from a few words (massive) to a paragraph (yahoo)."""
    ow = _template(backbone, ONE_WORD)
    times = []
    for d in dev:
        qo = _template(backbone, _question_options(d.labels))
        for text in d.texts[: per_set + 1]:
            t = time.perf_counter()
            ow.run(text, layers=[l_ow])
            qo.run(text, layers=[l_qo])
            times.append(time.perf_counter() - t)
        times.pop(-per_set - 1)   # the first call of each template warms it up
    return float(np.median(times) * 1000)
