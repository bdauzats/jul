"""Compare centering strategies for the vector method, on the dev datasets only (never the bench).

The center subtracted before the cosine can come from four places:

  none      no centering at all
  options   the mean of the question's own option vectors (free, today's fallback)
  generic   the mean of varied texts (data/generic), read through the same formulation
  task      the mean of the task's own unlabeled texts (what a Context(examples=...) gives)

"one word" does not mention the question, so its generic center is one vector per model and can be
shipped as an asset. "question + options" embeds the question, so its generic center has to be
recomputed for every new question -- which is only worth it if it clearly wins here.

State and option vectors do not depend on the center, so they are computed once and every strategy is
scored offline from them.

Usage: scripts/dev_generic_center.py <model> [n_dev] [n_generic]
Writes jul/lib/jul/assets/<model>.one_word.center.npy and runs/dev/center-<model>.json
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "scripts"))

from dev_common import short_names  # noqa: E402

from jul.backbone import MODELS, Backbone, PromptTemplate  # noqa: E402
from jul.presets import resolve  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5-9b"
N_DEV = int(sys.argv[2]) if len(sys.argv) > 2 else 50
N_GENERIC = int(sys.argv[3]) if len(sys.argv) > 3 else 200
DEV = ["yahootopics", "empathetic", "massive", "financialphrasebank"]
QUESTION = "Which single label best describes the input text?"

nrm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
preset = resolve(MODEL)
MODELS[preset.name] = preset.repo
bb = Backbone(preset.name)
ONE_WORD, QUESTION_OPTIONS = preset.formulations


def template(text: str) -> PromptTemplate:
    prefix, suffix = text.split("{state}")
    return PromptTemplate(bb, prefix, suffix)


def vectors(tpl: PromptTemplate, layer: int, texts: list[str]) -> np.ndarray:
    out = []
    for t in texts:
        h, _ = tpl.run(t, layers=[layer])
        out.append(np.array(h[layer][: h[layer].shape[0] // 2].astype(mx.float32)))
    return np.stack(out)


def question_prompt(labels: list[str]) -> str:
    return (QUESTION_OPTIONS.template.replace("{instructions}", QUESTION)
            .replace("{options}", ", ".join(short_names(labels))))


# Varied texts, balanced over the generic datasets (disjoint from the bench and from dev).
files = sorted((ROOT / "data" / "generic").glob("*.jsonl"))
per_file = max(1, N_GENERIC // len(files))
generic = [json.loads(l)["text"] for f in files for l in list(open(f))[:per_file]][:N_GENERIC]
print(f"{preset.name}: {len(generic)} generic texts from {len(files)} datasets, "
      f"{N_DEV} rows per dev dataset", flush=True)

t0 = time.perf_counter()
one_word_tpl = template(ONE_WORD.template)
vectors(one_word_tpl, ONE_WORD.layer, ["warm up"])
generic_center_ow = vectors(one_word_tpl, ONE_WORD.layer, generic).mean(0)
print(f"  one-word generic center: {time.perf_counter() - t0:.0f}s", flush=True)

results = {name: {} for name in ("none", "options", "generic", "task", "generic+options")}
for ds in DEV:
    rows = [json.loads(l) for l in open(ROOT / "data" / "dev" / f"{ds}.jsonl")][:N_DEV]
    unlabeled = [json.loads(l)["text"] for l in open(ROOT / "data" / "dev" / f"{ds}.unlabeled.jsonl")][:N_DEV]
    labels = rows[0]["labels"]
    y = np.array([r["target_index"] for r in rows])
    texts = [r["text"] for r in rows]

    qo_tpl = template(question_prompt(labels))
    pack = {}
    for key, tpl, layer in (("ow", one_word_tpl, ONE_WORD.layer), ("qo", qo_tpl, QUESTION_OPTIONS.layer)):
        pack[key] = {
            "X": vectors(tpl, layer, texts),
            "L": vectors(tpl, layer, labels),
            "task": vectors(tpl, layer, unlabeled).mean(0),
        }
    pack["ow"]["generic"] = generic_center_ow
    pack["qo"]["generic"] = vectors(qo_tpl, QUESTION_OPTIONS.layer, generic).mean(0)

    def score(centers: dict) -> float:
        total = 0
        for key, c in centers.items():
            p = pack[key]
            total = total + nrm(p["X"] - c) @ nrm(p["L"] - c).T
        return float(((total / len(centers)).argmax(1) == y).mean())

    zero = np.zeros_like(generic_center_ow)
    strategies = {
        "none": {"ow": zero, "qo": np.zeros_like(pack["qo"]["generic"])},
        "options": {"ow": pack["ow"]["L"].mean(0), "qo": pack["qo"]["L"].mean(0)},
        "generic": {"ow": pack["ow"]["generic"], "qo": pack["qo"]["generic"]},
        "task": {"ow": pack["ow"]["task"], "qo": pack["qo"]["task"]},
        "generic+options": {"ow": pack["ow"]["generic"], "qo": pack["qo"]["L"].mean(0)},
    }
    for name, centers in strategies.items():
        results[name][ds] = score(centers)
    print(f"  {ds:<22} " + "  ".join(f"{n}={results[n][ds]:.3f}" for n in results), flush=True)

print(f"\n{'strategy':<18} " + "  ".join(f"{d[:8]:>8}" for d in DEV) + "     mean")
ranked = sorted(results.items(), key=lambda kv: -np.mean(list(kv[1].values())))
for name, per_ds in ranked:
    mean = np.mean([per_ds[d] for d in DEV])
    print(f"{name:<18} " + "  ".join(f"{per_ds[d]:8.3f}" for d in DEV) + f"   {mean:.3f}")

asset = ROOT / "lib" / "jul" / "assets" / f"{preset.name}.one_word.center.npy"
np.save(asset, generic_center_ow.astype(np.float32))
out = ROOT / "runs" / "dev" / f"center-{preset.name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"model": preset.name, "n_dev": N_DEV, "n_generic": len(generic),
                           "layers": {"one_word": ONE_WORD.layer, "question_options": QUESTION_OPTIONS.layer},
                           "accuracy": results}, indent=2))
print(f"\nsaved {asset.relative_to(ROOT)} and {out.relative_to(ROOT)}")
print(f"total {time.perf_counter() - t0:.0f}s")
