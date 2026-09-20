"""Does a Context help? Two questions at once, on the dev datasets only.

  description   one sentence describing the data, prepended to both formulations as `Context: ...`.
                It sits in the prompt, so it changes the vectors: everything is encoded twice.
  examples      how many unlabeled texts the task center needs (10, 50, 200). The center does not
                change the vectors, so all sizes are scored from a single encoding.

Plan criterion for the description: mean gain >= 2 points, and no dataset losing more than 3.

Usage: scripts/dev_context_effect.py <model> [n_rows]
Writes runs/dev/context-effect-<model>.json
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

MODEL = sys.argv[1] if len(sys.argv) > 1 else "minicpm5-2b"
N_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
N_UNLABELED = 200
SIZES = [10, 50, 200]
QUESTION = "Which single label best describes the input text?"

# One neutral sentence per dataset: what the data IS, never a hint at the answer.
DESCRIPTIONS = {
    "yahootopics": "Questions and their answers posted by users on the Yahoo Answers forum, in English.",
    "empathetic": "Short messages in which a person describes a personal situation they went through, in English.",
    "massive": "Short spoken commands addressed to a voice assistant, transcribed to text, in English.",
    "financialphrasebank": "Sentences taken from financial news articles about listed companies, in English.",
}

nrm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
preset = resolve(MODEL)
MODELS[preset.name] = preset.repo
bb = Backbone(preset.name)


def encode(prompt: str, layer: int, texts: list[str]) -> np.ndarray:
    prefix, suffix = prompt.split("{state}")
    tpl = PromptTemplate(bb, prefix, suffix)
    out = []
    for t in texts:
        h, _ = tpl.run(t, layers=[layer])
        out.append(np.array(h[layer][: h[layer].shape[0] // 2].astype(mx.float32)))
    return np.stack(out)


def prompts(labels: list[str], description: str | None) -> dict:
    head = f"Context: {description}\n" if description else ""
    out = {}
    for f in preset.formulations:
        text = f.template
        if "{instructions}" in text:
            text = (text.replace("{instructions}", QUESTION)
                        .replace("{options}", ", ".join(short_names(labels))))
        out[f.name] = (head + text, f.layer)
    return out


results = {}
t_start = time.perf_counter()
for ds, description in DESCRIPTIONS.items():
    rows = [json.loads(l) for l in open(ROOT / "data" / "dev" / f"{ds}.jsonl")][:N_ROWS]
    unlabeled = [json.loads(l)["text"] for l in open(ROOT / "data" / "dev" / f"{ds}.unlabeled.jsonl")][:N_UNLABELED]
    labels = rows[0]["labels"]
    y = np.array([r["target_index"] for r in rows])
    texts = [r["text"] for r in rows]
    results[ds] = {}

    for variant, desc in (("without", None), ("with", description)):
        pack = {}
        for name, (prompt, layer) in prompts(labels, desc).items():
            pack[name] = {"X": encode(prompt, layer, texts),
                          "L": encode(prompt, layer, labels),
                          "U": encode(prompt, layer, unlabeled)}
            if name == "one_word" and preset.center == "generic":
                g = preset.generic_center(next(f for f in preset.formulations if f.name == name))
                pack[name]["default"] = g if g is not None else pack[name]["L"].mean(0)
            else:
                pack[name]["default"] = pack[name]["L"].mean(0)

        def accuracy(center_of) -> float:
            total = sum(nrm(p["X"] - center_of(p)) @ nrm(p["L"] - center_of(p)).T for p in pack.values())
            return float(((total / len(pack)).argmax(1) == y).mean())

        scores = {"default": accuracy(lambda p: p["default"])}
        for n in SIZES:
            scores[f"examples={n}"] = accuracy(lambda p, n=n: p["U"][:n].mean(0))
        results[ds][variant] = scores
        print(f"  {ds:<22} {variant:<8} " + "  ".join(f"{k}={v:.3f}" for k, v in scores.items()), flush=True)

print(f"\n--- description, at the preset's own center ({preset.center}) ---")
deltas = []
for ds in DESCRIPTIONS:
    a, b = results[ds]["without"]["default"], results[ds]["with"]["default"]
    deltas.append(b - a)
    print(f"  {ds:<22} without={a:.3f}  with={b:.3f}  delta={b-a:+.3f}")
mean_gain, worst = float(np.mean(deltas)), float(np.min(deltas))
verdict = "PASS" if mean_gain >= 0.02 and worst >= -0.03 else "FAIL"
print(f"  mean gain {mean_gain:+.3f} (needs >= +0.020), worst {worst:+.3f} (needs >= -0.030) -> {verdict}")

print("\n--- how many examples the task center needs (no description) ---")
print(f"  {'dataset':<22} " + "  ".join(f"{k:>12}" for k in ("default", *[f'examples={n}' for n in SIZES])))
for ds in DESCRIPTIONS:
    s = results[ds]["without"]
    print(f"  {ds:<22} " + "  ".join(f"{s[k]:12.3f}" for k in ("default", *[f'examples={n}' for n in SIZES])))
mean_row = {k: float(np.mean([results[ds]["without"][k] for ds in DESCRIPTIONS]))
            for k in ("default", *[f"examples={n}" for n in SIZES])}
print(f"  {'MEAN':<22} " + "  ".join(f"{mean_row[k]:12.3f}" for k in mean_row))

out = ROOT / "runs" / "dev" / f"context-effect-{MODEL}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"model": MODEL, "n_rows": N_ROWS, "descriptions": DESCRIPTIONS,
                           "results": results, "description_verdict": verdict,
                           "description_mean_gain": mean_gain}, indent=2))
print(f"\nsaved {out.relative_to(ROOT)}  ({time.perf_counter()-t_start:.0f}s)")
