"""Does the wording of the vector template matter? "in one word" vs "in one idea" / "one sentence" / ...

Single formulation, vector method, dev sets only (never the Jev bench). Two centers are reported:
"options" (mean of the option vectors, zero task data) and "task" (mean of unlabeled task texts).
Accuracy is read over a few layers around the tuned one, since a new wording may peak elsewhere.

Usage: scripts/dev_wording.py [model] [N]"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from jul.backbone import Backbone, PromptTemplate  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openbmb/MiniCPM5-2B-MLX"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
DEV = ["yahootopics", "empathetic", "massive", "financialphrasebank"]
LAYERS = [35, 37, 39, 40, 41]
TEMPLATES = {
    "one word": 'This text: "{x}" means in one word: "',
    "one idea": 'This text: "{x}" means in one idea: "',
    "one sentence": 'This text: "{x}" means in one sentence: "',
    "one phrase": 'This text: "{x}" means in one phrase: "',
    "few words": 'This text: "{x}" means in a few words: "',
}
nrm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
bb = Backbone(MODEL)


def vecs(tpl, texts):
    prefix, suffix = tpl.split("{x}")
    t = PromptTemplate(bb, prefix, suffix)
    out = []
    for s in texts:
        h, _ = t.run(s, layers=LAYERS)
        out.append({l: h[l][: h[l].shape[0] // 2] for l in LAYERS})
    return out


def acc(X, L, c, l, y):
    s = nrm(np.stack([v[l] for v in X]) - c) @ nrm(np.stack([v[l] for v in L]) - c).T
    return float((s.argmax(1) == y).mean())


res = {}  # (template, center, layer) -> [acc per dataset]
for ds in DEV:
    rows = [json.loads(l) for l in open(ROOT / "data" / "dev" / f"{ds}.jsonl")][:N]
    unl = [json.loads(l)["text"] for l in open(ROOT / "data" / "dev" / f"{ds}.unlabeled.jsonl")][:N]
    labels, y = rows[0]["labels"], np.array([r["target_index"] for r in rows])
    for name, tpl in TEMPLATES.items():
        X, L, U = vecs(tpl, [r["text"] for r in rows]), vecs(tpl, labels), vecs(tpl, unl)
        for l in LAYERS:
            for cname, c in (("options", np.mean([v[l] for v in L], 0)), ("task", np.mean([v[l] for v in U], 0))):
                res.setdefault((name, cname, l), []).append(acc(X, L, c, l, y))
    print(f"  {ds}: done", flush=True)

out = {f"{t}|{c}|{l}": v for (t, c, l), v in res.items()}
(ROOT / "runs" / "dev").mkdir(parents=True, exist_ok=True)
(ROOT / "runs" / "dev" / "wording.json").write_text(json.dumps({"model": MODEL, "n": N, "datasets": DEV, "acc": out}, indent=2))

for cname in ("options", "task"):
    print(f"\ncenter = {cname}   (mean dev accuracy; per-dataset at the best layer)")
    print(f"  {'template':<14}" + "".join(f"  L{l:<5}" for l in LAYERS) + "   best   " + "  ".join(d[:6] for d in DEV))
    for name in TEMPLATES:
        means = [np.mean(res[(name, cname, l)]) for l in LAYERS]
        b = LAYERS[int(np.argmax(means))]
        print(f"  {name:<14}" + "".join(f"  {m:.3f} " for m in means)
              + f"   L{b:<4} " + "  ".join(f"{a:.3f}" for a in res[(name, cname, b)]))
