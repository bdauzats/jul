"""Fit a model's vector-method temperature on the dev datasets (never on the benchmark).

softmax(cosine / tau) must be calibrated per backbone; tau is chosen by pooled NLL over the dev sets,
for "one word" and for "one word + question/options". Writes runs/dev/tau-<model>.json.

Usage: scripts/dev_fit_tau.py <model> <layer_one_word> <layer_combo> [N]"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "scripts"))
from dev_common import short_names  # noqa: E402

from jul.backbone import Backbone, PromptTemplate  # noqa: E402
from jul.calibration import metrics  # noqa: E402

MODEL, L_ONE, L_COMBO = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 50
DEV = ["yahootopics", "empathetic", "massive", "financialphrasebank"]
QUESTION = "Which single label best describes the input text?"
nrm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
bb = Backbone(MODEL)
layers = sorted({L_ONE, L_COMBO})


def vecs(tpl, texts):
    prefix, suffix = tpl.split("{x}")
    t = PromptTemplate(bb, prefix, suffix)
    out = []
    for s in texts:
        h, _ = t.run(s, layers=layers)
        out.append({l: np.array(h[l].astype(mx.float32))[: h[l].shape[0] // 2] for l in layers})
    return out


def cos(X, L, C, l):
    c = np.mean([v[l] for v in C], axis=0)
    return nrm(np.stack([v[l] for v in X]) - c) @ nrm(np.stack([v[l] for v in L]) - c).T


pooled = {"one word": [], "one word + question/options": []}
for ds in DEV:
    rows = [json.loads(l) for l in open(ROOT / "data" / "dev" / f"{ds}.jsonl")][:N]
    unl = [json.loads(l)["text"] for l in open(ROOT / "data" / "dev" / f"{ds}.unlabeled.jsonl")][:N]
    labels, y = rows[0]["labels"], np.array([r["target_index"] for r in rows])
    S = {}
    for name, tpl in {"one": 'This text: "{x}" means in one word: "',
                      "qo": f"{QUESTION}\nPossible answers: {', '.join(short_names(labels))}.\n" + 'Text: "{x}"\nIn one word, the answer is: "'}.items():
        S[name] = (vecs(tpl, [r["text"] for r in rows]), vecs(tpl, labels), vecs(tpl, unl))
    one = cos(*S["one"], L_ONE)
    combo = (cos(*S["one"], L_COMBO) + cos(*S["qo"], L_COMBO)) / 2
    pooled["one word"] += list(zip(one, y))
    pooled["one word + question/options"] += list(zip(combo, y))
    print(f"  {ds}: done", flush=True)


def nll(tau, data):
    out = []
    for s, t in data:
        z = s / tau
        out.append(-(z[t] - z.max() - np.log(np.exp(z - z.max()).sum())))
    return float(np.mean(out))


taus = {}
grid = np.exp(np.linspace(np.log(0.003), np.log(1.0), 300))
for name, data in pooled.items():
    tau = float(min(grid, key=lambda t: nll(t, data)))
    taus[name] = tau
    for tau_used, label in ((0.0456, "MiniCPM tau 0.0456"), (tau, f"fitted tau {tau:.4f}")):
        eces = []
        for ds_rows in [data[i * N:(i + 1) * N] for i in range(len(DEV))]:
            m = metrics(np.stack([s / tau_used for s, _ in ds_rows]), np.array([t for _, t in ds_rows]))
            eces.append(m["ece"])
        print(f"  {name:<30} {label:<22} mean dev ECE {np.mean(eces):.3f}")
out = ROOT / "runs" / "dev" / f"tau-{MODEL}.json"
out.write_text(json.dumps({"model": MODEL, "layer_one_word": L_ONE, "layer_combo": L_COMBO, "n_per_dataset": N, "tau": taus}, indent=2))
print("saved", out, taus)
