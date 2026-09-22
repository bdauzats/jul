"""Option 2: correct the options that attract everything (hubness), on the dev sets only.

Some option vectors are close to almost any text and win too often. For each option we measure how
close it is to varied texts that have nothing to do with the task (data/generic, 195 texts, neither
bench nor dev), and subtract that from its score:

  mean     score - beta * mean cosine(option, generic texts)
  csls-k   score - beta * mean cosine(option, its k nearest generic texts)    (CSLS, word translation)
  z        (score - mean) / std, both over the generic texts

Zero task data: only the options and the generic texts, like Jev. A "task" row uses the task's own
unlabeled texts instead of the generic ones, for reference only (not comparable to Jev).
Same preset as the lib: one word at L39 (generic center) + question/options at L40 (options center).
The one-word generic vectors are computed once; the question/options ones once per question.

Usage: scripts/dev_hubness.py [model] [N]"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "scripts"))
from dev_common import short_names  # noqa: E402

from jul.backbone import Backbone, PromptTemplate  # noqa: E402
from jul.presets import resolve  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "minicpm5-2b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
DEV = ["yahootopics", "empathetic", "massive", "financialphrasebank"]
QUESTION = "Which single label best describes the input text?"
ONE = 'This text: "{x}" means in one word: "'
QO = '{q}\nPossible answers: {o}.\nText: "{x}"\nIn one word, the answer is: "'
L_ONE, L_QO = 39, 40
BETAS = [0.5, 1.0]

preset = resolve(MODEL)
bb = Backbone(preset.repo)
generic_center = preset.generic_center(preset.formulations[0])
files = sorted((ROOT / "data" / "generic").glob("*.jsonl"))
generic = [json.loads(l)["text"] for f in files for l in list(open(f))[:13]]
nrm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)


def encode(tpl, layer, texts):
    prefix, suffix = tpl.split("{x}")
    t = PromptTemplate(bb, prefix, suffix)
    out = []
    for s in texts:
        h, _ = t.run(s, layers=[layer])
        out.append(h[layer][: h[layer].shape[0] // 2])
    return np.stack(out)


def cos(A, B, c):
    return nrm(A - c) @ nrm(B - c).T


def corrected(S, R, method, beta):
    """S: (n, K) scores of the texts, R: (m, K) scores of the reference texts against the options."""
    if method == "none":
        return S
    if method == "z":
        return (S - R.mean(0)) / R.std(0)
    k = R.shape[0] if method == "mean" else int(method.split("-")[1])
    return S - beta * np.sort(R, 0)[-k:].mean(0)


METHODS = [("none", 0)] + [(m, b) for m in ("mean", "csls-10", "csls-50") for b in BETAS] + [("z", 0)]
G_one = encode(ONE, L_ONE, generic)
res = {}  # (ref, method, beta) -> {ds: acc}
for ds in DEV:
    rows = [json.loads(l) for l in open(ROOT / "data" / "dev" / f"{ds}.jsonl")][:N]
    unl = [json.loads(l)["text"] for l in open(ROOT / "data" / "dev" / f"{ds}.unlabeled.jsonl")][:N]
    labels, y = rows[0]["labels"], np.array([r["target_index"] for r in rows])
    texts = [r["text"] for r in rows]
    qo = QO.replace("{q}", QUESTION).replace("{o}", ", ".join(short_names(labels)))
    parts = []
    for tpl, layer, G in ((ONE, L_ONE, G_one), (qo, L_QO, None)):
        X, L, U = encode(tpl, layer, texts), encode(tpl, layer, labels), encode(tpl, layer, unl)
        G = G if G is not None else encode(tpl, layer, generic)
        c = generic_center if tpl == ONE else L.mean(0)
        parts.append((cos(X, L, c), {"generic": cos(G, L, c), "task": cos(U, L, c)}))
    for ref in ("generic", "task"):
        for m, b in METHODS:
            S = sum(corrected(s, R[ref], m, b) for s, R in parts)
            res.setdefault((ref, m, b), {})[ds] = float((S.argmax(1) == y).mean())
    print(f"  {ds}: done", flush=True)

(ROOT / "runs" / "dev" / f"hubness-{MODEL}.json").write_text(json.dumps(
    {f"{r}|{m}|{b}": v for (r, m, b), v in res.items()}, indent=2))
for ref in ("generic", "task"):
    print(f"\nreference texts = {ref}  (combination one word + question/options, accuracy)")
    print(f"  {'method':<14}" + "".join(f"{d[:10]:>12}" for d in DEV) + "     mean")
    for m, b in METHODS:
        r = res[(ref, m, b)]
        name = m if m in ("none", "z") else f"{m} b={b}"
        print(f"  {name:<14}" + "".join(f"{r[d]:>12.3f}" for d in DEV) + f"    {np.mean(list(r.values())):.3f}")
