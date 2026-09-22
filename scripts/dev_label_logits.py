"""Option 3: read the probability of each label's first token in the same pass, and mix it with the cosine.

Both templates end on an opening quote, where the model is about to write its one-word answer. The
forward pass that gives the vector can also give the next-token logits (it only has to run the last
few layers). For each option we take the log-probability of the first token of its short name
(logsumexp over "Name", "name", " name" variants), normalised over the options, and mix:

  z = mean cosine / tau + lambda * log p(first token)

No extra pass, no task data. Limit: options whose short names share their first token (massive:
"The intent of ...") get no signal from the logits, only from the cosine.
Same preset as the lib (MiniCPM5-2B, one word L39 generic center + question/options L40 options center).

Usage: scripts/dev_label_logits.py [model] [N]"""

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
from jul.presets import resolve  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "minicpm5-2b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
DEV = ["yahootopics", "empathetic", "massive", "financialphrasebank"]
QUESTION = "Which single label best describes the input text?"
ONE = 'This text: "{x}" means in one word: "'
QO = '{q}\nPossible answers: {o}.\nText: "{x}"\nIn one word, the answer is: "'
L_ONE, L_QO = 39, 40
LAMBDAS = [0.1, 0.25, 0.5, 1.0, 2.0]

preset = resolve(MODEL)
bb = Backbone(preset.repo)
generic_center = preset.generic_center(preset.formulations[0])
nrm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)


def first_tokens(name):
    variants = {name, name.lower(), name[:1].upper() + name[1:], " " + name, " " + name.lower()}
    return sorted({bb.encode(v)[0] for v in variants if bb.encode(v)})


def encode(tpl, layer, texts, logits=False):
    prefix, suffix = tpl.split("{x}")
    t = PromptTemplate(bb, prefix, suffix)
    vecs, logps = [], []
    for s in texts:
        h, lg = t.run(s, layers=[layer], logits=logits)
        vecs.append(h[layer][: h[layer].shape[0] // 2])
        if logits:
            lg = np.array(lg)
            logps.append(lg - lg.max() - np.log(np.exp(lg - lg.max()).sum()))
    return np.stack(vecs), (np.stack(logps) if logits else None)


def label_logp(logp, toks):
    """(n, K) log p of each option's first token, renormalised over the options."""
    s = np.stack([np.logaddexp.reduce(logp[:, t], axis=1) for t in toks], 1)
    return s - np.logaddexp.reduce(s, axis=1, keepdims=True)


def cos(A, B, c):
    return nrm(A - c) @ nrm(B - c).T


res = {}  # (source, lambda) -> {ds: acc}
for ds in DEV:
    rows = [json.loads(l) for l in open(ROOT / "data" / "dev" / f"{ds}.jsonl")][:N]
    labels, y = rows[0]["labels"], np.array([r["target_index"] for r in rows])
    names = short_names(labels)
    toks = [first_tokens(n) for n in names]
    n_distinct = len({tuple(t) for t in toks})
    texts = [r["text"] for r in rows]
    qo = QO.replace("{q}", QUESTION).replace("{o}", ", ".join(names))
    X1, P1 = encode(ONE, L_ONE, texts, logits=True)
    L1, _ = encode(ONE, L_ONE, labels)
    X2, P2 = encode(qo, L_QO, texts, logits=True)
    L2, _ = encode(qo, L_QO, labels)
    C = (cos(X1, L1, generic_center) + cos(X2, L2, L2.mean(0))) / 2 / preset.tau
    lp = {"one": label_logp(P1, toks), "qo": label_logp(P2, toks)}
    lp["both"] = (lp["one"] + lp["qo"]) / 2
    acc = lambda S: float((S.argmax(1) == y).mean())
    res.setdefault(("cosine only", 0), {})[ds] = acc(C)
    for src, P in lp.items():
        res.setdefault((f"logits {src} only", 0), {})[ds] = acc(P + 1e-6 * C)  # cosine breaks ties
        for lam in LAMBDAS:
            res.setdefault((f"cos + {src}", lam), {})[ds] = acc(C + lam * P)
    print(f"  {ds}: done ({n_distinct} distinct first tokens for {len(labels)} options)", flush=True)

(ROOT / "runs" / "dev" / f"label-logits-{MODEL}.json").write_text(json.dumps(
    {f"{s}|{l}": v for (s, l), v in res.items()}, indent=2))
print(f"\n  {'variant':<22}" + "".join(f"{d[:10]:>12}" for d in DEV) + "     mean")
for (s, lam), r in res.items():
    name = s if lam == 0 else f"{s} l={lam}"
    print(f"  {name:<22}" + "".join(f"{r[d]:>12.3f}" for d in DEV) + f"    {np.mean(list(r.values())):.3f}")
