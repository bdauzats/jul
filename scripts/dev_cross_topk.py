"""Idea 4 on top of idea A: shortlist with the vector, then settle with the cross reading.

The vector (today's cosine, one word + question/options) ranks the options; only its top k get the
cross reading of scripts/dev_cross_options.py (question + option list + text cached, then one short
suffix per option, score = log p(yes) - log p(no)). Final choice among the top k:
z = cosine / tau + lambda * score. The option bias (mean score over 20 generic texts) is computed once
per question for every option. Cost per call: k short suffixes, whatever the number of options.
No task data anywhere. Scores are saved so that k and lambda can be re-read offline.

Usage: scripts/dev_cross_topk.py [model] [N] [K]"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from mlx_lm.models.cache import make_prompt_cache

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "scripts"))
from dev_common import short_names  # noqa: E402

from jul.backbone import Backbone, PromptTemplate  # noqa: E402
from jul.presets import resolve  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "minicpm5-2b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
N_GENERIC = 20
DEV = ["yahootopics", "empathetic", "massive", "financialphrasebank"]
QUESTION = "Which single label best describes the input text?"
ONE = 'This text: "{x}" means in one word: "'
QO = '{q}\nPossible answers: {o}.\nText: "{x}"\nIn one word, the answer is: "'
L_ONE, L_QO = 39, 40
SUFFIX = '"\nProposed answer: {option}\nIs the proposed answer correct? Answer yes or no.\nAnswer:'
LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0]

preset = resolve(MODEL)
bb = Backbone(preset.repo)
generic_center = preset.generic_center(preset.formulations[0])
files = sorted((ROOT / "data" / "generic").glob("*.jsonl"))
generic = [json.loads(l)["text"] for f in files for l in list(open(f))[:2]][:N_GENERIC]
nrm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
YES = sorted({bb.encode(v)[0] for v in (" yes", " Yes", "yes", "Yes")})
NO = sorted({bb.encode(v)[0] for v in (" no", " No", "no", "No")})


def trim_to(cache, n):
    for c in cache:
        if c.offset > n:
            c.trim(c.offset - n)


def cross_scores(prefix, texts, options, subsets):
    """Scores (len(texts), len(options)), NaN outside each text's subset; seconds per text."""
    cache = make_prompt_cache(bb.model)
    bb.forward(bb.encode(prefix), cache=cache, logits=True)
    n_prefix = cache[0].offset
    suffixes = [bb.encode(SUFFIX.format(option=o)) for o in options]
    out = np.full((len(texts), len(options)), np.nan)
    t0 = time.perf_counter()
    for i, (text, subset) in enumerate(zip(texts, subsets)):
        bb.forward(bb.encode(text), cache=cache, logits=True)
        n_text = cache[0].offset
        for j in subset:
            _, lg = bb.forward(suffixes[j], cache=cache, logits=True)
            lg = np.array(lg)
            out[i, j] = np.logaddexp.reduce(lg[YES]) - np.logaddexp.reduce(lg[NO])
            trim_to(cache, n_text)
        trim_to(cache, n_prefix)
    return out, (time.perf_counter() - t0) / len(texts)


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


def pick(C, S, k, lam):
    """Choice among the top k of C, by C + lam * S."""
    top = np.argsort(-C, 1)[:, :k]
    z = np.take_along_axis(C, top, 1) + lam * np.take_along_axis(S, top, 1)
    return top[np.arange(len(C)), z.argmax(1)]


res, saved = {}, {}
for ds in DEV:
    rows = [json.loads(l) for l in open(ROOT / "data" / "dev" / f"{ds}.jsonl")][:N]
    labels, y = rows[0]["labels"], np.array([r["target_index"] for r in rows])
    opts = [l.rstrip(" .") for l in labels]
    names = ", ".join(short_names(labels))
    texts = [r["text"] for r in rows]
    qo = QO.replace("{q}", QUESTION).replace("{o}", names)
    L2 = encode(qo, L_QO, labels)
    C = (cos(encode(ONE, L_ONE, texts), encode(ONE, L_ONE, labels), generic_center)
         + cos(encode(qo, L_QO, texts), L2, L2.mean(0))) / 2 / preset.tau
    k = min(K, len(opts))
    subsets = [list(r) for r in np.argsort(-C, 1)[:, :k]]
    prefix = f'{QUESTION}\nPossible answers: {names}.\nText: "'
    S, sec = cross_scores(prefix, texts, opts, subsets)
    bias = cross_scores(prefix, generic, opts, [range(len(opts))] * len(generic))[0].mean(0)
    saved[ds] = {"cos": C.tolist(), "cross": np.nan_to_num(S, nan=-1e9).tolist(), "bias": bias.tolist(), "y": y.tolist()}
    acc = lambda p, y=y: float((p == y).mean())
    res.setdefault("cosine only (today)", {})[ds] = acc(C.argmax(1))
    for label_k in sorted({2, 3, 5, K}):
        kk = min(label_k, len(opts))  # the key keeps the requested k, so every dataset fills every row
        res.setdefault(f"top-{label_k} recall", {})[ds] = float((np.argsort(-C, 1)[:, :kk] == y[:, None]).any(1).mean())
        for corr, Sx in (("raw", S), ("bias-corr", S - bias)):
            for lam in LAMBDAS:
                res.setdefault(f"top-{label_k} {corr} l={lam}", {})[ds] = acc(pick(C, np.nan_to_num(Sx, nan=-1e9), kk, lam))
    print(f"  {ds}: {sec * 1000:.0f} ms per text for {k} suffixes (looped)", flush=True)

(ROOT / "runs" / "dev" / f"cross-topk-{MODEL}.json").write_text(json.dumps({"acc": res, "scores": saved}))
print(f"\n  {'variant':<28}" + "".join(f"{d[:10]:>12}" for d in DEV) + "     mean")
for name, r in res.items():
    print(f"  {name:<28}" + "".join(f"{r[d]:>12.3f}" for d in DEV) + f"    {np.mean(list(r.values())):.3f}")
