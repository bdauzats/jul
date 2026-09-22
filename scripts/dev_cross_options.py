"""Idea A: read the text and each option together (cross-encoding), instead of comparing two vectors.

Today the text and the options are encoded separately; the model never sees them side by side. Here the
question (and optionally the list of options) is a cached prefix, the text is appended once, then each
option is appended in turn as a short suffix, reusing the cache of the text:

  {question}\\n[Possible answers: a, b, c.\\n]Text: "{text}"
  Proposed answer: {option}
  Is the proposed answer correct? Answer yes or no.
  Answer:                                          -> score = log p(yes) - log p(no)

Variants: with / without the option list in the prefix; raw score, or minus each option's mean score
over 20 generic texts (an option-bias correction with zero task data); alone or mixed with today's
cosine (z = cosine / tau + lambda * score). No task data anywhere.
The options are looped one by one here (with KV-cache trimming): the time printed is an upper bound,
batching the suffixes would cut it.

Usage: scripts/dev_cross_options.py [model] [N]"""

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
N_GENERIC = 20
DEV = ["yahootopics", "empathetic", "massive", "financialphrasebank"]
QUESTION = "Which single label best describes the input text?"
ONE = 'This text: "{x}" means in one word: "'
QO = '{q}\nPossible answers: {o}.\nText: "{x}"\nIn one word, the answer is: "'
L_ONE, L_QO = 39, 40
SUFFIX = '"\nProposed answer: {option}\nIs the proposed answer correct? Answer yes or no.\nAnswer:'
LAMBDAS = [0.25, 0.5, 1.0, 2.0]

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


def cross_scores(prefix, texts, options):
    """(len(texts), len(options)) log p(yes) - log p(no), and seconds per text."""
    cache = make_prompt_cache(bb.model)
    bb.forward(bb.encode(prefix), cache=cache, logits=True)
    n_prefix = cache[0].offset
    suffixes = [bb.encode(SUFFIX.format(option=o)) for o in options]
    out, t0 = [], time.perf_counter()
    for text in texts:
        bb.forward(bb.encode(text), cache=cache, logits=True)
        n_text = cache[0].offset
        row = []
        for suf in suffixes:
            _, lg = bb.forward(suf, cache=cache, logits=True)
            lg = np.array(lg)
            row.append(np.logaddexp.reduce(lg[YES]) - np.logaddexp.reduce(lg[NO]))
            trim_to(cache, n_text)
        out.append(row)
        trim_to(cache, n_prefix)
    return np.array(out), (time.perf_counter() - t0) / len(texts)


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


res, secs = {}, {}
for ds in DEV:
    rows = [json.loads(l) for l in open(ROOT / "data" / "dev" / f"{ds}.jsonl")][:N]
    labels, y = rows[0]["labels"], np.array([r["target_index"] for r in rows])
    opts = [l.rstrip(" .") for l in labels]
    names = ", ".join(short_names(labels))
    texts = [r["text"] for r in rows]
    acc = lambda S, y=y: float((S.argmax(1) == y).mean())

    qo = QO.replace("{q}", QUESTION).replace("{o}", names)
    L2 = encode(qo, L_QO, labels)
    C = (cos(encode(ONE, L_ONE, texts), encode(ONE, L_ONE, labels), generic_center)
         + cos(encode(qo, L_QO, texts), L2, L2.mean(0))) / 2 / preset.tau
    res.setdefault("cosine only (today)", {})[ds] = acc(C)

    for variant, prefix in (("no list", f'{QUESTION}\nText: "'),
                            ("with list", f'{QUESTION}\nPossible answers: {names}.\nText: "')):
        X, sec = cross_scores(prefix, texts, opts)
        bias = cross_scores(prefix, generic, opts)[0].mean(0)
        secs.setdefault(variant, {})[ds] = sec
        for corr, S in (("raw", X), ("bias-corrected", X - bias)):
            res.setdefault(f"cross {variant}, {corr}", {})[ds] = acc(S + 1e-6 * C)
            for lam in LAMBDAS:
                res.setdefault(f"cos + cross {variant}, {corr} l={lam}", {})[ds] = acc(C + lam * S)
        print(f"  {ds} / {variant}: {sec * 1000:.0f} ms per text for {len(opts)} options (looped)", flush=True)

(ROOT / "runs" / "dev" / f"cross-options-{MODEL}.json").write_text(json.dumps({"acc": res, "sec_per_text": secs}, indent=2))
print(f"\n  {'variant':<44}" + "".join(f"{d[:10]:>12}" for d in DEV) + "     mean")
for name, r in res.items():
    print(f"  {name:<44}" + "".join(f"{r[d]:>12.3f}" for d in DEV) + f"    {np.mean(list(r.values())):.3f}")
