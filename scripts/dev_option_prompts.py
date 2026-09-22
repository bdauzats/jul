"""How should an option be encoded? Compare option-side prompts, the text side unchanged.

The text vector is read where the model is about to say its answer. Today an option goes through the
same template as a text, as if it were a text to classify (V0). The variants keep the question and the
option list, and end on the same words as the text side, so both vectors are read at the same moment:

  question/options (qo):
    V0  ...Text: "<option>"\nIn one word, the answer is: "            (today)
    V1  ...The correct answer is <option>.\nIn one word, the answer is: "
    V2  ...Text: a text about <option>\nIn one word, the answer is: "
    V3  ..."<option>" in one word: "
  one word (ow):
    V0  This text: "<option>" means in one word: "                    (today)
    V2  This text: a text about <option> means in one word: "

<option> is the option exactly as given (dev: the dataset's label text); the list keeps the short names,
like the lib. Two centerings:
  shared  one center for both sides, as the lib does (generic for ow, mean of the options for qo)
  split   each side its own center: options minus their own mean; texts minus the mean of reference
          texts read through the same template -- "generic" (195 texts from data/generic, zero task
          data) or "task" (the task's unlabeled texts, reference only, not comparable to Jev).
Dev sets only.

Usage: scripts/dev_option_prompts.py [model] [N]"""

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
HEAD = '{q}\nPossible answers: {o}.\n'
TEXT_QO = HEAD + 'Text: "{x}"\nIn one word, the answer is: "'
TEXT_OW = 'This text: "{x}" means in one word: "'
OPT_QO = {
    "V0": HEAD + 'Text: "{x}"\nIn one word, the answer is: "',
    "V1": HEAD + 'The correct answer is {x}.\nIn one word, the answer is: "',
    "V2": HEAD + 'Text: a text about {x}\nIn one word, the answer is: "',
    "V3": HEAD + '"{x}" in one word: "',
}
OPT_OW = {"V0": 'This text: "{x}" means in one word: "', "V2": 'This text: a text about {x} means in one word: "'}
L_OW, L_QO = 39, 40

preset = resolve(MODEL)
bb = Backbone(preset.repo)
generic_center = preset.generic_center(preset.formulations[0])
nrm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)


def encode(tpl, layer, texts):
    prefix, suffix = tpl.split("{x}")
    t = PromptTemplate(bb, prefix, suffix)
    out = []
    for s in texts:
        h, _ = t.run(s, layers=[layer])
        out.append(h[layer][: h[layer].shape[0] // 2])
    return np.stack(out)


def cos(A, B, c, cb=None):
    return nrm(A - c) @ nrm(B - (c if cb is None else cb)).T


files = sorted((ROOT / "data" / "generic").glob("*.jsonl"))
generic = [json.loads(l)["text"] for f in files for l in list(open(f))[:13]]


res = {}  # variant name -> {ds: acc}
for ds in DEV:
    rows = [json.loads(l) for l in open(ROOT / "data" / "dev" / f"{ds}.jsonl")][:N]
    labels, y = rows[0]["labels"], np.array([r["target_index"] for r in rows])
    opts = [l.rstrip(" .") for l in labels]
    fill = lambda t: t.replace("{q}", QUESTION).replace("{o}", ", ".join(short_names(labels)))
    texts = [r["text"] for r in rows]
    unl = [json.loads(l)["text"] for l in open(ROOT / "data" / "dev" / f"{ds}.unlabeled.jsonl")][:N]
    X_ow, X_qo = encode(TEXT_OW, L_OW, texts), encode(fill(TEXT_QO), L_QO, texts)
    C_ow = {"generic": generic_center, "task": encode(TEXT_OW, L_OW, unl).mean(0)}
    C_qo = {"generic": encode(fill(TEXT_QO), L_QO, generic).mean(0), "task": encode(fill(TEXT_QO), L_QO, unl).mean(0)}
    Lo = {v: encode(t, L_OW, opts) for v, t in OPT_OW.items()}
    Lq = {v: encode(fill(t), L_QO, opts) for v, t in OPT_QO.items()}
    acc = lambda S: float((S.argmax(1) == y).mean())
    for mode in ("shared", "split generic", "split task"):
        if mode == "shared":
            S_ow = {v: cos(X_ow, L, generic_center) for v, L in Lo.items()}
            S_qo = {v: cos(X_qo, L, L.mean(0)) for v, L in Lq.items()}
        else:
            ref = mode.split()[1]
            S_ow = {v: cos(X_ow, L, C_ow[ref], L.mean(0)) for v, L in Lo.items()}
            S_qo = {v: cos(X_qo, L, C_qo[ref], L.mean(0)) for v, L in Lq.items()}
        for v, S in S_ow.items():
            res.setdefault(f"{mode} | ow {v} alone", {})[ds] = acc(S)
        for v, S in S_qo.items():
            res.setdefault(f"{mode} | qo {v} alone", {})[ds] = acc(S)
        for vo, So in S_ow.items():
            for vq, Sq in S_qo.items():
                res.setdefault(f"{mode} | ow {vo} + qo {vq}", {})[ds] = acc(So + Sq)
        if ds == "financialphrasebank":
            print(f"    {mode}: predicted classes, qo V1: {np.bincount(S_qo['V1'].argmax(1), minlength=3)}, truth {np.bincount(y, minlength=3)}")
    print(f"  {ds}: done", flush=True)

(ROOT / "runs" / "dev" / f"option-prompts-{MODEL}.json").write_text(json.dumps(res, indent=2))
print(f"\n  {'variant':<36}" + "".join(f"{d[:10]:>12}" for d in DEV) + "     mean")
for name, r in res.items():
    print(f"  {name:<36}" + "".join(f"{r[d]:>12.3f}" for d in DEV) + f"    {np.mean(list(r.values())):.3f}")
