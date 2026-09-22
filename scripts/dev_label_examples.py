"""Option 1: represent each option by texts the model writes for it, not only by its label.

Once per question, the model writes K short fictitious texts for every option (it sees the whole
option list, so it can aim at what sets each option apart). The option vector becomes
alpha * vector(label) + (1 - alpha) * mean(vector(fictitious texts)). No task data is used: only the
options, like Jev. Dev sets only, never the Jev bench.

Phase 1 (generation) is cached in runs/dev/label-examples-<model>.json; phase 2 scores offline.

The writer can be another model than the encoder (4th argument), e.g. Qwen3.5-9B writes and MiniCPM5-2B
encodes. Prompt v2 asks for the texts themselves, as their author would write or say them, since
MiniCPM with v1 often described the text ("A user wants to know...") instead of writing it.

Usage: scripts/dev_label_examples.py [model] [N] [K] [writer] [prompt v1|v2]"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "scripts"))
from dev_common import short_names  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "minicpm5-2b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
WRITER = sys.argv[4] if len(sys.argv) > 4 else MODEL
VERSION = sys.argv[5] if len(sys.argv) > 5 else "v1"
DEV = ["yahootopics", "empathetic", "massive", "financialphrasebank"]
QUESTION = "Which single label best describes the input text?"
ONE = 'This text: "{x}" means in one word: "'
QO = '{q}\nPossible answers: {o}.\nText: "{x}"\nIn one word, the answer is: "'
L_ONE, L_QO = 39, 40  # preset minicpm5-2b
ALPHAS = [1.0, 0.5, 0.25, 0.0]
TAG = MODEL if (WRITER, VERSION) == (MODEL, "v1") else f"{MODEL}-by-{WRITER}-{VERSION}"
CACHE = ROOT / "runs" / "dev" / f"label-examples-{WRITER}-{VERSION}.json" if TAG != MODEL else ROOT / "runs" / "dev" / f"label-examples-{MODEL}.json"
GEN_PROMPT = """A text classifier must choose one of these categories:
{listing}

Write {k} different, realistic short texts (one or two sentences each) that clearly belong to the category "{label}" and not to any other category. Vary the topic, style and wording. One text per line, no numbering, no quotes, nothing else."""
if VERSION == "v2":
    GEN_PROMPT = """A text classifier must choose one of these categories:
{listing}

Write {k} different texts that clearly belong to the category "{label}" and not to any other category. Write the texts themselves, exactly as their author would write or say them (a question, a message, a spoken command, a sentence from an article...), never a description of a text. Keep each one short and realistic, and vary the wording. One text per line, no numbering, no quotes, nothing else."""

datasets = {}
for ds in DEV:
    rows = [json.loads(l) for l in open(ROOT / "data" / "dev" / f"{ds}.jsonl")][:N]
    unl = [json.loads(l)["text"] for l in open(ROOT / "data" / "dev" / f"{ds}.unlabeled.jsonl")][:N]
    datasets[ds] = (rows, unl, rows[0]["labels"])

# --- phase 1: the model writes K texts per option -----------------------------------------------
gen = json.loads(CACHE.read_text()) if CACHE.exists() else {}
todo = [(ds, lab) for ds, (_, _, labels) in datasets.items() for lab in labels if lab not in gen.get(ds, {})]
if todo:
    from mlx_lm import generate, load
    from jul.presets import resolve
    model, tok = load(resolve(WRITER).repo)
    for ds, lab in todo:
        labels = datasets[ds][2]
        msg = GEN_PROMPT.format(listing="\n".join(f"- {l}" for l in labels), k=K, label=lab)
        prompt = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        out = generate(model, tok, prompt, max_tokens=60 * K, verbose=False).split("</think>")[-1]
        lines = [l.strip(" -*•\"'").strip() for l in out.splitlines()]
        gen.setdefault(ds, {})[lab] = [l for l in lines if len(l) > 8][:K]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(gen, indent=1, ensure_ascii=False))
    print(f"generated {len(todo)} options", flush=True)
    del model
    mx.clear_cache()

# --- phase 2: encode and score -----------------------------------------------------------------
from jul.backbone import Backbone, PromptTemplate  # noqa: E402
from jul.presets import resolve  # noqa: E402

preset = resolve(MODEL)
bb = Backbone(preset.repo)
generic = preset.generic_center(preset.formulations[0])
nrm = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)


def encoder(tpl, layer):
    prefix, suffix = tpl.split("{x}")
    t = PromptTemplate(bb, prefix, suffix)

    def enc(texts):
        out = []
        for s in texts:
            h, _ = t.run(s, layers=[layer])
            out.append(h[layer][: h[layer].shape[0] // 2])
        return np.stack(out)
    return enc


def cos(X, O, c):
    return nrm(X - c) @ nrm(O - c).T


res = {}  # (alpha, center) -> {ds: (acc, top3)}
for ds, (rows, unl, labels) in datasets.items():
    y = np.array([r["target_index"] for r in rows])
    texts = [r["text"] for r in rows]
    fake = [gen[ds][l] or [l] for l in labels]
    n_fake = [len(f) for f in fake]
    parts = {}
    for name, tpl, layer in (("one", ONE, L_ONE),
                             ("qo", QO.replace("{q}", QUESTION).replace("{o}", ", ".join(short_names(labels))), L_QO)):
        enc = encoder(tpl, layer)
        F = enc([t for f in fake for t in f])
        idx = np.cumsum([0] + n_fake)
        parts[name] = dict(X=enc(texts), L=enc(labels), U=enc(unl),
                           E=np.stack([F[idx[i]:idx[i + 1]].mean(0) for i in range(len(labels))]))
    for a in ALPHAS:
        for center in ("preset", "task"):
            S = 0
            for name, p in parts.items():
                O = a * p["L"] + (1 - a) * p["E"]
                if center == "task":
                    c = p["U"].mean(0)
                elif name == "one":
                    c = generic
                else:
                    c = O.mean(0)
                S = S + cos(p["X"], O, c)
            top3 = (np.argsort(-S, 1)[:, :3] == y[:, None]).any(1).mean()
            res.setdefault((a, center), {})[ds] = (float((S.argmax(1) == y).mean()), float(top3))
    print(f"  {ds}: done ({np.mean(n_fake):.1f} texts per option)", flush=True)

(ROOT / "runs" / "dev" / f"label-examples-scores-{TAG}.json").write_text(json.dumps(
    {f"{a}|{c}": v for (a, c), v in res.items()}, indent=2))
for center in ("preset", "task"):
    print(f"\ncenter = {center}  (combination one word + question/options; acc per dataset, mean, top-3)")
    print(f"  {'alpha(label)':<13}" + "".join(f"{d[:10]:>12}" for d in DEV) + "        mean    top3")
    for a in ALPHAS:
        r = res[(a, center)]
        print(f"  {a:<13}" + "".join(f"{r[d][0]:>12.3f}" for d in DEV)
              + f"     {np.mean([r[d][0] for d in DEV]):.3f}   {np.mean([r[d][1] for d in DEV]):.3f}")
