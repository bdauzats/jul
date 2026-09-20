"""Does a per-task head actually help, for which model, and from how many labeled examples?

Encodes each dataset once with the preset's vector method (the expensive part, cached on disk), then
trains a head on 25, 50, 100, ... examples and scores every size on the held-out `val` split.

The `test` split is deliberately never touched: it holds the benchmark rows, kept for etape 5.

Usage: scripts/dev_tuning_curve.py <model> [datasets...]
Writes features/jul-tuning/<model>/<ds>.npz and runs/dev/tuning-curve-<model>.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from jul import Choice, TypeSafeClient  # noqa: E402
from jul import tuning  # noqa: E402
from jul.types import options_of  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "minicpm5-2b"
DATASETS = sys.argv[2:] or ["agnews", "emotiondair", "banking77"]
SIZES = [25, 50, 100, 200, 500, 1000]
QUESTION = "Which single label best describes the input text?"
CACHE = ROOT / "features" / "jul-tuning" / MODEL


def load(ds, split):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"btzsc-{ds}" / f"{split}.jsonl")]
    return [r["text"] for r in rows], [r["label"] for r in rows]


def encode(client, ds):
    """(features, zero-shot scores, y) for train and val. Cached: the model pass is the slow part."""
    path = CACHE / f"{ds}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return {s: (d[f"{s}_x"], d[f"{s}_z"], d[f"{s}_y"]) for s in ("train", "val")}, list(d["labels"])

    labels = sorted(set(load(ds, "train")[1]))
    index = {l: i for i, l in enumerate(labels)}
    question = Choice(instructions=QUESTION, criteria={f"label_{i:03d}": l for i, l in enumerate(labels)})
    engine = client._engine_for(None)
    compiled = engine.compile("choice", question.instructions, options_of(question), None)

    out = {}
    for split in ("train", "val"):
        texts, ys = load(ds, split)
        t0 = time.perf_counter()
        x, z = [], []
        for i, text in enumerate(texts):
            scores, features, _ = engine.read(compiled, text, None)
            x.append(features)
            z.append(scores)
            if (i + 1) % 250 == 0:
                print(f"    {split} {i+1}/{len(texts)}  ({(i+1)/(time.perf_counter()-t0):.1f}/s)", flush=True)
        out[split] = (np.stack(x), np.stack(z), np.array([index[y] for y in ys]))
        print(f"    {split}: {len(texts)} in {time.perf_counter()-t0:.0f}s", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, labels=np.array(labels), **{
        f"{s}_{k}": v for s, arrays in out.items() for k, v in zip(("x", "z", "y"), arrays)})
    return out, labels


def head_on(features, y, k, seed=0):
    """Exactly what tuning.train fits, minus its own accept/reject gate."""
    inner, holdout = tuning.stratified_split(y, 0.2, seed + 1)
    mean, std = features[inner].mean(0), features[inner].std(0) + 1e-3
    norm = lambda a: (a - mean) / std
    W, b = tuning._fit_linear(norm(features[inner]), y[inner], norm(features[holdout]), y[holdout],
                              k, seed=seed)
    return {"W": W, "b": b, "mean": mean, "std": std, "temperature": np.array([1.0])}


client = TypeSafeClient(model=MODEL)
report = {}
for ds in DATASETS:
    print(f"[{MODEL}] {ds}", flush=True)
    data, labels = encode(client, ds)
    (xtr, ztr, ytr), (xva, zva, yva) = data["train"], data["val"]
    k = len(labels)
    zero_shot = float((zva.argmax(1) == yva).mean())
    rows = []
    for n in SIZES:
        if n > len(ytr) or n < 2 * k:
            continue
        keep, _ = tuning.stratified_split(ytr, 1 - n / len(ytr), seed=0)
        keep = keep[:n]
        if len(set(ytr[keep])) < k:
            continue
        payload = head_on(xtr[keep], ytr[keep], k)
        accuracy = float((tuning.apply(payload, xva).argmax(1) == yva).mean())
        gated, gate_report = tuning.train(xtr[keep], ytr[keep], ztr[keep], [str(i) for i in range(k)],
                                          ds, MODEL)
        rows.append({"n": int(len(keep)), "head_val": accuracy, "gain": accuracy - zero_shot,
                     "activated": gate_report.activated})
        print(f"  n={len(keep):<5} head_val={accuracy:.3f}  zero-shot={zero_shot:.3f}  "
              f"gain={accuracy-zero_shot:+.3f}  gate={'ON' if gate_report.activated else 'off'}",
              flush=True)
    report[ds] = {"classes": k, "zero_shot_val": zero_shot, "curve": rows}

out = ROOT / "runs" / "dev" / f"tuning-curve-{MODEL}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"model": MODEL, "report": report}, indent=2))
print(f"\nsaved {out.relative_to(ROOT)}")
