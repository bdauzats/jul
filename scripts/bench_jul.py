"""Final benchmark: jul against Jev on the BTZSC pilot v1, through the public API only.

Same 300 examples as the published run (manifest hash checked), same label descriptions, same
question, same metric code imported from the benchmark package. Jev and GLiNER numbers come from the
published report; their per-example predictions are not public.

Three variants, reported separately because only the first is a fair comparison:

  zero-shot   no task data at all, like Jev.
  context     a Context holding 50 unlabeled texts of the task (task center). NOT comparable: Jev
              gets nothing. The description is left off, measured harmful on average.
  tuned       a per-task head trained on 1000 labeled rows disjoint from the benchmark. NOT comparable.
  tuned-hybrid  the same, the head reading the vectors and the TF-IDF of the text (features="hybrid",
              needs jul[tune]). Not run by default.

Everything goes through `jul.TypeSafeClient`, never through the engine, so this measures the library
a user would install.

Usage: [JUL_BACKEND=onnx] scripts/bench_jul.py <model> [variants, comma-separated]
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lib"))

from jev_benchmarks.io import read_jsonl, sha256_file  # noqa: E402
from jev_benchmarks.metrics import score_predictions  # noqa: E402
from jev_benchmarks.models import Prediction  # noqa: E402

from jul import Choice, Context, TypeSafeClient  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "minicpm5-2b"
VARIANTS = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["zero-shot", "context", "tuned"])
BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
PUBLISHED = BENCH / "results" / "reports" / "btzsc-pilot-v1.json"
DATASETS = {"agnews": "AG News", "banking77": "Banking77", "emotiondair": "Emotion"}
QUESTION = "Which single label best describes the input text?"
N_CONTEXT = 50          # 10 is enough, 50 is the best of the sizes tried
OUT = ROOT / "runs" / "jev-bench-jul"

digest = sha256_file(MANIFEST)
assert digest == MANIFEST_SHA256, f"manifest changed: {digest}"
manifest = read_jsonl(MANIFEST)
rows_by_ds = {ds: [r for r in manifest if r["dataset"] == ds] for ds in DATASETS}
print(f"manifest verified: {len(manifest)} rows, {MANIFEST_SHA256[:12]}...")


def training_rows(ds):
    """The 1000 disjoint labeled rows prepared by scripts/prepare_btzsc.py."""
    return [json.loads(l) for l in open(ROOT / "data" / f"btzsc-{ds}" / "train.jsonl")]


# The benchmark rows must never appear in the training data.
bench_texts = {r["text"] for r in manifest}
for ds in DATASETS:
    overlap = bench_texts & {r["text"] for r in training_rows(ds)}
    assert not overlap, f"{ds}: {len(overlap)} training rows also in the benchmark"
print("training data verified disjoint from the benchmark")

client = TypeSafeClient(model=MODEL)
results, notes = {}, {}

for variant in VARIANTS:
    predictions = []
    for ds, rows in rows_by_ds.items():
        labels = list(rows[0]["labels"])
        criteria = {f"label_{i:03d}": label for i, label in enumerate(labels)}
        keys = list(criteria)
        questions = {"label": Choice(instructions=QUESTION, criteria=criteria)}

        context = None
        if variant in ("context", "tuned", "tuned-hybrid"):
            train = training_rows(ds)
            context = Context(examples=[r["text"] for r in train][:N_CONTEXT], use_description=False)
        if variant.startswith("tuned"):
            index = {label: key for key, label in criteria.items()}
            labeled = [(r["text"], {"label": index[r["label"]]}) for r in training_rows(ds)]
            t0 = time.perf_counter()
            features = "hybrid" if variant == "tuned-hybrid" else "vector"
            report = client.autotune(context, questions, labeled, save=False, features=features)["label"]
            notes[f"{ds}/{variant}"] = {"activated": report.activated, "reason": report.reason,
                                    "seconds": round(time.perf_counter() - t0, 1)}
            print(f"  [{ds}] head: {'ACTIVE' if report.activated else 'refused'} — {report.reason}",
                  flush=True)

        for r in rows:
            start = time.perf_counter()
            answer = client.system_one(state=r["text"], questions=questions, context=context).choices["label"]
            latency = time.perf_counter() - start
            probabilities = tuple(float(answer.probabilities[k]) for k in keys)
            predictions.append(Prediction(
                experiment_id="jul-final", backend=f"jul-{variant}", model_requested=MODEL,
                model_resolved=MODEL, dataset=ds, example_id=r["example_id"],
                target_index=r["target_index"], predicted_index=int(np.argmax(probabilities)),
                labels=tuple(r["labels"]), probabilities=probabilities,
                latency_seconds=latency, probability_sum_raw=float(sum(probabilities))))
        print(f"  [{ds}] {variant}: done", flush=True)
    results[variant] = predictions

published = json.loads(PUBLISHED.read_text())["results"]
OUT.mkdir(parents=True, exist_ok=True)


def row(name, scored, latency_ms):
    mean = lambda k: np.mean([scored[d][k] for d in DATASETS])
    return (f"| {name} | " + " | ".join(f"{scored[d]['accuracy']:.2f} / {scored[d]['ece']:.3f}"
                                        for d in DATASETS)
            + f" | {mean('accuracy'):.3f} | {mean('ece'):.3f} | {latency_ms:.0f} ms |")


lines = [f"# jul {MODEL} vs Jev — BTZSC pilot v1, 300 examples",
         "",
         f"Manifest `{MANIFEST_SHA256[:12]}…`, 100 rows per dataset. Cells: accuracy / ECE.",
         "Only the zero-shot row is comparable to Jev: the other two receive task data, Jev receives none.",
         "",
         "| Model | " + " | ".join(f"{t} acc / ECE" for t in DATASETS.values())
         + " | Mean acc | Mean ECE | p50 latency |",
         "|---|" + "---:|" * (len(DATASETS) + 3)]
for name, key in (("Jev (published)", "jev"), ("GLiNER2.5 (published)", "gliner")):
    m = published[key]
    lines.append(row(name, m, np.mean([m[d]["latency_p50_seconds"] for d in DATASETS]) * 1000))
for variant, preds in results.items():
    by = defaultdict(list)
    for p in preds:
        by[p.dataset].append(p)
    lines.append(row(f"**jul {MODEL} — {variant}**", {d: score_predictions(by[d]) for d in DATASETS},
                     np.median([p.latency_seconds for p in preds]) * 1000))
if notes:
    lines += ["", "Tuned heads:", ""] + [f"- `{k}`: {v['reason']} ({v['seconds']}s)" for k, v in notes.items()]

report = "\n".join(lines) + "\n"
(OUT / f"report-{MODEL}.md").write_text(report)
for variant, preds in results.items():
    with open(OUT / f"{MODEL}-{variant}.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p.to_dict() if hasattr(p, "to_dict") else p.__dict__, default=str) + "\n")
print("\n" + report)
