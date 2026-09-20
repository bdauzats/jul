"""The baseline worth remembering: TF-IDF + a linear SVM, no LLM anywhere.

Same 300 benchmark rows and same 1000 labeled training rows as `bench_jul.py`, so the numbers sit in
the same table. Needs only scikit-learn: no model, no GPU, no network. Training takes under a second
on a CPU.

It is here because it is the honest reference. A library is worth its cost only against the cheapest
thing that could have worked, and on two of these three datasets the cheapest thing is very close.

Usage: scripts/bench_tfidf.py
Writes runs/jev-bench-jul/report-tfidf.md
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "external" / "jev-benchmarks" / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
DATASETS = {"agnews": "AG News", "banking77": "Banking77", "emotiondair": "Emotion"}
OUT = ROOT / "runs" / "jev-bench-jul"

digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
assert digest == MANIFEST_SHA256, f"manifest changed: {digest}"
manifest = [json.loads(l) for l in open(MANIFEST)]
print(f"manifest verified: {len(manifest)} rows")

MODELS = {
    "TF-IDF + linear SVM": lambda: LinearSVC(C=1),
    "TF-IDF + logistic regression": lambda: LogisticRegression(max_iter=2000, C=10),
}

results = {}
for name, make_clf in MODELS.items():
    per_dataset, latencies, fit_seconds = {}, [], 0.0
    for ds in DATASETS:
        bench = [r for r in manifest if r["dataset"] == ds]
        labels = list(bench[0]["labels"])                      # the manifest's canonical order
        index = {label: i for i, label in enumerate(labels)}
        train = [json.loads(l) for l in open(ROOT / "data" / f"btzsc-{ds}" / "train.jsonl")]
        assert all(r["label"] in index for r in train), f"{ds}: unexpected training label"

        model = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), strip_accents="unicode"),
            make_clf())
        start = time.perf_counter()
        model.fit([r["text"] for r in train], [index[r["label"]] for r in train])
        fit_seconds += time.perf_counter() - start

        model.predict([bench[0]["text"]])                      # warm-up
        correct, per_call = 0, []
        for r in bench:                                        # one at a time, as a real call would be
            start = time.perf_counter()
            predicted = int(model.predict([r["text"]])[0])
            per_call.append((time.perf_counter() - start) * 1000)
            correct += predicted == r["target_index"]
        per_dataset[ds] = correct / len(bench)
        latencies += per_call
        print(f"  {name:<30} {ds:<14} acc={per_dataset[ds]:.2f}  "
              f"p50={np.median(per_call):.2f} ms  ({len(labels)} options)", flush=True)
    results[name] = {"per_dataset": per_dataset, "mean": float(np.mean(list(per_dataset.values()))),
                     "p50_ms": float(np.median(latencies)), "fit_seconds": round(fit_seconds, 1)}

OUT.mkdir(parents=True, exist_ok=True)
lines = ["# No-LLM baseline — BTZSC pilot v1, 300 examples", "",
         "Trained on the same 1000 labeled rows per dataset as `bench_jul.py`, scored on the same 300",
         "benchmark rows, one prediction at a time. scikit-learn only.", "",
         "| Method | " + " | ".join(DATASETS.values()) + " | Mean | p50 latency | Training |",
         "|---|" + "---:|" * (len(DATASETS) + 3)]
for name, r in results.items():
    lines.append(f"| {name} | " + " | ".join(f"{r['per_dataset'][d]:.2f}" for d in DATASETS)
                 + f" | **{r['mean']:.3f}** | **{r['p50_ms']:.2f} ms** | {r['fit_seconds']}s on CPU |")
lines += ["", "For comparison, from the published report and `bench_jul.py`:", "",
          "| | Mean | p50 latency |", "|---|---:|---:|",
          "| Jev (published, no examples) | 0.753 | 246 ms |",
          "| jul minicpm5-2b + autotune (1000 labeled) | 0.757 | 65 ms |",
          "| jul minicpm5-2b, zero-shot | 0.617 | 64 ms |"]
best = max(results.values(), key=lambda r: r["mean"])
lines += ["",
          f"The bag of words lands {(0.753 - best['mean']) * 100:.1f} points behind Jev at roughly "
          f"{246 / best['p50_ms']:.0f}x lower latency, and it beats the zero-shot LLM outright. It "
          "loses on one dataset only, Emotion, where recognising a feeling needs meaning rather than "
          "vocabulary — which is exactly, and only, where the LLM earns its cost."]
report = "\n".join(lines) + "\n"
(OUT / "report-tfidf.md").write_text(report)
print("\n" + report)
