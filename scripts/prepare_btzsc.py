"""Supervised variant of the Jev benchmark: train/val from BTZSC test rows disjoint from the 300 benchmark
rows, test = exactly the benchmark rows. Writes data/btzsc-<dataset>/ and tasks/btzsc-<dataset>.json."""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_benchmarks.config import BenchmarkConfig, load_config  # noqa: E402
from jev_benchmarks.data import load_examples, prepare_manifest  # noqa: E402
from jev_benchmarks.io import read_jsonl  # noqa: E402

CONFIG = ROOT / "external" / "jev-benchmarks" / "configs" / "pilot-v1.yaml"

ap = argparse.ArgumentParser()
ap.add_argument("--train", type=int, default=1000)
ap.add_argument("--val", type=int, default=200)
args = ap.parse_args()

cfg = load_config(CONFIG)
manifest = read_jsonl(prepare_manifest(cfg))
bench_ids = {r["example_id"] for r in manifest}
raw = copy.deepcopy(cfg.raw)
raw["seed"] = cfg.seed + 2000
raw["dataset"]["samples_per_dataset"] = 100_000  # every valid row, in balanced order
pool = load_examples(BenchmarkConfig(raw=raw, path=cfg.path))

by_class = defaultdict(lambda: defaultdict(list))
for e in pool:
    if e.example_id not in bench_ids:
        by_class[e.dataset][e.target_index].append(e)

question = cfg.raw["models"]["jev"]["question"]
for dataset, classes in by_class.items():
    queues = [list(q) for _, q in sorted(classes.items())]
    picked = []
    while len(picked) < args.train + args.val and any(queues):
        for q in queues:
            if q and len(picked) < args.train + args.val:
                picked.append(q.pop(0))
    val, train = picked[: args.val], picked[args.val :]
    test = [r for r in manifest if r["dataset"] == dataset]
    labels = list(test[0]["labels"])
    out = ROOT / "data" / f"btzsc-{dataset}"
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", [(e.text, e.target_index) for e in train]),
                        ("val", [(e.text, e.target_index) for e in val]),
                        ("test", [(r["text"], r["target_index"]) for r in test])):
        with open(out / f"{split}.jsonl", "w") as f:
            for text, y in rows:
                f.write(json.dumps({"text": text, "label": labels[y]}, ensure_ascii=False) + "\n")
    task = {"name": f"btzsc-{dataset}", "data_dir": f"data/btzsc-{dataset}", "instruction": question,
            "labels": [{"name": l} for l in labels]}
    task_path = ROOT / "tasks" / f"btzsc-{dataset}.json"
    task_path.parent.mkdir(parents=True, exist_ok=True)   # a fresh clone has no tasks/ directory
    task_path.write_text(json.dumps(task, indent=2, ensure_ascii=False))
    print(f"{dataset}: train {len(train)}  val {len(val)}  test {len(test)}  labels {len(labels)}")
