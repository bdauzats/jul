"""Development datasets for tuning the vector method: BTZSC sets NOT used by the Jev benchmark.

Writes data/dev/<dataset>.jsonl (labeled, balanced) and data/dev/<dataset>.unlabeled.jsonl (texts only)."""

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_benchmarks.config import BenchmarkConfig, load_config  # noqa: E402
from jev_benchmarks.data import load_examples  # noqa: E402

DEV = {"yahootopics": "topic", "empathetic": "emotion", "massive": "intent", "financialphrasebank": "sentiment"}
N_LABELED, N_UNLABELED = 200, 200

cfg = load_config(ROOT / "external" / "jev-benchmarks" / "configs" / "pilot-v1.yaml")
raw = copy.deepcopy(cfg.raw)
raw["seed"] = 7
raw["dataset"]["samples_per_dataset"] = N_LABELED + N_UNLABELED
raw["dataset"]["datasets"] = [{"name": n, "task": t} for n, t in DEV.items()]
examples = load_examples(BenchmarkConfig(raw=raw, path=cfg.path))

by_ds = defaultdict(lambda: defaultdict(list))
for e in examples:
    by_ds[e.dataset][e.target_index].append(e)
out = ROOT / "data" / "dev"
out.mkdir(parents=True, exist_ok=True)
for ds, classes in by_ds.items():
    queues = [list(q) for _, q in sorted(classes.items())]
    picked = []
    while any(queues):
        for q in queues:
            if q:
                picked.append(q.pop(0))
    labeled, unlabeled = picked[:N_LABELED], picked[N_LABELED:]
    with open(out / f"{ds}.jsonl", "w") as f:
        for e in labeled:
            f.write(json.dumps({"text": e.text, "target_index": e.target_index, "labels": list(e.labels)}, ensure_ascii=False) + "\n")
    with open(out / f"{ds}.unlabeled.jsonl", "w") as f:
        for e in unlabeled:
            f.write(json.dumps({"text": e.text}, ensure_ascii=False) + "\n")
    print(f"{ds}: labeled {len(labeled)}  unlabeled {len(unlabeled)}  labels {len(labeled[0].labels)}  e.g. {labeled[0].labels[0]!r}")
