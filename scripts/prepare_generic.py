"""Training data for a generic (task-agnostic) head: BTZSC datasets that are neither in the Jev
benchmark, nor dev/validation sets, nor aggregates of other datasets.

Writes data/generic/<dataset>.jsonl: up to N balanced rows {text, target_index, labels}."""

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import get_dataset_config_names  # noqa: E402
from jev_benchmarks.config import BenchmarkConfig, load_config  # noqa: E402
from jev_benchmarks.data import load_examples  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BENCH = {"agnews", "banking77", "emotiondair"}
DEV = {"yahootopics", "empathetic", "massive", "financialphrasebank"}
AGGREGATES = {"all", "emotion", "intent", "sentiment", "topic"}

cfg = load_config(ROOT / "external" / "jev-benchmarks" / "configs" / "pilot-v1.yaml")
names = get_dataset_config_names(cfg.raw["dataset"]["repository"], revision=cfg.raw["dataset"]["revision"])
train_sets = sorted(set(names) - BENCH - DEV - AGGREGATES)
print("training datasets:", train_sets, flush=True)

out = ROOT / "data" / "generic"
out.mkdir(parents=True, exist_ok=True)
for ds in train_sets:
    if (out / f"{ds}.jsonl").exists():
        continue
    raw = copy.deepcopy(cfg.raw)
    raw["seed"] = 11
    raw["dataset"]["samples_per_dataset"] = N
    raw["dataset"]["datasets"] = [{"name": ds, "task": "generic"}]
    try:
        rows = load_examples(BenchmarkConfig(raw=raw, path=cfg.path))
    except Exception as exc:  # a malformed config should not stop the others
        print(f"  {ds}: skipped ({type(exc).__name__}: {exc})", flush=True)
        continue
    with open(out / f"{ds}.jsonl", "w") as f:
        for e in rows:
            f.write(json.dumps({"text": e.text, "target_index": e.target_index, "labels": list(e.labels)}, ensure_ascii=False) + "\n")
    print(f"  {ds}: {len(rows)} rows, {len(rows[0].labels)} labels, e.g. {rows[0].labels[0]!r}", flush=True)
