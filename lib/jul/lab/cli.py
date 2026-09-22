"""quickfast CLI.

  python -m quickfast.cli extract  --task tasks/emotion.json --model minicpm5-2b [--limit N]
  python -m quickfast.cli evaluate --task tasks/emotion.json --model minicpm5-2b [--holdout LABEL]
  python -m quickfast.cli bench    --task tasks/emotion.json --model minicpm5-2b
  python -m quickfast.cli decide   --run runs/emotion/minicpm5-2b/hybrid-plain "some text"
  python -m quickfast.cli ask choice "Route this ticket" -o "billing:payments" -o "technical:bugs" -i "charged twice"
  python -m quickfast.cli ask noul   "Does the message report a bug?" -i "the app crashes on export"
  python -m quickfast.cli ask score  "How frustrated is the customer?" -o "Calm" -o "Annoyed" -o "Angry" -i "..."
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

from . import evaluate, features
from ..backbone import MODELS, Backbone
from .task import Task

ROOT = Path.cwd()  # research outputs land under the directory the command is run from


def _run_dir(task: Task, model: str, holdout: str | None) -> Path:
    return ROOT / "runs" / task.name / (model + (f"-holdout-{holdout}" if holdout else ""))


def cmd_extract(a):
    task = Task.load(a.task)
    for model in a.model:
        print(f"[extract] {model}", flush=True)
        features.extract(Backbone(model), task, features.feature_dir(ROOT / "features", task, model), limit=a.limit)


def cmd_evaluate(a):
    task = Task.load(a.task)
    for model in a.model:
        run_dir = _run_dir(task, model, a.holdout)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.task, run_dir / "task.json")
        evaluate.run(features.feature_dir(ROOT / "features", task, model), run_dir, holdout=a.holdout,
                     tied=not a.untied)


def _latency(fn, texts, warmup=5):
    for t in texts[:warmup]:
        fn(t)
    ts = []
    for t in texts:
        s = time.perf_counter()
        fn(t)
        ts.append((time.perf_counter() - s) * 1000)
    ts.sort()
    return statistics.median(ts), ts[int(0.9 * (len(ts) - 1))]


def cmd_bench(a):
    from .decider import Decider, LogitDecider
    from .task import Prompts

    task = Task.load(a.task)
    texts, _ = task.load_split("test")
    texts = texts[: a.n]
    out = {}
    for model in a.model:
        bb = Backbone(model)
        run_dir = _run_dir(task, model, None)
        rows = []
        no_cache = Prompts(bb, task, use_prefix_cache=False)
        rows.append(("A  logits (no prefix cache)", lambda t: no_cache.options.run(t, logits=True)))
        rows.append(("A  logits", LogitDecider(bb, task).decide))
        for variant in ("probe-plain", "probe-options", "hybrid-plain", "hybrid-options"):
            if (run_dir / variant).exists():
                d = Decider(run_dir / variant, backbone=bb)
                rows.append((f"{variant} (L≤{max(d.layers)}/{bb.n_layers - 1})", d.decide))
        print(f"\n=== latency {model}  ({len(texts)} texts, ms/decision) ===")
        out[model] = {}
        for name, fn in rows:
            med, p90 = _latency(fn, texts)
            out[model][name] = {"median_ms": med, "p90_ms": p90}
            print(f"  {name:<44} median {med:6.1f}   p90 {p90:6.1f}")
    path = ROOT / "runs" / task.name / "latency.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


BENCH_NAME = {"A  logits": "A  logits", "B  probe/plain": "probe-plain", "B  probe/options": "probe-options",
              "B+C hybrid/plain": "hybrid-plain", "B+C hybrid/options": "hybrid-options"}


def cmd_summary(a):
    """One table: accuracy, mean confidence and ECE (raw and calibrated), latency — per model and variant."""
    task = Task.load(a.task)
    latency = json.loads((ROOT / "runs" / task.name / "latency.json").read_text())
    lines = ["| model | variant | accuracy | mean conf (raw → cal) | ECE (raw → cal) | latency median / p90 |",
             "|---|---|---:|---:|---:|---:|"]
    for model in a.model:
        report = json.loads((_run_dir(task, model, None) / "report.json").read_text())
        for name, r in report["results"].items():
            key = next(v for k, v in BENCH_NAME.items() if name.startswith(k))
            lat = next((v for k, v in latency.get(model, {}).items() if k == key or k.startswith(key + " (L")), None)
            lat_s = f"{lat['median_ms']:.1f} / {lat['p90_ms']:.1f} ms" if lat else "–"
            lines.append(f"| {model} | {name.strip()} | {r['cal']['acc']:.3f} | "
                         f"{r['raw']['conf']:.3f} → {r['cal']['conf']:.3f} | "
                         f"{r['raw']['ece']:.3f} → {r['cal']['ece']:.3f} | {lat_s} |")
    table = "\n".join(lines)
    (ROOT / "runs" / task.name / "summary.md").write_text(table + "\n")
    print(table)


def cmd_ask(a):
    from .primitives import QuickFast

    qf = QuickFast(a.model)
    pairs = [tuple(o.split(":", 1)) if ":" in o else (o, "") for o in a.option or []]
    if a.kind == "choice":
        ask = lambda x: qf.choice(a.instructions, dict(pairs), x)
    elif a.kind == "noul":
        ask = lambda x: qf.noul(a.instructions, x, dict(pairs) or None)
    else:
        ask = lambda x: qf.score(a.instructions, [o for o in a.option], x)
    for x in a.input:
        s = time.perf_counter()
        out = ask(x)
        print(json.dumps(out, ensure_ascii=False), f"  ({(time.perf_counter() - s) * 1000:.1f} ms)")


def cmd_decide(a):
    from .decider import Decider

    d = Decider(a.run)
    for name, desc in (tuple(x.split(":", 1)) if ":" in x else (x, "") for x in a.add_label or []):
        d.add_label(name, desc)
    for text in a.text:
        s = time.perf_counter()
        probs = d.decide(text)
        ms = (time.perf_counter() - s) * 1000
        print(f"\n{text!r}  ({ms:.1f} ms)")
        for label, p in probs.items():
            print(f"  {label:<14} {p:6.3f}  {'█' * round(p * 40)}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="jul lab")
    sub = p.add_subparsers(required=True)
    models = dict(nargs="+", default=["minicpm5-2b"], help=f"aliases: {', '.join(MODELS)} (or any repo of the backend)")

    s = sub.add_parser("extract")
    s.add_argument("--task", required=True)
    s.add_argument("--model", **models)
    s.add_argument("--limit", type=int)
    s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("evaluate")
    s.add_argument("--task", required=True)
    s.add_argument("--model", **models)
    s.add_argument("--holdout", help="label excluded from training, to measure zero-shot on new labels")
    s.add_argument("--untied", action="store_true", help="separate projections for queries and labels")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("bench")
    s.add_argument("--task", required=True)
    s.add_argument("--model", **models)
    s.add_argument("-n", type=int, default=100)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("summary")
    s.add_argument("--task", required=True)
    s.add_argument("--model", **models)
    s.set_defaults(fn=cmd_summary)

    s = sub.add_parser("ask", help="zero-shot typed question: choice, noul or score")
    s.add_argument("kind", choices=["choice", "noul", "score"])
    s.add_argument("instructions")
    s.add_argument("-o", "--option", action="append",
                   help="choice: name[:description]; noul: yes:meaning / no:meaning; score: level description, lowest first")
    s.add_argument("-i", "--input", action="append", required=True)
    s.add_argument("--model", default="minicpm5-2b")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("decide")
    s.add_argument("--run", required=True)
    s.add_argument("--add-label", action="append", help="name[:description], hybrid runs only")
    s.add_argument("text", nargs="+")
    s.set_defaults(fn=cmd_decide)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
