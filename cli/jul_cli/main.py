"""The `jul` command line. Built only on the public API of the library.

  jul setup                    # backend, weights and a first check for the default model
  jul ask choice "Which team should handle this ticket?" \
      -o billing:"payments, invoices" -o technical:"bugs, errors" \
      --state "I was charged twice" --model minicpm5-2b

  jul run questions.yaml --input tickets.jsonl --output answers.jsonl --context tickets
  jul context create tickets --description "Support tickets of an online bank" --examples sample.txt
  jul synth questions.yaml --seeds sample.jsonl --per-option 30 --output synth.jsonl
  jul autotune tickets --questions questions.yaml --labeled labeled.jsonl
  jul models

Every command prints the same JSON shape as the Jev API response.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from jul import Choice, Context, Noul, NoulCriteria, Score, TypeSafeClient
from jul.backbone import BACKENDS
from jul.presets import ALIASES, PRESETS

TYPES = {"choice": Choice, "noul": Noul, "score": Score}


# --- loading question files ---------------------------------------------------------------------

def load_questions(path: str | Path) -> dict:
    """A YAML or JSON file mapping a question name to `{type, instructions, criteria}`."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("PyYAML is needed for YAML question files: pip install 'jul[yaml]' "
                             "(or use a .json file)") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: expected a mapping of question name -> question")
    return {name: _question(name, spec) for name, spec in raw.items()}


def _question(name: str, spec: dict):
    kind = str(spec.get("type", "choice")).lower()
    if kind not in TYPES:
        raise SystemExit(f"question {name!r}: unknown type {kind!r} (choice, noul or score)")
    instructions = spec.get("instructions", "")
    criteria = spec.get("criteria")
    if kind == "noul":
        if isinstance(criteria, dict):
            return Noul(instructions=instructions,
                        criteria=NoulCriteria(true=criteria.get("true", ""), false=criteria.get("false", "")))
        return Noul(instructions=instructions)
    if kind == "score":
        return Score(instructions=instructions, criteria=list(criteria or []))
    return Choice(instructions=instructions, criteria=criteria or {})


def read_jsonl(path: str | Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_examples(path: str | Path) -> list[str]:
    """One text per line (.txt) or a `text` field per line (.jsonl)."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [r["text"] if isinstance(r, dict) else str(r) for r in read_jsonl(path)]
    return [l for l in path.read_text().splitlines() if l.strip()]


# --- commands -----------------------------------------------------------------------------------

def cmd_ask(a):
    pairs = [o.split(":", 1) if ":" in o else (o, "") for o in a.option or []]
    if a.kind == "choice":
        if len(pairs) < 2:
            raise SystemExit("choice needs at least two -o options")
        question = Choice(instructions=a.instructions, criteria={k.strip(): v.strip() for k, v in pairs})
    elif a.kind == "noul":
        criteria = {k.strip(): v.strip() for k, v in pairs}
        question = Noul(instructions=a.instructions,
                        criteria=NoulCriteria(true=criteria.get("true", ""), false=criteria.get("false", ""))
                        if criteria else None)
    else:
        levels = [v.strip() or k.strip() for k, v in pairs]
        if len(levels) < 2:
            raise SystemExit("score needs at least two -o levels, lowest first")
        question = Score(instructions=a.instructions, criteria=levels)

    client = TypeSafeClient(model=a.model, backend=a.backend, context=a.context, method=a.method)
    for state in a.state:
        t = time.perf_counter()
        response = client.system_one(state=state, questions={a.kind: question})
        out = response.as_dict()
        out["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
        print(json.dumps(out, ensure_ascii=False))
    client.close()


def cmd_run(a):
    questions = load_questions(a.questions)
    client = TypeSafeClient(model=a.model, backend=a.backend, context=a.context, method=a.method)
    out = open(a.output, "w") if a.output else sys.stdout
    n, t0 = 0, time.perf_counter()
    try:
        for row in read_jsonl(a.input):
            state = row.get("state", row) if isinstance(row, dict) else row
            response = client.system_one(state=state, questions=questions)
            record = response.as_dict()
            if isinstance(row, dict) and "id" in row:
                record["id"] = row["id"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if a.output and n % 50 == 0:
                print(f"  {n} rows", file=sys.stderr, flush=True)
    finally:
        if a.output:
            out.close()
    rate = n / max(time.perf_counter() - t0, 1e-9)
    print(f"{n} rows in {time.perf_counter() - t0:.1f}s ({rate:.1f}/s)"
          + (f" -> {a.output}" if a.output else ""), file=sys.stderr)
    client.close()


def cmd_context(a):
    if a.action == "list":
        names = Context.list_saved()
        print("\n".join(names) if names else "no saved context")
        return
    if a.action == "create":
        examples = read_examples(a.examples) if a.examples else []
        ctx = Context(description=a.description or "", examples=examples, name=a.name,
                      use_description=a.use_description)
        if examples and not a.lazy:
            # Compile the centers now so later calls pay nothing.
            client = TypeSafeClient(model=a.model, backend=a.backend, context=ctx)
            client.system_one(state=examples[0],
                              questions={"_": Choice(instructions="warm up", criteria={"a": "a", "b": "b"})},
                              context=ctx)
            client.close()
        path = ctx.save(a.name)
        print(f"saved context {a.name!r} ({len(examples)} examples) -> {path}")
        return
    if a.action == "show":
        ctx = Context.load(a.name)
        print(json.dumps({"name": ctx.name, "description": ctx.description,
                          "description_in_prompts": ctx.use_description, "examples": len(ctx.examples),
                          "centers": sorted(ctx.centers), "tuned_questions": len(ctx.heads),
                          "calibrated_questions": len(ctx.calibration)}, indent=2))
        return
    if a.action == "delete":
        print(f"deleted {a.name!r}" if Context.delete(a.name) else f"no context named {a.name!r}")


def read_labeled(path: str | Path, questions: dict, require_answers: bool = True) -> list[tuple]:
    """`(state, answers)` pairs from JSONL lines `{state|text, answers: {question: answer}}`."""
    labeled = []
    for row in read_jsonl(path):
        if isinstance(row, str):
            row = {"state": row}
        state = row.get("state", row.get("text"))
        answers = row.get("answers") or {k: v for k, v in row.items() if k in questions}
        if state is None or (require_answers and not answers):
            raise SystemExit("each labeled line needs a `state` (or `text`) and one answer per question")
        labeled.append((state, answers))
    return labeled


def cmd_autotune(a):
    questions = load_questions(a.questions)
    labeled = read_labeled(a.labeled, questions)
    try:
        ctx = Context.load(a.context)
    except FileNotFoundError:
        ctx = Context(name=a.context)
    client = TypeSafeClient(model=a.model, backend=a.backend, context=ctx)
    print(f"tuning on {len(labeled)} labeled examples with {a.model or 'minicpm5-2b'} ...", file=sys.stderr)
    reports = client.autotune(ctx, questions, labeled)
    for report in reports.values():
        print(report)
        print()
    client.close()


def cmd_synth(a):
    from jul.synth import mlx_writer, synthesize
    questions = load_questions(a.questions)
    seeds = read_labeled(a.seeds, questions, require_answers=False) if a.seeds else []
    if not seeds:
        print("no seeds: texts are written from the questions alone", file=sys.stderr)
    print(f"loading writer {a.writer} ...", file=sys.stderr)
    write = mlx_writer(a.writer, temperature=a.temperature)
    t0 = time.perf_counter()
    with open(a.output, "w") as out:
        def save(rows):
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            if rows:
                print(f"  {next(iter(rows[0]['answers'].items()))}: {len(rows)} texts", file=sys.stderr)
        rows = synthesize(questions, seeds, a.per_option, write, batch=a.batch, seed=a.seed, on_option=save)
    print(f"{len(rows)} texts in {time.perf_counter() - t0:.0f}s -> {a.output}", file=sys.stderr)


def cmd_models(a):
    if a.action == "add":
        return cmd_models_add(a)
    from jul.presets import fitted_presets
    try:
        from huggingface_hub import scan_cache_dir
        cached = {r.repo_id for r in scan_cache_dir().repos}
    except Exception:
        cached = set()
    rows = [(name, "mlx", p) for name, p in PRESETS.items()]
    rows += [(p.name, p.backend, p) for p in fitted_presets()]
    print(f"{'preset':<14} {'backend':<8} {'downloaded':<11} {'latency':<10} quality")
    for name, backend, p in rows:
        repo = p.repos.get(backend)
        mark = "yes" if repo in cached else "no"
        print(f"{name:<14} {backend:<8} {mark:<11} {p.latency_ms + ' ms':<10} {p.quality}")
    print("\naliases: " + ", ".join(f"{a} -> {t}" for a, t in ALIASES.items()))
    for name, backend, p in rows:
        print(f"\n{name} ({backend}): " + ", ".join(f"{b} {r}" for b, r in p.repos.items()))
        if p.method == "pointer":
            print("  method: pointer (decision model; format, head and temperature in its decision.json)")
        else:
            print("  formulations: " + ", ".join(f"{f.name}@layer{f.layer}" for f in p.formulations)
                  + f", tau={p.tau}, center={p.center}")
        if p.notes:
            print(f"  note: {p.notes}")
    print("\nAdd a model: jul models add <name> --repo <hf repo> [--backend mlx|torch]")


def cmd_models_add(a):
    from jul.calibrate import CalibrationError, calibrate
    if not a.name:
        raise SystemExit("models add needs a name")
    from jul.decision import spec_source
    source = spec_source(a.repo) if a.repo else None
    if source:
        from jul.backbone import resolve_backend
        from jul.presets import pointer_preset, save_preset
        preset = pointer_preset(a.name, source, resolve_backend(a.backend))
        path = save_preset(preset)
        print(f"{a.name}: decision model on {preset.backend}, read with its decision.json (nothing to fit) -> {path}")
        print(f"\nUse it: jul ask ... --model {a.name} --backend {preset.backend}")
        return
    try:
        preset = calibrate(a.name, repo=a.repo, backend=a.backend, data=a.data,
                           n_dev=a.n_dev, n_generic=a.n_generic)
    except CalibrationError as exc:
        raise SystemExit(f"error: {exc}") from exc
    c = preset.calibration
    print(f"\n{preset.name} on {c['backend']}: layers "
          + ", ".join(f"{f.name}@{f.layer}" for f in preset.formulations)
          + f", center {preset.center}, tau {preset.tau}")
    print(f"  dev accuracy {c['dev_accuracy']:.3f} ± {c['dev_accuracy_stderr']:.3f} (n={c['n_dev']}), "
          f"{c['dev_accuracy_task_center']:.3f} with a task center; ECE {c['dev_ece']:.3f}; "
          f"{c['latency_ms_p50']:.0f} ms per decision")
    print("  by set: " + ", ".join(f"{k} {v:.2f}" for k, v in c["dev_accuracy_by_set"].items()))
    print("  by center: " + ", ".join(f"{k} {v:.3f}" for k, v in c["dev_accuracy_by_center"].items()))
    print(f"\nUse it: jul ask ... --model {preset.name} --backend {c['backend']}")


def cmd_setup(a):
    from jul_cli.setup import run
    run(a.model, a.backend, install=not a.no_install, skip_check=a.no_check)


def cmd_lab(a):
    """The prototype's research commands (feature extraction, head training, benchmarks)."""
    from jul.lab.cli import main as lab_main
    lab_main(a.rest)


# --- parser -------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jul", description="Juste Un LLM - local typed decisions.")
    model_kw = dict(default=None, help=f"preset: {', '.join(PRESETS)} (aliases: {', '.join(ALIASES)})")
    backend_kw = dict(choices=list(BACKENDS), default=None,
                      help="default: $JUL_BACKEND, else mlx on Apple Silicon, else torch")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="install the backend, download the model, check a first answer")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--no-install", action="store_true", help="do not pip install a missing backend")
    s.add_argument("--no-check", action="store_true", help="skip the final test decision")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("ask", help="one typed question, answered now")
    s.add_argument("kind", choices=list(TYPES))
    s.add_argument("instructions")
    s.add_argument("-o", "--option", action="append",
                   help="choice: key[:description]; noul: true:meaning / false:meaning; "
                        "score: a level description, lowest first")
    s.add_argument("--state", action="append", required=True, help="the text to judge (repeatable)")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--context", help="name of a saved context")
    s.add_argument("--method", choices=["vector", "letters"])
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("run", help="answer a question file over a JSONL input")
    s.add_argument("questions")
    s.add_argument("--input", required=True)
    s.add_argument("--output")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--context")
    s.add_argument("--method", choices=["vector", "letters"])
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("context", help="create, list, show or delete a saved context")
    s.add_argument("action", choices=["create", "list", "show", "delete"])
    s.add_argument("name", nargs="?")
    s.add_argument("--description")
    s.add_argument("--examples", help=".txt (one per line) or .jsonl with a `text` field")
    s.add_argument("--use-description", action="store_true",
                   help="also put the description in the prompts; measured harmful on average, "
                        "so it is off unless you ask and measure on your own data")
    s.add_argument("--lazy", action="store_true", help="do not compute the centers now")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.set_defaults(fn=cmd_context)

    s = sub.add_parser("autotune", help="train a per-task head from labeled examples")
    s.add_argument("context", help="name of the context the head is stored in")
    s.add_argument("--questions", required=True)
    s.add_argument("--labeled", required=True, help="JSONL: {state, answers: {question: answer}}")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.set_defaults(fn=cmd_autotune)

    s = sub.add_parser("synth", help="write a synthetic labeled dataset (for a later autotune)")
    s.add_argument("questions")
    s.add_argument("--seeds", help="JSONL of real examples: {state|text, answers?}; answers optional")
    s.add_argument("--output", required=True, help="JSONL in the format of autotune --labeled")
    s.add_argument("--per-option", type=int, default=30, help="texts per option (default 30)")
    s.add_argument("--writer", default="qwen3.5-9b", help="preset or mlx-lm repo (default qwen3.5-9b)")
    s.add_argument("--batch", type=int, default=8, help="texts asked per call (default 8)")
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("models", help="list the presets, or fit one for a new model (add)")
    s.add_argument("action", nargs="?", choices=["list", "add"], default="list")
    s.add_argument("name", nargs="?", help="add: the preset name")
    s.add_argument("--repo", help="add: Hugging Face repo (default: the known repo for this name), or a local "
                                  "directory; one holding a decision.json is added as a decision model")
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--data", type=Path, help="add: calibration data dir (default: fetched into "
                                               "~/.jul/calibration-data)")
    s.add_argument("--n-dev", type=int, default=50, help="add: examples per dev set (default 50)")
    s.add_argument("--n-generic", type=int, default=200, help="add: generic texts (default 200)")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("lab", help="research commands")
    s.add_argument("rest", nargs=argparse.REMAINDER,
                   help="passed through: extract, evaluate, bench, summary, ask, decide")
    s.set_defaults(fn=cmd_lab)
    return p


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    if a.command == "context" and a.action != "list" and not a.name:
        raise SystemExit(f"context {a.action} needs a name")
    if a.command in {"ask", "run", "autotune"} or (
            a.command == "context" and a.action == "create" and a.examples and not a.lazy):
        from jul_cli.setup import require_setup
        require_setup(a.model, a.backend)
    a.fn(a)


if __name__ == "__main__":
    main()
