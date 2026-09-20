"""The `jul` command line. Built only on the public API of the library.

  jul ask choice "Which team should handle this ticket?" \
      -o billing:"payments, invoices" -o technical:"bugs, errors" \
      --state "I was charged twice" --model minicpm5-2b

  jul run questions.yaml --input tickets.jsonl --output answers.jsonl --context tickets
  jul context create tickets --description "Support tickets of an online bank" --examples sample.txt
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
        except ImportError:
            raise SystemExit("PyYAML is needed for YAML question files: pip install 'jul[yaml]' "
                             "(or use a .json file)")
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

    client = TypeSafeClient(model=a.model, context=a.context, method=a.method)
    for state in a.state:
        t = time.perf_counter()
        response = client.system_one(state=state, questions={a.kind: question})
        out = response.as_dict()
        out["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
        print(json.dumps(out, ensure_ascii=False))
    client.close()


def cmd_run(a):
    questions = load_questions(a.questions)
    client = TypeSafeClient(model=a.model, context=a.context, method=a.method)
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
            client = TypeSafeClient(model=a.model, context=ctx)
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


def cmd_autotune(a):
    questions = load_questions(a.questions)
    labeled = []
    for row in read_jsonl(a.labeled):
        state = row.get("state", row.get("text"))
        answers = row.get("answers") or {k: v for k, v in row.items() if k in questions}
        if state is None or not answers:
            raise SystemExit("each labeled line needs a `state` (or `text`) and one answer per question")
        labeled.append((state, answers))
    try:
        ctx = Context.load(a.context)
    except FileNotFoundError:
        ctx = Context(name=a.context)
    client = TypeSafeClient(model=a.model, context=ctx)
    print(f"tuning on {len(labeled)} labeled examples with {a.model or 'minicpm5-2b'} ...", file=sys.stderr)
    reports = client.autotune(ctx, questions, labeled)
    for report in reports.values():
        print(report)
        print()
    client.close()


def cmd_models(a):
    try:
        from huggingface_hub import scan_cache_dir
        cached = {r.repo_id for r in scan_cache_dir().repos}
    except Exception:
        cached = set()
    print(f"{'preset':<14} {'downloaded':<11} {'latency':<10} quality")
    for name, p in PRESETS.items():
        mark = "yes" if p.repo in cached else "no"
        print(f"{name:<14} {mark:<11} {p.latency_ms + ' ms':<10} {p.quality}")
    print(f"\naliases: " + ", ".join(f"{a} -> {t}" for a, t in ALIASES.items()))
    for name, p in PRESETS.items():
        print(f"\n{name}: {p.repo}")
        print(f"  formulations: " + ", ".join(f"{f.name}@layer{f.layer}" for f in p.formulations)
              + f", tau={p.tau}, center={p.center}")
        if p.notes:
            print(f"  note: {p.notes}")


def cmd_lab(a):
    """The prototype's research commands (feature extraction, head training, benchmarks)."""
    from jul.lab.cli import main as lab_main
    lab_main(a.rest)


# --- parser -------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jul", description="Juste Un LLM - local typed decisions.")
    model_kw = dict(default=None, help=f"preset: {', '.join(PRESETS)} (aliases: {', '.join(ALIASES)})")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ask", help="one typed question, answered now")
    s.add_argument("kind", choices=list(TYPES))
    s.add_argument("instructions")
    s.add_argument("-o", "--option", action="append",
                   help="choice: key[:description]; noul: true:meaning / false:meaning; "
                        "score: a level description, lowest first")
    s.add_argument("--state", action="append", required=True, help="the text to judge (repeatable)")
    s.add_argument("--model", **model_kw)
    s.add_argument("--context", help="name of a saved context")
    s.add_argument("--method", choices=["vector", "letters"])
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("run", help="answer a question file over a JSONL input")
    s.add_argument("questions")
    s.add_argument("--input", required=True)
    s.add_argument("--output")
    s.add_argument("--model", **model_kw)
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
    s.set_defaults(fn=cmd_context)

    s = sub.add_parser("autotune", help="train a per-task head from labeled examples")
    s.add_argument("context", help="name of the context the head is stored in")
    s.add_argument("--questions", required=True)
    s.add_argument("--labeled", required=True, help="JSONL: {state, answers: {question: answer}}")
    s.add_argument("--model", **model_kw)
    s.set_defaults(fn=cmd_autotune)

    s = sub.add_parser("models", help="available presets, whether downloaded, indicative latency")
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
    a.fn(a)


if __name__ == "__main__":
    main()
