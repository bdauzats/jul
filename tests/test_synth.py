import itertools

from jul import Choice, Noul
from jul.synth import parse_lines, synthesize


def fake_writer():
    counter = itertools.count()

    def write(prompt):
        if prompt.startswith("Here are real texts"):
            return "Short customer messages, informal."
        key = prompt.split('the right answer is clearly "')[1].split('"')[0]
        # one duplicate per batch, to check it is dropped
        return "\n".join([f"1. text about {key} number {next(counter)}"] * 2
                         + [f"- another {key} text {next(counter)}"])
    return write


def test_synthesize_writes_per_option_rows_in_autotune_format():
    questions = {"team": Choice(instructions="Which team?", criteria={"billing": "payments", "tech": "bugs"}),
                 "urgent": Noul(instructions="Is it urgent?")}
    seeds = [("I was charged twice", {"team": "billing"}), ("app crashes on login", {})]
    rows = synthesize(questions, seeds, per_option=5, write=fake_writer(), batch=3)
    by = {}
    for r in rows:
        (q, k), = r["answers"].items()
        by.setdefault((q, k), []).append(r["state"])
    assert set(by) == {("team", "billing"), ("team", "tech"), ("urgent", "true"), ("urgent", "false")}
    assert all(len(v) == 5 and len(set(v)) == 5 for v in by.values())
    assert all(r["synthetic"] for r in rows)


def test_seeds_reach_the_prompt_of_their_option_only():
    prompts = []

    def write(p):
        prompts.append(p)
        return "a brand new generated text"
    questions = {"team": Choice(instructions="Which team?", criteria=["billing", "tech"])}
    synthesize(questions, [("I was charged twice", {"team": "billing"})], 1, write)
    billing = next(p for p in prompts if 'clearly "billing"' in p)
    tech = next(p for p in prompts if 'clearly "tech"' in p)
    assert "I was charged twice" in billing and "I was charged twice" not in tech


def test_parse_lines():
    assert parse_lines('<think>x</think>\n1. "Hello there, friend"\n- ok\n• Where is my card?') == [
        "Hello there, friend", "Where is my card?"]
