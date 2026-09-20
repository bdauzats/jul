"""Per-preset non-regression: accuracy on a frozen dev sample, with a tolerance, plus latency.

The sample is 30 rows each of two dev datasets (never the Jev benchmark). Baselines in
`fixtures/baseline.json` were measured with this code; the tolerance is wide because 30 rows carry an
error of roughly +/- 8 points, so this catches a broken method, not a small drift.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from jul import Choice, Context, TypeSafeClient

FIXTURES = Path(__file__).parent / "fixtures"
QUESTION = "Which single label best describes the input text?"
ACCURACY_TOLERANCE = 0.15
# The fixture ships 30 unlabeled texts per dataset, just under the advised minimum.
pytestmark = pytest.mark.filterwarnings("ignore:Context has")
LATENCY_FACTOR = 4          # generous: a loaded machine must not turn this red


def fixture():
    return json.loads((FIXTURES / "dev_sample.json").read_text())


def baseline():
    return json.loads((FIXTURES / "baseline.json").read_text())


def evaluate(client, dataset) -> tuple[float, float]:
    criteria = {f"label_{i:03d}": label for i, label in enumerate(dataset["labels"])}
    questions = {"label": Choice(instructions=QUESTION, criteria=criteria)}
    context = Context(examples=dataset["unlabeled"], use_description=False)
    keys = list(criteria)
    client.system_one(state=dataset["rows"][0]["text"], questions=questions, context=context)

    correct, latencies = 0, []
    for row in dataset["rows"]:
        start = time.perf_counter()
        answer = client.system_one(state=row["text"], questions=questions, context=context).choices["label"]
        latencies.append((time.perf_counter() - start) * 1000)
        correct += keys.index(answer.choice) == row["target_index"]
    return correct / len(dataset["rows"]), float(np.median(latencies))


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["minicpm5-2b", "qwen3.5-9b"])
def test_accuracy_and_latency_hold(preset):
    expected = baseline()[preset]
    client = TypeSafeClient(model=preset)
    try:
        for dataset in fixture():
            accuracy, p50 = evaluate(client, dataset)
            reference = expected[dataset["dataset"]]
            assert accuracy >= reference["acc"] - ACCURACY_TOLERANCE, (
                f"{preset} on {dataset['dataset']}: {accuracy:.3f} vs {reference['acc']:.3f} expected")
            assert p50 <= reference["p50_ms"] * LATENCY_FACTOR, (
                f"{preset} on {dataset['dataset']}: {p50:.0f} ms vs {reference['p50_ms']:.0f} ms expected")
    finally:
        client.close()


@pytest.mark.slow
def test_a_task_center_is_computed_once_and_reused():
    dataset = fixture()[0]
    client = TypeSafeClient(model="minicpm5-2b")
    context = Context(examples=dataset["unlabeled"], use_description=False)
    criteria = {f"label_{i:03d}": label for i, label in enumerate(dataset["labels"])}
    questions = {"label": Choice(instructions=QUESTION, criteria=criteria)}

    assert context.centers == {}
    client.system_one(state=dataset["rows"][0]["text"], questions=questions, context=context)
    assert len(context.centers) == 2          # one per formulation of the preset
    centers = {k: v.copy() for k, v in context.centers.items()}

    client.system_one(state=dataset["rows"][1]["text"], questions=questions, context=context)
    assert all(np.array_equal(centers[k], v) for k, v in context.centers.items())
    client.close()


@pytest.mark.slow
def test_a_context_survives_a_round_trip_to_disk(home):
    dataset = fixture()[0]
    criteria = {f"label_{i:03d}": label for i, label in enumerate(dataset["labels"])}
    questions = {"label": Choice(instructions=QUESTION, criteria=criteria)}
    client = TypeSafeClient(model="minicpm5-2b", context_home=home)

    context = Context(examples=dataset["unlabeled"], name="frozen", use_description=False)
    first = client.system_one(state=dataset["rows"][0]["text"], questions=questions, context=context)
    context.save(home=home)

    reloaded = Context.load("frozen", home=home)
    second = client.system_one(state=dataset["rows"][0]["text"], questions=questions, context=reloaded)
    assert first.choices["label"].probabilities == second.choices["label"].probabilities
    client.close()
