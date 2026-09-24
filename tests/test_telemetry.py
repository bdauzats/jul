"""OpenTelemetry export: off unless asked, content gated, and a telemetry failure never breaks a call.

No model is loaded: `_system_one` is replaced by a stub that returns a fixed response, so these tests
check what is exported, not how the answer was computed.
"""

import subprocess
import sys

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402

import jul  # noqa: E402
from jul import Choice, Noul, Score, TypeSafeClient, telemetry  # noqa: E402
from jul.types import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage  # noqa: E402

STATE = {"ticket": "I was charged twice, card ending 4242"}
QUESTIONS = {
    "team": Choice(instructions="Which team should handle this ticket?",
                   criteria={"billing": "payments", "technical": "bugs"}),
    "is_bug": Noul(instructions="Does the message report a software bug?"),
    "anger": Score(instructions="How angry is the customer?", criteria=["Calm", "Angry"]),
}


def _canned(self, state, questions, context, model, method, route_above, methods):
    methods.update({"team": "head", "is_bug": "vector", "anger": "letters"})
    return SystemOneResponse(
        answers={"team": ChoiceAnswer(choice="billing", probabilities={"billing": 0.9, "technical": 0.1},
                                      confidence=0.9),
                 "is_bug": NoulAnswer(noul=0.12),
                 "anger": ScoreAnswer(score=0.3, legend={"0": "Calm", "1": "Angry"},
                                      probabilities={"0": 0.7, "1": 0.3}, confidence=0.4)},
        model="minicpm5-2b", usage=Usage(input_tokens=42), request_id="req-1")


@pytest.fixture
def otel(monkeypatch):
    """Telemetry switched on, exporting to memory. Returns (logs exporter, metric reader)."""
    for name in ("JUL_OTEL_LOG_STATE", "JUL_OTEL_LOG_QUESTION_DETAILS", "JUL_OTEL_LOG_PROBABILITIES",
                 "JUL_OTEL_LOG_ANSWERS", "JUL_OTEL_CONTENT_MAX_LENGTH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JUL_ENABLE_TELEMETRY", "1")
    monkeypatch.setattr(TypeSafeClient, "_system_one", _canned)
    logs, metrics = InMemoryLogRecordExporter(), InMemoryMetricReader()
    telemetry.configure(metric_readers=[metrics], log_processors=[SimpleLogRecordProcessor(logs)])
    yield logs, metrics
    telemetry.reset()


def _events(logs):
    return [dict(r.log_record.attributes) for r in logs.get_finished_logs()]


def _metrics(reader):
    out = {}
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                out[m.name] = [dict(p.attributes) | {"_value": getattr(p, "value", None),
                                                     "_count": getattr(p, "count", None)}
                               for p in m.data.data_points]
    return out


def _ask(client=None):
    client = client or TypeSafeClient(model="minicpm5-2b")
    return client.system_one(state=STATE, questions=QUESTIONS)


def test_one_request_event_and_one_decision_event_per_question(otel):
    logs, _ = otel
    _ask()
    events = _events(logs)
    assert [e["event.name"] for e in events] == ["request", "decision", "decision", "decision"]
    assert [e["event.sequence"] for e in events] == [0, 1, 2, 3]
    request = events[0]
    assert request["model"] == "minicpm5-2b"
    assert request["input_tokens"] == 42
    assert request["question_count"] == 3
    assert request["request_id"] == "req-1"
    assert request["duration_ms"] >= 0
    assert len({e["session.id"] for e in events}) == 1


def test_decisions_carry_type_method_answer_and_confidence(otel):
    logs, _ = otel
    _ask()
    team, is_bug, anger = _events(logs)[1:]
    assert (team["question.type"], team["method"], team["answer"], team["confidence"]) == \
        ("choice", "head", "billing", 0.9)
    assert (is_bug["question.type"], is_bug["answer"]) == ("noul", 0.12)
    assert "confidence" not in is_bug                # a Noul's answer is already a probability
    assert (anger["question.type"], anger["method"], anger["answer"]) == ("score", "letters", 0.3)


def test_content_is_redacted_by_default(otel):
    logs, _ = otel
    _ask()
    events = _events(logs)
    assert events[0]["state"] == "<REDACTED>"
    assert events[0]["state_length"] == len(jul.types.serialize_state(STATE))
    blob = repr(events)
    for secret in ("4242", "charged twice", "Which team", "is_bug", "technical"):
        assert secret not in blob, secret


def test_each_gate_lets_in_only_its_own_content(otel, monkeypatch):
    logs, _ = otel
    monkeypatch.setenv("JUL_OTEL_LOG_STATE", "1")
    _ask()
    events = _events(logs)
    assert "4242" in events[0]["state"]
    assert "question.name" not in events[1] and "probabilities" not in events[1]

    logs.clear()
    monkeypatch.delenv("JUL_OTEL_LOG_STATE")
    monkeypatch.setenv("JUL_OTEL_LOG_QUESTION_DETAILS", "1")
    monkeypatch.setenv("JUL_OTEL_LOG_PROBABILITIES", "1")
    _ask()
    events = _events(logs)
    assert events[0]["state"] == "<REDACTED>"
    team = events[1]
    assert team["question.name"] == "team"
    assert team["question.instructions"] == "Which team should handle this ticket?"
    assert list(team["question.options"]) == ["billing", "technical"]
    assert list(team["probabilities"]) == [0.9, 0.1]


def test_answers_can_be_withheld(otel, monkeypatch):
    logs, _ = otel
    monkeypatch.setenv("JUL_OTEL_LOG_ANSWERS", "0")
    _ask()
    assert {e["answer"] for e in _events(logs)[1:]} == {"<REDACTED>"}


def test_content_is_truncated(otel, monkeypatch):
    logs, _ = otel
    monkeypatch.setenv("JUL_OTEL_LOG_STATE", "1")
    monkeypatch.setenv("JUL_OTEL_CONTENT_MAX_LENGTH", "40")
    _ask()
    state = _events(logs)[0]["state"]
    assert len(state) == 40 and "TRUNCATED" in state


def test_metrics(otel):
    _, reader = otel
    client = TypeSafeClient(model="minicpm5-2b")
    _ask(client)
    _ask(client)
    m = _metrics(reader)
    assert m["jul.session.count"][0]["_value"] == 1           # once per client, not per call
    assert m["jul.token.usage"][0]["_value"] == 84
    assert m["jul.token.usage"][0]["type"] == "input"
    assert m["jul.request.duration"][0]["_count"] == 2
    by_type = {p["question.type"]: p["_value"] for p in m["jul.decision.count"]}
    assert by_type == {"choice": 2, "noul": 2, "score": 2}
    assert {p["question.type"] for p in m["jul.decision.confidence"]} == {"choice", "score"}
    assert "session.id" in m["jul.token.usage"][0]


def test_an_error_is_counted_and_still_raised(otel, monkeypatch):
    logs, reader = otel

    def boom(*args, **kwargs):
        raise RuntimeError("model file missing at /secret/path")

    monkeypatch.setattr(TypeSafeClient, "_system_one", boom)
    with pytest.raises(RuntimeError):
        _ask()
    event = _events(logs)[0]
    assert event["event.name"] == "request_error"
    assert event["error_type"] == "RuntimeError"
    assert "error" not in event                                   # the message can hold paths or data
    assert _metrics(reader)["jul.request.error.count"][0]["_value"] == 1


def test_a_broken_exporter_never_breaks_a_decision(otel, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("collector down")

    monkeypatch.setattr(telemetry._Telemetry, "event", broken)
    assert _ask().choices["team"].choice == "billing"


def test_off_by_default_and_opentelemetry_is_not_even_imported():
    code = ("import sys, os; os.environ.pop('JUL_ENABLE_TELEMETRY', None); "
            "sys.path[:0] = ['lib', 'cli']; import jul; from jul import telemetry; "
            "assert not telemetry.active(); "
            "assert not [m for m in sys.modules if m.startswith('opentelemetry')], 'imported'")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(__import__("pathlib").Path(__file__).parents[1]))
