"""Compatibility: code written against `typesafe_sdk` must run by changing only the import.

The body of `jev_script` is deliberately written the way the Jev SDK documents it, down to the
per-call `model=` and `client.close()`, and it is handed the module under test as `sdk`.
"""

import jul
import pytest


def jev_script(sdk):
    """A script a Jev user could have written. Nothing here is jul-specific."""
    client = sdk.TypeSafeClient(model="minicpm5-2b")
    response = client.system_one(
        state={"ticket": "I was charged twice for my subscription this month."},
        questions={
            "team": sdk.Choice(instructions="Which team should handle this ticket?",
                               criteria={"billing": "payments, invoices, refunds",
                                         "technical": "bugs, errors, crashes",
                                         "sales": "pricing, plans, demos"}),
            "is_bug": sdk.Noul(instructions="Does the message report a software bug?"),
            "frustration": sdk.Score(instructions="How frustrated is the customer?",
                                     criteria=["Calm", "Frustrated but civil", "Very angry"]),
        },
        model="minicpm5-2b",
    )
    out = {
        "choice": response.choices["team"].choice,
        "probabilities": response.choices["team"].probabilities,
        "confidence": response.choices["team"].confidence,
        "noul": response.nouls["is_bug"].noul,
        "score": response.scores["frustration"].score,
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "request_id": response.request_id,
    }
    client.close()
    return out


@pytest.mark.slow
def test_a_jev_script_runs_unchanged():
    out = jev_script(jul)
    assert out["choice"] in {"billing", "technical", "sales"}
    assert set(out["probabilities"]) == {"billing", "technical", "sales"}
    assert abs(sum(out["probabilities"].values()) - 1) < 0.01
    assert 0 <= out["confidence"] <= 1
    assert 0 <= out["noul"] <= 1
    assert 0 <= out["score"] <= 2
    assert out["model"] == "minicpm5-2b"
    assert out["input_tokens"] > 0
    assert len(out["request_id"]) == 36        # a uuid4, generated locally


@pytest.mark.slow
def test_the_ticket_of_the_readme_is_routed_to_billing():
    assert jev_script(jul)["choice"] == "billing"


@pytest.mark.slow
def test_remote_only_arguments_are_accepted_and_ignored(client):
    response = client.system_one(
        state="the app crashes on export",
        questions={"is_bug": jul.Noul(instructions="Does the message report a software bug?")},
        response_model=None, retry=3, extra_body={"anything": 1}, timeout=30,
    )
    assert 0 <= response.nouls["is_bug"].noul <= 1


def test_the_constructor_accepts_the_remote_arguments():
    c = jul.TypeSafeClient(model="minicpm5-2b", api_key="unused", base_url="https://example.invalid",
                           timeout=10, max_retries=2)
    assert c.model == "minicpm5-2b"       # nothing is loaded until the first call


@pytest.mark.slow
def test_the_async_client_returns_the_same_answer():
    import asyncio

    async def go():
        client = jul.AsyncTypeSafeClient(model="minicpm5-2b")
        response = await client.system_one(
            state="I was charged twice",
            questions={"team": jul.Choice(instructions="Which team?",
                                          criteria={"billing": "payments, invoices",
                                                    "technical": "bugs, errors"})})
        await client.close()
        return response

    assert asyncio.run(go()).choices["team"].choice == "billing"


@pytest.mark.slow
def test_a_question_with_noul_criteria_runs(client):
    response = client.system_one(
        state="the export button crashes the app",
        questions={"bug": jul.Noul(instructions="Is this a software bug?",
                                   criteria=jul.NoulCriteria(true="it reports a defect",
                                                             false="it does not report a defect"))})
    assert 0 <= response.nouls["bug"].noul <= 1


def test_asking_nothing_is_an_error():
    c = jul.TypeSafeClient(model="minicpm5-2b")
    with pytest.raises(ValueError):
        c.system_one(state="x", questions={})


def test_every_type_is_read_by_vectors_by_default():
    """Measured on the dev sets; letters stays reachable with method="letters"."""
    from jul.client import DEFAULT_METHOD
    assert set(DEFAULT_METHOD.values()) == {"vector"}


@pytest.mark.slow
def test_the_letters_reading_is_still_available(client):
    response = client.system_one(
        state="the export button crashes the app",
        questions={"bug": jul.Noul(instructions="Is this a software bug?")},
        method="letters")
    assert 0 <= response.nouls["bug"].noul <= 1


@pytest.mark.slow
def test_a_tuned_head_wins_over_an_explicit_letters_method(client, home, tmp_path):
    """A head is trained on vector features, so asking for letters must not silently drop it."""
    question = {"bug": jul.Noul(instructions="Does the message report a software bug?")}
    labeled = [("the app crashes on export", {"bug": True}),
               ("error 500 every time I log in", {"bug": True}),
               ("the save button does nothing", {"bug": True}),
               ("please send me the invoice", {"bug": False}),
               ("how much is the Pro plan?", {"bug": False}),
               ("can I change my email address?", {"bug": False})] * 12

    context = jul.Context(name="bugs", use_description=False)
    report = client.autotune(context, question, labeled, save=False)["bug"]
    if not report.activated:
        pytest.skip(f"the safety net refused the head: {report.reason}")
    assert context.heads, "an activated head must be stored in the context"

    letters = client.system_one(state="the app crashes on export", questions=question,
                                context=context, method="letters")
    vector = client.system_one(state="the app crashes on export", questions=question,
                               context=context, method="vector")
    assert letters.nouls["bug"].noul == vector.nouls["bug"].noul
