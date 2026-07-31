"""The orchestration patterns, exercised without a network or a display."""

from __future__ import annotations

import threading

import pytest

from aichatlab.orchestrator import Orchestrator, Target
from aichatlab.session import Session
from tests.conftest import FakeClient

ALPHA = Target("local", "alpha:1b")
BETA = Target("local", "beta:2b")
GAMMA = Target("host", "gamma:3b")


@pytest.fixture
def rig():
    """An orchestrator wired to fake clients, recording every emitted event."""
    events = []
    client = FakeClient()
    session = Session()
    orchestrator = Orchestrator(
        clients={"local": client, "host": client},
        session=session,
        emit=lambda kind, **payload: events.append((kind, payload)),
    )
    return orchestrator, session, client, events


def kinds(events):
    return [kind for kind, _ in events]


def headings(events):
    return [p["heading"] for k, p in events if k == "turn_start"]


def test_broadcast_reaches_every_model(rig):
    orchestrator, session, client, events = rig

    results = orchestrator.broadcast([ALPHA, BETA, GAMMA], "hello")

    assert len(results) == 3
    assert {call["model"] for call in client.calls} == {
        "alpha:1b", "beta:2b", "gamma:3b"}
    # each conversation holds the prompt and the reply
    for target in (ALPHA, BETA, GAMMA):
        history = session.history(target.key)
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "hello"


def test_broadcast_streams_tokens_tagged_per_model(rig):
    orchestrator, _session, _client, events = rig

    orchestrator.broadcast([ALPHA, BETA], "hello")

    starts = [p["stream_id"] for k, p in events if k == "turn_start"]
    tokens = [p["stream_id"] for k, p in events if k == "token"]
    assert len(set(starts)) == 2          # distinct streams
    assert set(tokens) <= set(starts)     # every token belongs to a stream


def test_one_model_failing_does_not_stop_the_others():
    events = []
    client = FakeClient(fail_on="beta")
    orchestrator = Orchestrator(
        clients={"local": client}, session=Session(),
        emit=lambda kind, **payload: events.append((kind, payload)))

    results = orchestrator.broadcast([ALPHA, BETA], "hello")

    assert len(results) == 1
    assert "turn_error" in kinds(events)


def test_relay_files_the_answer_back_to_the_asker(rig):
    orchestrator, session, client, _events = rig

    orchestrator.relay(ALPHA, BETA, "what do you think of blue?")

    assert "Ask beta:2b" in session.history(ALPHA.key)[0]["content"]
    # the asker's history records what came back
    assert any("Received opinion from beta:2b" in m["content"]
               for m in session.history(ALPHA.key))
    assert "what do you think of blue?" in client.calls[0]["messages"][-1]["content"]


def test_debate_runs_every_speaker_each_round_then_judges(rig):
    orchestrator, _session, client, events = rig

    orchestrator.debate([ALPHA, BETA], "is tabs better than spaces?", rounds=2)

    # 2 speakers x 2 rounds, plus one judging turn
    assert len(client.calls) == 5
    assert headings(events)[-1].endswith("consensus")


def test_debate_speakers_see_the_running_transcript(rig):
    orchestrator, _session, client, _events = rig

    orchestrator.debate([ALPHA, BETA], "topic", rounds=1)

    first_prompt = client.calls[0]["messages"][-1]["content"]
    second_prompt = client.calls[1]["messages"][-1]["content"]
    assert "debate so far" not in first_prompt      # nothing said yet
    assert "alpha:1b said" in second_prompt          # beta can reply to alpha


def test_debate_needs_two_models(rig):
    orchestrator, _session, client, events = rig

    assert orchestrator.debate([ALPHA], "topic") is None
    assert client.calls == []
    assert "note" in kinds(events)


def test_critique_chain_drafts_critiques_then_revises(rig):
    orchestrator, _session, client, events = rig

    orchestrator.critique_chain([ALPHA, BETA], "write a haiku")

    assert len(client.calls) == 3
    assert [h.split("· ")[-1] for h in headings(events)] == [
        "draft", "critique", "final"]
    # the reviser sees both the draft and the critique
    final_prompt = client.calls[2]["messages"][-1]["content"]
    assert "Reviewer feedback" in final_prompt
    assert "First draft" in final_prompt


def test_critique_chain_uses_third_model_as_reviser_when_available(rig):
    orchestrator, _session, client, _events = rig

    orchestrator.critique_chain([ALPHA, BETA, GAMMA], "write a haiku")

    assert [call["model"] for call in client.calls] == [
        "alpha:1b", "beta:2b", "gamma:3b"]


def test_judge_panel_scores_answers_anonymously():
    # replies deliberately do not mention the model, so any model name in the
    # judge's prompt would have to have come from the orchestrator itself
    events = []
    client = FakeClient(reply="The answer is four.")
    orchestrator = Orchestrator(
        clients={"local": client, "host": client}, session=Session(),
        emit=lambda kind, **payload: events.append((kind, payload)))

    orchestrator.judge_panel([ALPHA, BETA], "what is 2+2?", judge=GAMMA)

    judge_prompt = client.calls[-1]["messages"][-1]["content"]
    assert "Answer 1:" in judge_prompt and "Answer 2:" in judge_prompt
    # the judge must not be told which model wrote which answer
    assert "alpha:1b" not in judge_prompt
    assert "beta:2b" not in judge_prompt
    # ...but the key is revealed to the user afterwards
    assert any("Answer 1 = alpha:1b" in p.get("text", "")
               for k, p in events if k == "note")


def test_cancelling_stops_before_the_next_turn(rig):
    orchestrator, _session, client, _events = rig
    orchestrator.cancel.set()

    orchestrator.debate([ALPHA, BETA], "topic", rounds=3)

    assert client.calls == []


def test_cancel_midway_through_a_debate():
    events = []
    client = FakeClient()
    cancel = threading.Event()
    orchestrator = Orchestrator(
        clients={"local": client}, session=Session(), cancel=cancel,
        emit=lambda kind, **payload: (
            events.append((kind, payload)),
            cancel.set() if len(client.calls) >= 2 else None))

    orchestrator.debate([ALPHA, BETA], "topic", rounds=5)

    assert len(client.calls) == 2      # stopped instead of running all 10
