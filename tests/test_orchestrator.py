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


# ------------------------------------------------------------ conversation

def test_conversation_alternates_speakers_until_the_turn_cap(rig):
    orchestrator, _session, client, _events = rig

    orchestrator.converse([ALPHA, BETA], "are tabs better than spaces?",
                          max_turns=5)

    assert len(client.calls) == 5
    assert [c["model"] for c in client.calls] == [
        "alpha:1b", "beta:2b", "alpha:1b", "beta:2b", "alpha:1b"]


def test_each_speaker_sees_what_came_before(rig):
    orchestrator, _session, client, _events = rig

    orchestrator.converse([ALPHA, BETA], "topic", max_turns=3)

    first = client.calls[0]["messages"][-1]["content"]
    third = client.calls[2]["messages"][-1]["content"]
    assert "conversation so far" not in first
    assert "alpha:1b said" in third and "beta:2b said" in third


def test_a_model_can_end_the_conversation_itself():
    events = []
    client = FakeClient(reply="I think we've covered it.\n[END]")
    orchestrator = Orchestrator(
        clients={"local": client}, session=Session(),
        emit=lambda kind, **payload: events.append((kind, payload)))

    orchestrator.converse([ALPHA, BETA], "topic", max_turns=20)

    # everyone gets one turn before an ending is honoured
    assert len(client.calls) == 2
    assert any("wrapped it up" in p.get("text", "") for k, p in events
               if k == "note")


def test_an_eager_end_on_the_opening_turn_is_ignored():
    client = FakeClient(reply="[END] nothing to discuss")
    orchestrator = Orchestrator(
        clients={"local": client}, session=Session(), emit=lambda *a, **k: None)

    orchestrator.converse([ALPHA, BETA, GAMMA], "topic", max_turns=6)

    assert len(client.calls) >= 3        # not cut short on turn one


def test_the_end_marker_is_stripped_from_what_is_stored():
    session = Session()
    client = FakeClient(reply="Good talk.\n[END]")
    Orchestrator(clients={"local": client}, session=session,
                 emit=lambda *a, **k: None
                 ).converse([ALPHA, BETA], "topic", max_turns=4)

    stored = " ".join(m["content"] for m in session.history(ALPHA.key))
    assert "Good talk." in stored
    assert "[END]" not in stored


def test_user_interjections_reach_the_next_speaker(rig):
    orchestrator, _session, client, events = rig
    queued = [["wait, what about cost?"]]

    def pending():
        return queued.pop(0) if queued else []

    orchestrator.converse([ALPHA, BETA], "topic", max_turns=3, pending=pending)

    first = client.calls[0]["messages"][-1]["content"]
    assert "wait, what about cost?" in first
    assert any("You joined in" in p.get("text", "") for k, p in events
               if k == "note")


def test_interjections_persist_in_the_transcript(rig):
    orchestrator, _session, client, _events = rig
    queued = [["first point"], [], []]

    def pending():
        return queued.pop(0) if queued else []

    orchestrator.converse([ALPHA, BETA], "topic", max_turns=3, pending=pending)

    last = client.calls[-1]["messages"][-1]["content"]
    assert "first point" in last          # still visible several turns later


def test_conversation_needs_two_models(rig):
    orchestrator, _session, client, events = rig

    assert orchestrator.converse([ALPHA], "topic") == []
    assert client.calls == []
    assert any("at least two" in p.get("text", "") for k, p in events
               if k == "note")


def test_stopping_ends_the_conversation():
    events = []
    client = FakeClient()
    cancel = threading.Event()
    orchestrator = Orchestrator(
        clients={"local": client}, session=Session(), cancel=cancel,
        emit=lambda kind, **payload: (
            events.append((kind, payload)),
            cancel.set() if len(client.calls) >= 3 else None))

    orchestrator.converse([ALPHA, BETA], "topic", max_turns=30)

    assert len(client.calls) == 3
    assert any("Stopped after" in p.get("text", "") for k, p in events
               if k == "note")


def test_reaching_the_cap_says_so(rig):
    orchestrator, _session, _client, events = rig

    orchestrator.converse([ALPHA, BETA], "topic", max_turns=2)

    assert any("2-turn limit" in p.get("text", "") for k, p in events
               if k == "note")


# --------------------------------------------------------------- research

RESEARCH_RESULTS = [
    {"title": "Top speed records", "url": "https://a.example/speed",
     "snippet": "Records listed."},
    {"title": "Road test", "url": "https://b.example/test",
     "snippet": "Measured figures."},
]


class ResearchClient(FakeClient):
    """Returns a different reply depending on which pipeline step is asking."""

    def __init__(self, verification="OK", answer="It does 308mph (Source 1)."):
        super().__init__()
        self.verification = verification
        self.answer = answer

    def chat(self, model, messages, options=None, on_token=None, cancel=None,
             think=None, on_thought=None):
        prompt = messages[-1]["content"]
        if "Searches:" in prompt:
            self.reply = "top speed record\n0-60 acceleration time"
        elif "Findings:" in prompt:
            self.reply = self.verification
        elif "Revised answer:" in prompt:
            self.reply = "Corrected answer (Source 2)."
        else:
            self.reply = self.answer
        return super().chat(model, messages, options, on_token, cancel,
                            think, on_thought)


def research_rig(client=None, results=None):
    events = []
    client = client or ResearchClient()
    session = Session()
    orchestrator = Orchestrator(
        clients={"local": client, "host": client}, session=session,
        emit=lambda kind, **payload: events.append((kind, payload)))
    searched = []

    def search_fn(query):
        searched.append(query)
        return list(results if results is not None else RESEARCH_RESULTS)

    return orchestrator, session, client, events, searched, search_fn


def test_research_searches_every_planned_angle():
    orchestrator, _s, _c, _e, searched, search_fn = research_rig()

    orchestrator.research(ALPHA, "how fast is it?", search_fn,
                          fetch_fn=lambda url: "page text", follow_up=False)

    assert searched == ["top speed record", "0-60 acceleration time"]


def test_the_report_records_which_searches_were_run():
    orchestrator, _s, _c, _e, _q, search_fn = research_rig()

    report = orchestrator.research(ALPHA, "how fast?", search_fn,
                                   fetch_fn=lambda url: "text", follow_up=False)

    assert "**Searches run**" in report
    assert "top speed record" in report


def test_follow_up_searches_appear_in_the_report():
    client = ResearchClient(verification="GAP: no pricing\nSEARCH: U9 price")
    orchestrator, _s, _c, _e, _q, search_fn = research_rig(client)

    report = orchestrator.research(ALPHA, "how fast?", search_fn,
                                   fetch_fn=lambda url: "text", follow_up=True)

    assert "U9 price" in report


def test_research_produces_a_report_with_sources():
    orchestrator, _s, _c, events, _q, search_fn = research_rig()

    report = orchestrator.research(ALPHA, "how fast?", search_fn,
                                   fetch_fn=lambda url: "page text",
                                   follow_up=False)

    assert "**Sources**" in report
    assert "https://a.example/speed" in report
    assert any(k == "report" for k, _ in events)


def test_research_records_the_report_in_the_session():
    orchestrator, session, _c, _e, _q, search_fn = research_rig()

    orchestrator.research(ALPHA, "how fast?", search_fn,
                          fetch_fn=lambda url: "text", follow_up=False)

    history = session.history(ALPHA.key)
    assert history[0]["content"].startswith("[Research]")
    assert "**Sources**" in history[1]["content"]


def test_research_revises_when_the_check_finds_problems():
    client = ResearchClient(verification="UNSUPPORTED: 308mph is invented")
    orchestrator, _s, _c, events, _q, search_fn = research_rig(client)

    report = orchestrator.research(ALPHA, "how fast?", search_fn,
                                   fetch_fn=lambda url: "text", follow_up=False)

    assert "Corrected answer" in report
    headings = [p["heading"] for k, p in events if k == "turn_start"]
    assert any(h.endswith("final") for h in headings)


def test_research_skips_revision_when_everything_checks_out():
    orchestrator, _s, _c, events, _q, search_fn = research_rig()

    orchestrator.research(ALPHA, "how fast?", search_fn,
                          fetch_fn=lambda url: "text", follow_up=False)

    headings = [p["heading"] for k, p in events if k == "turn_start"]
    assert not any(h.endswith("final") for h in headings)


def test_research_follows_up_on_gaps():
    client = ResearchClient(
        verification="GAP: no pricing\nSEARCH: U9 Xtreme price")
    orchestrator, _s, _c, _e, searched, search_fn = research_rig(client)

    orchestrator.research(ALPHA, "how fast?", search_fn,
                          fetch_fn=lambda url: "text", follow_up=True)

    assert "U9 Xtreme price" in searched


def test_an_invented_citation_forces_a_revision():
    """Citing a source that was never retrieved must not survive the pass."""
    client = ResearchClient(answer="It costs $40,000 (Source 9).")
    orchestrator, _s, _c, events, _q, search_fn = research_rig(client)

    report = orchestrator.research(ALPHA, "how much?", search_fn,
                                   fetch_fn=lambda url: "text", follow_up=False)

    headings = [p["heading"] for k, p in events if k == "turn_start"]
    assert any(h.endswith("final") for h in headings)
    assert "Source 9" not in report


def test_research_gives_up_gracefully_when_nothing_is_found():
    orchestrator, _s, _c, events, _q, search_fn = research_rig(results=[])

    assert orchestrator.research(ALPHA, "obscure question", search_fn,
                                 fetch_fn=lambda url: "") is None
    assert any("nothing to read" in p.get("text", "")
               for k, p in events if k == "note")


def test_research_survives_a_failing_search_engine():
    orchestrator, _s, _c, events, _q, _sf = research_rig()

    def broken(query):
        raise RuntimeError("searxng down")

    assert orchestrator.research(ALPHA, "q", broken,
                                 fetch_fn=lambda url: "") is None
    assert any("Search failed" in p.get("text", "")
               for k, p in events if k == "note")


def test_research_survives_unreachable_pages():
    orchestrator, _s, _c, _e, _q, search_fn = research_rig()

    def dead(url):
        raise RuntimeError("connection refused")

    report = orchestrator.research(ALPHA, "how fast?", search_fn,
                                   fetch_fn=dead, follow_up=False)
    assert report and "**Sources**" in report


def test_stopping_research_ends_it_early():
    client = ResearchClient()
    cancel = threading.Event()
    orchestrator = Orchestrator(
        clients={"local": client}, session=Session(), cancel=cancel,
        emit=lambda kind, **payload: None)
    cancel.set()

    assert orchestrator.research(ALPHA, "q", lambda q: RESEARCH_RESULTS,
                                 fetch_fn=lambda url: "") is None
    assert client.calls == []


# -- reasoning models ------------------------------------------------------
def test_reasoning_is_emitted_as_its_own_event():
    """The UI needs it separately: reasoning is not part of the reply."""
    events = []
    client = FakeClient(server="local", thinking="working it out")
    session = Session()
    orchestrator = Orchestrator(
        clients={"local": client}, session=session,
        emit=lambda kind, **payload: events.append((kind, payload)),
        options={"num_predict": 192})

    orchestrator.broadcast([Target("local", "qwen3:8b")], "hello")

    thoughts = [p["text"] for kind, p in events if kind == "thought"]
    assert thoughts == ["working it out"]


def test_reasoning_never_reaches_the_saved_conversation():
    """It is working, not something the model said — and it would eat the
    context budget on every subsequent turn."""
    session = Session()
    client = FakeClient(server="local", reply="42", thinking="6 times 7")
    orchestrator = Orchestrator(
        clients={"local": client}, session=session, emit=lambda *a, **k: None)

    target = Target("local", "qwen3:8b")
    orchestrator.broadcast([target], "what is six sevens?")

    saved = " ".join(m["content"] for m in session.history(target.key))
    assert "42" in saved
    assert "6 times 7" not in saved


def test_the_reply_allowance_is_raised_for_a_reasoning_model():
    """192 tokens is the Quick end of the slider; qwen3 would spend the lot
    thinking and hand back nothing."""
    client = FakeClient(server="local")
    orchestrator = Orchestrator(
        clients={"local": client}, session=Session(),
        emit=lambda *a, **k: None, options={"num_predict": 192})

    orchestrator.broadcast([Target("local", "qwen3:8b")], "hello")

    assert client.calls[-1]["options"]["num_predict"] > 192


def test_an_ordinary_model_keeps_the_allowance_it_was_given():
    client = FakeClient(server="local")
    orchestrator = Orchestrator(
        clients={"local": client}, session=Session(),
        emit=lambda *a, **k: None, options={"num_predict": 192})

    orchestrator.broadcast([Target("local", "llama3:8b")], "hello")

    assert client.calls[-1]["options"]["num_predict"] == 192
    assert client.calls[-1]["think"] is None


def test_turning_reasoning_off_reaches_the_request():
    client = FakeClient(server="local")
    orchestrator = Orchestrator(
        clients={"local": client}, session=Session(),
        emit=lambda *a, **k: None, think=False)

    orchestrator.broadcast([Target("local", "qwen3:8b")], "hello")

    assert client.calls[-1]["think"] is False


def test_machinery_calls_never_reason():
    """Triage, summarising and query rewriting run on 48 to 700 tokens.  A
    reasoning model would spend all of them thinking and return nothing."""
    client = FakeClient(server="local", reply="a short answer")
    orchestrator = Orchestrator(
        clients={"local": client}, session=Session(),
        emit=lambda *a, **k: None, think=True)

    orchestrator._quiet_turn(Target("local", "qwen3:8b"), "rewrite this", 48)

    assert client.calls[-1]["think"] is False


def test_a_long_run_compacts_between_turns_not_only_at_send():
    """Compaction is decided once when Send is pressed, but plan-and-work
    runs a dozen turns inside that one send — so a conversation can sail past
    the ceiling mid-run with nothing left to notice."""
    session = Session()
    target = Target("local", "alpha:1b")
    for _ in range(12):
        session.add(target.key, "user", "question " * 400)
        session.add(target.key, "assistant", "answer " * 400)

    folded = []

    def compactor(targets):
        folded.append(list(targets))
        # Shrink it, the way a real compaction would.
        session.conversations[targets[0].key] = [
            {"role": "user", "content": "[Earlier conversation, summarised] "
                                        "they discussed the loader."}]
        return True

    orchestrator = Orchestrator(
        clients={"local": FakeClient(server="local")}, session=session,
        emit=lambda *a, **k: None, budget_tokens=6_000, compactor=compactor)

    orchestrator.broadcast([target], "carry on")

    assert folded, "the run should have folded before taking its turn"


def test_it_does_not_try_to_compact_the_same_thread_every_turn():
    """A conversation that is one huge pinned attachment cannot be folded at
    all; retrying each step would spend a summarising call to be told so."""
    session = Session()
    target = Target("local", "alpha:1b")
    session.add(target.key, "user", "x " * 20_000)

    attempts = []
    orchestrator = Orchestrator(
        clients={"local": FakeClient(server="local")}, session=session,
        emit=lambda *a, **k: None, budget_tokens=6_000,
        compactor=lambda targets: attempts.append(1) or False)

    orchestrator.broadcast([target], "one")
    orchestrator.broadcast([target], "two")
    orchestrator.broadcast([target], "three")

    assert len(attempts) == 1


def test_a_compactor_that_throws_never_breaks_the_run():
    session = Session()
    target = Target("local", "alpha:1b")
    for _ in range(12):
        session.add(target.key, "user", "question " * 400)

    def boom(_targets):
        raise RuntimeError("the summariser fell over")

    orchestrator = Orchestrator(
        clients={"local": FakeClient(server="local", reply="fine")},
        session=session, emit=lambda *a, **k: None, budget_tokens=6_000,
        compactor=boom)

    results = orchestrator.broadcast([target], "carry on")

    assert results and results[0].text == "fine"


def test_a_short_conversation_is_left_alone():
    session = Session()
    target = Target("local", "alpha:1b")
    session.add(target.key, "user", "hello")
    folded = []

    orchestrator = Orchestrator(
        clients={"local": FakeClient(server="local")}, session=session,
        emit=lambda *a, **k: None, budget_tokens=28_000,
        compactor=lambda targets: folded.append(1))

    orchestrator.broadcast([target], "hi")

    assert not folded


# ------------------------------------------------- planning, with and without
# a folder attached
#
# Plan mode had no test at all, which is how it went a long time sending every
# step out as a bare user message: no system prompt, so no editing
# instructions, and no attachments, so no file to quote.  A model asked to
# replace a function it had never been shown produced a plausible replacement
# for a function that did not exist, and every one was refused downstream.

PLAN_REPLY = """1. Open the file engine.py
2. Locate the function evaluate_and_schedule
3. Replace it with a version that schedules concurrently
4. Save the modified file
5. Test the modifications"""


class PlanningClient(FakeClient):
    """Answers the first call with a plan and the rest with work."""

    def chat(self, model, messages, options=None, on_token=None, cancel=None,
             think=None, on_thought=None):
        self.reply = PLAN_REPLY if not self.calls else "some work"
        return super().chat(model, messages, options=options,
                            on_token=on_token, cancel=cancel, think=think,
                            on_thought=on_thought)


def planning_rig(**kwargs):
    client = PlanningClient()
    session = Session()
    orchestrator = Orchestrator(
        clients={"local": client}, session=session,
        emit=lambda kind, **payload: None, system_prompt="THE RULES",
        **kwargs)
    return orchestrator, session, client


def roles_of(call):
    return [message["role"] for message in call["messages"]]


def test_every_step_carries_the_system_prompt():
    """Without it a step has no persona, no recalled lessons and — the one
    that actually bit — no instructions for how to propose an edit."""
    orchestrator, _session, client = planning_rig()

    orchestrator.plan_and_work(ALPHA, "make it concurrent")

    working = client.calls[1:]            # call 0 is the plan itself
    assert working, "the plan produced no steps"
    for call in working:
        assert call["messages"][0]["role"] == "system"
        assert "THE RULES" in call["messages"][0]["content"]


def test_a_step_that_has_to_edit_can_see_the_file():
    """The attached file lives in the conversation, not in the step prompt.
    A step sent on its own cannot quote what it was never shown."""
    orchestrator, session, client = planning_rig(editing=True)
    session.add(ALPHA.key, "user",
                "Attached file engine.py:\ndef evaluate_and_schedule(self):\n"
                "    return []")

    orchestrator.plan_and_work(ALPHA, "make it concurrent")

    for call in client.calls[1:]:
        blob = "\n".join(m["content"] for m in call["messages"])
        assert "def evaluate_and_schedule" in blob


def test_a_step_that_is_not_editing_stays_cheap():
    """Isolation is the whole point of the mode; only editing pays for the
    conversation to travel with every step."""
    orchestrator, session, client = planning_rig()
    session.add(ALPHA.key, "user", "a long earlier conversation " * 50)

    orchestrator.plan_and_work(ALPHA, "write me a summary")

    for call in client.calls[1:]:
        assert roles_of(call) == ["system", "user"]


def test_steps_a_model_cannot_do_are_struck_from_the_plan():
    orchestrator, _session, client = planning_rig(editing=True)

    orchestrator.plan_and_work(ALPHA, "make it concurrent")

    plan = "\n".join(m["content"] for m in client.calls[1]["messages"])
    assert "Locate the function" in plan
    assert "Open the file" not in plan
    assert "Save the modified file" not in plan


def test_the_plan_is_left_alone_when_nobody_is_editing():
    """"Run the tests" is a fine thing to write about when the answer is
    prose; it is only theatre when the app is meant to be writing files."""
    orchestrator, _session, client = planning_rig()

    orchestrator.plan_and_work(ALPHA, "how should I ship this?")

    plan = "\n".join(m["content"] for m in client.calls[1]["messages"])
    assert "Open the file" in plan


def test_the_final_answer_is_asked_for_edit_blocks():
    orchestrator, _session, client = planning_rig(editing=True)

    orchestrator.plan_and_work(ALPHA, "make it concurrent")

    summary = client.calls[-1]["messages"][-1]["content"]
    assert "edit block" in summary
    assert "changes nothing" in summary
