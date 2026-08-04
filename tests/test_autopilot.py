"""Unattended running: when to press Continue, and when to stop pressing it."""

from __future__ import annotations

from aichatlab.autopilot import Limits, Run, human_hours, normalise

LONG = ("The first improvement is to split the symbol table so the loader "
        "stops rebuilding it on every open, which is where most of the "
        "start-up time goes on a large binary. ")


def more(index: int) -> str:
    """A genuinely different continuation — real work, not a restatement."""
    return (f"Improvement {index}: the {index}th pass over the relocation "
            f"table can be folded into the symbol walk above it, which saves "
            f"a full traversal of a structure that is already in cache at "
            f"that point in the load. ")


def fresh(**kwargs) -> Run:
    run = Run(limits=Limits(**kwargs))
    run.begin(now=0.0)
    return run


def test_a_growing_reply_is_continued():
    run = fresh()
    text = LONG
    assert run.consider(text, now=1.0)
    text += more(2)
    assert run.consider(text, now=2.0)
    assert run.continues == 2


def test_the_continuation_cap_stops_it():
    run = fresh(continues=3)
    text = ""
    for index in range(3):
        text += more(index)
        assert run.consider(text, now=index)

    text += more(99)
    decision = run.consider(text, now=4)

    assert not decision
    assert "limit of 3" in decision.reason


def test_a_continuation_that_adds_nothing_stops_it():
    """A model that has finished still reports the length cap; the tell is
    that it emits a few words and stops again."""
    run = fresh()
    run.consider(LONG, now=1.0)

    decision = run.consider(LONG + " ok", now=2.0)

    assert not decision
    assert "almost nothing" in decision.reason


def test_a_model_that_restarts_is_caught():
    """Character counts miss this: the restatement is long."""
    run = fresh()
    text = LONG
    run.consider(text, now=1.0)
    text += more(2)
    assert run.consider(text, now=2.0), "genuine new material"

    # Now it goes back to the beginning instead of carrying on.
    text += LONG
    decision = run.consider(text, now=3.0)

    assert not decision
    assert "repeating itself" in decision.reason


def test_the_clock_stops_it_even_while_it_progresses():
    run = fresh(hours=2)
    assert run.consider(LONG, now=60.0)

    decision = run.consider(LONG + more(2), now=2 * 3600 + 1)

    assert not decision
    assert "2 hours" in decision.reason


def test_running_out_of_context_stops_it():
    """Continuing here means deleting the start of the work to fit the end."""
    run = fresh()

    decision = run.consider(LONG, now=1.0, context_exhausted=True)

    assert not decision
    assert "context budget" in decision.reason


def test_the_limits_can_be_switched_off():
    run = fresh(continues=0, hours=0)
    text = ""
    for index in range(40):
        text += more(index)
        assert run.consider(text, now=index * 10_000)

    assert run.continues == 40


def test_the_summary_says_what_happened_and_why_it_stopped():
    run = fresh(continues=2)
    text = LONG
    run.consider(text, now=1.0)
    text += more(2)
    run.consider(text, now=2.0)
    text += more(3)
    run.consider(text, now=3.0)

    summary = run.summary(now=3.0)

    assert "carried on 2 times" in summary
    assert "then stopped" in summary
    assert "limit of 2" in summary


def test_decisions_made_unattended_are_reported():
    run = fresh()
    run.note("carried the conversation across to Qwen3 8B")
    run.note("answered without searching (no SearXNG configured)")

    summary = run.summary(now=10.0)

    assert "Decisions made for you" in summary
    assert "Qwen3 8B" in summary
    assert "SearXNG" in summary


def test_a_run_that_did_nothing_says_nothing():
    assert fresh().summary(now=1.0) == ""


def test_beginning_again_forgets_the_previous_run():
    run = fresh()
    run.consider(LONG, now=1.0)
    run.note("something")

    run.begin(now=100.0)

    assert run.continues == 0
    assert not run.decisions
    assert run.summary(now=101.0) == ""
    # and the echo check must not fire on the earlier run's text
    assert run.consider(LONG, now=101.0)


def test_normalise_ignores_whitespace_and_case():
    assert normalise("  Hello   THERE\n") == "hello there"


def test_human_hours_reads_naturally():
    assert human_hours(45) == "45s"
    assert "minutes" in human_hours(600)
    assert "hours" in human_hours(3 * 3600)
