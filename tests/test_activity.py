"""The live status line: what is running, for how long, and whether it stalled."""

from __future__ import annotations

import pytest

from aichatlab.activity import (
    STALL_SECONDS,
    Activity,
    Tracker,
    describe,
    human_duration,
    looks_urgent,
    stall_note,
)

# ------------------------------------------------------------------ duration

@pytest.mark.parametrize("seconds, expected", [
    (0, "0s"), (9, "9s"), (59, "59s"), (60, "1m 00s"), (92, "1m 32s"),
    (3599, "59m 59s"), (3600, "1h 00m"), (7325, "2h 02m"),
])
def test_durations_read_the_way_a_person_says_them(seconds, expected):
    assert human_duration(seconds) == expected


def test_a_negative_duration_does_not_produce_nonsense():
    assert human_duration(-5) == "0s"


# ------------------------------------------------------------------ activity

def test_a_fresh_activity_is_thinking_until_something_arrives():
    activity = Activity(step="thinking", label="Qwen3 8B", started=100.0,
                        last_at=100.0)

    assert "is thinking" in describe(activity, 105.0)
    activity.touch(106.0)
    assert "is writing" in describe(activity, 106.0)


def test_the_status_line_carries_elapsed_tokens_and_rate():
    activity = Activity(step="writing", label="GPT-OSS 20B", started=0.0,
                        last_at=0.0)
    for tick in range(1, 61):
        activity.touch(float(tick))

    line = describe(activity, 60.0)

    assert "GPT-OSS 20B" in line
    assert "60 tokens" in line
    assert "1m 00s" in line
    assert "1 tok/s" in line


def test_no_rate_is_claimed_from_a_single_token():
    activity = Activity(started=0.0, last_at=0.0)
    activity.touch(1.0)

    assert "tok/s" not in describe(activity, 1.0)


def test_nothing_running_reads_as_ready():
    assert describe(None, 10.0) == "Ready"


def test_other_running_work_is_counted_in():
    activity = Activity(label="Llama3", started=0.0, last_at=0.0)

    assert "(+2 more running)" in describe(activity, 3.0, extra=2)


def test_an_activity_with_a_detail_shows_it():
    activity = Activity(step="compacting", label="Gemma2", started=0.0,
                        last_at=0.0, detail="14 messages")

    assert "14 messages" in describe(activity, 1.0)


# -------------------------------------------------------------------- stalls

def test_silence_past_the_threshold_counts_as_stalled():
    activity = Activity(started=0.0, last_at=0.0)

    assert not activity.stalled(STALL_SECONDS - 1)
    assert activity.stalled(STALL_SECONDS + 1)


def test_an_arriving_token_clears_the_stall():
    activity = Activity(started=0.0, last_at=0.0)
    assert activity.stalled(100.0)

    activity.touch(100.0)

    assert not activity.stalled(101.0)


def test_the_stall_note_distinguishes_slow_from_never_started():
    started = Activity(label="Qwen3", started=0.0, last_at=0.0, tokens=40)
    silent = Activity(label="Qwen3", started=0.0, last_at=0.0)

    assert "may still be working" in stall_note(started, 60.0)
    assert "loaded into memory" in stall_note(silent, 60.0)
    assert "1m 00s" in stall_note(silent, 60.0)


def test_a_stall_is_only_reported_once():
    tracker = Tracker()
    tracker.start("a", "writing", "M", now=0.0)

    first = tracker.stalled(100.0)
    assert first
    first[0][1].warned = True

    assert tracker.stalled(200.0) == []


# ------------------------------------------------------------------- tracker

def test_the_tracker_follows_several_models_at_once():
    tracker = Tracker()
    tracker.start(1, "writing", "Alpha", now=0.0)
    tracker.start(2, "writing", "Beta", now=5.0)

    assert tracker.busy()
    assert tracker.current().label == "Beta"        # most recently started

    tracker.finish(2)
    assert tracker.current().label == "Alpha"
    tracker.finish(1)
    assert not tracker.busy()
    assert tracker.current() is None


def test_touching_an_unknown_key_is_harmless():
    Tracker().touch("nope", 1.0)                    # must not raise


def test_clearing_forgets_everything():
    tracker = Tracker()
    tracker.start(1, "writing", now=0.0)
    tracker.clear()

    assert not tracker.busy()


# -------------------------------------------------------- typing while busy

@pytest.mark.parametrize("text", [
    "stop", "STOP that", "wait, that's wrong",
    "actually can you do it in Python instead",
    "never mind, different question", "hold on", "forget it",
])
def test_an_interruption_is_recognised(text):
    assert looks_urgent(text)


@pytest.mark.parametrize("text", [
    "what does the cli module do?",
    "summarise the tests folder",
    "and how would I run it on Windows?",
    "",
])
def test_an_ordinary_follow_up_is_not_urgent(text):
    assert not looks_urgent(text)


def test_urgency_is_judged_from_the_opening_not_the_whole_message():
    """"...and then it stops" is a description, not a command."""
    long_tail = ("Explain how the pipeline works end to end, " + "x" * 200
                 + " and then tell me why it stops")

    assert not looks_urgent(long_tail)
