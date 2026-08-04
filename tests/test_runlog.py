"""The run log: what ran, how long it took, and how it ended."""

from __future__ import annotations

from datetime import datetime

from aichatlab.client import ChatResult
from aichatlab.runlog import Entry, RunLog, outcome_of, write_csv

AT = datetime(2026, 8, 2, 14, 30, 5)


def result(**kwargs) -> ChatResult:
    base = dict(model="gpt-oss:20b", server="host", elapsed_s=32.8,
                prompt_tokens=24_000, eval_tokens=192)
    base.update(kwargs)
    return ChatResult(**base)


# ------------------------------------------------------------------ outcomes

def test_a_normal_reply_is_recorded_as_finished():
    assert outcome_of(result(done_reason="stop"))[0] == "finished"


def test_a_capped_reply_is_recorded_as_cut_off():
    assert outcome_of(result(done_reason="length"))[0] == "length"


def test_a_cancelled_reply_beats_its_done_reason():
    """You stopped it; that is not the model running out of room."""
    assert outcome_of(result(done_reason="length", cancelled=True))[0] == "cancelled"


def test_the_outcome_is_written_in_english():
    log = RunLog()
    entry = log.record(result(done_reason="length"), now=AT)

    assert entry.outcome_text == "cut off at the token cap"


# ------------------------------------------------------------------ entries

def test_a_recorded_entry_keeps_the_numbers_that_matter():
    log = RunLog()
    entry = log.record(result(done_reason="stop"), purpose="chat", now=AT)

    assert entry.at == "14:30:05"
    assert entry.model == "gpt-oss:20b"
    assert entry.server == "host"
    assert entry.purpose == "chat"
    assert entry.duration_s == 32.8
    assert entry.eval_tokens == 192
    assert round(entry.rate) == 6


def test_a_row_renders_missing_numbers_as_dashes():
    entry = Entry(model="m", at="10:00:00")

    assert "—" in entry.row()


def test_a_failure_is_logged_with_its_message():
    log = RunLog()
    log.record_failure("gpt-oss:20b", "host", "compaction",
                       "HTTP 500 (Internal Server Error)", now=AT)

    assert log.entries[0].outcome == "error"
    assert "HTTP 500" in log.entries[0].row()[-1]


def test_the_log_is_bounded_so_a_long_session_cannot_grow_forever():
    log = RunLog(limit=10)
    for _ in range(50):
        log.record(result(), now=AT)

    assert len(log) == 10


# ------------------------------------------------------------------ summary

def test_an_empty_log_says_so_plainly():
    assert "Nothing has run yet" in RunLog().summary()


def test_the_summary_counts_requests_time_and_the_bad_outcomes():
    log = RunLog()
    for _ in range(3):
        log.record(result(done_reason="length"), now=AT)
    log.record(result(done_reason="stop", elapsed_s=10.0), now=AT)
    log.record_failure("m", "host", "chat", "boom", now=AT)

    summary = log.summary()

    assert "5 requests" in summary
    assert "3 cut off at the token cap" in summary
    assert "1 failed" in summary
    assert "1m 48s" in summary          # 3*32.8 + 10


def test_the_slowest_request_can_be_found():
    log = RunLog()
    log.record(result(elapsed_s=5.0), now=AT)
    log.record(result(model="slow:70b", elapsed_s=240.0), now=AT)

    assert log.slowest(1)[0].model == "slow:70b"


def test_clearing_empties_the_log():
    log = RunLog()
    log.record(result(), now=AT)
    log.clear()

    assert len(log) == 0


# ---------------------------------------------------------------- exporting

def test_csv_has_a_header_and_one_row_per_request():
    log = RunLog()
    log.record(result(done_reason="length"), purpose="chat", now=AT)
    log.record(result(done_reason="stop"), purpose="compaction", now=AT)

    lines = log.to_csv().strip().splitlines()

    assert lines[0].startswith("time,model,server,purpose")
    assert len(lines) == 3
    assert "compaction" in lines[2]


def test_csv_can_be_written_to_disk(tmp_path):
    log = RunLog()
    log.record(result(), now=AT)
    path = tmp_path / "runlog.csv"

    write_csv(path, log.entries)

    assert "gpt-oss:20b" in path.read_text(encoding="utf-8")
