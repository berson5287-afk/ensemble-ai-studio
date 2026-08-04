"""The record of why nothing was written.

These tests care about one thing: that the six ways an edit check can end up
writing nothing stay distinguishable. If two of them collapse into the same
report, the module has stopped being useful — telling them apart is the only
reason it exists.
"""

from __future__ import annotations

import json

from aichatlab import editdebug

PROSE = """\
I'll update the config loader to read the timeout from the environment, and
change the retry helper so it backs off exponentially. That should fix it.
"""

CODE_ONLY = """\
Here's the fix:

```python
def load(path):
    return json.loads(path.read_text())
```

Save that over the old version.
"""

UNCLOSED = """\
=== FILE: loader.py ===
def load(path):
    return path.read_text()
"""

GOOD_PATCH = """\
=== EDIT: loader.py ===
--- FIND
def load(path):
--- REPLACE
def load(path, encoding="utf-8"):
=== END EDIT ===
"""


# -- reading the shape of a reply -----------------------------------------
def test_prose_is_reported_as_having_no_markers_and_no_code():
    note = editdebug.shape_note(editdebug.shape(PROSE))

    assert "prose only" in note
    assert "never anything to apply" in note


def test_a_reply_that_is_only_a_code_fence_says_so():
    info = editdebug.shape(CODE_ONLY)

    assert info["code_fences"] == 2
    assert info["file_openers"] == 0
    note = editdebug.shape_note(info)
    assert "fenced code block" in note
    assert "cannot apply" in note


def test_an_unclosed_file_block_is_named_as_the_problem():
    info = editdebug.shape(UNCLOSED)

    assert info["file_openers"] == 1
    assert info["file_closers"] == 0
    assert "END FILE" in editdebug.shape_note(info)


def test_ordinary_prose_is_not_mistaken_for_a_malformed_marker():
    # "Replace the timeout" at the start of a sentence is English, not a
    # botched FIND/REPLACE block.
    reply = ("Replace the timeout with 30 seconds.\n"
             "Find the retry helper and change the backoff.\n")

    assert editdebug.shape(reply)["near_misses"] == 0
    assert "prose only" in editdebug.shape_note(editdebug.shape(reply))


def test_a_marker_the_model_decorated_wrongly_is_still_spotted():
    reply = ("### EDIT: loader.py\n"
             "some code here\n")
    info = editdebug.shape(reply)

    assert info["near_misses"] >= 1
    assert info["edit_openers"] == 0
    assert "wrong syntax" in editdebug.shape_note(info)


def test_a_well_formed_patch_draws_no_complaint():
    info = editdebug.shape(GOOD_PATCH)

    assert info["edit_blocks"] == 1
    assert info["find_parts"] == 1
    assert editdebug.shape_note(info) == ""


def test_shape_counts_survive_an_empty_reply():
    info = editdebug.shape("")

    assert info["chars"] == 0
    assert editdebug.shape_note(info)      # says something rather than crashing


# -- a trace start to finish ----------------------------------------------
def test_a_trace_records_the_steps_it_was_given():
    trace = editdebug.begin(PROSE, project="/tmp/proj")
    trace.step("whole-file blocks parsed", [])
    trace.step("anchored patches parsed", [])
    trace.done("prose-unmatched")

    assert not trace.offered
    text = trace.explain()
    assert "prose-unmatched" in text
    assert "/tmp/proj" in text
    assert "whole-file blocks parsed: 0" in text


def test_a_trace_that_reached_a_diff_is_marked_as_offered():
    trace = editdebug.begin(GOOD_PATCH).done("offered")

    assert trace.offered

    auto = editdebug.begin(GOOD_PATCH).done("auto-applied")
    assert auto.offered


def test_rejections_keep_their_reasons():
    class Refusal:
        name = "loader.py"
        reason = "the lines it quoted are not in the file"

    trace = editdebug.begin(GOOD_PATCH)
    trace.refused([Refusal()])
    trace.done("all-rejected")

    assert "the lines it quoted are not in the file" in trace.explain()


def test_a_step_lists_the_files_it_saw_without_running_on_forever():
    class Edit:
        def __init__(self, name):
            self.name = name

    trace = editdebug.begin("")
    trace.step("changes", [Edit(f"f{n}.py") for n in range(12)])

    line = trace.steps[0].line()
    assert line.startswith("changes: 12")
    assert "f0.py" in line
    assert "f11.py" not in line       # truncated, with an ellipsis
    assert "…" in line


def test_a_claim_to_have_written_is_flagged():
    trace = editdebug.begin(PROSE)
    trace.claimed_to_write = "I've saved the file"
    trace.done("prose-unmatched")

    assert "claimed to have written" in trace.explain()


# -- the recorder ----------------------------------------------------------
def test_traces_are_kept_and_summarised(tmp_path):
    debug = editdebug.EditDebug(path=tmp_path / "trace.jsonl")
    debug.record(editdebug.begin(GOOD_PATCH).done("offered"))
    debug.record(editdebug.begin(PROSE).done("prose-unmatched"))
    debug.record(editdebug.begin(PROSE).done("prose-unmatched"))

    assert len(debug) == 3
    assert len(debug.failures()) == 2
    summary = debug.summary()
    assert "3 checks" in summary
    assert "1 reached a diff" in summary
    assert "2 prose-unmatched" in summary


def test_the_recorder_is_bounded():
    debug = editdebug.EditDebug(path=None, limit=3, enabled=True)
    debug.enabled = False           # don't touch the real home directory
    for _ in range(10):
        debug.record(editdebug.begin("").done("offered"))

    assert len(debug) == 0          # disabled records nothing

    debug.enabled = True
    debug.path = None


def test_the_limit_drops_the_oldest(tmp_path):
    debug = editdebug.EditDebug(path=tmp_path / "t.jsonl", limit=3)
    for n in range(10):
        trace = editdebug.begin("")
        trace.at = f"00:00:{n:02d}"
        debug.record(trace.done("offered"))

    assert len(debug) == 3
    assert debug.last().at == "00:00:09"


def test_traces_are_written_as_readable_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    debug = editdebug.EditDebug(path=path)
    debug.record(editdebug.begin(CODE_ONLY, project="/p").done("prose-only"))

    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["verdict"] == "prose-only"
    assert rows[0]["project"] == "/p"
    assert "fenced code block" in rows[0]["shape_note"]


def test_reading_traces_back_skips_a_half_written_line(tmp_path):
    path = tmp_path / "trace.jsonl"
    debug = editdebug.EditDebug(path=path)
    debug.record(editdebug.begin("").done("offered"))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"verdict": "trunc')      # app closed mid-append

    found = editdebug.read_traces(path)

    assert len(found) == 1
    assert found[0]["verdict"] == "offered"


def test_reading_a_missing_file_is_not_an_error(tmp_path):
    assert editdebug.read_traces(tmp_path / "nope.jsonl") == []


def test_a_write_failure_never_reaches_the_caller(tmp_path):
    # A debugging aid that can break the feature it debugs is worse than none.
    debug = editdebug.EditDebug(path=tmp_path / "no-such-dir" / "t.jsonl")
    debug.record(editdebug.begin("").done("offered"))

    assert len(debug) == 1          # kept in memory regardless


def test_the_reply_is_kept_but_bounded():
    trace = editdebug.begin("x" * (editdebug.REPLY_KEEP + 5000))

    assert len(trace.reply_head) == editdebug.REPLY_KEEP


def test_every_verdict_the_app_uses_has_wording():
    for verdict in ("offered", "auto-applied", "no-folder", "edits-off",
                    "all-rejected", "no-change", "prose-offered",
                    "prose-unmatched", "no-target"):
        assert editdebug.VERDICTS[verdict]


def test_report_is_empty_but_polite_before_anything_runs():
    assert "not run yet" in editdebug.EditDebug().report()


# -- the bridge handshake, when several windows fight over one file --------
def test_the_handshake_is_taken_back_when_it_goes_stale(tmp_path):
    """A second window that is killed leaves the file naming a dead port."""
    import json
    import os

    from aichatlab.bridge import Bridge

    class FakeApp:
        root = None

    path = tmp_path / "bridge.json"
    live = Bridge(FakeApp(), handshake_path=path, port=0)
    live.port = 5555
    live._write_handshake()
    assert live.handshake_is_mine()
    assert not live.reassert(), "nothing to do while it is already ours"

    # another window takes it over, then dies without tidying up
    path.write_text(json.dumps(
        {"host": "127.0.0.1", "port": 6666, "token": "other", "pid": 999999}),
        encoding="utf-8")
    assert not live.handshake_is_mine()

    assert live.reassert(), "the live window takes it back"
    found = json.loads(path.read_text(encoding="utf-8"))
    assert found["port"] == 5555
    assert found["pid"] == os.getpid()
    assert found["token"] == live.token


def test_a_deleted_handshake_is_rewritten(tmp_path):
    from aichatlab.bridge import Bridge

    class FakeApp:
        root = None

    path = tmp_path / "bridge.json"
    live = Bridge(FakeApp(), handshake_path=path, port=0)
    live.port = 4444
    live._write_handshake()
    path.unlink()

    assert live.reassert()
    assert path.exists()


def test_a_stopped_bridge_does_not_reclaim_the_file(tmp_path):
    from aichatlab.bridge import Bridge

    class FakeApp:
        root = None

    path = tmp_path / "bridge.json"
    live = Bridge(FakeApp(), handshake_path=path, port=0)
    live.port = 3333
    live._write_handshake()
    live.stop()

    assert not live.reassert()
    assert not path.exists()
