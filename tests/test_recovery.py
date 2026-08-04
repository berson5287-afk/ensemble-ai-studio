"""Keeping an unfinished conversation across a close or a crash."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from aichatlab.recovery import Recovered, discard, read, write
from aichatlab.session import Session

NOW = datetime(2026, 8, 2, 14, 32, 0)


def chat() -> Session:
    session = Session()
    session.add("host::gpt-oss:20b", "user", "take a look at my app")
    session.add("host::gpt-oss:20b", "assistant", "Here are a few ideas.")
    return session


# ------------------------------------------------------------ round tripping

def test_an_unfinished_chat_survives_a_write_and_read(tmp_path):
    path = tmp_path / "unfinished.json"

    assert write(chat(), selected=["host::gpt-oss:20b"], mode="plan",
                 attachments=["udbg-phase1/"], path=path, now=NOW)
    found = read(path)

    assert found is not None
    assert found.messages == 2
    assert found.selected == ["host::gpt-oss:20b"]
    assert found.mode == "plan"
    assert found.attachments == ["udbg-phase1/"]
    assert "take a look at my app" in found.session.transcript()


def test_an_attachment_is_still_pinned_after_recovery(tmp_path):
    """Recovery goes through the same cleaning as opening a saved file."""
    path = tmp_path / "unfinished.json"
    session = Session()
    session.add("local::alpha:1b", "user",
                "[Folder: app — 2 of 2 files included]\nx\n\n"
                "[End of folder contents.]")
    write(session, path=path, now=NOW)

    found = read(path)

    assert found.session.conversations["local::alpha:1b"][0]["pinned"]


def test_an_empty_chat_is_not_worth_saving(tmp_path):
    path = tmp_path / "unfinished.json"

    assert not write(Session(), path=path, now=NOW)
    assert not path.exists()


def test_nothing_is_recovered_when_there_is_no_file(tmp_path):
    assert read(tmp_path / "missing.json") is None


def test_a_corrupt_file_recovers_nothing_rather_than_crashing(tmp_path):
    path = tmp_path / "unfinished.json"
    path.write_text("{not json at all", encoding="utf-8")

    assert read(path) is None


def test_a_file_with_no_conversation_recovers_nothing(tmp_path):
    path = tmp_path / "unfinished.json"
    path.write_text(json.dumps({"schema": 1, "chat": {"conversations": {}}}),
                    encoding="utf-8")

    assert read(path) is None


def test_writing_is_atomic_so_a_crash_cannot_leave_half_a_file(tmp_path):
    path = tmp_path / "unfinished.json"
    write(chat(), path=path, now=NOW)

    assert path.exists()
    assert not path.with_suffix(".part").exists()
    json.loads(path.read_text(encoding="utf-8"))       # must parse


def test_a_second_write_replaces_the_first(tmp_path):
    path = tmp_path / "unfinished.json"
    write(chat(), path=path, now=NOW)

    longer = chat()
    longer.add("host::gpt-oss:20b", "user", "and the tests?")
    write(longer, path=path, now=NOW)

    assert read(path).messages == 3


def test_discarding_removes_the_file_and_any_leftovers(tmp_path):
    path = tmp_path / "unfinished.json"
    write(chat(), path=path, now=NOW)
    path.with_suffix(".part").write_text("junk", encoding="utf-8")

    discard(path)

    assert not path.exists()
    assert not path.with_suffix(".part").exists()


def test_discarding_nothing_is_harmless(tmp_path):
    discard(tmp_path / "never-existed.json")


def test_a_write_to_an_impossible_place_fails_quietly(tmp_path):
    """Insurance must never be the thing that breaks the app."""
    assert not write(chat(), path=tmp_path / "no" / "such" / "dir.json")


# --------------------------------------------------------------- describing

def test_the_note_says_when_and_what():
    found = Recovered(session=chat(), at=NOW.isoformat(),
                      attachments=["udbg-phase1/"])

    text = found.describe(now=NOW)

    assert "14:32 today" in text
    assert "udbg-phase1/" in text
    assert "2 messages" in text


def test_yesterdays_chat_says_yesterday():
    found = Recovered(session=chat(),
                      at=(NOW - timedelta(days=1)).isoformat())

    assert found.when(now=NOW) == "yesterday at 14:32"


def test_an_older_chat_gives_the_date():
    found = Recovered(session=chat(),
                      at=(NOW - timedelta(days=4)).isoformat())

    assert "at 14:32" in found.when(now=NOW)
    assert "July" in found.when(now=NOW)


def test_a_missing_timestamp_does_not_produce_nonsense():
    assert Recovered(session=chat(), at="").when(now=NOW) == "earlier"
    assert Recovered(session=chat(), at="banana").when(now=NOW) == "earlier"


def test_one_message_is_not_pluralised():
    session = Session()
    session.add("local::alpha:1b", "user", "hello")

    assert "1 message." in Recovered(session=session, at=NOW.isoformat()).describe(NOW)
