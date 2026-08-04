"""History, context budgeting and save/load round-trips."""

from __future__ import annotations

import json

import pytest

from aichatlab.session import (
    Session,
    attachment_allowance,
    attachment_details,
    describe_attachments,
    estimate_tokens,
    make_key,
    split_key,
)


def test_split_key_handles_v1_files():
    assert split_key("host::llama3:8b") == ("host", "llama3:8b")
    assert split_key("local::gemma2:9b") == ("local", "gemma2:9b")
    # v1 saved bare model names, and model names contain colons
    assert split_key("llama3:8b") == ("local", "llama3:8b")


def test_add_and_history():
    session = Session()
    key = make_key("local", "m:1b")
    session.add(key, "user", "hello")
    session.add(key, "assistant", "hi")

    assert [m["role"] for m in session.history(key)] == ["user", "assistant"]
    assert session.is_empty() is False


def test_rejects_unknown_role():
    with pytest.raises(ValueError):
        Session().add("k", "wizard", "hello")


def test_build_context_keeps_recent_turns_within_budget():
    session = Session()
    key = "local::m"
    for _index in range(30):
        session.add(key, "user", "x" * 400)        # ~100 tokens each

    context = session.build_context(key, budget_tokens=500)
    kept = [m for m in context if m["role"] == "user"]

    assert 1 <= len(kept) <= 6                     # far fewer than 30
    assert context[0]["role"] == "system"
    assert "trimmed" in context[0]["content"]


def test_build_context_keeps_system_prompt():
    session = Session()
    key = "local::m"
    session.add(key, "user", "hello")

    context = session.build_context(key, budget_tokens=1000,
                                    system_prompt="You are terse.")

    assert context[0] == {"role": "system", "content": "You are terse."}
    assert context[-1]["content"] == "hello"


def test_build_context_always_keeps_at_least_the_latest_message():
    """A single huge attachment must not be trimmed away to nothing."""
    session = Session()
    key = "local::m"
    session.add(key, "user", "x" * 100_000)

    context = session.build_context(key, budget_tokens=100)
    assert any(m["role"] == "user" for m in context)


def test_save_load_round_trip(tmp_path):
    session = Session()
    key = make_key("host", "qwen:7b")
    session.add(key, "user", "say hello")
    session.add(key, "assistant", "Hello! 👋 café")

    path = session.save(tmp_path / "chat.json")
    loaded, _selected = Session.load(path)

    assert loaded.conversations == session.conversations
    assert "👋" in loaded.history(key)[1]["content"]


def test_load_normalises_legacy_files(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "conversations": {"llama3:8b": [{"role": "user", "content": "hi"}]},
        "selected_models": ["llama3:8b"],
    }), encoding="utf-8")

    session, selected = Session.load(path)

    assert "local::llama3:8b" in session.conversations
    assert selected == ["local::llama3:8b"]


def test_load_drops_malformed_messages(tmp_path):
    path = tmp_path / "messy.json"
    path.write_text(json.dumps({"conversations": {"local::m": [
        {"role": "user", "content": "good"},
        {"role": "user", "content": 12345},        # not a string
        "not a dict",
        {"role": "hacker", "content": "nope"},
    ]}}), encoding="utf-8")

    session, _ = Session.load(path)
    assert session.history("local::m") == [{"role": "user", "content": "good"}]


def test_load_rejects_nonsense(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"conversations": "not a dict"}), encoding="utf-8")

    with pytest.raises(ValueError):
        Session.load(path)


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("word " * 100) > estimate_tokens("word " * 50)
    doubled = estimate_tokens("word " * 200) / estimate_tokens("word " * 100)
    assert 1.8 < doubled < 2.2


def test_code_is_priced_higher_per_character_than_prose():
    """The whole point: a folder of code does not tokenise like English.

    Pricing it as English built prompts that overflowed the window, and
    Ollama's answer to an overflowing prompt is to keep the most recent half —
    throwing away the front, where the system prompt lives.
    """
    prose = ("the quick brown fox jumps over the lazy dog and keeps on "
             "running through the field ") * 40
    code = ('    def handle(self, row):\n'
            '        if (row.get("ai_confidence") or 0.0) < self.min_conf():\n'
            '            return None\n') * 20

    prose_rate = len(prose) / estimate_tokens(prose)
    code_rate = len(code) / estimate_tokens(code)

    assert code_rate < prose_rate
    assert code_rate < 3.0, "code runs about 2.5 characters per token"


def test_indentation_and_punctuation_are_paid_for():
    plain = "x = 1\n" * 20
    nested = "                x = 1\n" * 20

    assert estimate_tokens(nested) > estimate_tokens(plain)


# ------------------------------------------- attachments must not be trimmed

FOLDER = ("[Folder: udbg-phase1 — 10 of 24 files included]\nFiles included:\n"
          "  udbg/cli.py (17 KB)\n\n--- udbg/cli.py ---\n```\n"
          + "source code line\n" * 4000 +
          "```\n\n[End of folder contents.]")
FILE = "[Attached file: notes.txt]\n```\n" + "note line\n" * 100 + "```"


def test_a_message_carrying_a_folder_is_pinned_automatically():
    session = Session()
    message = session.add("local::m", "user", FOLDER + "\n\nwhat does this do?")

    assert message["pinned"]


def test_an_ordinary_message_is_not_pinned():
    session = Session()

    assert "pinned" not in session.add("local::m", "user", "hello there")
    assert "pinned" not in session.add("local::m", "assistant", FOLDER)


def test_an_attachment_survives_follow_up_turns():
    """The bug: reopen a chat with a folder, ask one more thing, and the
    document the whole conversation is about is silently trimmed away —
    because it is the oldest message and trimming is oldest-first."""
    session = Session()
    session.add("k", "user", FOLDER + "\n\nwhat does this do?")
    session.add("k", "assistant", "It is a debugger.")
    for index in range(6):
        session.add("k", "user", f"follow-up {index} " + "x" * 2000)
        session.add("k", "assistant", f"answer {index} " + "y" * 2000)

    context = session.build_context("k", budget_tokens=28_000)

    assert any("[Folder:" in m["content"] for m in context)


def test_an_attachment_too_big_for_the_budget_is_shortened_not_dropped():
    session = Session()
    session.add("k", "user", FOLDER)
    session.add("k", "user", "and now a question")

    context = session.build_context("k", budget_tokens=2_000)

    folder = next(m for m in context if "[Folder:" in m["content"])
    assert "attachment shortened" in folder["content"]
    assert "Files included:" in folder["content"]      # the manifest survives
    assert estimate_tokens(folder["content"]) <= 2_000


def test_the_attachment_never_takes_the_whole_budget():
    """A document sized to the entire budget leaves no room for the question
    about it, so the next turn pushes it straight back out."""
    session = Session()
    session.add("k", "user", FOLDER)

    context = session.build_context("k", budget_tokens=10_000)
    used = sum(estimate_tokens(m["content"]) for m in context)

    assert used < 10_000
    assert attachment_allowance(10_000) < 10_000


def test_ordinary_turns_are_still_trimmed_oldest_first():
    session = Session()
    for index in range(20):
        session.add("k", "user", f"turn {index} " + "x" * 4000)

    context = session.build_context("k", budget_tokens=3_000)

    assert any("trimmed to fit" in m["content"] for m in context)
    assert "turn 19" in context[-1]["content"]
    assert not any("turn 0 " in m["content"] for m in context)


def test_the_newest_message_is_kept_even_if_it_alone_blows_the_budget():
    session = Session()
    session.add("k", "user", "old")
    session.add("k", "user", "huge " + "x" * 100_000)

    context = session.build_context("k", budget_tokens=100)

    assert "huge" in context[-1]["content"]


def test_our_bookkeeping_never_reaches_the_inference_server():
    session = Session()
    session.add("k", "user", FOLDER)
    session.add("k", "assistant", "ok")

    context = session.build_context("k", budget_tokens=50_000, system_prompt="p")

    assert all(set(m) == {"role", "content"} for m in context)


def test_two_attachments_favour_the_more_recent_one():
    session = Session()
    session.add("k", "user", FOLDER)
    session.add("k", "user", FILE)

    context = session.build_context("k", budget_tokens=1_200)

    assert any("notes.txt" in m["content"] for m in context)


# ---------------------------------------------------- saving and reloading

def test_pinning_survives_a_save_and_load(tmp_path):
    session = Session()
    session.add("local::m", "user", FOLDER)
    path = tmp_path / "chat.json"
    session.save(path)

    loaded, _selected = Session.load(path)

    assert loaded.conversations["local::m"][0]["pinned"]


def test_a_chat_saved_before_pinning_existed_is_still_protected(tmp_path):
    """The marker is in the message text, so old files get the fix too."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "schema": 2,
        "conversations": {"host::gpt-oss:20b": [
            {"role": "user", "content": FOLDER},
            {"role": "assistant", "content": "some reply"}]},
        "selected_models": ["host::gpt-oss:20b"],
    }), encoding="utf-8")

    loaded, selected = Session.load(path)
    context = loaded.build_context("host::gpt-oss:20b", budget_tokens=28_000)

    assert loaded.conversations["host::gpt-oss:20b"][0]["pinned"]
    assert any("[Folder:" in m["content"] for m in context)
    assert selected == ["host::gpt-oss:20b"]


# ------------------------------------------------- describing what was sent

def test_a_folder_message_reduces_to_its_name_and_the_typed_question():
    names, typed = describe_attachments(FOLDER + "\n\nwhat does this do?")

    assert names == ["udbg-phase1/"]
    assert typed == "what does this do?"


def test_a_file_message_reduces_to_its_name():
    names, typed = describe_attachments(FILE + "\n\nsummarise this")

    assert names == ["notes.txt"]
    assert typed == "summarise this"


def test_several_attachments_are_all_named():
    names, _typed = describe_attachments(f"{FOLDER}\n\n{FILE}\n\nboth please")

    assert names == ["udbg-phase1/", "notes.txt"]


def test_an_ordinary_message_is_returned_unchanged():
    names, typed = describe_attachments("just a question")

    assert names == []
    assert typed == "just a question"


def test_an_attachment_sent_with_no_typed_text_says_so():
    names, typed = describe_attachments(FOLDER)

    assert names == ["udbg-phase1/"]
    assert typed == ""


def test_an_attached_file_that_mentions_the_end_marker_is_still_fully_stripped():
    """Attaching this project includes folderscan.py, whose source contains
    the literal end marker — an unanchored match stopped there and left half
    the block sitting in the transcript."""
    tricky = ('[Folder: ai-chat-lab — 2 of 2 files included]\nFiles included:\n'
              '  folderscan.py (9 KB)\n\n--- folderscan.py ---\n```\n'
              'return header + "\\n\\n[End of folder contents.]"\n'
              'more source\n```\n\n[End of folder contents.]\n\nreview this')

    names, typed = describe_attachments(tricky)

    assert names == ["ai-chat-lab/"]
    assert typed == "review this"
    assert "more source" not in typed


def test_a_file_whose_contents_contain_a_fence_is_still_stripped():
    tricky = ("[Attached file: guide.md]\n```\n"
              "here is an example:\n```\nprint(1)\n```\ndone\n```\n\nsummarise")

    names, typed = describe_attachments(tricky)

    assert names == ["guide.md"]
    assert typed == "summarise"


def test_an_attachment_reports_its_size_so_a_chip_is_not_mistaken_for_nothing():
    """A reopened chat showing one question and one chip reads as empty when
    it is in fact carrying 24,000 tokens."""
    details = attachment_details(FOLDER + "\n\nwhat does this do?")

    assert len(details) == 1
    name, tokens = details[0]
    assert name == "udbg-phase1/"
    assert tokens > 15_000


def test_several_attachments_are_each_sized():
    details = attachment_details(f"{FOLDER}\n\n{FILE}\n\nboth please")

    assert [name for name, _ in details] == ["udbg-phase1/", "notes.txt"]
    assert all(tokens > 0 for _, tokens in details)


def test_a_message_with_no_attachment_has_no_details():
    assert attachment_details("just a question") == []
