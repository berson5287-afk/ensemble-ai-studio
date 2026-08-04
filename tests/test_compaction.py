"""Folding an old conversation into a digest instead of dropping it."""

from __future__ import annotations

import pytest

from aichatlab.compaction import (
    KEEP_RECENT,
    SUMMARY_PREFIX,
    apply,
    build_message,
    build_prompt,
    clean_summary,
    is_summary,
    plan,
    should_compact,
    transcript_for,
    usage,
)


def chat(turns: int, size: int = 400) -> list[dict]:
    """A conversation of alternating turns, each `size` characters long."""
    messages = []
    for index in range(turns):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"m{index} " + "x" * size})
    return messages


# ------------------------------------------------------------ when to compact

def test_a_short_conversation_is_left_alone():
    assert not should_compact(chat(2), budget=28_000)


def test_a_conversation_near_the_budget_is_worth_compacting():
    messages = chat(40, size=2000)          # ~20k tokens

    assert should_compact(messages, budget=24_000)


def test_a_long_conversation_with_a_huge_budget_is_left_alone():
    assert not should_compact(chat(40, size=2000), budget=500_000)


def test_a_tiny_conversation_is_never_compacted_however_small_the_budget():
    """Below a floor, folding costs a model call and buys almost nothing."""
    assert not should_compact(chat(2, size=100), budget=100)


# ------------------------------------------------------------------ planning

def test_recent_turns_are_kept_verbatim():
    result = plan(chat(20))

    assert len(result.keep) == KEEP_RECENT
    assert result.keep[-1]["content"].startswith("m19")
    assert result.fold[0]["content"].startswith("m0")
    assert result.folded_count == 16


def test_nothing_is_folded_when_the_chat_is_all_recent():
    result = plan(chat(3))

    assert result.fold == []
    assert not result.worth_doing


def test_an_existing_summary_is_carried_forward_not_re_folded():
    """Otherwise each pass summarises the previous summary and detail rots."""
    messages = [build_message("older stuff happened", 8), *chat(20)]

    result = plan(messages)

    assert result.previous_summary.startswith(SUMMARY_PREFIX)
    assert not any(is_summary(m) for m in result.fold)
    assert "older stuff happened" in build_prompt(result)
    assert "Carry anything still relevant" in build_prompt(result)


def test_a_folded_stretch_of_real_size_is_worth_doing():
    assert plan(chat(20)).worth_doing


# -------------------------------------------------------------------- prompt

def test_the_prompt_asks_for_the_things_a_successor_would_need():
    prompt = build_prompt(plan(chat(20)))

    for wanted in ("trying to do", "Decisions made", "still open",
                   "file paths", "not present"):
        assert wanted in prompt


def test_the_transcript_keeps_the_end_when_it_must_cut():
    messages = [{"role": "user", "content": "OLDEST " + "x" * 5000},
                {"role": "user", "content": "NEWEST"}]

    text = transcript_for(messages, max_chars=500)

    assert "NEWEST" in text
    assert "earlier turns omitted" in text


def test_empty_messages_are_skipped_in_the_transcript():
    text = transcript_for([{"role": "user", "content": "   "},
                           {"role": "user", "content": "real"}])

    assert text == "User: real"


# ------------------------------------------------------------------ applying

def test_applying_replaces_the_old_turns_with_one_summary():
    messages = chat(20)

    result = apply(messages, "They agreed to use Postgres.")

    assert len(result) == KEEP_RECENT + 1
    assert is_summary(result[0])
    assert "Postgres" in result[0]["content"]
    assert result[-1]["content"].startswith("m19")


def test_compaction_actually_saves_tokens():
    messages = chat(30, size=2000)

    after = apply(messages, "a short summary")

    assert usage(after) < usage(messages) / 2


def test_compacting_twice_does_not_stack_digests():
    once = apply(chat(20), "first summary")
    twice = apply(once + chat(10), "second summary")

    assert sum(1 for m in twice if is_summary(m)) == 1
    assert "second summary" in twice[0]["content"]


def test_the_second_digest_counts_the_first_in_its_total():
    once = apply(chat(20), "first summary")          # [digest, m16..m19]
    twice = apply(once + chat(10), "second summary")

    # 14 messages remain after the digest is set aside; 4 are kept verbatim,
    # so 10 are folded — plus the digest that was absorbed into the new one.
    assert "11 earlier message(s)" in twice[0]["content"]


def test_nothing_happens_when_there_is_nothing_to_fold():
    messages = chat(3)

    assert apply(messages, "summary") == messages


def test_the_digest_is_a_system_message_so_it_is_not_mistaken_for_a_turn():
    result = apply(chat(20), "summary")

    assert result[0]["role"] == "system"
    assert "established context" in result[0]["content"]


# --------------------------------------------------------------- tidying up

@pytest.mark.parametrize("reply, expected", [
    ("Here is a summary: they chose Postgres.", "they chose Postgres."),
    ("Summary: they chose Postgres.", "they chose Postgres."),
    ("Here's the summary of the chat: X", "X"),
    ("They chose Postgres.", "They chose Postgres."),
])
def test_model_preambles_are_stripped(reply, expected):
    assert clean_summary(reply) == expected


def test_a_summary_that_quotes_the_marker_cannot_forge_a_second_digest():
    cleaned = clean_summary(f"{SUMMARY_PREFIX} sneaky text")

    assert not cleaned.startswith(SUMMARY_PREFIX)


# --------------------------------- compaction must not eat your attachments

FOLDER = ("[Folder: udbg-phase1 — 10 of 24 files included]\nFiles included:\n"
          "  udbg/cli.py (17 KB)\n\n--- udbg/cli.py ---\n```\n"
          + "source line\n" * 2000 + "```\n\n[End of folder contents.]")


def test_an_attached_folder_is_never_folded_into_the_summary():
    """Summarising an attachment is destructive in a way nothing else is: a
    four-line précis of a source tree cannot be turned back into the tree.
    build_context already refuses to trim them — the Compact button must not
    delete them instead."""
    messages = [{"role": "user", "content": FOLDER, "pinned": True}, *chat(20)]

    result = plan(messages)

    assert not any("[Folder:" in m["content"] for m in result.fold)
    assert any("[Folder:" in m["content"] for m in result.pinned)


def test_the_folder_is_still_there_after_compacting():
    messages = [{"role": "user", "content": FOLDER, "pinned": True}, *chat(20)]

    after = apply(messages, "they discussed the debugger")

    assert any("[Folder:" in m["content"] for m in after)
    assert "they discussed the debugger" in after[1]["content"]


def test_an_attachment_in_a_chat_saved_before_pinning_is_also_spared():
    """Detected from the message text, so old saved chats are safe too."""
    messages = [{"role": "user", "content": FOLDER}, *chat(20)]

    assert any("[Folder:" in m["content"] for m in plan(messages).pinned)


def test_the_attachment_comes_before_the_digest():
    """Roughly chronological: an attachment is what started the conversation."""
    messages = [{"role": "user", "content": FOLDER, "pinned": True}, *chat(20)]

    after = apply(messages, "summary")

    assert "[Folder:" in after[0]["content"]
    assert is_summary(after[1])


def test_compaction_still_saves_tokens_with_an_attachment_present():
    messages = [{"role": "user", "content": FOLDER, "pinned": True},
                *chat(30, size=2000)]

    after = apply(messages, "a short summary")

    assert usage(after) < usage(messages)


def test_a_chat_that_is_only_an_attachment_has_nothing_worth_folding():
    messages = [{"role": "user", "content": FOLDER, "pinned": True},
                {"role": "assistant", "content": "here are some tips"}]

    assert not plan(messages).worth_doing
    assert apply(messages, "summary") == messages


# -- why folding will not help --------------------------------------------
FOLDER = ("[Folder: udbg-phase1 — 41 files]\n" + "x" * 200_000
          + "\n[End of folder contents.]")


def big_attachment_chat() -> list[dict]:
    """The shape that broke it: one huge pinned folder, a short discussion."""
    return [
        {"role": "user", "content": FOLDER, "pinned": True},
        {"role": "assistant", "content": "I have read all 41 files."},
        {"role": "user", "content": "what would you change first?"},
        {"role": "assistant", "content": "The loader rebuilds the symbol "
                                         "table on every open."},
    ]


def test_a_conversation_full_of_attachment_is_not_called_short():
    """It said "still short enough" while the line above it said the oldest
    turns were being dropped — two statements that cannot both be true."""
    from aichatlab.compaction import explain_no_compaction

    reason = explain_no_compaction(big_attachment_chat(), budget=28_000)

    assert "short enough" not in reason
    assert "nothing here that can be folded" in reason
    assert "udbg-phase1" in reason


def test_the_explanation_gives_the_actual_numbers():
    from aichatlab.compaction import explain_no_compaction, usage

    chat = big_attachment_chat()
    reason = explain_no_compaction(chat, budget=28_000)

    assert f"{usage(chat):,}" in reason, "it should say how big it really is"
    assert "new chat" in reason and "context budget" in reason


def test_a_genuinely_short_chat_is_still_called_short():
    from aichatlab.compaction import explain_no_compaction

    reason = explain_no_compaction(
        [{"role": "user", "content": "hello"},
         {"role": "assistant", "content": "hi"}], budget=28_000)

    assert "short enough" in reason


def test_a_long_chat_with_nothing_old_enough_says_so():
    from aichatlab.compaction import explain_no_compaction

    # Over budget, no attachments, but only a couple of recent turns.
    chat = [{"role": "user", "content": "q " * 60_000},
            {"role": "assistant", "content": "a " * 60_000}]

    reason = explain_no_compaction(chat, budget=28_000)

    assert "nothing old enough to fold" in reason
    assert "context budget" in reason


def test_pinned_tokens_are_measured_not_guessed():
    from aichatlab.compaction import pinned_tokens, usage

    chat = big_attachment_chat()

    assert pinned_tokens(chat) > usage(chat) * 0.9
