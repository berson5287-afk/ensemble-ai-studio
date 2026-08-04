"""Deciding whether a question needs the web before going to fetch it."""

from __future__ import annotations

import pytest

from aichatlab.triage import (
    Triage,
    build_prompt,
    describe_material,
    parse,
    worth_triaging,
)

# ------------------------------------------------------------------ parsing

def test_a_plain_answer_verdict_means_no_search():
    verdict = parse("ANSWER")

    assert not verdict.needs_search
    assert "no search needed" in verdict.note


def test_a_search_verdict_carries_the_model_s_own_query():
    verdict = parse("SEARCH: Ollama num_ctx default value")

    assert verdict.needs_search
    assert verdict.query == "Ollama num_ctx default value"
    assert "Ollama num_ctx" in verdict.note


@pytest.mark.parametrize("reply", [
    "I think the answer is: SEARCH: latest RTX 5090 price",
    "SEARCH:latest RTX 5090 price",
    'search: "latest RTX 5090 price"',
    "Decision: SEARCH: latest RTX 5090 price\nBecause prices change.",
])
def test_the_verdict_is_found_however_the_model_wraps_it(reply):
    verdict = parse(reply)

    assert verdict.needs_search
    assert verdict.query == "latest RTX 5090 price"


def test_only_the_first_line_of_a_query_is_used():
    verdict = parse("SEARCH: copper prices today\nI'll check LME too.")

    assert verdict.query == "copper prices today"


def test_a_very_long_query_is_trimmed_at_a_word_boundary():
    verdict = parse("SEARCH: " + "word " * 80)

    assert len(verdict.query) <= 160
    assert not verdict.query.endswith("wor")


def test_an_unclear_reply_defaults_to_not_searching():
    """A needless search costs a slow round-trip and pollutes the prompt;
    a needlessly skipped one costs one follow-up message."""
    verdict = parse("Well, it depends on what you mean.")

    assert not verdict.needs_search
    assert "unclear" in verdict.reason


def test_an_empty_reply_defaults_to_not_searching():
    assert not parse("").needs_search
    assert not parse("   ").needs_search


def test_a_search_verdict_with_no_query_falls_back_when_one_is_offered():
    verdict = parse("SEARCH:", fallback_query="original question")

    assert verdict.needs_search
    assert verdict.query == "original question"


def test_a_search_verdict_with_no_query_and_no_fallback_does_not_search():
    assert not parse("SEARCH: ").needs_search


# ------------------------------------------------------------------ material

def test_the_inventory_names_files_and_folders_distinctly():
    material = describe_material(
        [{"name": "udbg-phase1/", "content": "x" * 4000},
         {"name": "notes.txt", "content": "x" * 400}], history_turns=6)

    assert "folder: udbg-phase1/" in material
    assert "file: notes.txt" in material
    assert "1,000 tokens" in material
    assert "6 earlier message(s)" in material


def test_an_empty_inventory_is_stated_plainly_in_the_prompt():
    assert "nothing attached" in build_prompt("what is X?", "")


def test_the_prompt_steers_away_from_searching_the_user_s_own_files():
    prompt = build_prompt("take a look at my app", describe_material(
        [{"name": "app/", "content": "x"}]))

    assert "own files" in prompt
    assert "almost never need a search" in prompt


# ------------------------------------------------------- when to bother

def test_triage_is_worth_it_when_something_is_attached():
    assert worth_triaging("take a look at my app",
                          [{"name": "app/", "content": "x"}], [])


def test_triage_is_worth_it_once_there_is_a_conversation_to_draw_on():
    assert worth_triaging("what about the other one?", [],
                          [{"role": "user", "content": "a"},
                           {"role": "assistant", "content": "b"}])


def test_a_bare_first_question_skips_triage_entirely():
    """With nothing to weigh a search against, triage can only say yes —
    and paying a round-trip to be told what we assumed is not an improvement."""
    assert not worth_triaging("what's the latest on the port strike?", [], [])


def test_an_empty_question_is_never_triaged():
    assert not worth_triaging("", [{"name": "app/", "content": "x"}], [])


def test_the_note_reads_as_something_a_person_would_say():
    assert "no search needed" in Triage(False).note
    assert "Checking the web" in Triage(True, query="x").note
