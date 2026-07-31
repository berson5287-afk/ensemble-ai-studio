"""Follow-up detection and context-aware search queries.

The bug these exist for: asking "what is the 0-60mph?" straight after a chat
about the Yangwang U9 Xtreme searched those literal words and came back with
generic definitions of the term.
"""

from __future__ import annotations

import pytest

from aichatlab.research import clean_query, contextual_query, extract_subject, looks_like_followup

CAR_HISTORY = [
    {"role": "user", "content": "lookup online the world's fastest production car"},
    {"role": "assistant",
     "content": ("The YANGWANG U9 Xtreme has been confirmed as the world's "
                 "fastest production car with a top speed of 496 km/h, set at "
                 "the ATP Automotive Testing Papenburg track in Germany.")},
]


@pytest.mark.parametrize("question", [
    "what is the 0-60mph?",
    "how much is it?",
    "why?",
    "and the price",
    "is that faster than the Bugatti?",
    "who makes them",
])
def test_short_or_referential_questions_are_follow_ups(question):
    assert looks_like_followup(question) is True


@pytest.mark.parametrize("question", [
    "Explain how a lithium iron phosphate battery differs from NMC chemistry",
    "What were the main causes of the 1970s oil crisis in the United States",
])
def test_self_contained_questions_are_not_follow_ups(question):
    assert looks_like_followup(question) is False


def test_empty_question_is_not_a_followup():
    assert looks_like_followup("") is False
    assert looks_like_followup("   ") is False


def test_subject_is_pulled_from_recent_turns():
    assert "YANGWANG U9" in extract_subject(CAR_HISTORY)


def test_subject_ignores_injected_research_blocks():
    history = [
        {"role": "user",
         "content": "[Web research results for: x]\n\nSource 1: Wikipedia "
                    "Foundation Page\n[End of web research. Answer using]"},
        {"role": "assistant", "content": "The Honda Civic Type R is quick."},
    ]
    assert "Honda Civic" in extract_subject(history)


def test_no_subject_when_there_is_nothing_to_find():
    assert extract_subject([{"role": "user", "content": "hello there"}]) == ""


def test_model_rewriter_is_used_when_available():
    seen = {}

    def rewriter(prompt):
        seen["prompt"] = prompt
        return "Yangwang U9 Xtreme 0-60 mph acceleration time"

    query = contextual_query("what is the 0-60mph?", CAR_HISTORY, rewriter)

    assert query == "Yangwang U9 Xtreme 0-60 mph acceleration time"
    # the rewriter must actually see the conversation
    assert "U9 Xtreme" in seen["prompt"]
    assert "what is the 0-60mph?" in seen["prompt"]


def test_falls_back_to_the_subject_when_no_rewriter():
    query = contextual_query("what is the 0-60mph?", CAR_HISTORY)

    assert "U9" in query
    assert "0-60mph" in query


def test_falls_back_when_the_rewriter_fails():
    def broken(_prompt):
        raise RuntimeError("model died")

    query = contextual_query("what is the 0-60mph?", CAR_HISTORY, broken)
    assert "U9" in query


def test_falls_back_when_the_rewriter_returns_junk():
    query = contextual_query("how fast?", CAR_HISTORY, lambda _p: "  ")
    assert "U9" in query


def test_self_contained_questions_are_left_alone():
    question = "Explain how a lithium iron phosphate battery differs from NMC"
    assert contextual_query(question, CAR_HISTORY, lambda _p: "REWRITTEN") == question


def test_first_question_in_a_chat_is_left_alone():
    assert contextual_query("what is the 0-60mph?", []) == "what is the 0-60mph?"


@pytest.mark.parametrize("raw, expected", [
    ('"Yangwang U9 Xtreme top speed"', "Yangwang U9 Xtreme top speed"),
    ("Search query: U9 Xtreme 0-60", "U9 Xtreme 0-60"),
    ("Here is the query: U9 price", "U9 price"),
    ("U9 speed\nand some rambling after", "U9 speed"),
    ("", ""),
])
def test_clean_query_strips_model_padding(raw, expected):
    assert clean_query(raw) == expected


def test_clean_query_caps_length():
    assert len(clean_query("word " * 100)) <= 160
