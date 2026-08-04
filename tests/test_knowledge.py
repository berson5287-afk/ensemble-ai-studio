"""Learning mode: extraction, storage, retrieval and restraint."""

from __future__ import annotations

import json

import pytest

from aichatlab.knowledge import (
    KnowledgeBase,
    build_extraction_prompt,
    format_for_prompt,
    parse_lessons,
    worth_learning_from,
)

PLUMBING = ("Plumbing", "Shut the main valve off before opening a P-trap or "
                        "the standing water in the line will empty onto the floor.")
PYTHON = ("Python", "dict.setdefault avoids the extra KeyError branch when "
                    "building a dictionary of lists.")


@pytest.fixture
def base(tmp_path):
    return KnowledgeBase(tmp_path / "knowledge.json")


# ---------------------------------------------------------------- parsing

def test_parses_topic_and_lesson_lines():
    reply = ("Plumbing | Shut the main valve before opening a P-trap.\n"
             "Tools | A basin wrench reaches nuts behind a sink that pliers cannot.")
    lessons = parse_lessons(reply)

    assert len(lessons) == 2
    assert lessons[0][0] == "Plumbing"
    assert "main valve" in lessons[0][1]


def test_parses_bulleted_and_numbered_output():
    reply = ("- Plumbing | Shut the main valve before opening a P-trap.\n"
             "2. Wiring | Kill the breaker and test the line before touching it.")
    lessons = parse_lessons(reply)

    assert len(lessons) == 2
    assert lessons[1][0] == "Wiring"
    assert not lessons[1][1].startswith("2.")


def test_colon_format_is_accepted_too():
    lessons = parse_lessons("Soldering: heat the joint, not the solder itself.")
    assert lessons[0][0] == "Soldering"
    assert "heat the joint" in lessons[0][1]


def test_none_reply_yields_nothing():
    assert parse_lessons("NONE") == []
    assert parse_lessons("none.") == []
    assert parse_lessons("") == []


def test_time_sensitive_lines_are_refused():
    reply = ("Weather | It will rain on Saturday in Pearl River.\n"
             "Plumbing | Shut the main valve before opening a P-trap.")
    lessons = parse_lessons(reply)

    assert len(lessons) == 1
    assert lessons[0][0] == "Plumbing"


def test_at_most_three_lessons_per_exchange():
    reply = "\n".join(f"Topic{i} | A durable lesson number {i} worth keeping."
                      for i in range(10))
    assert len(parse_lessons(reply)) == 3


def test_extraction_prompt_contains_both_sides():
    prompt = build_extraction_prompt("how do I fix a leak?", "Shut the valve.")
    assert "how do I fix a leak?" in prompt
    assert "Shut the valve." in prompt
    assert "NONE" in prompt


# ---------------------------------------------------------------- storage

def test_lessons_are_stored_and_persist(tmp_path):
    path = tmp_path / "k.json"
    KnowledgeBase(path).add(*PLUMBING)

    assert len(KnowledgeBase(path)) == 1


def test_duplicate_lessons_are_not_stored_twice(base):
    base.add(*PLUMBING)
    again = base.add("Plumbing", "Shut the main valve off before opening a "
                                 "P-trap or standing water empties onto the floor.")

    assert again is None
    assert len(base) == 1


def test_different_lessons_are_both_kept(base):
    base.add(*PLUMBING)
    base.add(*PYTHON)
    assert len(base) == 2


def test_volatile_lessons_are_refused(base):
    assert base.add("Weather", "It will rain tomorrow afternoon in Pearl River") is None
    assert len(base) == 0


def test_trivial_lessons_are_refused(base):
    assert base.add("General", "yes") is None
    assert len(base) == 0


def test_add_many_returns_only_what_was_kept(base):
    added = base.add_many([PLUMBING, ("Weather", "Rain is expected tonight"),
                           PYTHON])
    assert len(added) == 2


# ---------------------------------------------------------------- recall

def test_relevant_lessons_are_found_by_topic(base):
    base.add(*PLUMBING)
    base.add(*PYTHON)

    found = base.relevant("my kitchen sink P-trap is leaking, plumbing help")

    assert len(found) == 1
    assert found[0].topic == "Plumbing"


def test_unrelated_questions_recall_nothing(base):
    base.add(*PLUMBING)
    assert base.relevant("what is the capital of Peru") == []


def test_recall_counts_uses(base):
    base.add(*PYTHON)
    base.recall("python dictionary setdefault question")
    base.recall("python dictionary setdefault question")

    assert base.lessons[0].uses == 2


def test_recall_is_capped(base):
    for index in range(20):
        base.add("Wiring", f"Electrical rule number {index} about circuit "
                           f"breakers and wiring in a panel box.")
    assert len(base.relevant("wiring a circuit breaker panel", limit=5)) <= 5


def test_format_for_prompt_reads_as_prior_knowledge(base):
    base.add(*PLUMBING)
    text = format_for_prompt(base.relevant("P-trap plumbing leak"))

    assert "earlier conversations" in text
    assert "main valve" in text
    assert "(Plumbing)" in text


def test_format_of_nothing_is_empty():
    assert format_for_prompt([]) == ""


# ---------------------------------------------------------------- management

def test_lessons_can_be_removed(base):
    lesson = base.add(*PLUMBING)
    assert base.remove(lesson.id) is True
    assert len(base) == 0
    assert base.remove("nonexistent") is False


def test_clear_empties_everything(base):
    base.add(*PLUMBING)
    base.add(*PYTHON)
    base.clear()
    assert len(base) == 0


def test_topics_are_listed_alphabetically(base):
    base.add(*PYTHON)
    base.add(*PLUMBING)
    assert base.topics() == ["Plumbing", "Python"]


def test_a_corrupt_knowledge_file_is_ignored(tmp_path):
    path = tmp_path / "k.json"
    path.write_text("not json at all", encoding="utf-8")

    base = KnowledgeBase(path)
    assert len(base) == 0
    base.add(*PLUMBING)
    assert len(base) == 1


# ---------------------------------------------------------------- gating

def test_short_exchanges_are_not_mined():
    assert worth_learning_from("hi", "Hello!") is False


def test_substantial_exchanges_are_mined():
    question = "How do I replace the trap under my kitchen sink?"
    answer = (" ".join(["Shut off the water, place a bucket underneath, then "
                        "loosen both slip nuts by hand before pulling the trap "
                        "free and cleaning the threads."] * 2))
    assert worth_learning_from(question, answer) is True


def test_a_lesson_starting_with_a_number_keeps_it():
    """Same stripper bug as the research parser: "3D" must not become "D"."""
    lessons = parse_lessons("Printing | 3D prints warp when the bed is too cold")
    assert lessons[0][1].startswith("3D")


# ------------------------------------------- lessons stay in their own project
#
# "Make some performance upgrades" matches every project the user has ever
# opened.  A lesson carrying another app's file paths does not become good
# advice by being relevant-sounding — it becomes a model confidently
# proposing `udbg/cache.py` inside a folder that has no `udbg` in it.

def test_a_lesson_learned_in_one_project_stays_there(tmp_path):
    base = KnowledgeBase(tmp_path / "k.json")
    base.add("Caching", "udbg/cache.py holds the lru_cache helpers",
             project="udbg")

    assert base.relevant("caching helpers", project="udbg")
    assert base.relevant("caching helpers", project="Replyit") == []


def test_a_lesson_learned_outside_any_project_still_applies_everywhere(tmp_path):
    base = KnowledgeBase(tmp_path / "k.json")
    base.add("Caching", "lru_cache is not safe on methods taking self")

    assert base.relevant("caching methods", project="Replyit")
    assert base.relevant("caching methods")


def test_recall_is_scoped_the_same_way_as_relevant(tmp_path):
    base = KnowledgeBase(tmp_path / "k.json")
    base.add("Caching", "udbg/cache.py holds the lru_cache helpers",
             project="udbg")

    assert base.recall("caching helpers", project="Replyit") == []


def test_the_project_survives_a_round_trip(tmp_path):
    path = tmp_path / "k.json"
    KnowledgeBase(path).add_many([("Caching", "a lesson about caching here")],
                                 project="Replyit")

    assert KnowledgeBase(path).lessons[0].project == "Replyit"


def test_an_older_knowledge_file_without_projects_still_loads(tmp_path):
    """Everything learned before this existed is general, which is the same
    behaviour those lessons already had."""
    path = tmp_path / "k.json"
    path.write_text(json.dumps({"lessons": [
        {"topic": "Caching", "text": "a lesson from before projects existed",
         "id": "abc", "created_at": "2026-01-01", "source": "", "uses": 0}]}),
        encoding="utf-8")

    base = KnowledgeBase(path)

    assert base.lessons[0].project == ""
    assert base.relevant("caching lesson", project="anything")
