"""Planning a big job, then working the plan."""

from __future__ import annotations

import pytest

from aichatlab.planning import (
    MAX_STEPS,
    build_plan_prompt,
    build_step_prompt,
    build_summary_prompt,
    drop_impossible_steps,
    looks_complex,
    parse_plan,
)

PLAN = """1. Audit error handling across udbg/cli.py
2. Check the test coverage gaps
3. Review the assembler for edge cases
4. Write up the full list of upgrades"""


# ------------------------------------------------------------- when to plan

def test_a_long_multi_part_request_is_worth_planning():
    assert looks_complex(
        "my debugging App I have fully backed up. I would like you to take a "
        "look and make improvements where you see fit and please give me a "
        "complete list of the upgrades that you have given it. Thank you!")


def test_a_short_request_about_attached_files_is_worth_planning():
    assert looks_complex("review this and list the upgrades",
                         [{"name": "app/"}])


def test_a_plain_question_is_not_worth_the_extra_round_trips():
    assert not looks_complex("what does the cli module do?")
    assert not looks_complex("how do I run the tests?", [{"name": "app/"}])
    assert not looks_complex("")


# ------------------------------------------------------------ reading a plan

def test_a_numbered_plan_becomes_a_list_of_steps():
    assert parse_plan(PLAN) == [
        "Audit error handling across udbg/cli.py",
        "Check the test coverage gaps",
        "Review the assembler for edge cases",
        "Write up the full list of upgrades",
    ]


@pytest.mark.parametrize("prefix", [
    "Here's the plan:\n", "Sure! ", "Plan:\n", "Steps:\n",
])
def test_the_models_preamble_is_stripped(prefix):
    assert parse_plan(prefix + PLAN)[0].startswith("Audit error handling")


def test_bullets_are_accepted_as_well_as_numbers():
    steps = parse_plan("- First thing\n* Second thing\n• Third thing")

    assert steps == ["First thing", "Second thing", "Third thing"]


def test_trailing_commentary_is_not_mistaken_for_a_step():
    steps = parse_plan(PLAN + "\n\nLet me know if you'd like me to start!")

    assert len(steps) == 4
    assert not any("Let me know" in s for s in steps)


def test_a_leading_number_in_a_step_is_not_eaten():
    """Same trap as the search queries: "0-60 time" must keep its subject."""
    assert parse_plan("1. 0-60 acceleration figures") == [
        "0-60 acceleration figures"]


def test_duplicate_steps_are_dropped():
    assert parse_plan("1. Do the thing\n2. Do the thing") == ["Do the thing"]


def test_the_plan_is_capped():
    long_plan = "\n".join(f"{n}. Step number {n}" for n in range(1, 20))

    assert len(parse_plan(long_plan)) == MAX_STEPS


def test_an_over_long_step_is_trimmed_at_a_word_boundary():
    step = parse_plan("1. " + "word " * 60)[0]

    assert len(step) <= 111
    assert step.endswith("…")


def test_prose_with_no_list_yields_no_plan():
    assert parse_plan("I would approach this carefully and thoughtfully.") == [
        "I would approach this carefully and thoughtfully."]
    assert parse_plan("") == []


# ---------------------------------------------------------------- prompts

def test_the_plan_prompt_forbids_filler_steps():
    prompt = build_plan_prompt("improve my app", "- folder: app/")

    assert "not 'understand the request'" in prompt
    assert "folder: app/" in prompt
    assert "2-6 steps" in prompt


def test_a_step_prompt_shows_the_plan_and_what_is_already_done():
    steps = parse_plan(PLAN)
    prompt = build_step_prompt("improve my app", steps, 2,
                               done=["found three bare excepts",
                                     "coverage is 40%"])

    assert "Now do step 3 only: Review the assembler" in prompt
    assert "found three bare excepts" in prompt
    assert "do not add a summary" in prompt


def test_the_first_step_has_nothing_done_yet():
    prompt = build_step_prompt("improve my app", parse_plan(PLAN), 0)

    assert "already done" not in prompt
    assert "Now do step 1 only" in prompt


def test_the_summary_prompt_asks_for_the_answer_not_a_process_report():
    prompt = build_summary_prompt("improve my app", parse_plan(PLAN),
                                  ["a", "b", "c", "d"])

    assert "do not describe your process" in prompt
    assert "asked for" in prompt


# --------------------------------------------------- plans for changing files

def test_the_plan_prompt_says_the_model_has_no_hands():
    prompt = build_plan_prompt("make it concurrent", editing=True)

    assert "cannot open, save, run or test anything" in prompt
    assert "must not appear in the plan" in prompt


def test_an_ordinary_plan_prompt_is_not_burdened_with_it():
    assert "cannot open" not in build_plan_prompt("summarise this")


def test_the_theatre_is_struck_and_the_work_is_kept():
    steps = drop_impossible_steps([
        "Open the file engine.py",
        "Locate the function evaluate_and_schedule",
        "Replace it with a concurrent version",
        "Define a new private method _schedule_email",
        "Save the modified file",
        "Test the modifications",
    ])

    assert steps == ["Locate the function evaluate_and_schedule",
                     "Replace it with a concurrent version",
                     "Define a new private method _schedule_email"]


def test_writing_something_is_work_and_writing_it_out_is_not():
    """"Write the changes to the file" is theatre. "Write the new helper" is
    the entire job, and striking it would leave the plan with nothing in it."""
    steps = drop_impossible_steps([
        "Write a new retry helper",
        "Write the changes to the file",
        "Rework the caller to use it",
    ])

    assert steps == ["Write a new retry helper", "Rework the caller to use it"]


def test_a_plan_that_is_all_theatre_is_left_alone_to_be_seen():
    """Better a visibly silly plan than a silently empty one — and a plan
    this short falls back to answering in one go anyway."""
    steps = ["Open the file", "Save the file"]

    assert drop_impossible_steps(steps) == steps


def test_a_step_is_told_to_quote_rather_than_recall():
    prompt = build_step_prompt("make it concurrent", ["Replace the function"],
                               0, editing=True)

    assert "quote from it" in prompt
    assert "rather than from memory" in prompt


def test_the_summary_demands_blocks_rather_than_a_description():
    prompt = build_summary_prompt("make it concurrent", ["Replace it"],
                                  ["here is the new function"], editing=True)

    assert "edit block" in prompt
    assert "Prose describing a change changes nothing" in prompt
    assert "Do not claim any file has been modified or saved" in prompt


def test_an_ordinary_summary_is_unchanged():
    prompt = build_summary_prompt("summarise this", ["Read it"], ["done"])

    assert "edit block" not in prompt
