"""Project memory: what a new chat already knows."""

from __future__ import annotations

from aichatlab import projectmemory as pm


def test_an_unseen_project_remembers_nothing():
    memory = pm.load("/tmp/definitely-not-a-project-93812")

    assert memory.empty
    assert memory.as_block() == ""


def test_what_is_written_comes_back(tmp_path):
    pm.save_brief(tmp_path, "udbg is a small ELF debugger.")
    pm.save_progress(tmp_path, "Halfway through folding the symbol table.")
    pm.append_decision(tmp_path, "Load lazily — start-up cost dominates.")

    memory = pm.load(tmp_path)

    assert "ELF debugger" in memory.brief
    assert "symbol table" in memory.progress
    assert "Load lazily" in memory.decisions
    assert not memory.empty


def test_the_block_leads_with_what_the_project_is(tmp_path):
    pm.save_brief(tmp_path, "udbg is a small ELF debugger.")
    pm.save_progress(tmp_path, "Halfway through the symbol table.")
    pm.append_decision(tmp_path, "Load lazily.")

    block = pm.load(tmp_path).as_block()

    assert block.index("What this project is") < block.index("Where we got to")
    assert block.index("Where we got to") < block.index("Decisions already")
    assert block.startswith("[Project memory")
    assert block.rstrip().endswith(pm.FOOTER)


def test_memory_cannot_grow_into_another_context_problem(tmp_path):
    """Unbounded memory stops being memory and becomes the thing it was
    meant to prevent."""
    pm.save_brief(tmp_path, "b" * 50_000)
    for index in range(200):
        pm.append_decision(tmp_path, f"Decision number {index} " + "d" * 400)

    block = pm.load(tmp_path).as_block()

    assert len(block) <= pm.MAX_PROMPT_CHARS


def test_the_newest_decisions_survive_the_cap(tmp_path):
    for index in range(200):
        pm.append_decision(tmp_path, f"Decision number {index} " + "d" * 200)

    decisions = pm.load(tmp_path).decisions

    assert "Decision number 199" in decisions
    assert "Decision number 0 " not in decisions


def test_decisions_are_appended_not_overwritten(tmp_path):
    pm.append_decision(tmp_path, "First decision.")
    pm.append_decision(tmp_path, "Second decision.")

    decisions = pm.load(tmp_path).decisions

    assert "First decision." in decisions
    assert "Second decision." in decisions


def test_the_same_decision_is_not_recorded_twice(tmp_path):
    pm.append_decision(tmp_path, "Load lazily.")
    pm.append_decision(tmp_path, "Load lazily.")

    assert pm.load(tmp_path).decisions.count("Load lazily.") == 1


def test_progress_is_replaced_because_it_is_a_position_not_a_history(tmp_path):
    pm.save_progress(tmp_path, "Started the loader.")
    pm.save_progress(tmp_path, "Finished the loader, starting symbols.")

    progress = pm.load(tmp_path).progress

    assert "Finished the loader" in progress
    assert "Started the loader" not in progress


def test_empty_and_none_are_not_written(tmp_path):
    assert not pm.save_progress(tmp_path, "")
    assert not pm.save_progress(tmp_path, "none")
    assert not pm.append_decision(tmp_path, "  NONE  ")
    assert pm.load(tmp_path).empty


def test_an_unwritable_folder_is_not_an_error(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")

    assert not pm.save_brief(blocker, "anything")
    assert pm.load(blocker).empty


# -- asking the model to write the hand-off --------------------------------
REPLY = """
PROGRESS:
The loader now re-checks the privacy gate at read time. The symbol table
work is half done — rebuild_symbol_table still runs on every open.

DECISION:
Re-classify files at read time rather than trusting the onboard pass, because
a file's contents can change between the two.
"""


def test_the_handoff_is_split_into_progress_and_decision():
    progress, decision = pm.parse_handoff(REPLY)

    assert "half done" in progress
    assert "Re-classify files at read time" in decision
    assert "DECISION" not in progress, "the sections must not bleed"


def test_a_conversation_that_decided_nothing_records_nothing():
    progress, decision = pm.parse_handoff(
        "PROGRESS:\nPoked at it a bit.\n\nDECISION:\nNONE")

    assert progress
    assert decision == ""


def test_a_reply_in_the_wrong_shape_is_survivable():
    progress, decision = pm.parse_handoff("I'm not sure what you mean.")

    assert progress == "" and decision == ""


def test_the_prompt_keeps_the_end_of_a_long_conversation():
    """Where it got to is at the end; the beginning is the least useful part
    of a hand-off."""
    prompt = pm.build_handoff_prompt("start " * 5_000 + "THE LAST THING",
                                     limit=2_000)

    assert "THE LAST THING" in prompt
    assert "earlier turns omitted" in prompt


def test_a_chat_where_nothing_happened_gets_no_handoff():
    assert not pm.worth_a_handoff([])
    assert not pm.worth_a_handoff([{"role": "user", "content": "hi"}])
    assert pm.worth_a_handoff(
        [{"role": "user", "content": "x" * 100},
         {"role": "assistant", "content": "y" * 100},
         {"role": "user", "content": "z" * 100},
         {"role": "assistant", "content": "w" * 100}])
