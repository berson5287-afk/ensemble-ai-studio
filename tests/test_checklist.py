"""The checklist: every tick has to be earned."""

from __future__ import annotations

from aichatlab.checklist import (
    DONE,
    SKIPPED,
    Checklist,
    pipeline_for,
)


def test_a_fresh_checklist_shows_everything_still_to_do():
    checklist = Checklist(title="This request")
    checklist.add("a", "Read the folder")
    checklist.add("b", "Ask the model")

    rendered = checklist.render()

    assert "This request   (0/2)" in rendered
    assert rendered.count("☐") == 2


def test_a_step_ticks_from_pending_to_running_to_done():
    checklist = Checklist()
    checklist.add("a", "Ask the model")

    checklist.start("a", now=0.0)
    assert "▸ Ask the model" in checklist.render(3.0)
    assert checklist.running().key == "a"

    checklist.finish("a", now=12.0)
    assert "☑ Ask the model" in checklist.render(12.0)
    assert checklist.find("a").state == DONE


def test_a_running_step_shows_how_long_it_has_been_running():
    checklist = Checklist()
    checklist.add("a", "Ask the model")
    checklist.start("a", now=0.0)

    assert "1m 32s" in checklist.render(92.0)


def test_a_finished_step_shows_what_it_took():
    checklist = Checklist()
    checklist.add("a", "Ask the model")
    checklist.start("a", now=0.0)
    checklist.finish("a", now=45.0)

    assert "(45s)" in checklist.render(45.0)


def test_a_step_under_a_second_is_not_padded_with_a_timing():
    checklist = Checklist()
    checklist.add("a", "Decide")
    checklist.start("a", now=0.0)
    checklist.finish("a", now=0.2)

    step_line = checklist.render(1.0).splitlines()[1]
    assert "(" not in step_line


def test_failure_and_skipping_are_shown_rather_than_hidden():
    checklist = Checklist()
    checklist.add("a", "Search the web")
    checklist.add("b", "Ask the model")
    checklist.fail("a", "SearXNG refused the request")
    checklist.skip("b", "nothing to ask")

    rendered = checklist.render()
    assert "✗ Search the web — SearXNG refused" in rendered
    assert "⊘ Ask the model — nothing to ask" in rendered
    assert checklist.complete


def test_stopping_early_marks_the_rest_rather_than_leaving_them_pending():
    """A checklist frozen mid-way looks like it is still running."""
    checklist = Checklist()
    checklist.add("a", "One")
    checklist.add("b", "Two")
    checklist.add("c", "Three")
    checklist.finish("a")
    checklist.start("b", now=0.0)

    checklist.skip_remaining("not reached")

    assert [s.state for s in checklist.steps] == [DONE, SKIPPED, SKIPPED]
    assert checklist.complete


def test_ticking_an_unknown_step_is_harmless():
    checklist = Checklist()
    checklist.start("nope")
    checklist.finish("nope")
    checklist.fail("nope")

    assert checklist.render() == ""


def test_the_count_tracks_finished_steps_of_any_kind():
    checklist = Checklist()
    for key in "abcd":
        checklist.add(key, key)
    checklist.finish("a")
    checklist.fail("b")
    checklist.skip("c")

    assert "(3/4)" in checklist.render()
    assert not checklist.complete


# ------------------------------------------------------------ the pipeline

def test_the_pipeline_lists_only_work_that_will_actually_happen():
    checklist = pipeline_for(models=["Qwen2.5 32B Coder"])

    rendered = checklist.render()
    assert "Ask Qwen2.5 32B Coder" in rendered
    assert "Read" not in rendered            # nothing was attached
    assert "web" not in rendered             # search is off
    assert "remembering" not in rendered     # learning is off


def test_the_pipeline_names_the_attachments_it_will_read():
    checklist = pipeline_for(
        attachments=[{"name": "udbg-phase1/"}, {"name": "notes.txt"}],
        models=["Qwen3 8B"])

    assert "Read udbg-phase1/, notes.txt" in checklist.render()


def test_every_optional_stage_appears_when_it_is_switched_on():
    checklist = pipeline_for(attachments=[{"name": "app/"}], searching=True,
                             models=["A", "B"], learning=True, compacting=True)

    keys = [step.key for step in checklist.steps]
    assert keys == ["compact", "attach", "search", "model:A", "model:B", "learn"]


def test_an_empty_pipeline_renders_as_nothing():
    assert pipeline_for().render() == ""
