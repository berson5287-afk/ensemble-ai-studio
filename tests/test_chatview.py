"""Transcript rendering, including the concurrent-streaming regression.

These need a display, so they are skipped automatically on headless machines
(CI runs them under xvfb).
"""

from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def view():
    from aichatlab.ui.chatview import ChatView
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    widget = ChatView(root)
    widget.pack()
    yield widget
    root.destroy()


def lines_containing(view, needle):
    return [line for line in view.transcript().splitlines() if needle in line]


def test_two_models_streaming_at_once_stay_in_their_own_bubbles(view):
    """Regression: both replies used to interleave into one bubble.

    The marks were being set before the bubble padding was inserted, so
    right-gravity pushed every mark to the end of the widget.
    """
    alpha = view.start_stream(1, "alpha")
    beta = view.start_stream(2, "beta")

    for index in range(5):
        view.append_stream(alpha, f"A{index} ")
        view.append_stream(beta, f"B{index} ")

    view.end_stream(alpha, meta="30 tok/s")
    view.end_stream(beta, meta="25 tok/s")

    alpha_line = lines_containing(view, "A0")[0]
    beta_line = lines_containing(view, "B0")[0]

    assert "B" not in alpha_line
    assert "A" not in beta_line
    assert alpha_line.split() == ["A0", "A1", "A2", "A3", "A4"]


def test_stream_meta_lands_on_the_heading_not_in_the_message(view):
    stream = view.start_stream(1, "alpha")
    view.append_stream(stream, "the answer")
    view.end_stream(stream, meta="42 tok/s")

    heading = lines_containing(view, "alpha")[0]
    body = lines_containing(view, "the answer")[0]

    assert "42 tok/s" in heading
    assert "42 tok/s" not in body


def test_copy_chip_holds_the_finished_text(view):
    stream = view.start_stream(1, "alpha")
    view.append_stream(stream, "hello ")
    view.append_stream(stream, "world")
    view.end_stream(stream)

    view.copy_all()
    assert "hello world" in view.clipboard_get()


def test_transcript_survives_a_failed_stream(view):
    stream = view.start_stream(1, "alpha")
    view.append_stream(stream, "partial")
    view.fail_stream(stream, "server exploded")

    text = view.transcript()
    assert "partial" in text
    assert "server exploded" in text


def test_selection_highlight_outranks_bubble_backgrounds(view):
    """Regression: AI replies looked unselectable.

    Tk gives tags created after the widget a higher priority than the built-in
    "sel" tag, so the opaque bubble background painted over the highlight.
    tag_names() returns tags lowest-priority first, so "sel" must come last.
    """
    order = view.text.tag_names()

    assert order.index("sel") > order.index("bot_msg")
    assert order.index("sel") > order.index("user_msg")
    assert order[-1] == "sel"


def test_selection_has_a_visible_colour(view):
    # a raised tag ignores the widget-level -selectbackground, so the tag
    # itself must carry the colour
    assert view.text.tag_cget("sel", "background") not in ("", None)


def test_ai_text_can_actually_be_selected_and_copied(view):
    stream = view.start_stream(1, "llama3.2:3b")
    view.append_stream(stream, "the Yangwang U9 Xtreme")
    view.end_stream(stream)

    view.select_all()
    view.copy_selection()

    assert "Yangwang U9 Xtreme" in view.clipboard_get()


def test_markdown_is_rendered_when_the_reply_finishes(view):
    stream = view.start_stream(1, "Qwen2.5 32B")
    view.append_stream(stream, "The **YANGWANG U9** hit `496 km/h`.\n- fast\n")
    view.end_stream(stream)

    text = view.transcript()
    assert "**" not in text and "`" not in text
    assert "YANGWANG U9" in text
    assert "•" in text


def test_copy_chip_holds_the_clean_text_not_the_markdown(view):
    stream = view.start_stream(1, "Qwen2.5 32B")
    view.append_stream(stream, "it is **very** fast")
    view.end_stream(stream)

    view.copy_all()
    assert "very" in view.clipboard_get()


def test_plain_replies_are_left_exactly_as_they_streamed(view):
    stream = view.start_stream(1, "Llama3.2 3B")
    view.append_stream(stream, "No markup here at all.")
    view.end_stream(stream)

    assert "No markup here at all." in view.transcript()


def test_consecutive_replies_from_one_model_share_a_header(view):
    view.add_user("hi")
    first = view.start_stream(1, "Qwen2.5 32B", speaker="host::qwen")
    view.append_stream(first, "one")
    view.end_stream(first)
    second = view.start_stream(2, "Qwen2.5 32B", speaker="host::qwen")
    view.append_stream(second, "two")
    view.end_stream(second)

    assert view.transcript().count("Qwen2.5 32B") == 1


def test_a_new_question_starts_a_fresh_header(view):
    first = view.start_stream(1, "Qwen2.5 32B", speaker="host::qwen")
    view.end_stream(first)
    view.add_user("another question")
    second = view.start_stream(2, "Qwen2.5 32B", speaker="host::qwen")
    view.end_stream(second)

    assert view.transcript().count("Qwen2.5 32B") == 2


def test_different_models_each_get_their_own_header(view):
    first = view.start_stream(1, "Qwen2.5 32B", speaker="host::qwen")
    view.end_stream(first)
    second = view.start_stream(2, "Llama3.2 3B", speaker="local::llama")
    view.end_stream(second)

    text = view.transcript()
    assert "Qwen2.5 32B" in text and "Llama3.2 3B" in text


def test_typing_indicator_appears_then_is_replaced_by_the_reply(view):
    stream = view.start_stream(1, "Qwen2.5 32B")
    assert "•" in view.transcript()          # dots while waiting

    view.append_stream(stream, "hello there")
    text = view.transcript()
    assert "hello there" in text
    assert "•" not in text                   # dots cleared by the first token
    view.end_stream(stream)


def test_typing_indicator_is_cleared_on_failure(view):
    stream = view.start_stream(1, "Qwen2.5 32B")
    view.fail_stream(stream, "server exploded")

    text = view.transcript()
    assert "server exploded" in text
    assert "•••" not in text


def test_a_note_with_an_id_updates_in_place(view):
    view.add_note("🔍 Searching…", note_id="research")
    view.add_note("📖 Reading source 1…", note_id="research")
    view.add_note("✅ Done — 5 sources", note_id="research")

    text = view.transcript()
    assert "Searching" not in text
    assert "Reading source 1" not in text
    assert "✅ Done — 5 sources" in text


def test_closing_a_note_lets_the_next_one_start_a_new_line(view):
    view.add_note("first", note_id="research")
    view.close_note("research")
    view.add_note("second", note_id="research")

    text = view.transcript()
    assert "first" in text and "second" in text


def test_notes_without_an_id_stack_up_normally(view):
    view.add_note("one")
    view.add_note("two")
    assert "one" in view.transcript() and "two" in view.transcript()


def test_clear_removes_everything(view):
    view.add_user("hi")
    view.add_note("note")
    view.clear()
    assert view.transcript().strip() == ""


def test_finished_lines_are_styled_while_the_reply_is_still_streaming(view):
    """Raw ** should not linger on lines that are already complete."""
    stream = view.start_stream(1, "Qwen2.5 32B")
    view.append_stream(stream, "The **U9 Xtreme** is quick.\n")
    view.append_stream(stream, "Still typing **this")

    text = view.transcript()
    assert "**U9 Xtreme**" not in text      # settled line already styled
    assert "U9 Xtreme" in text
    assert "**this" in text                 # the in-progress line is untouched

    view.end_stream(stream)


def test_incremental_styling_keeps_the_final_text_intact(view):
    stream = view.start_stream(1, "Qwen2.5 32B")
    for chunk in ("Line **one**\n", "- bullet ", "two\n", "tail text"):
        view.append_stream(stream, chunk)
    view.end_stream(stream)

    text = view.transcript()
    assert "one" in text and "bullet two" in text and "tail text" in text
    assert "**" not in text
    assert "•" in text


def test_end_stream_redraws_when_the_final_text_was_cleaned(view):
    """Control markers like [END] are stripped from what the user sees."""
    stream = view.start_stream(1, "Llama3.2 3B")
    view.append_stream(stream, "Agreed, we've landed in the same place.\n[END]")
    view.end_stream(stream, "Agreed, we've landed in the same place.")

    text = view.transcript()
    assert "[END]" not in text
    assert "landed in the same place" in text


# ------------------------------------------------- inline actions (buttons)

def test_an_action_shows_its_question_and_buttons(view):
    view.add_action("Enable web research?",
                    [("Yes", lambda: None, True), ("No", lambda: None, False)],
                    action_id="a1")

    assert "Enable web research?" in view.transcript()
    assert view.has_action("a1")


def test_clicking_a_button_runs_its_callback(view):
    clicked = []
    view.add_action("Search?", [("Yes", lambda: clicked.append("yes"), True)],
                    action_id="a1")

    _click_action_button(view, 0)

    assert clicked == ["yes"]


def test_resolving_replaces_the_question_with_the_outcome(view):
    view.add_action("Search?", [("Yes", lambda: None, True)], action_id="a1")

    view.resolve_action("a1", "🔍 Web research switched on.")

    assert "Search?" not in view.transcript()
    assert "Web research switched on." in view.transcript()
    assert not view.has_action("a1")


def test_the_buttons_go_away_once_the_choice_is_made(view):
    """A stale click must not send the same message twice.

    Answering deletes the block, and the embedded buttons go with it, so
    there is nothing left to click a second time.
    """
    clicks = []

    def choose():
        clicks.append(1)
        view.resolve_action("a1", "done")

    view.add_action("Search?", [("Yes", choose, True)], action_id="a1")
    assert _action_buttons(view)

    _click_action_button(view, 0)

    assert clicks == [1]
    assert _action_buttons(view) == []
    assert not view.has_action("a1")


def test_resolving_an_unknown_action_is_harmless(view):
    view.resolve_action("nope", "text")   # must not raise


def test_surrounding_messages_survive_an_action_being_resolved(view):
    view.add_note("before")
    view.add_action("Search?", [("Yes", lambda: None, True)], action_id="a1")
    view.add_note("after")

    view.resolve_action("a1", "chosen")

    transcript = view.transcript()
    assert "before" in transcript and "after" in transcript
    assert "chosen" in transcript and "Search?" not in transcript


def test_clearing_the_view_forgets_pending_actions(view):
    view.add_action("Search?", [("Yes", lambda: None, True)], action_id="a1")

    view.clear()

    assert not view.has_action("a1")


def _action_buttons(view):
    buttons = []
    for name in view.text.window_names():
        widget = view.text.nametowidget(name)
        buttons.extend(child for child in widget.winfo_children()
                       if isinstance(child, tk.Button))
    return buttons


def _click_action_button(view, index):
    _action_buttons(view)[index].invoke()


# -- reasoning models ------------------------------------------------------
def test_reasoning_streams_above_the_answer_as_it_arrives(view):
    view.start_stream(1, "Qwen3 8B")
    view.append_thought(1, "The user is asking about VRAM. ")
    view.append_thought(1, "Two cards is not one pool.")

    transcript = view.transcript()
    assert "Thinking" in transcript
    assert "Two cards is not one pool." in transcript
    assert view.thought_text(1) == ("The user is asking about VRAM. "
                                    "Two cards is not one pool.")


def test_the_reasoning_folds_away_when_the_answer_starts(view):
    """It exists to show life during the silence, not to crowd the reply."""
    view.start_stream(1, "Qwen3 8B")
    view.append_thought(1, "working it out")
    assert not view._thoughts[1]["collapsed"]

    view.append_stream(1, "They are separate 11 GB pools.")

    assert view._thoughts[1]["collapsed"]
    assert "click to show" in view.transcript()


def test_the_reasoning_survives_the_final_markdown_redraw(view):
    """`end_stream` deletes and repaints the answer; the working must sit
    outside that range or a bold marker would wipe it."""
    view.start_stream(1, "Qwen3 8B")
    view.append_thought(1, "a private thought")
    view.append_stream(1, "**bold** answer")
    view.end_stream(1, "**bold** answer", thought_s=12.0)

    assert view.thought_text(1) == "a private thought"
    assert "Thought for 12s" in view.transcript()


def test_a_reasoning_time_under_a_second_is_not_reported(view):
    """"Thought for 0s" is worse than saying nothing about the time."""
    view.start_stream(1, "Qwen3 8B")
    view.append_thought(1, "quick one")
    view.end_stream(1, "answer", thought_s=0.4)

    assert "Thought for 0s" not in view.transcript()
    assert "click to show" in view.transcript()


def test_an_empty_thought_block_leaves_nothing_behind(view):
    """Whitespace-only reasoning must not leave a stray header on screen."""
    view.start_stream(1, "Qwen3 8B")
    view.append_thought(1, "   ")
    view.end_stream(1, "answer")

    assert "Thinking" not in view.transcript()
    assert "Thought for" not in view.transcript()


def test_clicking_the_header_expands_and_collapses(view):
    view.start_stream(1, "Qwen3 8B")
    view.append_thought(1, "some working")
    view.end_stream(1, "answer", thought_s=3.0)
    assert view._thoughts[1]["collapsed"]

    view.toggle_thought(1)
    assert not view._thoughts[1]["collapsed"]
    assert "▾" in view.transcript()

    view.toggle_thought(1)
    assert view._thoughts[1]["collapsed"]


def test_a_reply_that_never_reasons_gets_no_block_at_all(view):
    view.start_stream(1, "Llama3 8B")
    view.append_stream(1, "Straight to the answer.")
    view.end_stream(1, "Straight to the answer.")

    assert 1 not in view._thoughts
    assert "Thinking" not in view.transcript()


def test_two_reasoning_models_keep_their_working_apart(view):
    view.start_stream(1, "Qwen3 8B")
    view.start_stream(2, "DeepSeek R1")
    view.append_thought(1, "first model's working")
    view.append_thought(2, "second model's working")

    assert view.thought_text(1) == "first model's working"
    assert view.thought_text(2) == "second model's working"


def test_a_failed_reply_still_tidies_its_reasoning_away(view):
    view.start_stream(1, "Qwen3 8B")
    view.append_thought(1, "half a thought")
    view.fail_stream(1, "the server went away")

    assert view._thoughts[1]["collapsed"]
    assert "the server went away" in view.transcript()


def test_clearing_the_view_forgets_the_reasoning_blocks(view):
    view.start_stream(1, "Qwen3 8B")
    view.append_thought(1, "something")

    view.clear()

    assert not view._thoughts
