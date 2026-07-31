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
