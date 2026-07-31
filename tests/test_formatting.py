"""Friendly model names and markdown rendering."""

from __future__ import annotations

import pytest

from aichatlab.formatting import (
    friendly_model_name,
    has_markdown,
    initials,
    parse_inline,
    render_markdown,
    tidy,
)


@pytest.mark.parametrize("model, expected", [
    ("qwen2.5:32b-instruct-q4_K_M", "Qwen2.5 32B"),
    ("qwen2.5:14b-instruct-q8_0", "Qwen2.5 14B"),
    ("llama3.2:3b", "Llama3.2 3B"),
    ("llama3.2-vision:latest", "Llama3.2 Vision"),
    ("gemma3:27b", "Gemma3 27B"),
    ("gpt-oss:20b", "GPT-OSS 20B"),
    ("qwen2.5-coder:32b", "Qwen2.5 32B Coder"),
    ("minicpm-v:latest", "MiniCPM-V"),
    ("llama3:latest", "Llama3"),
    ("mixtral:8x7b", "Mixtral 8X7B"),
])
def test_friendly_model_names(model, expected):
    assert friendly_model_name(model) == expected


def test_friendly_name_of_nothing_is_nothing():
    assert friendly_model_name("") == ""


def test_initials_are_two_letters():
    assert initials("qwen2.5:32b-instruct-q4_K_M") == "Qw"
    assert initials("") == "Ai"


# ------------------------------------------------------------------ markdown

def texts(runs):
    return "".join(text for text, _tags in runs)


def tagged(runs, tag):
    return [text for text, tags in runs if tag in tags]


def test_bold_becomes_a_styled_run_without_asterisks():
    runs = parse_inline("The **YANGWANG U9** is fast")

    assert texts(runs) == "The YANGWANG U9 is fast"
    assert tagged(runs, "md_bold") == ["YANGWANG U9"]


def test_italic_and_code_and_bold_italic():
    runs = parse_inline("use *care* with `rm -rf` and ***never*** guess")

    assert tagged(runs, "md_italic") == ["care", "never"]
    assert tagged(runs, "md_code") == ["rm -rf"]
    assert "never" in tagged(runs, "md_bold")
    assert "*" not in texts(runs)


def test_underscores_inside_words_are_left_alone():
    """q4_K_M must not turn into italics."""
    runs = parse_inline("model qwen2.5:32b-instruct-q4_K_M loaded")
    assert texts(runs) == "model qwen2.5:32b-instruct-q4_K_M loaded"
    assert tagged(runs, "md_italic") == []


def test_links_show_label_then_url():
    runs = parse_inline("see [the guide](http://x.example/g) now")

    assert tagged(runs, "md_link") == ["the guide"]
    assert "http://x.example/g" in texts(runs)
    assert "](" not in texts(runs)


def test_bullets_get_a_real_bullet_character():
    runs = render_markdown("- first item\n- second item")

    assert "•" in texts(runs)
    assert "- first" not in texts(runs)
    assert tagged(runs, "md_bullet")


def test_numbered_lists_keep_their_numbers():
    runs = render_markdown("1. first\n2. second")
    assert "1" in texts(runs) and "2" in texts(runs)
    assert tagged(runs, "md_bullet")


def test_headings_lose_their_hashes():
    runs = render_markdown("## Performance\nfast")

    assert "Performance" in texts(runs)
    assert "#" not in texts(runs)
    assert "Performance" in tagged(runs, "md_heading")
    assert "fast" not in tagged(runs, "md_heading")


def test_fenced_code_blocks_keep_their_lines_but_lose_the_fence():
    runs = render_markdown("try:\n```\npip install pypdf\n```\ndone")

    assert "pip install pypdf" in tagged(runs, "md_code_block")[0]
    assert "```" not in texts(runs)
    assert "try:" in texts(runs) and "done" in texts(runs)


def test_paragraph_line_breaks_survive():
    runs = render_markdown("one\n\ntwo")
    assert texts(runs) == "one\n\ntwo"


def test_plain_text_passes_through_unchanged():
    plain = "Just a normal sentence, nothing fancy."
    assert texts(render_markdown(plain)) == plain


def test_has_markdown_detects_only_real_markup():
    assert has_markdown("this is **bold**")
    assert has_markdown("- a bullet")
    assert has_markdown("# heading")
    assert not has_markdown("plain sentence with no markup")
    assert not has_markdown("2 * 3 is 6")


def test_tidy_collapses_excess_blank_lines():
    assert tidy("\n\nhello\n\n\n\nworld\n  ") == "hello\n\nworld"
