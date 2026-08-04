"""Reasoning models: detection, the `think` field, and the reply allowance."""

from __future__ import annotations

import pytest

from aichatlab.thinking import (
    MIN_ALLOWANCE,
    OFF_LEVEL,
    allowance,
    split_inline,
    summarise,
    supports_thinking,
    takes_levels,
    think_value,
)


@pytest.mark.parametrize("model", [
    "qwen3:8b", "qwen3:30b-a3b", "deepseek-r1:7b", "gpt-oss:20b",
    "magistral:latest", "qwq:32b", "DeepSeek-R1:14B",
])
def test_reasoning_models_are_recognised(model):
    assert supports_thinking(model)


@pytest.mark.parametrize("model", [
    "qwen2.5:14b-instruct-q8_0", "llama3:latest", "gemma3:12b",
    "qwen2.5-coder:32b", "phi3:mini", "mistral:7b",
])
def test_ordinary_models_are_left_alone(model):
    """Sending `think` to these is an error, not a no-op."""
    assert not supports_thinking(model)
    assert think_value(model, True) is None
    assert think_value(model, False) is None


def test_a_reasoning_model_gets_a_boolean():
    assert think_value("qwen3:8b", True) is True
    assert think_value("qwen3:8b", False) is False


def test_gpt_oss_gets_a_level_because_booleans_are_ignored():
    """Ollama documents that GPT-OSS ignores true/false — a boolean would
    silently leave it reasoning at whatever the default is."""
    assert takes_levels("gpt-oss:20b")
    assert think_value("gpt-oss:20b", True) == "medium"
    assert think_value("gpt-oss:20b", False) == OFF_LEVEL
    assert think_value("gpt-oss:20b", False) is not False


def test_the_reply_allowance_is_raised_so_reasoning_cannot_eat_the_answer():
    """Quick on the speed slider is 192 tokens — less than qwen3 spends
    thinking, so the answer would arrive empty."""
    assert allowance(192, "qwen3:8b", True) >= MIN_ALLOWANCE


def test_the_allowance_is_untouched_for_models_that_do_not_reason():
    assert allowance(192, "llama3:latest", True) == 192
    assert allowance(192, "qwen3:8b", False) == 192


def test_an_unlimited_allowance_stays_unlimited():
    """0 means "no cap" to Ollama; raising it to 1,500 would impose one."""
    assert allowance(0, "qwen3:8b", True) == 0


def test_a_generous_allowance_still_gains_room_for_the_reasoning():
    assert allowance(4000, "qwen3:8b", True) > 4000


# -- the inline fallback ---------------------------------------------------
def test_inline_reasoning_is_split_out_of_the_answer():
    reasoning, answer = split_inline(
        "<think>The user wants X. I should check Y.</think>\n\nHere is X.")
    assert reasoning == "The user wants X. I should check Y."
    assert answer == "Here is X."


def test_an_unclosed_thought_does_not_leak_into_the_reply():
    """A reply cut off mid-reasoning never closes its tag, and pasting the
    remainder into the bubble reads as the model talking nonsense."""
    reasoning, answer = split_inline("Hello.\n<think>Wait, actually the user")
    assert answer == "Hello."
    assert "Wait, actually" in reasoning


def test_ordinary_text_passes_through_untouched():
    assert split_inline("Just an answer.") == ("", "Just an answer.")


def test_several_thought_blocks_are_all_collected():
    reasoning, answer = split_inline(
        "<think>one</think>A<think>two</think>B")
    assert "one" in reasoning and "two" in reasoning
    assert answer == "AB"


def test_summarise_gives_a_one_line_gist():
    assert summarise("a " * 200).endswith("…")
    assert "\n" not in summarise("line one\nline two")
