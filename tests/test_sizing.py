"""Whether the window we are about to ask for will fit on the machine."""

from __future__ import annotations

import pytest

from aichatlab.sizing import (
    find_loaded,
    gpu_share,
    heavy_request,
    parse_size,
    placement_note,
    suggest_after_timeout,
)


@pytest.mark.parametrize("model, expected", [
    ("qwen2.5:32b-instruct-q4_K_M", 32.0),
    ("llama3.2:3b", 3.0),
    ("gpt-oss:20b", 20.0),
    ("qwen2.5-coder:32b", 32.0),
    ("gemma2:9b", 9.0),
    ("qwen3:1.5b", 1.5),
])
def test_the_parameter_count_is_read_from_the_name(model, expected):
    assert parse_size(model) == expected


@pytest.mark.parametrize("model", ["phi3:mini", "llama3", "mistral:latest", ""])
def test_a_name_that_does_not_say_gives_no_answer(model):
    assert parse_size(model) is None


def test_a_version_number_is_not_mistaken_for_a_parameter_count():
    """"qwen2.5" is a version, not a 2.5B model."""
    assert parse_size("qwen2.5:32b") == 32.0
    assert parse_size("qwen2.5") is None


def test_a_big_model_with_a_big_window_is_flagged():
    warning = heavy_request("qwen2.5-coder:32b", 29_696)

    assert "32B model" in warning
    assert "29,696-token window" in warning
    assert "ollama ps" in warning
    assert "moves layers to the CPU" in warning


def test_a_small_model_is_not_flagged_however_big_the_window():
    assert heavy_request("llama3.2:3b", 30_000) == ""


def test_a_big_model_with_an_ordinary_window_is_not_flagged():
    assert heavy_request("qwen2.5-coder:32b", 8_192) == ""


def test_a_model_that_does_not_state_its_size_is_left_alone():
    assert heavy_request("mistral:latest", 30_000) == ""


def test_the_advice_after_a_timeout_quotes_the_measured_speed():
    advice = suggest_after_timeout("qwen2.5-coder:32b", 23_890, 1800)

    assert "13 tokens per second" in advice
    assert "CPU speed" in advice
    assert "7B or 14B" in advice
    assert "ollama ps" in advice


def test_a_small_model_timing_out_gets_different_advice():
    advice = suggest_after_timeout("llama3.2:3b", 23_890, 1800)

    assert "fewer files" in advice
    assert "7B or 14B" not in advice


def test_no_division_by_zero_when_no_time_elapsed():
    assert suggest_after_timeout("qwen2.5:32b", 1000, 0)


# ---------------------------------------------- where the model really is

def test_a_model_entirely_on_the_cpu_is_called_out():
    """`ollama ps` said 100% CPU. Almost nobody knows to run it."""
    entry = {"name": "qwen2.5-coder:32b", "size": 23_000_000_000,
             "size_vram": 0}

    note = placement_note("Qwen2.5 32B Coder", entry)

    assert "entirely on the CPU" in note
    assert "23 GB" in note
    assert "did not refuse to run" in note


def test_a_partly_offloaded_model_gives_the_percentage():
    entry = {"size": 20_000_000_000, "size_vram": 12_000_000_000}

    assert "only 60% on the GPU" in placement_note("M", entry)


def test_a_model_on_the_gpu_says_nothing_at_all():
    entry = {"size": 8_000_000_000, "size_vram": 8_000_000_000}

    assert placement_note("M", entry) == ""


def test_a_missing_or_broken_entry_makes_no_claim():
    assert gpu_share({}) is None
    assert gpu_share({"size": 0, "size_vram": 0}) is None
    assert gpu_share({"size": "banana", "size_vram": 1}) is None
    assert placement_note("M", {}) == ""


def test_the_loaded_entry_is_found_however_ollama_names_it():
    models = [{"name": "qwen2.5-coder:32b", "model": "qwen2.5-coder:32b"}]

    assert find_loaded(models, "qwen2.5-coder:32b") is not None
    assert find_loaded(models, "qwen2.5-coder:32b-q4_K_M") is not None
    assert find_loaded(models, "llama3:8b") is None
    assert find_loaded([], "anything") is None
    assert find_loaded(models, "") is None
