"""Settings persistence and the speed↔quality slider mapping."""

from __future__ import annotations

import json

from aichatlab.config import Settings


def make(quality: int) -> Settings:
    return Settings({"speed_quality": quality})


def test_slider_extremes():
    label, cap, directive = make(0).profile()
    assert label == "Fastest"
    assert cap < 300
    assert "brief" in directive.lower() or "sentences" in directive.lower()

    label, cap, directive = make(100).profile()
    assert label == "Max quality"
    assert cap == -1
    assert "thorough" in directive.lower()


def test_slider_midpoint_is_balanced_with_no_directive():
    label, _cap, directive = make(50).profile()
    assert label == "Balanced"
    assert directive == ""


def test_quick_profiles_cap_the_response_length():
    fast = make(10).sampling_options()
    balanced = make(50).sampling_options()
    maxed = make(100).sampling_options()

    assert fast["num_predict"] < balanced["num_predict"]
    assert "num_predict" not in maxed          # uncapped at the top


def test_slider_value_is_clamped_and_survives_junk():
    assert make(-40).profile()[0] == "Fastest"
    assert make(400).profile()[0] == "Max quality"
    assert Settings({"speed_quality": "banana"}).profile()[0] == "Balanced"


def test_settings_round_trip(tmp_path):
    settings = Settings(path=tmp_path / "settings.json")
    settings["searxng_url"] = "http://192.168.1.20:8080"
    settings["speed_quality"] = 70
    assert settings.save()

    loaded = Settings.load(tmp_path / "settings.json")
    assert loaded["searxng_url"] == "http://192.168.1.20:8080"
    assert loaded.profile()[0] == "Detailed"


def test_corrupt_settings_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    loaded = Settings.load(path)
    assert loaded["local_ip"] == "127.0.0.1"


# ----------------------------------------------------- context window sizing

def test_the_model_window_is_derived_from_our_own_budget():
    """Regression: we trimmed to 6,000 tokens and then asked for Ollama's
    4,096-token default, so the server re-trimmed a prompt we had already
    decided was small enough."""
    settings = Settings({"context_budget_tokens": 6000, "speed_quality": 50})

    window = settings.context_window()

    assert window >= 6000 + 1024
    assert window % 1024 == 0
    assert settings.sampling_options()["num_ctx"] == window


def test_a_bigger_budget_asks_for_a_bigger_window():
    small = Settings({"context_budget_tokens": 6000}).context_window()
    large = Settings({"context_budget_tokens": 24000}).context_window()

    assert large > small


def test_the_window_is_capped_so_a_huge_budget_cannot_exhaust_vram():
    settings = Settings({"context_budget_tokens": 500_000,
                         "max_context_window": 32768})

    assert settings.context_window() == 32768


def test_a_tiny_budget_still_gets_a_usable_window():
    assert Settings({"context_budget_tokens": 10}).context_window() >= 2048


def test_junk_budget_values_do_not_crash_the_window_calculation():
    assert Settings({"context_budget_tokens": "banana"}).context_window() >= 2048
    assert Settings({"max_context_window": None}).context_window() >= 2048


def test_the_trimming_budget_never_exceeds_the_window_we_ask_for():
    """Regression: a folder attachment could raise the budget to 96,000 while
    we still requested a 32,768-token window, so the server silently dropped
    two thirds of the prompt instead of us trimming it deliberately."""
    settings = Settings({"context_budget_tokens": 96_000,
                         "max_context_window": 32_768})

    assert settings.effective_budget() < settings.context_window()
    assert settings.effective_budget() < 96_000


def test_an_ordinary_budget_is_used_as_written():
    settings = Settings({"context_budget_tokens": 6000})

    assert settings.effective_budget() == 6000


def test_a_junk_budget_still_yields_something_usable():
    assert Settings({"context_budget_tokens": None}).effective_budget() >= 500


# ------------------------------------------------- how long to allow silence

def test_the_timeout_covers_the_time_spent_reading_a_big_prompt():
    """Not a total time limit — how long the server may say nothing. It says
    nothing for the whole time it is reading, and a 32B model reading 24,000
    tokens exceeds ten minutes, so requests were being cancelled while the
    model was still working."""
    settings = Settings({"request_timeout": 600})

    assert settings.timeout_for(24_000) > 600
    assert settings.timeout_for(24_000) >= 24_000 / 15


def test_a_small_prompt_just_uses_the_configured_limit():
    settings = Settings({"request_timeout": 1800})

    assert settings.timeout_for(500) == 1800
    assert settings.timeout_for(0) == 1800


def test_a_measured_machine_gets_a_tighter_estimate_than_the_guess():
    settings = Settings({"request_timeout": 60})

    measured = settings.timeout_for(24_000, prompt_rate=600.0)
    guessed = settings.timeout_for(24_000)

    assert measured < guessed


def test_the_configured_limit_is_always_a_floor():
    settings = Settings({"request_timeout": 5000})

    assert settings.timeout_for(1_000, prompt_rate=600.0) == 5000


def test_a_junk_timeout_does_not_produce_nonsense():
    assert Settings({"request_timeout": None}).timeout_for(1000) > 0


def test_the_old_ten_minute_default_is_moved_on(tmp_path):
    """load() prefers the file over the default, so raising the default does
    nothing for anyone who has already run the app."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"request_timeout": 600}), encoding="utf-8")

    assert Settings.load(path)["request_timeout"] > 600


def test_a_timeout_someone_actually_chose_is_left_alone(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"request_timeout": 900}), encoding="utf-8")

    assert Settings.load(path)["request_timeout"] == 900


def test_the_window_is_sized_to_the_prompt_rather_than_the_whole_budget():
    """The window is not free — its KV cache sits next to the weights, so
    reserving the full budget for a short question costs gigabytes to hold a
    conversation that does not exist yet."""
    settings = Settings({"context_budget_tokens": 28_000,
                         "max_context_window": 32_768})

    small = settings.context_window(500)
    large = settings.context_window(23_890)

    assert small < large < settings.context_window()
    assert settings.sampling_options(500)["num_ctx"] == small


def test_the_window_never_exceeds_the_budget_however_big_the_prompt():
    settings = Settings({"context_budget_tokens": 8_000,
                         "max_context_window": 32_768})

    assert settings.context_window(100_000) <= settings.context_window()


def test_the_window_only_changes_in_steps_so_ollama_stops_reloading():
    """num_ctx is part of the runner configuration, so every change unloads
    and reloads the model — twenty gigabytes for a 32B, per message."""
    settings = Settings({"context_budget_tokens": 32_000,
                         "max_context_window": 32_768})

    windows = {settings.context_window(n) for n in range(11_000, 14_000, 200)}

    assert len(windows) == 1        # a conversation growing slowly stays put


def test_the_window_still_grows_when_the_prompt_genuinely_does():
    settings = Settings({"context_budget_tokens": 32_000,
                         "max_context_window": 32_768})

    assert settings.context_window(5_000) < settings.context_window(25_000)
