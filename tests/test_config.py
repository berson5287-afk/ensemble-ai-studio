"""Settings persistence and the speed↔quality slider mapping."""

from __future__ import annotations

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
