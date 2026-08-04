"""Research caching: TTL, near-match reuse and session cleanup."""

from __future__ import annotations

import json

import pytest

from aichatlab.cache import ResearchCache, is_volatile, normalise, similarity, ttl_for


class Clock:
    """A hand-cranked clock so TTL tests don't have to wait."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def cache(tmp_path):
    return ResearchCache(tmp_path / "cache.json", clock=Clock())


def test_stores_and_returns_research(cache):
    cache.put("how to sweat a copper pipe", "BLOCK")
    hit = cache.get("how to sweat a copper pipe")

    assert hit is not None
    assert hit.block == "BLOCK"
    assert hit.exact is True


def test_lookup_ignores_case_and_spacing(cache):
    cache.put("How To Sweat  A Copper Pipe", "BLOCK")
    assert cache.get("how to sweat a copper pipe") is not None


def test_miss_returns_nothing(cache):
    cache.put("plumbing", "BLOCK")
    assert cache.get("quantum chromodynamics") is None


def test_near_identical_questions_reuse_the_same_research(cache):
    cache.put("pearl river ny weather forecast this weekend", "BLOCK")
    hit = cache.get("weather forecast pearl river ny this weekend")

    assert hit is not None
    assert hit.exact is False           # matched by overlap, not exactly


def test_unrelated_questions_do_not_share_research(cache):
    cache.put("pearl river ny weather forecast", "BLOCK")
    assert cache.get("how do I rewire a light switch") is None


@pytest.mark.parametrize("query", [
    "what is the weather this weekend",
    "todays news headlines",
    "current price of copper",
    "tomorrow's forecast",
])
def test_time_sensitive_questions_are_spotted(query):
    assert is_volatile(query) is True


@pytest.mark.parametrize("query", [
    "how to sweat a copper pipe",
    "difference between AC and DC wiring",
])
def test_durable_questions_are_not_time_sensitive(query):
    assert is_volatile(query) is False


def test_weather_expires_far_sooner_than_general_knowledge():
    assert ttl_for("weather this weekend", 360) == 900
    assert ttl_for("how to sweat a copper pipe", 360) == 360 * 60


def test_expired_entries_are_not_returned(tmp_path):
    clock = Clock()
    cache = ResearchCache(tmp_path / "c.json", clock=clock)
    cache.put("weather in pearl river ny", "OLD FORECAST")

    clock.advance(600)                       # 10 minutes — still fresh
    assert cache.get("weather in pearl river ny") is not None

    clock.advance(600)                       # 20 minutes — stale weather
    assert cache.get("weather in pearl river ny") is None


def test_durable_research_survives_the_same_gap(tmp_path):
    clock = Clock()
    cache = ResearchCache(tmp_path / "c.json", clock=clock)
    cache.put("how to sweat a copper pipe", "GUIDE")

    clock.advance(3600)
    assert cache.get("how to sweat a copper pipe") is not None


def test_age_is_reported_in_words(tmp_path):
    clock = Clock()
    cache = ResearchCache(tmp_path / "c.json", clock=clock)
    cache.put("how to wire a switch", "GUIDE")

    clock.advance(120)
    assert cache.get("how to wire a switch").age_phrase == "2 minutes ago"
    clock.advance(3600)
    assert "hour" in cache.get("how to wire a switch").age_phrase


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.json"
    ResearchCache(path, clock=Clock()).put("wiring a three way switch", "GUIDE")

    reopened = ResearchCache(path, clock=Clock())
    assert reopened.get("wiring a three way switch") is not None


def test_discarding_a_session_removes_only_what_it_added(tmp_path):
    path = tmp_path / "c.json"
    earlier = ResearchCache(path, clock=Clock())
    earlier.put("kept from a saved chat", "OLD")
    earlier.keep_session()

    current = ResearchCache(path, clock=Clock())
    current.put("added in this unsaved chat", "NEW")
    current.discard_session()

    reopened = ResearchCache(path, clock=Clock())
    assert reopened.get("kept from a saved chat") is not None
    assert reopened.get("added in this unsaved chat") is None


def test_discarding_everything_removes_the_file(tmp_path):
    path = tmp_path / "c.json"
    cache = ResearchCache(path, clock=Clock())
    cache.put("only entry", "BLOCK")
    cache.discard_session()

    assert not path.exists()


def test_keeping_a_session_leaves_entries_for_next_time(tmp_path):
    path = tmp_path / "c.json"
    cache = ResearchCache(path, clock=Clock())
    cache.put("saved chat research", "BLOCK")
    cache.keep_session()

    assert ResearchCache(path, clock=Clock()).get("saved chat research")


def test_a_corrupt_cache_file_is_ignored(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{ not json", encoding="utf-8")

    cache = ResearchCache(path, clock=Clock())
    assert len(cache) == 0
    cache.put("something", "BLOCK")
    assert cache.get("something") is not None


def test_expired_entries_are_pruned_on_write(tmp_path):
    clock = Clock()
    cache = ResearchCache(path=tmp_path / "c.json", clock=clock)
    cache.put("weather today", "OLD")
    clock.advance(2000)
    cache.put("how to solder", "NEW")

    saved = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert "weather today" not in saved["entries"]
    assert len(cache) == 1


def test_empty_input_is_not_cached(cache):
    cache.put("", "BLOCK")
    cache.put("question", "")
    assert len(cache) == 0


def test_similarity_and_normalise_helpers():
    assert normalise("  Hello   World ") == "hello world"
    assert similarity("copper pipe soldering", "soldering copper pipe") == 1.0
    assert similarity("copper pipe", "tax law") == 0.0


# ------------------------------------------------- "0 minutes" = never expire

def test_zero_minutes_means_never_for_durable_questions():
    assert ttl_for("how to sweat a copper pipe", 0) is None


def test_zero_minutes_still_expires_time_sensitive_questions():
    """No setting should let the app serve last week's forecast."""
    assert ttl_for("weather this weekend", 0) == 900


def test_never_expiring_research_survives_a_year(tmp_path):
    clock = Clock()
    cache = ResearchCache(tmp_path / "c.json", base_ttl_minutes=0, clock=clock)
    cache.put("how to wire a three way switch", "GUIDE")

    clock.advance(365 * 24 * 3600)
    assert cache.get("how to wire a three way switch") is not None


def test_weather_still_goes_stale_when_set_to_never(tmp_path):
    clock = Clock()
    cache = ResearchCache(tmp_path / "c.json", base_ttl_minutes=0, clock=clock)
    cache.put("weather tomorrow in pearl river", "FORECAST")

    clock.advance(1200)                       # 20 minutes
    assert cache.get("weather tomorrow in pearl river") is None


def test_negative_and_junk_ttl_are_handled():
    assert ttl_for("how to solder", -5) is None
    assert ttl_for("how to solder", "banana") == 360 * 60
