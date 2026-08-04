"""SearXNG search, page reading and the research block."""

from __future__ import annotations

import pytest
import requests

from aichatlab.research import (
    ResearchError,
    SearxngClient,
    gather,
    html_to_text,
    wants_web_search,
)
from tests.conftest import FakeResponse

RESULTS = {
    "results": [
        {"title": "Pipe sizing guide", "url": "http://a.example/guide",
         "content": "How to size pipes.", "engine": "ddg"},
        {"title": "Steel handbook", "url": "http://b.example/steel",
         "content": "All about steel.", "engine": "brave"},
        {"title": "Forum thread", "url": "http://c.example/forum",
         "content": "", "engine": "ddg"},
    ]
}

PAGE_HTML = """<html><head><title>x</title><style>.a{color:red}</style></head>
<body><nav>menu menu</nav><script>var x = 1;</script>
<h1>Pipe sizing</h1><p>Use schedule 40 for most jobs.</p>
<p>Pressure rating falls with temperature.</p></body></html>"""


@pytest.fixture
def client():
    return SearxngClient("http://searx.local:8080")


def test_search_parses_results(client, monkeypatch):
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse(payload=RESULTS))
    results = client.search("pipes")

    assert [r["title"] for r in results] == [
        "Pipe sizing guide", "Steel handbook", "Forum thread"]
    assert results[0]["url"] == "http://a.example/guide"


def test_search_requests_json_format(client, monkeypatch):
    captured = {}

    def fake_get(url, params=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse(payload=RESULTS)

    monkeypatch.setattr(requests, "get", fake_get)
    client.search("pipes")

    assert captured["url"] == "http://searx.local:8080/search"
    assert captured["params"]["format"] == "json"
    assert captured["params"]["q"] == "pipes"


def test_403_explains_the_settings_yml_fix(client, monkeypatch):
    """The most common SearXNG gotcha: JSON format not enabled."""
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse(status_code=403,
                                                     reason="Forbidden"))
    with pytest.raises(ResearchError, match="settings.yml"):
        client.search("pipes")


def test_unconfigured_client_refuses(monkeypatch):
    with pytest.raises(ResearchError, match="Settings"):
        SearxngClient("").search("pipes")


def test_html_page_becomes_readable_text():
    text = html_to_text(PAGE_HTML)

    assert "schedule 40" in text
    assert "Pressure rating" in text
    assert "var x" not in text          # script stripped
    assert "color:red" not in text      # style stripped
    assert "menu menu" not in text      # nav stripped


def test_gather_builds_cited_block_and_reads_pages(client, monkeypatch):
    def fake_get(url, params=None, **kwargs):
        if "searx.local" in url:
            return FakeResponse(payload=RESULTS)
        return FakeResponse(text="", payload=None, lines=None) or None

    class PageResponse:
        status_code = 200
        headers = {"Content-Type": "text/html"}
        text = PAGE_HTML

    def routed_get(url, params=None, **kwargs):
        if "searx.local" in url:
            return FakeResponse(payload=RESULTS)
        return PageResponse()

    monkeypatch.setattr(requests, "get", routed_get)
    notes = []
    block = gather(client, "pipe sizing", fetch_pages=1, emit=notes.append)

    assert "Source 1: Pipe sizing guide" in block
    assert "Source 2: Steel handbook" in block
    assert "schedule 40" in block               # page one was actually read
    assert "(Source N)" in block
    assert any("Searching" in note for note in notes)
    assert any("Read " in note or "Skimmed" in note for note in notes)


def test_gather_with_no_results_raises(client, monkeypatch):
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse(payload={"results": []}))
    with pytest.raises(ResearchError, match="returned nothing"):
        gather(client, "xyzzy")


def test_unreachable_page_keeps_the_snippet(client, monkeypatch):
    def routed_get(url, params=None, **kwargs):
        if "searx.local" in url:
            return FakeResponse(payload=RESULTS)
        raise requests.ConnectionError("dead site")

    monkeypatch.setattr(requests, "get", routed_get)
    block = gather(client, "pipes", fetch_pages=2)

    assert "How to size pipes." in block        # snippet survives
    assert "Page content" not in block          # but no page text claimed


def test_domain_is_used_in_progress_notes(client, monkeypatch):
    class PageResponse:
        status_code = 200
        headers = {"Content-Type": "text/html"}
        text = PAGE_HTML

    def routed_get(url, params=None, **kwargs):
        if "searx.local" in url:
            return FakeResponse(payload=RESULTS)
        return PageResponse()

    monkeypatch.setattr(requests, "get", routed_get)
    notes = []
    gather(client, "pipes", fetch_pages=1, emit=notes.append)

    assert any("a.example" in note for note in notes)
    assert not any("http://" in note for note in notes)


def test_research_block_tells_the_model_to_sound_natural(client, monkeypatch):
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse(payload=RESULTS))
    block = gather(client, "pipes", fetch_pages=0)

    assert "as if you simply know this" in block
    assert "(Source N)" in block


# --------------------------------------------- "did you mean to search?"

@pytest.mark.parametrize("question", [
    "Can you look up the new Ryzen prices?",
    "look it up for me",
    "search the web for the 2026 tax brackets",
    "Search for reviews of the Yangwang U9",
    "google it and tell me",
    "check online whether the store is open",
    "what's the latest on the port strike?",
    "find the current price of copper",
    "give me up-to-date information on the recall",
    "do some research online about heat pumps",
])
def test_explicit_lookup_requests_are_recognised(question):
    assert wants_web_search(question)


@pytest.mark.parametrize("question", [
    "explain how a heat pump works",
    "what is my current project structure?",       # "current" is not a search
    "search this file for the bug",                # searching, but not the web
    "Google was founded in 1998 — who by?",        # a company, not a command
    "summarise the attached document",
    "",
])
def test_ordinary_questions_do_not_trigger_the_prompt(question):
    assert not wants_web_search(question)


def test_an_explicit_refusal_to_search_wins():
    assert not wants_web_search("what's the latest kernel? don't search, "
                                "just tell me what you remember")
    assert not wants_web_search("answer without searching the web")


def test_the_matched_phrase_comes_back_so_it_can_be_quoted():
    assert wants_web_search("please look up the spec").lower() == "look up"


# -- questions that need a lookup without asking for one -------------------
def test_a_question_about_the_newest_thing_is_flagged():
    """The model answers "the newest Audi R8" from training data, confidently,
    and nothing says its newest and the world's newest are different cars."""
    from aichatlab.research import looks_time_sensitive

    assert looks_time_sensitive("what is the 0-60 of the newest audi r8?")
    assert looks_time_sensitive("what's the latest version of Ollama?")
    assert looks_time_sensitive("who is the CEO of Intel these days?")
    assert looks_time_sensitive("how much is a 5090 right now?")
    assert looks_time_sensitive("what's the weather today?")


def test_a_timeless_question_is_not_flagged():
    from aichatlab.research import looks_time_sensitive

    assert not looks_time_sensitive("how does a heat pump work?")
    assert not looks_time_sensitive("what is the best pizza in New York?")
    assert not looks_time_sensitive("explain the difference between q4 and q8")


def test_questions_about_the_attached_material_are_left_alone():
    """"the current implementation" is about the code in hand, not the world,
    and prompting for a web search there is pure noise."""
    from aichatlab.research import looks_time_sensitive

    assert not looks_time_sensitive("what does this code currently do?")
    assert not looks_time_sensitive("is my current approach in this file ok?")
    assert not looks_time_sensitive("you said the latest change was safe — why?")


def test_an_explicit_refusal_still_wins():
    from aichatlab.research import looks_time_sensitive

    assert not looks_time_sensitive(
        "what is the latest version? don't search, just tell me")


def test_the_matched_phrase_is_returned_so_it_can_be_quoted():
    from aichatlab.research import looks_time_sensitive

    assert looks_time_sensitive("the newest audi r8") == "newest"
