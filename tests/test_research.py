"""SearXNG search, page reading and the research block."""

from __future__ import annotations

import pytest
import requests

from aichatlab.research import ResearchError, SearxngClient, gather, html_to_text
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
