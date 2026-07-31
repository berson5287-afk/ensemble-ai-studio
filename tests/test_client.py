"""Streaming, cancellation and error reporting."""

from __future__ import annotations

import json
import threading

import pytest
import requests

from aichatlab.client import OllamaClient, OllamaError
from tests.conftest import FakeResponse, ndjson


@pytest.fixture
def client():
    return OllamaClient("http://box:11434", server="host", timeout=5)


def test_streams_tokens_in_order(client, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        lines=ndjson("Hello", " ", "world")))

    seen = []
    result = client.chat("m:1b", [{"role": "user", "content": "hi"}],
                         on_token=seen.append)

    assert result.text == "Hello world"
    assert seen == ["Hello", " ", "world"]
    assert result.cancelled is False


def test_captures_token_stats_for_benchmarking(client, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        lines=ndjson("hi", done={"eval_count": 20, "prompt_eval_count": 7,
                                 "eval_duration": 500_000_000})))

    result = client.chat("m:1b", [{"role": "user", "content": "hi"}])

    assert result.eval_tokens == 20
    assert result.prompt_tokens == 7
    # 20 tokens in half a second
    assert result.tokens_per_second == pytest.approx(40.0)


def test_server_error_message_is_surfaced(client, monkeypatch):
    """The 500 that started all this must carry the server's explanation."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        status_code=500, reason="Internal Server Error",
        payload={"error": "model requires more system memory"}))

    with pytest.raises(OllamaError) as excinfo:
        client.chat("big:70b", [{"role": "user", "content": "hi"}])

    assert "more system memory" in str(excinfo.value)
    assert "500" in str(excinfo.value)


def test_error_inside_stream_is_raised(client, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        lines=[json.dumps({"error": "out of memory"}).encode()]))

    with pytest.raises(OllamaError, match="out of memory"):
        client.chat("m:1b", [{"role": "user", "content": "hi"}])


def test_falls_back_to_generate_on_old_servers(client, monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        if url.endswith("/api/chat"):
            return FakeResponse(status_code=404, reason="Not Found")
        return FakeResponse(lines=[
            json.dumps({"response": "legacy reply", "done": True}).encode()])

    monkeypatch.setattr(requests, "post", fake_post)
    result = client.chat("m:1b", [{"role": "user", "content": "hi"}])

    assert result.text == "legacy reply"
    assert calls == ["http://box:11434/api/chat", "http://box:11434/api/generate"]


def test_cancel_stops_stream_and_closes_connection(client, monkeypatch):
    cancel = threading.Event()
    response = FakeResponse(lines=ndjson(*[f"chunk{i}" for i in range(50)]))
    monkeypatch.setattr(requests, "post", lambda *a, **k: response)

    def on_token(_piece):
        cancel.set()          # cancel as soon as the first token lands

    result = client.chat("m:1b", [{"role": "user", "content": "hi"}],
                         on_token=on_token, cancel=cancel)

    assert result.cancelled is True
    assert result.text == "chunk0"        # partial text is kept
    assert response.closed is True        # the connection really was dropped


def test_list_models_is_sorted(client, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(
        payload={"models": [{"name": "zeta:1b"}, {"name": "alpha:2b"}]}))

    assert client.list_models() == ["alpha:2b", "zeta:1b"]


def test_unconfigured_server_refuses_politely():
    with pytest.raises(OllamaError, match="no address"):
        OllamaClient("").chat("m", [])
