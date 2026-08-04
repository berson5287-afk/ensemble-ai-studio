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


# --------------------------------------------- replies that were cut short

def test_a_reply_stopped_by_the_length_cap_is_flagged(monkeypatch):
    """Ollama says done_reason "length" when generation hit num_predict.
    Without surfacing it, a capped answer that stops mid-sentence looks
    exactly like a finished one."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        lines=ndjson("Here are a few ideas: 1. Centralise",
                     done={"done_reason": "length", "eval_count": 192})))

    result = OllamaClient("http://x").chat("m", [{"role": "user", "content": "hi"}])

    assert result.truncated
    assert result.done_reason == "length"


def test_a_reply_that_finished_normally_is_not_flagged(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        lines=ndjson("All done.", done={"done_reason": "stop"})))

    assert not OllamaClient("http://x").chat(
        "m", [{"role": "user", "content": "hi"}]).truncated


def test_a_reply_with_no_done_reason_is_not_flagged(monkeypatch):
    """Older Ollama builds omit the field entirely."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        lines=ndjson("All done.")))

    assert not OllamaClient("http://x").chat(
        "m", [{"role": "user", "content": "hi"}]).truncated


def test_a_cancelled_reply_is_not_reported_as_truncated(monkeypatch):
    """You stopped it on purpose — that is not the model running out of room."""
    from aichatlab.client import ChatResult

    result = ChatResult(model="m", done_reason="length", cancelled=True)

    assert not result.truncated


# -- reasoning models ------------------------------------------------------
def reasoning_stream(thoughts, answers, done=None):
    """An Ollama stream that reasons first, then answers."""
    lines = [json.dumps({"message": {"role": "assistant", "thinking": t},
                         "done": False}).encode() for t in thoughts]
    lines += [json.dumps({"message": {"role": "assistant", "content": c},
                          "done": False}).encode() for c in answers]
    payload = {"message": {"role": "assistant", "content": ""}, "done": True}
    payload.update(done or {})
    lines.append(json.dumps(payload).encode())
    return lines


def test_reasoning_is_captured_separately_from_the_answer(client, monkeypatch):
    """Reading only `content` is what makes a reasoning model look hung."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        lines=reasoning_stream(["The user wants ", "a number."],
                               ["It is ", "42."])))

    thoughts, tokens = [], []
    result = client.chat("qwen3:8b", [{"role": "user", "content": "hi"}],
                         on_token=tokens.append, on_thought=thoughts.append,
                         think=True)

    assert result.text == "It is 42."
    assert result.thinking == "The user wants a number."
    assert thoughts == ["The user wants ", "a number."]
    assert tokens == ["It is ", "42."]


def test_reasoning_streams_live_rather_than_arriving_at_the_end(client,
                                                               monkeypatch):
    """The whole point is showing life during the silence before the answer."""
    order = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        lines=reasoning_stream(["thinking…"], ["answer"])))

    client.chat("qwen3:8b", [{"role": "user", "content": "hi"}],
                on_token=lambda t: order.append(("token", t)),
                on_thought=lambda t: order.append(("thought", t)),
                think=True)

    assert order[0][0] == "thought"
    assert order[-1][0] == "token"


def test_the_think_field_is_sent_only_when_asked(client, monkeypatch):
    sent = {}

    def fake_post(url, **kwargs):
        sent.update(kwargs.get("json") or {})
        return FakeResponse(lines=ndjson("hi"))

    monkeypatch.setattr(requests, "post", fake_post)

    client.chat("qwen3:8b", [{"role": "user", "content": "hi"}], think=True)
    assert sent["think"] is True

    sent.clear()
    client.chat("llama3:8b", [{"role": "user", "content": "hi"}])
    assert "think" not in sent, "Ollama rejects `think` on a model that " \
                                "does not reason, so it must be omitted"


def test_a_level_string_is_passed_through_for_gpt_oss(client, monkeypatch):
    sent = {}

    def fake_post(url, **kwargs):
        sent.update(kwargs.get("json") or {})
        return FakeResponse(lines=ndjson("hi"))

    monkeypatch.setattr(requests, "post", fake_post)
    client.chat("gpt-oss:20b", [{"role": "user", "content": "hi"}],
                think="medium")

    assert sent["think"] == "medium"


def test_time_spent_reasoning_is_measured(client, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        lines=reasoning_stream(["mm"], ["done"])))

    result = client.chat("qwen3:8b", [{"role": "user", "content": "hi"}],
                         think=True)

    assert result.thought_s >= 0
    assert result.thinking


def test_inline_thought_tags_are_split_out_on_old_servers(client, monkeypatch):
    """Servers predating the `thinking` field leave it in the content."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        lines=ndjson("<think>", "hmm, 6*7", "</think>", "It is 42.")))

    caught = []
    result = client.chat("qwen3:8b", [{"role": "user", "content": "hi"}],
                         on_thought=caught.append)

    assert result.text == "It is 42."
    assert "hmm, 6*7" in result.thinking
    assert caught, "the reasoning was found but never handed to the caller"
