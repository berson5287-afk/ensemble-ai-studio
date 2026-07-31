"""Shared test doubles — no network, no display."""

from __future__ import annotations

import json
import threading

import pytest

from aichatlab.client import ChatResult


class FakeResponse:
    """Stands in for a streaming requests.Response."""

    def __init__(self, status_code: int = 200, payload=None,
                 lines: list[bytes] | None = None, text: str = "",
                 reason: str = "OK", on_close=None):
        self.status_code = status_code
        self.reason = reason
        self._payload = payload
        self._lines = lines or []
        self.text = text
        self.closed = False
        self._on_close = on_close

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            if self.closed:
                return
            yield line

    def close(self):
        self.closed = True
        if self._on_close:
            self._on_close()


def ndjson(*texts: str, done: dict | None = None) -> list[bytes]:
    """Build an Ollama-style NDJSON stream from chunks of text."""
    lines = [json.dumps({"message": {"role": "assistant", "content": chunk},
                         "done": False}).encode()
             for chunk in texts]
    payload = {"message": {"role": "assistant", "content": ""}, "done": True}
    payload.update(done or {})
    lines.append(json.dumps(payload).encode())
    return lines


class FakeClient:
    """A stand-in for OllamaClient that records what it was asked."""

    def __init__(self, server: str = "local", reply: str = "",
                 fail_on: str | None = None):
        self.base_url = "http://fake:11434"
        self.server = server
        self.reply = reply
        self.fail_on = fail_on
        self.calls: list[dict] = []

    def chat(self, model, messages, options=None, on_token=None, cancel=None):
        self.calls.append({"model": model, "messages": messages,
                           "options": options})
        if self.fail_on and self.fail_on in model:
            raise RuntimeError("boom")
        text = self.reply or f"reply from {model}"
        if on_token:
            on_token(text)
        return ChatResult(model=model, server=self.server, text=text,
                          elapsed_s=0.25, eval_tokens=12,
                          raw={"eval_duration": 250_000_000})

    def list_models(self, timeout=15):
        return ["alpha:1b", "beta:2b"]

    @property
    def last_prompt(self) -> str:
        return self.calls[-1]["messages"][-1]["content"]


@pytest.fixture
def fake_client():
    return FakeClient()


@pytest.fixture
def cancel_event():
    return threading.Event()
