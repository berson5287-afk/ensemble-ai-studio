"""The Gemini adapter, exercised against a fake SDK — no network, no keys."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from aichatlab.client import OllamaError
from aichatlab.gemini import (
    DEFAULT_MODELS,
    GeminiClient,
    build_config,
    configured_models,
    explain,
    split_system,
)
from aichatlab.orchestrator import Orchestrator, Target
from aichatlab.session import Session


def chunk(text: str = "", thought: str = "", finish: str | None = None,
          usage: dict | None = None):
    """One streamed GenerateContentResponse, as attribute bags."""
    parts = []
    if thought:
        parts.append(SimpleNamespace(text=thought, thought=True))
    if text:
        parts.append(SimpleNamespace(text=text, thought=False))
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        finish_reason=SimpleNamespace(name=finish) if finish else None)
    meta = SimpleNamespace(**usage) if usage else None
    return SimpleNamespace(candidates=[candidate], usage_metadata=meta)


class FakeSDK:
    """Stands in for google.genai.Client, recording every request."""

    def __init__(self, chunks=None, error: Exception | None = None):
        self.chunks = chunks or []
        self.error = error
        self.calls: list[dict] = []
        self.models = self

    def generate_content_stream(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        yield from self.chunks


def client(chunks=None, error=None) -> tuple[GeminiClient, FakeSDK]:
    sdk = FakeSDK(chunks, error)
    return GeminiClient(["gemini-test"], project="p", client=sdk), sdk


# -- message shaping ---------------------------------------------------------
def test_system_messages_become_one_instruction():
    system, contents = split_system([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "be kind"},
        {"role": "assistant", "content": "hello"},
    ])
    assert system == "be brief\n\nbe kind"
    assert [c["role"] for c in contents] == ["user", "model"]


def test_adjacent_same_role_turns_are_merged():
    _system, contents = split_system([
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ])
    assert len(contents) == 1
    assert contents[0]["parts"][0]["text"] == "first\n\nsecond"


def test_conversation_cannot_open_with_the_model():
    _system, contents = split_system([
        {"role": "assistant", "content": "[Earlier conversation, summarised] …"},
        {"role": "user", "content": "go on"},
    ])
    assert contents[0]["role"] == "user"


def test_options_translate_to_gemini_config():
    config = build_config({"temperature": 0.3, "num_predict": 500, "num_ctx": 8192},
                          "sys", think=None)
    assert config["temperature"] == 0.3
    assert config["max_output_tokens"] == 500
    assert config["system_instruction"] == "sys"
    assert "num_ctx" not in config
    assert config["thinking_config"] == {"include_thoughts": True}


def test_uncapped_reply_sends_no_max_tokens():
    assert "max_output_tokens" not in build_config({"num_predict": -1}, "", None)


def test_thinking_off_requests_no_thoughts():
    assert "thinking_config" not in build_config({}, "", think=False)


# -- streaming ---------------------------------------------------------------
def test_chat_streams_tokens_and_thoughts_separately():
    gemini, sdk = client([
        chunk(thought="let me think"),
        chunk(text="Hel"),
        chunk(text="lo", finish="STOP",
              usage={"prompt_token_count": 7, "candidates_token_count": 2,
                     "thoughts_token_count": 3}),
    ])
    tokens, thoughts = [], []

    result = gemini.chat("gemini-test",
                         [{"role": "system", "content": "sys"},
                          {"role": "user", "content": "hi"}],
                         options={"temperature": 0.5, "num_predict": 100},
                         on_token=tokens.append, on_thought=thoughts.append)

    assert tokens == ["Hel", "lo"]
    assert thoughts == ["let me think"]
    assert result.text == "Hello"
    assert result.thinking == "let me think"
    assert result.prompt_tokens == 7 and result.eval_tokens == 2
    assert result.done_reason == "stop"
    assert not result.truncated
    assert result.server == "cloud"
    call = sdk.calls[0]
    assert call["model"] == "gemini-test"
    assert call["config"]["system_instruction"] == "sys"
    assert call["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_max_tokens_reads_as_truncated_like_ollama():
    gemini, _sdk = client([chunk(text="partial", finish="MAX_TOKENS")])
    result = gemini.chat("gemini-test", [{"role": "user", "content": "hi"}])
    assert result.done_reason == "length"
    assert result.truncated


def test_cancel_stops_the_stream_and_keeps_what_arrived():
    cancel = threading.Event()
    gemini, _sdk = client([chunk(text="one "), chunk(text="two "), chunk(text="three")])

    def on_token(piece):
        if piece == "two ":
            cancel.set()

    result = gemini.chat("gemini-test", [{"role": "user", "content": "hi"}],
                         on_token=on_token, cancel=cancel)
    assert result.cancelled
    assert result.text == "one two "


def test_sdk_errors_become_ollama_errors_with_advice():
    gemini, _sdk = client(error=RuntimeError("403 PERMISSION_DENIED: nope"))
    with pytest.raises(OllamaError) as caught:
        gemini.chat("gemini-test", [{"role": "user", "content": "hi"}])
    assert "aiplatform.user" in str(caught.value)


def test_empty_conversation_is_refused():
    gemini, _sdk = client()
    with pytest.raises(OllamaError):
        gemini.chat("gemini-test", [{"role": "system", "content": "only a system prompt"}])


@pytest.mark.parametrize("message, needle", [
    ("404 NOT_FOUND model", "GEMINI_MODELS"),
    ("429 RESOURCE_EXHAUSTED", "rate-limit"),
    ("Could not automatically determine credentials", "application-default"),
    ("something odd", "something odd"),
])
def test_explanations_say_what_to_do(message, needle):
    assert needle in explain(RuntimeError(message), "m")


# -- configuration -----------------------------------------------------------
def test_models_come_from_the_environment_or_defaults(monkeypatch):
    assert configured_models("a, b ,,c") == ["a", "b", "c"]
    assert configured_models("") == list(DEFAULT_MODELS)
    monkeypatch.setenv("GEMINI_MODELS", "x")
    assert configured_models() == ["x"]


def test_base_url_marks_the_backend_without_being_a_url(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    vertex = GeminiClient(["m"], project="proj", location="global", client=object())
    assert vertex.base_url == "vertex://proj/global"
    direct = GeminiClient(["m"], api_key="k", client=object())
    assert direct.base_url.startswith("gemini-api://")


def test_missing_project_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(OllamaError) as caught:
        GeminiClient(["m"], project="")._sdk()
    assert "GOOGLE_CLOUD_PROJECT" in str(caught.value)


# -- through the orchestrator ------------------------------------------------
def test_orchestrator_drives_gemini_like_any_other_client():
    gemini, sdk = client([chunk(text="reply", finish="STOP")])
    events = []
    orchestrator = Orchestrator(
        clients={"cloud": gemini}, session=Session(),
        emit=lambda kind, **payload: events.append((kind, payload)),
        options={"temperature": 0.7, "num_predict": 2048})

    results = orchestrator.broadcast([Target("cloud", "gemini-test")], "hello")

    assert [r.text for r in results] == ["reply"]
    assert [k for k, _ in events] == ["turn_start", "token", "turn_end"]
    assert sdk.calls[0]["config"]["max_output_tokens"] == 2048


def test_sdk_client_is_built_exactly_once_under_concurrency(monkeypatch):
    """A broadcast hits `_sdk` from several threads at once; one client must win."""
    import sys
    import types

    built = []

    class Client:
        def __init__(self, **kwargs):
            time.sleep(0.02)                 # widen the window for the race
            built.append(self)

    fake_genai = types.SimpleNamespace(Client=Client)
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    gemini = GeminiClient(["m"], project="p")
    seen = []
    workers = [threading.Thread(target=lambda: seen.append(gemini._sdk()))
               for _ in range(8)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    assert len(built) == 1
    assert all(s is built[0] for s in seen)
