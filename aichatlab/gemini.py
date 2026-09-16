"""A Gemini client with the same shape as `OllamaClient`.

The orchestrator does not care where a model lives.  It asks a client for a
streamed chat turn, hands it a cancel event and two callbacks, and gets a
`ChatResult` back.  That is the whole contract, and it is small enough that a
second backend is one file: this one talks to Gemini through Vertex AI (or the
Gemini Developer API) instead of to an Ollama server on the LAN.

Why this exists at all: the desktop app runs on a GPU under the desk.  The
hosted demo runs on Cloud Run, which has no GPU and scales to zero, so the
models have to be somewhere else — and "somewhere else" with a service account
and no keys to leak is Vertex AI.

What is deliberately *not* here: model discovery.  Vertex publishes dozens of
models and most are not chat models, so the list is configuration
(`GEMINI_MODELS`) rather than a lookup.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from typing import Any

from .client import ChatResult, OllamaError, TokenCallback

# Chat models the hosted demo offers when nothing else is configured.  Three
# distinct tiers so a debate or a judge panel is not the same model arguing
# with itself.  Overridable with GEMINI_MODELS=a,b,c.
DEFAULT_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
)

# Vertex's "global" endpoint routes to wherever there is capacity, which is
# the right default for a demo that must not fail on a regional stockout.
DEFAULT_LOCATION = "global"

# Ollama's role names → Gemini's.  System prompts travel separately.
ROLE_MAP = {"user": "user", "assistant": "model"}


def configured_models(raw: str | None = None) -> list[str]:
    """Parse GEMINI_MODELS, falling back to the defaults."""
    text = os.environ.get("GEMINI_MODELS", "") if raw is None else raw
    models = [m.strip() for m in text.split(",") if m.strip()]
    return models or list(DEFAULT_MODELS)


def split_system(messages: Sequence[dict]) -> tuple[str, list[dict]]:
    """Lift system messages out into one instruction; merge adjacent turns.

    Gemini takes the system prompt as a separate field and expects the
    conversation to alternate.  Two user messages in a row — which happens
    after a compaction summary — are joined rather than rejected.
    """
    system_parts: list[str] = []
    contents: list[dict] = []
    for message in messages:
        role = message.get("role", "user")
        text = str(message.get("content") or "")
        if role == "system":
            if text:
                system_parts.append(text)
            continue
        gemini_role = ROLE_MAP.get(role, "user")
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"][0]["text"] += "\n\n" + text
        else:
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
    # A conversation must open with the user, not the model.
    while contents and contents[0]["role"] != "user":
        contents.pop(0)
    return "\n\n".join(system_parts), contents


def build_config(options: dict[str, Any] | None, system: str,
                 think: bool | str | None) -> dict[str, Any]:
    """Translate Ollama-style options into a GenerateContentConfig dict.

    `num_ctx` has no Gemini equivalent (the window is what it is) and is
    dropped.  `num_predict` becomes `max_output_tokens`.  Thoughts are
    requested so the UI can show them the way it shows Ollama's `thinking`.
    """
    options = options or {}
    config: dict[str, Any] = {}
    if system:
        config["system_instruction"] = system
    if "temperature" in options:
        try:
            config["temperature"] = float(options["temperature"])
        except (TypeError, ValueError):
            pass
    cap = int(options.get("num_predict") or 0)
    if cap > 0:
        config["max_output_tokens"] = cap
    if think is not False:
        config["thinking_config"] = {"include_thoughts": True}
    return config


class GeminiClient:
    """Talks to Gemini, quacking exactly like `OllamaClient`.

    Authentication is whatever the environment provides: on Cloud Run that is
    the service account, locally it is `gcloud auth application-default
    login`.  Set GEMINI_API_KEY to use the Developer API instead of Vertex.
    """

    def __init__(self, models: Sequence[str] | None = None, *,
                 project: str | None = None, location: str | None = None,
                 api_key: str | None = None, server: str = "cloud",
                 timeout: int = 600, client: Any = None) -> None:
        self.models = list(models or configured_models())
        self.server = server
        self.timeout = timeout
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION",
                                                   DEFAULT_LOCATION)
        self.api_key = api_key if api_key is not None else os.environ.get(
            "GEMINI_API_KEY", "")
        # The orchestrator treats an empty base_url as "not configured".
        # There is no URL here, so this is a label that says where we point.
        self.base_url = (f"gemini-api://{self.server}" if self.api_key
                         else f"vertex://{self.project}/{self.location}")
        self.server_version = "gemini"
        self._client = client
        self._client_lock = threading.Lock()

    # -- SDK ---------------------------------------------------------------
    def _sdk(self) -> Any:
        """The google-genai client, built once on first use.

        Built under a lock, because a broadcast is the first thing a fresh
        instance does: four threads arrive here together, each builds a
        client, one assignment wins, and the losers are garbage-collected
        mid-stream — the SDK closes its HTTP client on the way out, and the
        three streams still using it fail with "client has been closed".
        """
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                from google import genai  # imported here: optional dependency

                if self.api_key:
                    self._client = genai.Client(api_key=self.api_key)
                else:
                    if not self.project:
                        raise OllamaError(
                            "No Google Cloud project configured. Set "
                            "GOOGLE_CLOUD_PROJECT (Vertex AI) or GEMINI_API_KEY.")
                    self._client = genai.Client(
                        vertexai=True, project=self.project,
                        location=self.location)
        return self._client

    # -- discovery (the shape the app expects) -----------------------------
    def list_models(self, timeout: int = 15) -> list[str]:
        return list(self.models)

    def model_sizes(self, timeout: int = 15) -> dict:
        return {}

    def version(self, timeout: int = 8) -> str:
        return self.server_version

    def running_models(self, timeout: int = 8) -> list[dict]:
        return []

    def is_reachable(self, timeout: int = 8) -> bool:
        try:
            self._sdk()
            return True
        except Exception:
            return False

    # -- generation --------------------------------------------------------
    def chat(self, model: str, messages: list[dict[str, str]],
             options: dict[str, Any] | None = None,
             on_token: TokenCallback | None = None,
             cancel: threading.Event | None = None,
             think: bool | str | None = None,
             on_thought: TokenCallback | None = None) -> ChatResult:
        """Stream one turn.  Cancel by setting the event; the stream is dropped."""
        system, contents = split_system(messages)
        if not contents:
            raise OllamaError("Nothing to send: the conversation has no user turn.")
        config = build_config(options, system, think)

        started = time.monotonic()
        first_answer_at: float | None = None
        chunks: list[str] = []
        thoughts: list[str] = []
        result = ChatResult(model=model, server=self.server)
        usage: Any = None
        finish = ""

        try:
            stream = self._sdk().models.generate_content_stream(
                model=model, contents=contents, config=config)
            for chunk in stream:
                if cancel is not None and cancel.is_set():
                    result.cancelled = True
                    break
                for candidate in (getattr(chunk, "candidates", None) or []):
                    reason = getattr(candidate, "finish_reason", None)
                    if reason:
                        finish = str(getattr(reason, "name", reason))
                    content = getattr(candidate, "content", None)
                    for part in (getattr(content, "parts", None) or []):
                        text = getattr(part, "text", None) or ""
                        if not text:
                            continue
                        if getattr(part, "thought", False):
                            thoughts.append(text)
                            if on_thought is not None:
                                on_thought(text)
                        else:
                            if first_answer_at is None:
                                first_answer_at = time.monotonic()
                            chunks.append(text)
                            if on_token is not None:
                                on_token(text)
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage = chunk.usage_metadata
        except OllamaError:
            raise
        except Exception as exc:
            if cancel is not None and cancel.is_set():
                result.cancelled = True
            else:
                raise OllamaError(explain(exc, model)) from exc

        result.text = "".join(chunks)
        result.thinking = "".join(thoughts)
        result.elapsed_s = time.monotonic() - started
        if thoughts:
            end = first_answer_at if first_answer_at is not None else time.monotonic()
            result.thought_s = max(0.0, end - started)
        if usage is not None:
            result.prompt_tokens = getattr(usage, "prompt_token_count", None)
            result.eval_tokens = getattr(usage, "candidates_token_count", None)
            thought_tokens = getattr(usage, "thoughts_token_count", None)
            result.raw = {
                "prompt_eval_count": result.prompt_tokens,
                "eval_count": result.eval_tokens,
                "thoughts_token_count": thought_tokens,
                # Ollama reports nanoseconds; give ChatResult the same so
                # tokens/sec works without a special case.
                "eval_duration": int(result.elapsed_s * 1e9),
            }
        # Gemini says MAX_TOKENS where Ollama says "length".
        result.done_reason = "length" if finish == "MAX_TOKENS" else finish.lower()
        return result


def explain(exc: Exception, model: str) -> str:
    """Turn an SDK exception into one sentence that says what to do."""
    text = str(exc)
    lowered = text.lower()
    if "404" in text or "not found" in lowered:
        return (f"Vertex AI does not know a model called '{model}'. Check "
                f"GEMINI_MODELS against the current Gemini model list.")
    if "403" in text or "permission" in lowered:
        return ("Permission denied by Vertex AI. The service account needs "
                "roles/aiplatform.user and the Vertex AI API must be enabled.")
    if "429" in text or "quota" in lowered or "resource_exhausted" in lowered:
        return "Gemini is rate-limiting this project right now. Try again shortly."
    if "401" in text or "credential" in lowered or "default credentials" in lowered:
        return ("No Google credentials found. On Cloud Run this is the service "
                "account; locally run `gcloud auth application-default login`.")
    return text[:300]
