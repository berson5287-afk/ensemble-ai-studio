"""A small streaming Ollama client.

Two things matter here and neither is true of the naive `requests.post(...)`
version this replaced:

1. Responses stream token by token, so the UI can render as the model types.
2. Cancellation is real.  Setting the cancel event closes the HTTP response,
   which tells the server to stop generating instead of quietly finishing the
   work and having the client throw the answer away.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

TokenCallback = Callable[[str], None]


class OllamaError(RuntimeError):
    """A request failed, carrying whatever explanation the server gave."""


@dataclass
class ChatResult:
    """The outcome of one model turn, including timing for benchmarking."""

    model: str
    server: str = ""
    text: str = ""
    cancelled: bool = False
    elapsed_s: float = 0.0
    prompt_tokens: int | None = None
    eval_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float | None:
        """Generation speed, preferring the server's own nanosecond timings."""
        eval_ns = self.raw.get("eval_duration")
        if self.eval_tokens and eval_ns:
            seconds = eval_ns / 1e9
            if seconds > 0:
                return self.eval_tokens / seconds
        if self.eval_tokens and self.elapsed_s > 0:
            return self.eval_tokens / self.elapsed_s
        return None

    @property
    def label(self) -> str:
        return f"{self.model} ({self.server})" if self.server else self.model


def _error_from_response(response: requests.Response) -> OllamaError:
    """Build an error that includes the server's own message, not just a code."""
    detail = ""
    try:
        detail = (response.json() or {}).get("error", "")
    except ValueError:
        detail = (response.text or "").strip()[:300]
    return OllamaError(
        f"HTTP {response.status_code} ({response.reason}) — "
        f"{detail or 'the server gave no details'}")


class OllamaClient:
    """Talks to one Ollama server."""

    def __init__(self, base_url: str, server: str = "",
                 timeout: int = 600, connect_timeout: int = 10) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.server = server
        self.timeout = timeout
        self.connect_timeout = connect_timeout

    # -- discovery ---------------------------------------------------------
    def list_models(self, timeout: int = 15) -> list[str]:
        if not self.base_url:
            raise OllamaError("This server has no address configured.")
        response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
        if response.status_code >= 400:
            raise _error_from_response(response)
        payload = response.json()
        return sorted(m["name"] for m in payload.get("models", []) if "name" in m)

    def is_reachable(self, timeout: int = 8) -> bool:
        try:
            self.list_models(timeout=timeout)
            return True
        except Exception:
            return False

    # -- generation --------------------------------------------------------
    def chat(self, model: str, messages: list[dict[str, str]],
             options: dict[str, Any] | None = None,
             on_token: TokenCallback | None = None,
             cancel: threading.Event | None = None) -> ChatResult:
        """Stream a chat completion.

        Falls back to the older /api/generate endpoint when a server predates
        /api/chat.  Returns whatever text arrived even when cancelled.
        """
        if not self.base_url:
            raise OllamaError("This server has no address configured.")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options

        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, stream=True,
                timeout=(self.connect_timeout, self.timeout))
        except requests.RequestException as exc:
            raise OllamaError(str(exc)) from exc

        if response.status_code in (404, 405):
            response.close()
            return self._generate(model, messages, options, on_token, cancel)
        if response.status_code >= 400:
            error = _error_from_response(response)
            response.close()
            raise error

        return self._consume(response, model, on_token, cancel)

    def _generate(self, model: str, messages: list[dict[str, str]],
                  options: dict[str, Any] | None,
                  on_token: TokenCallback | None,
                  cancel: threading.Event | None) -> ChatResult:
        """Compatibility path for servers without /api/chat."""
        parts = []
        for message in messages:
            prefix = {"user": "User", "assistant": "Assistant",
                      "system": "System"}.get(message.get("role", "user"), "User")
            parts.append(f"{prefix}: {message.get('content', '')}")
        parts.append("Assistant:")

        payload: dict[str, Any] = {
            "model": model,
            "prompt": "\n\n".join(parts),
            "stream": True,
        }
        if options:
            payload["options"] = options

        try:
            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, stream=True,
                timeout=(self.connect_timeout, self.timeout))
        except requests.RequestException as exc:
            raise OllamaError(str(exc)) from exc

        if response.status_code >= 400:
            error = _error_from_response(response)
            response.close()
            raise error
        return self._consume(response, model, on_token, cancel)

    def _consume(self, response: requests.Response, model: str,
                 on_token: TokenCallback | None,
                 cancel: threading.Event | None) -> ChatResult:
        """Read the NDJSON stream, emitting tokens as they arrive."""
        started = time.monotonic()
        chunks: list[str] = []
        result = ChatResult(model=model, server=self.server)

        try:
            for line in response.iter_lines(decode_unicode=False):
                if cancel is not None and cancel.is_set():
                    result.cancelled = True
                    break
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue

                if data.get("error"):
                    raise OllamaError(str(data["error"]))

                piece = data.get("message", {}).get("content") or data.get("response") or ""
                if piece:
                    chunks.append(piece)
                    if on_token is not None:
                        on_token(piece)

                if data.get("done"):
                    result.raw = data
                    result.prompt_tokens = data.get("prompt_eval_count")
                    result.eval_tokens = data.get("eval_count")
                    break
        except requests.RequestException as exc:
            if cancel is not None and cancel.is_set():
                result.cancelled = True
            else:
                raise OllamaError(str(exc)) from exc
        finally:
            # Closing the connection is what actually stops the server working.
            response.close()

        result.text = "".join(chunks)
        result.elapsed_s = time.monotonic() - started
        return result
