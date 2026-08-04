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

from .thinking import (
    outdated_server_note,
    split_inline,
    supports_think_api,
)

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
    done_reason: str = ""
    thinking: str = ""
    thought_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """Did the server stop because it ran out of reply allowance?

        Ollama says `done_reason: "length"` when generation hit `num_predict`.
        Without surfacing it, a capped answer is indistinguishable from a
        finished one — it just stops mid-sentence and the user is left
        guessing whether the model had more to say.
        """
        return self.done_reason == "length" and not self.cancelled

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
                 timeout: int = 600, connect_timeout: int = 10,
                 server_version: str = "") -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.server = server
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        # Which Ollama this is.  Empty means "not asked yet", which is treated
        # as capable — see `thinking.supports_think_api`.
        self.server_version = server_version

    # -- discovery ---------------------------------------------------------
    def list_models(self, timeout: int = 15) -> list[str]:
        if not self.base_url:
            raise OllamaError("This server has no address configured.")
        response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
        if response.status_code >= 400:
            raise _error_from_response(response)
        payload = response.json()
        return sorted(m["name"] for m in payload.get("models", []) if "name" in m)

    def model_sizes(self, timeout: int = 15) -> dict:
        """{name: bytes on disk} — what /api/tags already knows and we threw
        away.  Size is the single number that decides whether a model will run
        on the graphics card or crawl along on the processor."""
        if not self.base_url:
            return {}
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            if response.status_code >= 400:
                return {}
            models = response.json().get("models", [])
        except (requests.RequestException, ValueError):
            return {}
        return {m["name"]: int(m.get("size") or 0)
                for m in models if isinstance(m, dict) and m.get("name")}

    def version(self, timeout: int = 8) -> str:
        """The server's own version string, or "" if it will not say.

        Worth one request: a model that downloads perfectly but will not run
        is almost always a server too old for its architecture, and that is
        indistinguishable from a broken model without this.
        """
        if not self.base_url:
            return ""
        try:
            response = requests.get(f"{self.base_url}/api/version",
                                    timeout=timeout)
            if response.status_code >= 400:
                return ""
            return str((response.json() or {}).get("version") or "")
        except (requests.RequestException, ValueError):
            return ""

    def running_models(self, timeout: int = 8) -> list[dict]:
        """What is loaded right now, and how much of it is on the GPU.

        `/api/ps` is the same thing `ollama ps` prints.  It is the only way
        to find out that a model has been quietly placed on the CPU, which
        Ollama does silently when it will not fit in VRAM.
        """
        if not self.base_url:
            return []
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=timeout)
            if response.status_code >= 400:
                return []
            return list(response.json().get("models", []))
        except (requests.RequestException, ValueError):
            return []

    def pull(self, model: str, on_progress=None,
             cancel: threading.Event | None = None,
             timeout: int = 60) -> str:
        """Download a model, through the server rather than from here.

        `/api/pull` makes the *server* fetch the weights, so this works from a
        machine whose own outbound access is blocked, and it works for a
        server in a container that has no shell available.  Progress arrives
        as NDJSON: a status line, then per-layer `completed`/`total` byte
        counts, then `{"status": "success"}`.

        Returns "" on success, or the server's reason for refusing.
        """
        if not self.base_url:
            return "This server has no address configured."
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"model": model, "stream": True}, stream=True,
                timeout=(self.connect_timeout, timeout))
        except requests.RequestException as exc:
            return str(exc)
        if response.status_code >= 400:
            error = _error_from_response(response)
            response.close()
            return str(error)

        failure = ""
        try:
            for line in response.iter_lines(decode_unicode=False):
                if cancel is not None and cancel.is_set():
                    failure = "cancelled"
                    break
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue
                if data.get("error"):
                    failure = str(data["error"])
                    break
                if on_progress is not None:
                    on_progress(data)
        except requests.RequestException as exc:
            failure = str(exc)
        finally:
            response.close()
        return failure

    def embed(self, model: str, inputs: list[str],
              timeout: int = 120) -> list[list[float]]:
        """Vectors for a batch of texts, via /api/embed.

        Falls back to the older singular /api/embeddings on servers that
        predate the batch endpoint.  Returns [] rather than raising: retrieval
        that cannot embed should quietly degrade to keyword search, not take
        the chat down with it.
        """
        if not self.base_url or not inputs:
            return []
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": list(inputs)}, timeout=timeout)
            if response.status_code < 400:
                vectors = (response.json() or {}).get("embeddings") or []
                if vectors:
                    return [[float(x) for x in v] for v in vectors]
        except (requests.RequestException, ValueError, TypeError):
            pass
        return self._embed_one_by_one(model, inputs, timeout)

    def _embed_one_by_one(self, model: str, inputs: list[str],
                          timeout: int) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in inputs:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text}, timeout=timeout)
                if response.status_code >= 400:
                    return []
                vector = (response.json() or {}).get("embedding") or []
            except (requests.RequestException, ValueError, TypeError):
                return []
            if not vector:
                return []
            vectors.append([float(x) for x in vector])
        return vectors

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
             cancel: threading.Event | None = None,
             think: bool | str | None = None,
             on_thought: TokenCallback | None = None) -> ChatResult:
        """Stream a chat completion.

        Falls back to the older /api/generate endpoint when a server predates
        /api/chat.  Returns whatever text arrived even when cancelled.

        `think` is passed straight through to Ollama and must be left as None
        for models that do not reason — the server rejects the request rather
        than ignoring the field.  `aichatlab.thinking.think_value` works out
        the right value, including the level strings GPT-OSS insists on.
        """
        if not self.base_url:
            raise OllamaError("This server has no address configured.")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if think is not None and supports_think_api(self.server_version):
            payload["think"] = think
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
            return self._generate(model, messages, options, on_token, cancel,
                                  on_thought)
        if response.status_code >= 400:
            error = _error_from_response(response)
            response.close()
            raise self._explain(error, model)

        return self._consume(response, model, on_token, cancel, on_thought)

    def _explain(self, error: OllamaError, model: str) -> OllamaError:
        """Replace an unhelpful server error with one that says what to do."""
        note = outdated_server_note(str(error), model, self.server_version)
        return OllamaError(note) if note else error

    def _generate(self, model: str, messages: list[dict[str, str]],
                  options: dict[str, Any] | None,
                  on_token: TokenCallback | None,
                  cancel: threading.Event | None,
                  on_thought: TokenCallback | None = None) -> ChatResult:
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
        return self._consume(response, model, on_token, cancel, on_thought)

    def _consume(self, response: requests.Response, model: str,
                 on_token: TokenCallback | None,
                 cancel: threading.Event | None,
                 on_thought: TokenCallback | None = None) -> ChatResult:
        """Read the NDJSON stream, emitting tokens as they arrive.

        Reasoning models send their working in `message.thinking` and the
        answer in `message.content`, interleaved in the same stream.  Both are
        surfaced, separately: dropping the reasoning is what makes a model that
        thinks for ninety seconds look like one that has hung.
        """
        started = time.monotonic()
        chunks: list[str] = []
        thoughts: list[str] = []
        first_answer_at: float | None = None
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
                    raise self._explain(OllamaError(str(data["error"])), model)

                message = data.get("message") or {}
                thought = message.get("thinking") or ""
                if thought:
                    thoughts.append(thought)
                    if on_thought is not None:
                        on_thought(thought)

                piece = message.get("content") or data.get("response") or ""
                if piece:
                    if first_answer_at is None:
                        first_answer_at = time.monotonic()
                    chunks.append(piece)
                    if on_token is not None:
                        on_token(piece)

                if data.get("done"):
                    result.raw = data
                    result.prompt_tokens = data.get("prompt_eval_count")
                    result.eval_tokens = data.get("eval_count")
                    result.done_reason = str(data.get("done_reason") or "")
                    break
        except requests.exceptions.ReadTimeout as exc:
            if cancel is not None and cancel.is_set():
                result.cancelled = True
            else:
                raise OllamaError(
                    f"The server sent nothing for {self.timeout} seconds, so "
                    f"the request was given up on. A large model reading a "
                    f"big prompt can take longer than that before its first "
                    f"word — raise 'Give up after' in ⚙ Settings."
                ) from exc
        except requests.RequestException as exc:
            if cancel is not None and cancel.is_set():
                result.cancelled = True
            else:
                raise OllamaError(str(exc)) from exc
        finally:
            # Closing the connection is what actually stops the server working.
            response.close()

        result.text = "".join(chunks)
        result.thinking = "".join(thoughts)
        result.elapsed_s = time.monotonic() - started
        if thoughts:
            # How long the model spent before writing a word, which is the
            # number that explains a long silence.
            end = first_answer_at if first_answer_at is not None else time.monotonic()
            result.thought_s = max(0.0, end - started)

        # Servers without the `thinking` field, and the /api/generate fallback,
        # leave the reasoning inline in the answer instead.
        if not result.thinking and "<think" in result.text.lower():
            reasoning, answer = split_inline(result.text)
            if reasoning:
                result.thinking, result.text = reasoning, answer
                if on_thought is not None:
                    on_thought(reasoning)
        return result
