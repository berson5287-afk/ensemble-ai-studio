"""HTTP front end for the orchestrator.

The desktop app puts orchestration on a worker thread and drains an event
queue into Tk widgets.  This does the same thing with the widgets replaced by
a browser: one POST starts a run, the response is a stream of newline-
delimited JSON events — the very same `emit` events the Tk app renders — and
the page draws cards as they arrive.

Nothing here is model-specific.  The clients dict decides what a "server" is;
on Cloud Run it holds one `GeminiClient`, in tests it holds a fake.

Boundaries, because the URL is public:

* Prompts are capped in length and runs are rate-limited per caller.
* Conversations live in memory, keyed by a browser-generated session id, and
  are forgotten after an idle hour.  Cloud Run may also recycle the instance;
  a demo does not need durable history, and saying so is more honest than a
  half-working database.
* An optional passphrase (DEMO_PASSPHRASE) turns the demo private without
  any change to the page.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..client import ChatResult
from ..formatting import friendly_model_name
from ..orchestrator import MODES, Orchestrator, Target
from ..session import Session

STATIC = Path(__file__).parent / "static"

# -- limits ----------------------------------------------------------------
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "6000"))
MAX_MODELS_PER_RUN = 4
MAX_ROUNDS = 3
MAX_TURNS = 8
RUNS_PER_MINUTE = int(os.environ.get("RUNS_PER_MINUTE", "6"))
RUN_TIMEOUT_S = int(os.environ.get("RUN_TIMEOUT_S", "600"))
SESSION_IDLE_S = 3600
# Gemini's window is enormous; the budget is what we *choose* to carry.
CONTEXT_BUDGET_TOKENS = 32_000

# The speed↔quality slider, in three stops.  Gemini's reasoning shares the
# output allowance with the answer, so the floor is higher than the desktop
# profiles use for a 7B model.
QUALITY = {
    "quick": (1024, "Be concise. Short paragraphs, only the essentials."),
    "balanced": (2048, ""),
    "detailed": (4096, "Give a detailed, well-structured answer with reasoning."),
}

WEB_MODES = ("chat", "debate", "critique", "judge", "converse", "plan")


# -- request models --------------------------------------------------------
class RunRequest(BaseModel):
    session: str = Field(min_length=8, max_length=64)
    mode: str = "chat"
    models: list[str] = Field(default_factory=list)
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    rounds: int = Field(default=2, ge=1, le=MAX_ROUNDS)
    max_turns: int = Field(default=6, ge=2, le=MAX_TURNS)
    judge: str | None = None
    quality: str = "balanced"
    system_prompt: str = Field(default="", max_length=2000)


class StopRequest(BaseModel):
    run_id: str


class SayRequest(BaseModel):
    run_id: str
    text: str = Field(min_length=1, max_length=1000)


class ResetRequest(BaseModel):
    session: str


# -- state -------------------------------------------------------------------
class Run:
    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.pending: list[str] = []
        self.lock = threading.Lock()

    def say(self, text: str) -> None:
        with self.lock:
            self.pending.append(text)

    def drain(self) -> list[str]:
        with self.lock:
            items, self.pending = self.pending, []
        return items


class Store:
    """Everything the server remembers between requests."""

    def __init__(self) -> None:
        self.sessions: dict[str, tuple[Session, float]] = {}
        self.runs: dict[str, Run] = {}
        self.calls: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def session(self, key: str) -> Session:
        now = time.monotonic()
        with self.lock:
            for stale in [k for k, (_s, seen) in self.sessions.items()
                          if now - seen > SESSION_IDLE_S]:
                del self.sessions[stale]
            session, _seen = self.sessions.get(key, (Session(), now))
            self.sessions[key] = (session, now)
            return session

    def forget(self, key: str) -> None:
        with self.lock:
            self.sessions.pop(key, None)

    def allow(self, caller: str) -> bool:
        """A sliding one-minute window per caller."""
        now = time.monotonic()
        with self.lock:
            window = self.calls[caller]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= RUNS_PER_MINUTE:
                return False
            window.append(now)
            return True

    def start(self) -> tuple[str, Run]:
        run_id = secrets.token_urlsafe(12)
        run = Run()
        with self.lock:
            self.runs[run_id] = run
        return run_id, run

    def finish(self, run_id: str) -> None:
        with self.lock:
            self.runs.pop(run_id, None)


# -- serialisation -----------------------------------------------------------
def jsonable(value: Any) -> Any:
    """Make an `emit` payload JSON-safe without the orchestrator knowing."""
    if isinstance(value, Target):
        return {"server": value.server, "model": value.model,
                "label": model_label(value.model), "short": model_label(value.model)}
    if isinstance(value, ChatResult):
        return {
            "model": value.model, "server": value.server,
            "elapsed_s": round(value.elapsed_s, 2),
            "thought_s": round(value.thought_s, 2),
            "prompt_tokens": value.prompt_tokens,
            "eval_tokens": value.eval_tokens,
            "tokens_per_second": (round(value.tokens_per_second, 1)
                                  if value.tokens_per_second else None),
            "truncated": value.truncated, "cancelled": value.cancelled,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def model_label(model: str) -> str:
    """"gemini-3.8-flash" → "Gemini 3.8 Flash"; Ollama tags keep their prettifier."""
    if model.lower().startswith("gemini"):
        return " ".join(part.capitalize() if part.isalpha() else part
                        for part in model.split("-"))
    return friendly_model_name(model) or model


def polish(payload: dict, targets: list[Target]) -> dict:
    """Swap the orchestrator's model names for the web labels in prose.

    The orchestrator writes headings and notes with `friendly_model_name`,
    which was tuned for Ollama tags and turns "gemini-3.8-flash" into
    "Gemini-3.8-Flash".  Rather than teach it about a second naming scheme,
    the web layer rewrites the two text fields it renders as titles.
    """
    swaps = {}
    for target in targets:
        raw = friendly_model_name(target.model) or target.model
        pretty = model_label(target.model)
        if raw != pretty:
            swaps[raw] = pretty
    if not swaps:
        return payload
    polished = dict(payload)
    for field in ("heading", "text"):
        value = polished.get(field)
        if isinstance(value, str):
            for raw, pretty in swaps.items():
                value = value.replace(raw, pretty)
            polished[field] = value
    return polished


# -- the app -----------------------------------------------------------------
def create_app(clients: dict[str, Any] | None = None,
               store: Store | None = None) -> FastAPI:
    """Build the app.  `clients` is injected so tests never touch a network."""
    app = FastAPI(title="Ensemble AI Studio", version=__version__,
                  docs_url=None, redoc_url=None)
    store = store or Store()
    passphrase = os.environ.get("DEMO_PASSPHRASE", "")

    if clients is None:
        from ..gemini import GeminiClient
        clients = {"cloud": GeminiClient()}
    app.state.clients = clients
    app.state.store = store

    def available() -> list[Target]:
        targets = []
        for server, client in clients.items():
            for model in client.list_models():
                targets.append(Target(server, model))
        return targets

    def caller_of(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def guard(request: Request) -> None:
        if passphrase and request.headers.get("x-passphrase", "") != passphrase:
            raise HTTPException(401, "This demo needs a passphrase.")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/api/models")
    def models(request: Request) -> dict:
        guard(request)
        return {
            "models": [{"id": t.model, "server": t.server,
                        "label": model_label(t.model)} for t in available()],
            "modes": [{"id": mode, "label": MODES[mode]} for mode in WEB_MODES],
            "limits": {"max_prompt_chars": MAX_PROMPT_CHARS,
                       "max_models": MAX_MODELS_PER_RUN,
                       "max_rounds": MAX_ROUNDS, "max_turns": MAX_TURNS,
                       "runs_per_minute": RUNS_PER_MINUTE},
            "quality": list(QUALITY),
            "version": __version__,
            "private": bool(passphrase),
        }

    @app.post("/api/run")
    def run(body: RunRequest, request: Request) -> StreamingResponse:
        guard(request)
        if body.mode not in WEB_MODES:
            raise HTTPException(400, f"Unknown mode '{body.mode}'.")
        if body.quality not in QUALITY:
            raise HTTPException(400, f"Unknown quality '{body.quality}'.")
        by_id = {t.model: t for t in available()}
        targets = [by_id[m] for m in body.models if m in by_id]
        if not targets:
            raise HTTPException(400, "Pick at least one model.")
        if len(targets) > MAX_MODELS_PER_RUN:
            raise HTTPException(400, f"At most {MAX_MODELS_PER_RUN} models per run.")
        judge = by_id.get(body.judge) if body.judge else None
        if not store.allow(caller_of(request)):
            raise HTTPException(429, f"Easy — {RUNS_PER_MINUTE} runs a minute is "
                                     f"the limit on the public demo.")

        run_id, handle = store.start()
        events: queue.Queue = queue.Queue()
        session = store.session(body.session)
        cap, directive = QUALITY[body.quality]
        system_prompt = "\n\n".join(p for p in (body.system_prompt.strip(),
                                                directive) if p)
        orchestrator = Orchestrator(
            clients=clients, session=session,
            emit=lambda kind, **payload: events.put((kind, payload)),
            cancel=handle.cancel,
            options={"temperature": 0.7, "num_predict": cap},
            budget_tokens=CONTEXT_BUDGET_TOKENS,
            system_prompt=system_prompt)

        def work() -> None:
            try:
                dispatch(orchestrator, body, targets, judge, handle)
            except Exception as exc:  # the stream must always close
                events.put(("note", {"text": f"⚠ {exc}"}))
            finally:
                events.put(("done", {"cancelled": handle.cancel.is_set()}))

        threading.Thread(target=work, name=f"run-{run_id}", daemon=True).start()

        named = targets + ([judge] if judge and judge not in targets else [])

        def stream() -> Iterator[bytes]:
            deadline = time.monotonic() + RUN_TIMEOUT_S
            yield encode("run_start", {"run_id": run_id, "mode": body.mode,
                                       "targets": targets})
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        handle.cancel.set()
                        yield encode("note", {"text": "⏱ Run timed out."})
                        yield encode("done", {"cancelled": True})
                        break
                    try:
                        kind, payload = events.get(timeout=min(15.0, remaining))
                    except queue.Empty:
                        yield b"\n"                      # keep-alive
                        continue
                    yield encode(kind, polish(payload, named))
                    if kind == "done":
                        break
            finally:
                # The browser went away, or we are finished either way.
                handle.cancel.set()
                store.finish(run_id)

        return StreamingResponse(stream(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store",
                                          "X-Accel-Buffering": "no"})

    @app.post("/api/stop")
    def stop(body: StopRequest, request: Request) -> dict:
        guard(request)
        handle = store.runs.get(body.run_id)
        if handle is not None:
            handle.cancel.set()
        return {"stopped": handle is not None}

    @app.post("/api/say")
    def say(body: SayRequest, request: Request) -> dict:
        """Hand a line to whichever model speaks next in a conversation."""
        guard(request)
        handle = store.runs.get(body.run_id)
        if handle is None:
            raise HTTPException(404, "That conversation has finished.")
        handle.say(body.text.strip())
        return {"queued": True}

    @app.post("/api/reset")
    def reset(body: ResetRequest, request: Request) -> dict:
        guard(request)
        store.forget(body.session)
        return {"cleared": True}

    @app.exception_handler(HTTPException)
    def _http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app


def dispatch(orchestrator: Orchestrator, body: RunRequest,
             targets: list[Target], judge: Target | None, handle: Run) -> None:
    """Route one request to the orchestration pattern it asked for."""
    if body.mode == "chat":
        orchestrator.broadcast(targets, body.prompt)
    elif body.mode == "debate":
        orchestrator.debate(targets, body.prompt, rounds=body.rounds, judge=judge)
    elif body.mode == "critique":
        orchestrator.critique_chain(targets, body.prompt)
    elif body.mode == "judge":
        orchestrator.judge_panel(targets, body.prompt, judge=judge)
    elif body.mode == "converse":
        orchestrator.converse(targets, body.prompt, max_turns=body.max_turns,
                              pending=handle.drain)
    elif body.mode == "plan":
        orchestrator.plan_and_work(targets[0], body.prompt)


def encode(kind: str, payload: dict) -> bytes:
    record = {"kind": kind}
    record.update(jsonable(payload))
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


# `uvicorn aichatlab.web.server:app`
app = create_app() if os.environ.get("AICHATLAB_WEB_AUTOSTART", "1") == "1" else None
