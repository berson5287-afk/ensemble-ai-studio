"""A control channel into a running window, for when the log is not enough.

The edit log answers "what did the app decide". It cannot answer "what happens
if I attach this folder and ask that question", and the gap between those two
is where an afternoon goes: a change is made, the app is restarted by hand, a
folder is attached by hand, a question is typed by hand, and one data point
comes back. Doing that fifty times is how a fix gets verified, and nobody does
it fifty times.

So the window can be driven from outside. It listens on loopback for a few
plain commands — what state are you in, attach this, send that, what did you
say, what did the edit check decide — and runs each one on the Tk thread the
same way a click would.

Three rules make that safe enough to ship:

*Off unless asked for.* It starts only when `AICHATLAB_BRIDGE=1` is set or the
`control_bridge` setting is on. No default-on port.

*Loopback and a token.* It binds 127.0.0.1 and nothing else, and every command
must carry a token written into the user's own home directory when the server
starts. A port on localhost is still a port.

*No new powers.* Every command goes through the same methods the buttons call,
so the review window still stands in front of the filesystem. The bridge can
ask the app to propose an edit; it cannot write a file behind the dialog that
exists to prevent exactly that.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import socket
import threading
from pathlib import Path

HANDSHAKE_PATH = Path.home() / ".ai_chat_lab_bridge.json"

# Loopback only.  Not configurable on purpose: "which interface should the
# remote-control port listen on" is not a question this app should be asking.
HOST = "127.0.0.1"

# A command runs on the UI thread, so it waits for whatever that thread is
# doing.  Long enough for a folder scan, short enough that a wedged window
# reports as wedged instead of hanging the caller forever.
CALL_TIMEOUT = 120.0

MAX_LINE = 1_000_000


def enabled(settings=None) -> bool:
    if os.environ.get("AICHATLAB_BRIDGE", "").strip() in ("1", "true", "yes"):
        return True
    return bool(settings.get("control_bridge", False)) if settings else False


class Bridge:
    """Serves commands to one app instance until it is stopped."""

    def __init__(self, app, handshake_path: Path | None = None,
                 port: int = 0) -> None:
        self.app = app
        self.handshake_path = handshake_path or HANDSHAKE_PATH
        self.token = secrets.token_hex(16)
        self.port = port
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> int:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((HOST, self.port))
        self._server.listen(4)
        self.port = self._server.getsockname()[1]
        self._write_handshake()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self.port

    def _write_handshake(self) -> None:
        """Where to connect and what to say, readable by this user."""
        try:
            self.handshake_path.write_text(json.dumps(
                {"host": HOST, "port": self.port, "token": self.token,
                 "pid": os.getpid()}), encoding="utf-8")
            os.chmod(self.handshake_path, 0o600)
        except OSError:
            pass

    def handshake_is_mine(self) -> bool:
        """Does the file on disk still point at this window?

        One file, several windows. Open a second one and it takes the file
        over; kill that second one before it can tidy up and the file is left
        naming a port nothing is listening on, while a perfectly good window
        sits behind a port nobody can find. The symptom is a connection
        refused that looks like the app is shut when it is not.
        """
        try:
            found = json.loads(
                self.handshake_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return found.get("port") == self.port and found.get("pid") == os.getpid()

    def reassert(self) -> bool:
        """Take the handshake back if it has gone or gone stale.

        Cheap enough to run on a timer, which is what makes the stale case
        heal itself rather than needing the user to know it happened.
        """
        if self._stop.is_set() or self.handshake_is_mine():
            return False
        self._write_handshake()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        try:
            self.handshake_path.unlink()
        except OSError:
            pass

    # -- serving -----------------------------------------------------------
    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                client, _addr = self._server.accept()
            except OSError:
                return                      # closed, or shutting down
            threading.Thread(target=self._handle, args=(client,),
                             daemon=True).start()

    def _handle(self, client: socket.socket) -> None:
        with client:
            try:
                data = self._read_line(client)
                if data is None:
                    return
                request = json.loads(data)
                if not secrets.compare_digest(
                        str(request.get("token", "")), self.token):
                    self._send(client, {"ok": False, "error": "bad token"})
                    return
                reply = self.call(str(request.get("command", "")),
                                  request.get("args") or {})
            except (ValueError, OSError) as exc:
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            try:
                self._send(client, reply)
            except OSError:
                pass

    @staticmethod
    def _read_line(client: socket.socket) -> str | None:
        chunks: list[bytes] = []
        size = 0
        while b"\n" not in b"".join(chunks[-1:]) and size < MAX_LINE:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        raw = b"".join(chunks).strip()
        return raw.decode("utf-8", "replace") if raw else None

    @staticmethod
    def _send(client: socket.socket, payload: dict) -> None:
        client.sendall(
            (json.dumps(payload, default=str) + "\n").encode("utf-8"))

    # -- running a command on the UI thread --------------------------------
    def call(self, command: str, args: dict) -> dict:
        """Hop onto the Tk thread, run the command, bring the answer back.

        Everything the app owns — widgets, the session, the project — belongs
        to that thread, and touching it from here is the kind of bug that
        shows up once a week and never in a test.
        """
        handler = COMMANDS.get(command)
        if handler is None:
            return {"ok": False, "error": f"unknown command: {command}",
                    "commands": sorted(COMMANDS)}
        answer: queue.Queue = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                answer.put({"ok": True, "result": handler(self.app, args)})
            except Exception as exc:                       # noqa: BLE001
                answer.put({"ok": False,
                            "error": f"{type(exc).__name__}: {exc}"})

        try:
            self.app.root.after(0, run)
        except Exception as exc:                           # noqa: BLE001
            return {"ok": False, "error": f"window is gone: {exc}"}
        try:
            return answer.get(timeout=CALL_TIMEOUT)
        except queue.Empty:
            return {"ok": False, "error": "the window did not answer in time "
                                          "— it may be busy or blocked on a "
                                          "dialog"}


# -- the commands ----------------------------------------------------------
def _state(app, _args) -> dict:
    from . import edits as edit_tools

    return {
        "project": str(app.project) if app.project else "",
        "project_files": len(app.project_files),
        "files_held": len(app.project_texts),
        "attachments": [a["name"] for a in app.attachments],
        "selected_models": [f"{t.server}:{t.model}"
                            for t in app.selected_targets()],
        "available_models": [f"{server}:{model}"
                             for (server, model) in app.model_vars],
        "busy": bool(app.worker and app.worker.is_alive()),
        "mode": app.current_mode(),
        "allow_edits": bool(app.settings.get("allow_edits", True)),
        "review_popup": bool(app.settings.get("review_popup", True)),
        "auto_apply_edits": bool(app.auto_apply_edits),
        "pending_edits": {key: [e.name for e in value]
                          for key, value in app.pending_edits.items()},
        "pending_changes": sorted(app.pending_changes),
        "edit_checks": len(app.edit_debug),
        "last_verdict": (app.edit_debug.last().verdict
                         if app.edit_debug.last() else ""),
        "instructions_in_prompt": bool(
            app.project is not None
            and app.settings.get("allow_edits", True)),
        "bridge_note": "the review window still gates every write",
    }


def _attach_impl(app, args) -> dict:
    from .folderscan import Limits, survey

    path = Path(str(args.get("path", ""))).expanduser()
    if not path.is_dir():
        return {"attached": False, "reason": f"not a folder: {path}"}
    found = survey(path)
    if not found.entries:
        return {"attached": False, "reason": "no readable files"}
    limits = Limits.everything() if args.get("everything") else Limits()
    app._folder_accepted(path, found, limits, app.settings.effective_budget())
    return {"attached": True, "path": str(path),
            "files_found": len(found.entries),
            "note": "reading runs on a worker; poll `state` for files_held"}


def _select_models(app, args) -> dict:
    wanted = [str(name) for name in (args.get("models") or [])]
    chosen = []
    for (server, model), var in app.model_vars.items():
        on = any(name in (model, f"{server}:{model}") for name in wanted)
        var.set(on)
        if on:
            chosen.append(f"{server}:{model}")
    return {"selected": chosen}


def _send(app, args) -> dict:
    """Type a message and press Send, and say plainly if nothing happened.

    `send` has several gates that hold a message back rather than sending it —
    an offer to carry another model's history, a question about searching the
    web, a mode that needs two models. Each is reasonable on its own and each
    leaves the box full and the transcript unchanged, which from a script (or
    from across the room) is indistinguishable from a dead button. So the
    result says which one, rather than reporting a send that did not happen.
    """
    from .orchestrator import parse_relay

    text = str(args.get("text", ""))
    if app.worker and app.worker.is_alive():
        return {"sent": False, "held_by": "busy",
                "reason": "a request is already running"}
    targets = app.selected_targets()
    if not targets:
        return {"sent": False, "held_by": "no-models",
                "reason": "no models are ticked; send would open a modal"}

    if args.get("force"):
        # Answer the two gates that ask a question before sending: carrying
        # another model's history, and whether to search the web. A script
        # driving this has already decided.
        app.skip_carry_prompt = True
        app.skip_search_prompt = True

    # Clear the greyed-out placeholder *first*: it wipes the box when it goes,
    # so hiding it after typing throws the message away and `send` then sees
    # an empty input and returns without a word.
    app._hide_placeholder()
    app.input.delete("1.0", "end")
    app.input.insert("1.0", text)
    app.send()
    busy = bool(app.worker and app.worker.is_alive())
    # "The transcript grew" is not the same as "it went". Both gates below
    # *add* to the transcript while holding the message back, so growth alone
    # reports a send that never happened. The message itself appearing is the
    # only honest signal.
    went = busy or (text[:60] in app.chat.transcript() if text else False)
    if went:
        return {"sent": True, "busy": busy}

    # Nothing moved. Work out which gate closed, by asking them the same
    # question `send` did — none of these has a side effect on its own.
    mode = app.current_mode()
    relay = parse_relay(text, app.available_targets()) if text else None
    held = "unknown"
    hint = "retry with force=true to answer the gate and send anyway"
    if any(key.startswith("carry") for key in app.chat._actions):
        held = "carry-history-offer"
        hint = ("this model has none of the conversation on screen; "
                "force=true sends without carrying it over")
    elif any(key.startswith("search") for key in app.chat._actions):
        held = "search-offer"
    elif app.pending_changes or app.pending_edits:
        held = "waiting-on-a-decision"
    elif relay is None and mode not in ("chat", "research", "plan") and len(
            targets) < 2:
        held = "mode-needs-two-models"
    elif app.chat._actions:
        held = "an-unanswered-action-row"
    return {"sent": False, "held_by": held, "hint": hint,
            "reason": "send returned without sending; the message is still "
                      "in the input box",
            "open_actions": sorted(app.chat._actions),
            "mode": mode}


def _transcript(app, args) -> dict:
    text = app.chat.transcript()
    tail = int(args.get("tail", 0) or 0)
    if tail:
        text = "\n".join(text.splitlines()[-tail:])
    return {"transcript": text}


def _traces(app, args) -> dict:
    count = int(args.get("count", 5) or 5)
    return {"traces": [t.to_json() for t in app.edit_debug.traces[-count:]],
            "summary": app.edit_debug.summary()}


def _pending(app, _args) -> dict:
    out = {}
    for action_id, proposed in app.pending_edits.items():
        flaws = app.pending_flaws.get(action_id) or {}
        out[action_id] = [
            {"file": edit.name,
             "breaks": [problem.describe()
                        for problem in flaws.get(edit.name, [])]}
            for edit in proposed]
    return {"pending": out, "changes": sorted(app.pending_changes)}


def _diff(app, args) -> dict:
    """The diffs waiting for approval, so a script can read them before a
    human is asked to."""
    from . import edits as edit_tools

    action_id = str(args.get("action_id", "")) or next(
        iter(app.pending_edits), "")
    proposed = app.pending_edits.get(action_id) or []
    return {"action_id": action_id,
            "diffs": {edit.name: edit_tools.diff_for(app.project, edit)
                      for edit in proposed}}


def _settings(app, args) -> dict:
    changes = args.get("set") or {}
    for key, value in changes.items():
        app.settings[str(key)] = value
    if changes:
        app.settings.save()
    return {"changed": sorted(changes),
            "values": {key: app.settings.get(key)
                       for key in ("allow_edits", "review_popup",
                                   "context_budget_tokens",
                                   "max_context_window", "speed_quality")}}


def _pick(app, args) -> dict:
    """Pick a numbered candidate from an offered menu."""
    index = int(args.get("index", 1)) - 1
    if not app.menu_left:
        return {"picked": False, "reason": "no menu is open"}
    if not (0 <= index < len(app.menu_left)):
        return {"picked": False, "reason": f"index out of range 1..{len(app.menu_left)}"}
    chosen = app.menu_left[index]
    app._pick_suggestion(app.menu_action_id, index)
    return {"picked": True, "file": chosen.file,
            "description": chosen.description}


def _menu(app, _args) -> dict:
    return {"menu": [{"n": n, "file": c.file, "description": c.description}
                     for n, c in enumerate(app.menu_left, 1)]}


def _apply(app, args) -> dict:
    """Apply a pending set of edits — the review window's Apply button.

    Added when the user asked for driven end-to-end runs with real applies.
    It spends the same pending action the window would and goes through
    `_apply_edits`, so the backup, the undo offer and the transcript record
    are identical to a click.
    """
    action_id = str(args.get("action_id", "")) or next(
        iter(app.pending_edits), "")
    proposed = app.pending_edits.pop(action_id, None)
    if not proposed:
        return {"applied": False, "reason": "nothing is pending"}
    app.pending_flaws.pop(action_id, None)
    app.chat.resolve_action(action_id, "✏ Applied via the bridge.")
    app._apply_edits(proposed, always=False)
    from . import edits as edit_tools
    stamps = edit_tools.snapshots(app.project)
    app._advance_change_queue()
    return {"applied": True, "files": [e.name for e in proposed],
            "undo_stamp": next(iter(stamps), "")}


def _undo(app, args) -> dict:
    """Put a snapshot back — the transcript's Undo button."""
    from . import edits as edit_tools

    stamp = str(args.get("stamp", "")) or next(
        iter(edit_tools.snapshots(app.project)), "")
    if not stamp:
        return {"undone": False, "reason": "no snapshot to restore"}
    app._undo_edits(stamp, action_id="bridge-undo")
    return {"undone": True, "stamp": stamp}


def _realise(app, args) -> dict:
    """Press "Make these changes" — go and fetch the real diffs."""
    action_id = str(args.get("action_id", ""))
    if not action_id:
        action_id = next(iter(app.pending_changes), "")
    if action_id not in app.pending_changes:
        return {"started": False, "reason": "no such pending change",
                "available": sorted(app.pending_changes)}
    app._realise_changes(action_id)
    return {"started": True, "action_id": action_id}


def _new_chat(app, _args) -> dict:
    app.new_chat()
    return {"started": True}


def _ping(app, _args) -> dict:
    return {"alive": True, "version": getattr(app, "version", "")}


COMMANDS = {
    "ping": _ping,
    "state": _state,
    "attach": _attach_impl,
    "select_models": _select_models,
    "send": _send,
    "transcript": _transcript,
    "traces": _traces,
    "pending": _pending,
    "diff": _diff,
    "settings": _settings,
    "pick": _pick,
    "apply": _apply,
    "undo": _undo,
    "menu": _menu,
    "realise": _realise,
    "new_chat": _new_chat,
}
