"""Drive a running Ensemble AI Studio window from the command line.

Start the app with the bridge on:

    set AICHATLAB_BRIDGE=1        (Windows)   /   export AICHATLAB_BRIDGE=1
    python "Ensemble AI Studio.pyw"

then:

    python tools/bridge_cli.py state
    python tools/bridge_cli.py attach --path ../Replyit
    python tools/bridge_cli.py select_models --models qwen2.5:14b-instruct-q8_0
    python tools/bridge_cli.py send --text "Find the deficiencies and fix them."
    python tools/bridge_cli.py wait
    python tools/bridge_cli.py traces --count 1
    python tools/bridge_cli.py transcript --tail 40

The connection details are written to ~/.ai_chat_lab_bridge.json when the app
starts, so nothing has to be passed in by hand.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aichatlab.bridge import HANDSHAKE_PATH  # noqa: E402


def handshake(path=None) -> dict:
    target = Path(path) if path else HANDSHAKE_PATH
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"No bridge to talk to ({target}: {exc}).\n"
            f"Start the app with AICHATLAB_BRIDGE=1.") from exc


def call(command: str, args: dict | None = None, path=None,
         timeout: float = 180.0) -> dict:
    where = handshake(path)
    payload = {"token": where["token"], "command": command,
               "args": args or {}}
    try:
        sock = socket.create_connection((where["host"], where["port"]),
                                        timeout=timeout)
    except OSError as exc:
        # The handshake outlives a window that was closed without running
        # `stop` — a crash, or the process being killed. Saying "connection
        # refused" sends people looking for a firewall; the real answer is
        # almost always that the app is not open.
        raise SystemExit(
            f"Nothing is listening on {where['host']}:{where['port']} "
            f"({exc.__class__.__name__}).\n"
            f"The window that wrote {HANDSHAKE_PATH} (pid "
            f"{where.get('pid', '?')}) is gone. Start Ensemble AI Studio again — the "
            f"bridge comes up with it when `control_bridge` is on in "
            f"settings.") from exc
    with sock:
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        chunks = []
        while b"\n" not in b"".join(chunks[-1:]):
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return json.loads(b"".join(chunks).decode("utf-8", "replace"))


def wait_until_idle(path=None, limit: float = 600.0,
                    poll: float = 2.0) -> dict:
    """Block until the window stops working, so a script can read the reply."""
    started = time.time()
    while time.time() - started < limit:
        reply = call("state", path=path)
        if not reply.get("ok"):
            return reply
        if not reply["result"].get("busy"):
            return reply
        time.sleep(poll)
    return {"ok": False, "error": f"still busy after {limit:.0f}s"}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command")
    parser.add_argument("--path", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--tail", type=int, default=0)
    parser.add_argument("--set", default="", help="key=value,key=value")
    parser.add_argument("--everything", action="store_true")
    parser.add_argument("--handshake", default="")
    parser.add_argument("--limit", type=float, default=600.0)
    args = parser.parse_args()

    where = args.handshake or None
    if args.command == "wait":
        print(json.dumps(wait_until_idle(where, args.limit), indent=2))
        return 0

    payload: dict = {}
    if args.path:
        payload["path"] = args.path
    if args.text:
        payload["text"] = args.text
    if args.models:
        payload["models"] = [m.strip() for m in args.models.split(",")
                             if m.strip()]
    if args.count:
        payload["count"] = args.count
    if args.tail:
        payload["tail"] = args.tail
    if args.everything:
        payload["everything"] = True
    if args.set:
        changes = {}
        for pair in args.set.split(","):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            text = value.strip()
            if text.lower() in ("true", "false"):
                changes[key.strip()] = text.lower() == "true"
            elif text.lstrip("-").isdigit():
                changes[key.strip()] = int(text)
            else:
                changes[key.strip()] = text
        payload["set"] = changes

    reply = call(args.command, payload, where)
    print(json.dumps(reply, indent=2, default=str))
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
