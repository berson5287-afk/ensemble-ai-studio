"""Manual smoke test: a reasoning model, end to end.

The mock server sends `message.thinking` before `message.content`, exactly as
Ollama does for Qwen3, and refuses `think` on a model that cannot reason — so
this also proves the app never sends it to the wrong model.

    python tools/mock_ollama.py 11504 &
    xvfb-run -a python tools/smoke_thinking.py /tmp/shots
"""

from __future__ import annotations

import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aichatlab import config as config_module  # noqa: E402

config_module.SETTINGS_PATH = Path("/tmp/smoke_thinking_settings.json")
if config_module.SETTINGS_PATH.exists():
    config_module.SETTINGS_PATH.unlink()

from aichatlab import recovery as recovery_module  # noqa: E402

recovery_module.RECOVERY_PATH = Path("/tmp/smoke_thinking_unfinished.json")
if recovery_module.RECOVERY_PATH.exists():
    recovery_module.RECOVERY_PATH.unlink()

from aichatlab.ui.app import ChatLabApp  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/shots")
OUT.mkdir(parents=True, exist_ok=True)


def shot(name: str) -> None:
    subprocess.run(["import", "-window", "root", str(OUT / f"{name}.png")],
                   check=False, timeout=20)
    print("captured", name, flush=True)


def spin(root, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.02)


def settle(root, app, seconds=90) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        if not (app.worker and app.worker.is_alive()):
            root.update()
            return
        time.sleep(0.05)


def pick(app, wanted: str) -> None:
    for (_server, model), var in app.model_vars.items():
        var.set(wanted in model)


def main() -> None:
    root = tk.Tk()
    app = ChatLabApp(root)
    app.settings["host_ip"] = "127.0.0.1"
    app.settings["host_port"] = "11504"
    app.settings["local_ip"] = ""
    app.settings["speed_quality"] = 0        # Quick: the 192-token allowance
    app.settings.save()
    app.refresh_models()
    for _ in range(120):
        root.update()
        if app.model_vars:
            break
        time.sleep(0.1)
    if not app.model_vars:
        raise SystemExit("start tools/mock_ollama.py 11504 first")

    # ---------------------------------------------------------- reasoning on
    pick(app, "qwen3")
    app.thinking_var.set(True)
    root.update()

    app._hide_placeholder()
    app.input.insert("1.0", "which of my two cards should hold the model?")
    app.send()

    # Sample the transcript while the model reasons.  The mock streams the
    # working in well under a second, so a single check at a fixed delay is a
    # coin toss — watch for it instead.
    live, live_status, shot_taken = "", "", False
    deadline = time.time() + 60
    while time.time() < deadline:
        root.update()
        current = app.chat.transcript()
        if "💭 Thinking" in current and not shot_taken:
            live, live_status, shot_taken = current, app.status.get(), True
            shot("22_thinking_live")
        if not (app.worker and app.worker.is_alive()):
            break
        time.sleep(0.02)

    print("live status:", live_status or app.status.get(), flush=True)
    print("reasoning visible while working:", bool(live), flush=True)

    settle(root, app)
    spin(root, 0.5)
    done = app.chat.transcript()
    blocks = dict(app.chat._thoughts)
    print("blocks:", {k: v["collapsed"] for k, v in blocks.items()}, flush=True)
    print("folded after the answer:", "click to show" in done, flush=True)
    shot("23_thinking_folded")

    assert live, "the reasoning was never shown while it ran"
    assert blocks, "no reasoning block was created"
    assert all(b["collapsed"] for b in blocks.values()), "still expanded"
    assert "click to show" in done, "no way to read the reasoning back"

    # The reasoning must not have been saved as something the model said.
    target = app.selected_targets()[0]
    saved = " ".join(m["content"] for m in app.session.history(target.key))
    assert "Let me work through" not in saved, "reasoning leaked into history"
    print("reasoning kept out of the saved history: True", flush=True)

    # Expand it again, the way the user would.
    stream_id = next(iter(blocks))
    app.chat.toggle_thought(stream_id)
    root.update()
    assert "Let me work through" in app.chat.transcript(), "cannot reopen it"
    print("reopens on click: True", flush=True)
    shot("24_thinking_expanded")

    # ----------------------------------------- a model that cannot reason
    # The mock returns HTTP 400 if `think` is sent to it, so a clean reply
    # here proves the app knows the difference.
    pick(app, "llama3")
    app._hide_placeholder()
    app.input.insert("1.0", "and what about the other card?")
    app.send()
    settle(root, app)
    spin(root, 0.3)

    transcript = app.chat.transcript()
    assert "does not support thinking" not in transcript, \
        "`think` was sent to a model that cannot reason"
    print("ordinary model unaffected: True", flush=True)

    print("run log:", app.runlog.summary(), flush=True)
    if app.pump_id:
        root.after_cancel(app.pump_id)
    if app.tick_id:
        root.after_cancel(app.tick_id)
    root.destroy()
    print("\nOK", flush=True)


if __name__ == "__main__":
    main()
