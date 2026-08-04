"""Manual smoke test: switching to a model that has not seen the conversation.

Reproduces the real failure — a folder is attached and discussed with one
model, a second model is ticked, and the second one answers "I'll need some
specifics about your app" because its history is empty while the screen is
full of the first model's conversation.

    python tools/mock_ollama.py 11503 &
    xvfb-run -a python tools/smoke_carry.py /tmp/shots
"""

from __future__ import annotations

import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aichatlab import config as config_module  # noqa: E402

config_module.SETTINGS_PATH = Path("/tmp/smoke_carry_settings.json")
if config_module.SETTINGS_PATH.exists():
    config_module.SETTINGS_PATH.unlink()

from aichatlab import recovery as recovery_module  # noqa: E402

# Without this the app restores whatever unfinished chat an earlier smoke run
# left in the temp directory, and the screenshot is of someone else's session.
recovery_module.RECOVERY_PATH = Path("/tmp/smoke_carry_unfinished.json")
if recovery_module.RECOVERY_PATH.exists():
    recovery_module.RECOVERY_PATH.unlink()

from aichatlab.session import make_key  # noqa: E402
from aichatlab.ui.app import ChatLabApp  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/shots")
OUT.mkdir(parents=True, exist_ok=True)

FOLDER = ("[Folder: udbg-phase1 — 41 files, 24,000 tokens]\n"
          "manifest…\n" + "x" * 60_000 + "\n[End of folder contents.]")


def shot(name: str) -> None:
    subprocess.run(["import", "-window", "root", str(OUT / f"{name}.png")],
                   check=False, timeout=20)
    print("captured", name, flush=True)


def spin(root, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.05)


def buttons(app) -> list[str]:
    labels = []
    for name in app.chat.text.window_names():
        widget = app.chat.text.nametowidget(name)
        for child in widget.winfo_children():
            labels.append(child.cget("text"))
    return labels


def main() -> None:
    root = tk.Tk()
    app = ChatLabApp(root)
    app.settings["host_ip"] = "127.0.0.1"
    app.settings["host_port"] = "11503"
    app.settings["local_ip"] = ""
    app.settings.save()
    app.refresh_models()
    for _ in range(120):
        root.update()
        if app.model_vars:
            break
        time.sleep(0.1)
    if not app.model_vars:
        raise SystemExit("start tools/mock_ollama.py 11503 first")

    names = [model for _server, model in app.model_vars]
    if len(names) < 2:
        raise SystemExit("this smoke test needs two models on the mock server")
    first, second = names[0], names[1]
    server = next(iter(app.model_vars))[0]

    # A conversation about a folder, belonging entirely to the first model.
    donor = make_key(server, first)
    app.session.add(donor, "user", FOLDER, pinned=True)
    app.session.add(donor, "assistant",
                    "I have read all 41 files. The biggest win is the "
                    "duplicated symbol table in loader.c.")
    app._redraw()
    root.update()

    # Now tick the *second* model instead, exactly as the user did.
    for (srv, model), var in app.model_vars.items():
        var.set(srv == server and model == second)
    app._update_context_label()
    root.update()
    print("context label:", app.context_label.cget("text"), flush=True)
    shot("19_carry_before")

    app._hide_placeholder()
    app.input.insert("1.0", "so what would you change first?")
    app.send()
    spin(root, 1)

    transcript = app.chat.transcript()
    print("status:", app.status.get(), flush=True)
    print("buttons:", buttons(app), flush=True)
    shot("20_carry_prompt")

    assert "has not seen any of this" in transcript, "no offer was made"
    assert "udbg-phase1" in transcript, "did not say what it would carry"
    assert app.input.get("1.0", "end-1c").strip(), "the message was thrown away"

    # Say yes, and check the second model really did inherit the folder.
    action_id = next(iter(app.chat._actions))
    target = app.selected_targets()[0]
    app.send = lambda: None                 # do not actually re-send
    app._carry_history(donor, target, action_id)
    root.update()

    carried = app.session.history(target.key)
    print("carried messages:", len(carried), flush=True)
    print("still pinned:", bool(carried and carried[0].get("pinned")),
          flush=True)
    print("context label:", app.context_label.cget("text"), flush=True)
    shot("21_carry_done")

    assert len(carried) == 2, "the conversation did not come across"
    assert carried[0].get("pinned"), "the folder lost its pin on the way over"
    assert len(app.session.conversations[donor]) == 2, "the donor was damaged"

    if app.pump_id:
        root.after_cancel(app.pump_id)
    if app.tick_id:
        root.after_cancel(app.tick_id)
    root.destroy()
    print("\nOK", flush=True)


if __name__ == "__main__":
    main()
