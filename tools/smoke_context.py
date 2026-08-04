"""Manual smoke test: cut-off detection, compaction and search triage.

    python tools/mock_ollama.py 11434 &
    xvfb-run -a python tools/smoke_context.py /tmp/shots
"""

from __future__ import annotations

import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aichatlab import config as config_module  # noqa: E402

config_module.SETTINGS_PATH = Path("/tmp/smoke_ctx_settings.json")
if config_module.SETTINGS_PATH.exists():
    config_module.SETTINGS_PATH.unlink()

from aichatlab.compaction import is_summary, usage  # noqa: E402
from aichatlab.ui.app import ChatLabApp  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/shots")
OUT.mkdir(parents=True, exist_ok=True)


def shot(name: str) -> None:
    time.sleep(0.4)
    try:
        subprocess.run(["import", "-window", "root", str(OUT / f"{name}.png")],
                       check=False, timeout=20)
    except subprocess.TimeoutExpired:
        return
    print("captured", name, flush=True)


def settle(root, app, seconds=25):
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        if not (app.worker and app.worker.is_alive()):
            root.update()
            return True
        time.sleep(0.05)
    return False


def say(app, root, text):
    app._hide_placeholder()
    app.input.delete("1.0", tk.END)
    app.input.insert("1.0", text)
    app.send()
    settle(root, app)


def main() -> None:
    root = tk.Tk()
    app = ChatLabApp(root)
    root.update()
    for _ in range(100):
        root.update()
        if app.model_vars:
            break
        time.sleep(0.1)
    if not app.model_vars:
        raise SystemExit("start tools/mock_ollama.py 11434 first")

    target = next(iter(app.model_vars))
    app.model_vars[target].set(True)
    key = f"{target[0]}::{target[1]}"

    # ---------------------------------------------- 1. a reply cut off short
    app.settings["speed_quality"] = 10            # Fastest — a 192-token cap
    app.speed_scale.set(10)
    root.update()
    print("profile:", app.settings.profile(), flush=True)

    for attempt in range(8):
        say(app, root, f"give me some tips, round {attempt}")
        if any(a.startswith("cut") for a in app.chat._actions):
            print("truncated on attempt", attempt, flush=True)
            break
    else:
        print("!! never got a truncated reply", flush=True)
    shot("07_cut_off")

    action = next((a for a in app.chat._actions if a.startswith("cut")), None)
    if action:
        before = len(app.session.history(key))
        app._continue_reply(app.selected_targets()[0], action)
        settle(root, app)
        print("continue added", len(app.session.history(key)) - before,
              "messages", flush=True)
        shot("08_continued")

    # ------------------------------------------------------ 2. compaction
    app.settings["speed_quality"] = 50
    for index in range(8):
        app.session.history(key).append(
            {"role": "user" if index % 2 == 0 else "assistant",
             "content": f"turn {index}: " + "filler text " * 300})
    before_tokens = usage(app.session.history(key))
    before_count = len(app.session.history(key))
    app._update_context_label()
    root.update()
    print("context label:", app.context_label["text"], flush=True)
    shot("09_context_filling")

    app.compact_chat()
    settle(root, app, seconds=40)
    history = app.session.history(key)
    print(f"compacted: {before_count} → {len(history)} messages, "
          f"{before_tokens:,} → {usage(history):,} tokens", flush=True)
    print("first is a digest:", is_summary(history[0]), flush=True)
    shot("10_compacted")

    assert usage(history) < before_tokens, "compaction saved nothing"
    assert is_summary(history[0]), "no digest at the front"
    assert len(history) < before_count

    # ------------------------------------------------------- 3. triage
    app.settings["searxng_url"] = "http://127.0.0.1:9/nowhere"
    app.research_var.set(True)
    app.attachments = [{"name": "my-project/", "content": "x" * 8000}]
    app._rebuild_chips()
    root.update()
    say(app, root, "take a look at my app and give me some tips")
    for _ in range(40):                     # let the event pump drain
        root.update()
        time.sleep(0.05)
    shot("11_triage")
    print("--- transcript tail ---", flush=True)
    print("\n".join(app.chat.transcript().splitlines()[-14:]), flush=True)

    # The triage note updates in place (one note_id), so only the verdict
    # survives — "Reading what you've given me…" is replaced, not appended.
    transcript = app.chat.transcript()
    searched = "Searching the web" in transcript
    vetoed = "no search needed" in transcript
    print("searched the web:", searched, flush=True)
    print("triage vetoed the search:", vetoed, flush=True)
    assert vetoed and not searched, "triage did not veto a question about files"

    root.destroy()


if __name__ == "__main__":
    main()
