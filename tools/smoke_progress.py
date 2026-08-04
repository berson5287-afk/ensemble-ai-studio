"""Manual smoke test: the live status line, stalls, the run log and interrupts.

    python /tmp/slow_ollama.py 11500 &
    xvfb-run -a python tools/smoke_progress.py /tmp/shots
"""

from __future__ import annotations

import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aichatlab import config as config_module  # noqa: E402

config_module.SETTINGS_PATH = Path("/tmp/smoke_progress_settings.json")
if config_module.SETTINGS_PATH.exists():
    config_module.SETTINGS_PATH.unlink()

from aichatlab.ui.app import ChatLabApp  # noqa: E402
from aichatlab.ui.runlog_window import RunLogWindow  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/shots")
OUT.mkdir(parents=True, exist_ok=True)


def shot(name: str) -> None:
    try:
        subprocess.run(["import", "-window", "root", str(OUT / f"{name}.png")],
                       check=False, timeout=20)
    except subprocess.TimeoutExpired:
        return
    print("captured", name, flush=True)


def spin(root, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.05)


def settle(root, app, seconds=120) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        if not (app.worker and app.worker.is_alive()):
            root.update()
            return
        time.sleep(0.05)


def main() -> None:
    root = tk.Tk()
    app = ChatLabApp(root)
    app.settings["host_ip"] = "127.0.0.1"
    app.settings["host_port"] = "11500"
    app.settings["local_ip"] = ""
    app.settings.save()
    app.refresh_models()
    for _ in range(120):
        root.update()
        if app.model_vars:
            break
        time.sleep(0.1)
    if not app.model_vars:
        raise SystemExit("start /tmp/slow_ollama.py 11500 first")
    next(iter(app.model_vars.values())).set(True)
    app.settings["speed_quality"] = 10          # Fastest — a 192-token cap
    app.speed_scale.set(10)
    root.update()

    # 1. the live status line while a slow model works
    app._hide_placeholder()
    app.input.insert("1.0", "please take a look at my app and give me tips")
    app.send()
    spin(root, 6)
    print("status @6s: ", app.status.get(), flush=True)
    shot("13_live_progress")
    spin(root, 6)
    print("status @12s:", app.status.get(), flush=True)

    # 2. typing while it is still working
    app._hide_placeholder()
    app.input.insert("1.0", "what does the cli module do?")
    app.send()
    root.update()
    print("queued:", repr(app.queued_message), flush=True)
    shot("14_typed_while_busy")
    action = next((a for a in app.chat._actions if a.startswith("queue")), None)
    app._queue_after(action)
    root.update()

    settle(root, app)
    spin(root, 2)
    print("after it finished, queued is:", repr(app.queued_message), flush=True)
    settle(root, app)
    spin(root, 1)

    # 3. the cut-off treadmill, second time round
    print("cut-off streak:", app.cutoff_streak, flush=True)
    shot("15_repeat_cutoff")

    # 4. the run log
    print("run log:", app.runlog.summary(), flush=True)
    for entry in app.runlog.entries:
        print("   ", entry.row(), flush=True)
    window = RunLogWindow(root, app.runlog)
    root.update()
    spin(root, 1)
    shot("16_run_log")
    window.destroy()

    assert len(app.runlog) >= 2, "nothing was logged"
    assert app.runlog.counts().get("length"), "no cut-off was recorded"

    if app.pump_id:
        root.after_cancel(app.pump_id)
    if app.tick_id:
        root.after_cancel(app.tick_id)
    root.destroy()


if __name__ == "__main__":
    main()
