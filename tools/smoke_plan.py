"""Manual smoke test: the checklist, plan mode, and the long-prompt warning.

    python /tmp/plan_ollama.py 11502 &
    xvfb-run -a python tools/smoke_plan.py /tmp/shots
"""

from __future__ import annotations

import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aichatlab import config as config_module  # noqa: E402

config_module.SETTINGS_PATH = Path("/tmp/smoke_plan_settings.json")
if config_module.SETTINGS_PATH.exists():
    config_module.SETTINGS_PATH.unlink()

from aichatlab.orchestrator import MODES  # noqa: E402
from aichatlab.ui.app import ChatLabApp  # noqa: E402

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
    app.settings["host_port"] = "11502"
    app.settings["local_ip"] = ""
    app.settings.save()
    app.refresh_models()
    for _ in range(120):
        root.update()
        if app.model_vars:
            break
        time.sleep(0.1)
    if not app.model_vars:
        raise SystemExit("start /tmp/plan_ollama.py 11502 first")
    next(iter(app.model_vars.values())).set(True)
    app.settings["speed_quality"] = 100
    root.update()

    question = ("my debugging App I have fully backed up. I would like you to "
                "take a look and make improvements where you see fit and "
                "please give me a complete list of the upgrades you gave it.")

    # a folder attachment, web research on — the exact shape of the report
    app.attachments = [{"name": "udbg-phase1/", "content": "x" * 95_000}]
    app._rebuild_chips()
    app.research_var.set(True)
    app.learning_var.set(True)
    app.settings["searxng_url"] = "http://127.0.0.1:9/nowhere"

    app.mode.set(MODES["plan"])
    app._sync_mode_bar()
    root.update()

    app._hide_placeholder()
    app.input.insert("1.0", question)
    app.send()
    spin(root, 3)
    print("status:", app.status.get(), flush=True)
    shot("17_plan_checklist")

    settle(root, app)
    spin(root, 1)

    transcript = app.chat.transcript()
    print("--- notes ---", flush=True)
    for line in transcript.splitlines():
        line = line.strip()
        if line.startswith(("⏳", "📄", "🔍", "🗂", "☑", "☐", "▸", "⊘", "✗",
                            "The plan", "This request")):
            print("   ", line, flush=True)
    shot("18_plan_done")

    print("\nsearched the web:", "Searching the web" in transcript, flush=True)
    print("warned about prompt size:", "before the first word" in transcript,
          flush=True)
    print("run log:", app.runlog.summary(), flush=True)
    print("prompt rate measured:", app.runlog.prompt_rate(), "tok/s", flush=True)

    assert "Searching the web" not in transcript, "searched despite attachments"
    assert "The plan" in transcript, "no plan checklist appeared"
    assert "☑" in transcript, "nothing was ever ticked off"
    assert app.runlog.prompt_rate(), "prompt speed was never measured"

    if app.pump_id:
        root.after_cancel(app.pump_id)
    if app.tick_id:
        root.after_cancel(app.tick_id)
    root.destroy()


if __name__ == "__main__":
    main()
