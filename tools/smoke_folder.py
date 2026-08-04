"""Manual smoke test: drive the new folder + search-prompt UI and photograph it.

Run under a display (xvfb-run works) with a mock Ollama on 11434:

    python tools/mock_ollama.py 11434 &
    xvfb-run -a python tools/smoke_folder.py /path/to/shots
"""

from __future__ import annotations

import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aichatlab import config as config_module  # noqa: E402

# never touch the real settings file — this script deliberately mangles the
# context budget, and a smoke test should not leave that behind
config_module.SETTINGS_PATH = Path("/tmp/smoke_settings.json")
if config_module.SETTINGS_PATH.exists():
    config_module.SETTINGS_PATH.unlink()

from aichatlab.folderscan import select, survey  # noqa: E402
from aichatlab.ui.app import ChatLabApp  # noqa: E402
from aichatlab.ui.folder_dialog import FolderScanDialog  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/shots")
OUT.mkdir(parents=True, exist_ok=True)


def shot(name: str) -> None:
    time.sleep(0.4)
    subprocess.run(["import", "-window", "root", str(OUT / f"{name}.png")],
                   check=False)
    print("captured", name)


def main() -> None:
    root = tk.Tk()
    app = ChatLabApp(root)
    root.update()

    # wait for the mock server's models to arrive, then tick one — otherwise
    # send() stops at the modal "no models selected" warning
    for _ in range(100):
        root.update()
        if app.model_vars:
            break
        time.sleep(0.1)
    if not app.model_vars:
        raise SystemExit("no models — start tools/mock_ollama.py 11434 first")
    first = next(iter(app.model_vars.values()))
    first.set(True)
    root.update()
    print("models:", len(app.model_vars), flush=True)

    # 1. the inline "did you mean to search?" prompt
    app.research_var.set(False)
    app._hide_placeholder()
    app.input.insert("1.0", "look up the latest firmware for the RX 7900")
    app.send()
    root.update()
    shot("01_search_prompt")

    pending = next(iter(app.chat._actions))
    app.chat.resolve_action(pending, "💬 Answering from memory — no search was made.")
    root.update()
    shot("02_search_prompt_resolved")

    # 2. the folder dialog, pointed at this very repository
    repo = Path(__file__).resolve().parent.parent
    result = survey(repo)
    print(f"survey: {result.describe()}  skipped={sorted(set(result.skipped_dirs))}")

    accepted = {}
    dialog = FolderScanDialog(
        root, result, app.settings.effective_budget(),
        on_accept=lambda limits, budget: accepted.update(
            limits=limits, budget=budget),
        max_window=int(app.settings.get("max_context_window", 32768)))
    root.update()
    shot("03_folder_dialog_limited")
    print("limited:", dialog.selection.summary(), "|", dialog.plan.message())

    dialog.access.set("all")
    dialog._recompute()
    root.update()
    shot("04_folder_dialog_everything")
    print("everything:", dialog.selection.summary(), "|", dialog.plan.message())

    dialog.access.set("limited")
    dialog.kind_vars["docs"].set(False)
    dialog.max_depth.set(2)
    dialog._recompute()
    root.update()
    shot("05_folder_dialog_tuned")
    print("tuned:", dialog.selection.summary(), "|", dialog.plan.message())

    dialog.access.set("all")
    dialog._recompute()
    dialog._accept()
    root.update()
    print("accepted:", accepted["budget"], "tokens budget")

    # 3. read it for real and attach
    app._folder_accepted(repo, result, accepted["limits"], accepted["budget"])
    for _ in range(200):
        root.update()
        if any(a["name"].endswith("/") for a in app.attachments):
            break
        time.sleep(0.05)
    root.update()
    shot("06_folder_attached")

    attachment = app.attachments[-1]
    print("attached:", attachment["name"], len(attachment["content"]), "chars")
    print("budget now:", app.settings["context_budget_tokens"],
          "effective:", app.settings.effective_budget(),
          "num_ctx:", app.settings.sampling_options()["num_ctx"])
    assert app.settings.effective_budget() < app.settings.context_window()
    composed = app._compose("what does this project do?")
    assert composed.startswith("[Folder:")     # not re-fenced
    print("composed prompt:", len(composed), "chars")

    selection = select(result, accepted["limits"])
    print("files:", len(selection.chosen), "estimated tokens:", selection.tokens)

    root.destroy()


if __name__ == "__main__":
    main()
