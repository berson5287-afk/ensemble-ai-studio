"""Import the edited file in a throwaway subprocess, before it is written.

The validator reads the code; the test runner runs it after the apply.  This
is the layer between: the edited text is imported — top-level code executed,
imports resolved against the real project — in a subprocess, before the diff
is even shown.  A module-level NameError, a missing import, a circular
import: none of those are visible to a scope-checker that only reads
function bodies, and all of them crash the project the moment anything
imports the file.

The edited text never touches the project.  It is written to a temp
directory that shadows the project on sys.path, so `import engine` finds the
*edited* engine while the engine's own imports find its *real* siblings.

Deliberately narrow, and honest at the edges.  Only a clean traceback and a
nonzero exit count as failure.  A timeout is reported as *inconclusive* —
never red — because a GUI script with unguarded top-level code starts an
event loop when imported, and "your edit hangs my checker" is not the same
finding as "your edit is broken".
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

IMPORT_TIMEOUT_S = 12
TAIL_CHARS = 1200


@dataclass
class Verdict:
    """What importing the edited file did."""

    checked: bool               # a real import ran to a real conclusion
    ok: bool = False
    seconds: float = 0.0
    detail: str = ""            # traceback tail, or why it was inconclusive


def check_import(project, name: str, new_text: str,
                 timeout_s: int = IMPORT_TIMEOUT_S) -> Verdict:
    """Import the edited text as its module, shadowing the real file."""
    stem = Path(name).stem
    if not str(name).endswith((".py", ".pyw")) or not stem.isidentifier():
        return Verdict(checked=False, detail="not an importable Python module")
    started = time.time()
    try:
        with tempfile.TemporaryDirectory() as shadow:
            (Path(shadow) / f"{stem}.py").write_text(new_text or "",
                                                     encoding="utf-8")
            code = (
                "import sys\n"
                f"sys.path.insert(0, {str(project)!r})\n"
                f"sys.path.insert(0, {shadow!r})\n"
                f"import {stem}\n")
            done = subprocess.run(
                [sys.executable, "-B", "-c", code], cwd=str(project),
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=max(5, int(timeout_s)))
    except subprocess.TimeoutExpired:
        return Verdict(
            checked=False, seconds=time.time() - started,
            detail="importing did not finish — top-level code may start "
                   "the app, so this check is inconclusive, not failed")
    except (OSError, ValueError) as exc:
        return Verdict(checked=False, detail=f"could not run the check: "
                                             f"{exc}")
    output = ((done.stdout or "") + "\n" + (done.stderr or "")).strip()
    return Verdict(checked=True, ok=(done.returncode == 0),
                   seconds=time.time() - started,
                   detail=output[-TAIL_CHARS:])


def problem_from(verdict: Verdict, name: str):
    """A validator-shaped Problem for a failed import, or None.

    Shaped like validate.Problem so the review window, the flags and
    Apply-all treat an import crash exactly like a syntax error — un-ticked,
    red, and never swept up by accident.
    """
    from .validate import Problem

    if not verdict.checked or verdict.ok:
        return None
    last = ""
    for line in reversed(verdict.detail.splitlines()):
        if line.strip():
            last = line.strip()[:160]
            break
    return Problem(
        "import",
        f"importing the edited {name} crashes — {last or 'nonzero exit'}. "
        f"Top-level code runs on import, and this edit breaks it",
        symbol=f"import:{name}")
