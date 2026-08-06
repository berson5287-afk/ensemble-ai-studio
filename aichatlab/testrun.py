"""Run the project's own tests after an edit is applied, and mean it.

The validator answers "does this parse"; the review window answers "is this
the change I wanted".  Neither answers the only question that finally counts —
does the project still work — and the one live edit that reached disk broken
sailed through both, because plausible and correct look identical on a diff.

So an applied edit is now followed by the project's own test suite, run in a
subprocess.  Green means "kept, and here is the proof".  Red means the edit
is rolled back to the snapshot the apply just made, with the failure shown —
the strongest guarantee this app can make, because it comes from running the
code rather than reading it.

Two boundaries keep this honest.  *Only test entry points the project itself
declares* — a pytest layout or an obvious self-test script — are run; the app
never invents a way to execute someone's code.  And *absence is reported, not
papered over*: a project with no discoverable tests gets "no tests found to
run", never a green light it did not earn.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# A suite that cannot finish inside this is a suite the user should be
# running themselves, with a terminal to watch it in.
DEFAULT_TIMEOUT_S = 180

# How much failure output to carry back.  The last lines are where pytest
# and every hand-rolled harness put the verdict.
TAIL_CHARS = 2000

# Self-test scripts, in the order a project is likely to mean them.
SELFTEST_NAMES = ("selftest_harness.py", "selftest.py", "self_test.py",
                  "run_tests.py", "runtests.py", "test_all.py")


@dataclass
class Runner:
    """One way to run this project's tests."""

    kind: str                   # "pytest" | "script"
    command: list = field(default_factory=list)
    label: str = ""


@dataclass
class Outcome:
    """What happened when the tests ran."""

    ran: bool
    ok: bool = False
    label: str = ""
    seconds: float = 0.0
    tail: str = ""
    reason: str = ""            # why nothing ran, when ran is False

    def summary(self) -> str:
        if not self.ran:
            return f"no tests were run — {self.reason}"
        state = "passed" if self.ok else "FAILED"
        return f"{self.label} {state} in {self.seconds:.0f}s"


def find_runner(project) -> Runner | None:
    """The project's own way of testing itself, or None.

    pytest first, because a `tests/` directory or a `conftest.py` is an
    unambiguous declaration.  A self-test script second, matched by the
    handful of names people actually use.  Nothing else: guessing at entry
    points is how a "test runner" becomes "arbitrary code execution with
    extra steps".
    """
    root = Path(project)
    has_pytest_layout = ((root / "tests").is_dir()
                         or (root / "conftest.py").is_file())
    if has_pytest_layout:
        return Runner(kind="pytest",
                      command=[sys.executable, "-m", "pytest", "-q",
                               "--maxfail=5"],
                      label="pytest")
    for name in SELFTEST_NAMES:
        if (root / name).is_file():
            return Runner(kind="script",
                          command=[sys.executable, name],
                          label=name)
    return None


def run(project, runner: Runner | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S, cancel=None) -> Outcome:
    """Run the tests and report plainly, never raising.

    A test runner that can crash the apply path would make edits *less*
    safe than having no runner at all, so every failure here — no runner,
    a dead interpreter, a hung suite — comes back as an Outcome that says
    exactly what happened.
    """
    runner = runner or find_runner(project)
    if runner is None:
        return Outcome(ran=False,
                       reason="no tests found to run (no tests/ directory, "
                              "conftest.py, or self-test script)")
    started = time.time()
    try:
        done = subprocess.run(
            runner.command, cwd=str(project), capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=max(10, int(timeout_s)))
    except subprocess.TimeoutExpired:
        return Outcome(ran=True, ok=False, label=runner.label,
                       seconds=time.time() - started,
                       tail=f"(killed after {timeout_s}s — the suite did "
                            f"not finish)")
    except (OSError, ValueError) as exc:
        return Outcome(ran=False, reason=f"could not start {runner.label}: "
                                         f"{exc}")
    if cancel is not None and getattr(cancel, "is_set", lambda: False)():
        return Outcome(ran=False, reason="stopped before the result was read")
    output = ((done.stdout or "") + "\n" + (done.stderr or "")).strip()
    return Outcome(ran=True, ok=(done.returncode == 0), label=runner.label,
                   seconds=time.time() - started,
                   tail=output[-TAIL_CHARS:])
