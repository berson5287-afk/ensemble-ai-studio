"""Import the edited file before it is written, and be honest at the edges.

Real subprocesses on real temp projects — a mocked import checker checks
the mock.
"""

from __future__ import annotations

from aichatlab import sandbox


def test_a_clean_module_imports_green(tmp_path):
    verdict = sandbox.check_import(tmp_path, "engine.py",
                                   "X = 1\n\ndef f():\n    return X\n")

    assert verdict.checked and verdict.ok


def test_a_module_level_name_error_is_caught(tmp_path):
    """The case every other layer misses: fatal, and outside any function."""
    verdict = sandbox.check_import(
        tmp_path, "engine.py", "LIMIT = compute_limit()\n")

    assert verdict.checked and not verdict.ok
    assert "compute_limit" in verdict.detail


def test_a_missing_import_is_caught(tmp_path):
    verdict = sandbox.check_import(
        tmp_path, "engine.py", "import no_such_module_xyz\n")

    assert verdict.checked and not verdict.ok
    assert "no_such_module_xyz" in verdict.detail


def test_the_edited_text_shadows_the_real_file(tmp_path):
    """The whole mechanism: the project's file is broken, the edit fixes it,
    and the import must see the edit, not the disk."""
    (tmp_path / "engine.py").write_text("raise RuntimeError('old broken')\n",
                                        encoding="utf-8")

    verdict = sandbox.check_import(tmp_path, "engine.py", "X = 1\n")

    assert verdict.checked and verdict.ok, "the edit shadows the original"


def test_siblings_resolve_from_the_real_project(tmp_path):
    (tmp_path / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")

    verdict = sandbox.check_import(
        tmp_path, "engine.py", "from helper import VALUE\nassert VALUE == 7\n")

    assert verdict.checked and verdict.ok


def test_a_hung_import_is_inconclusive_never_red(tmp_path):
    """A GUI script starts its event loop on import; that is not a broken
    edit, and calling it one would teach people to ignore the red."""
    verdict = sandbox.check_import(
        tmp_path, "app.pyw", "import time\ntime.sleep(60)\n", timeout_s=5)

    assert not verdict.checked
    assert "inconclusive" in verdict.detail
    assert sandbox.problem_from(verdict, "app.pyw") is None


def test_a_non_module_name_is_skipped(tmp_path):
    verdict = sandbox.check_import(tmp_path, "notes.md", "# hi\n")

    assert not verdict.checked
    assert sandbox.problem_from(verdict, "notes.md") is None


def test_a_failure_becomes_a_validator_shaped_problem(tmp_path):
    verdict = sandbox.check_import(
        tmp_path, "engine.py", "LIMIT = compute_limit()\n")

    problem = sandbox.problem_from(verdict, "engine.py")

    assert problem is not None
    assert problem.kind == "import"
    assert "compute_limit" in problem.message
    assert problem.key == ("import", "import:engine.py")


def test_a_green_import_yields_no_problem(tmp_path):
    verdict = sandbox.check_import(tmp_path, "engine.py", "X = 1\n")

    assert sandbox.problem_from(verdict, "engine.py") is None
