"""Run the project's own tests after an apply, and mean it.

The one edit that reached a user's disk broken had passed every static gate;
running the code is the only judge a plausible diff cannot fool.  These tests
use real subprocesses on tiny throwaway projects, because a mocked test
runner tests the mock.
"""

from __future__ import annotations

from aichatlab import testrun


# -- finding the project's own way of testing itself ------------------------
def test_a_pytest_layout_is_detected(tmp_path):
    (tmp_path / "tests").mkdir()

    runner = testrun.find_runner(tmp_path)

    assert runner is not None and runner.kind == "pytest"
    assert "-m" in runner.command and "pytest" in runner.command


def test_a_conftest_alone_counts_as_pytest(tmp_path):
    (tmp_path / "conftest.py").write_text("", encoding="utf-8")

    assert testrun.find_runner(tmp_path).kind == "pytest"


def test_a_selftest_script_is_detected(tmp_path):
    (tmp_path / "selftest_harness.py").write_text("print('ok')",
                                                  encoding="utf-8")

    runner = testrun.find_runner(tmp_path)

    assert runner.kind == "script"
    assert runner.command[-1] == "selftest_harness.py"


def test_pytest_layout_wins_over_a_script(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "selftest.py").write_text("", encoding="utf-8")

    assert testrun.find_runner(tmp_path).kind == "pytest"


def test_no_entry_point_means_none(tmp_path):
    (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")

    assert testrun.find_runner(tmp_path) is None


# -- running, for real -------------------------------------------------------
def test_a_passing_script_comes_back_green(tmp_path):
    (tmp_path / "selftest.py").write_text(
        "print('all good')\nraise SystemExit(0)\n", encoding="utf-8")

    outcome = testrun.run(tmp_path)

    assert outcome.ran and outcome.ok
    assert "passed" in outcome.summary()
    assert "all good" in outcome.tail


def test_a_failing_script_comes_back_red_with_the_output(tmp_path):
    (tmp_path / "selftest.py").write_text(
        "print('expected 3, got 7')\nraise SystemExit(1)\n", encoding="utf-8")

    outcome = testrun.run(tmp_path)

    assert outcome.ran and not outcome.ok
    assert "FAILED" in outcome.summary()
    assert "expected 3, got 7" in outcome.tail


def test_a_crashing_script_is_a_failure_not_an_exception(tmp_path):
    (tmp_path / "selftest.py").write_text(
        "raise RuntimeError('boom')\n", encoding="utf-8")

    outcome = testrun.run(tmp_path)

    assert outcome.ran and not outcome.ok
    assert "boom" in outcome.tail


def test_a_hung_suite_is_killed_and_reported(tmp_path):
    (tmp_path / "selftest.py").write_text(
        "import time\ntime.sleep(60)\n", encoding="utf-8")

    outcome = testrun.run(tmp_path, timeout_s=10)

    assert outcome.ran and not outcome.ok
    assert "did not finish" in outcome.tail


def test_no_tests_is_reported_never_a_green_light(tmp_path):
    outcome = testrun.run(tmp_path)

    assert not outcome.ran and not outcome.ok
    assert "no tests found" in outcome.reason


def test_the_tail_is_bounded(tmp_path):
    (tmp_path / "selftest.py").write_text(
        "print('x' * 100000)\n", encoding="utf-8")

    outcome = testrun.run(tmp_path)

    assert len(outcome.tail) <= testrun.TAIL_CHARS
