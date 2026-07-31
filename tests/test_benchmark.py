"""Benchmark runs, aggregation and CSV export."""

from __future__ import annotations

import csv
import threading

from aichatlab.benchmark import run_benchmark, summarise, write_csv
from aichatlab.orchestrator import Target
from tests.conftest import FakeClient

ALPHA = Target("local", "alpha:1b")
BETA = Target("local", "beta:2b")


def test_runs_every_model_the_requested_number_of_times():
    client = FakeClient()
    rows = run_benchmark({"local": client}, [ALPHA, BETA], "hi", repeats=3)

    assert len(rows) == 6
    assert sorted({row.run for row in rows}) == [1, 2, 3]
    assert all(row.ok for row in rows)


def test_records_timing_and_throughput():
    rows = run_benchmark({"local": FakeClient()}, [ALPHA], "hi")

    row = rows[0]
    assert row.seconds == 0.25
    assert row.eval_tokens == 12
    assert row.tokens_per_second is not None


def test_failures_are_recorded_not_raised():
    client = FakeClient(fail_on="beta")
    rows = run_benchmark({"local": client}, [ALPHA, BETA], "hi")

    good = [r for r in rows if r.ok]
    bad = [r for r in rows if not r.ok]
    assert len(good) == 1 and len(bad) == 1
    assert "boom" in bad[0].error


def test_unconfigured_server_is_reported_per_row():
    rows = run_benchmark({}, [ALPHA], "hi")
    assert rows[0].ok is False
    assert "not configured" in rows[0].error


def test_cancel_stops_the_run():
    cancel = threading.Event()
    cancel.set()
    rows = run_benchmark({"local": FakeClient()}, [ALPHA, BETA], "hi", cancel=cancel)
    assert rows == []


def test_summarise_orders_fastest_first():
    rows = run_benchmark({"local": FakeClient()}, [ALPHA, BETA], "hi", repeats=2)
    rows[0].seconds = 9.0            # make alpha's first run slow
    rows[2].seconds = 9.5

    summaries = summarise(rows)
    assert len(summaries) == 2
    assert summaries[0].median_seconds <= summaries[1].median_seconds
    assert summaries[0].runs == 2


def test_summarise_counts_failures():
    client = FakeClient(fail_on="beta")
    summaries = summarise(run_benchmark({"local": client}, [ALPHA, BETA], "hi"))
    by_model = {s.model: s for s in summaries}
    assert by_model["beta:2b"].failures == 1


def test_csv_export_has_a_header_and_one_row_each(tmp_path):
    rows = run_benchmark({"local": FakeClient()}, [ALPHA, BETA], "hi")
    path = write_csv(rows, tmp_path / "bench.csv")

    with path.open(encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))

    assert len(parsed) == 2
    assert parsed[0]["model"] == "alpha:1b"
    assert parsed[0]["tokens_per_second"]
