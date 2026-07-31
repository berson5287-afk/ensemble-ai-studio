"""Run one prompt across many models and measure them.

Useful on its own, and the reason the client returns timing data: with a dozen
models on a box it is genuinely hard to know which ones are worth using until
you have latency and throughput side by side.
"""

from __future__ import annotations

import csv
import statistics
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .client import OllamaClient
from .orchestrator import Target

FIELDNAMES = [
    "model", "server", "run", "ok", "seconds", "eval_tokens",
    "tokens_per_second", "characters", "error",
]


@dataclass
class BenchmarkRow:
    model: str
    server: str
    run: int = 1
    ok: bool = True
    seconds: float = 0.0
    eval_tokens: int | None = None
    tokens_per_second: float | None = None
    characters: int = 0
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "server": self.server,
            "run": self.run,
            "ok": self.ok,
            "seconds": round(self.seconds, 3),
            "eval_tokens": self.eval_tokens if self.eval_tokens is not None else "",
            "tokens_per_second": (round(self.tokens_per_second, 2)
                                  if self.tokens_per_second else ""),
            "characters": self.characters,
            "error": self.error,
        }


@dataclass
class BenchmarkSummary:
    """Per-model aggregate across repeated runs."""

    model: str
    server: str
    runs: int = 0
    failures: int = 0
    seconds: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)

    @property
    def median_seconds(self) -> float | None:
        return statistics.median(self.seconds) if self.seconds else None

    @property
    def median_speed(self) -> float | None:
        return statistics.median(self.speeds) if self.speeds else None


def run_benchmark(clients: dict[str, OllamaClient], targets: Sequence[Target],
                  prompt: str, repeats: int = 1,
                  emit: Callable[..., None] | None = None,
                  cancel: threading.Event | None = None,
                  options: dict | None = None) -> list[BenchmarkRow]:
    """Run `prompt` against each target `repeats` times, sequentially.

    Sequential on purpose: running models in parallel on one machine makes them
    fight over the GPU and the timings stop meaning anything.
    """
    rows: list[BenchmarkRow] = []
    total = len(targets) * max(1, repeats)
    done = 0

    for run_index in range(1, max(1, repeats) + 1):
        for target in targets:
            if cancel is not None and cancel.is_set():
                return rows

            if emit:
                emit("progress", done=done, total=total,
                     text=f"Running {target.label} (run {run_index})…")

            row = BenchmarkRow(model=target.model, server=target.server,
                               run=run_index)
            client = clients.get(target.server)
            if client is None or not client.base_url:
                row.ok = False
                row.error = f"{target.server} server not configured"
            else:
                try:
                    result = client.chat(
                        target.model,
                        [{"role": "user", "content": prompt}],
                        options=options, cancel=cancel)
                    row.seconds = result.elapsed_s
                    row.eval_tokens = result.eval_tokens
                    row.tokens_per_second = result.tokens_per_second
                    row.characters = len(result.text)
                    row.ok = not result.cancelled and bool(result.text)
                    if result.cancelled:
                        row.error = "cancelled"
                except Exception as exc:
                    row.ok = False
                    row.error = str(exc)[:200]

            rows.append(row)
            done += 1
            if emit:
                emit("row", row=row, done=done, total=total)

    if emit:
        emit("finished", rows=rows)
    return rows


def summarise(rows: Sequence[BenchmarkRow]) -> list[BenchmarkSummary]:
    """Collapse repeated runs into one row per model, sorted fastest first."""
    summaries: dict[str, BenchmarkSummary] = {}
    for row in rows:
        key = f"{row.server}::{row.model}"
        summary = summaries.setdefault(
            key, BenchmarkSummary(model=row.model, server=row.server))
        summary.runs += 1
        if row.ok:
            summary.seconds.append(row.seconds)
            if row.tokens_per_second:
                summary.speeds.append(row.tokens_per_second)
        else:
            summary.failures += 1

    ordered = sorted(
        summaries.values(),
        key=lambda s: (s.median_seconds is None, s.median_seconds or 0.0))
    return ordered


def write_csv(rows: Sequence[BenchmarkRow], path) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path
