"""A record of every request the app made, and what became of it.

Written after two hours of "I have no idea what it's doing". A transcript
shows what the models said; it does not show that a summarisation call ran for
four minutes, that six requests were cut off at the same token cap, or that
the host spent most of the afternoon loading a model. Those questions need a
log, and a log is only useful if it records the boring cases too.

Kept in memory and bounded — this is for the session you are in, not an audit
trail. Exporting to CSV is there because the moment you want to compare
numbers you want them in a spreadsheet.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

MAX_ENTRIES = 500

OUTCOMES = {
    "finished": "finished",
    "length": "cut off at the token cap",
    "cancelled": "stopped by you",
    "error": "failed",
}


@dataclass
class Entry:
    """One model call, start to finish."""

    model: str
    server: str = ""
    purpose: str = "chat"
    at: str = ""                       # wall clock, HH:MM:SS
    duration_s: float = 0.0
    prompt_tokens: int | None = None
    eval_tokens: int | None = None
    outcome: str = "finished"
    detail: str = ""
    prompt_eval_s: float = 0.0     # time spent reading, before the first word

    @property
    def outcome_text(self) -> str:
        return OUTCOMES.get(self.outcome, self.outcome)

    @property
    def rate(self) -> float | None:
        if self.eval_tokens and self.duration_s > 0:
            return self.eval_tokens / self.duration_s
        return None

    def row(self) -> tuple:
        return (
            self.at,
            self.model,
            self.server,
            self.purpose,
            f"{self.duration_s:.1f}s",
            f"{self.prompt_tokens:,}" if self.prompt_tokens else "—",
            f"{self.eval_tokens:,}" if self.eval_tokens else "—",
            f"{self.rate:.0f}" if self.rate else "—",
            self.outcome_text + (f" — {self.detail}" if self.detail else ""),
        )


def outcome_of(result) -> tuple[str, str]:
    """Read a ChatResult and say plainly how it ended."""
    if getattr(result, "cancelled", False):
        return "cancelled", ""
    if getattr(result, "done_reason", "") == "length":
        return "length", ""
    return "finished", ""


@dataclass
class RunLog:
    entries: list = field(default_factory=list)
    limit: int = MAX_ENTRIES

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: Entry) -> Entry:
        self.entries.append(entry)
        if len(self.entries) > self.limit:
            del self.entries[:-self.limit]
        return entry

    def record(self, result, purpose: str = "chat",
               now: datetime | None = None) -> Entry:
        """Log a finished ChatResult."""
        outcome, detail = outcome_of(result)
        raw = getattr(result, "raw", None) or {}
        prompt_ns = raw.get("prompt_eval_duration") or 0
        return self.add(Entry(
            prompt_eval_s=float(prompt_ns) / 1e9 if prompt_ns else 0.0,
            model=getattr(result, "model", "?"),
            server=getattr(result, "server", ""),
            purpose=purpose,
            at=(now or datetime.now()).strftime("%H:%M:%S"),
            duration_s=float(getattr(result, "elapsed_s", 0.0) or 0.0),
            prompt_tokens=getattr(result, "prompt_tokens", None),
            eval_tokens=getattr(result, "eval_tokens", None),
            outcome=outcome, detail=detail))

    def record_failure(self, model: str, server: str, purpose: str,
                       message: str, duration_s: float = 0.0,
                       now: datetime | None = None) -> Entry:
        return self.add(Entry(
            model=model, server=server, purpose=purpose,
            at=(now or datetime.now()).strftime("%H:%M:%S"),
            duration_s=duration_s, outcome="error",
            detail=str(message)[:160]))

    def clear(self) -> None:
        self.entries.clear()

    # -- reading it back ---------------------------------------------------
    def total_seconds(self) -> float:
        return sum(entry.duration_s for entry in self.entries)

    def counts(self) -> dict:
        tally: dict[str, int] = {}
        for entry in self.entries:
            tally[entry.outcome] = tally.get(entry.outcome, 0) + 1
        return tally

    def summary(self) -> str:
        if not self.entries:
            return "Nothing has run yet this session."
        from .activity import human_duration

        tally = self.counts()
        bits = [f"{len(self.entries)} request"
                f"{'s' if len(self.entries) != 1 else ''}",
                f"{human_duration(self.total_seconds())} of model time"]
        for outcome in ("length", "error", "cancelled"):
            if tally.get(outcome):
                bits.append(f"{tally[outcome]} {OUTCOMES[outcome]}")
        return " · ".join(bits)

    def prompt_rate(self) -> float | None:
        """Tokens per second this machine reads prompts at, from real runs.

        Generation speed is the number everyone quotes; prompt speed is the
        one that decides how long you stare at nothing before the first word,
        and it is usually several times faster and entirely different.
        """
        rates = [e.prompt_tokens / e.prompt_eval_s for e in self.entries
                 if e.prompt_tokens and e.prompt_eval_s > 0.05]
        if not rates:
            return None
        rates.sort()
        return rates[len(rates) // 2]

    def slowest(self, count: int = 1) -> list:
        return sorted(self.entries, key=lambda e: e.duration_s,
                      reverse=True)[:count]

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["time", "model", "server", "purpose", "duration",
                         "prompt tokens", "reply tokens", "tokens/s",
                         "outcome"])
        for entry in self.entries:
            writer.writerow(entry.row())
        return buffer.getvalue()


def write_csv(path, entries: Sequence[Entry]) -> None:
    log = RunLog(list(entries))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(log.to_csv())
