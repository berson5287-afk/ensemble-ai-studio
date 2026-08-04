"""Unattended running: keep going without a human clicking Continue.

A reply that stops at the length cap is not finished, it is paused — and the
only thing standing between it and a finished answer is somebody pressing a
button.  Overnight, nobody does.  This module decides when to press it
automatically and, more importantly, when to stop pressing it.

Stopping is the hard half.  Four different things go wrong on a long
unattended run, and each of them looks like progress from the inside:

* **It never finishes.**  Some prompts produce output indefinitely.  A plain
  continuation cap catches this.
* **It finishes but keeps going.**  A model that has said everything starts
  restating it, or emits a few words and stops again.  Character counts alone
  miss this, because the restatement is long — it has to be compared against
  what came before.
* **It progresses, slowly, forever.**  Real progress, no useful end.  Only a
  clock catches that one.
* **It runs out of room.**  Once the conversation fills the budget and there
  is nothing left to compact, continuing means silently deleting the start of
  the work to make room for the end of it.

The costs are asymmetric, so the defaults lean towards stopping early.  A run
that halts at 3am with a note explaining why has cost one night.  A run that
loops until morning has cost a night *and* left a transcript nobody can
trust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Roughly one sentence.  Less than this added by a whole continuation means
# the model is stopping immediately, not writing.
MIN_PROGRESS_CHARS = 40

# How much of the front of a continuation has to already exist in what came
# before for it to count as a restatement rather than a continuation.
ECHO_WINDOW = 180

DEFAULT_CONTINUES = 20
DEFAULT_HOURS = 8.0

WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Compare on words, not on whitespace and capitals."""
    return WHITESPACE.sub(" ", (text or "").strip().lower())


@dataclass(frozen=True)
class Limits:
    """What the user is willing to let happen while they are not watching."""

    continues: int = DEFAULT_CONTINUES
    hours: float = DEFAULT_HOURS
    min_progress: int = MIN_PROGRESS_CHARS


@dataclass(frozen=True)
class Decision:
    """Whether to press Continue, and what to say about it either way."""

    go: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.go


@dataclass
class Run:
    """One unattended run, across however many continuations it takes."""

    limits: Limits = field(default_factory=Limits)
    # None, not 0.0 — a run that began at t=0 is a real run, and `if
    # self.started` would quietly treat it as one that never started.  This
    # is the same trap that stopped checklist steps showing their timing.
    started: float | None = None
    continues: int = 0
    stopped: str = ""
    decisions: list[str] = field(default_factory=list)
    _seen: list[str] = field(default_factory=list)

    # -- lifecycle ---------------------------------------------------------
    def begin(self, now: float) -> None:
        self.started = now
        self.continues = 0
        self.stopped = ""
        self.decisions.clear()
        self._seen.clear()

    def note(self, text: str) -> None:
        """Record a choice made on the user's behalf, for the morning."""
        self.decisions.append(text)

    def elapsed(self, now: float) -> float:
        if self.started is None:
            return 0.0
        return max(0.0, now - self.started)

    # -- the decision ------------------------------------------------------
    def consider(self, text: str, now: float,
                 context_exhausted: bool = False) -> Decision:
        """Should this cut-off reply be continued automatically?

        `text` is the whole reply as it currently stands, not just the newest
        piece — the echo check needs the history to compare against.
        """
        if self.limits.continues and self.continues >= self.limits.continues:
            return self._stop(
                f"reached the limit of {self.limits.continues} automatic "
                f"continues on one message")

        if self.limits.hours and self.elapsed(now) >= self.limits.hours * 3600:
            return self._stop(
                f"ran for {self.limits.hours:g} hours, which is the limit")

        if context_exhausted:
            return self._stop(
                "the conversation filled the context budget and there is "
                "nothing left to compact — continuing would start deleting "
                "the beginning of the work to make room for the end of it")

        fresh = self._new_material(text)
        if len(fresh.strip()) < self.limits.min_progress:
            return self._stop(
                "the last continuation added almost nothing, so the model "
                "has finished even though it stopped at the length cap")

        if self._is_echo(fresh):
            return self._stop(
                "the model started repeating itself rather than carrying on")

        self._seen.append(normalise(text))
        self.continues += 1
        return Decision(True, f"continuing ({self.continues})")

    def _stop(self, reason: str) -> Decision:
        self.stopped = reason
        return Decision(False, reason)

    def _new_material(self, text: str) -> str:
        """Whatever this reply has that the previous one did not."""
        if not self._seen:
            return text or ""
        previous = self._seen[-1]
        current = normalise(text)
        if current.startswith(previous):
            return current[len(previous):]
        return current

    def _is_echo(self, fresh: str) -> bool:
        """Has the model gone back to the beginning instead of carrying on?"""
        head = normalise(fresh)[:ECHO_WINDOW]
        if len(head) < 60:          # too short to judge either way
            return False
        return any(head in seen for seen in self._seen)

    # -- reporting ---------------------------------------------------------
    def summary(self, now: float = 0.0) -> str:
        """What happened, written to be read after the fact."""
        if not self.continues and not self.decisions:
            return ""
        bits = []
        if self.continues:
            plural = "time" if self.continues == 1 else "times"
            bits.append(f"carried on {self.continues} {plural}")
        if self.started is not None and now:
            bits.append(f"over {human_hours(self.elapsed(now))}")
        line = "🌙 Auto mode " + ", ".join(bits) if bits else "🌙 Auto mode"
        if self.stopped:
            line += f", then stopped: {self.stopped}."
        else:
            line += "."
        if self.decisions:
            line += " Decisions made for you: " + "; ".join(self.decisions) + "."
        return line


def human_hours(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0))
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} minutes"
    return f"{minutes / 60:.1f} hours"
