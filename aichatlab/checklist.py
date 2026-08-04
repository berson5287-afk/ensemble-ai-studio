"""A list of steps that ticks itself off as the work happens.

A spinner says "something is happening". A checklist says what has already
been done, what is happening now, and what is still to come — which is the
difference between waiting and waiting *for a known thing*.

The rule this module exists to enforce is that every tick is earned. Items
appear when the app genuinely has that step to do and are marked done when it
genuinely finished, including the unglamorous outcomes: skipped, failed. A
checklist that ticks itself off optimistically is worse than none, because it
looks like progress.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .activity import human_duration

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"

MARKS = {
    PENDING: "☐",
    RUNNING: "▸",
    DONE: "☑",
    FAILED: "✗",
    SKIPPED: "⊘",
}


@dataclass
class Step:
    key: str
    label: str
    state: str = PENDING
    detail: str = ""
    started: float | None = None      # None, not 0.0: t=0 is falsy
    seconds: float = 0.0

    def line(self, now: float = 0.0) -> str:
        mark = MARKS.get(self.state, MARKS[PENDING])
        text = f"{mark} {self.label}"
        if self.detail:
            text += f" — {self.detail}"
        if self.state == RUNNING and self.started is not None:
            text += f"   {human_duration(max(0.0, now - self.started))}"
        elif self.state == DONE and self.seconds >= 1:
            text += f"   ({human_duration(self.seconds)})"
        return text


@dataclass
class Checklist:
    """The steps of one run, in the order they will happen."""

    title: str = ""
    steps: list = field(default_factory=list)

    # -- building ----------------------------------------------------------
    def add(self, key: str, label: str, detail: str = "") -> Step:
        step = Step(key=key, label=label, detail=detail)
        self.steps.append(step)
        return step

    def find(self, key: str) -> Step | None:
        for step in self.steps:
            if step.key == key:
                return step
        return None

    # -- ticking -----------------------------------------------------------
    def start(self, key: str, now: float = 0.0, detail: str = "") -> None:
        step = self.find(key)
        if step is None:
            return
        step.state = RUNNING
        step.started = now
        if detail:
            step.detail = detail

    def finish(self, key: str, now: float = 0.0, detail: str = "",
               state: str = DONE) -> None:
        step = self.find(key)
        if step is None:
            return
        step.state = state
        if detail:
            step.detail = detail
        if step.started is not None:
            step.seconds = max(0.0, now - step.started)

    def fail(self, key: str, detail: str = "", now: float = 0.0) -> None:
        self.finish(key, now=now, detail=detail, state=FAILED)

    def skip(self, key: str, detail: str = "") -> None:
        self.finish(key, detail=detail, state=SKIPPED)

    def skip_remaining(self, detail: str = "not reached") -> None:
        """Stopped early — say so rather than leaving items looking pending."""
        for step in self.steps:
            if step.state in (PENDING, RUNNING):
                step.state = SKIPPED
                step.detail = step.detail or detail

    # -- reading -----------------------------------------------------------
    @property
    def finished(self) -> int:
        return sum(1 for s in self.steps if s.state in (DONE, SKIPPED, FAILED))

    @property
    def complete(self) -> bool:
        return bool(self.steps) and self.finished == len(self.steps)

    def running(self) -> Step | None:
        for step in self.steps:
            if step.state == RUNNING:
                return step
        return None

    def render(self, now: float = 0.0) -> str:
        if not self.steps:
            return ""
        head = self.title or "Working through this"
        head = f"{head}   ({self.finished}/{len(self.steps)})"
        return "\n".join([head, *(step.line(now) for step in self.steps)])


def pipeline_for(*, attachments: Sequence[dict] = (), searching: bool = False,
                 models: Sequence[str] = (), learning: bool = False,
                 compacting: bool = False) -> Checklist:
    """The steps this particular send is actually going to perform.

    Built from what the app knows it will do, not from a fixed template, so
    an item never appears for work that was never going to happen.
    """
    checklist = Checklist(title="This request")
    if compacting:
        checklist.add("compact", "Summarise the older turns to make room")
    if attachments:
        names = ", ".join(str(a.get("name", "")) for a in attachments)
        checklist.add("attach", f"Read {names}")
    if searching:
        checklist.add("search", "Decide whether the web is needed")
    for name in models:
        checklist.add(f"model:{name}", f"Ask {name}")
    if learning:
        checklist.add("learn", "Save anything worth remembering")
    return checklist
