"""What the app is doing right now, in words a waiting person can use.

"Working…" is not a progress report. It does not say which model, what step,
how long it has been going, or whether anything is still arriving — so a run
that has quietly wedged looks exactly like one that is thinking hard, and the
only way to tell them apart is to wait and find out.

This module holds the state behind a live status line: the current step, when
it started, how much has arrived and when the last token landed. Time is
always passed in rather than read, so every one of these behaviours is
testable without sleeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Nothing at all for this long means something is wrong, or a big model is
# still being loaded into memory — either way the user deserves to be told.
STALL_SECONDS = 30.0

STEPS = {
    "thinking": "is thinking",
    "reasoning": "is reasoning it through",
    "writing": "is writing",
    "reading": "is reading your files",
    "searching": "is searching the web",
    "compacting": "is summarising the older turns",
    "learning": "is looking for anything worth remembering",
    "planning": "is planning the research",
    "checking": "is fact-checking its answer",
}


def human_duration(seconds: float) -> str:
    """"92" -> "1m 32s".  Long waits are the ones that need reading quickly."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


@dataclass
class Activity:
    """One thing the app is currently doing."""

    step: str = "thinking"
    label: str = ""                 # usually the model's friendly name
    started: float = 0.0
    last_at: float = 0.0
    tokens: int = 0
    warned: bool = False            # a stall has already been reported
    detail: str = ""
    prompt_tokens: int = 0          # how much it has to read before writing

    def touch(self, now: float, tokens: int = 1) -> None:
        self.tokens += tokens
        self.last_at = now
        self.warned = False
        if self.step == "thinking":
            self.step = "writing"

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.started)

    def silent_for(self, now: float) -> float:
        return max(0.0, now - (self.last_at or self.started))

    def rate(self, now: float) -> float | None:
        seconds = self.elapsed(now)
        if self.tokens < 2 or seconds <= 0:
            return None
        return self.tokens / seconds

    def stalled(self, now: float, after: float | None = None) -> bool:
        # Read the threshold at call time, not at import time, so it stays
        # one number that can be turned down rather than a frozen default.
        limit = STALL_SECONDS if after is None else after
        return self.silent_for(now) >= limit


@dataclass
class Tracker:
    """Every in-flight activity, keyed by stream or task."""

    items: dict = field(default_factory=dict)

    def start(self, key, step: str, label: str = "", now: float = 0.0,
              detail: str = "") -> Activity:
        activity = Activity(step=step, label=label, started=now, last_at=now,
                            detail=detail)
        self.items[key] = activity
        return activity

    def touch(self, key, now: float, tokens: int = 1) -> None:
        activity = self.items.get(key)
        if activity is not None:
            activity.touch(now, tokens)

    def set_step(self, key, step: str) -> None:
        """Rename what a running activity is doing, without restarting it."""
        activity = self.items.get(key)
        if activity is not None:
            activity.step = step

    def finish(self, key) -> Activity | None:
        return self.items.pop(key, None)

    def clear(self) -> None:
        self.items.clear()

    def busy(self) -> bool:
        return bool(self.items)

    def current(self) -> Activity | None:
        """The activity worth showing — the one that started most recently."""
        if not self.items:
            return None
        return max(self.items.values(), key=lambda a: a.started)

    def stalled(self, now: float, after: float | None = None) -> list:
        return [(key, a) for key, a in self.items.items()
                if a.stalled(now, after) and not a.warned]


def describe(activity: Activity | None, now: float, extra: int = 0) -> str:
    """The status line: who, what, for how long, and how fast."""
    if activity is None:
        return "Ready"

    verb = STEPS.get(activity.step, activity.step)
    who = activity.label or "The model"
    parts = [f"{who} {verb}…"]
    if activity.tokens:
        parts.append(f"{activity.tokens:,} tokens")
    parts.append(human_duration(activity.elapsed(now)))
    rate = activity.rate(now)
    if rate:
        parts.append(f"{rate:.0f} tok/s")
    if activity.detail:
        parts.append(activity.detail)
    line = "  ·  ".join(parts)
    if extra > 0:
        line += f"   (+{extra} more running)"
    return line


# Below this, a prompt is small enough that silence is not explained by it.
BIG_PROMPT_TOKENS = 4_000


def stall_note(activity: Activity, now: float) -> str:
    """Said once, when a step goes quiet — never a silent spinner.

    Silence before the first word usually means something quite different
    from silence in the middle: the model is reading the prompt, and a big
    prompt through a big model takes minutes at that stage with nothing to
    show for it.  Calling that a stall is simply wrong.
    """
    who = activity.label or "The model"
    step = STEPS.get(activity.step, "is working")
    waited = human_duration(activity.silent_for(now))

    if activity.tokens:
        return (f"⚠ Nothing more from {who} for {waited} — it {step} and has "
                f"written {activity.tokens:,} tokens so far. It may still be "
                f"working, or the connection may have dropped; Stop takes "
                f"effect as soon as the server responds.")
    if activity.prompt_tokens >= BIG_PROMPT_TOKENS:
        return (f"⏳ {who} has not started writing yet — it is still reading "
                f"the {activity.prompt_tokens:,}-token prompt, which a large "
                f"model can spend several minutes on before the first word. "
                f"({waited} so far.)")
    return (f"⚠ {who} has sent nothing for {waited} while it {step}. A large "
            f"model is often still being loaded into memory on the first "
            f"request; if this keeps up, check the server is reachable.")


# ------------------------------------------------ interrupting a running job

# Words that mean "stop what you're doing", as opposed to a new question that
# can politely wait its turn.
URGENT = (
    "stop", "cancel", "abort", "never mind", "nevermind", "forget it",
    "wait", "hold on", "no no", "that's wrong", "thats wrong", "wrong",
    "not what i", "instead", "actually", "scrap that", "start over",
)


def looks_urgent(text: str) -> bool:
    """Does this read like an interruption rather than the next question?"""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    opening = lowered[:60]
    return any(word in opening for word in URGENT)
