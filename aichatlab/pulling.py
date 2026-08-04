"""Reading Ollama's download progress into something worth watching.

`/api/pull` streams NDJSON: a status line, then a line per layer carrying
`completed` and `total` byte counts, then `{"status": "success"}`.  Rendered
literally that is a thousand lines a second of digests nobody can read.

What a person actually wants is one line: how far through, how fast, and how
long is left.  The awkward part is that "how far through" is not any single
layer — a model is several layers pulled in sequence, each reporting its own
`completed`/`total`, and a naive reading of the newest line makes the bar
leap backwards to zero every time a layer finishes.  So progress is tracked
per digest and summed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Nothing meaningful can be said about the rate until a little has moved.
MIN_SAMPLE_S = 1.0


def human_bytes(size: float) -> str:
    size = float(size or 0)
    if size < 1024:
        return f"{size:.0f} B"
    for unit in ("KB", "MB", "GB", "TB"):
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {unit}" if size < 10 else f"{size:.0f} {unit}"
    return f"{size:.0f} PB"


def human_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0))
    if seconds < 60:
        return f"{seconds:.0f}s left"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s left"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m left"


@dataclass
class Progress:
    """Running totals across every layer of one download."""

    status: str = ""
    layers: dict = field(default_factory=dict)     # digest -> (done, total)
    started: float | None = None
    done: bool = False
    # Bytes already on disk when we started, so a resumed pull does not
    # report a wildly optimistic rate from its first instant.
    baseline: int = 0

    def update(self, payload: dict, now: float) -> None:
        if self.started is None:
            self.started = now
        status = str(payload.get("status") or "")
        if status:
            self.status = status
        if status == "success":
            self.done = True

        digest = payload.get("digest")
        if not digest:
            return
        try:
            total = int(payload.get("total") or 0)
            completed = int(payload.get("completed") or 0)
        except (TypeError, ValueError):
            return
        if total <= 0:
            return
        if digest not in self.layers and completed >= total:
            # A layer that is *complete* the first time it is mentioned was
            # already on disk.  A partly-done one is ambiguous, but far more
            # likely to be a live download whose first progress line we
            # happened to catch — and counting that as cached would
            # under-report the rate rather than wildly over-report it.
            self.baseline += completed
        self.layers[digest] = (completed, total)

    # -- derived numbers ---------------------------------------------------
    @property
    def completed(self) -> int:
        return sum(done for done, _total in self.layers.values())

    @property
    def total(self) -> int:
        return sum(total for _done, total in self.layers.values())

    @property
    def fraction(self) -> float:
        total = self.total
        return min(1.0, self.completed / total) if total else 0.0

    def rate(self, now: float) -> float:
        """Bytes per second, measured over the whole download."""
        if self.started is None:
            return 0.0
        elapsed = now - self.started
        if elapsed < MIN_SAMPLE_S:
            return 0.0
        moved = max(0, self.completed - self.baseline)
        return moved / elapsed if moved else 0.0

    def eta(self, now: float) -> float:
        rate = self.rate(now)
        if rate <= 0:
            return 0.0
        return max(0.0, (self.total - self.completed) / rate)

    def describe(self, model: str, now: float) -> str:
        """The one line worth showing."""
        if self.done:
            return f"✅ {model} downloaded ({human_bytes(self.total)})."
        if not self.total:
            # "pulling manifest", "verifying sha256 digest", and so on.
            return f"⬇ {model} — {self.status or 'starting'}…"
        percent = self.fraction * 100
        line = (f"⬇ {model} — {percent:.0f}%  "
                f"({human_bytes(self.completed)} of {human_bytes(self.total)})")
        rate = self.rate(now)
        if rate > 0:
            line += f"  ·  {human_bytes(rate)}/s  ·  {human_eta(self.eta(now))}"
        return line


def looks_like_model_name(text: str) -> bool:
    """A light sanity check, so a typo does not become a confusing 500.

    Deliberately permissive: registries, namespaces and tags all vary, and
    guessing too strictly here would block a name that is perfectly valid.
    """
    name = (text or "").strip()
    if not name or " " in name or "\n" in name:
        return False
    return all(character.isalnum() or character in "._-:/" for character in name)
