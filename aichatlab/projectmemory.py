"""What a new chat should already know about a project it has seen before.

Attaching a folder gives a model the code, freshly read, every time.  What it
does not give is any of the thinking: what was tried, what was rejected and
why, what is half-finished.  Start a new chat and the source is current while
the reasoning about it is gone — which is the wrong way round, because the
source was never the part at risk.

Learning mode does not fill this gap and should not be made to.  It stores
durable general facts and deliberately rejects anything time-sensitive, which
is exactly what project state is: "we are midway through folding the symbol
table into the relocation walk" is not a fact, it is a position.

So three small files live beside the project, in plain Markdown that a person
can read and edit:

* `project.md`  — what this is, how it is built, the conventions to respect.
* `decisions.md` — append-only: what was decided, and why.
* `progress.md` — where we got to, and what is next.

They are capped hard.  Memory that grows without limit stops being memory and
becomes another context problem, and the whole point is to spend a few hundred
tokens to save re-explaining the project every time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DIRECTORY = ".aichatlab"

BRIEF = "project.md"
DECISIONS = "decisions.md"
PROGRESS = "progress.md"

# Per-file ceilings.  Beyond these, memory is competing with the material it
# is supposed to make room for.
MAX_BRIEF_CHARS = 6_000
MAX_DECISIONS_CHARS = 8_000
MAX_PROGRESS_CHARS = 3_000

# What the whole lot is allowed to cost in a prompt.
MAX_PROMPT_CHARS = 9_000

HEADER = "[Project memory — %s]"
FOOTER = "[End of project memory.]"

NOTHING = {"", "none", "none.", "nothing", "n/a", "no change", "no changes"}


def directory(folder) -> Path:
    return Path(folder) / DIRECTORY


def paths(folder) -> dict:
    base = directory(folder)
    return {"brief": base / BRIEF,
            "decisions": base / DECISIONS,
            "progress": base / PROGRESS}


def _read(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    # Keep the end: the most recent decisions and the latest progress are
    # what matter, and the top of a long file is the oldest part of it.
    return "…[earlier entries trimmed]…\n" + text[-limit:]


@dataclass(frozen=True)
class Memory:
    """Everything remembered about one project."""

    name: str = ""
    brief: str = ""
    decisions: str = ""
    progress: str = ""

    @property
    def empty(self) -> bool:
        return not (self.brief or self.decisions or self.progress)

    def as_block(self, limit: int = MAX_PROMPT_CHARS) -> str:
        """The pinned block a chat starts with, or "" when there is nothing.

        Ordered by what a model most needs first, and trimmed from the least
        important end, so a cap never costs you the brief.
        """
        if self.empty:
            return ""
        parts = []
        if self.brief:
            parts.append(f"## What this project is\n\n{self.brief}")
        if self.progress:
            parts.append(f"## Where we got to\n\n{self.progress}")
        if self.decisions:
            parts.append(f"## Decisions already taken\n\n{self.decisions}")

        body = "\n\n".join(parts)
        head = HEADER % (self.name or "this project")
        marker = "\n…[project memory trimmed]…"
        # The scaffolding is charged for up front.  Trimming the body to the
        # limit and *then* adding a marker and a footer is how a cap ends up
        # being politely exceeded by exactly the length of its own apology.
        room = max(0, limit - len(head) - len(FOOTER) - len(marker) - 2)
        if len(body) > room:
            body = body[:room].rstrip() + marker
        return f"{head}\n{body}\n{FOOTER}"


def load(folder) -> Memory:
    """Read whatever is remembered.  Missing files are simply empty."""
    found = paths(folder)
    return Memory(
        name=Path(folder).name,
        brief=_read(found["brief"], MAX_BRIEF_CHARS),
        decisions=_read(found["decisions"], MAX_DECISIONS_CHARS),
        progress=_read(found["progress"], MAX_PROGRESS_CHARS))


def exists(folder) -> bool:
    return directory(folder).is_dir()


def _write(path: Path, text: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic, so a crash mid-write cannot leave half a memory behind.
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
        return True
    except OSError:
        return False


def _trim_front(text: str, limit: int) -> str:
    """Keep the newest entries when a file outgrows its cap."""
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    # Do not start mid-entry; find the first heading after the cut.
    match = re.search(r"^##? ", cut, re.MULTILINE)
    if match:
        cut = cut[match.start():]
    return "…[earlier entries removed]…\n\n" + cut


def save_brief(folder, text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    return _write(paths(folder)["brief"], text[:MAX_BRIEF_CHARS] + "\n")


def save_progress(folder, text: str, when: str = "") -> bool:
    """Replace the hand-off note.  Progress is a position, not a history —
    the history lives in decisions.md."""
    text = (text or "").strip()
    if not text or text.lower() in NOTHING:
        return False
    stamp = when or datetime.now().strftime("%Y-%m-%d %H:%M")
    body = f"_Last updated {stamp}_\n\n{text}"
    return _write(paths(folder)["progress"], body[:MAX_PROGRESS_CHARS] + "\n")


def append_decision(folder, text: str, when: str = "") -> bool:
    """Add to the log without rewriting what is already there."""
    text = (text or "").strip()
    if not text or text.lower() in NOTHING:
        return False
    stamp = when or datetime.now().strftime("%Y-%m-%d")
    path = paths(folder)["decisions"]
    existing = ""
    try:
        existing = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        existing = ""
    if text in existing:
        return False                    # already recorded; do not duplicate
    entry = f"## {stamp}\n\n{text}\n"
    combined = f"{existing.rstrip()}\n\n{entry}" if existing.strip() else entry
    return _write(path, _trim_front(combined, MAX_DECISIONS_CHARS))


# -- asking a model to write it -------------------------------------------
HANDOFF_PROMPT = """\
This conversation is ending. Write a hand-off for whoever picks this project
up next — most likely the same person tomorrow, in a fresh chat with no memory
of what we just did.

Reply in exactly this shape, and put nothing else in your answer:

PROGRESS:
<where the work stands now and what the obvious next step is — at most six
lines. Say what is half-finished. Do not summarise the conversation.>

DECISION:
<one decision taken in this conversation that should still hold next month,
and the reason for it. Write NONE if nothing was really decided.>

The conversation:

{transcript}
"""

PROGRESS_RE = re.compile(r"PROGRESS\s*:\s*(.*?)(?=^\s*DECISION\s*:|\Z)",
                         re.DOTALL | re.MULTILINE | re.IGNORECASE)
DECISION_RE = re.compile(r"DECISION\s*:\s*(.*)",
                         re.DOTALL | re.IGNORECASE)


def build_handoff_prompt(transcript: str, limit: int = 12_000) -> str:
    text = (transcript or "").strip()
    if len(text) > limit:
        # The end of a conversation is where it got to; the start is not.
        text = "…[earlier turns omitted]…\n\n" + text[-limit:]
    return HANDOFF_PROMPT.format(transcript=text)


def parse_handoff(reply: str) -> tuple[str, str]:
    """(progress, decision) — either may be "" when the model had nothing."""
    text = reply or ""
    progress_match = PROGRESS_RE.search(text)
    decision_match = DECISION_RE.search(text)
    progress = (progress_match.group(1).strip() if progress_match else "")
    decision = (decision_match.group(1).strip() if decision_match else "")
    if decision.lower().strip(" .") in NOTHING:
        decision = ""
    if progress.lower().strip(" .") in NOTHING:
        progress = ""
    return progress, decision


def worth_a_handoff(messages) -> bool:
    """Do not write a hand-off for a chat where nothing happened."""
    real = [m for m in messages or []
            if m.get("role") in ("user", "assistant")
            and len(str(m.get("content", ""))) > 40]
    return len(real) >= 4
