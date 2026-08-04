"""Why nothing was written.

The edit pipeline fails closed on purpose, and that is the right call — but it
means the most common complaint about this app is "it told me what it was
going to change and then did nothing". From the outside every one of those
failures looks identical: a reply full of confident prose, and a folder that
did not move.

They are not identical on the inside. The reply might have had no edit markers
at all; it might have had markers the parser could not close; it might have had
perfectly good blocks that were refused for quoting lines the file does not
contain; it might have proposed changes that were already on disk. Those need
four different fixes, and none of them is guessable from the chat window.

So this module records the decision, not the outcome. Every run of the edit
check writes one `Trace`: which gate it fell through, how many blocks were seen
at each stage, what shape the reply actually had, and why each rejection
happened. Traces are kept in memory for the session and appended to a JSONL
file so they survive the app closing and can be read by something other than a
human squinting at a chat log.

Nothing here changes what the pipeline does. It is a window, not a lever.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

TRACE_PATH = Path.home() / ".ai_chat_lab_edit_trace.jsonl"

# The session's own memory of what happened. Bounded because this is a
# debugging aid for the session you are in, not an audit trail.
MAX_TRACES = 60

# How much of the reply to keep. Enough to see the shape of the thing and read
# the region where the markers should have been; not so much that the file
# becomes a transcript.
REPLY_KEEP = 6000

# Every way the check can end. The value is what to tell a person.
VERDICTS = {
    "offered": "the review window was offered",
    "auto-applied": "applied without asking (you chose that earlier in "
                    "this chat)",
    "no-folder": "no folder is attached, so there was nothing to write to",
    "edits-off": "editing is switched off in settings",
    "all-rejected": "every block was refused",
    "no-change": "the changes were identical to what is already on disk",
    "prose-offered": "no edit blocks, but the described changes were "
                     "recognised and offered as a follow-up",
    "prose-unmatched": "no edit blocks, and the prose did not name a file "
                       "the app could match",
    "no-target": "no edit blocks, and there was no model to go back to",
    "menu-offered": "a menu of candidate changes was offered to pick "
                    "from",
}

# What the reply *looked* like. These deliberately duplicate nothing: they are
# imported from the parser so the shape report cannot drift away from the
# thing it is describing.
FENCE = re.compile(r"^[ \t]*```", re.MULTILINE)

# A model that has the right idea and the wrong syntax. Worth telling apart
# from one that never tried, because the fix is a prompt change, not a
# different model.
#
# The decoration is required, not optional. Without it this matches every
# reply that happens to begin a sentence with "Replace the timeout…", and a
# hint that fires on ordinary prose is worse than no hint — it sends you
# hunting for a malformed block that was never there.
NEAR_MISS = re.compile(
    r"^[ \t]*(?:"
    r"(?:#{1,6}|\*{2}|={2,}|-{2,})[ \t]*"
    r"(?:FILE|EDIT|FIND|REPLACE|SEARCH|INSERT)\b"
    r"|(?:BEGIN|UPDATED?|NEW|MODIFIED|CHANGED)[ \t]+FILE\b"
    r")",
    re.MULTILINE | re.IGNORECASE)


def shape(reply: str) -> dict:
    """What the reply looked like, structurally.

    This is the single most useful thing in a trace. "Nothing was written"
    plus "the reply contained six fenced code blocks and no edit markers"
    is a diagnosis; "nothing was written" on its own is a shrug.
    """
    from . import edits as edit_tools

    text = reply or ""
    return {
        "chars": len(text),
        "file_openers": len(edit_tools.OPENER.findall(text)),
        "file_closers": len(edit_tools.CLOSER.findall(text)),
        "edit_openers": len(edit_tools.EDIT_OPENER.findall(text)),
        "edit_blocks": len(edit_tools.EDIT_BLOCK.findall(text)),
        "find_parts": len(edit_tools.FIND_PART.findall(text)),
        "code_fences": len(FENCE.findall(text)),
        "near_misses": len(NEAR_MISS.findall(text)),
    }


def shape_note(info: dict) -> str:
    """The one sentence a person should read about that shape."""
    if info["file_openers"] and info["file_openers"] > info["file_closers"]:
        return (f"{info['file_openers']} FILE block(s) opened but only "
                f"{info['file_closers']} closed — the model left off the "
                f"`=== END FILE ===` marker, or the reply was cut short")
    if info["edit_openers"] and not info["edit_blocks"]:
        return (f"{info['edit_openers']} EDIT block(s) opened but none "
                f"parsed — usually a missing `=== END EDIT ===` or a "
                f"missing `--- FIND` section")
    if info["edit_blocks"] and not info["find_parts"]:
        return "EDIT blocks with no FIND section — nothing to anchor to"
    if not info["file_openers"] and not info["edit_openers"]:
        if info["code_fences"] >= 2:
            return (f"no edit markers at all, but {info['code_fences'] // 2} "
                    f"fenced code block(s) — the model answered with code "
                    f"instead of an edit block, which the app cannot apply")
        if info["near_misses"]:
            return (f"no usable markers, but {info['near_misses']} line(s) "
                    f"that look like an attempt at one — the model has the "
                    f"idea and the wrong syntax")
        return ("no edit markers and no code — the model replied with prose "
                "only, so there was never anything to apply")
    return ""


@dataclass
class Step:
    """One stage of the pipeline and what came out of it."""

    name: str
    count: int = 0
    names: list = field(default_factory=list)
    detail: str = ""

    def line(self) -> str:
        bit = f"{self.name}: {self.count}"
        if self.names:
            bit += f" ({', '.join(self.names[:6])}"
            bit += ", …)" if len(self.names) > 6 else ")"
        return bit + (f" — {self.detail}" if self.detail else "")


@dataclass
class Trace:
    """One run of the edit check, start to verdict."""

    at: str = ""
    verdict: str = ""
    offered: bool = False
    project: str = ""
    truncated: bool = False
    second_pass: bool = False
    had_target: bool = False
    steps: list = field(default_factory=list)
    rejections: list = field(default_factory=list)
    # Problems the edit would introduce into a file that parses today. These
    # do not stop it being offered — the user may know better — but a diff
    # that breaks the file must never reach them looking clean.
    flaws: list = field(default_factory=list)
    claimed_to_write: str = ""
    reply_shape: dict = field(default_factory=dict)
    reply_head: str = ""

    # -- filled in as the check runs ---------------------------------------
    def step(self, name: str, items=(), detail: str = "") -> Trace:
        """Record a stage. `items` may be Edits, Rejecteds or plain names."""
        found = list(items)
        self.steps.append(Step(
            name=name, count=len(found), detail=detail,
            names=[getattr(item, "name", str(item)) for item in found]))
        return self

    def broke(self, name: str, items) -> Trace:
        for item in items:
            self.flaws.append({
                "name": name,
                "line": getattr(item, "line", 0),
                "problem": getattr(item, "describe", lambda: str(item))()})
        return self

    def refused(self, items) -> Trace:
        for item in items:
            self.rejections.append({
                "name": getattr(item, "name", str(item)),
                "reason": getattr(item, "reason", "")})
        return self

    def done(self, verdict: str) -> Trace:
        self.verdict = verdict
        self.offered = verdict in ("offered", "auto-applied")
        return self

    # -- reading it back ---------------------------------------------------
    def explain(self) -> str:
        """The whole trace as something a person can read in one go."""
        lines = [f"[{self.at}] {self.verdict} — "
                 f"{VERDICTS.get(self.verdict, self.verdict)}"]
        if self.project:
            lines.append(f"  folder: {self.project}")
        flags = []
        if self.truncated:
            flags.append("reply was cut off at the token cap")
        if self.second_pass:
            flags.append("second pass (one change on its own)")
        if not self.had_target:
            flags.append("no model recorded to retry against")
        if flags:
            lines.append("  " + "; ".join(flags))

        note = shape_note(self.reply_shape) if self.reply_shape else ""
        if note:
            lines.append(f"  shape: {note}")
        if self.reply_shape:
            lines.append("  counts: " + ", ".join(
                f"{key}={value}" for key, value in self.reply_shape.items()))
        for step in self.steps:
            lines.append(f"  {step.line()}")
        for item in self.rejections:
            lines.append(f"  refused {item['name']}: {item['reason']}")
        for item in self.flaws:
            lines.append(f"  ⚠ {item['name']} would break: {item['problem']}")
        if self.claimed_to_write:
            lines.append(f"  ⚠ the model claimed to have written: "
                         f"“{self.claimed_to_write}”")
        return "\n".join(lines)

    def to_json(self) -> dict:
        data = asdict(self)
        data["verdict_text"] = VERDICTS.get(self.verdict, self.verdict)
        data["shape_note"] = (shape_note(self.reply_shape)
                              if self.reply_shape else "")
        return data


def begin(reply: str, project=None, truncated: bool = False,
          second_pass: bool = False, had_target: bool = False,
          now: datetime | None = None) -> Trace:
    """Open a trace for a reply that is about to be checked for edits."""
    text = reply or ""
    return Trace(
        at=(now or datetime.now()).strftime("%H:%M:%S"),
        project=str(project) if project else "",
        truncated=bool(truncated),
        second_pass=bool(second_pass),
        had_target=bool(had_target),
        reply_shape=shape(text),
        reply_head=text[:REPLY_KEEP])


@dataclass
class EditDebug:
    """The session's traces, in memory and on disk.

    Writing to disk is best-effort on purpose. A debugging aid that can crash
    the feature it is debugging is worse than no debugging aid, so every
    filesystem error here is swallowed — the in-memory copy is the one the app
    relies on, and the file is a convenience for reading the record from
    somewhere other than the app.
    """

    traces: list = field(default_factory=list)
    path: Path | None = None
    enabled: bool = True
    limit: int = MAX_TRACES

    def __len__(self) -> int:
        return len(self.traces)

    def record(self, trace: Trace) -> Trace:
        if not self.enabled:
            return trace
        self.traces.append(trace)
        if len(self.traces) > self.limit:
            del self.traces[:-self.limit]
        self._append(trace)
        return trace

    def _append(self, trace: Trace) -> None:
        target = self.path if self.path is not None else TRACE_PATH
        try:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace.to_json(),
                                        ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            pass        # never let the record-keeping break the feature

    def last(self) -> Trace | None:
        return self.traces[-1] if self.traces else None

    def failures(self) -> list:
        return [t for t in self.traces if not t.offered]

    def clear(self) -> None:
        self.traces.clear()

    def summary(self) -> str:
        """The session in one line: how often the pipeline reached a diff."""
        if not self.traces:
            return "The edit check has not run yet this session."
        tally: dict[str, int] = {}
        for trace in self.traces:
            tally[trace.verdict] = tally.get(trace.verdict, 0) + 1
        offered = sum(1 for t in self.traces if t.offered)
        bits = [f"{len(self.traces)} check"
                f"{'s' if len(self.traces) != 1 else ''}",
                f"{offered} reached a diff"]
        bits += [f"{count} {verdict}" for verdict, count in
                 sorted(tally.items(), key=lambda kv: -kv[1])
                 if verdict not in ("offered", "auto-applied")]
        return " · ".join(bits)

    def report(self) -> str:
        """Every trace, most recent last. This is the thing to paste."""
        if not self.traces:
            return "The edit check has not run yet this session."
        parts = [self.summary(), ""]
        parts += [trace.explain() for trace in self.traces]
        return "\n".join(parts)


def read_traces(path=None, limit: int = 20) -> list[dict]:
    """The most recent traces from the log file, oldest first.

    Reading is as forgiving as writing: a half-written final line from an app
    that was closed mid-append should not stop the rest being readable.
    """
    target = Path(path) if path is not None else TRACE_PATH
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            found.append(json.loads(line))
        except ValueError:
            continue
    return found[-limit:] if limit else found
