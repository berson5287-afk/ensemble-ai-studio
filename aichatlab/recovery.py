"""Keeping an unfinished conversation, so closing the window is not deleting it.

Until this existed, closing the app discarded everything: no prompt, no file,
no trace. Not only a crash — a deliberate close on a chat you had not saved
threw away the whole conversation, and a long one on a slow local model can
represent an hour of waiting.

The design deliberately avoids a "are you sure?" dialog on exit, because that
is a tax paid on every close to protect the rare one. Instead the conversation
is written out as it goes, and the *next* launch offers it back. Closing
becomes cheap again, and nothing is lost either way.

The file is kept when the chat was never saved anywhere, and thrown away when
it was — if there is a real .json on disk with this conversation in it, a
second copy in the temp directory is clutter, not insurance.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .session import Session

RECOVERY_PATH = Path(tempfile.gettempdir()) / "ai_chat_lab_unfinished.json"

SCHEMA = 1


@dataclass
class Recovered:
    """An unfinished conversation found on disk."""

    session: Session
    selected: list = field(default_factory=list)
    mode: str = "chat"
    at: str = ""                       # ISO timestamp of the last write
    attachments: list = field(default_factory=list)

    @property
    def messages(self) -> int:
        return sum(len(m) for m in self.session.conversations.values())

    def when(self, now: datetime | None = None) -> str:
        """"14:32 today" / "yesterday at 14:32" / "on 29 July at 14:32"."""
        try:
            stamp = datetime.fromisoformat(self.at)
        except (TypeError, ValueError):
            return "earlier"
        now = now or datetime.now()
        clock = stamp.strftime("%H:%M")
        days = (now.date() - stamp.date()).days
        if days <= 0:
            return f"{clock} today"
        if days == 1:
            return f"yesterday at {clock}"
        return f"{stamp.strftime('%-d %B') if os.name != 'nt' else stamp.strftime('%d %B')} at {clock}"

    def describe(self, now: datetime | None = None) -> str:
        bits = [f"{self.messages} message{'s' if self.messages != 1 else ''}"]
        if self.attachments:
            bits.insert(0, ", ".join(self.attachments))
        return (f"🔄 Found an unfinished chat from {self.when(now)} — "
                + " · ".join(bits) + ".")


def write(session: Session, selected=(), mode: str = "chat",
          attachments=(), path: Path | None = None,
          now: datetime | None = None) -> bool:
    """Save the in-progress conversation.  Never raises — this is insurance.

    Written to a neighbouring file and moved into place, so a crash during
    the write cannot leave a half-file where the recovery data should be.
    """
    target = Path(path) if path else RECOVERY_PATH
    if session.is_empty():
        return False
    payload = {
        "schema": SCHEMA,
        "at": (now or datetime.now()).isoformat(),
        "mode": mode,
        "attachments": [str(name) for name in attachments],
        "chat": session.to_dict(selected),
    }
    temporary = target.with_suffix(".part")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False),
                             encoding="utf-8")
        os.replace(temporary, target)
        return True
    except (OSError, ValueError, TypeError):
        try:
            temporary.unlink()
        except OSError:
            pass
        return False


def read(path: Path | None = None) -> Recovered | None:
    """Load an unfinished conversation, or None if there isn't a usable one."""
    source = Path(path) if path else RECOVERY_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    chat = raw.get("chat")
    if not isinstance(chat, dict):
        return None
    try:
        session, selected = Session.from_dict(chat)
    except (ValueError, TypeError, KeyError):
        return None
    if session.is_empty():
        return None

    attachments = [str(a) for a in raw.get("attachments", [])
                   if isinstance(a, str)]
    return Recovered(session=session, selected=selected,
                     mode=str(raw.get("mode", "chat")),
                     at=str(raw.get("at", "")), attachments=attachments)


def discard(path: Path | None = None) -> None:
    target = Path(path) if path else RECOVERY_PATH
    for candidate in (target, target.with_suffix(".part")):
        try:
            candidate.unlink()
        except OSError:
            pass
