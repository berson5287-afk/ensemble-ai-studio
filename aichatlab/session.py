"""Conversation state, persistence and context budgeting.

Each model gets its own message history, keyed "<server>::<model>".  The
interesting part is `build_context`: model context windows are finite, and a
long chat — or one big attached document — will silently overflow them.  We
estimate tokens, keep the most recent turns that fit, and tell the model
plainly that earlier turns were dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

ROLES = ("user", "assistant", "system")
SCHEMA_VERSION = 3

# Source code, not English.  English averages ~4 characters per token, and
# this was 4 for that reason — but almost everything this app measures is a
# folder of code, where indentation, punctuation and identifiers like
# `_PRICING_DELIVERED_RE` tokenise far worse.  Measured against a real
# tokeniser on a real project: 4.2 characters per token through the README,
# 3.0 across the whole attachment, and about 2.5 through the dense middle of a
# source file.
#
# Guessing high is not a symmetric mistake.  Guess low and the app builds a
# prompt it believes fits; the server then finds it does not, and Ollama's
# answer to that is to keep only the most recent half — which throws away the
# *front* of the prompt, where the system prompt and the editing instructions
# live.  The model then answers a folder of code with no idea it was ever
# allowed to edit anything, and the user is told their files were listed and
# nothing happened.  That was this app's commonest editing failure, and it was
# caused by this constant.
CHARS_PER_TOKEN = 3

# Turning a *token* budget back into "how many characters may I read" is the
# other direction, and it cannot use the measured estimate because the text
# does not exist yet.  Measured on real projects the dense parts run to about
# 2.5 characters per token, so 2 is the honest floor to read against: it
# under-fills the window rather than overflowing it, and overflowing it is the
# failure that silently costs the system prompt.
DENSE_CHARS_PER_TOKEN = 2

# Tokenisers split on word boundaries, give most punctuation a token of its
# own, and charge again for runs of indentation.  Counting those three things
# tracks a real tokeniser far better than dividing by any constant can: across
# a 115,000-character project attachment this lands within 1% of the true
# count, where characters-over-four was 34% under.
_WORDS = re.compile(r"[A-Za-z0-9_]+")
_SYMBOLS = re.compile(r"[^A-Za-z0-9_\s]")
_INDENTS = re.compile(r"[ \t]{2,}")

# Room kept clear of attachments so the conversation itself still fits.
CHAT_RESERVE_TOKENS = 2_000

ATTACHMENT_PREFIXES = ("[Folder: ", "[Attached file: ")

# A folder block runs to its end marker; a file block to the close of its fence.
#
# Both markers are anchored to whole lines, because attaching a project can
# easily include a source file that mentions them — this app's own
# folderscan.py contains the string "[End of folder contents.]" — and an
# unanchored match stops at that mention, leaving half the block behind.
ATTACHMENT_BLOCK = re.compile(
    r"^\[Folder: (?P<folder>[^\]—\n]+?)(?: —[^\]\n]*)?\]"
    r".*?^\[End of folder contents\.\]$"
    # `\Z`, not `$`: under MULTILINE `$` matches at every line end, so a
    # fence inside the attached file would satisfy the lookahead and close
    # the block early.
    r"|^\[Attached file: (?P<file>[^\]\n]+)\]\n```\n.*?\n```(?=\n\n|\s*\Z)",
    re.DOTALL | re.MULTILINE)

SHORTENED_MARKER = "\n\n…[attachment shortened to fit the context budget]…"


# Counting every character of a 150,000-character attachment takes about
# 13 milliseconds, and this is called in loops — over every message on every
# compaction check, and again whenever the context label is redrawn. Past this
# size the text is sampled instead: three windows spread through it, scaled by
# what fraction they covered. Prose and code are not evenly mixed inside a
# file, but they are even enough across three samples to hold the accuracy
# that matters here, and bounded work is what keeps the window responsive.
_EXACT_LIMIT = 12_000
_SAMPLES = 3

# Sampling three windows lands a few percent low on a mixed attachment, and a
# few percent low is the direction that overflows a context window. The margin
# buys that back and costs a little room we would rather waste than lose the
# system prompt over.
_SAMPLE_MARGIN = 1.05


def _count(text: str) -> int:
    words = sum(max(1, (len(word) + 3) // 4) for word in _WORDS.findall(text))
    return (words + len(_SYMBOLS.findall(text))
            + len(_INDENTS.findall(text)) + text.count("\n"))


def estimate_tokens(text: str) -> int:
    """How many tokens this text is likely to cost.

    Erring high is the safe direction and erring low is not, so this counts
    what a tokeniser charges for rather than dividing by an average. On plain
    English it comes out about a third high, which costs a little room; on
    source code it is accurate, which is what stops the app building a prompt
    the server has to throw the front off.
    """
    if not text:
        return 0
    total = len(text)
    if total <= _EXACT_LIMIT:
        return max(1, _count(text))
    window = _EXACT_LIMIT // _SAMPLES
    step = total // _SAMPLES
    taken, counted = 0, 0
    for index in range(_SAMPLES):
        chunk = text[index * step:index * step + window]
        taken += len(chunk)
        counted += _count(chunk)
    if not taken:
        return max(1, total // CHARS_PER_TOKEN)
    return max(1, int(counted * (total / taken) * _SAMPLE_MARGIN))


def carries_attachment(content: str) -> bool:
    """Does this message contain a file or folder the user attached?

    Detected from the text rather than recorded as a flag, so chats saved
    before this existed get the same protection when they are reopened.
    """
    return (content or "").lstrip().startswith(ATTACHMENT_PREFIXES)


def describe_attachments(content: str) -> tuple[list[str], str]:
    """Split a composed message into (attachment names, what was typed).

    Redrawing a loaded chat used to paste the entire folder — a hundred
    thousand characters of source code — into the transcript as if the user
    had typed it. The names are what a person wants to see; the block is for
    the model.
    """
    names: list[str] = []

    def collect(match: re.Match) -> str:
        folder, single = match.group("folder"), match.group("file")
        if folder:
            names.append(f"{folder.strip()}/")
        elif single:
            names.append(single.strip())
        return ""

    remainder = ATTACHMENT_BLOCK.sub(collect, content or "").strip()
    return names, remainder


def attachment_details(content: str) -> list[tuple[str, int]]:
    """(name, estimated tokens) for every attachment in a composed message.

    Reopening a chat shows attachments as chips rather than a hundred
    thousand characters of pasted source, which is right — but a bare chip
    gives no sense of whether the folder is really still there, and a
    conversation that is one question and one chip reads as empty when it is
    in fact carrying 24,000 tokens.  The size is the reassurance.
    """
    details: list[tuple[str, int]] = []
    for match in ATTACHMENT_BLOCK.finditer(content or ""):
        folder, single = match.group("folder"), match.group("file")
        name = f"{folder.strip()}/" if folder else (single or "").strip()
        if name:
            details.append((name, estimate_tokens(match.group(0))))
    return details


def attachment_allowance(budget_tokens: int,
                         reserve: int = CHAT_RESERVE_TOKENS) -> int:
    """How much of the budget attached documents may occupy.

    Never all of it: a document sized to the whole budget leaves no room for
    the question about it, so the next turn pushes it out entirely.
    """
    budget = max(0, int(budget_tokens))
    return max(0, budget - min(reserve, budget // 4))


def shorten_attachment(content: str, budget_tokens: int) -> str:
    """Trim an attachment from the end, keeping its manifest and first files."""
    limit = max(0, int(budget_tokens) * CHARS_PER_TOKEN) - len(SHORTENED_MARKER)
    if limit <= 0:
        return SHORTENED_MARKER.strip()
    if len(content) <= limit:
        return content
    return content[:limit].rstrip() + SHORTENED_MARKER


def make_key(server: str, model: str) -> str:
    return f"{server}::{model}"


def split_key(key: str) -> tuple[str, str]:
    """Split "<server>::<model>", tolerating v1 files that stored bare names."""
    if "::" in key:
        server, model = key.split("::", 1)
        if server in ("local", "host"):
            return server, model
    return "local", key


def short_model(model: str) -> str:
    return model.split(":")[0]


class Session:
    """All conversations in one chat session."""

    def __init__(self, conversations: dict[str, list[dict]] | None = None) -> None:
        self.conversations: dict[str, list[dict]] = conversations or {}
        self.path: Path | None = None

    # -- history -----------------------------------------------------------
    def history(self, key: str) -> list[dict]:
        return self.conversations.setdefault(key, [])

    def add(self, key: str, role: str, content: str,
            pinned: bool | None = None) -> dict:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        message = {"role": role, "content": content}
        if pinned is None:
            pinned = role == "user" and carries_attachment(content)
        if pinned:
            message["pinned"] = True
        self.history(key).append(message)
        return message

    def clear(self) -> None:
        self.conversations = {}
        self.path = None

    def is_empty(self) -> bool:
        return not any(self.conversations.values())

    def total_tokens(self, key: str) -> int:
        return sum(estimate_tokens(m.get("content", ""))
                   for m in self.conversations.get(key, []))

    # -- context budgeting -------------------------------------------------
    def build_context(self, key: str, budget_tokens: int = 6000,
                      system_prompt: str = "") -> list[dict]:
        """Newest-first trim of a conversation to fit a token budget.

        The system prompt is always kept.  Whole messages are dropped rather
        than cut in half, so a model never sees a truncated sentence.

        Attached files and folders are the exception, and they are the reason
        this method is more than a loop.  An attachment arrives as an ordinary
        user message, so plain oldest-first trimming throws it away first —
        and it is the oldest message in exactly the conversations that are
        entirely about it.  Reopen a saved chat, ask one follow-up, and the
        document the whole thread discusses is silently gone.  So attachments
        are kept ahead of ordinary turns, and shortened rather than dropped
        when even they will not fit.
        """
        messages = list(self.conversations.get(key, []))
        context: list[dict] = []
        used = 0

        if system_prompt.strip():
            context.append({"role": "system", "content": system_prompt.strip()})
            used += estimate_tokens(system_prompt)

        # -- attachments first, newest attachment winning the room ----------
        allowance = attachment_allowance(max(0, budget_tokens - used))
        pinned: dict[int, str] = {}
        pinned_used = 0
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not (message.get("pinned")
                    or carries_attachment(message.get("content", ""))):
                continue
            content = message.get("content", "")
            room = allowance - pinned_used
            if room <= 0:
                continue
            if estimate_tokens(content) > room:
                content = shorten_attachment(content, room)
            pinned[index] = content
            pinned_used += estimate_tokens(content)
        used += pinned_used

        # -- then ordinary turns, newest first ------------------------------
        newest = len(messages) - 1
        kept = set(pinned)
        dropped = 0
        for index in range(newest, -1, -1):
            if index in pinned:
                continue
            cost = estimate_tokens(messages[index].get("content", ""))
            if index != newest and used + cost > budget_tokens:
                dropped += 1
                continue
            kept.add(index)
            used += cost

        if dropped:
            context.append({
                "role": "system",
                "content": (f"[{dropped} earlier message(s) were trimmed to fit "
                            f"the context budget.]"),
            })
        # Rebuilt as plain role/content pairs: "pinned" is our bookkeeping and
        # has no business being posted to the inference server.
        for index in sorted(kept):
            context.append({"role": messages[index].get("role", "user"),
                            "content": pinned.get(
                                index, messages[index].get("content", ""))})
        return context

    # -- persistence -------------------------------------------------------
    def to_dict(self, selected: Iterable[str] | None = None) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "saved_at": datetime.now().isoformat(),
            "conversations": self.conversations,
            "selected_models": list(selected or []),
        }

    def save(self, path, selected: Iterable[str] | None = None) -> Path:
        """Write the chat to disk.

        `selected` matters more than it looks: without it the file records no
        model tick-boxes, so reopening restores the conversations but not who
        was answering — and the context counter, which is per-model, reads
        near-zero under a screen full of restored transcript.
        """
        path = Path(path)
        payload = self.to_dict(selected)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        self.path = path
        return path

    @classmethod
    def load(cls, path) -> tuple[Session, list[str]]:
        path = Path(path)
        session, selected = cls.from_dict(
            json.loads(path.read_text(encoding="utf-8")))
        session.path = path
        return session, selected

    @classmethod
    def from_dict(cls, raw: dict) -> tuple[Session, list[str]]:
        """Rebuild a session from saved data, wherever that data came from.

        Shared with the recovery file so a crashed session is cleaned and
        pinned by exactly the same rules as one opened from disk.
        """
        conversations = raw.get("conversations", {})
        if not isinstance(conversations, dict):
            raise ValueError("This file does not look like a saved chat.")

        cleaned: dict[str, list[dict]] = {}
        for key, messages in conversations.items():
            if not isinstance(messages, list):
                continue
            server, model = split_key(key)
            good = []
            for m in messages:
                if not (isinstance(m, dict) and m.get("role") in ROLES
                        and isinstance(m.get("content"), str)):
                    continue
                message = {"role": m["role"], "content": m["content"]}
                # Chats saved before attachments were pinned still get the
                # protection, because the marker is in the text itself.
                if m.get("pinned") or (m["role"] == "user"
                                       and carries_attachment(m["content"])):
                    message["pinned"] = True
                good.append(message)
            cleaned[make_key(server, model)] = good

        session = cls(cleaned)
        selected = [make_key(*split_key(k))
                    for k in raw.get("selected_models", [])
                    if isinstance(k, str)]
        return session, selected

    # -- export ------------------------------------------------------------
    def transcript(self) -> str:
        lines: list[str] = []
        for key, messages in self.conversations.items():
            server, model = split_key(key)
            lines.append(f"=== {model} ({server}) ===")
            for message in messages:
                role = message.get("role", "")
                who = {"user": "You", "assistant": short_model(model)}.get(role, "System")
                lines.append(f"{who}: {message.get('content', '')}")
            lines.append("")
        return "\n".join(lines).strip()
