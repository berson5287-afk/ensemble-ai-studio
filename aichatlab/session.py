"""Conversation state, persistence and context budgeting.

Each model gets its own message history, keyed "<server>::<model>".  The
interesting part is `build_context`: model context windows are finite, and a
long chat — or one big attached document — will silently overflow them.  We
estimate tokens, keep the most recent turns that fit, and tell the model
plainly that earlier turns were dropped.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

ROLES = ("user", "assistant", "system")
SCHEMA_VERSION = 2

# Rough but stable: English averages ~4 characters per token across models.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


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

    def add(self, key: str, role: str, content: str) -> dict:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        message = {"role": role, "content": content}
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
        """
        messages = list(self.conversations.get(key, []))
        context: list[dict] = []
        used = 0

        if system_prompt.strip():
            context.append({"role": "system", "content": system_prompt.strip()})
            used += estimate_tokens(system_prompt)

        kept: list[dict] = []
        dropped = 0
        for message in reversed(messages):
            cost = estimate_tokens(message.get("content", ""))
            if kept and used + cost > budget_tokens:
                dropped += 1
                continue
            kept.append(message)
            used += cost
        kept.reverse()

        if dropped:
            context.append({
                "role": "system",
                "content": (f"[{dropped} earlier message(s) were trimmed to fit "
                            f"the context budget.]"),
            })
        context.extend(kept)
        return context

    # -- persistence -------------------------------------------------------
    def to_dict(self, selected: Iterable[str] | None = None) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "saved_at": datetime.now().isoformat(),
            "conversations": self.conversations,
            "selected_models": list(selected or []),
        }

    def save(self, path) -> Path:
        path = Path(path)
        payload = self.to_dict()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        self.path = path
        return path

    @classmethod
    def load(cls, path) -> tuple[Session, list[str]]:
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        conversations = raw.get("conversations", {})
        if not isinstance(conversations, dict):
            raise ValueError("This file does not look like a saved chat.")

        cleaned: dict[str, list[dict]] = {}
        for key, messages in conversations.items():
            if not isinstance(messages, list):
                continue
            server, model = split_key(key)
            good = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if isinstance(m, dict)
                and m.get("role") in ROLES
                and isinstance(m.get("content"), str)
            ]
            cleaned[make_key(server, model)] = good

        session = cls(cleaned)
        session.path = path
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
