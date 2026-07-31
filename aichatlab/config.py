"""User settings: server addresses, sampling options and context budget.

Settings live in a single JSON file in the user's home directory so the app
remembers servers between launches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path.home() / ".ai_chat_lab_settings.json"

SERVERS = ("local", "host")

DEFAULTS: dict[str, Any] = {
    "local_ip": "127.0.0.1",
    "local_port": "11434",
    "host_ip": "",
    "host_port": "11434",
    "temperature": 0.7,
    "context_budget_tokens": 6000,
    "request_timeout": 600,
    "system_prompt": "",
    "speed_quality": 50,       # 0 = fastest answers, 100 = most detailed
    "searxng_url": "",         # e.g. http://192.168.1.20:8080
    "conversational": True,    # talk like a person, not like a reference book
    "smart_followups": True,   # resolve "what's its 0-60?" against the thread
    "learning": False,         # remember durable lessons between chats
    "research_cache": True,    # reuse recent research instead of re-searching
    "research_cache_ttl_minutes": 360,
}

# Without this, models answer a chat message the way they'd write an
# encyclopedia entry: restating the question, defining terms you obviously
# already know, and announcing what they're about to do.
CONVERSATIONAL_PROMPT = (
    "You are talking with someone in a live chat, so write the way a "
    "knowledgeable person talks.\n"
    "- Answer the actual question first. Never restate it back to them.\n"
    "- Don't open with 'The term X refers to' or define words they clearly "
    "already understand.\n"
    "- Don't narrate what you're about to do — just do it.\n"
    "- Use plain language and contractions. Keep paragraphs short.\n"
    "- If the question depends on something earlier in the conversation, "
    "answer it in that context rather than starting over.\n"
    "- If it's genuinely ambiguous, ask one short clarifying question instead "
    "of guessing."
)

# The speed↔quality slider maps to a response-length cap (Ollama's
# num_predict — fewer tokens to generate is the single biggest speed lever)
# plus a style directive prepended to the system prompt.  -1 means uncapped.
PROFILES = (
    # (upper bound, label, num_predict, directive)
    (19, "Fastest", 192,
     "Answer in a few sentences at most. No preamble, no filler."),
    (39, "Quick", 448,
     "Be concise. Short paragraphs, only the essentials."),
    (59, "Balanced", 1024, ""),
    (79, "Detailed", 2048,
     "Give a detailed, well-structured answer with reasoning."),
    (100, "Max quality", -1,
     "Be thorough and comprehensive. Cover nuances, edge cases and "
     "trade-offs, and explain your reasoning."),
)


class Settings:
    """A small dict-backed settings object with load/save helpers."""

    def __init__(self, values: dict[str, Any] | None = None,
                 path: Path | None = None) -> None:
        self.path = Path(path) if path else SETTINGS_PATH
        self.values: dict[str, Any] = dict(DEFAULTS)
        if values:
            self.values.update(values)

    # -- dict-ish access ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        settings = cls(path=path)
        try:
            if settings.path.exists():
                raw = json.loads(settings.path.read_text(encoding="utf-8"))
                for key, default in DEFAULTS.items():
                    if key in raw and isinstance(raw[key], type(default)):
                        settings.values[key] = raw[key]
        except (OSError, ValueError):
            pass  # a corrupt settings file should never stop the app starting
        return settings

    def save(self) -> bool:
        try:
            self.path.write_text(json.dumps(self.values, indent=2),
                                 encoding="utf-8")
            return True
        except OSError:
            return False

    # -- derived values ----------------------------------------------------
    def base_url(self, server: str) -> str:
        """Return http://ip:port for a server, or "" when unconfigured."""
        ip = str(self.get(f"{server}_ip", "")).strip()
        port = str(self.get(f"{server}_port", "")).strip() or "11434"
        if not ip:
            return ""
        if ip.startswith(("http://", "https://")):
            return f"{ip.rstrip('/')}:{port}" if ":" not in ip.split("//")[1] else ip.rstrip("/")
        return f"http://{ip}:{port}"

    def profile(self) -> tuple[str, int, str]:
        """(label, num_predict, directive) for the speed↔quality slider."""
        try:
            value = int(self.get("speed_quality", 50))
        except (TypeError, ValueError):
            value = 50
        value = max(0, min(100, value))
        for upper, label, num_predict, directive in PROFILES:
            if value <= upper:
                return label, num_predict, directive
        return PROFILES[-1][1:]

    def sampling_options(self) -> dict[str, Any]:
        """Options passed straight through to Ollama's `options` field."""
        try:
            temperature = float(self.get("temperature", 0.7))
        except (TypeError, ValueError):
            temperature = 0.7
        options: dict[str, Any] = {"temperature": temperature}
        _label, num_predict, _directive = self.profile()
        if num_predict > 0:
            options["num_predict"] = num_predict
        return options

    def style_directive(self) -> str:
        """The slider's brevity/thoroughness instruction, if any."""
        return self.profile()[2]
