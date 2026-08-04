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

# Pessimistic prompt-reading speed, used until this machine has been
# measured.  Being wrong slow costs patience; being wrong fast cancels
# a request that was going to succeed.
SLOW_PROMPT_RATE = 15.0        # tokens per second

# `load` prefers whatever is in the file over the default, so raising a
# default does nothing for anyone who has already run the app.  This one
# value was never a deliberate choice by anybody — it was ten minutes of
# silence, which a 32B model reading 24,000 tokens exceeds — so a settings
# file still carrying it gets moved on.
SUPERSEDED_TIMEOUT = 600
TIMEOUT_HEADROOM = 180         # seconds

# num_ctx is part of a model's runner configuration, so changing it makes
# Ollama unload and reload the model — twenty-odd gigabytes for a 32B, on
# every message.  Sizing the window to each prompt exactly would do that, so
# the answer is rounded up into a handful of stable steps instead.
# Steps of 4k: coarse enough that a growing conversation only crosses a
# boundary occasionally, fine enough that a 24k prompt does not reserve
# 32k of key/value cache it will never use.
WINDOW_STEPS = (2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672,
                32768, 40960, 49152, 65536, 98304, 131072)

DEFAULTS: dict[str, Any] = {
    "local_ip": "127.0.0.1",
    "local_port": "11434",
    "host_ip": "",
    "host_port": "11434",
    "temperature": 0.7,
    "context_budget_tokens": 6000,
    "request_timeout": 1800,   # silence allowed before giving up
    "system_prompt": "",
    "speed_quality": 50,       # 0 = fastest answers, 100 = most detailed
    "searxng_url": "",         # e.g. http://192.168.1.20:8080
    "conversational": True,    # talk like a person, not like a reference book
    "smart_followups": True,   # resolve "what's its 0-60?" against the thread
    "learning": False,         # remember durable lessons between chats
    "research_cache": True,    # reuse recent research instead of re-searching
    "research_cache_ttl_minutes": 360,
    "auto_compact": True,      # fold old turns into a digest before they fall off
    "assess_before_search": True,   # read what we have before searching the web
    "review_popup": True,      # show proposed changes without waiting for a click
    # A described change is re-asked for automatically. The second pass
    # writes nothing — the review window still gates the disk — so the
    # click it replaces was only ever costing the user their request.
    "auto_realise": True,
    # Ask for one change, let it be decided, then ask for the next.
    # Fetching them all first costs minutes per change before
    # anything is shown, and pays for every one even when the
    # first makes it obvious the run is going nowhere.
    "one_at_a_time": True,
    # A broad wish ("make it faster") becomes a menu of small,
    # verified, one-place candidates to pick from, because sending
    # it straight to a local model breaks things four times out of
    # four and the menu shape lands.
    "suggest_menu": True,
    # Lets a script drive this window (aichatlab/bridge.py). Off by default,
    # and a real setting rather than only an environment variable because the
    # ordinary way to start this app is to double-click it, where there is
    # nowhere to put an environment variable.
    "control_bridge": False,
    # Ceiling on the num_ctx we ask Ollama for.  Context costs VRAM, so a
    # folder attachment that pushes the budget to 120k should not silently try
    # to allocate a 120k window on a laptop GPU.
    "max_context_window": 32768,
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
        if settings.values.get("request_timeout") == SUPERSEDED_TIMEOUT:
            settings.values["request_timeout"] = DEFAULTS["request_timeout"]
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

    def sampling_options(self, prompt_tokens: int = 0) -> dict[str, Any]:
        """Options passed straight through to Ollama's `options` field."""
        try:
            temperature = float(self.get("temperature", 0.7))
        except (TypeError, ValueError):
            temperature = 0.7
        options: dict[str, Any] = {"temperature": temperature}
        _label, num_predict, _directive = self.profile()
        if num_predict > 0:
            options["num_predict"] = num_predict
        options["num_ctx"] = self.context_window(prompt_tokens)
        return options

    def timeout_for(self, prompt_tokens: int = 0,
                    prompt_rate: float | None = None) -> int:
        """How long to allow with no bytes at all before giving up.

        This is not a total time limit — it is how long the server may say
        nothing.  While a model reads its prompt it says exactly nothing, so
        the limit has to cover that, and a fixed ten minutes does not: a 32B
        model reading 24,000 tokens can spend longer than that before the
        first word, and the request was being cancelled while the model was
        still working perfectly well.

        `prompt_rate` comes from the run log when this machine has been
        measured; the fallback is deliberately pessimistic.
        """
        base = int(self.get("request_timeout", 1800) or 1800)
        tokens = max(0, int(prompt_tokens))
        if not tokens:
            return base
        rate = prompt_rate if prompt_rate and prompt_rate > 0 else SLOW_PROMPT_RATE
        return max(base, int(tokens / rate) + TIMEOUT_HEADROOM)

    def effective_budget(self) -> int:
        """The budget we actually trim to — never larger than the window.

        A budget above `max_context_window` is not ambitious, it is broken:
        we stop trimming at 96,000 tokens, ask the server for 32,768, and the
        server discards the difference itself, oldest-first and without any of
        the care `build_context` takes.  Clamping here means the two numbers
        can never disagree, however the settings file got that way.
        """
        try:
            budget = max(500, int(self.get("context_budget_tokens", 6000) or 6000))
        except (TypeError, ValueError):
            budget = 6000
        _label, num_predict, _directive = self.profile()
        reply = 1024 if num_predict < 0 else num_predict
        return max(500, min(budget, self.context_window() - reply - 512))

    def context_window(self, prompt_tokens: int = 0) -> int:
        """The num_ctx to ask Ollama for, given our own trimming budget.

        Sized to what this prompt actually needs when that is known.  The
        window is not free — its key/value cache is allocated next to the
        weights — so asking for the full budget on every short question
        reserves gigabytes to hold a conversation that does not exist yet.

        Trimming the conversation to 6,000 tokens is pointless if the model is
        only given a 4,096-token window — Ollama's default — because the
        server then drops the oldest part of a prompt we had already decided
        was small enough to keep.  The budget and the window have to agree, so
        the window is derived from the budget plus room for the reply.
        """
        try:
            budget = int(self.get("context_budget_tokens", 6000) or 6000)
        except (TypeError, ValueError):
            budget = 6000
        _label, num_predict, _directive = self.profile()
        reply = 1024 if num_predict < 0 else num_predict
        wanted = min(budget, int(prompt_tokens)) if prompt_tokens else budget
        needed = wanted + reply + 512
        try:
            ceiling = int(self.get("max_context_window", 32768) or 32768)
        except (TypeError, ValueError):
            ceiling = 32768
        ceiling = max(2048, ceiling)
        for step in WINDOW_STEPS:
            if step >= needed:
                return min(step, ceiling)
        return ceiling

    def style_directive(self) -> str:
        """The slider's brevity/thoroughness instruction, if any."""
        return self.profile()[2]
