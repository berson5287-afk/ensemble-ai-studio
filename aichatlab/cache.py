"""Caching for web research.

Searching the same thing twice in one conversation is wasted time — several
seconds of search plus page fetches for an answer we already have.  But some
answers go stale fast: cached weather from this morning is simply wrong by the
evening, while "how do I sweat a copper pipe" is good indefinitely.  So the
time-to-live depends on what was asked.

The cache lives in a temp file.  Entries added during a run are thrown away
when the app closes unless the chat was saved, on the reasoning that a
throwaway conversation shouldn't leave anything behind.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TTL_MINUTES = 360          # six hours for ordinary questions
VOLATILE_TTL_SECONDS = 900         # fifteen minutes for anything time-sensitive
NEAR_MATCH_THRESHOLD = 0.8         # token overlap needed to reuse another query

CACHE_PATH = Path(tempfile.gettempdir()) / "ai_chat_lab_research_cache.json"

# Questions whose answer changes through the day.
VOLATILE = re.compile(
    r"\b(weather|forecast|temperature|rain|snow|wind|humidity|storm|"
    r"today|tonight|tomorrow|now|currently|current|this (morning|afternoon|"
    r"evening|weekend|week)|price|cost|stock|share|score|game|news|headline|"
    r"latest|breaking|open|closed|traffic|flight|delay)\b",
    re.IGNORECASE)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "in", "on",
    "at", "to", "for", "and", "or", "but", "what", "whats", "how", "when",
    "where", "why", "who", "which", "it", "its", "this", "that", "do", "does",
    "did", "can", "could", "would", "should", "please", "me", "my", "i",
    "you", "your", "about", "with", "from", "there", "going",
}


def normalise(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def tokens(query: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalise(query))
    return {word for word in words if word not in STOPWORDS and len(word) > 1}


def similarity(left: str, right: str) -> float:
    """Jaccard overlap of significant words — 1.0 means the same question."""
    first, second = tokens(left), tokens(right)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def is_volatile(query: str) -> bool:
    return bool(VOLATILE.search(query or ""))


def ttl_for(query: str, base_minutes: int = DEFAULT_TTL_MINUTES) -> int:
    """Seconds a result for this query stays usable."""
    base = max(60, int(base_minutes) * 60)
    return min(base, VOLATILE_TTL_SECONDS) if is_volatile(query) else base


@dataclass
class CacheHit:
    query: str
    block: str
    age_seconds: float
    exact: bool

    @property
    def age_phrase(self) -> str:
        minutes = int(self.age_seconds // 60)
        if minutes < 1:
            return "just now"
        if minutes == 1:
            return "a minute ago"
        if minutes < 60:
            return f"{minutes} minutes ago"
        hours = minutes // 60
        return "an hour ago" if hours == 1 else f"{hours} hours ago"


class ResearchCache:
    """A small JSON-backed cache of research blocks, keyed by query."""

    def __init__(self, path: Path | None = None,
                 base_ttl_minutes: int = DEFAULT_TTL_MINUTES,
                 clock=time.time) -> None:
        self.path = Path(path) if path else CACHE_PATH
        self.base_ttl_minutes = base_ttl_minutes
        self._clock = clock
        self.entries: dict[str, dict] = {}
        self.session_keys: set[str] = set()
        self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                entries = raw.get("entries", {})
                if isinstance(entries, dict):
                    self.entries = {
                        key: value for key, value in entries.items()
                        if isinstance(value, dict) and "block" in value
                    }
        except (OSError, ValueError):
            self.entries = {}       # a corrupt cache is not worth complaining about

    def save(self) -> bool:
        try:
            self.path.write_text(
                json.dumps({"entries": self.entries}, ensure_ascii=False),
                encoding="utf-8")
            return True
        except OSError:
            return False

    # -- reading -----------------------------------------------------------
    def get(self, query: str) -> CacheHit | None:
        """Return usable cached research, exact match first then near match."""
        now = self._clock()
        key = normalise(query)

        entry = self.entries.get(key)
        if entry and not self._expired(entry, now):
            return CacheHit(query=entry.get("query", query),
                            block=entry["block"],
                            age_seconds=now - entry.get("at", now),
                            exact=True)

        best, best_score = None, 0.0
        for candidate_key, candidate in self.entries.items():
            if self._expired(candidate, now):
                continue
            score = similarity(key, candidate_key)
            if score >= NEAR_MATCH_THRESHOLD and score > best_score:
                best, best_score = candidate, score
        if best is not None:
            return CacheHit(query=best.get("query", query), block=best["block"],
                            age_seconds=now - best.get("at", now), exact=False)
        return None

    def _expired(self, entry: dict, now: float) -> bool:
        ttl = entry.get("ttl") or ttl_for(entry.get("query", ""),
                                          self.base_ttl_minutes)
        return (now - entry.get("at", 0)) > ttl

    # -- writing -----------------------------------------------------------
    def put(self, query: str, block: str) -> None:
        if not query or not block:
            return
        key = normalise(query)
        self.entries[key] = {
            "query": query,
            "block": block,
            "at": self._clock(),
            "ttl": ttl_for(query, self.base_ttl_minutes),
        }
        self.session_keys.add(key)
        self.prune()
        self.save()

    def prune(self) -> None:
        now = self._clock()
        for key in [k for k, v in self.entries.items() if self._expired(v, now)]:
            self.entries.pop(key, None)
            self.session_keys.discard(key)

    # -- lifecycle ---------------------------------------------------------
    def discard_session(self) -> None:
        """Drop everything this run added, leaving earlier entries alone."""
        for key in self.session_keys:
            self.entries.pop(key, None)
        self.session_keys.clear()
        if self.entries:
            self.save()
        else:
            self.delete_file()

    def keep_session(self) -> None:
        self.session_keys.clear()
        self.save()

    def delete_file(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass

    def clear(self) -> None:
        self.entries.clear()
        self.session_keys.clear()
        self.delete_file()

    def __len__(self) -> int:
        return len(self.entries)
