"""Deciding whether a question actually needs the web before searching it.

With web research switched on, every message used to go straight to the search
engine as literal text. Attach a folder and ask "take a look at my app and
give me some tips" and the app dutifully searched for *those words*, found
nothing useful, and spent a round-trip proving it.

The fix is the one a person would apply: look at what you already have first.
A model is shown the question and an inventory of the material in hand, and
answers one of two ways — I can answer from this, or I need to look up X. When
it does need a lookup it supplies the query itself, which is a far better
search than the raw message ("Ollama num_ctx default value", not "please take
a look at my app and don't edit anything").

Pure module: prompt construction and reply parsing only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

ANSWER_TOKEN = "ANSWER"
SEARCH_TOKEN = "SEARCH:"

# Models love to wrap a one-word answer in explanation.  Find the decision
# anywhere in the reply rather than demanding it be the whole thing.
# `(.*)` rather than `(.+)`: a bare "SEARCH:" with no query still has to be
# recognised as a request to search, so it can fall back to the raw question
# instead of being misread as an unclear reply and silently skipped.
SEARCH_RE = re.compile(r"\bSEARCH\s*:\s*(.*)", re.IGNORECASE)
ANSWER_RE = re.compile(r"\bANSWER\b", re.IGNORECASE)

MAX_QUERY_CHARS = 160


@dataclass(frozen=True)
class Triage:
    """The verdict: search for `query`, or answer from what we already have."""

    needs_search: bool
    query: str = ""
    reason: str = ""

    @property
    def note(self) -> str:
        if self.needs_search:
            return f"🔍 Checking the web for “{self.query}”"
        return "📄 Working from what you've given me — no search needed"


def describe_material(attachments: Sequence[dict] | None = None,
                      history_turns: int = 0) -> str:
    """An inventory of what the model can already see, for the prompt."""
    lines = []
    for attachment in attachments or []:
        name = str(attachment.get("name", "")).strip()
        if not name:
            continue
        size = len(str(attachment.get("content", "")))
        kind = "folder" if name.endswith("/") else "file"
        lines.append(f"- {kind}: {name} (~{max(1, size // 4):,} tokens of text)")
    if history_turns:
        lines.append(f"- {history_turns} earlier message(s) in this conversation")
    return "\n".join(lines)


def build_prompt(question: str, material: str) -> str:
    return (
        "Decide whether answering this question needs a web search.\n\n"
        "You already have access to the material listed below. Searching is "
        "only worth it for things that material cannot tell you: current "
        "events, live prices, recent releases, or facts about the outside "
        "world.\n\n"
        "Questions about the user's own files, code or earlier conversation "
        "almost never need a search.\n\n"
        f"Material you already have:\n{material or '- nothing attached'}\n\n"
        f"Question: {question}\n\n"
        "Reply with exactly one line, nothing else:\n"
        f"  {ANSWER_TOKEN}            — if the material above is enough\n"
        f"  {SEARCH_TOKEN} <query>    — if you need to look something up, "
        "where <query> is what you would type into a search engine\n\n"
        "Decision:")


def parse(reply: str, fallback_query: str = "") -> Triage:
    """Read the model's verdict, defaulting to *not* searching when unclear.

    Defaulting to no-search is deliberate. A needless search costs a slow
    round-trip and pollutes the prompt with irrelevant sources; a needlessly
    skipped one costs a follow-up message. The cheap mistake is the one to
    make when the model has been ambiguous.
    """
    text = (reply or "").strip()
    if not text:
        return Triage(False, reason="no reply from the triage model")

    match = SEARCH_RE.search(text)
    if match:
        lines = match.group(1).strip().splitlines()
        query = (lines[0] if lines else "").strip().strip('"').strip("'").strip()
        if len(query) > MAX_QUERY_CHARS:
            query = query[:MAX_QUERY_CHARS].rsplit(" ", 1)[0]
        if len(query) > 2:
            return Triage(True, query=query, reason="model asked for a lookup")
        if fallback_query:
            return Triage(True, query=fallback_query,
                          reason="model asked for a lookup but gave no query")
        return Triage(False, reason="model asked for a lookup but gave no query")

    if ANSWER_RE.search(text):
        return Triage(False, reason="model can answer from what it has")
    return Triage(False, reason="unclear reply, so no search")


def worth_triaging(question: str, attachments: Sequence[dict] | None,
                   history: Sequence[dict] | None = None) -> bool:
    """Only spend a model call when there is something to weigh it against.

    With nothing attached and no history, there is no "material" for the model
    to prefer over a search, so triage can only ever say yes — and paying for
    a round-trip to be told what we assumed is not an improvement.
    """
    if not (question or "").strip():
        return False
    return bool(attachments) or len(list(history or [])) >= 2
