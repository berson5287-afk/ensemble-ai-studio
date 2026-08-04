"""A cache of what each file is, so the work is done once per version.

Summarising and embedding a folder is the expensive part of retrieval — on a
local 8B model, a few hundred files is minutes of work, and paying it on every
question would be worse than the problem it solves.  But almost nothing
changes between one question and the next, so almost all of that work is
repeated for no reason.

The cache is keyed on the *content*, not the path or the timestamp.  Renaming
a file, touching it, or checking it out again on another machine should all
cost nothing, and a file that genuinely changed should be the only one that
does.  A hash makes all four cases fall out for free.

Embeddings additionally record which model produced them.  Vectors from two
different embedding models are not comparable, and silently mixing them
produces a ranking that is subtly wrong in a way nobody would ever trace back
to here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .projectmemory import directory

INDEX = "index.json"
SCHEMA = 1

# A summary that is longer than this is not a summary.
MAX_SUMMARY_CHARS = 240


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class Entry:
    """What we know about one file, and which version of it we knew it for."""

    name: str
    digest: str
    size: int = 0
    summary: str = ""
    vector: list = field(default_factory=list)
    embedded_with: str = ""

    def matches(self, digest: str) -> bool:
        return bool(self.digest) and self.digest == digest

    def usable_vector(self, model: str) -> list:
        """Vectors from a different model are not comparable — refuse them."""
        if self.vector and self.embedded_with == model:
            return list(self.vector)
        return []


class Index:
    """Per-file knowledge for one project, persisted beside it."""

    def __init__(self, entries: dict | None = None) -> None:
        self.entries: dict = entries or {}

    # -- lookup ------------------------------------------------------------
    def get(self, name: str, digest: str) -> Entry | None:
        entry = self.entries.get(name)
        return entry if entry is not None and entry.matches(digest) else None

    def put(self, entry: Entry) -> None:
        self.entries[entry.name] = entry

    def summary_for(self, name: str, digest: str) -> str:
        entry = self.get(name, digest)
        return entry.summary if entry else ""

    def vector_for(self, name: str, digest: str, model: str) -> list:
        entry = self.get(name, digest)
        return entry.usable_vector(model) if entry else []

    # -- what still needs doing -------------------------------------------
    def needs_summary(self, files) -> list:
        """(name, text) for every file whose summary is missing or stale."""
        return [(name, text) for name, text in files
                if not self.summary_for(name, content_hash(text))]

    def needs_embedding(self, files, model: str) -> list:
        if not model:
            return []
        return [(name, text) for name, text in files
                if not self.vector_for(name, content_hash(text), model)]

    def record_summary(self, name: str, text: str, summary: str) -> None:
        digest = content_hash(text)
        existing = self.entries.get(name)
        if existing is not None and existing.matches(digest):
            existing.summary = summary[:MAX_SUMMARY_CHARS]
            return
        # The file changed, so anything we knew about the old version — the
        # vector especially — describes a file that no longer exists.
        self.put(Entry(name=name, digest=digest, size=len(text),
                       summary=summary[:MAX_SUMMARY_CHARS]))

    def record_vector(self, name: str, text: str, vector, model: str) -> None:
        digest = content_hash(text)
        existing = self.entries.get(name)
        if existing is not None and existing.matches(digest):
            existing.vector = list(vector)
            existing.embedded_with = model
            return
        self.put(Entry(name=name, digest=digest, size=len(text),
                       vector=list(vector), embedded_with=model))

    def forget_missing(self, names) -> int:
        """Drop files that are no longer in the folder."""
        keep = set(names)
        gone = [name for name in self.entries if name not in keep]
        for name in gone:
            del self.entries[name]
        return len(gone)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"schema": SCHEMA,
                "entries": [asdict(e) for e in self.entries.values()]}

    @classmethod
    def from_dict(cls, raw: dict) -> Index:
        entries = {}
        for item in (raw or {}).get("entries", []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            try:
                entry = Entry(
                    name=str(item["name"]),
                    digest=str(item.get("digest", "")),
                    size=int(item.get("size") or 0),
                    summary=str(item.get("summary", ""))[:MAX_SUMMARY_CHARS],
                    vector=[float(x) for x in item.get("vector") or []],
                    embedded_with=str(item.get("embedded_with", "")))
            except (TypeError, ValueError):
                continue
            entries[entry.name] = entry
        return cls(entries)


def path_for(folder) -> Path:
    return directory(folder) / INDEX


def load(folder) -> Index:
    """Read the cache, treating any damage as "we know nothing yet"."""
    try:
        raw = json.loads(path_for(folder).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Index()
    return Index.from_dict(raw)


def save(folder, index: Index) -> bool:
    target = path_for(folder)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".part")
        temporary.write_text(json.dumps(index.to_dict()), encoding="utf-8")
        temporary.replace(target)
        return True
    except OSError:
        return False


SUMMARY_PROMPT = """\
In one sentence of at most 25 words, say what this file is for. Name what it
does, not what language it is in. No preamble, no file name, just the sentence.

{name}:

{body}
"""

# Enough to see what a file is; the whole file is not needed to say so, and
# sending it would make indexing cost as much as not indexing.
SUMMARY_SAMPLE_CHARS = 4_000


def build_summary_prompt(name: str, text: str) -> str:
    body = (text or "")[:SUMMARY_SAMPLE_CHARS]
    return SUMMARY_PROMPT.format(name=name, body=body)


def clean_summary(reply: str) -> str:
    """Take the first real sentence and drop the model's throat-clearing."""
    text = " ".join((reply or "").split())
    for opener in ("This file ", "The file ", "Sure, ", "Certainly, "):
        if text.startswith(opener):
            text = text[len(opener):]
    return text[:MAX_SUMMARY_CHARS].strip()


def embedding_text(name: str, summary: str, text: str,
                   limit: int = 2_000) -> str:
    """What actually gets embedded for a file.

    The name and the summary go first because they survive truncation, and a
    path like `ui/folder_dialog.py` is a strong signal that the first two
    thousand characters of the file might not contain.
    """
    head = f"{name}\n{summary}\n" if summary else f"{name}\n"
    return head + (text or "")[:limit]
