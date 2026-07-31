"""Learning mode: durable lessons carried between conversations.

When Learning is on, each exchange is mined for things worth remembering —
"shut the main valve before you open a P-trap", "in Python, `dict.setdefault`
avoids a KeyError branch" — and stored on disk.  Later chats retrieve whatever
looks relevant and hand it to the model as prior knowledge.

What matters here is restraint: a knowledge base that eats every passing fact
becomes noise, so we only keep things that would still be true next month.
Anything time-sensitive is rejected outright, using the same test the research
cache uses to decide what goes stale.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .cache import STOPWORDS, is_volatile, tokens

KNOWLEDGE_PATH = Path.home() / ".ai_chat_lab_knowledge.json"

MAX_LESSON_CHARS = 400
MAX_INJECTED = 5
MIN_RELEVANCE = 0.08
MIN_EXCHANGE_CHARS = 120       # below this there is nothing worth mining

NONE_MARKERS = {"none", "none.", "nothing", "n/a", "no lessons"}


@dataclass
class Lesson:
    topic: str
    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""
    uses: int = 0

    @property
    def haystack(self) -> str:
        return f"{self.topic} {self.text}"


# ------------------------------------------------------------------ prompts

def build_extraction_prompt(question: str, answer: str) -> str:
    return (
        "Read this exchange and pull out anything worth remembering for future, "
        "unrelated conversations.\n\n"
        "Keep only durable, reusable knowledge — techniques, rules of thumb, "
        "gotchas, how something works. Ignore anything tied to this moment "
        "(weather, prices, news, dates) and anything too obvious to be worth "
        "writing down.\n\n"
        "Write at most 3 lines, each formatted exactly as:\n"
        "Topic | the lesson in one sentence\n\n"
        "If there is nothing worth keeping, reply with just: NONE\n\n"
        f"User asked: {question[:800]}\n\n"
        f"Answer given: {answer[:2000]}\n\n"
        "Lessons:")


def parse_lessons(text: str) -> list[tuple[str, str]]:
    """Parse "Topic | lesson" lines out of a model's reply."""
    lessons: list[tuple[str, str]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
        if not line or line.lower() in NONE_MARKERS:
            continue
        if line.lower().startswith("lessons"):
            continue

        if "|" in line:
            topic, _, lesson = line.partition("|")
        elif ":" in line and len(line.split(":", 1)[0]) < 40:
            topic, _, lesson = line.partition(":")
        else:
            topic, lesson = "General", line

        topic = topic.strip().strip("*").title()[:40] or "General"
        lesson = lesson.strip().strip("*")[:MAX_LESSON_CHARS]
        if len(lesson) < 12:
            continue
        # a "lesson" about tonight's forecast is not worth keeping
        if is_volatile(lesson):
            continue
        lessons.append((topic, lesson))
    return lessons[:3]


def format_for_prompt(lessons: Sequence[Lesson]) -> str:
    if not lessons:
        return ""
    lines = [f"- ({lesson.topic}) {lesson.text}" for lesson in lessons]
    return ("Things you worked out in earlier conversations that may be "
            "relevant here:\n" + "\n".join(lines) +
            "\n\nUse them if they help. Don't mention that you were reminded.")


# ------------------------------------------------------------------ storage

class KnowledgeBase:
    """Lessons on disk, with keyword retrieval."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else KNOWLEDGE_PATH
        self.lessons: list[Lesson] = []
        self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.lessons = [
                    Lesson(**{k: v for k, v in item.items()
                              if k in Lesson.__dataclass_fields__})
                    for item in raw.get("lessons", [])
                    if isinstance(item, dict) and item.get("text")
                ]
        except (OSError, ValueError, TypeError):
            self.lessons = []

    def save(self) -> bool:
        try:
            self.path.write_text(
                json.dumps({"lessons": [asdict(x) for x in self.lessons]},
                           indent=2, ensure_ascii=False),
                encoding="utf-8")
            return True
        except OSError:
            return False

    # -- writing -----------------------------------------------------------
    def add(self, topic: str, text: str, source: str = "") -> Lesson | None:
        """Store a lesson unless we already know something very like it."""
        text = (text or "").strip()
        if len(text) < 12 or is_volatile(text):
            return None
        if self._duplicate(text):
            return None
        lesson = Lesson(topic=(topic or "General").strip()[:40],
                        text=text[:MAX_LESSON_CHARS], source=source[:200])
        self.lessons.append(lesson)
        self.save()
        return lesson

    def add_many(self, pairs: Sequence[tuple[str, str]],
                 source: str = "") -> list[Lesson]:
        added = [self.add(topic, text, source) for topic, text in pairs]
        return [lesson for lesson in added if lesson]

    def _duplicate(self, text: str) -> bool:
        new = tokens(text)
        if not new:
            return True
        for lesson in self.lessons:
            existing = tokens(lesson.text)
            if not existing:
                continue
            overlap = len(new & existing) / len(new | existing)
            if overlap > 0.7:
                return True
        return False

    # -- reading -----------------------------------------------------------
    def relevant(self, query: str, limit: int = MAX_INJECTED) -> list[Lesson]:
        """Lessons whose wording overlaps the question, best first."""
        wanted = tokens(query)
        if not wanted or not self.lessons:
            return []

        scored: list[tuple[float, Lesson]] = []
        for lesson in self.lessons:
            available = tokens(lesson.haystack)
            if not available:
                continue
            shared = wanted & available
            if not shared:
                continue
            score = len(shared) / len(wanted)
            # a topic-word match is a strong signal, so weight it
            if tokens(lesson.topic) & wanted:
                score += 0.25
            if score >= MIN_RELEVANCE:
                scored.append((score, lesson))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [lesson for _score, lesson in scored[:limit]]

    def recall(self, query: str, limit: int = MAX_INJECTED) -> list[Lesson]:
        """Like `relevant`, but counts the retrieval against each lesson."""
        found = self.relevant(query, limit)
        for lesson in found:
            lesson.uses += 1
        if found:
            self.save()
        return found

    def topics(self) -> list[str]:
        return sorted({lesson.topic for lesson in self.lessons})

    # -- management --------------------------------------------------------
    def remove(self, lesson_id: str) -> bool:
        before = len(self.lessons)
        self.lessons = [x for x in self.lessons if x.id != lesson_id]
        if len(self.lessons) != before:
            self.save()
            return True
        return False

    def clear(self) -> None:
        self.lessons = []
        self.save()

    def __len__(self) -> int:
        return len(self.lessons)


def worth_learning_from(question: str, answer: str) -> bool:
    """Skip greetings, one-liners and anything obviously time-bound."""
    if len((question or "") + (answer or "")) < MIN_EXCHANGE_CHARS:
        return False
    if len((answer or "").split()) < 20:
        return False
    return True


__all__ = ["KnowledgeBase", "Lesson", "build_extraction_prompt",
           "format_for_prompt", "parse_lessons", "worth_learning_from",
           "STOPWORDS"]
