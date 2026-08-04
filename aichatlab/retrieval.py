"""Sending the part of the project that matters, instead of all of it.

Pasting a token-capped slice of a folder into every prompt has a ceiling you
cannot raise your way out of.  An 8B model with a 28,000-token window will
never hold a real codebase, and a bigger window costs VRAM at roughly a
gigabyte per ten thousand tokens — so the answer is not to send more, it is to
send less and choose better.

Two rankers live here, and the fallback is not an afterthought.

*Embeddings* handle the case that matters most: you ask "where do we decide
whether a file is safe to read" and the relevant file never uses the word
"safe".  Keyword search cannot bridge that; a vector model can.

*Keyword ranking* is what runs when no embedding model is installed, or when
the server refuses.  It is genuinely worse at paraphrase and genuinely better
at exact identifiers — `read_source_guarded` is a strong signal that gets
diluted by an embedding of the whole file — so the two are blended when both
are available rather than treating one as a degraded version of the other.

Nothing here talks to a UI or a network directly.  The client is passed in,
which is what lets the ranking be tested without either.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")

# Identifiers are the strongest signal a code question carries, and splitting
# them means `read_source_guarded` also matches a question about "guarded
# reads".  Both forms are kept.
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

STOP = {
    "the", "and", "for", "that", "this", "with", "you", "your", "from", "are",
    "was", "not", "but", "have", "has", "can", "does", "did", "what", "how",
    "why", "when", "where", "which", "who", "our", "its", "it", "is", "in",
    "of", "to", "a", "an", "on", "at", "by", "or", "be", "as", "if", "we",
    "do", "so", "than", "then", "there", "into", "about", "would", "should",
    "could", "any", "all", "some", "more", "most", "please", "me", "my",
}

# Below this, a file is not relevant enough to be worth its tokens.
MIN_SCORE = 0.02

# How much of the blend comes from vectors when both rankers have an opinion.
VECTOR_WEIGHT = 0.65


def terms(text: str) -> list[str]:
    """Words worth matching on, with identifiers split as well as kept."""
    found: list[str] = []
    for match in WORD.findall(text or ""):
        lowered = match.lower()
        if lowered not in STOP:
            found.append(lowered)
        for piece in CAMEL.sub(" ", match).replace("_", " ").split():
            piece = piece.lower()
            if len(piece) > 2 and piece not in STOP and piece != lowered:
                found.append(piece)
    return found


@dataclass
class Chunk:
    """One retrievable piece — usually a file, sometimes part of a big one."""

    name: str
    text: str
    summary: str = ""
    vector: list[float] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        # The measured estimator, not characters-over-four.  This was the
        # last hiding place of the old ratio, and it let a "budgeted"
        # selection build a 105,000-character block for a 32,768-token
        # window — which Ollama answered by silently dropping the front of
        # the prompt, system prompt and all.
        from .session import estimate_tokens

        return estimate_tokens(self.text)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    left = math.sqrt(sum(x * x for x in a))
    right = math.sqrt(sum(y * y for y in b))
    if left <= 0 or right <= 0:
        return 0.0
    return dot / (left * right)


def keyword_scores(question: str, chunks) -> dict:
    """BM25-ish ranking: rare words count for more, long files are not
    rewarded for merely being long."""
    wanted = terms(question)
    if not wanted or not chunks:
        return {}

    documents = []
    for chunk in chunks:
        documents.append(terms(f"{chunk.name} {chunk.summary} {chunk.text}"))
    total = len(documents)
    average = sum(len(d) for d in documents) / total if total else 1.0

    # How many documents each wanted term appears in.
    appears = {}
    for term in set(wanted):
        appears[term] = sum(1 for d in documents if term in d) or 0

    k1, b = 1.5, 0.75
    scores = {}
    for chunk, document in zip(chunks, documents):
        if not document:
            continue
        counts: dict = {}
        for word in document:
            counts[word] = counts.get(word, 0) + 1
        length = len(document)
        score = 0.0
        for term in set(wanted):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            # Rare terms are the informative ones.
            idf = math.log(1 + (total - appears[term] + 0.5)
                           / (appears[term] + 0.5))
            score += idf * (frequency * (k1 + 1)) / (
                frequency + k1 * (1 - b + b * length / average))
        if score > 0:
            scores[chunk.name] = score
    return _normalise(scores)


def vector_scores(question_vector: list[float], chunks) -> dict:
    if not question_vector:
        return {}
    scores = {}
    for chunk in chunks:
        if chunk.vector:
            similarity = cosine(question_vector, chunk.vector)
            if similarity > 0:
                scores[chunk.name] = similarity
    return _normalise(scores)


def _normalise(scores: dict) -> dict:
    """Scale to 0–1 so two rankers on different scales can be blended."""
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0:
        return {}
    return {name: value / top for name, value in scores.items()}


def blend(keyword: dict, vector: dict,
          vector_weight: float = VECTOR_WEIGHT) -> dict:
    """Combine the two rankings, using whichever are available.

    Neither is a degraded version of the other: vectors find the file that
    never uses your words, keywords find the exact identifier that an
    embedding of a whole file dilutes into nothing.
    """
    if not vector:
        return keyword
    if not keyword:
        return vector
    names = set(keyword) | set(vector)
    weight = min(1.0, max(0.0, vector_weight))
    return {name: weight * vector.get(name, 0.0)
            + (1 - weight) * keyword.get(name, 0.0)
            for name in names}


@dataclass(frozen=True)
class Hit:
    name: str
    score: float
    tokens: int


def rank(question: str, chunks, question_vector=None,
         vector_weight: float = VECTOR_WEIGHT) -> list[Hit]:
    """Every chunk worth considering, best first."""
    scores = blend(keyword_scores(question, chunks),
                   vector_scores(question_vector or [], chunks),
                   vector_weight)
    by_name = {chunk.name: chunk for chunk in chunks}
    hits = [Hit(name, score, by_name[name].tokens)
            for name, score in scores.items()
            if score >= MIN_SCORE and name in by_name]
    return sorted(hits, key=lambda hit: (-hit.score, hit.name))


# Identifier pieces this generic appear in every codebase, so sharing one
# says nothing about whether the question is about *this* code.
GENERIC_IDENTIFIER = {
    "get", "set", "add", "run", "new", "old", "name", "text", "data", "value",
    "item", "list", "dict", "key", "path", "file", "line", "self", "none",
    "true", "false", "main", "init", "args", "kwargs", "result", "return",
    "print", "test", "str", "int", "len", "type", "make", "read", "load",
}


def identifier_vocabulary(chunks) -> set:
    """Every identifier piece the code actually uses, lowercased.

    Strings and comments are deliberately included by `terms` for *ranking* —
    but for deciding whether a question is about the code at all, they are
    the trap.  "say hello" matches the greeting templates in a mail app's
    draft engine with a perfect score, because scores are normalised and the
    best match is always 1.0.  What separates a question about the code from
    smalltalk that happens to share a word with a string literal is the
    identifiers, so only identifier-shaped lines feed this set.
    """
    from .edits import defined_names

    vocabulary: set = set()
    for chunk in chunks:
        text = chunk.text or ""
        # Function and class names across the languages this app sees —
        # a C `void rebuild_symbol_table(void) {` binds a name every bit as
        # much as a Python `def` does, just without a keyword to spot.
        for name in defined_names(text):
            for piece in terms(name):
                if len(piece) > 2 and piece not in GENERIC_IDENTIFIER:
                    vocabulary.add(piece)
        for line in text.splitlines():
            stripped = line.strip()
            # Lines that *bind* names: def/class headers, assignments, and
            # C-style declarations ending in an open brace.
            code = stripped.split("#")[0]
            if not (stripped.startswith(("def ", "class ", "async def "))
                    or "=" in code or "{" in code):
                continue
            head = code.split("=")[0].split("{")[0]
            for piece in terms(head):
                if len(piece) > 2 and piece not in GENERIC_IDENTIFIER:
                    vocabulary.add(piece)
    return vocabulary


def worth_retrieving(question: str, chunks) -> bool:
    """Does this question plausibly need the code in front of the model?

    "say hello" does not, and sending three files anyway buries a nine-
    character greeting under a hundred thousand characters of source — the
    model answers the code, because the code is almost all of what it was
    given.  A question earns retrieval by sharing a word with the code's own
    identifiers, or by talking about files and code at all.
    """
    from .edits import ABOUT_FILES

    said = question or ""
    if ABOUT_FILES.search(said):
        return True
    # A very short message needs a hard signal, not a shared word.  This app
    # answers *email* — "thanks!" and "good morning" are identifiers in its
    # own draft engine (`_THANKS_ONLY_RE`, the greeting code), so vocabulary
    # overlap proves nothing until there are enough words to be a question.
    if len(said.split()) <= 4:
        return bool(re.search(r"\w+_\w+|\w+\(\)|\.\w{1,4}\b", said))
    vocabulary = identifier_vocabulary(chunks)
    return any(term in vocabulary for term in terms(said))


def select(question: str, chunks, budget_tokens: int,
           question_vector=None, max_files: int = 12,
           vector_weight: float = VECTOR_WEIGHT) -> list:
    """The chunks to actually send, best first, inside a token budget.

    Ordering is by relevance but the budget is spent greedily, so one enormous
    marginally-relevant file cannot crowd out four small highly-relevant ones:
    a chunk that does not fit is skipped rather than ending the selection.
    """
    by_name = {chunk.name: chunk for chunk in chunks}
    chosen, spent = [], 0
    for hit in rank(question, chunks, question_vector, vector_weight):
        if len(chosen) >= max_files:
            break
        chunk = by_name[hit.name]
        if spent + chunk.tokens > budget_tokens:
            continue
        chosen.append(chunk)
        spent += chunk.tokens
    return chosen


def describe_selection(chosen, considered: int) -> str:
    """What was sent and what was not — never a silent subset."""
    if not chosen:
        return (f"No file out of {considered} looked relevant enough to "
                f"include, so the question was answered from the "
                f"conversation alone.")
    names = ", ".join(chunk.name for chunk in chosen)
    return (f"📂 Sent {len(chosen)} of {considered} files, chosen for this "
            f"question: {names}.")
