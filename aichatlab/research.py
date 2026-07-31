"""Web research through a self-hosted SearXNG instance.

SearXNG is a privacy-respecting metasearch engine that many people run in
Docker on their home network.  Given its URL, this module turns a question
into a research block: search results plus the actual text of the top pages,
formatted so a model can answer from the sources and cite them by number.

Note: SearXNG only answers `format=json` requests when the JSON format is
enabled in its settings.yml (`search: formats: [html, json]`).  A 403 here
almost always means that line is missing, so the error says exactly that.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from html.parser import HTMLParser

import requests

USER_AGENT = "AIChatLab/2.3.0 (+https://github.com/ai-chat-lab)"

DEFAULT_MAX_RESULTS = 5
DEFAULT_FETCH_PAGES = 2
PAGE_CHAR_LIMIT = 3500
SNIPPET_CHAR_LIMIT = 400


class ResearchError(RuntimeError):
    """Search failed in a way worth telling the user about."""


class _TextExtractor(HTMLParser):
    """Strip an HTML page down to its readable text — no dependencies."""

    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer",
            "iframe", "form", "button"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "article", "section", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass  # keep whatever was parsed before the markup went bad
    return parser.text()


class SearxngClient:
    """Talks to one SearXNG instance."""

    def __init__(self, base_url: str, timeout: int = 15) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.base_url)

    def search(self, query: str,
               max_results: int = DEFAULT_MAX_RESULTS) -> list[dict]:
        """Return [{title, url, snippet, engine}, ...] for a query."""
        if not self.base_url:
            raise ResearchError("No SearXNG URL configured — set it in ⚙ Settings.")
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "safesearch": "1"},
                headers={"User-Agent": USER_AGENT},
                timeout=self.timeout)
        except requests.RequestException as exc:
            raise ResearchError(f"Could not reach SearXNG: {exc}") from exc

        if response.status_code == 403:
            raise ResearchError(
                "SearXNG refused the JSON request (HTTP 403). Enable it in "
                "searxng/settings.yml:\n\n  search:\n    formats:\n"
                "      - html\n      - json\n\nthen restart the container.")
        if response.status_code >= 400:
            raise ResearchError(
                f"SearXNG returned HTTP {response.status_code}.")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ResearchError(
                "SearXNG did not return JSON — check the URL points at the "
                "SearXNG web root.") from exc

        results = []
        for item in payload.get("results", [])[:max_results]:
            url = item.get("url", "")
            if not url:
                continue
            results.append({
                "title": (item.get("title") or url)[:200],
                "url": url,
                "snippet": (item.get("content") or "")[:SNIPPET_CHAR_LIMIT],
                "engine": item.get("engine", ""),
            })
        return results

    def ping(self) -> int:
        """Connection test: how many results does a trivial query return?"""
        return len(self.search("test", max_results=10))


def domain_of(url: str) -> str:
    """"https://en.wikipedia.org/wiki/X" → "en.wikipedia.org"."""
    match = re.match(r"https?://([^/]+)", url or "", re.IGNORECASE)
    return match.group(1).lower() if match else (url or "")[:40]


def fetch_page_text(url: str, timeout: int = 12,
                    max_chars: int = PAGE_CHAR_LIMIT) -> str:
    """Download one page and reduce it to readable text.  Empty on failure."""
    try:
        response = requests.get(url, timeout=timeout, headers={
            "User-Agent": USER_AGENT}, allow_redirects=True)
        content_type = response.headers.get("Content-Type", "")
        if response.status_code >= 400 or "html" not in content_type:
            return ""
        text = html_to_text(response.text[:600_000])
    except requests.RequestException:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars] + " …[page truncated]"
    return text


# --------------------------------------------------- context-aware questions

# Words that mean "you know what I'm referring to" — the tell-tale sign of a
# question that only makes sense given what was just said.
REFERENCE_WORDS = re.compile(
    r"\b(it|its|it's|that|those|these|this|they|them|their|the same|"
    r"he|she|his|her|there|one|ones|another|instead|too|also|"
    r"the car|the model|the first|the second|the last)\b",
    re.IGNORECASE)

# "Search query: ...", "Here is the query: ...", "Rewritten query: ..."
QUERY_NOISE = re.compile(
    r"^[^:\n]{0,40}\b(query|search|answer|rewritten|here)\b[^:\n]{0,20}:\s*",
    re.IGNORECASE)

# leading articles that are never part of the thing being discussed
LEADING_NOISE = re.compile(
    r"^(the|a|an|in|at|on|according to|based on|its|this|that)\s+",
    re.IGNORECASE)

STOP_ENTITIES = {
    "I", "The", "This", "That", "These", "Those", "It", "A", "An", "And",
    "But", "Source", "Sources", "You", "We", "They", "According", "Based",
    "Page", "URL", "Summary", "Answer", "Note", "Web",
}

ENTITY_RE = re.compile(r"\b([A-Z][\w.\-]*(?:\s+[A-Z0-9][\w.\-]*){0,3})\b")


def looks_like_followup(question: str) -> bool:
    """Would a person need the previous turn to understand this question?"""
    text = (question or "").strip()
    if not text:
        return False
    words = text.split()
    if len(words) <= 4:
        return True
    if len(words) <= 16 and REFERENCE_WORDS.search(text):
        return True
    return False


def clean_query(text: str) -> str:
    """Take a model's answer and make it usable as a search box query."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    line = QUERY_NOISE.sub("", line).strip().strip('"').strip("'").strip()
    if len(line) > 160:
        line = line[:160].rsplit(" ", 1)[0]
    return line


def extract_subject(history: Sequence[dict]) -> str:
    """Best guess at what the conversation is about, newest turns first.

    Prefers the *first* multi-word proper noun in a message rather than the
    longest: subjects tend to be introduced early, while the longest phrase is
    usually incidental detail like a test track's full name.
    """
    for message in reversed(list(history)[-6:]):
        content = message.get("content", "") or ""
        # our own injected research blocks are full of misleading proper nouns
        if content.startswith("[Web research results"):
            content = content.split("[End of web research", 1)[0]
            content = re.sub(r"^(Source \d+:|URL:|Summary:).*$", "", content,
                             flags=re.MULTILINE)

        candidates = []
        for match in ENTITY_RE.finditer(content):
            phrase = LEADING_NOISE.sub("", match.group(1).strip()).strip()
            if len(phrase) > 2 and phrase not in STOP_ENTITIES:
                candidates.append(phrase)

        for candidate in candidates:
            if " " in candidate:
                return candidate
        if candidates:
            return candidates[0]
    return ""


def build_rewrite_prompt(question: str, history: Sequence[dict]) -> str:
    recent = []
    for message in list(history)[-4:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = (message.get("content", "") or "")
        if content.startswith("[Web research results"):
            continue
        recent.append(f"{role}: {content[:400]}")
    return (
        "Rewrite the user's latest question as a standalone web search query.\n"
        "Replace pronouns like 'it' or 'that' with what they actually refer to.\n"
        "Reply with the search query only — no quotes, no explanation.\n\n"
        f"Conversation so far:\n{chr(10).join(recent)}\n\n"
        f"Latest question: {question}\n\nSearch query:")


def contextual_query(question: str, history: Sequence[dict] | None = None,
                     rewriter: Callable[[str], str] | None = None) -> str:
    """Turn a follow-up into a query that stands on its own.

    "what is the 0-60mph?" after a chat about the Yangwang U9 Xtreme should
    search for the car, not for the definition of 0-60.  A model rewrites the
    query when one is available; otherwise we fall back to gluing the most
    recent subject onto the question, which is crude but much better than
    searching the literal words.
    """
    question = (question or "").strip()
    history = list(history or [])
    if not question or not history or not looks_like_followup(question):
        return question

    if rewriter is not None:
        try:
            rewritten = clean_query(rewriter(build_rewrite_prompt(question, history)))
        except Exception:
            rewritten = ""
        if rewritten and len(rewritten) > 3:
            return rewritten

    subject = extract_subject(history)
    return f"{subject} {question}" if subject else question


def gather(client: SearxngClient, query: str,
           max_results: int = DEFAULT_MAX_RESULTS,
           fetch_pages: int = DEFAULT_FETCH_PAGES,
           emit: Callable[[str], None] | None = None) -> str:
    """Search, read the top pages, and build a sourced context block.

    Raises ResearchError when the search itself fails; unreachable individual
    pages are skipped silently (their snippet still appears).
    """
    say = emit or (lambda _msg: None)
    say(f"🔍 Searching the web for “{query}”…")
    results = client.search(query, max_results=max_results)
    if not results:
        raise ResearchError(f"The web search returned nothing for: {query}")

    fetched = 0
    read_from = []
    sections = []
    for number, result in enumerate(results, 1):
        section = [f"Source {number}: {result['title']}\nURL: {result['url']}"]
        if result["snippet"]:
            section.append(f"Summary: {result['snippet']}")
        if fetched < fetch_pages:
            site = domain_of(result["url"])
            say(f"📖 Reading {site}…")
            page_text = fetch_page_text(result["url"])
            if page_text:
                section.append(f"Page content:\n{page_text}")
                if site not in read_from:
                    read_from.append(site)
                fetched += 1
        sections.append("\n".join(section))

    if read_from:
        say(f"🔍 Read {', '.join(read_from)} and skimmed "
            f"{len(results)} results")
    else:
        say(f"🔍 Skimmed {len(results)} search results")

    return (
        f"[Web research results for: {query}]\n\n"
        + "\n\n---\n\n".join(sections)
        + "\n\n[End of web research. Use these sources to answer the user's "
        "question directly — write as if you simply know this, not as if you "
        "are reporting on search results. Add (Source N) after specific facts "
        "so they can be checked. If the sources don't cover something, say so "
        "plainly instead of guessing.]")
