"""The pieces of a proper research pass, with no UI and no network.

Single-shot retrieval — one query, one search, one answer — is what most
"chat with the web" features do, and it fails in predictable ways: one query
misses whole facets of a question, and nothing ever checks that the citations
the model wrote actually point at the sources it claims.

This module holds the parts that make the difference, all of them pure
functions so they can be tested without a model or a search engine:

* decomposing a question into several distinct searches
* deduplicating results so one chatty domain can't crowd out other views
* a deterministic citation check the model has no say in
* parsing the verification pass into unsupported claims, gaps and follow-ups
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

MAX_QUERIES = 3
MAX_SOURCES = 8
MAX_PER_DOMAIN = 2
MAX_BLOCK_CHARS = 14_000
MAX_FOLLOW_UPS = 2

CITATION_RE = re.compile(r"\(\s*Sources?\s+([\d,\s and]+)\)", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+")
# Strips "1. ", "- " and "* " from a list item — but deliberately not bare
# leading digits, or a query like "0-60 acceleration time" loses its subject.
LIST_PREFIX = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s+")


@dataclass
class Source:
    number: int
    title: str
    url: str
    snippet: str = ""
    text: str = ""

    @property
    def domain(self) -> str:
        match = re.match(r"https?://([^/]+)", self.url or "", re.IGNORECASE)
        return match.group(1).lower() if match else ""

    def as_block(self, text_limit: int = 3500) -> str:
        parts = [f"Source {self.number}: {self.title}", f"URL: {self.url}"]
        if self.snippet:
            parts.append(f"Summary: {self.snippet}")
        if self.text:
            body = self.text[:text_limit]
            if len(self.text) > text_limit:
                body += " …[truncated]"
            parts.append(f"Page content:\n{body}")
        return "\n".join(parts)


# ------------------------------------------------------------------ planning

def build_plan_prompt(question: str, limit: int = MAX_QUERIES) -> str:
    return (
        "You are planning web research. Break the question below into up to "
        f"{limit} web searches that between them would answer it.\n\n"
        "Make each search cover a different facet — don't just reword the "
        "question. Write them as search queries, not sentences. One per line, "
        "no numbering, nothing else.\n\n"
        f"Question: {question}\n\nSearches:")


def parse_queries(text: str, fallback: str,
                  limit: int = MAX_QUERIES) -> list[str]:
    """Pull search queries out of the planner's reply, always returning one."""
    queries: list[str] = []
    for raw in (text or "").splitlines():
        line = LIST_PREFIX.sub("", raw).strip().strip('"').strip("'")
        if not line or line.lower().startswith(("searches", "here", "sure")):
            continue
        if len(line) < 3 or len(line) > 160:
            continue
        if line.lower() not in {q.lower() for q in queries}:
            queries.append(line)
    return queries[:limit] or [fallback]


# ------------------------------------------------------------------ gathering

def dedupe_sources(results: Sequence[dict], limit: int = MAX_SOURCES,
                   per_domain: int = MAX_PER_DOMAIN) -> list[Source]:
    """Drop repeats and cap how much of the result set one site can occupy."""
    sources: list[Source] = []
    seen_urls: set[str] = set()
    per_domain_count: dict[str, int] = {}

    for item in results:
        url = (item.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        candidate = Source(number=0, title=item.get("title") or url,
                           url=url, snippet=item.get("snippet", ""))
        domain = candidate.domain
        if per_domain_count.get(domain, 0) >= per_domain:
            continue
        seen_urls.add(url)
        per_domain_count[domain] = per_domain_count.get(domain, 0) + 1
        sources.append(candidate)
        if len(sources) >= limit:
            break

    for index, source in enumerate(sources, 1):
        source.number = index
    return sources


def format_sources(sources: Sequence[Source],
                   max_chars: int = MAX_BLOCK_CHARS) -> str:
    """Render sources for a prompt, sharing the character budget between them."""
    if not sources:
        return ""
    with_text = [s for s in sources if s.text] or list(sources)
    per_source = max(600, max_chars // max(1, len(with_text)))
    return "\n\n---\n\n".join(s.as_block(per_source) for s in sources)


# ----------------------------------------------------------------- synthesis

def build_synthesis_prompt(question: str, sources: Sequence[Source]) -> str:
    return (
        "Answer the question below using only the sources provided.\n\n"
        "Rules:\n"
        "- Put a citation like (Source 2) directly after every specific fact, "
        "figure or claim you take from a source.\n"
        "- Where sources disagree, say so explicitly and give both figures.\n"
        "- If the sources don't answer part of the question, say that plainly "
        "rather than filling the gap from memory.\n"
        "- Lead with the answer. Structure it with short paragraphs or bullets "
        "where that genuinely helps.\n"
        "- Do not invent sources or cite numbers that aren't listed.\n\n"
        f"Question: {question}\n\n"
        f"Sources:\n\n{format_sources(sources)}\n\n"
        "Answer:")


# --------------------------------------------------------------- verification

@dataclass
class Verification:
    unsupported: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.unsupported and not self.gaps


def build_verification_prompt(question: str, answer: str,
                              sources: Sequence[Source]) -> str:
    return (
        "You are fact-checking a research answer against the sources it was "
        "written from. Be strict and sceptical.\n\n"
        "Report using these exact prefixes, one item per line:\n"
        "UNSUPPORTED: a claim in the answer that the sources do not actually "
        "support, or that cites the wrong source\n"
        "GAP: something the question asked that the sources never covered\n"
        "SEARCH: a web search that would close one of those gaps "
        f"(at most {MAX_FOLLOW_UPS})\n\n"
        "If the answer is fully supported and complete, reply with exactly: OK\n\n"
        f"Question: {question}\n\n"
        f"Answer under review:\n{answer}\n\n"
        f"Sources:\n\n{format_sources(sources, max_chars=8000)}\n\n"
        "Findings:")


def parse_verification(text: str) -> Verification:
    result = Verification()
    for raw in (text or "").splitlines():
        line = LIST_PREFIX.sub("", raw).strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered in {"ok", "ok.", "none", "no issues"}:
            continue
        if lowered.startswith("unsupported:"):
            claim = line.split(":", 1)[1].strip()
            if claim:
                result.unsupported.append(claim)
        elif lowered.startswith("gap:"):
            gap = line.split(":", 1)[1].strip()
            if gap:
                result.gaps.append(gap)
        elif lowered.startswith("search:"):
            query = line.split(":", 1)[1].strip().strip('"')
            if query and len(result.follow_ups) < MAX_FOLLOW_UPS:
                result.follow_ups.append(query)
    return result


@dataclass
class CitationReport:
    cited: set[int] = field(default_factory=set)
    invalid: set[int] = field(default_factory=set)
    unused: set[int] = field(default_factory=set)

    @property
    def clean(self) -> bool:
        return not self.invalid

    @property
    def summary(self) -> str:
        if self.invalid:
            listed = ", ".join(str(n) for n in sorted(self.invalid))
            return f"cites Source {listed}, which does not exist"
        if not self.cited:
            return "no citations at all"
        return f"{len(self.cited)} source(s) cited"


def check_citations(answer: str, sources: Sequence[Source]) -> CitationReport:
    """Deterministically check the citations — the model gets no say in this.

    Catches the failure that matters most in retrieval: an answer that cites
    a source number that was never provided, which reads as verified but is
    pure invention.
    """
    available = {source.number for source in sources}
    cited: set[int] = set()
    for match in CITATION_RE.finditer(answer or ""):
        for number in NUMBER_RE.findall(match.group(1)):
            cited.add(int(number))
    return CitationReport(cited=cited,
                          invalid={n for n in cited if n not in available},
                          unused=available - cited)


# -------------------------------------------------------------- revision

def build_revision_prompt(question: str, answer: str, sources: Sequence[Source],
                          verification: Verification,
                          citations: CitationReport) -> str:
    problems = []
    for claim in verification.unsupported:
        problems.append(f"- Unsupported: {claim}")
    for gap in verification.gaps:
        problems.append(f"- Not covered by the sources: {gap}")
    if citations.invalid:
        listed = ", ".join(str(n) for n in sorted(citations.invalid))
        problems.append(f"- Cited Source {listed}, which does not exist")

    return (
        "Revise the research answer below. New sources have been added and a "
        "fact-check found problems.\n\n"
        "Fix every problem listed. Remove or correct anything the sources "
        "don't support, use the new sources where they help, and state "
        "plainly anything that still can't be confirmed. Keep the citation "
        "style — (Source N) after each specific claim.\n\n"
        f"Question: {question}\n\n"
        f"Problems found:\n" + ("\n".join(problems) or "- none") + "\n\n"
        f"Previous answer:\n{answer}\n\n"
        f"All sources (renumbered):\n\n{format_sources(sources)}\n\n"
        "Revised answer:")


# ----------------------------------------------------------------- reporting

def format_report(answer: str, sources: Sequence[Source],
                  verification: Verification,
                  citations: CitationReport,
                  queries: Sequence[str] = ()) -> str:
    """The written-up result: answer, sources, searches run, open questions."""
    parts = [answer.strip()]

    if sources:
        listed = "\n".join(f"{s.number}. {s.title} — {s.url}" for s in sources)
        parts.append(f"**Sources**\n{listed}")

    if queries:
        parts.append("**Searches run**\n"
                     + "\n".join(f"- {query}" for query in queries))

    unresolved = []
    if verification.gaps:
        unresolved.extend(verification.gaps)
    if citations.invalid:
        listed = ", ".join(str(n) for n in sorted(citations.invalid))
        unresolved.append(f"The answer cited Source {listed}, which was never "
                          f"retrieved — treat that claim with suspicion.")
    if unresolved:
        parts.append("**Could not confirm**\n"
                     + "\n".join(f"- {item}" for item in unresolved))

    return "\n\n".join(parts)
