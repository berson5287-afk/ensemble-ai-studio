"""The research pipeline's pure parts: planning, dedup, citations, reporting."""

from __future__ import annotations

from aichatlab.deepresearch import (
    CitationReport,
    Source,
    build_plan_prompt,
    build_revision_prompt,
    build_synthesis_prompt,
    build_verification_prompt,
    check_citations,
    dedupe_sources,
    format_report,
    format_sources,
    parse_queries,
    parse_verification,
)

RESULTS = [
    {"title": "Copper pipe guide", "url": "https://plumbing.example/guide",
     "snippet": "How to sweat a joint."},
    {"title": "Same site, second page", "url": "https://plumbing.example/two",
     "snippet": "More detail."},
    {"title": "Same site, third page", "url": "https://plumbing.example/three",
     "snippet": "Even more."},
    {"title": "A different view", "url": "https://diy.example/solder",
     "snippet": "Alternative method."},
    {"title": "Duplicate link", "url": "https://plumbing.example/guide",
     "snippet": "Repeat."},
]


def sources(count: int = 3) -> list[Source]:
    return [Source(number=i, title=f"Title {i}", url=f"https://x{i}.example/p",
                   snippet=f"snippet {i}", text=f"body {i}")
            for i in range(1, count + 1)]


# ------------------------------------------------------------------ planning

def test_plan_prompt_asks_for_distinct_searches():
    prompt = build_plan_prompt("how fast is the U9 Xtreme?", limit=3)
    assert "how fast is the U9 Xtreme?" in prompt
    assert "3" in prompt
    assert "different facet" in prompt


def test_parses_one_query_per_line():
    queries = parse_queries("U9 Xtreme top speed\nU9 Xtreme 0-60 time\n"
                            "Yangwang U9 production numbers", "fallback")
    assert len(queries) == 3
    assert "U9 Xtreme top speed" in queries


def test_strips_numbering_quotes_and_preamble():
    queries = parse_queries('Here are the searches:\n1. "first query"\n'
                            '- second query\n* third query', "fallback")
    assert queries[0] == "first query"
    assert "second query" in queries
    assert not any(q.startswith(("-", "*", "1.")) for q in queries)


def test_duplicate_queries_are_dropped():
    assert len(parse_queries("same thing\nSAME THING\nother", "fb")) == 2


def test_query_limit_is_respected():
    assert len(parse_queries("\n".join(f"query {i}" for i in range(10)),
                             "fb", limit=3)) == 3


def test_empty_plan_falls_back_to_the_question():
    assert parse_queries("", "the original question") == ["the original question"]
    assert parse_queries("OK\n\n", "the original question") == [
        "the original question"]


# ----------------------------------------------------------------- gathering

def test_duplicate_urls_are_dropped():
    found = dedupe_sources(RESULTS)
    assert len({s.url for s in found}) == len(found)


def test_one_domain_cannot_crowd_out_the_rest():
    found = dedupe_sources(RESULTS, per_domain=2)
    domains = [s.domain for s in found]
    assert domains.count("plumbing.example") == 2
    assert "diy.example" in domains


def test_sources_are_numbered_from_one():
    found = dedupe_sources(RESULTS)
    assert [s.number for s in found] == list(range(1, len(found) + 1))


def test_source_limit_is_respected():
    many = [{"title": f"t{i}", "url": f"https://s{i}.example/p"}
            for i in range(30)]
    assert len(dedupe_sources(many, limit=8)) == 8


def test_results_without_a_url_are_skipped():
    assert dedupe_sources([{"title": "no link", "url": ""}]) == []


def test_format_sources_shares_the_character_budget():
    long_sources = [Source(number=i, title=f"T{i}", url=f"https://x{i}.example",
                           text="x" * 50_000) for i in range(1, 5)]
    block = format_sources(long_sources, max_chars=8000)
    assert len(block) < 20_000                 # trimmed, not concatenated raw
    assert "Source 4" in block                 # every source still present


# ---------------------------------------------------------------- citations

def test_citations_are_extracted():
    report = check_citations("Fast (Source 1). Also true (Source 3).", sources(3))
    assert report.cited == {1, 3}
    assert report.invalid == set()
    assert report.unused == {2}


def test_a_citation_to_a_nonexistent_source_is_caught():
    """The failure that matters: an invented citation reads as verified."""
    report = check_citations("It costs $40,000 (Source 7).", sources(3))

    assert report.invalid == {7}
    assert report.clean is False
    assert "does not exist" in report.summary


def test_multiple_numbers_in_one_citation():
    report = check_citations("Both agree (Sources 1 and 2).", sources(3))
    assert report.cited == {1, 2}


def test_an_answer_with_no_citations_is_reported():
    report = check_citations("Just some prose.", sources(2))
    assert report.cited == set()
    assert "no citations" in report.summary


# -------------------------------------------------------------- verification

def test_verification_parses_each_finding_type():
    result = parse_verification(
        "UNSUPPORTED: the 308mph figure is not in any source\n"
        "GAP: nothing covers pricing\n"
        "SEARCH: Yangwang U9 Xtreme price")

    assert result.unsupported == ["the 308mph figure is not in any source"]
    assert result.gaps == ["nothing covers pricing"]
    assert result.follow_ups == ["Yangwang U9 Xtreme price"]
    assert result.clean is False


def test_a_clean_verification_reports_nothing():
    result = parse_verification("OK")
    assert result.clean is True
    assert result.follow_ups == []


def test_bulleted_findings_are_accepted():
    result = parse_verification("- UNSUPPORTED: made up\n* GAP: missing bit")
    assert result.unsupported and result.gaps


def test_follow_up_searches_are_capped():
    result = parse_verification("\n".join(f"SEARCH: query {i}" for i in range(9)))
    assert len(result.follow_ups) <= 2


# ------------------------------------------------------------------ prompts

def test_synthesis_prompt_demands_citations_and_honesty():
    prompt = build_synthesis_prompt("how fast?", sources(2))
    assert "(Source 2)" in prompt
    assert "disagree" in prompt
    assert "don't answer part of the question" in prompt
    assert "body 1" in prompt                  # the source text is included


def test_verification_prompt_carries_the_answer_and_sources():
    prompt = build_verification_prompt("q", "the answer text", sources(2))
    assert "the answer text" in prompt
    assert "UNSUPPORTED:" in prompt and "SEARCH:" in prompt


def test_revision_prompt_lists_every_problem():
    verification = parse_verification("UNSUPPORTED: bad claim\nGAP: no pricing")
    citations = CitationReport(cited={9}, invalid={9})
    prompt = build_revision_prompt("q", "old answer", sources(2), verification,
                                   citations)

    assert "bad claim" in prompt
    assert "no pricing" in prompt
    assert "Source 9" in prompt
    assert "old answer" in prompt


# ----------------------------------------------------------------- reporting

def test_report_carries_answer_sources_and_open_questions():
    verification = parse_verification("GAP: pricing was never covered")
    citations = check_citations("Fast (Source 1).", sources(2))
    report = format_report("Fast (Source 1).", sources(2), verification,
                           citations)

    assert "Fast (Source 1)." in report
    assert "**Sources**" in report
    assert "https://x1.example/p" in report
    assert "Could not confirm" in report
    assert "pricing was never covered" in report


def test_a_clean_report_has_no_caveat_section():
    citations = check_citations("Fast (Source 1).", sources(2))
    report = format_report("Fast (Source 1).", sources(2),
                           parse_verification("OK"), citations)

    assert "Could not confirm" not in report
    assert "**Sources**" in report


def test_invented_citations_are_flagged_in_the_report():
    citations = check_citations("Costs $40k (Source 9).", sources(2))
    report = format_report("Costs $40k (Source 9).", sources(2),
                           parse_verification("OK"), citations)

    assert "Could not confirm" in report
    assert "Source 9" in report


def test_leading_numbers_in_a_query_are_not_eaten():
    """Regression: "0-60 acceleration time" was being cut to "acceleration time"."""
    queries = parse_queries("0-60 acceleration time\n2024 model pricing", "fb")

    assert "0-60 acceleration time" in queries
    assert "2024 model pricing" in queries


def test_real_list_markers_are_still_stripped():
    queries = parse_queries("1. first query\n2) second query\n- third query", "fb")
    assert queries == ["first query", "second query", "third query"]


def test_report_lists_the_searches_that_were_run():
    """Provenance: which queries produced this answer."""
    report = format_report("Fast (Source 1).", sources(2),
                           parse_verification("OK"),
                           check_citations("Fast (Source 1).", sources(2)),
                           queries=["top speed record", "0-60 time"])

    assert "**Searches run**" in report
    assert "top speed record" in report
    assert "0-60 time" in report
