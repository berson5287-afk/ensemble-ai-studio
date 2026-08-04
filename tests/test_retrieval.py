"""Choosing which parts of a project to send."""

from __future__ import annotations

from aichatlab.retrieval import (
    Chunk,
    blend,
    cosine,
    describe_selection,
    keyword_scores,
    rank,
    select,
    terms,
    vector_scores,
    worth_retrieving,
)

LOADER = Chunk("loader.c", """
    static int read_source_guarded(const char *path, buffer *out) {
        verdict v = classify_file(path);
        if (!v.ingest_eligible) return WITHHELD;
        return slurp(path, out);
    }
""")

SYMBOLS = Chunk("symbols.c", """
    void rebuild_symbol_table(image *img) {
        for (size_t i = 0; i < img->n_sections; i++) hash_insert(...);
    }
""")

UI = Chunk("window.c", """
    void draw_titlebar(window *w) { blit(w->surface, w->title); }
""")

README = Chunk("README.md", "udbg is a small debugger for ELF binaries.")

ALL = [LOADER, SYMBOLS, UI, README]


def test_identifiers_are_split_as_well_as_kept():
    """`read_source_guarded` should match a question about "guarded reads"."""
    found = terms("read_source_guarded and rebuildSymbolTable")

    assert "read_source_guarded" in found
    assert "guarded" in found
    assert "symbol" in found


def test_noise_words_are_dropped():
    assert "the" not in terms("the thing that we should have")
    assert "thing" in terms("the thing that we should have")


def test_keyword_search_finds_the_right_file():
    scores = keyword_scores("where do we decide if a file is ingestible?", ALL)

    assert max(scores, key=scores.get) == "loader.c"


def test_an_exact_identifier_beats_everything():
    scores = keyword_scores("what does rebuild_symbol_table do?", ALL)

    assert max(scores, key=scores.get) == "symbols.c"


def test_an_unrelated_question_matches_little():
    scores = keyword_scores("what is the capital of France?", ALL)

    assert not scores or max(scores.values()) <= 1.0


def test_cosine_is_sane():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([], [1, 0]) == 0.0
    assert cosine([1, 0], [1, 0, 0]) == 0.0, "different lengths mean nothing"


def test_vectors_find_the_file_that_never_uses_your_words():
    """The case keyword search cannot do: ask about "safety", get the file
    that only ever says "guarded" and "eligible"."""
    loader = Chunk("loader.c", LOADER.text, vector=[1.0, 0.0, 0.0])
    ui = Chunk("window.c", UI.text, vector=[0.0, 1.0, 0.0])
    question = [0.98, 0.02, 0.0]

    scores = vector_scores(question, [loader, ui])

    assert max(scores, key=scores.get) == "loader.c"


def test_blending_uses_whichever_rankers_have_an_opinion():
    assert blend({"a": 1.0}, {}) == {"a": 1.0}
    assert blend({}, {"b": 1.0}) == {"b": 1.0}

    mixed = blend({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0},
                  vector_weight=0.5)

    assert mixed["a"] == mixed["b"] == 0.5


def test_ranking_puts_the_best_first():
    hits = rank("where is the symbol table rebuilt?", ALL)

    assert hits
    assert hits[0].name == "symbols.c"
    assert all(hits[i].score >= hits[i + 1].score
               for i in range(len(hits) - 1))


# -- the budget ------------------------------------------------------------
def test_selection_stays_inside_the_budget():
    big = [Chunk(f"f{i}.c", "x" * 40_000) for i in range(10)]
    chosen = select("x", big, budget_tokens=20_000)

    assert sum(c.tokens for c in chosen) <= 20_000


def test_one_huge_file_does_not_crowd_out_several_small_ones():
    """A chunk that does not fit is skipped, not treated as the end of the
    list — otherwise one marginally-relevant monster costs you four good
    matches."""
    monster = Chunk("monster.c", "ingestible " * 40_000)
    smalls = [Chunk(f"small{i}.c", "ingestible classify_file verdict")
              for i in range(4)]

    chosen = select("ingestible classify_file", [monster, *smalls],
                    budget_tokens=2_000)

    assert "monster.c" not in [c.name for c in chosen]
    assert len(chosen) == 4


def test_the_file_count_is_capped_even_when_the_budget_is_huge():
    many = [Chunk(f"f{i}.c", "classify_file verdict") for i in range(50)]

    chosen = select("classify_file", many, budget_tokens=1_000_000,
                    max_files=5)

    assert len(chosen) == 5


def test_nothing_relevant_selects_nothing():
    chosen = select("the capital of France", ALL, budget_tokens=100_000)

    assert not chosen


def test_what_was_sent_is_always_stated():
    """A silently-chosen subset is worse than no retrieval at all — you would
    never know which files the answer was based on."""
    chosen = select("where is the symbol table rebuilt?", ALL,
                    budget_tokens=100_000)

    note = describe_selection(chosen, considered=len(ALL))

    assert "symbols.c" in note
    assert f"of {len(ALL)} files" in note


def test_selecting_nothing_says_so_rather_than_staying_quiet():
    note = describe_selection([], considered=41)

    assert "41" in note
    assert "conversation alone" in note


def test_an_empty_project_is_harmless():
    assert rank("anything", []) == []
    assert select("anything", [], budget_tokens=1000) == []
    assert keyword_scores("anything", []) == {}


# -- smalltalk does not need three source files in front of it --------------
GREETING_CODE = Chunk(name="draft.py", text=(
    "_THANKS_ONLY_RE = re.compile(r'thanks')\n"
    "MORNING_GREETING = 'good morning'\n"
    "def make_draft(template_text):\n"
    "    drip_wait = compute_drip_wait()\n"
    "    return template_text\n"))


def test_smalltalk_is_not_worth_retrieving():
    for said in ("say hello", "thanks!", "good morning",
                 "how are you today", "tell me a joke",
                 "what is the capital of france"):
        assert not worth_retrieving(said, [GREETING_CODE]), said


def test_questions_about_the_code_are():
    for said in ("how does make_draft build the reply",
                 "where is drip_wait computed",
                 "what does this app do",
                 "explain the code"):
        assert worth_retrieving(said, [GREETING_CODE]), said


def test_a_short_ask_with_a_hard_signal_still_retrieves():
    assert worth_retrieving("fix make_draft()", [GREETING_CODE])
    assert worth_retrieving("check draft.py", [GREETING_CODE])
    assert worth_retrieving("about drip_wait", [GREETING_CODE])


def test_string_literals_do_not_make_a_question_about_the_code():
    """'hello' lives in the greeting template, not in any identifier."""
    chunk = Chunk(name="d.py", text="TEMPLATE = 'hello there, thanks'\n")

    assert not worth_retrieving("say hello please and thanks", [chunk])


def test_chunk_tokens_use_the_measured_estimator():
    """The last hiding place of characters-over-four: a 'budgeted' selection
    built a 105,000-character block for a 32,768-token window, and the server
    dropped the front of the prompt — system prompt and all."""
    dense = Chunk(name="a.py", text=(
        "    if (r.get(\"ai_confidence\") or 0.0) < self.min_conf():\n" * 200))

    assert dense.tokens > len(dense.text) // 4, "code prices above chars/4"
