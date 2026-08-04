"""The per-file cache: done once per version of a file, not once per question."""

from __future__ import annotations

from aichatlab import projectindex as pi

FILES = [("loader.c", "int read_source_guarded(void) { return 0; }"),
         ("symbols.c", "void rebuild_symbol_table(void) {}")]


def test_everything_needs_doing_when_nothing_is_known():
    index = pi.Index()

    assert len(index.needs_summary(FILES)) == 2
    assert len(index.needs_embedding(FILES, "nomic-embed-text")) == 2


def test_work_already_done_is_not_repeated():
    index = pi.Index()
    for name, text in FILES:
        index.record_summary(name, text, "does a thing")

    assert index.needs_summary(FILES) == []


def test_a_changed_file_is_the_only_one_that_costs_anything():
    index = pi.Index()
    for name, text in FILES:
        index.record_summary(name, text, "does a thing")

    changed = [("loader.c", "int read_source_guarded(void) { return 1; }"),
               FILES[1]]

    outstanding = index.needs_summary(changed)

    assert [name for name, _ in outstanding] == ["loader.c"]


def test_touching_a_file_without_changing_it_costs_nothing():
    """Keyed on content, so a checkout, a copy or a `touch` are all free."""
    index = pi.Index()
    index.record_summary("loader.c", FILES[0][1], "reads guarded sources")

    assert index.needs_summary([FILES[0]]) == []


def test_a_changed_file_loses_the_vector_for_its_old_self():
    """The old vector describes a file that no longer exists; keeping it is
    worse than having none, because it still ranks."""
    index = pi.Index()
    index.record_vector("loader.c", FILES[0][1], [0.1, 0.2], "nomic")

    index.record_summary("loader.c", "completely different contents", "new")

    assert index.vector_for("loader.c", pi.content_hash(
        "completely different contents"), "nomic") == []


def test_vectors_from_another_model_are_refused():
    """Two embedding models produce incomparable vectors, and mixing them
    gives a ranking that is subtly wrong and untraceable."""
    index = pi.Index()
    index.record_vector("loader.c", FILES[0][1], [0.1, 0.2], "nomic-embed-text")
    digest = pi.content_hash(FILES[0][1])

    assert index.vector_for("loader.c", digest, "nomic-embed-text")
    assert index.vector_for("loader.c", digest, "mxbai-embed-large") == []
    assert len(index.needs_embedding(FILES, "mxbai-embed-large")) == 2


def test_summary_and_vector_can_coexist_for_one_file():
    index = pi.Index()
    index.record_summary("loader.c", FILES[0][1], "reads guarded sources")
    index.record_vector("loader.c", FILES[0][1], [0.5], "nomic")
    digest = pi.content_hash(FILES[0][1])

    assert index.summary_for("loader.c", digest) == "reads guarded sources"
    assert index.vector_for("loader.c", digest, "nomic") == [0.5]


def test_deleted_files_are_forgotten():
    index = pi.Index()
    for name, text in FILES:
        index.record_summary(name, text, "x")

    dropped = index.forget_missing(["loader.c"])

    assert dropped == 1
    assert "symbols.c" not in index.entries


def test_no_embedding_model_means_no_embedding_work():
    assert pi.Index().needs_embedding(FILES, "") == []


# -- persistence -----------------------------------------------------------
def test_the_cache_survives_a_round_trip(tmp_path):
    index = pi.Index()
    index.record_summary("loader.c", FILES[0][1], "reads guarded sources")
    index.record_vector("loader.c", FILES[0][1], [0.25, 0.5], "nomic")

    assert pi.save(tmp_path, index)
    reloaded = pi.load(tmp_path)

    digest = pi.content_hash(FILES[0][1])
    assert reloaded.summary_for("loader.c", digest) == "reads guarded sources"
    assert reloaded.vector_for("loader.c", digest, "nomic") == [0.25, 0.5]


def test_a_damaged_cache_is_treated_as_an_empty_one(tmp_path):
    path = pi.path_for(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json", encoding="utf-8")

    assert pi.load(tmp_path).entries == {}


def test_junk_entries_are_skipped_rather_than_crashing():
    index = pi.Index.from_dict({"entries": [
        {"name": "good.c", "digest": "abc", "summary": "fine"},
        {"no_name": True},
        "not even a dict",
        {"name": "bad.c", "vector": ["not a number"]},
    ]})

    assert "good.c" in index.entries
    assert "bad.c" not in index.entries


def test_an_unwritable_folder_is_not_an_error(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file, not a directory", encoding="utf-8")

    assert not pi.save(blocker, pi.Index())


# -- prompts ---------------------------------------------------------------
def test_the_summary_prompt_does_not_send_the_whole_file():
    prompt = pi.build_summary_prompt("big.c", "x" * 200_000)

    assert len(prompt) < 6_000, "indexing must not cost as much as not indexing"


def test_throat_clearing_is_stripped_from_summaries():
    assert pi.clean_summary("This file reads sources.") == "reads sources."
    assert pi.clean_summary("Sure, handles the loader.") == "handles the loader."
    assert "\n" not in pi.clean_summary("two\nlines")


def test_the_name_and_summary_survive_truncation():
    """`ui/folder_dialog.py` is a strong signal the first 2,000 characters of
    the file might not contain."""
    text = pi.embedding_text("ui/folder_dialog.py", "asks how much to read",
                             "x" * 100_000, limit=100)

    assert text.startswith("ui/folder_dialog.py")
    assert "asks how much to read" in text
    assert len(text) < 200
