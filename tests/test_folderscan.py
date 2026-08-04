"""Folder scanning: what gets measured, what gets sent, and what it costs."""

from __future__ import annotations

from pathlib import Path

import pytest

from aichatlab.folderscan import (
    Limits,
    build_block,
    human_size,
    kind_of,
    plan_budget,
    select,
    survey,
)


@pytest.fixture
def project(tmp_path):
    """A small project with the usual noise around it."""
    (tmp_path / "README.md").write_text("# Demo project\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("some notes\n", encoding="utf-8")

    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("x = 1\n", encoding="utf-8")
    deep = src / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "buried.py").write_text("y = 2\n", encoding="utf-8")

    for noise in (".git", "node_modules", "__pycache__"):
        folder = tmp_path / noise
        folder.mkdir()
        (folder / "junk.py").write_text("nope\n", encoding="utf-8")

    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return tmp_path


# ------------------------------------------------------------------- survey

def test_survey_skips_build_and_vcs_folders(project):
    result = survey(project)

    found = {entry.relative for entry in result.entries}
    assert "README.md" in found
    assert "src/app.py" in found
    assert not any(name.startswith((".git/", "node_modules/", "__pycache__/"))
                   for name in found)
    assert {".git", "node_modules", "__pycache__"} <= set(result.skipped_dirs)


def test_survey_counts_but_does_not_include_unreadable_files(project):
    result = survey(project)

    assert result.unreadable == 1                       # the PNG
    assert not any(e.relative.endswith(".png") for e in result.entries)


def test_survey_can_be_told_to_include_the_noise(project):
    result = survey(project, skip_noise=False, skip_hidden=False)

    found = {entry.relative for entry in result.entries}
    assert "node_modules/junk.py" in found


def test_survey_records_depth_relative_to_the_root(project):
    depths = {entry.relative: entry.depth for entry in survey(project).entries}

    assert depths["README.md"] == 0
    assert depths["src/app.py"] == 1
    assert depths["src/a/b/c/d/buried.py"] == 5


def test_survey_of_a_file_or_missing_path_is_empty(tmp_path):
    a_file = tmp_path / "x.txt"
    a_file.write_text("hi", encoding="utf-8")

    assert survey(a_file).entries == []
    assert survey(tmp_path / "nope").entries == []


def test_survey_stops_at_the_limit_and_says_so(tmp_path):
    for index in range(30):
        (tmp_path / f"f{index}.txt").write_text("x", encoding="utf-8")

    result = survey(tmp_path, limit=10)

    assert result.truncated
    assert len(result.entries) == 10


# ------------------------------------------------------------------- select

def test_depth_limit_excludes_deeply_nested_files(project):
    selection = select(survey(project), Limits(max_depth=1))

    assert not any("buried" in e.relative for e in selection.chosen)
    assert selection.dropped["depth"] == 1


def test_kind_filter_keeps_only_the_requested_types(project):
    selection = select(survey(project),
                       Limits(kinds=frozenset({"code"}), max_depth=64))

    assert {e.relative for e in selection.chosen} == {
        "main.py", "src/app.py", "src/a/b/c/d/buried.py"}
    assert selection.dropped["kind"] == 2        # README.md and notes.txt


def test_oversized_files_are_dropped_with_a_reason(tmp_path):
    (tmp_path / "small.txt").write_text("x" * 100, encoding="utf-8")
    (tmp_path / "huge.txt").write_text("x" * 50_000, encoding="utf-8")

    selection = select(survey(tmp_path), Limits(max_file_bytes=1000))

    assert [e.relative for e in selection.chosen] == ["small.txt"]
    assert selection.dropped["size"] == 1


def test_file_count_cap_is_respected(project):
    selection = select(survey(project), Limits(max_files=2))

    assert len(selection.chosen) == 2
    assert selection.dropped["cap"]


def test_readme_is_read_first(project):
    selection = select(survey(project), Limits())

    assert selection.chosen[0].relative == "README.md"


def test_token_ceiling_stops_before_it_is_exceeded(tmp_path):
    from aichatlab.session import CHARS_PER_TOKEN

    for index in range(10):
        (tmp_path / f"f{index}.txt").write_text("x" * 4000, encoding="utf-8")

    per_file = 4000 // CHARS_PER_TOKEN
    ceiling = per_file * 3
    selection = select(survey(tmp_path), Limits(max_tokens=ceiling))

    # exactly three fit, and the rest are refused rather than squeezed in
    assert len(selection.chosen) == 3
    assert selection.dropped["budget"] == 7
    assert selection.tokens <= ceiling


def test_one_oversized_file_still_gets_through_alone(tmp_path):
    """A token ceiling smaller than the first file should not send nothing."""
    (tmp_path / "big.txt").write_text("x" * 40_000, encoding="utf-8")

    selection = select(survey(tmp_path), Limits(max_tokens=100))

    assert len(selection.chosen) == 1


def test_everything_limits_take_the_whole_folder(project):
    selection = select(survey(project), Limits.everything())

    assert len(selection.chosen) == 5
    assert selection.dropped_count == 0


def test_documents_are_estimated_below_their_raw_size(tmp_path):
    """A 1 MB PowerPoint is mostly images — pricing it by bytes is nonsense."""
    (tmp_path / "deck.pptx").write_bytes(b"\x00" * 1_000_000)
    (tmp_path / "notes.txt").write_bytes(b"x" * 1_000_000)

    by_name = {e.relative: e for e in survey(tmp_path).entries}

    from aichatlab.session import CHARS_PER_TOKEN

    assert by_name["notes.txt"].tokens == pytest.approx(
        1_000_000 / CHARS_PER_TOKEN, rel=0.01)
    assert by_name["deck.pptx"].tokens < by_name["notes.txt"].tokens / 10


def test_summary_mentions_what_was_left_out(project):
    selection = select(survey(project), Limits(max_files=1))

    assert "1 file" in selection.summary()
    assert "left out" in selection.summary()
    assert "over the file limit" in selection.dropped_detail()


# ------------------------------------------------------------------- budget

def test_a_folder_that_fits_needs_no_change():
    plan = plan_budget(1000, 6000)

    assert plan.fits
    assert plan.recommended == 6000
    assert "fits inside" in plan.message()


def test_a_folder_that_does_not_fit_recommends_a_bigger_budget():
    plan = plan_budget(40_000, 6000)

    assert not plan.fits
    assert plan.recommended >= 42_000
    assert plan.recommended % 4000 == 0
    assert "40,000" in plan.message() and "budget is 6,000" in plan.message()


def test_the_recommendation_leaves_room_for_the_conversation():
    plan = plan_budget(9000, 6000)

    assert plan.recommended >= 11_000


def test_small_overruns_round_to_the_nearest_thousand():
    plan = plan_budget(7000, 6000)

    assert plan.recommended == 9000


# -------------------------------------------------------------------- block

def test_block_contains_a_manifest_and_every_chosen_file(project):
    result = survey(project)
    selection = select(result, Limits())

    block = build_block(project, selection.chosen, len(result.entries))

    assert "Files included:" in block
    assert "src/app.py" in block
    assert "print('hello')" in block
    assert block.rstrip().endswith("[End of folder contents.]")


def test_block_tells_the_model_what_it_cannot_see(project):
    result = survey(project)
    selection = select(result, Limits(max_files=1))

    block = build_block(project, selection.chosen, len(result.entries))

    assert "4 other file(s)" in block
    assert "say so plainly" in block


def test_one_unreadable_file_does_not_lose_the_others(project):
    from aichatlab.documents import BinaryFileError

    result = survey(project)
    selection = select(result, Limits())

    def read(path):
        if path.name == "main.py":
            raise BinaryFileError(path.name)
        return "CONTENT"

    block = build_block(project, selection.chosen, len(result.entries),
                        read_fn=read)

    assert "this file is binary" in block
    assert block.count("CONTENT") == len(selection.chosen) - 1


def test_a_read_error_is_reported_inline_rather_than_raising(project):
    result = survey(project)
    selection = select(result, Limits(max_files=1))

    def read(path):
        raise OSError("permission denied")

    block = build_block(project, selection.chosen, len(result.entries),
                        read_fn=read)

    assert "could not read this file: permission denied" in block


def test_the_char_budget_is_a_hard_stop_even_if_the_estimate_was_wrong(project):
    """Estimation is a heuristic; the block builder must not trust it."""
    result = survey(project)
    selection = select(result, Limits())

    block = build_block(project, selection.chosen, len(result.entries),
                        read_fn=lambda path: "z" * 10_000, char_budget=5_000)

    assert block.count("z") <= 5_100
    assert "shortened to fit" in block


def test_progress_is_reported_for_every_file(project):
    result = survey(project)
    selection = select(result, Limits())
    seen = []

    build_block(project, selection.chosen, len(result.entries),
                progress=lambda done, total: seen.append((done, total)))

    assert seen[0] == (1, len(selection.chosen))
    assert seen[-1] == (len(selection.chosen), len(selection.chosen))


# --------------------------------------------------------------- small bits

@pytest.mark.parametrize("name, expected", [
    ("app.py", "code"), ("Main.JS", "code"), ("notes.md", "text"),
    ("data.csv", "text"), ("report.pdf", "docs"), ("deck.pptx", "docs"),
    ("photo.jpeg", None), ("binary.exe", None), (".gitignore", "text"),
])
def test_kind_of_classifies_by_extension(name, expected):
    assert kind_of(name) == expected


def test_human_size_reads_like_a_file_manager():
    assert human_size(512) == "512 B"
    assert human_size(2048) == "2.0 KB"
    assert human_size(5 * 1024 * 1024) == "5.0 MB"


# --------------------------------------- the budget must not exceed the window

def test_a_folder_bigger_than_the_model_window_is_flagged_not_accommodated():
    """Regression: raising our budget past the window we request is worse
    than not raising it.  We stop trimming and the server drops the oldest
    two thirds of the prompt instead, silently and without choosing well."""
    plan = plan_budget(92_000, 6000, ceiling=32_768)

    assert plan.over_ceiling
    assert plan.recommended == 32_768         # never above the window
    assert "more than the 32,768" in plan.message()


def test_a_folder_under_the_ceiling_is_accommodated_normally():
    plan = plan_budget(20_000, 6000, ceiling=32_768)

    assert not plan.over_ceiling
    assert 22_000 <= plan.recommended <= 32_768


def test_without_a_ceiling_nothing_is_capped():
    plan = plan_budget(92_000, 6000)

    assert not plan.over_ceiling
    assert plan.recommended > 92_000


def test_a_fitting_folder_is_never_over_the_ceiling():
    assert not plan_budget(500, 6000, ceiling=32_768).over_ceiling


def test_the_folder_header_names_a_relative_path(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = survey(".")
    selection = select(result, Limits())

    block = build_block(".", selection.chosen, len(result.entries))

    assert f"[Folder: {tmp_path.name}" in block


def test_the_char_budget_covers_the_whole_block_not_just_the_file_text(project):
    """It is a promise to the caller about the finished string: the manifest,
    the per-file headings and the fences come out of the same allowance."""
    result = survey(project)
    selection = select(result, Limits())

    for budget in (2_000, 6_000, 20_000):
        block = build_block(project, selection.chosen, len(result.entries),
                            read_fn=lambda path: "z" * 50_000,
                            char_budget=budget)
        assert len(block) <= budget, f"{len(block)} > {budget}"


# ------------------------------- what the app holds vs what one prompt carries

def _entry(name, size=100, kind="code", depth=0):
    from aichatlab.folderscan import FileEntry
    return FileEntry(path=Path("/x") / name, relative=name, size=size,
                     kind=kind, depth=depth)


def test_the_readable_set_survives_the_token_budget():
    """Which files fit one prompt is a property of the prompt.  The readable
    set is a property of the folder, and the budget must not shrink it."""
    from aichatlab.folderscan import Limits, Survey, select

    result = Survey(root=Path("/x"))
    result.entries = [_entry(f"f{n}.py", size=4_000) for n in range(14)]

    selection = select(result, Limits(max_tokens=6_000))

    assert len(selection.chosen) < 14, "the budget should bite in this test"
    assert len(selection.readable) == 14


def test_files_dropped_for_what_they_are_stay_dropped_everywhere():
    """kind/depth/size say something about the file itself; those drops
    apply to holding as well as sending."""
    from aichatlab.folderscan import Limits, Survey, select

    result = Survey(root=Path("/x"))
    result.entries = [_entry("good.py"),
                      _entry("huge.py", size=10_000_000),
                      _entry("deep.py", depth=9)]

    selection = select(result, Limits())

    names = [entry.relative for entry in selection.readable]
    assert names == ["good.py"]


def test_holding_keeps_the_sent_files_even_under_a_tight_cap():
    """Under the byte cap, the files actually in the prompt are never the
    ones sacrificed."""
    from aichatlab.folderscan import Limits, Survey, select, worth_holding

    result = Survey(root=Path("/x"))
    result.entries = [_entry(f"f{n}.py", size=1_000) for n in range(10)]
    selection = select(result, Limits(max_tokens=1))   # roughly one file fits

    held = worth_holding(selection, cap_bytes=3_000)

    names = [entry.relative for entry in held]
    assert names[0] == selection.chosen[0].relative
    assert len(held) == 3


def test_an_ordinary_folder_is_held_in_full():
    from aichatlab.folderscan import Limits, Survey, select, worth_holding

    result = Survey(root=Path("/x"))
    result.entries = [_entry(f"f{n}.py", size=2_000) for n in range(14)]
    selection = select(result, Limits(max_tokens=2_000))

    held = worth_holding(selection, cap_bytes=20_000_000)

    assert len(held) == 14
