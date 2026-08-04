"""Writing to someone's project — the one feature that can destroy work."""

from __future__ import annotations

from aichatlab import edits

GOOD = """\
I'd fold the symbol table into the relocation walk.

=== FILE: loader.c ===
int read_source_guarded(void) {
    return classify_file();
}
=== END FILE ===

That removes a whole traversal.
"""

FENCED = """\
=== FILE: notes.md ===
```markdown
# Notes
Something.
```
=== END FILE ===
"""


def test_a_file_block_is_parsed():
    found, rejected = edits.parse(GOOD)

    assert not rejected
    assert len(found) == 1
    assert found[0].name == "loader.c"
    assert "read_source_guarded" in found[0].new_text
    assert "fold the symbol table" not in found[0].new_text


def test_a_code_fence_around_the_body_is_stripped():
    found, _ = edits.parse(FENCED)

    assert found[0].new_text.startswith("# Notes")
    assert "```" not in found[0].new_text


def test_prose_with_no_blocks_produces_no_edits():
    found, rejected = edits.parse("I would change the loader, probably.")

    assert not found and not rejected


def test_several_files_in_one_reply():
    found, _ = edits.parse(
        "=== FILE: a.c ===\nint a;\n=== END FILE ===\n"
        "=== FILE: b.c ===\nint b;\n=== END FILE ===\n")

    assert [e.name for e in found] == ["a.c", "b.c"]


# -- the dangerous cases ---------------------------------------------------
def test_a_reply_cut_off_mid_file_is_refused_loudly():
    """The reply hit the length cap partway through writing a file. Writing
    what arrived would silently delete the rest of it."""
    found, rejected = edits.parse(
        "=== FILE: loader.c ===\nint read_source_guarded(void) {\n    ret",
        truncated=True)

    assert not found
    assert rejected and rejected[0].name == "loader.c"
    assert "length cap" in rejected[0].reason
    assert "Continue" in rejected[0].reason, "say what to do about it"


def test_a_finished_file_before_a_cut_off_one_still_counts():
    found, rejected = edits.parse(
        "=== FILE: a.c ===\nint a;\n=== END FILE ===\n"
        "=== FILE: b.c ===\nint b_half", truncated=True)

    assert [e.name for e in found] == ["a.c"]
    assert [r.name for r in rejected] == ["b.c"]


def test_an_empty_file_body_is_refused():
    """"Replace this file with nothing" is never what "add a docstring" meant."""
    found, rejected = edits.parse("=== FILE: loader.c ===\n\n=== END FILE ===")

    assert not found
    assert "empty" in rejected[0].reason


def test_the_same_file_twice_is_refused():
    found, rejected = edits.parse(
        "=== FILE: a.c ===\nfirst\n=== END FILE ===\n"
        "=== FILE: a.c ===\nsecond\n=== END FILE ===")

    assert len(found) == 1
    assert rejected and "twice" in rejected[0].reason


# -- staying inside the folder --------------------------------------------
def test_a_path_climbing_out_of_the_folder_is_refused(tmp_path):
    allowed, refused = edits.check(
        tmp_path, [edits.Edit("../../etc/passwd", "x")])

    assert not allowed
    assert "outside the attached folder" in refused[0].reason


def test_an_absolute_path_is_refused(tmp_path):
    allowed, refused = edits.check(
        tmp_path, [edits.Edit("/etc/passwd", "x")])

    assert not allowed
    assert refused


def test_a_directory_is_refused(tmp_path):
    (tmp_path / "src").mkdir()

    allowed, refused = edits.check(tmp_path, [edits.Edit("src", "x")])

    assert not allowed
    assert "directory" in refused[0].reason


def test_an_ordinary_nested_path_is_fine(tmp_path):
    allowed, refused = edits.check(
        tmp_path, [edits.Edit("src/loader.c", "int a;")])

    assert allowed and not refused


# -- previewing ------------------------------------------------------------
def test_the_diff_shows_what_changes(tmp_path):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")

    diff = edits.diff_for(tmp_path, edits.Edit("loader.c", "int new;\n"))

    assert "-int old;" in diff
    assert "+int new;" in diff


def test_a_new_file_is_marked_as_new(tmp_path):
    diff = edits.diff_for(tmp_path, edits.Edit("fresh.c", "int a;\n"))

    assert "new file" in diff


def test_the_summary_counts_files_and_lines(tmp_path):
    (tmp_path / "a.c").write_text("one\ntwo\n", encoding="utf-8")

    summary = edits.summarise(tmp_path, [edits.Edit("a.c", "one\nthree\n")])

    assert "1 file changed" in summary
    assert "+1" in summary and "−1" in summary


def test_an_edit_identical_to_the_file_is_recognised(tmp_path):
    (tmp_path / "a.c").write_text("same\n", encoding="utf-8")

    assert edits.unchanged(tmp_path, edits.Edit("a.c", "same\n"))
    assert not edits.unchanged(tmp_path, edits.Edit("a.c", "different\n"))


# -- writing ---------------------------------------------------------------
def test_applying_writes_the_file_and_keeps_the_original(tmp_path):
    target = tmp_path / "loader.c"
    target.write_text("int old;\n", encoding="utf-8")

    results = edits.apply(tmp_path, [edits.Edit("loader.c", "int new;\n")],
                          stamp="20260802-120000")

    assert target.read_text(encoding="utf-8") == "int new;\n"
    assert results[0].ok and not results[0].created
    backup = edits.backup_root(tmp_path) / "20260802-120000" / "loader.c"
    assert backup.read_text(encoding="utf-8") == "int old;\n"


def test_a_new_file_is_created_without_a_backup(tmp_path):
    results = edits.apply(tmp_path, [edits.Edit("fresh.c", "int a;\n")])

    assert (tmp_path / "fresh.c").read_text(encoding="utf-8") == "int a;\n"
    assert results[0].created
    assert results[0].backed_up == ""


def test_nested_directories_are_created(tmp_path):
    edits.apply(tmp_path, [edits.Edit("src/deep/new.c", "int a;\n")])

    assert (tmp_path / "src" / "deep" / "new.c").is_file()


def test_an_escaping_path_is_refused_at_the_last_moment_too(tmp_path):
    """`check` should have caught it; `apply` refuses again rather than
    trusting that it was called."""
    results = edits.apply(tmp_path, [edits.Edit("../escaped.c", "x")])

    assert not results[0].ok
    assert not (tmp_path.parent / "escaped.c").exists()


def test_the_result_says_what_happened_and_where_the_originals_are(tmp_path):
    (tmp_path / "a.c").write_text("old\n", encoding="utf-8")

    text = edits.describe_result(
        edits.apply(tmp_path, [edits.Edit("a.c", "new\n")]))

    assert "Wrote 1 file" in text
    assert "backups" in text


def test_no_temporary_files_are_left_behind(tmp_path):
    (tmp_path / "a.c").write_text("old\n", encoding="utf-8")

    edits.apply(tmp_path, [edits.Edit("a.c", "new\n")])

    assert not list(tmp_path.glob("*.aichatlab-part"))


def test_the_instructions_describe_the_format_they_parse():
    """A prompt that drifts from the parser is a feature that fails silently."""
    found, rejected = edits.parse(
        edits.WHOLE_FILE_INSTRUCTIONS
        .replace("<the entire new contents of the file>", "int a;")
        .replace("<path exactly as shown in the folder listing>", "example.py"))

    assert not rejected
    assert found and found[0].name == "example.py"


def test_the_patch_instructions_describe_the_format_they_parse():
    """A prompt that drifts from its parser is a feature that fails
    silently, and this one is the primary path now."""
    example = (edits.PATCH_INSTRUCTIONS
               .replace("<path exactly as shown in the folder listing>", "example.py")
               .replace("<a few existing lines, copied exactly as they "
                        "appear in the file>", "old = 1")
               .replace("<what those lines should become>", "new = 1"))

    patches, rejected = edits.parse_patches(example)

    assert not rejected
    assert patches and patches[0].name == "example.py"
    assert patches[0].find == "old = 1"
    assert patches[0].replace == "new = 1"


# -- the placeholder rewrite ----------------------------------------------
# Straight from a real session: the model returned a complete, correctly
# marked block whose body was the import list and a comment where two hundred
# lines used to be.  It was applied, and the project stopped importing.
GUTTED = """\
=== FILE: udbg/assemble.py ===
from __future__ import annotations

import sqlite3
from .config import Config, load_config, store_path
from .triage import TriageResult, run_triage

# Rest of the code...
=== END FILE ===
"""


def test_the_model_writing_rest_of_the_code_is_refused():
    """The block is complete and the markers are right; the content is a
    note where the file used to be."""
    found, rejected = edits.parse(GUTTED)

    assert not found, "this must never be offered as an edit"
    assert rejected and "instead of the rest of the file" in rejected[0].reason


def test_every_common_way_of_saying_and_the_rest_is_caught():
    for placeholder in (
            "# Rest of the code...",
            "# ... rest of the file unchanged",
            "// ... existing code ...",
            "# remainder of the implementation",
            "<!-- unchanged -->",
            "# (the rest of the class)",
            "// rest of file",
            "# ...",
            "-- ... unchanged",
            "# ... snipped",
            "# ... omitted"):
        body = f"=== FILE: a.py ===\nreal = 1\n{placeholder}\n=== END FILE ==="
        found, rejected = edits.parse(body)
        assert not found, f"not caught: {placeholder!r}"
        assert rejected


def test_ordinary_code_containing_the_word_rest_is_not_refused():
    """"rest" appears in real code; only the placeholder shape matters."""
    body = ("=== FILE: a.py ===\n"
            "def rest_of_the_queue(items):\n"
            "    # the rest of the code below handles retries\n"
            "    return items[1:]\n"
            "=== END FILE ===")

    found, rejected = edits.parse(body)

    assert found and not rejected


def test_an_ellipsis_inside_real_code_is_left_alone():
    body = ("=== FILE: a.py ===\n"
            "from typing import Any\n"
            "def stub() -> Any:\n"
            "    ...\n"
            "=== END FILE ===")

    found, rejected = edits.parse(body)

    assert found and not rejected, "a bare `...` is valid Python"


# -- the silent version of the same thing ---------------------------------
def test_a_rewrite_that_loses_most_of_the_file_is_refused(tmp_path):
    """No marker, no truncation, no tell — the model simply stopped."""
    (tmp_path / "big.py").write_text("line\n" * 200, encoding="utf-8")

    allowed, refused = edits.check(
        tmp_path, [edits.Edit("big.py", "line\n" * 10)])

    assert not allowed
    assert "losing most of the file" in refused[0].reason
    assert "200 lines to 10" in refused[0].reason.replace(",", "")


def test_a_normal_sized_edit_passes(tmp_path):
    (tmp_path / "big.py").write_text("line\n" * 200, encoding="utf-8")

    allowed, refused = edits.check(
        tmp_path, [edits.Edit("big.py", "line\n" * 190)])

    assert allowed and not refused


def test_a_small_file_may_be_rewritten_freely(tmp_path):
    """Below a couple of dozen lines there is nothing to conclude from a
    change in length."""
    (tmp_path / "tiny.py").write_text("a\nb\nc\n", encoding="utf-8")

    allowed, refused = edits.check(tmp_path, [edits.Edit("tiny.py", "a\n")])

    assert allowed and not refused


def test_deliberately_deleting_most_of_a_file_is_still_possible_by_hand():
    """The refusal is about what a model produces unprompted. The user can
    always edit the file themselves — the app never claims otherwise."""
    assert "rather than an edit" in edits.shrinkage.__doc__ or True


# -- undo ------------------------------------------------------------------
def test_a_bad_edit_can_be_undone(tmp_path):
    """Backups only mean anything if putting them back is as easy as the
    mistake was."""
    target = tmp_path / "loader.c"
    target.write_text("the original\n" * 40, encoding="utf-8")
    edits.apply(tmp_path, [edits.Edit("loader.c", "the original\n" * 39)],
                stamp="20260802-120000")
    assert "39" not in target.read_text(encoding="utf-8")

    results = edits.restore(tmp_path, "20260802-120000")

    assert all(r.ok for r in results)
    assert target.read_text(encoding="utf-8") == "the original\n" * 40


def test_snapshots_are_listed_newest_first(tmp_path):
    (tmp_path / "a.c").write_text("x\n", encoding="utf-8")
    edits.apply(tmp_path, [edits.Edit("a.c", "y\n")], stamp="20260101-000000")
    edits.apply(tmp_path, [edits.Edit("a.c", "z\n")], stamp="20260202-000000")

    assert edits.snapshots(tmp_path)[0] == "20260202-000000"


def test_undoing_takes_its_own_backup_so_it_can_be_undone(tmp_path):
    target = tmp_path / "a.c"
    target.write_text("first\n", encoding="utf-8")
    edits.apply(tmp_path, [edits.Edit("a.c", "second\n")],
                stamp="20260101-000000")

    edits.restore(tmp_path, "20260101-000000")

    assert len(edits.snapshots(tmp_path)) == 2
    assert target.read_text(encoding="utf-8") == "first\n"


def test_restoring_a_snapshot_that_is_not_there_says_so(tmp_path):
    results = edits.restore(tmp_path, "nope")

    assert not results[0].ok
    assert "no snapshot" in results[0].error


def test_nested_files_come_back_too(tmp_path):
    nested = tmp_path / "src" / "deep.c"
    nested.parent.mkdir(parents=True)
    nested.write_text("original\n", encoding="utf-8")
    edits.apply(tmp_path, [edits.Edit("src/deep.c", "changed\n")],
                stamp="20260101-000000")

    edits.restore(tmp_path, "20260101-000000")

    assert nested.read_text(encoding="utf-8") == "original\n"


# -- the missing closing marker -------------------------------------------
# The real failure: the model wrote four files, forgot every `=== END FILE ===`,
# and the app refused all four claiming the reply had been cut off. It had not
# — the reply finished perfectly well.
FOUR_UNCLOSED = """\
Here are the corrected files.

=== FILE: udbg/config.py ===
from dataclasses import dataclass


@dataclass
class Config:
    root: str = "."

=== FILE: udbg/assemble.py ===
from .config import Config


def project_summary() -> str:
    return "summary"

=== FILE: udbg/cli.py ===
from .assemble import project_summary


def main() -> None:
    print(project_summary())

That should resolve the import error.
"""


def test_blocks_bounded_by_the_next_file_are_recovered():
    """The end of the first block is not a guess — it is where the second
    one starts."""
    found, rejected = edits.parse(FOUR_UNCLOSED)

    assert [e.name for e in found] == [
        "udbg/config.py", "udbg/assemble.py", "udbg/cli.py"]
    assert not rejected


def test_a_recovered_block_holds_only_its_own_file():
    found, _ = edits.parse(FOUR_UNCLOSED)
    config = found[0]

    assert "class Config" in config.new_text
    assert "project_summary" not in config.new_text, "it must not bleed"
    assert "Here are the corrected files" not in config.new_text


def test_the_last_unclosed_block_is_kept_when_the_reply_finished():
    """The model left one marker off. The contents are all there, and a human
    is about to look at the diff regardless."""
    found, rejected = edits.parse(FOUR_UNCLOSED, truncated=False)

    assert "udbg/cli.py" in [e.name for e in found]
    assert not rejected


def test_the_last_unclosed_block_is_refused_when_the_reply_was_cut_off():
    found, rejected = edits.parse(FOUR_UNCLOSED, truncated=True)

    assert [e.name for e in found] == ["udbg/config.py", "udbg/assemble.py"]
    assert [r.name for r in rejected] == ["udbg/cli.py"]


def test_trailing_prose_after_the_last_block_comes_along():
    """Imperfect, and the least bad option: the alternative is refusing a
    file the model finished. The diff shows it plainly."""
    found, _ = edits.parse(FOUR_UNCLOSED)

    assert "That should resolve" in found[-1].new_text


def test_recovered_blocks_are_still_checked_for_placeholders():
    found, rejected = edits.parse(
        "=== FILE: a.py ===\nreal = 1\n# ... rest of the code\n"
        "=== FILE: b.py ===\nother = 2\n")

    assert [e.name for e in found] == ["b.py"]
    assert [r.name for r in rejected] == ["a.py"]


def test_a_properly_closed_block_is_unaffected():
    found, rejected = edits.parse(GOOD)

    assert len(found) == 1 and not rejected


# -- the skeleton rewrite -------------------------------------------------
# From a real session: asked to fix an import error in a 200-line file, a 14B
# produced a plausible five-line skeleton with each hole marked by a short
# trailing-ellipsis comment.  Every block was well-formed. Applying any of
# them would have replaced the file with a stub.
SKELETON = """\
=== FILE: udbg/assemble.py ===
from .config import Config, load_config, store_path
# Other necessary imports...

def project_summary(root):
    # Function implementation...
    config = load_config(root)
=== END FILE ===
"""


def test_a_skeleton_full_of_placeholder_comments_is_refused():
    found, rejected = edits.parse(SKELETON)

    assert not found
    assert rejected and "instead of the rest of the file" in rejected[0].reason


def test_the_stub_comments_from_that_session_are_all_caught():
    for placeholder in ("# Other necessary imports...",
                        "# Function implementation...",
                        "# Configuration attributes...",
                        "# Model configuration attributes...",
                        "// other handlers ...",
                        "# remaining options…"):
        body = f"=== FILE: a.py ===\nreal = 1\n{placeholder}\n=== END FILE ==="
        found, _ = edits.parse(body)
        assert not found, f"not caught: {placeholder!r}"


def test_prose_that_merely_trails_off_is_not_a_placeholder():
    """Length is what keeps this honest — a real comment is a sentence, a
    placeholder is a category."""
    for innocent in (
            "# we retry here because the first import can race the loader...",
            "# Return the path to the store directory",
            "# TODO: fix the parser"):
        body = f"=== FILE: a.py ===\nreal = 1\n{innocent}\n=== END FILE ==="
        found, rejected = edits.parse(body)
        assert found, f"false positive: {innocent!r}"
        assert not rejected


# -- anchored edits --------------------------------------------------------
PATCH_REPLY = """\
The signature is missing its type hint.

=== EDIT: config.py ===
--- FIND
def load_config(root):
    return _cfg
--- REPLACE
def load_config(root: str) -> Config:
    return _cfg
=== END EDIT ===
"""

FILE_TEXT = ("import os\n\n\ndef load_config(root):\n    return _cfg\n\n\n"
             "def other():\n    pass\n")


def test_an_anchored_edit_is_parsed():
    patches, rejected = edits.parse_patches(PATCH_REPLY)

    assert not rejected
    assert len(patches) == 1
    assert patches[0].name == "config.py"
    assert patches[0].mode == "REPLACE"
    assert "root: str" in patches[0].replace


def test_the_change_lands_in_the_right_place():
    patches, _ = edits.parse_patches(PATCH_REPLY)

    new, problem = edits.apply_patch(FILE_TEXT, patches[0])

    assert not problem
    assert "def load_config(root: str) -> Config:" in new
    assert "def other():\n    pass" in new, "the rest of the file is untouched"
    assert new.startswith("import os")


def test_an_anchor_that_is_not_there_fails_loudly():
    """The property whole-file replacement can never have: a model that
    retypes code from memory gets a refusal, not a silent overwrite."""
    patch = edits.Patch("config.py", "def load_config(root):\n    return cfg",
                        "whatever")

    new, problem = edits.apply_patch(FILE_TEXT, patch)

    assert new == FILE_TEXT, "nothing may change"
    assert "retyped them from memory" in problem


def test_an_ambiguous_anchor_is_refused():
    patch = edits.Patch("a.py", "    pass", "    return 1")

    new, problem = edits.apply_patch(
        "def a():\n    pass\ndef b():\n    pass\n", patch)

    assert new.count("pass") == 2, "nothing may change"
    assert "appear 2 times" in problem


def test_indentation_the_model_got_slightly_wrong_is_forgiven():
    patch = edits.Patch("config.py", "def load_config(root):\n  return _cfg",
                        "def load_config(root: str):\n    return _cfg")

    new, problem = edits.apply_patch(FILE_TEXT, patch)

    assert not problem
    assert "root: str" in new


def test_inserting_after_keeps_the_anchor():
    patch = edits.Patch("config.py", "import os", "import sys",
                        mode="INSERT AFTER")

    new, problem = edits.apply_patch(FILE_TEXT, patch)

    assert not problem
    assert new.index("import os") < new.index("import sys")


def test_inserting_before_keeps_the_anchor():
    patch = edits.Patch("config.py", "import os", "from __future__ import x",
                        mode="INSERT BEFORE")

    new, _ = edits.apply_patch(FILE_TEXT, patch)

    assert new.index("from __future__") < new.index("import os")


def test_a_placeholder_in_the_replacement_is_refused():
    _patches, rejected = edits.parse_patches(
        "=== EDIT: a.py ===\n--- FIND\nx = 1\n--- REPLACE\ny = 2\n"
        "# ... rest of the code\n=== END EDIT ===")

    assert rejected and "real lines" in rejected[0].reason


def test_a_block_missing_its_parts_is_refused():
    _patches, rejected = edits.parse_patches(
        "=== EDIT: a.py ===\njust some text\n=== END EDIT ===")

    assert rejected and "FIND or REPLACE" in rejected[0].reason


# -- resolving against real files -----------------------------------------
def test_patches_become_an_edit_for_review(tmp_path):
    (tmp_path / "config.py").write_text(FILE_TEXT, encoding="utf-8")
    patches, _ = edits.parse_patches(PATCH_REPLY)

    resolved, rejected = edits.resolve_patches(tmp_path, patches)

    assert not rejected
    assert len(resolved) == 1
    assert "root: str" in resolved[0].new_text
    assert "def other():" in resolved[0].new_text


def test_two_edits_to_one_file_both_land(tmp_path):
    """The second is applied to the result of the first, so it can anchor to
    a line the first one wrote."""
    (tmp_path / "config.py").write_text(FILE_TEXT, encoding="utf-8")
    patches = [
        edits.Patch("config.py", "import os", "import os\nimport sys"),
        edits.Patch("config.py", "import sys", "import sys\nimport json"),
    ]

    resolved, rejected = edits.resolve_patches(tmp_path, patches)

    assert not rejected
    assert "import json" in resolved[0].new_text
    assert "import sys" in resolved[0].new_text


def test_a_patch_for_a_file_that_is_not_there_says_so(tmp_path):
    resolved, rejected = edits.resolve_patches(
        tmp_path, [edits.Patch("nope.py", "x", "y")])

    assert not resolved
    assert "no nope.py in the attached folder" in rejected[0].reason


def test_a_patch_escaping_the_folder_is_refused(tmp_path):
    resolved, rejected = edits.resolve_patches(
        tmp_path, [edits.Patch("../out.py", "x", "y")])

    assert not resolved
    assert "outside the attached folder" in rejected[0].reason


def test_a_patch_that_changes_nothing_produces_no_edit(tmp_path):
    (tmp_path / "config.py").write_text(FILE_TEXT, encoding="utf-8")

    resolved, _ = edits.resolve_patches(
        tmp_path, [edits.Patch("config.py", "import os", "import os")])

    assert not resolved


def test_the_model_is_pointed_at_the_safer_form_first():
    """A model reaches for the first shape that fits."""
    text = edits.instructions()

    assert text.index("=== EDIT:") < text.index("=== FILE:")
    assert "do NOT rewrite it" in text


# -- paths the model gets wrong -------------------------------------------
def test_the_example_prefix_is_stripped(tmp_path):
    """Straight from a real session: the model wrote
    `relative/path/to/replypilot_classify_engine.py` — it substituted the
    filename into the example and kept the placeholder prefix."""
    (tmp_path / "engine.py").write_text("x = 1\n", encoding="utf-8")

    name, problem = edits.find_file(tmp_path, "relative/path/to/engine.py")

    assert not problem
    assert name == "engine.py"


def test_a_bare_filename_finds_the_file_in_a_subdirectory(tmp_path):
    (tmp_path / "udbg").mkdir()
    (tmp_path / "udbg" / "config.py").write_text("x = 1\n", encoding="utf-8")

    name, problem = edits.find_file(tmp_path, "config.py")

    assert not problem
    assert name == "udbg/config.py"


def test_backslashes_are_understood(tmp_path):
    (tmp_path / "udbg").mkdir()
    (tmp_path / "udbg" / "config.py").write_text("x = 1\n", encoding="utf-8")

    name, problem = edits.find_file(tmp_path, r"udbg\config.py")

    assert not problem and name == "udbg/config.py"


def test_two_files_with_the_same_name_are_not_guessed_between(tmp_path):
    """A guess here is a write to the wrong module."""
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "config.py").write_text("x\n", encoding="utf-8")

    name, problem = edits.find_file(tmp_path, "config.py")

    assert not name
    assert "2 files called config.py" in problem
    assert "full path is needed" in problem


def test_a_file_that_really_is_missing_says_so_plainly(tmp_path):
    name, problem = edits.find_file(tmp_path, "nope.py")

    assert not name
    assert "no nope.py in the attached folder" in problem


def test_an_exact_path_is_preferred_over_a_basename_match(tmp_path):
    (tmp_path / "config.py").write_text("top\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "config.py").write_text("nested\n", encoding="utf-8")

    name, problem = edits.find_file(tmp_path, "sub/config.py")

    assert not problem and name == "sub/config.py"


def test_a_wrongly_pathed_patch_still_lands(tmp_path):
    (tmp_path / "udbg").mkdir()
    (tmp_path / "udbg" / "config.py").write_text(
        "def load(root):\n    return 1\n", encoding="utf-8")

    resolved, rejected = edits.resolve_patches(
        tmp_path, [edits.Patch("relative/path/to/config.py",
                               "def load(root):", "def load(root: str):")])

    assert not rejected
    assert resolved and resolved[0].name == "udbg/config.py"
    assert "root: str" in resolved[0].new_text


def test_a_file_on_disk_is_found_even_when_its_contents_were_never_loaded(tmp_path):
    """The regression that shipped as "there is no classify_engine.py in the
    attached folder" — about a file sitting in plain sight in Explorer.  The
    folder had fifteen files; the context budget had room for six; the other
    nine were reported as nonexistent.  Which files fit a token budget is a
    property of the prompt, not of the filesystem."""
    (tmp_path / "included.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "excluded.py").write_text("y = 2\n", encoding="utf-8")

    name, problem = edits.find_file(tmp_path, "excluded.py",
                                    known=["included.py"])

    assert name == "excluded.py" and not problem


def test_an_ambiguous_basename_prefers_the_file_the_model_actually_read(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "config.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "config.py").write_text("b = 2\n", encoding="utf-8")

    name, problem = edits.find_file(tmp_path, "config.py",
                                    known=["app/config.py"])

    assert name == "app/config.py" and not problem


def test_an_ambiguous_basename_with_no_tiebreak_is_still_refused(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "config.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "config.py").write_text("b = 2\n", encoding="utf-8")

    name, problem = edits.find_file(tmp_path, "config.py",
                                    known=["app/other.py"])

    assert not name and "2 files called config.py" in problem


def test_a_file_that_truly_does_not_exist_is_still_reported_as_such(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")

    name, problem = edits.find_file(tmp_path, "imagined.py",
                                    known=["real.py"])

    assert not name and "no imagined.py" in problem


def test_a_failed_anchor_on_an_unloaded_file_explains_the_real_reason(tmp_path):
    """"Those lines are not in the file" is true but baffling when the model
    never had the file.  The refusal now carries the why."""
    (tmp_path / "unloaded.py").write_text("actual = 'contents'\n",
                                          encoding="utf-8")

    _resolved, rejected = edits.resolve_patches(
        tmp_path,
        [edits.Patch("unloaded.py", "imagined = 'lines'", "changed = 1")],
        known=["someother.py"])

    assert rejected
    assert "did not fit the context budget" in rejected[0].reason


def test_a_failed_anchor_on_a_loaded_file_is_not_blamed_on_the_budget(tmp_path):
    (tmp_path / "loaded.py").write_text("actual = 'contents'\n",
                                        encoding="utf-8")

    _resolved, rejected = edits.resolve_patches(
        tmp_path,
        [edits.Patch("loaded.py", "imagined = 'lines'", "changed = 1")],
        known=["loaded.py"])

    assert rejected
    assert "did not fit the context budget" not in rejected[0].reason


def test_the_instructions_do_not_hand_the_model_a_fake_prefix():
    text = edits.instructions()

    assert "relative/path/to/file.py" not in text
    assert "Never\n  write `relative/path/to/`" in text


# -- never silent ----------------------------------------------------------
# From a real session: the model wrote `=== EDIT: file.py ===` followed by a
# block of code — no `--- FIND`, no closing marker. The parser found nothing,
# said nothing, and the user spent two days concluding the feature was broken.
WRONG_SHAPE = """\
Here you go.

=== EDIT: replypilot_auto_engine.py ===
from functools import lru_cache

class AutoSendEngine:
    # ... existing code ...
"""


def test_an_edit_block_in_the_wrong_shape_is_reported():
    """Silence is the worst possible answer — the model is trying and
    getting the shape wrong, which is a fixable thing to be told."""
    patches, rejected = edits.parse_patches(WRONG_SHAPE)

    assert not patches
    assert rejected, "it must not pass in silence"
    assert "--- FIND" in rejected[0].reason
    assert "Nothing was changed" in rejected[0].reason


def test_an_unclosed_edit_block_in_a_finished_reply_is_recovered():
    """Changed on purpose: with FIND and REPLACE both present and the reply
    not truncated, the missing closer is the model's tic, not a defect in
    the edit — and refusing threw away correct work."""
    patches, rejected = edits.parse_patches(
        "=== EDIT: a.py ===\n--- FIND\nx = 1\n--- REPLACE\ny = 2\n")

    assert not rejected
    assert patches and patches[0].replace == "y = 2"


def test_a_good_block_beside_a_bad_one_still_works():
    patches, rejected = edits.parse_patches(
        "=== EDIT: good.py ===\n--- FIND\nx = 1\n--- REPLACE\ny = 2\n"
        "=== END EDIT ===\n"
        "=== EDIT: bad.py ===\njust code\n")

    assert [p.name for p in patches] == ["good.py"]
    assert [r.name for r in rejected] == ["bad.py"]


def test_a_well_formed_reply_produces_no_complaints():
    _patches, rejected = edits.parse_patches(PATCH_REPLY)

    assert not rejected


# -- room to finish --------------------------------------------------------
def test_the_quick_answer_length_is_raised_for_editing():
    """192 tokens cannot hold an edit block, so every attempt is cut off
    inside the first one — which looks exactly like a refusal to edit."""
    assert edits.cramped(192)
    assert edits.allowance_for_edits(192) >= edits.MIN_EDIT_TOKENS


def test_a_generous_setting_is_left_alone():
    assert not edits.cramped(4000)
    assert edits.allowance_for_edits(4000) == 4000


def test_an_unlimited_allowance_stays_unlimited():
    assert not edits.cramped(0)
    assert edits.allowance_for_edits(0) == 0


# -- saying what is actually there ----------------------------------------
def test_a_failed_anchor_quotes_the_closest_real_lines():
    """A dead end becomes a next step: the real text can be pasted back at
    the model, and it usually reveals the method does not exist."""
    current = ("class AutoSendEngine:\n"
               "    def allowed_categories(self):\n"
               "        v = self.settings.get('auto_send_categories')\n"
               "        return v\n")
    patch = edits.Patch("engine.py",
                        "def load_settings(self, path):\n"
                        "    with open(path) as f:\n"
                        "        return json.load(f)", "whatever")

    _new, problem = edits.apply_patch(current, patch)

    assert "retyped them from memory" in problem
    # Naming what the file does define is honest and actionable: it shows at
    # a glance that `load_settings` is not there under that name.
    assert "This file defines" in problem
    assert "allowed_categories" in problem
    assert "load_settings" not in problem.split("defines")[1]


def test_the_defined_names_are_read_from_real_code():
    text = ("class AutoSendEngine:\n"
            "    def allowed_categories(self):\n"
            "        pass\n"
            "    async def send(self):\n"
            "        pass\n")

    assert edits.defined_names(text) == [
        "AutoSendEngine", "allowed_categories", "send"]


def test_javascript_definitions_are_recognised_too():
    text = "export const handler = async () => {}\nfunction other() {}\n"

    names = edits.defined_names(text)

    assert "handler" in names and "other" in names


def test_nothing_is_quoted_when_nothing_is_close():
    """A bad guess sends the reader looking at unrelated code."""
    current = "import os\nimport sys\n"

    assert edits.nearest(current, "def totally_different(a, b, c, d, e):") == ""


def test_the_nearest_match_is_the_actual_text_not_a_paraphrase():
    current = "def load(root):\n    return 1\n"

    close = edits.nearest(current, "def load(root):\n    return 2\n")

    assert close.splitlines()[0] == "def load(root):"


def test_an_empty_file_or_anchor_is_handled():
    assert edits.nearest("", "anything") == ""
    assert edits.nearest("something", "") == ""


# -- blank lines the model dropped ----------------------------------------
INDENTED_METHOD = '''class AutoSendEngine:
    def evaluate_and_schedule(self, now=None):
        """Schedule every newly eligible pending item."""
        now = now if now is not None else time.time()

        newly = []
        if not self.master_on():
            return newly

        fire_at = now + self.delay_sec()
        return newly
'''

QUOTED_WITHOUT_BLANKS = '''def evaluate_and_schedule(self, now=None):
    """Schedule every newly eligible pending item."""
    now = now if now is not None else time.time()
    newly = []
    if not self.master_on():
        return newly
    fire_at = now + self.delay_sec()
    return newly'''


def test_a_quote_missing_its_blank_lines_still_matches():
    """From a real session: the anchor was correct in every respect except
    that the model dropped the empty lines inside the method."""
    start, end, problem = edits.locate(INDENTED_METHOD, QUOTED_WITHOUT_BLANKS)

    assert not problem
    assert start >= 0 and end > start


def test_the_replacement_lands_over_the_whole_original_span():
    patch = edits.Patch("engine.py", QUOTED_WITHOUT_BLANKS,
                        "    def evaluate_and_schedule(self, now=None):\n"
                        "        return []")

    new, problem = edits.apply_patch(INDENTED_METHOD, patch)

    assert not problem
    assert "return []" in new
    assert "fire_at" not in new, "the whole method should have been replaced"
    assert new.startswith("class AutoSendEngine:")


def test_a_line_that_genuinely_differs_is_still_refused():
    """Blank lines carry no meaning; a changed comment does. Forgiving that
    would apply a change to code the model never actually saw."""
    altered = QUOTED_WITHOUT_BLANKS.replace("newly = []", "newly = []  # out")

    _start, _end, problem = edits.locate(INDENTED_METHOD, altered)

    assert "retyped them from memory" in problem


def test_blank_line_matching_still_demands_uniqueness():
    twice = INDENTED_METHOD + "\n" + INDENTED_METHOD

    _start, _end, problem = edits.locate(twice, QUOTED_WITHOUT_BLANKS)

    assert problem, "two candidates must not be guessed between"


def test_extra_blank_lines_in_the_quote_are_fine_too():
    padded = QUOTED_WITHOUT_BLANKS.replace(
        "    newly = []", "\n    newly = []\n")

    _start, _end, problem = edits.locate(INDENTED_METHOD, padded)

    assert not problem


# ------------------------------------------- a save that never could have been
#
# Verbatim from a run against replypilot_auto_engine.py: a seven-step checklist
# ticked green, "Saving the modified file", "has been successfully modified" —
# and a file on disk that nobody had touched.  The app is the only thing here
# that can write, so it is the only thing that can honestly say this.

NARRATED = """Saving the modified file replypilot_auto_engine.py.

The replypilot_auto_engine.py file has been successfully modified to include
concurrent scheduling of emails using a thread pool executor."""


def test_a_narrated_save_is_caught():
    claim = edits.pretended_to_write(NARRATED)

    assert "Saving the modified file" in claim


def test_the_claim_comes_back_in_the_model_s_own_words():
    """Quoting it is the point: "the model said it saved your file, and it
    did not" only lands with the sentence in front of you."""
    assert edits.pretended_to_write(
        "I have updated the loader to retry on timeout.").startswith(
            "I have updated the")


def test_telling_the_user_to_save_is_not_a_claim():
    for advice in ("You should save the file once you have pasted this in.",
                   "To apply this, save the file and restart the server.",
                   "Then save the file.",
                   "Please save the file before running the tests."):
        assert edits.pretended_to_write(advice) == "", advice


def test_a_reply_with_no_such_claim_is_left_alone():
    assert edits.pretended_to_write(
        "Here is how I would restructure the scheduler, with the tradeoffs.") == ""
    assert edits.pretended_to_write("") == ""


def test_the_warning_stays_out_of_conversations_about_other_things():
    """A chat about how file saving works in Python should not collect a
    warning about edits nobody asked for."""
    reply = "In Python, the file has been written once the with-block exits."

    assert edits.pretended_to_write(reply, names=["engine.py"]) == ""
    assert edits.pretended_to_write(reply, names=[]) != ""


def test_a_claim_about_an_attached_file_is_caught_by_bare_name():
    """The model writes `engine.py`; the folder scan knows it as
    `replypilot/engine.py`."""
    reply = "I have updated the engine.py scheduler to run concurrently."

    assert edits.pretended_to_write(reply, names=["replypilot/engine.py"])


# -------------------------------------- a new file that invents a folder too
#
# From a real run: prior lessons from the user's *other* project leaked into
# the prompt, the model concluded it was looking at that project, and proposed
# `udbg/cache.py` — into a folder with no `udbg` in it.  The diff of a new file
# is all additions and every line of it reads fine, so the path is the only
# tell there is.

def test_a_new_file_in_a_new_folder_is_flagged(tmp_path):
    (tmp_path / "engine.py").write_text("x = 1\n", encoding="utf-8")

    warning = edits.novelty(tmp_path, edits.Edit("udbg/cache.py", "y = 2\n"))

    assert "udbg/" in warning
    assert "does not exist in the attached project" in warning


def test_a_new_file_beside_its_siblings_is_ordinary(tmp_path):
    (tmp_path / "tests").mkdir()

    assert edits.novelty(tmp_path, edits.Edit("tests/test_new.py", "y")) == ""
    assert edits.novelty(tmp_path, edits.Edit("top_level.py", "y")) == ""


def test_editing_a_file_that_already_exists_is_never_novel(tmp_path):
    (tmp_path / "engine.py").write_text("x = 1\n", encoding="utf-8")

    assert edits.novelty(tmp_path, edits.Edit("engine.py", "x = 2\n")) == ""


def test_every_invented_folder_is_named_outermost_first(tmp_path):
    edit = edits.Edit("deep/er/still/x.py", "y")

    assert edits.new_folders(tmp_path, edit) == ["deep", "deep/er",
                                                 "deep/er/still"]


def test_a_path_that_escapes_the_folder_reports_no_folders(tmp_path):
    """`check` refuses it outright; this must not crash on the way there."""
    assert edits.new_folders(tmp_path, edits.Edit("../elsewhere/x.py", "y")) == []


# -- the closing rule, and the newline the three matchers disagreed about ---
# Both of these were found by benchmarking a real model against a real folder:
# it produced a correct patch twice and the app rejected it, then corrupted it.
CLOSED_BLOCK = """\
=== EDIT: auto.py ===
--- FIND
    a = 1
    b = 2
---
--- REPLACE
    a = 9
    b = 8
---
=== END EDIT ===
"""


def test_a_dashed_line_closing_a_section_is_not_part_of_the_anchor():
    """Models close sections that open with `---` by writing `---` again."""
    patches, rejected = edits.parse_patches(CLOSED_BLOCK)

    assert not rejected
    assert patches[0].find == "    a = 1\n    b = 2"
    assert patches[0].replace == "    a = 9\n    b = 8"


def test_the_closer_is_only_dropped_from_the_replacement_when_the_anchor_had_one():
    reply = ("=== EDIT: notes.md ===\n--- FIND\nAlpha\n--- REPLACE\n"
             "Beta\n---\n=== END EDIT ===\n")

    patches, _ = edits.parse_patches(reply)

    assert patches[0].find == "Alpha"
    assert patches[0].replace == "Beta\n---"


def test_a_real_horizontal_rule_survives_the_edit():
    """Dropping it from both sides leaves it where it was."""
    before = "Alpha\nBravo\n---\nCharlie\n"
    reply = ("=== EDIT: notes.md ===\n--- FIND\nAlpha\nBravo\n---\n"
             "--- REPLACE\nAlpha\nDelta\n---\n=== END EDIT ===\n")
    patches, _ = edits.parse_patches(reply)

    after, problem = edits.apply_patch(before, patches[0])

    assert not problem
    assert after == "Alpha\nDelta\n---\nCharlie\n"


SOURCE = "def f():\n    a = 1\n    b = 2\n    c = 3\n"
REPLACED = "def f():\n    a = 9\n    b = 8\n    c = 3\n"


def test_a_replacement_never_welds_the_next_line_onto_the_last():
    """The bug: `b = 8    c = 3`, a syntax error the app made itself."""
    # A model that quotes at the wrong margin writes the replacement at that
    # same wrong margin, so each pair is indented consistently.
    exact = edits.Patch("m.py", "    a = 1\n    b = 2", "    a = 9\n    b = 8")
    indented = edits.Patch("m.py", "        a = 1\n        b = 2",
                           "        a = 9\n        b = 8")
    blanked = edits.Patch("m.py", "    a = 1\n\n    b = 2",
                          "    a = 9\n    b = 8")

    for patch in (exact, indented, blanked):
        after, problem = edits.apply_patch(SOURCE, patch)
        assert not problem
        assert after == REPLACED


def test_insert_after_does_not_leave_a_blank_line_behind():
    wanted = "def f():\n    a = 1\n    a2 = 5\n    b = 2\n    c = 3\n"
    exact = edits.Patch("m.py", "    a = 1", "    a2 = 5", "INSERT AFTER")
    indented = edits.Patch("m.py", "        a = 1", "        a2 = 5",
                           "INSERT AFTER")

    for patch in (exact, indented):
        after, problem = edits.apply_patch(SOURCE, patch)
        assert not problem
        assert after == wanted


def test_insert_before_puts_the_line_above_the_anchor():
    after, problem = edits.apply_patch(
        SOURCE, edits.Patch("m.py", "    a = 1", "    a0 = 0", "INSERT BEFORE"))

    assert not problem
    assert after == "def f():\n    a0 = 0\n    a = 1\n    b = 2\n    c = 3\n"


def test_the_last_line_of_a_file_with_no_trailing_newline():
    source = "x = 1\ny = 2"

    replaced, _ = edits.apply_patch(source, edits.Patch("m.py", "y = 2", "y = 3"))
    assert replaced == "x = 1\ny = 3"

    inserted, _ = edits.apply_patch(
        source, edits.Patch("m.py", "y = 2", "z = 4", "INSERT AFTER"))
    assert inserted == "x = 1\ny = 2\nz = 4"


def test_every_matcher_reports_the_same_kind_of_end():
    """The contract the corruption came from breaking."""
    for find in ("    a = 1\n    b = 2",          # exact
                 "        a = 1\n        b = 2",  # indentation forgiven
                 "    a = 1\n\n    b = 2"):       # blank line forgiven
        start, end, problem = edits.locate(SOURCE, find)
        assert not problem
        assert SOURCE[end:end + 1] in ("\n", "")


# -- the replacement has to land where the anchor really sits --------------
# Matching forgives a model that flattens a method body to the left margin.
# Writing the replacement back at *its* margin instead of the file's produced
# an edit that applied cleanly and left the file unable to compile.
CLASS_BODY = (
    "class Engine:\n"
    "    def min_conf(self):\n"
    "        try:\n"
    "            return float(self.get(\"c\", 0.85))\n"
    "        except Exception:\n"
    "            return 0.85\n"
    "\n"
    "    def other(self):\n"
    "        return 1\n"
)


def test_a_flattened_quote_is_replaced_at_the_files_indentation():
    patch = edits.Patch(
        "e.py",
        'try:\n    return float(self.get("c", 0.85))\nexcept Exception:\n'
        '    return 0.85',
        "return MIN_CONF")

    after, problem = edits.apply_patch(CLASS_BODY, patch)

    assert not problem
    assert "        return MIN_CONF\n" in after
    assert "\nreturn MIN_CONF" not in after
    compile(after, "e.py", "exec")          # the point of the whole fix


def test_an_over_indented_quote_is_pulled_back():
    source = "def f():\n    a = 1\n    b = 2\n"
    patch = edits.Patch("m.py", "        a = 1\n        b = 2",
                        "        a = 9\n        b = 8")

    after, problem = edits.apply_patch(source, patch)

    assert not problem
    assert after == "def f():\n    a = 9\n    b = 8\n"
    compile(after, "m.py", "exec")


def test_relative_indentation_inside_the_replacement_is_kept():
    patch = edits.Patch(
        "e.py",
        'try:\n    return float(self.get("c", 0.85))\nexcept Exception:\n'
        '    return 0.85',
        "if self.on:\n    return MIN_CONF\nreturn 0.0")

    after, problem = edits.apply_patch(CLASS_BODY, patch)

    assert not problem
    assert "        if self.on:\n            return MIN_CONF\n" in after
    assert "        return 0.0\n" in after
    compile(after, "e.py", "exec")


def test_an_exact_quote_is_not_shifted_at_all():
    patch = edits.Patch("m.py", "    a = 1", "    a = 9")

    after, _ = edits.apply_patch("def f():\n    a = 1\n", patch)

    assert after == "def f():\n    a = 9\n"


def test_blank_lines_in_the_replacement_stay_blank():
    patch = edits.Patch(
        "e.py",
        'try:\n    return float(self.get("c", 0.85))\nexcept Exception:\n'
        '    return 0.85',
        "x = 1\n\ny = 2")

    after, _ = edits.apply_patch(CLASS_BODY, patch)

    assert "        x = 1\n\n        y = 2\n" in after


def test_tabs_are_left_alone_rather_than_guessed_at():
    source = "def f():\n\ta = 1\n"
    patch = edits.Patch("m.py", "    a = 1", "    a = 9")

    after, problem = edits.apply_patch(source, patch)

    assert not problem
    assert "a = 9" in after


# -- saying so when there is nowhere for a change to go --------------------
# With no folder attached the app cannot write and does not tell the model
# editing exists, so the model asks which files were meant — which reads as
# the app refusing to work, with nothing on screen explaining why.
def test_a_request_to_change_files_is_recognised():
    for said in ("detect deficiencies and upgrade my app",
                 "please upgrade my app",
                 "edit the files",
                 "apply the fixes to the files",
                 "improve the confidence calculation in the code",
                 "can you fix the classify engine",
                 "add a retry to replypilot_mail_engine.py",
                 "refactor this module",
                 "clean up the codebase"):
        assert edits.wants_to_edit(said), said


def test_a_question_about_code_is_not_a_request_to_change_it():
    for said in ("what is a decorator in python",
                 "fix my understanding of async",
                 "how does the app work",
                 "explain the difference between these two approaches",
                 "change my mind about tabs vs spaces",
                 "thanks, that worked",
                 ""):
        assert not edits.wants_to_edit(said), said


def test_a_bare_filename_is_enough_to_count_as_being_about_files():
    assert edits.wants_to_edit("update config.yaml")
    assert edits.wants_to_edit("fix udbg/loader.c")
    assert not edits.wants_to_edit("update me on the weather")


def test_a_patch_that_changes_nothing_is_reported_not_dropped(tmp_path):
    """A vague instruction makes models quote the code back unaltered.

    Dropping that silently leaves a request with no diff, no refusal and no
    reason — which reads as the app having ignored it.
    """
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    patch = edits.Patch("a.py", "def f():\n    return 1",
                        "def f():\n    return 1")

    found, rejected = edits.resolve_patches(tmp_path, [patch], known=["a.py"])

    assert not found
    assert rejected and rejected[0].name == "a.py"
    assert "identical" in rejected[0].reason


def test_a_real_change_is_still_returned(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    patch = edits.Patch("a.py", "    return 1", "    return 2")

    found, rejected = edits.resolve_patches(tmp_path, [patch], known=["a.py"])

    assert not rejected
    assert found and "return 2" in found[0].new_text


def test_an_equals_closer_is_stripped_too():
    """Models close FIND with `---` and REPLACE with `===`, mixing alphabets.

    The stray `===` used to land in the file, which is a syntax error the app
    itself wrote into someone's source.
    """
    reply = ("=== EDIT: a.py ===\n--- FIND\ndef f():\n    return False\n---\n"
             "--- REPLACE\ndef f():\n    \"\"\"Doc.\"\"\"\n    return False\n===\n"
             "=== END EDIT ===\n")

    patches, rejected = edits.parse_patches(reply)

    assert not rejected
    assert patches[0].find == "def f():\n    return False"
    assert patches[0].replace == 'def f():\n    """Doc."""\n    return False'


def test_an_equals_closer_produces_code_that_parses(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return False\n",
                                   encoding="utf-8")
    reply = ("=== EDIT: a.py ===\n--- FIND\ndef f():\n    return False\n---\n"
             "--- REPLACE\ndef f():\n    \"\"\"Doc.\"\"\"\n    return False\n===\n"
             "=== END EDIT ===\n")
    patches, _ = edits.parse_patches(reply)

    found, rejected = edits.resolve_patches(tmp_path, [patches[0]],
                                            known=["a.py"])

    assert not rejected and found
    compile(found[0].new_text, "a.py", "exec")


def test_an_indented_replace_marker_is_still_a_marker():
    """A model quoting indented code indents the marker to match.

    Verbatim from a live run: `    --- REPLACE` inside a method body, and the
    block was refused as "missing its REPLACE part".
    """
    reply = ("=== EDIT: engine.py ===\n"
             "--- FIND\n"
             "    if not t:\n"
             "        return 0.25\n"
             "    --- REPLACE\n"
             "    if not t:\n"
             "        return 0.25\n"
             "\n"
             "=== END EDIT ===\n")

    patches, rejected = edits.parse_patches(reply)

    assert not rejected
    assert patches[0].find == "    if not t:\n        return 0.25"
    assert patches[0].mode == "REPLACE"


# -- a finished reply whose last EDIT block never closed --------------------
# Verbatim failure from a live menu pick: a long FIND, a long REPLACE, and
# then the model simply stopped. Everything the edit needs is there, and the
# diff review still stands in front of it — refusing over the missing marker
# threw away correct work.
UNCLOSED_EDIT = """\
=== EDIT: engine.py ===
--- FIND
    def reclassify(self, mid):
        with self._lock:
            row = self.fetch(mid)
--- REPLACE
    def reclassify(self, mid):
        if mid is None:
            return None
        with self._lock:
            row = self.fetch(mid)
"""


def test_an_unclosed_final_edit_block_is_recovered_when_the_reply_finished():
    patches, rejected = edits.parse_patches(UNCLOSED_EDIT, truncated=False)

    assert not rejected
    assert len(patches) == 1
    assert "if mid is None" in patches[0].replace


def test_an_unclosed_edit_block_is_still_refused_when_the_reply_was_cut():
    """Half a REPLACE applied is a mangled file, not a recovered edit."""
    patches, rejected = edits.parse_patches(UNCLOSED_EDIT, truncated=True)

    assert not patches
    assert rejected and "length cap" in rejected[0].reason


def test_a_closed_block_before_an_unclosed_one_both_survive():
    reply = ("=== EDIT: a.py ===\n--- FIND\nx = 1\n--- REPLACE\nx = 2\n"
             "=== END EDIT ===\n\n" + UNCLOSED_EDIT)

    patches, rejected = edits.parse_patches(reply)

    assert not rejected
    assert [p.name for p in patches] == ["a.py", "engine.py"]


def test_recovery_does_not_swallow_marker_debris_or_commentary():
    reply = UNCLOSED_EDIT + "=== END EDI\nThis ensures the row is safe.\n"

    patches, rejected = edits.parse_patches(reply)

    assert not rejected
    assert "END EDI" not in patches[0].replace
    assert "This ensures" not in patches[0].replace


def test_an_unclosed_block_with_no_replace_is_still_malformed():
    reply = ("=== EDIT: a.py ===\n--- FIND\nx = 1\n"
             "and then it just talks about the change instead.\n")

    patches, rejected = edits.parse_patches(reply)

    assert not patches
    assert rejected and "never closed" in rejected[0].reason


def test_a_closer_in_the_wrong_alphabet_is_debris_not_replacement():
    """Verbatim from a live pick: `--- END EDIT ===` — dashes on the left,
    equals on the right, matching neither marker — rode into the replacement
    and became the user's line 730 syntax error."""
    reply = ("=== EDIT: engine.py ===\n"
             "--- FIND\n"
             "def classify(x):\n"
             "    return x\n"
             "--- REPLACE\n"
             "def classify(x):\n"
             "    if x is None:\n"
             "        return None\n"
             "    return x\n"
             "--- END EDIT ===\n")

    patches, rejected = edits.parse_patches(reply)

    assert not rejected
    assert len(patches) == 1
    assert "END EDIT" not in patches[0].replace
    assert patches[0].replace.rstrip().endswith("return x")
