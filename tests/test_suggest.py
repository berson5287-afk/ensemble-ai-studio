"""A broad wish becomes a menu of changes that can actually land.

The numbers behind the design, measured against this app's own pipeline:
one-place edits applied cleanly every run; two-place edits broke most runs;
"find the deficiencies" broke or produced nothing four runs out of four.  And
on the first live menus, seven of the nine functions the model named did not
exist — so the menu is grounded in a symbol index and verified before it is
ever shown.
"""

from __future__ import annotations

from aichatlab import suggest
from aichatlab.intent import Change

KNOWN = ["replypilot_auto_engine.py", "replypilot_mail_engine.py",
         "replypilot_classify_engine.py"]

MENU = """\
Here are some candidate improvements:

=== SUGGESTIONS ===
1. [replypilot_auto_engine.py] Add a guard to eligible_rows() for an empty queue
2. [replypilot_mail_engine.py] Add a docstring to html_to_text() describing the fallback
3. [replypilot_classify_engine.py] Early return in extract_features() when body is empty
=== END SUGGESTIONS ===
"""


# -- when the menu fires ----------------------------------------------------
def test_a_broad_wish_is_recognised():
    for said in ("I want performance enhancements in this app",
                 "make the error handling more robust",
                 "clean up the codebase",
                 "what can be improved here?",
                 "upgrade the code and fix any issues"):
        assert suggest.is_broad(said), said


def test_a_specific_instruction_goes_straight_through():
    """The menu would be answering a question with a questionnaire."""
    for said in ("replace 0.85 with a constant to improve reliability",
                 "improve html_to_text() so it returns '' for None",
                 "rename `min_conf` to threshold to make it better",
                 "set the timeout to 30 to speed up failures",
                 "improve the check on line 42"):
        assert not suggest.is_broad(said), said


def test_a_question_that_is_not_an_edit_request_is_not_broad():
    for said in ("how does the classifier perform?",
                 "is this robust enough for production?",
                 ""):
        assert not suggest.is_broad(said), said


# -- reading the menu back --------------------------------------------------
def test_a_well_formed_menu_is_parsed():
    menu = suggest.parse_menu(MENU, KNOWN)

    assert len(menu) == 3
    assert menu[0].file == "replypilot_auto_engine.py"
    assert "eligible_rows()" in menu[0].description


def test_the_markers_are_optional_but_the_shape_is_not():
    bare = "\n".join(MENU.splitlines()[2:6])      # lines without the fences

    assert len(suggest.parse_menu(bare, KNOWN)) == 3


def test_one_bracketed_line_in_prose_is_not_a_menu():
    reply = ("I looked at [replypilot_auto_engine.py] and it seems fine.\n"
             "1. [replypilot_auto_engine.py] Add a guard to eligible_rows()\n")

    assert suggest.parse_menu(reply, KNOWN) == []


def test_unknown_files_are_dropped():
    reply = ("1. [invented_module.py] Add a guard to something()\n"
             "2. [replypilot_auto_engine.py] Add a guard to eligible_rows()\n"
             "3. [replypilot_mail_engine.py] Add a check to html_to_text()\n")

    menu = suggest.parse_menu(reply, KNOWN)

    assert [c.file for c in menu] == ["replypilot_auto_engine.py",
                                      "replypilot_mail_engine.py"]


def test_prose_files_are_not_offered_as_code_changes():
    reply = ("1. [README.md] Document the retry behaviour in setup()\n"
             "2. [replypilot_auto_engine.py] Guard eligible_rows() for empty\n"
             "3. [replypilot_mail_engine.py] Check html_to_text() for None\n")

    menu = suggest.parse_menu(reply, KNOWN + ["README.md"])

    assert all(not c.file.endswith(".md") for c in menu)


def test_candidates_naming_a_function_are_ranked_first():
    reply = ("1. [replypilot_auto_engine.py] Improve the scheduling somehow\n"
             "2. [replypilot_mail_engine.py] Add a check to html_to_text()\n")

    menu = suggest.parse_menu(reply, KNOWN)

    assert "html_to_text()" in menu[0].description


def test_duplicates_collapse():
    reply = ("1. [replypilot_auto_engine.py] Guard eligible_rows() for empty\n"
             "2. [replypilot_auto_engine.py] Guard eligible_rows() for empty\n"
             "3. [replypilot_mail_engine.py] Check html_to_text() for None\n")

    assert len(suggest.parse_menu(reply, KNOWN)) == 2


# -- verification against the real files ------------------------------------
TEXTS = {
    "replypilot_auto_engine.py": "def eligible_rows(self):\n    return []\n",
    "replypilot_mail_engine.py": "def html_to_text(html):\n    return ''\n",
}


def test_a_candidate_naming_a_real_function_is_kept():
    kept, dropped = suggest.verify(
        [Change("Guard eligible_rows() against an empty queue",
                "replypilot_auto_engine.py")], TEXTS)

    assert len(kept) == 1 and not dropped


def test_an_invented_function_is_dropped_with_the_name_in_the_reason():
    """The live measurement: seven of nine named functions did not exist."""
    kept, dropped = suggest.verify(
        [Change("Add caching to classify_endpoint() for speed",
                "replypilot_auto_engine.py")], TEXTS)

    assert not kept
    assert "classify_endpoint() does not exist" in dropped[0][1]


def test_a_candidate_naming_no_function_is_dropped():
    kept, dropped = suggest.verify(
        [Change("Optimise the parsing logic to reduce complexity",
                "replypilot_mail_engine.py")], TEXTS)

    assert not kept
    assert "names no function" in dropped[0][1]


def test_every_named_function_must_exist_not_just_one():
    kept, dropped = suggest.verify(
        [Change("Make eligible_rows() call missing_helper() first",
                "replypilot_auto_engine.py")], TEXTS)

    assert not kept
    assert "missing_helper()" in dropped[0][1]


# -- the prompt and the guard-rails ----------------------------------------
def test_the_wrap_forbids_the_shapes_that_break():
    wrapped = suggest.wrap_ask("make it faster", "a.py: f(), g()")

    assert "ONE change in ONE place" in wrapped
    assert "No code" in wrapped
    assert suggest.MENU_OPEN in wrapped
    assert "a.py: f(), g()" in wrapped
    assert "does not exist" in wrapped      # the grounding clause


def test_the_wrap_survives_having_no_index():
    wrapped = suggest.wrap_ask("make it faster")

    assert suggest.MENU_OPEN in wrapped
    assert "functions that exist" not in wrapped


def test_the_symbol_index_lists_real_functions_only():
    index = suggest.symbol_index({
        "a.py": "def alpha():\n    pass\n\ndef beta():\n    pass\n",
        "NOTES.md": "# alpha() looks wrong\n"})

    assert "alpha()" in index and "beta()" in index
    assert "NOTES.md" not in index


def test_a_two_place_candidate_is_flagged_not_hidden():
    assert suggest.looks_multi_location(
        "Add MAX_DEPTH and use it in list_mail_folders()")
    assert suggest.looks_multi_location(
        "Apply the new check throughout the module")
    assert not suggest.looks_multi_location(
        "Add a guard clause to eligible_rows()")


# -- what a menu is built from ----------------------------------------------
def test_menu_files_are_code_not_the_docs_that_mention_the_topic():
    """"Performance enhancements" matches the README, which discusses
    performance in those words, while the engine that has the performance
    stays home. A menu built from files it cannot propose edits for is a
    menu of guesses."""
    texts = {
        "README.md": "Performance enhancements are planned. " * 200,
        "SESSION_NOTES.md": "We discussed performance at length. " * 200,
        ".gitignore": "*.pyc\n__pycache__/\n",
        "engine.py": "def tally(rows):\n    return len(rows)\n" * 50,
        "mail.py": "def send(msg):\n    return msg\n" * 50,
    }

    chosen = suggest.pick_menu_files(
        "I want performance enhancements in this app.", texts, 50_000)

    names = [c.name for c in chosen]
    assert "engine.py" in names and "mail.py" in names
    assert all(not n.endswith(".md") for n in names)
    assert all(not n.startswith(".") for n in names)


def test_menu_files_respect_the_budget():
    texts = {f"m{i}.py": "def f():\n    return 1\n" * 400 for i in range(30)}

    chosen = suggest.pick_menu_files("improve things", texts, 5_000)

    assert chosen, "something always goes"
    assert sum(c.tokens for c in chosen) <= 5_000


def test_breadth_beats_one_giant_file():
    """Three small engines teach the menu more than one huge one."""
    texts = {"huge.py": "def big():\n    return 1\n" * 3000,
             "a.py": "def a():\n    return 1\n" * 20,
             "b.py": "def b():\n    return 1\n" * 20,
             "c.py": "def c():\n    return 1\n" * 20}

    chosen = suggest.pick_menu_files("improve the code", texts, 2_000)

    names = {c.name for c in chosen}
    assert {"a.py", "b.py", "c.py"} <= names


# -- the map: what each file is for, in the authors' own words --------------
HEADERED = (
    "# engine.py\n"
    "# ReplyPilot Draft Engine v1.0.0\n"
    "# Deterministic templates per category, polished by the local LLM.\n"
    "# v1.2.0: acknowledgement, AI Review, checkboxes, auto-send.\n"
    "def make_draft():\n    return 1\n")


def test_the_map_reads_the_header_not_the_filename():
    text = HEADERED.replace("# engine.py", "# draft.py")

    result = suggest.project_map({"draft.py": text})

    assert "Deterministic templates" in result
    assert "draft.py — draft.py" not in result


def test_changelog_lines_are_not_a_description():
    result = suggest.project_map({"engine.py": HEADERED.replace(
        "# engine.py", "# engine.py")})

    assert "acknowledgement, AI Review" not in result


def test_a_module_docstring_wins_where_there_is_one():
    text = '"""The loader.\n\nReads things carefully."""\ndef load():\n    pass\n'

    assert "The loader" in suggest.project_map({"loader.py": text})


def test_a_file_with_no_header_falls_back_to_what_it_defines():
    text = "def alpha():\n    pass\n\ndef beta():\n    pass\n"

    result = suggest.project_map({"bare.py": text})

    assert "alpha()" in result and "beta()" in result


def test_docs_and_dotfiles_stay_off_the_map():
    result = suggest.project_map({
        "README.md": "# Everything about performance\n",
        ".gitignore": "# never commit real mail\n*.pyc\n",
        "engine.py": HEADERED})

    assert "README" not in result
    assert "gitignore" not in result


def test_the_map_rides_into_the_wrap():
    wrapped = suggest.wrap_ask("make it faster", "a.py: f()",
                               file_map="a.py — the frobnicator")

    assert "a.py — the frobnicator" in wrapped
    assert "map of the project" in wrapped


# -- does this already exist? ----------------------------------------------
# The most-measured failure of the pipeline: models proposing guards the code
# already has. This is the grep we ran by hand all day, automated at both
# moments it helped.
GUARDED = '''\
def polish_draft(template_text, subject):
    """Returns (text_or_None, reason)."""
    if not template_text:
        return None, "empty_template"
    voice = (subject or "").strip()
    return voice, "ok"

def unguarded(rows):
    total = rows[0].count
    return total
'''
TEXTS2 = {"draft.py": GUARDED}


def test_function_source_extracts_the_named_body():
    src = suggest.function_source(GUARDED, "polish_draft")

    assert src.startswith("def polish_draft")
    assert "empty_template" in src
    assert "unguarded" not in src


def test_function_source_handles_a_missing_name():
    assert suggest.function_source(GUARDED, "imaginary") == ""


def test_evidence_puts_the_named_function_under_the_models_nose():
    change = Change("Add a guard clause to polish_draft() for an empty "
                    "template_text", "draft.py")

    evidence = suggest.existing_evidence(change, TEXTS2)

    assert "Current body of polish_draft()" in evidence
    assert "empty_template" in evidence


def test_evidence_greps_the_claims_distinctive_terms():
    change = Change("Handle empty template_text in polish_draft()",
                    "draft.py")

    evidence = suggest.existing_evidence(change, TEXTS2)

    assert "line " in evidence
    assert "template" in evidence.lower()


def test_claim_terms_drop_the_edit_verbs():
    terms = suggest.claim_terms(
        "Add a guard clause to check that template_text is not empty")

    assert "guard" not in terms and "check" not in terms and "add" not in terms
    assert any("template" in t for t in terms)


def test_a_guard_claim_against_a_guarded_function_is_flagged():
    """The five-for-five case from live testing."""
    change = Change("Add a guard clause to polish_draft() to return early "
                    "when template_text is empty", "draft.py")

    note = suggest.may_already_exist(change, TEXTS2)

    assert "polish_draft() already carries guards" in note


def test_a_guard_claim_against_an_unguarded_function_stays_silent():
    change = Change("Add a check to unguarded() for an empty rows list",
                    "draft.py")

    assert suggest.may_already_exist(change, TEXTS2) == ""


def test_a_non_guard_claim_is_never_flagged():
    change = Change("Add a docstring to polish_draft() describing the "
                    "return shape", "draft.py")

    assert suggest.may_already_exist(change, TEXTS2) == ""


def test_the_second_pass_prompt_carries_the_evidence_and_the_exit():
    from aichatlab.intent import build_change_prompt

    prompt = build_change_prompt(
        Change("guard polish_draft() for empty input", "draft.py"),
        GUARDED, "INSTRUCTIONS",
        evidence="Current body of polish_draft():\n...")

    assert "READ THIS BEFORE WRITING" in prompt
    assert "Current body of polish_draft()" in prompt
    assert "NOCHANGE" in prompt


def test_no_evidence_means_no_extra_section():
    from aichatlab.intent import build_change_prompt

    prompt = build_change_prompt(
        Change("guard something()", "draft.py"), GUARDED, "INSTRUCTIONS")

    assert "READ THIS BEFORE WRITING" not in prompt
