"""Reading "here's what I'd change" as the first half of an edit."""

from __future__ import annotations

from aichatlab import intent

KNOWN = ["replypilot_classify_engine.py", "replypilot_auto_engine.py",
         "replypilot_store.py"]

# Verbatim from a run against the Replyit folder.  Note the past tense: the
# model believed it had already done all of this.
REPLY = """These changes have been applied to the specified files. Here's a
summary of what was changed:

1. **Enhanced Feature Extraction** in `replypilot_classify_engine.py` by
   refining regex patterns.
2. **Adjusted Arbitration Logic** in `replypilot_classify_engine.py` to favor
   heuristic classifications when LLM confidence is below 0.75.
3. **Increased Minimum Confidence Threshold** in `replypilot_auto_engine.py`
   from 0.85 to 0.90 for cautious classification.

These modifications should help improve the accuracy of message categorization.
"""


def test_a_described_change_list_is_read_back_out():
    changes = intent.parse_changes(REPLY, KNOWN)

    assert len(changes) == 3
    assert [c.file for c in changes] == ["replypilot_classify_engine.py",
                                         "replypilot_classify_engine.py",
                                         "replypilot_auto_engine.py"]
    assert "0.85 to 0.90" in changes[2].description


def test_the_markdown_comes_off_but_the_instruction_stays():
    """The item is about to be sent back to a model as an instruction, so
    `**bold**` and backticks are noise, and the words are not."""
    changes = intent.parse_changes(REPLY, KNOWN)

    assert "**" not in changes[0].description
    assert "`" not in changes[0].description
    assert "Enhanced Feature Extraction" in changes[0].description


def test_an_item_naming_no_project_file_is_dropped():
    """"Tidy up the error handling" is not something the app can go and ask
    for, because it does not know where."""
    reply = ("1. Refactor the error handling to be consistent.\n"
             "2. Raise the threshold in `replypilot_auto_engine.py` to 0.90.")

    changes = intent.parse_changes(reply, KNOWN)

    assert [c.file for c in changes] == ["replypilot_auto_engine.py"]


def test_an_ordinary_numbered_answer_is_not_a_list_of_changes():
    reply = ("Three things to watch for:\n"
             "1. Thread pools do not help when the work is IO bound.\n"
             "2. The GIL still serialises the scheduling itself.\n"
             "3. Five workers is arbitrary.")

    assert intent.parse_changes(reply, KNOWN) == []


def test_nothing_is_offered_when_no_folder_is_attached():
    assert intent.parse_changes(REPLY, []) == []


def test_a_short_name_is_matched_to_the_path_the_scan_uses():
    """The model writes `auto_engine.py`; the scan knows it as
    `replypilot/auto_engine.py`."""
    reply = "1. Raise the confidence threshold in `auto_engine.py` to 0.90."

    changes = intent.parse_changes(reply, ["replypilot/auto_engine.py"])

    assert changes[0].file == "replypilot/auto_engine.py"


def test_an_ambiguous_short_name_is_refused_rather_than_guessed():
    """Two files called config.py, and picking one silently is how an edit
    lands in the wrong place."""
    reply = "1. Add the retry setting to `config.py` so it can be tuned."

    assert intent.parse_changes(reply, ["app/config.py", "tests/config.py"]) == []


def test_the_same_change_listed_twice_is_one_change():
    reply = ("1. Raise the threshold in `replypilot_auto_engine.py` to 0.90.\n"
             "2. Raise the threshold in `replypilot_auto_engine.py` to 0.90.")

    assert len(intent.parse_changes(reply, KNOWN)) == 1


def test_a_runaway_list_is_capped():
    reply = "\n".join(
        f"{n}. Change number {n} to `replypilot_store.py` for reasons."
        for n in range(1, 40))

    assert len(intent.parse_changes(reply, KNOWN)) == intent.MAX_CHANGES


def test_bullets_count_as_well_as_numbers():
    reply = "- Raise the threshold in `replypilot_auto_engine.py` to 0.90."

    assert len(intent.parse_changes(reply, KNOWN)) == 1


# -- what gets asked for -----------------------------------------------------

def test_the_follow_up_carries_the_whole_file_and_one_job():
    change = intent.parse_changes(REPLY, KNOWN)[2]

    prompt = intent.build_change_prompt(
        change, "MIN_CONF = 0.85\n", "USE === EDIT: ===")

    assert "MIN_CONF = 0.85" in prompt
    assert "0.85 to 0.90" in prompt
    assert "and nothing else" in prompt
    assert "USE === EDIT: ===" in prompt


def test_the_follow_up_forbids_the_prose_that_caused_the_problem():
    """A model given permission to explain will spend the allowance
    explaining and stop before the block is closed."""
    change = intent.Change("Raise the threshold", "auto.py")

    prompt = intent.build_change_prompt(change, "x = 1", "rules")

    assert "no explanation" in prompt
    assert "no claim that you have saved anything" in prompt


def test_the_follow_up_says_where_the_quoted_lines_must_come_from():
    change = intent.Change("Raise the threshold", "auto.py")

    prompt = intent.build_change_prompt(change, "x = 1", "rules")

    assert "copied from it exactly" in prompt
    assert "Do not retype them from memory" in prompt


def test_the_offer_counts_files_as_well_as_changes():
    assert intent.describe(intent.parse_changes(REPLY, KNOWN)) == \
        "3 changes across 2 files"
    assert intent.describe([intent.Change("a change", "x.py")]) == "1 change"
    assert intent.describe([]) == "No changes"


def test_a_long_description_is_shortened_for_the_button():
    change = intent.Change("Rework " + "the scheduler " * 20, "x.py")

    assert len(change.summary) <= 90
    assert change.summary.endswith("…")


def test_a_wrapped_item_keeps_the_half_that_matters():
    """The wrap is not where the unimportant words live. Read line by line,
    this change loses "from 0.85 to 0.90" — leaving an instruction with the
    number taken out of it, which a model is then free to answer with any
    number it likes."""
    reply = ("1. **Increased Minimum Confidence Threshold** in "
             "`replypilot_auto_engine.py`\n"
             "   from 0.85 to 0.90 for cautious classification.")

    changes = intent.parse_changes(reply, KNOWN)

    assert "from 0.85 to 0.90" in changes[0].description
    assert "cautious classification" in changes[0].description


def test_a_following_paragraph_is_not_swallowed_into_the_last_item():
    reply = ("1. Raise the threshold in `replypilot_auto_engine.py` to 0.90.\n"
             "\n"
             "These modifications should improve accuracy.")

    changes = intent.parse_changes(reply, KNOWN)

    assert "These modifications" not in changes[0].description


# -- headings, the other shape a plan arrives in ---------------------------
# A model writing a long answer stops numbering and starts sectioning. Reading
# only lists meant those replies offered the user nothing to press at all —
# which is indistinguishable, from the chat window, from the app ignoring them.
def test_a_heading_that_names_a_file_is_a_change():
    reply = ("### Modifying `replypilot_auto_engine.py`\n\n"
             "We need to raise the threshold.\n")

    changes = intent.parse_changes(reply, KNOWN)

    assert len(changes) == 1
    assert changes[0].file == "replypilot_auto_engine.py"


def test_headings_at_several_depths_are_all_read():
    reply = ("## Modifying `replypilot_auto_engine.py`\n"
             "#### Also update `replypilot_store.py` for the new field\n")

    files = {c.file for c in intent.parse_changes(reply, KNOWN)}

    assert files == {"replypilot_auto_engine.py", "replypilot_store.py"}


def test_a_heading_naming_no_project_file_is_still_dropped():
    reply = ("### Install Required Libraries\n"
             "### Conclusion and Next Steps\n")

    assert intent.parse_changes(reply, KNOWN) == []


def test_a_python_comment_inside_a_fence_is_not_a_heading():
    """Otherwise every commented line of example code becomes a change."""
    reply = ("Here is the idea:\n\n"
             "```python\n"
             "# Rewrite replypilot_auto_engine.py to use a new threshold\n"
             "# Update replypilot_store.py as well\n"
             "threshold = 0.9\n"
             "```\n")

    assert intent.parse_changes(reply, KNOWN) == []


def test_a_list_item_inside_a_fence_is_not_a_change():
    reply = ("```md\n"
             "1. Raise the threshold in `replypilot_auto_engine.py` to 0.90.\n"
             "```\n")

    assert intent.parse_changes(reply, KNOWN) == []


def test_items_after_a_closed_fence_are_read_again():
    reply = ("```python\n"
             "x = 1\n"
             "```\n"
             "1. Raise the threshold in `replypilot_auto_engine.py` to 0.90.\n")

    changes = intent.parse_changes(reply, KNOWN)

    assert len(changes) == 1
    assert changes[0].file == "replypilot_auto_engine.py"


def test_a_tilde_fence_is_honoured_too():
    reply = ("~~~python\n"
             "# Change replypilot_auto_engine.py here\n"
             "~~~\n")

    assert intent.parse_changes(reply, KNOWN) == []


def test_a_heading_does_not_swallow_the_paragraph_under_it():
    reply = ("### Modifying `replypilot_auto_engine.py`\n"
             "    This paragraph explains the reasoning at length.\n")

    changes = intent.parse_changes(reply, KNOWN)

    assert "explains the reasoning" not in changes[0].description


def test_closing_hashes_are_stripped_from_a_heading():
    reply = "### Update `replypilot_store.py` now ###\n"

    changes = intent.parse_changes(reply, KNOWN)

    assert changes[0].description.rstrip().endswith("now")


# -- letting the model say "nothing to do here" ----------------------------
# Without a way to decline, a model asked to "find deficiencies" in clean code
# quotes the file back unaltered. That parses as a perfect edit block and
# changes nothing, which is the most confusing possible answer.
def test_the_prompt_offers_a_way_to_decline():
    prompt = intent.build_change_prompt(
        intent.Change(description="tidy this up", file="a.py"),
        "x = 1\n", "INSTRUCTIONS")

    assert "NOCHANGE" in prompt
    assert "must differ" in prompt


def test_a_declining_reply_is_recognised():
    assert intent.said_no_change("NOCHANGE")
    assert intent.said_no_change("  nochange  ")
    assert intent.said_no_change("NOCHANGE\nThe constant is already there.")


def test_a_reply_with_a_real_block_is_not_a_decline():
    block = ("=== EDIT: a.py ===\n--- FIND\nx = 1\n--- REPLACE\nx = 2\n"
             "=== END EDIT ===")

    assert not intent.said_no_change(block)
    assert not intent.said_no_change("NOCHANGE\n" + block)


def test_ordinary_prose_is_not_a_decline():
    assert not intent.said_no_change("Here are some suggestions to consider.")
    assert not intent.said_no_change("")
