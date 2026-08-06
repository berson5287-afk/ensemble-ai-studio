"""Turning "here's what I'd change" into changes the app can actually make.

A 14B model asked to explain a change *and* quote the exact lines it touches
will, reliably, do the first and skip the second.  Both jobs want the same
attention and the same tokens, and the prose is the one it is better at, so
the prose is what you get:

    These changes have been applied to the specified files. Here's a summary:
    1. Enhanced Feature Extraction in `classify_engine.py` by refining regex.
    2. Increased Minimum Confidence Threshold in `auto_engine.py` to 0.90.

Nothing was applied to anything — the model has no hands — but that list is
not worthless.  It is a perfectly good statement of intent, produced by a
model that had the files in front of it, and it is exactly what a person
would want to approve or refuse.  The mistake was treating it as a failed
edit rather than as the first half of one.

So this module reads that list, and the app offers to go back and ask for
each item on its own: one file, one change, the whole token allowance, no
prose asked for.  That second request is a much easier question than the
first, which is why it works where a single combined attempt does not.

The approval the user gives here is approval of the *intent*, and it is not
what puts anything on disk — the diff review still stands between the model
and the filesystem.  A description is not a diff, and agreeing to one is not
agreeing to the other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A list item, numbered or bulleted.  Models mix the two within one answer.
LIST_ITEM = re.compile(r"^\s{0,6}(?:\d{1,2}[.)]|[-*•])\s+(?P<body>.+)$")

# A markdown heading, which is the *other* shape this list arrives in and was
# for a long time the one that got dropped.  A model writing a long answer
# stops numbering and starts sectioning:
#
#     ### Modifying `replypilot_draft_engine.py`
#     #### Step 2: Check for `pywin32` Availability
#
# Those are statements of intent naming a real file, structurally identical to
# a numbered item and every bit as approvable — and reading only lists meant a
# reply full of them offered the user nothing at all to press.
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<body>.+?)\s*#*\s*$")

# ``` … ``` — everything inside is the model's code, not its plan.  Without
# this a heading scan reads every `# comment` in a Python block as a change,
# which is worse than reading none: the list fills with noise and the real
# items are buried in it.
FENCE = re.compile(r"^\s{0,3}(?:```|~~~)")

# `auto_engine.py` or a bare auto_engine.py, with or without a folder in front.
BACKTICKED = re.compile(r"`([^`\n]+?)`")
BARE_PATH = re.compile(r"\b([\w.\-/\\]+\.[A-Za-z]{1,6})\b")

# Markdown that carries no meaning once the item is an instruction again.
DECORATION = re.compile(r"\*\*|__|`")

MAX_CHANGES = 12
MIN_DESCRIPTION = 12


@dataclass(frozen=True)
class Change:
    """One thing the model said it would do, and the file it named."""

    description: str
    file: str

    @property
    def summary(self) -> str:
        """Short enough for a button's worth of context."""
        text = self.description
        return text if len(text) <= 90 else text[:87].rsplit(" ", 1)[0] + "…"


def _known_by_basename(known) -> dict[str, str]:
    """Bare filename → the name the folder scan actually uses.

    The model writes `auto_engine.py`; the scan knows it as
    `replypilot/auto_engine.py`.  A basename claimed by two different files
    is dropped rather than guessed at — picking one of them silently is how
    an edit lands in the wrong file.
    """
    seen: dict[str, list[str]] = {}
    for name in known or ():
        bare = str(name).replace("\\", "/").rsplit("/", 1)[-1]
        seen.setdefault(bare, []).append(str(name))
    return {bare: names[0] for bare, names in seen.items() if len(names) == 1}


def _file_in(text: str, known: dict[str, str]) -> str:
    """The project file this item is about, or "" if it names none.

    Backticked candidates are read first, because a model that marks up a
    filename has told you which token it meant.
    """
    candidates = BACKTICKED.findall(text) + BARE_PATH.findall(text)
    for candidate in candidates:
        cleaned = candidate.strip().strip(".,;:()[]").replace("\\", "/")
        if cleaned in known.values():
            return cleaned
        bare = cleaned.rsplit("/", 1)[-1]
        if bare in known:
            return known[bare]
    return ""


def _items(reply: str) -> list[str]:
    """List items, with their wrapped continuation lines folded back in.

    Worth the extra pass, because the wrap is not where the unimportant
    words live.  A real one:

        3. **Increased Minimum Confidence Threshold** in `auto_engine.py`
           from 0.85 to 0.90 for cautious classification.

    Read line by line, that change becomes "Increased Minimum Confidence
    Threshold in auto_engine.py" — an instruction with the number taken out
    of it, which a model would then be free to answer with any number it
    liked.  A continuation is an indented, non-empty line that is not itself
    a new item.
    """
    items: list[str] = []
    open_item = False
    fenced = False
    for line in (reply or "").splitlines():
        if FENCE.match(line):
            fenced = not fenced
            open_item = False
            continue
        if fenced:
            continue
        match = LIST_ITEM.match(line)
        if match:
            items.append(match.group("body").strip())
            open_item = True
            continue
        heading = HEADING.match(line)
        if heading:
            # A heading owns no continuation lines: the prose under it is
            # explanation, and folding a paragraph into the description would
            # turn one change into an essay the second pass has to re-read.
            items.append(heading.group("body").strip())
            open_item = False
            continue
        if open_item and line.strip() and line[:1] in " \t":
            items[-1] = f"{items[-1]} {line.strip()}"
            continue
        open_item = False
    return items


def parse_changes(reply: str, known=()) -> list[Change]:
    """The changes a reply describes, each tied to a file in the project.

    An item that names no file in the project is dropped rather than kept
    with a guess attached: "refactor the error handling" is not something
    this app can go and ask for, because it does not know where.  And a list
    where *nothing* names a project file is not a list of changes at all —
    it is an ordinary numbered answer that happens to be near a folder.
    """
    lookup = _known_by_basename(known)
    if not lookup:
        return []

    changes: list[Change] = []
    seen: set[tuple[str, str]] = set()
    for raw in _items(reply):
        body = DECORATION.sub("", raw).strip()
        if len(body) < MIN_DESCRIPTION:
            continue
        name = _file_in(raw, lookup)
        if not name:
            continue
        key = (name, body.lower())
        if key in seen:
            continue
        seen.add(key)
        changes.append(Change(description=body, file=name))
        if len(changes) >= MAX_CHANGES:
            break
    return changes


def describe(changes) -> str:
    """"3 changes across 2 files", for the line above the buttons."""
    count = len(changes)
    files = len({change.file for change in changes})
    if not count:
        return "No changes"
    part = "1 change" if count == 1 else f"{count} changes"
    if files > 1:
        return f"{part} across {files} files"
    return part


def build_change_prompt(change: Change, current: str, instructions: str,
                        evidence: str = "") -> str:
    """Ask for one change, on one file, with nothing else wanted.

    Everything here is aimed at the failure this exists to fix.  The file is
    included in full and marked as the only acceptable source of the FIND
    lines, because quoting from memory is what breaks anchored edits.  Prose
    is refused outright rather than merely discouraged, because a model given
    permission to explain will spend its allowance explaining and stop before
    the block is closed.  And the change is stated back to it verbatim, so
    the request is "do this one thing" rather than "decide again".
    """
    already = ""
    if evidence:
        # The model is handed the whole file and demonstrably does not read
        # the part it is about to rewrite — measured live, every redundant
        # guard it produced already existed in the function it named.  So
        # the function is put directly under its nose, with permission to
        # say so if the work is already done.
        already = (
            f"READ THIS BEFORE WRITING ANYTHING. What the file already "
            f"contains that bears on this change:\n\n{evidence}\n\n"
            f"If what was asked for is already present above — an existing "
            f"guard, an existing default, an existing check that covers it — "
            f"reply with the single word NOCHANGE. Do not restate an "
            f"existing protection in new words; that is not a change, and "
            f"it will be refused.\n\n")
    return (
        f"You already looked at this project and said you would make this "
        f"change:\n\n    {change.description}\n\n{already}"
        f"Here is the current, complete contents of {change.file}. The lines "
        f"you quote must be copied from it exactly — character for "
        f"character, including indentation and blank lines. Do not retype "
        f"them from memory and do not tidy them up on the way past.\n\n"
        f"--- {change.file} ---\n{current}\n--- end of {change.file} ---\n\n"
        f"Make exactly that one change to {change.file}, and nothing else.\n\n"
        f"The REPLACE section must differ from the FIND section. Copying the "
        f"lines back unaltered is not an edit — it is refused, and it wastes "
        f"the request. If the change is already present in the file, or the "
        f"instruction was too vague for you to know what to write, reply with "
        f"the single word NOCHANGE and no block at all. That is a useful "
        f"answer; a block that changes nothing is not.\n\n"
        f"Reply with the edit block and nothing else: no explanation, no "
        f"preamble, no summary afterwards, and no claim that you have saved "
        f"anything. The block is the whole reply.\n\n"
        f"{instructions}")


# The model's way of saying "there is nothing here to do", which is a real
# answer and needs to read as one rather than as a failure.
NO_CHANGE = re.compile(r"^\s*NOCHANGE\b", re.IGNORECASE | re.MULTILINE)


def said_no_change(reply: str) -> bool:
    """Did the model decline, having been given a way to decline?"""
    text = (reply or "").strip()
    return bool(NO_CHANGE.search(text)) and "=== EDIT:" not in text


def worth_offering(reply: str, known=()) -> bool:
    """Is there anything here worth going back to the model about?"""
    return bool(parse_changes(reply, known))


def from_question(question: str, reply: str, known=()) -> list[Change]:
    """A change to work on when the *reply* named no file but the ask did.

    The commonest shape of a failed edit request is not a model that refused.
    It is "fix the retry logic in `mail_engine.py`" answered with four
    paragraphs of advice that never mention the filename again. Read only the
    reply, there is nothing to place; read the question too and the target is
    obvious, and the second pass — one file, its real contents, prose refused
    — is exactly the request that succeeds where the first one did not.

    Deliberately at most one change. The question named one thing to work on,
    and inventing several out of a sentence would be guessing at a plan the
    user never wrote.
    """
    lookup = _known_by_basename(known)
    if not lookup:
        return []
    name = _file_in(question or "", lookup)
    if not name:
        return []
    said = DECORATION.sub("", question or "").strip()
    if len(said) < MIN_DESCRIPTION:
        return []
    return [Change(description=said, file=name)]
