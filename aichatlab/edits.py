"""Turning "here is what I would change" into files on disk, carefully.

Everything else in this app is read-only.  This is the one feature that can
destroy a morning's work, so it is built to fail closed: it refuses more
readily than it writes, and nothing reaches the disk without a human looking
at a diff first.

Three decisions shape the whole module.

*Whole files, not patches.*  A local 14B model asked for a unified diff
produces invalid hunks often enough to be useless, and the failure is quiet —
a patch that does not apply, or worse, one that applies in the wrong place.
Asked for the complete new contents of a file it succeeds nearly every time,
and when it does fail the damage is visible in the diff rather than hidden in
a line number.

*A truncated reply is the dangerous case.*  Replies get cut off at the length
cap all the time; this app already detects and continues them.  A cut-off
edit means the last file block simply stops mid-function, and writing that
would silently delete the rest of the file.  So a block without its closing
marker is discarded, loudly.

*Nothing outside the folder.*  Paths are resolved and checked against the
project root before anything is opened, because "../.." in a model-generated
filename is not a hypothetical.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

BEGIN = "=== FILE:"
END = "=== END FILE ==="

BLOCK = re.compile(
    r"^===\s*FILE:\s*(?P<name>[^\n=]+?)\s*===\s*\n"
    r"(?P<body>.*?)"
    r"^===\s*END\s+FILE\s*===\s*$",
    re.DOTALL | re.MULTILINE)

OPENER = re.compile(r"^===\s*FILE:\s*(?P<name>[^\n=]+?)\s*===[ \t]*$",
                    re.MULTILINE)
CLOSER = re.compile(r"^===\s*END\s+FILE\s*===[ \t]*$", re.MULTILINE)

# The way a small model destroys a file: it writes the part it was thinking
# about and leaves a note where the rest used to be.  The block is complete,
# the markers are correct, the content is a placeholder — and applying it
# deletes everything the model could not be bothered to repeat.
#
# This is not a rare failure.  It is the single most common way a 14B answers
# "give me the whole file", and it is far more dangerous than a truncated
# reply because nothing about it looks broken.
# Strips the comment scaffolding off a line so what remains can be judged on
# its words alone.  One enormous alternation covering every comment syntax and
# every phrasing was unreadable and quietly missed three of the eleven cases
# it was written for.
COMMENT_EDGES = re.compile(
    r"^[ \t]*(?:#+|//+|--|/\*+|<!--|;+|\*)?[ \t]*"
    r"(?P<body>.*?)"
    r"[ \t]*(?:\*/|-->)?[ \t]*$")

# "and the rest goes here", in the forms models actually write it.
REST_OF = re.compile(
    r"^(?:the\s+)?(?:rest|remainder|remaining|others?)\s+of\s+(?:the\s+)?"
    r"(?:code|file|class|function|method|module|implementation|contents?|"
    r"logic|body|imports?)"
    r"(?:\s+(?:unchanged|is\s+unchanged|as\s+before|goes?\s+here|here|"
    r"below|above|follows?|remains?))?$",
    re.IGNORECASE)

BARE_ELISIONS = {
    "unchanged", "no change", "no changes", "existing code", "same as before",
    "same as above", "as before", "as above", "snip", "snipped", "omitted",
    "truncated", "etc", "and so on", "unchanged code", "existing imports",
    "keep existing code", "rest unchanged", "remaining unchanged",
}

# How much of a file may vanish before a rewrite stops looking like an edit
# and starts looking like an accident.
SHRINK_FLOOR = 0.5
MIN_LINES_TO_JUDGE = 25


def _bare(line: str) -> str:
    """A line reduced to its words: comment markers, ellipses and brackets
    removed.  "# (… rest of the class …)" becomes "rest of the class"."""
    match = COMMENT_EDGES.match(line)
    body = match.group("body") if match else line
    body = body.replace("…", " ").replace("...", " ")
    body = re.sub(r"[()\[\]{}]", " ", body)
    body = re.sub(r"[\s,.;:!-]+$", "", body.strip())
    return re.sub(r"\s+", " ", body).strip().lower()


# The other half of the placeholder vocabulary, and the half that nearly got
# a project gutted a second time.  A model asked to reproduce a 200-line file
# it cannot hold in its head writes a plausible skeleton instead, marking each
# hole with a short trailing-ellipsis comment: "# Other necessary imports...",
# "# Function implementation...", "# Configuration attributes...".  None of
# them say "rest of the file", and every one of them means it.
STUB_NOUNS = (
    "import", "implementation", "attribute", "config", "configuration",
    "code", "method", "function", "class", "field", "property", "definition",
    "setup", "initialisation", "initialization", "logic", "detail", "member",
    "variable", "constant", "handler", "helper", "option", "setting",
    "parameter", "argument", "docstring", "body", "content",
)


def _is_stub_comment(line: str) -> bool:
    """A short comment ending in "…" that names a category, not a thing.

    Length is what keeps this honest: "# Other necessary imports..." is four
    words and a category, while "# we retry here because the first import
    can race the loader..." is prose that happens to trail off.
    """
    stripped = line.strip()
    if not re.search(r"(\.\.\.|…)\s*(?:\*/|-->)?$", stripped):
        return False
    words = _bare(line).split()
    if not words or len(words) > 5:
        return False
    return any(word.rstrip("s").endswith(noun) or noun in word
               for word in words for noun in STUB_NOUNS)


def _is_elision(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    # A line of real code is not a comment; only comment-shaped lines can be
    # placeholders, or the check would refuse `...` in valid Python.
    if not re.match(r"^[ \t]*(?:#|//|--|/\*|<!--|;|\*)", line):
        return False
    words = _bare(line)
    if not words:
        # A comment containing nothing but dots — "# ..." — is the shortest
        # way a model says "and the rest".
        return bool(re.search(r"\.\.\.|…", stripped))
    if words in BARE_ELISIONS:
        return True
    if REST_OF.match(words):
        return True
    return _is_stub_comment(line)


def elided(text: str) -> str:
    """The placeholder line that means "and the rest", or ""."""
    for line in (text or "").splitlines():
        if _is_elision(line):
            return line.strip()
    return ""


# Models wrap the body in a code fence about half the time.
FENCE = re.compile(r"\A\s*```[^\n]*\n(?P<inner>.*?)\n?```\s*\Z", re.DOTALL)

WHOLE_FILE_INSTRUCTIONS = f"""\
To create a *new* file, give its complete contents:

{BEGIN} <path exactly as shown in the folder listing> ===
<the entire new contents of the file>
{END}

Rules that matter:
- Only for files that do not exist yet. To change an existing file, use the
  anchored EDIT form above instead — it is far more reliable.
- Every block MUST end with the {END} line.
"""


@dataclass(frozen=True)
class Edit:
    """One file the model wants to replace."""

    name: str
    new_text: str

    @property
    def lines(self) -> int:
        return len(self.new_text.splitlines())


@dataclass(frozen=True)
class Rejected:
    """Something that looked like an edit and will not be applied."""

    name: str
    reason: str


def _strip_fence(body: str) -> str:
    match = FENCE.match(body)
    return match.group("inner") if match else body


def _recover_unterminated(text: str) -> list[tuple[str, str, bool]]:
    """Blocks the model opened and never closed: (name, body, bounded).

    Forgetting `=== END FILE ===` is much commoner than being cut off, and
    the two are not the same problem.  When another `=== FILE:` follows, the
    first block's end is not a guess — it is exactly where the next one
    starts, so the contents are complete and recoverable.  Only the *last*
    unclosed block is genuinely ambiguous, and only that one depends on
    whether the reply ran out of room.
    """
    starts = [(m.start(), m.end(), m.group("name").strip())
              for m in OPENER.finditer(text)]
    found = []
    for index, (_begin, after, name) in enumerate(starts):
        following = starts[index + 1][0] if index + 1 < len(starts) else None
        body = text[after:following] if following is not None else text[after:]
        # A closed block is handled elsewhere; this is only for the rest.
        if CLOSER.search(body):
            continue
        found.append((name, body, following is not None))
    return found


def parse(reply: str, truncated: bool = False) -> tuple[list[Edit], list[Rejected]]:
    """(edits, rejected) — never raises, and never guesses.

    `truncated` is what the server said about the reply hitting its length
    cap.  It is the difference between "the model forgot a marker" and "the
    model was interrupted", which look identical in the text and need
    opposite answers.
    """
    text = reply or ""
    edits: list[Edit] = []
    rejected: list[Rejected] = []
    seen: set[str] = set()

    for match in BLOCK.finditer(text):
        name = match.group("name").strip()
        body = _strip_fence(match.group("body"))
        if not name:
            continue
        if name in seen:
            rejected.append(Rejected(name, "the same file was given twice"))
            continue
        seen.add(name)
        if not body.strip():
            # A model that returns an empty file has almost always lost its
            # place, and "replace this file with nothing" is never what was
            # meant by "add a docstring".
            rejected.append(Rejected(name, "the new contents came back empty"))
            continue
        placeholder = elided(body)
        if placeholder:
            rejected.append(Rejected(
                name, f"the model wrote “{placeholder}” instead of the rest "
                      f"of the file — applying that would delete everything "
                      f"it did not repeat"))
            continue
        edits.append(Edit(name=name, new_text=body))

    for name, body, bounded in _recover_unterminated(text):
        if not name or name in seen:
            continue
        seen.add(name)
        body = _strip_fence(body).rstrip()
        if not body.strip():
            rejected.append(Rejected(name, "the new contents came back empty"))
            continue
        placeholder = elided(body)
        if placeholder:
            rejected.append(Rejected(
                name, f"the model wrote “{placeholder}” instead of the rest "
                      f"of the file — applying that would delete everything "
                      f"it did not repeat"))
            continue
        if bounded:
            # Its end is where the next file begins: not a guess.
            edits.append(Edit(name=name, new_text=body + "\n"))
            continue
        if truncated:
            rejected.append(Rejected(
                name, "the reply hit its length cap before this file was "
                      "finished — writing it would have deleted the rest of "
                      "the file. Press Continue, or raise the answer length "
                      "on the speed slider, and ask again"))
            continue
        # The reply finished; the model simply left the closing marker off
        # the last block.  The contents are all here, and the diff is about
        # to be shown to a human either way.
        edits.append(Edit(name=name, new_text=body + "\n"))
    return edits, rejected


# -- safety ----------------------------------------------------------------
def resolve(root, name: str) -> Path | None:
    """The real path this edit refers to, or None if it escapes the folder.

    `..` in a model-generated filename is not hypothetical, and neither is an
    absolute path — both have to be refused before anything is opened.
    """
    try:
        base = Path(root).resolve()
        candidate = (base / name).resolve()
    except (OSError, ValueError):
        return None
    if Path(name).is_absolute():
        return None
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def check(root, edits) -> tuple[list, list[Rejected]]:
    """Split edits into those that may be written and those that may not."""
    allowed, refused = [], []
    for edit in edits:
        target = resolve(root, edit.name)
        if target is None:
            refused.append(Rejected(
                edit.name, "that path is outside the attached folder"))
            continue
        if target.is_dir():
            refused.append(Rejected(edit.name, "that is a directory"))
            continue
        shrink = shrinkage(root, edit)
        if shrink:
            refused.append(Rejected(edit.name, shrink))
            continue
        allowed.append(edit)
    return allowed, refused


def shrinkage(root, edit: Edit) -> str:
    """A complaint when a rewrite loses most of the file, or "".

    The elision check catches a model that *says* it left the rest out.  This
    catches the one that simply stops, with no marker and no truncation —
    same outcome, no tell.
    """
    before = read_current(root, edit.name)
    if not before:
        return ""                       # a new file cannot shrink
    old_lines = len(before.splitlines())
    if old_lines < MIN_LINES_TO_JUDGE:
        return ""                       # too small to draw a conclusion from
    new_lines = len(edit.new_text.splitlines())
    if new_lines >= old_lines * SHRINK_FLOOR:
        return ""
    return (f"this would cut the file from {old_lines:,} lines to "
            f"{new_lines:,} — that is a rewrite losing most of the file "
            f"rather than an edit, which is what a model does when it "
            f"summarises instead of reproducing")


def new_folders(root, edit) -> list[str]:
    """Directories this edit would bring into existence, outermost first.

    A new file inside a folder that already exists is ordinary — a test
    beside its tests, a module beside its siblings.  A new file that also
    invents the folder around it is a different animal, and it is worth
    stopping on, because it is what a *confused* model produces rather than
    a wrong one.

    The case this comes from: prior lessons from another project leaked into
    the prompt, the model concluded it was looking at that project, and
    proposed `udbg/cache.py` — correct-looking, entirely reasonable in the
    app it was thinking of, and written into a folder with no `udbg` in it.
    Nothing about the diff of a brand-new file gives that away; the diff is
    all additions and every line looks fine. The path is the only tell, so
    the path is what gets checked.
    """
    target = resolve(root, edit.name)
    if target is None or target.exists():
        return []
    base = Path(root).resolve()
    missing: list[str] = []
    for parent in reversed(target.parents):
        try:
            inside = parent.relative_to(base)
        except ValueError:
            continue                       # at or above the project root
        if inside == Path(".") or parent.exists():
            continue
        missing.append(inside.as_posix())
    return missing


def novelty(root, edit) -> str:
    """A sentence about what is unfamiliar here, or "" if nothing is."""
    folders = new_folders(root, edit)
    if not folders:
        return ""
    listed = ", ".join(f"{name}/" for name in folders)
    plural = "folder" if len(folders) == 1 else "folders"
    return (f"this would also create the {plural} {listed}, which "
            f"{'does' if len(folders) == 1 else 'do'} not exist in the "
            f"attached project — check it belongs here")


# -- previewing ------------------------------------------------------------
def read_current(root, name: str) -> str:
    target = resolve(root, name)
    if target is None or not target.is_file():
        return ""
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def diff_for(root, edit: Edit) -> str:
    """A unified diff of what would change, for a human to read."""
    before = read_current(root, edit.name)
    label = "new file" if not before else edit.name
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        edit.new_text.splitlines(keepends=True),
        fromfile=f"a/{edit.name}" if before else "/dev/null",
        tofile=f"b/{edit.name}", n=3)
    body = "".join(lines)
    if not body.strip():
        return f"{edit.name}: no change\n"
    return body if before else f"# {label}\n{body}"


def counts(root, edit: Edit) -> tuple[int, int]:
    """(added, removed) lines, for a one-line summary."""
    before = read_current(root, edit.name).splitlines()
    after = edit.new_text.splitlines()
    added = removed = 0
    for line in difflib.ndiff(before, after):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return added, removed


def summarise(root, edits) -> str:
    if not edits:
        return "no files changed"
    added = removed = 0
    for edit in edits:
        plus, minus = counts(root, edit)
        added += plus
        removed += minus
    files = f"{len(edits)} file" + ("" if len(edits) == 1 else "s")
    return f"{files} changed, +{added} −{removed}"


def unchanged(root, edit: Edit) -> bool:
    return read_current(root, edit.name) == edit.new_text


# -- writing ---------------------------------------------------------------
BACKUPS = "backups"


def backup_root(root) -> Path:
    from .projectmemory import directory
    return directory(root) / BACKUPS


@dataclass(frozen=True)
class Applied:
    name: str
    created: bool
    backed_up: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def apply(root, edits, stamp: str = "") -> list[Applied]:
    """Write the edits, keeping a copy of whatever was there first.

    Backups are not optional and not configurable.  The whole feature is one
    mistyped filename away from costing someone a day, and a copy on disk is
    the difference between an annoyance and a disaster.
    """
    when = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = backup_root(root) / when
    results: list[Applied] = []
    for edit in edits:
        target = resolve(root, edit.name)
        if target is None:
            results.append(Applied(edit.name, False,
                                   error="outside the attached folder"))
            continue
        existed = target.is_file()
        saved = ""
        try:
            if existed:
                copy = folder / edit.name
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_text(read_current(root, edit.name),
                                encoding="utf-8")
                saved = str(copy)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Written through a temporary file so an interrupted write cannot
            # leave the original half-replaced.
            temporary = target.with_suffix(target.suffix + ".aichatlab-part")
            temporary.write_text(edit.new_text, encoding="utf-8")
            temporary.replace(target)
        except OSError as exc:
            results.append(Applied(edit.name, not existed, saved, str(exc)))
            continue
        results.append(Applied(edit.name, not existed, saved))
    return results


def snapshots(root) -> list[str]:
    """Every backup taken for this project, newest first."""
    base = backup_root(root)
    try:
        return sorted((d.name for d in base.iterdir() if d.is_dir()),
                      reverse=True)
    except OSError:
        return []


def restore(root, stamp: str) -> list[Applied]:
    """Put a snapshot back.

    The backups exist so that a bad edit costs a click rather than an
    evening, and that is only true if putting them back is also a click.
    Restoring takes its own backup first, so undoing an undo is possible too.
    """
    folder = backup_root(root) / stamp
    if not folder.is_dir():
        return [Applied("", False, error=f"no snapshot called {stamp}")]
    edits_back = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(folder)).replace("\\", "/")
        try:
            edits_back.append(Edit(name, path.read_text(encoding="utf-8",
                                                        errors="replace")))
        except OSError:
            continue
    if not edits_back:
        return [Applied("", False, error="that snapshot is empty")]
    return apply(root, edits_back)


def describe_result(results) -> str:
    written = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    created = sum(1 for r in written if r.created)
    parts = []
    if written:
        parts.append(f"✅ Wrote {len(written)} file"
                     + ("" if len(written) == 1 else "s")
                     + (f" ({created} new)" if created else ""))
    if failed:
        parts.append("⚠ Could not write: "
                     + ", ".join(f"{r.name} — {r.error}" for r in failed))
    if written:
        parts.append("The originals are in .aichatlab/backups.")
    return " ".join(parts) or "Nothing was written."


# ---------------------------------------------------------------- patching
"""Targeted edits: the model writes only what changes.

Whole-file replacement asks a model to reproduce every line it is not
changing, and a local 14B cannot do that for a file of any size — it writes a
plausible skeleton and marks the holes, which is how a project gets gutted.
The guards above catch that, but catching it every time is not the same as
being able to do the work.

An anchored edit asks for the five lines that change and where they go.  That
is inside any model's reach, and it has a property whole-file replacement can
never have: **an anchor that does not match fails loudly**.  A model that
paraphrases the code it is anchoring to gets a refusal naming the file, not a
silent overwrite.  The dangerous failure becomes the noisy one.
"""

EDIT_BLOCK = re.compile(
    r"^===\s*EDIT:\s*(?P<name>[^\n=]+?)\s*===[ \t]*\n"
    r"(?P<body>.*?)"
    r"^===\s*END\s+EDIT\s*===[ \t]*$",
    re.DOTALL | re.MULTILINE)

# `[ \t]*` before the dashes: a model quoting indented code indents the
# marker to match — `    --- REPLACE` inside a method body — and requiring
# column zero turned a perfectly good block into "missing its REPLACE part".
FIND_PART = re.compile(
    r"^[ \t]*-{2,}\s*FIND[ \t]*\n(?P<find>.*?)"
    r"(?=^[ \t]*-{2,}\s*(?:REPLACE|INSERT)|\Z)",
    re.DOTALL | re.MULTILINE)
REPLACE_PART = re.compile(
    r"^[ \t]*-{2,}\s*(?P<mode>REPLACE|INSERT AFTER|INSERT BEFORE)[ \t]*\n"
    r"(?P<replace>.*)\Z",
    re.DOTALL | re.MULTILINE)

PATCH_INSTRUCTIONS = """\
HOW TO CHANGE A FILE

You have no hands. You cannot open, save or write anything. The ONLY way a
change reaches the disk is an EDIT block in your reply, exactly as shown here.
Describing a change, or printing the new code in a ``` fence, changes nothing —
the user sees your description, their file stays as it was, and the work is
lost. If you are asked to change a file, the reply must contain EDIT blocks.

DO NOT ASK WHICH FILES — YOU CAN SEE THEM

The files are in front of you. A request like "upgrade my app", "fix the
deficiencies" or "make it better" is a request to *do the work*, not a request
for advice about doing it, and asking which file was meant hands the job back
to the person who asked you to do it.

- Pick the change yourself. Name the file, make the smallest real edit that
  improves it, and write the block. Do that for two or three of the clearest
  problems rather than listing twenty.
- Never reply with only a numbered plan, a checklist, or "here are some steps
  you could consider". A plan the user cannot apply is worth less to them than
  one small change they can.
- Never answer with the finished code in a ``` fence and leave them to paste
  it in. That is the work half-done and it is their time you are spending.
- Only ask a question when you genuinely cannot proceed — a missing password,
  a decision only they can make. "Which file?" is never that question.
- If you are unsure whether a change is wanted, make it anyway and say why in
  one line above the block. They see the diff before anything is written, so a
  proposal costs them a glance and refusing costs them one click.

To change a file, do NOT rewrite it. Give only the lines that change, anchored
to lines that are already there:

=== EDIT: <path exactly as shown in the folder listing> ===
--- FIND
<a few existing lines, copied exactly as they appear in the file>
--- REPLACE
<what those lines should become>
=== END EDIT ===

Use `--- INSERT AFTER` or `--- INSERT BEFORE` in place of `--- REPLACE` to add
lines without removing the anchor.

RULES ABOUT THE BLOCK

- Never put an EDIT block inside a ``` fence. The markers are the syntax;
  a fence around them stops the block being read at all.
- The path is a real path from the listing, like `udbg/config.py`. Never
  write `relative/path/to/` — that is a placeholder, not part of any path.
- Copy the FIND lines character for character from the file. Do not retype them
  from memory and do not tidy them up — if they do not match exactly, the edit
  is refused.
- Include enough lines in FIND to appear exactly once in the file. Three or
  four is usually right; one line is often not.
- Every block MUST end with its own `=== END EDIT ===` line.
- One block per change. Several blocks in one reply is fine, including several
  for the same file.
- Never write "# rest of the code" or similar. Only write real lines.

RULES ABOUT THE CODE

The user approves your change by reading it, so it has to be right on the
page — there is no test run and no second chance before it is written.

- Any variable you introduce must be given a starting value in the same scope,
  on a path that always runs. Adding `skipped += 1` without adding
  `skipped = 0` beside the other counters raises UnboundLocalError every time
  the function is called. This is the single most common way an edit that
  reads perfectly breaks the program.
- Only call functions, methods and attributes you can see in the file or in
  what you were given. Do not invent a helper and do not assume one exists.
- Only compare against values the code can actually produce. If you branch on
  a function's return value, check what that function really returns rather
  than guessing at a new case.
- If a change needs something you cannot see, say so in prose and make no
  block for it. A refused edit costs a retry; a wrong one costs the user's
  working program.

Explain your reasoning outside the blocks, in normal prose. Never claim to
have saved, written or modified a file — you have not, and saying so sends the
user looking at a file that has not changed.
"""

# The worked example is the most convincing part of the instructions and the
# most expensive: it roughly doubles their length, and they sit in the system
# prompt of every message once a folder is attached. So it is kept for the
# second pass — one file, one change, the whole allowance — where the model
# has already failed to produce a block once and the budget is not competing
# with a conversation.
PATCH_EXAMPLE = """\
A COMPLETE EXAMPLE

Suppose the file `app/counter.py` contains:

    def tally(rows):
        n = 0
        for row in rows:
            if row.ok:
                n += 1
        return n

To also count the skipped rows, the whole reply would be:

I'll add a second counter alongside `n`, initialised beside it so it always
has a value.

=== EDIT: app/counter.py ===
--- FIND
    def tally(rows):
        n = 0
        for row in rows:
--- REPLACE
    def tally(rows):
        n = 0
        skipped = 0
        for row in rows:
=== END EDIT ===

=== EDIT: app/counter.py ===
--- FIND
            if row.ok:
                n += 1
        return n
--- REPLACE
            if row.ok:
                n += 1
            else:
                skipped += 1
        return n, skipped
=== END EDIT ===

Note what the example does *not* do: it does not print the finished function
in a ``` fence, it does not say the file has been updated, and it introduces
`skipped` by adding `skipped = 0` in the same breath as the line that
increments it.
"""


@dataclass(frozen=True)
class Patch:
    """One anchored change: find these lines, do this to them."""

    name: str
    find: str
    replace: str
    mode: str = "REPLACE"       # REPLACE | INSERT AFTER | INSERT BEFORE


def _normalise_ws(text: str) -> str:
    """Compare on content, ignoring how the model felt about indentation."""
    return "\n".join(line.strip() for line in text.strip().splitlines())


EDIT_OPENER = re.compile(r"^===\s*EDIT:\s*(?P<name>[^\n=]+?)\s*===[ \t]*$",
                         re.MULTILINE)


def _malformed_edit_blocks(reply: str, handled) -> list[Rejected]:
    """EDIT blocks the parser could not use, so they are never silent.

    Silence is the worst possible answer here.  A model that writes
    `=== EDIT: file.py ===` and then dumps code — no `--- FIND`, no closing
    marker — produces nothing the app can apply, and saying nothing about it
    leaves the user pressing send again and concluding the feature does not
    work.  It half does: the model is trying and getting the shape wrong.
    """
    complaints: list[Rejected] = []
    for match in EDIT_OPENER.finditer(reply or ""):
        name = match.group("name").strip()
        if not name or name in handled:
            continue
        handled.add(name)
        tail = reply[match.end():]
        nxt = EDIT_OPENER.search(tail)
        body = tail[:nxt.start()] if nxt else tail
        if FIND_PART.search(body):
            complaints.append(Rejected(
                name, "the edit block was never closed — it needs a "
                      "`=== END EDIT ===` line after the replacement"))
        else:
            complaints.append(Rejected(
                name, "that was written as a block of code rather than an "
                      "edit: an EDIT block needs a `--- FIND` section quoting "
                      "the existing lines and a `--- REPLACE` section saying "
                      "what they become. Nothing was changed"))
    return complaints


# A line of nothing but rule characters, which models write to *close* a
# section because every section in this format opens with one.  Nothing in the
# instructions asks for it and nothing forbids it, so it arrives regularly —
# and in either alphabet, since the block markers use `===` and the section
# markers use `---`.  A model that closes FIND with `---` will happily close
# REPLACE with `===`, and that line lands in the file: three equals signs on
# their own in the middle of a function is a syntax error the app wrote.
CLOSING_RULE = re.compile(r"\n[ \t]*(?:-{3,}|={3,})[ \t]*$")


def _without_closer(find: str, replace: str) -> tuple[str, str]:
    """Drop a trailing `---` the model used as a section terminator.

    This was quietly costing valid edits.  A model that writes

        --- FIND
        <two real lines>
        ---
        --- REPLACE
        <two new lines>
        ---

    has quoted the file correctly, and the app was folding the closing rule
    into the anchor and then reporting that the lines were not in the file —
    blaming the model for the one part it got right, and hiding the real
    difference from anyone reading the message.

    Stripping it is safe whichever way the dashes were meant. If they were a
    terminator they were never file content, and both sections lose a line
    that should not have been there. If they were genuine content — a rule in
    a markdown file — then dropping it from *both* sides just shortens the
    anchor by a line that stays where it is, and the edit lands the same. The
    replacement is only trimmed when the anchor was, so the two can never fall
    out of step.
    """
    if not CLOSING_RULE.search(find):
        return find, replace
    return (CLOSING_RULE.sub("", find),
            CLOSING_RULE.sub("", replace) if CLOSING_RULE.search(replace)
            else replace)


def _patch_from_body(name: str, body: str):
    """(Patch, None) or (None, Rejected) from the inside of one EDIT block."""
    find_match = FIND_PART.search(body)
    replace_match = REPLACE_PART.search(body)
    if not find_match or not replace_match:
        return None, Rejected(
            name, "the edit block was missing its FIND or REPLACE part")
    find = find_match.group("find").strip("\n")
    replace = replace_match.group("replace").strip("\n")
    find, replace = _without_closer(find, replace)
    if not find.strip():
        return None, Rejected(
            name, "the FIND part was empty — there is nothing to anchor to")
    placeholder = elided(replace)
    if placeholder:
        return None, Rejected(
            name, f"the replacement contains “{placeholder}” rather than "
                  f"real lines")
    return Patch(name=name, find=find, replace=replace,
                 mode=match_mode(replace_match.group("mode"))), None


def parse_patches(reply: str,
                  truncated: bool = False) -> tuple[list[Patch], list[Rejected]]:
    """Anchored edits from a reply, and the ones that were malformed.

    `truncated` is what the server said about the reply hitting its length
    cap, and it decides what an unclosed final block means — the same
    question FILE blocks have always answered.  A model that writes a long
    FIND and a long REPLACE and then simply stops has given everything the
    edit needs; refusing it over a missing `=== END EDIT ===` throws away
    correct work, and this was the most common way a picked suggestion
    failed.  A block cut off *mid-replacement* is different: applying half a
    REPLACE would mangle the file, so when the server says the reply was
    interrupted, the refusal stands.
    """
    patches: list[Patch] = []
    rejected: list[Rejected] = []
    handled: set = set()
    text = reply or ""
    for match in EDIT_BLOCK.finditer(text):
        name = match.group("name").strip()
        if not name:
            continue
        patch, refusal = _patch_from_body(name, match.group("body"))
        if patch is not None:
            patches.append(patch)
        else:
            rejected.append(refusal)
        handled.add(name)

    # A final block that never closed.  Its end is the next opener or the end
    # of the reply — not a guess, unless the reply itself was cut short.
    openers = list(EDIT_OPENER.finditer(text))
    for index, match in enumerate(openers):
        name = match.group("name").strip()
        if not name or name in handled:
            continue
        tail = text[match.end():]
        nxt = openers[index + 1] if index + 1 < len(openers) else None
        body = text[match.end():nxt.start()] if nxt else tail
        # Anything from a line opening with `===` onward is marker debris —
        # a half-written `=== END EDIT`, or the commentary after it — and
        # must not ride into the replacement.  So is a closer in the wrong
        # alphabet: `--- END EDIT ===` matches neither marker, and it went
        # into a user's file as their line 730 syntax error.
        body = re.split(r"^[ \t]*(?:===|[-=]{2,}[ \t]*END[ \t]+EDIT\b)",
                        body, maxsplit=1, flags=re.MULTILINE)[0]
        if not FIND_PART.search(body) or not REPLACE_PART.search(body):
            continue                    # _malformed_edit_blocks will say why
        if truncated and nxt is None:
            handled.add(name)
            rejected.append(Rejected(
                name, "the reply hit its length cap before this edit was "
                      "finished — applying half a replacement would mangle "
                      "the file. Press Continue, or raise the answer length, "
                      "and ask again"))
            continue
        patch, refusal = _patch_from_body(name, body)
        handled.add(name)
        if patch is not None:
            patches.append(patch)
        else:
            rejected.append(refusal)

    for item in rejected:
        handled.add(item.name)
    rejected += _malformed_edit_blocks(text, handled)
    return patches, rejected


def match_mode(raw: str) -> str:
    cleaned = " ".join((raw or "REPLACE").split()).upper()
    return cleaned if cleaned in ("REPLACE", "INSERT AFTER",
                                  "INSERT BEFORE") else "REPLACE"


def _to_line_end(current: str, end: int) -> int:
    """Pull `end` back off the newline that terminates the matched text.

    The three ways of matching disagreed about this, and the disagreement was
    invisible until it corrupted a file.  An exact match returns the end of
    the anchor *text*; the two forgiving matches were built out of whole lines
    and returned the end of the last *line*, one character further on.  A
    replacement then landed on the wrong side of the newline and welded the
    following line onto it —

        b = 8    c = 3

    — which is a syntax error produced by the app, out of a patch the model
    got right. So every path now returns the same thing: the end of the
    matched text, never including its line ending.
    """
    if end > 0 and current[end - 1:end] == "\n":
        end -= 1
    if end > 0 and current[end - 1:end] == "\r":
        end -= 1
    return end


def locate(current: str, find: str) -> tuple[int, int, str]:
    """(start, end, complaint) for the one place this anchor matches.

    `end` is the end of the matched text and excludes the newline after it,
    whichever of the three matching strategies found it.

    Exactly once, or it is refused.  Zero matches means the model retyped the
    code from memory instead of copying it; several means it chose an anchor
    that does not identify a place.  Both are the model's mistake, and both
    are recoverable by asking again — unlike a silent overwrite.
    """
    if not current:
        return -1, -1, "the file is empty or could not be read"

    count = current.count(find)
    if count == 1:
        start = current.index(find)
        return start, start + len(find), ""
    if count > 1:
        return -1, -1, (f"those lines appear {count} times in the file, so it "
                        f"is not clear which one was meant — a longer FIND "
                        f"with more surrounding lines would say")

    # Exact match failed.  Try again ignoring indentation, which is the thing
    # models most often get subtly wrong while quoting otherwise correctly.
    wanted = _normalise_ws(find)
    lines = current.splitlines(keepends=True)
    height = len(wanted.splitlines())
    if height:
        hits = []
        for index in range(0, len(lines) - height + 1):
            window = "".join(lines[index:index + height])
            if _normalise_ws(window) == wanted:
                hits.append((sum(len(x) for x in lines[:index]),
                             sum(len(x) for x in lines[:index + height])))
        if len(hits) == 1:
            return hits[0][0], _to_line_end(current, hits[0][1]), ""
        if len(hits) > 1:
            return -1, -1, ("those lines appear more than once once "
                            "indentation is ignored — a longer FIND would say "
                            "which was meant")
    # Third: ignore blank lines entirely.  A model quoting a method almost
    # always drops the empty lines inside it — they carry no meaning, and
    # refusing over them means refusing a perfectly good anchor.  Anything
    # beyond this (a differing comment, a renamed variable) is *not* forgiven,
    # because that would be applying a change to code the model never saw.
    start, end, ok = _match_ignoring_blanks(current, find)
    if ok:
        return start, _to_line_end(current, end), ""

    close = nearest(current, find)
    complaint = ("those lines are not in the file — the model retyped them "
                 "from memory rather than copying them, so there is nothing "
                 "to anchor the change to")
    if not close:
        # No honest near-match.  Naming what the file *does* define is still
        # useful and cannot mislead: nine times out of ten the method being
        # "fixed" simply is not there under that name, and seeing the real
        # list is what makes that obvious.
        names = defined_names(current)
        if names:
            complaint += (". This file defines: " + ", ".join(names[:12])
                          + ("…" if len(names) > 12 else ""))
    if close:
        # Saying what *is* there turns a dead end into a next step: it can be
        # pasted straight back at the model, and it usually reveals that the
        # method being "fixed" does not exist under that name at all.
        complaint += f". The closest thing actually in the file is:\n{close}"
    return -1, -1, complaint


# Below this, "closest" is not close enough to be worth quoting.
NEAR_ENOUGH = 0.45
NEAR_MAX_LINES = 12


def nearest(current: str, find: str, limit: float = NEAR_ENOUGH) -> str:
    """The real lines that most resemble a failed anchor, or "".

    Deliberately conservative: a bad guess here is worse than none, because
    it sends the reader looking at code that has nothing to do with the
    problem.
    """
    import difflib

    wanted = (find or "").strip().splitlines()
    lines = (current or "").splitlines()
    if not wanted or not lines:
        return ""
    height = min(len(wanted), NEAR_MAX_LINES)
    best, score = "", 0.0
    for index in range(0, max(1, len(lines) - height + 1)):
        window = lines[index:index + height]
        ratio = difflib.SequenceMatcher(
            None, "\n".join(w.strip() for w in wanted[:height]),
            "\n".join(w.strip() for w in window)).ratio()
        if ratio > score:
            best, score = "\n".join(window), ratio
    return best if score >= limit else ""


# Enough to name what a file contains across the languages this app sees.
DEFINITION = re.compile(
    r"^[ \t]*(?:async\s+)?(?:def|class|function|fn|sub)\s+(?P<name>\w+)"
    r"|^[ \t]*(?:export\s+)?(?:const|let|var)\s+(?P<js>\w+)\s*=\s*"
    r"(?:async\s*)?(?:\(|function)"
    r"|^[ \t]*(?P<c>\w+)\s*\([^;]*\)\s*\{",
    re.MULTILINE)


def defined_names(text: str) -> list[str]:
    """Functions and classes the file actually declares, in order."""
    found: list[str] = []
    for match in DEFINITION.finditer(text or ""):
        name = match.group("name") or match.group("js") or match.group("c")
        if name and name not in found and not name.isdigit():
            found.append(name)
    return found


def _match_ignoring_blanks(current: str, find: str) -> tuple[int, int, bool]:
    """(start, end, matched) comparing only the lines that carry meaning."""
    lines = current.splitlines(keepends=True)
    numbered = [(index, line.strip())
                for index, line in enumerate(lines) if line.strip()]
    wanted = [line.strip() for line in (find or "").splitlines() if line.strip()]
    if not wanted or len(numbered) < len(wanted):
        return -1, -1, False

    hits = []
    for offset in range(len(numbered) - len(wanted) + 1):
        window = [text for _index, text in numbered[offset:offset + len(wanted)]]
        if window == wanted:
            first = numbered[offset][0]
            last = numbered[offset + len(wanted) - 1][0]
            hits.append((first, last))
    if len(hits) != 1:
        return -1, -1, False            # none, or ambiguous — refuse either way
    first, last = hits[0]
    start = sum(len(line) for line in lines[:first])
    end = sum(len(line) for line in lines[:last + 1])
    return start, end, True


def _indent_of(text: str) -> str:
    """The leading whitespace of the first line that has any content."""
    for line in (text or "").splitlines():
        if line.strip():
            return line[:len(line) - len(line.lstrip())]
    return ""


def _reindent(replace: str, find: str, anchor: str) -> str:
    """Shift the replacement to sit where the anchor really sits.

    Matching already forgives indentation, because a model quoting a method
    body from the middle of a class very often flattens it to the left margin.
    Forgiving it in the *anchor* and then writing the replacement back at the
    model's margin is half a feature: the edit applies, the diff looks
    plausible, and the file it produces does not compile.

    So the two are measured against each other and the difference is applied
    to every line. Tabs are left alone rather than guessed at — mixing them
    with spaces by arithmetic is its own bug.
    """
    quoted, real = _indent_of(find), _indent_of(anchor)
    if quoted == real or "\t" in quoted or "\t" in real:
        return replace
    if real.startswith(quoted):                 # the file is further right
        extra = real[len(quoted):]
        return "\n".join(extra + line if line.strip() else line
                         for line in replace.splitlines())
    if quoted.startswith(real):                 # the model was further right
        drop = len(quoted) - len(real)
        shifted = []
        for line in replace.splitlines():
            if not line.strip():
                shifted.append(line)
                continue
            lead = len(line) - len(line.lstrip())
            shifted.append(line[min(drop, lead):])
        return "\n".join(shifted)
    return replace


def apply_patch(current: str, patch: Patch) -> tuple[str, str]:
    """(new contents, complaint).  On any complaint the text is unchanged."""
    start, end, problem = locate(current, patch.find)
    if problem:
        return current, problem
    patch = Patch(name=patch.name, mode=patch.mode, find=patch.find,
                  replace=_reindent(patch.replace, patch.find,
                                    current[start:end]))
    # `end` sits at the end of the matched text, before its newline, so the
    # tail always begins with that newline (or is the end of the file).  Each
    # mode can therefore say plainly where the new lines go, without having to
    # guess which side of the line ending it was handed.
    if patch.mode == "INSERT AFTER":
        return current[:end] + "\n" + patch.replace + current[end:], ""
    if patch.mode == "INSERT BEFORE":
        return current[:start] + patch.replace + "\n" + current[start:], ""
    return current[:start] + patch.replace + current[end:], ""


# Prefixes a model copies out of an example and never removes.
TEMPLATE_PREFIXES = ("relative/path/to/", "path/to/", "your/project/",
                     "src/path/to/", "./", "/")


def _strip_template_prefix(name: str) -> str:
    cleaned = (name or "").replace("\\", "/").strip()
    changed = True
    while changed:
        changed = False
        for prefix in TEMPLATE_PREFIXES:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):]
                changed = True
    return cleaned


def find_file(root, name: str, known=None) -> tuple[str, str]:
    """(the real relative path, complaint).

    A model naming a file it can see is still perfectly capable of writing
    the path wrong — pasting the example's `relative/path/to/` prefix, giving
    a bare filename when the file is in a subdirectory, or using backslashes.
    None of those are reasons to refuse work the user asked for, as long as
    the file it lands on is unambiguous.

    The disk is the authority on what exists.  `known` — the files whose
    contents were loaded into the conversation — is a preference for
    breaking ties, never a boundary.  It used to be a boundary, and the
    result was the app telling the user "there is no classify_engine.py in
    the attached folder" about a file sitting in plain sight in the folder,
    because the folder had fifteen files and the context budget had room
    for six.  Which six fit a token budget is an implementation detail of
    the *prompt*; making it a property of the *filesystem* turned every
    larger-than-the-window project into a liar.  The danger the boundary
    guarded against — a model editing a file it never saw — is real, but it
    is handled where it actually bites: anchors that do not match the real
    file are refused with the reason, and the retry hands the real file
    back.
    """
    # Normalised at the door, so a Windows-style `udbg\\config.py` resolves
    # to the same answer on every platform.  On Windows the raw name would
    # *work* and come back unnormalised; on Linux it is one strange filename
    # that exists nowhere — same model output, two different behaviours.
    name = (name or "").replace("\\", "/")
    preferred = [str(c).replace("\\", "/") for c in known] if known else []

    def exists(candidate: str) -> bool:
        target = resolve(root, candidate)
        return target is not None and target.is_file()

    # An escaping path is refused as an escaping path.  Reporting it as "no
    # such file" would be friendlier and would hide the thing worth knowing.
    if name and resolve(root, name) is None and not _strip_template_prefix(
            name).startswith(("relative/", "path/")):
        if ".." in name.replace("\\", "/").split("/") or Path(name).is_absolute():
            return "", "that path is outside the attached folder"

    if name and exists(name):
        return name, ""

    stripped = _strip_template_prefix(name)
    if stripped and stripped != name and exists(stripped):
        return stripped, ""

    # Last resort: match on the file name alone, but only when it lands on
    # one file.  Two `config.py` files and a guess would be a write to the
    # wrong module — unless exactly one of them was in the conversation, in
    # which case the model is talking about the one it read.
    base = Path(stripped or name or "").name
    if not base:
        return "", "no file name was given"
    try:
        matches = [str(p.relative_to(Path(root))).replace("\\", "/")
                   for p in Path(root).rglob(base) if p.is_file()]
    except OSError:
        matches = []
    if len(matches) > 1:
        seen = [m for m in matches if m in preferred]
        if len(seen) == 1:
            return seen[0], ""
        return "", (f"there are {len(matches)} files called {base} "
                    f"({', '.join(sorted(matches)[:4])}…) — the full path is "
                    f"needed to say which")
    if len(matches) == 1:
        return matches[0], ""
    return "", (f"there is no {base} in the attached folder")


def resolve_patches(root, patches, known=None) -> tuple[list[Edit], list[Rejected]]:
    """Turn anchored edits into whole-file results, or say why not.

    Several patches to one file are applied in order against the running
    text, so a second edit anchored to a line the first one wrote still
    finds it.
    """
    working: dict = {}
    applied: set = set()
    rejected: list[Rejected] = []
    for patch in patches:
        name, problem = find_file(root, patch.name, known)
        if problem:
            rejected.append(Rejected(patch.name, problem))
            continue
        if resolve(root, name) is None:
            rejected.append(Rejected(
                patch.name, "that path is outside the attached folder"))
            continue
        if name not in working:
            working[name] = read_current(root, name)
        updated, problem = apply_patch(working[name], patch)
        if problem:
            # Anchors from a file the model never read cannot match it.
            # That is not the model being sloppy — the folder was bigger
            # than the context budget, so this file went in as a name in
            # the manifest and nothing else.  Saying so turns a baffling
            # refusal into an explanation, and points at the fix (the
            # retry, which sends the real contents).
            if known is not None and name not in known:
                problem += (" (Only this file's name was sent to the model — "
                            "its contents did not fit the context budget, so "
                            "the model was working from the name alone.)")
            rejected.append(Rejected(name, problem))
            continue
        working[name] = updated
        applied.add(name)

    edits = []
    for name, text in working.items():
        if text != read_current(root, name):
            edits.append(Edit(name=name, new_text=text))
            continue
        if name not in applied:
            # Every patch for this file was refused above and already has its
            # own reason. The unchanged text is the consequence of that, not
            # a second, separate complaint to make about it.
            continue
        # The patch applied and changed nothing: the model copied its FIND
        # lines into REPLACE unaltered. It happens whenever the instruction
        # was vague — "find the deficiencies and fix them" — and the model
        # quotes the code back rather than admitting it has no change to
        # make. Dropping it quietly leaves the user with a request that
        # produced no diff, no refusal and no reason, which is the single
        # most demoralising thing this feature can do.
        rejected.append(Rejected(
            name, "the replacement was identical to the lines it quoted, so "
                  "there is no change to make — the model repeated the code "
                  "back instead of altering it. Naming the specific change "
                  "you want usually fixes this"))
    return edits, rejected


def instructions(prefer_patches: bool = True, detailed: bool = False) -> str:
    """What the model is told it can do.

    Patches first and whole-file second, because the order is the advice: a
    model reaches for the first shape that fits, and for anything but a new
    file the first shape should be the one that cannot silently delete
    everything it did not repeat.

    `detailed` adds the worked example.  It roughly doubles the length, and
    these sit in the system prompt of every message once a folder is attached,
    so the ambient copy stays lean and the second pass — which has already
    watched this model fail to produce a block, and has its whole allowance
    for one change — gets the full version.
    """
    if not prefer_patches:
        return WHOLE_FILE_INSTRUCTIONS
    parts = [PATCH_INSTRUCTIONS]
    if detailed:
        parts.append(PATCH_EXAMPLE)
    parts.append(WHOLE_FILE_INSTRUCTIONS)
    return "\n".join(parts)


# An edit block costs a few hundred tokens before the model has said anything
# about why.  At the Quick end of the speed slider the whole reply allowance
# is 192 tokens, so every attempt is cut off inside the first block — which
# looks exactly like a model refusing to edit, and is not.
MIN_EDIT_TOKENS = 1_200


def allowance_for_edits(num_predict: int) -> int:
    """Enough room for the model to finish what it starts."""
    if num_predict <= 0:                # already unlimited
        return num_predict
    return max(num_predict, MIN_EDIT_TOKENS)


def cramped(num_predict: int) -> bool:
    return 0 < num_predict < MIN_EDIT_TOKENS


# What someone asking for their files to be changed actually types.  Not an
# exhaustive grammar — it only has to be right often enough to say "there is
# no folder attached" at the moment that sentence is useful.
WANTS_EDIT = re.compile(
    r"\b(?:edit|change|modify|update|upgrade|improve|refactor|rewrite|"
    r"implement|apply|fix|patch|add(?:\s+a)?|remove|delete|rename|"
    r"optimi[sz]e|clean\s*up|replace|swap|introduce|extract|convert|"
    r"migrate|simplify|harden|tidy|correct|deficienc(?:y|ies))\b"
    r"(?!\s+(?:me|us)\b)",
    re.IGNORECASE)

# …but only when it is aimed at something on disk.  "fix my understanding of
# decorators" is not a request to write a file.
ABOUT_FILES = re.compile(
    r"\b(?:file|files|code|codebase|project|app|application|script|module|"
    r"class|function|method|repo|repository|program|source|engine|handler|"
    r"handling|logic|parser|parsing|server|folder|directory)\b"
    # A filename settles it on its own: nobody types `classify.py` about an
    # idea.  The extension list is deliberately short — real ones, not every
    # suffix that exists.
    r"|\b[\w./\\-]+\.(?:py|pyw|js|ts|tsx|jsx|java|c|h|cpp|cs|go|rs|rb|php|"
    r"sh|ps1|sql|json|ya?ml|toml|ini|cfg|md|txt|html?|css)\b",
    re.IGNORECASE)


def wants_to_edit(text: str) -> bool:
    """Did the user just ask for their files to be changed?

    Used to decide whether silence is acceptable.  With no folder attached the
    app cannot write anything at all and does not even tell the model that
    editing exists — so the model asks which files are meant, the user reads
    that as the app refusing to do the work, and nothing in the window
    mentions the one fact that explains it.
    """
    said = text or ""
    return bool(WANTS_EDIT.search(said) and ABOUT_FILES.search(said))


# -- the most convincing failure this app has ------------------------------
# A model with no hands narrating "Saving the modified file." is worse than a
# model that refuses.  The user watches a checklist tick past "Save the
# modified file ✓", reads "the file has been successfully modified", and
# reasonably concludes the app wrote it.  Nothing was written; nothing could
# have been.  The app knows that for certain — it is the thing that does the
# writing — so staying quiet is a choice to let the user be misled.
CLAIMED_WRITE = re.compile(
    r"\bsav(?:e|es|ed|ing)\s+(?:the\s+|this\s+|your\s+)?"
    r"(?:modified\s+|updated\s+|edited\s+|changed\s+|new\s+)?files?\b"
    r"|\bfiles?\s+(?:has|have)\s+been\s+(?:successfully\s+)?"
    r"(?:modified|updated|saved|edited|changed|written|rewritten)\b"
    r"|\b(?:i|i'?ve|i\s+have)\s+(?:now\s+)?(?:successfully\s+)?"
    r"(?:updated|modified|edited|saved|written|applied|replaced|rewritten)\s+"
    r"(?:the|your|this|it)\b"
    r"|\bchanges?\s+(?:has|have)\s+been\s+(?:successfully\s+)?"
    r"(?:applied|made|saved|written|implemented)\b"
    r"|\b(?:successfully\s+)?(?:modified|updated|rewrote|edited)\s+the\s+file\b",
    re.IGNORECASE)

# Advice to the user is not a claim about what happened.  "You should save
# the file" and "To apply this, save the file" are both fine: the model is
# telling someone who does have hands what to do with them.
ADVICE_OPENER = re.compile(
    r"^(?:you|please|to\b|then\b|next\b|now\s+(?:you|save)|if\s+you|"
    r"after\b|once\b|finally,?\s+you|make\s+sure)", re.IGNORECASE)


def pretended_to_write(reply: str, names=()) -> str:
    """The sentence in which the model claimed to have written to disk.

    Returned rather than a bare True so the warning can quote it back: being
    told "the model said it saved your file, and it did not" only lands with
    the model's own words in front of you.

    `names` narrows this to replies that actually discuss a file in the
    attached folder, so a general conversation about how saving works does
    not collect a warning about edits nobody asked for.
    """
    text = reply or ""
    if names and not any(_mentions(text, name) for name in names):
        return ""
    for sentence in re.split(r"(?<=[.!?:])\s+|\n", text):
        line = sentence.strip().lstrip("*-•# ").strip()
        if not line or len(line) > 300 or ADVICE_OPENER.match(line):
            continue
        if CLAIMED_WRITE.search(line):
            return line
    return ""


def _mentions(text: str, name: str) -> bool:
    """Is this file talked about, by path or by bare filename?"""
    bare = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    return bool(bare) and bare in text
