"""Turning "make it better" into a menu of changes that can actually land.

Measured against this app's own pipeline, the difference between an edit that
applies and one that breaks is not the model's mood — it is the shape of the
request. Ten runs of specific asks split cleanly: a change in *one place*
(a docstring, a strip() in one function) landed every time; a change spread
across *two places* ("add a constant and use it") broke most of the time; and
"find the deficiencies and upgrade it" produced something broken or empty
four runs out of four.

So a broad request is not sent to the model as an instruction to edit.  It is
sent as a request for a *menu*: three to six candidate changes, each confined
to one place, each described in a line — no code, no blocks.  Describing
changes is the one job these models do reliably.  The user picks one, and the
pick becomes a scoped second-pass request through the machinery that already
works.  The menu costs one model call and turns the worst-performing kind of
ask into a sequence of the best-performing kind.

The user is the drill-down.  That is deliberate: every candidate is approved
twice — once as an idea, once as a diff — and nothing about a broad wish ever
reaches the disk without having been narrowed by a human on the way.
"""

from __future__ import annotations

import re

from .edits import ABOUT_FILES, wants_to_edit
from .intent import Change, _file_in, _known_by_basename

# The asks that produce broken edits when sent straight to a local model:
# a direction of travel rather than a destination.
BROAD = re.compile(
    r"\b(?:improve|upgrade|enhance|optimi[sz]e|deficienc(?:y|ies)|"
    r"performance|better|modernvi[sz]e|moderni[sz]e|clean\s*up|cleanup|"
    r"harden|robust|refactor|tidy|polish|speed\s*up|"
    r"any\s+(?:bugs?|issues?|problems?)|what\s+(?:can|should)\s+be\s+"
    r"(?:improved|fixed|better))\b",
    re.IGNORECASE)

# Signals that the user already knows the destination.  "Replace 0.85 with a
# constant to improve reliability" contains 'improve', but it is a specific
# instruction and turning it into a menu would be answering a question with a
# questionnaire.
NARROW = re.compile(
    r"`[^`]+`"                                  # a quoted identifier
    r"|\w+\(\)"                                 # a named function
    r"|\b(?:replace|rename|change|set|convert)\b.{0,60}\b(?:to|with|into)\b"
    r"|\bline\s+\d+"
    r"|=\s*[\w\"']",                            # an assignment spelled out
    re.IGNORECASE | re.DOTALL)

MENU_OPEN = "=== SUGGESTIONS ==="
MENU_CLOSE = "=== END SUGGESTIONS ==="

# `1. [file.py] change one thing` — the shape the wrap asks for.  Bracketed
# file first, because the file is the part the app must be able to trust.
MENU_LINE = re.compile(
    r"^\s*\d{1,2}[.)]\s*\[(?P<file>[^\]\n]+)\]\s*(?P<body>\S.*)$",
    re.MULTILINE)

MIN_CANDIDATES = 2       # one bracketed line in prose is not a menu
MAX_CANDIDATES = 8
MIN_BODY = 12


def is_broad(text: str) -> bool:
    """Is this an edit request with a direction but no destination?

    Only these get the menu treatment.  A specific instruction goes straight
    through, because the menu would cost a model call to ask the user what
    they already said.
    """
    said = text or ""
    if NARROW.search(said):
        return False
    # "What can be improved?" needs no other context: in a chat with a
    # folder attached — the only place this check runs — there is nothing
    # else it could be about.
    if re.search(r"\bwhat\s+(?:can|should|could)\s+be\s+"
                 r"(?:improved|fixed|better|upgraded)", said, re.IGNORECASE):
        return True
    if not BROAD.search(said):
        return False
    # Broad wishes are often noun-shaped — "performance enhancements", "more
    # robust" — so demanding an edit *verb* here would miss exactly the asks
    # this exists for.  The direction word plus something-about-the-code is
    # the signal; wants_to_edit covers the verb-shaped rest.
    return bool(ABOUT_FILES.search(said)) or wants_to_edit(said)


def symbol_index(texts: dict) -> str:
    """Every function that really exists, per file, in a few hundred tokens.

    The folder attachment is truncated to fit the context window, so the
    model sees whole files' names and half their contents — and asked to
    suggest changes, it invents plausible functions for the parts it cannot
    see.  Measured on a real menu: seven of the nine functions it named did
    not exist.  This index is the antidote, and it is cheap: names only, no
    signatures, no code.
    """
    from .edits import defined_names

    lines = []
    for name in sorted(texts):
        if PROSE_FILE.search(name):
            continue
        found = defined_names(texts[name])
        if found:
            lines.append(f"{name}: " + ", ".join(
                f"{fn}()" for fn in found[:40]))
    return "\n".join(lines)


def wrap_ask(ask: str, index: str = "", file_map: str = "") -> str:
    """The user's broad wish, recast as a request for a menu.

    Everything here is shaped by what failed.  No code and no EDIT blocks,
    because a model allowed to write code writes code instead of the list.
    One place per line, because two-location changes are the ones that break.
    The exact format is spelled out and closed with a marker, because this
    reply is parsed, not read.
    """
    orientation = ""
    if file_map:
        orientation = (
            f"A map of the project, one line per file, in the authors' own "
            f"words:\n\n{file_map}\n\n"
            f"Spread your suggestions across the files where the request "
            f"actually lives — the map says what each file is for.\n\n")
    grounding = ""
    if index:
        grounding = (
            f"These are the functions that exist, per file. Suggestions must "
            f"name a function from this list and no other — a function not "
            f"on it does not exist, whatever the file contents above may "
            f"have suggested:\n\n{index}\n\n")
    return (
        f"{ask}\n\n"
        f"Do not make any changes yet and do not write any code or EDIT "
        f"blocks. First, give me a menu of specific candidate changes to "
        f"choose from.\n\n{orientation}{grounding}"
        f"Use exactly this format:\n\n"
        f"{MENU_OPEN}\n"
        f"1. [<file path exactly as shown in the listing>] <one specific "
        f"change, in one place — name the function or section and say what "
        f"changes>\n"
        f"2. [<file>] <another change>\n"
        f"{MENU_CLOSE}\n\n"
        f"Rules:\n"
        f"- 3 to 6 suggestions.\n"
        f"- Each line is ONE change in ONE place. Never 'add X and use it "
        f"in Y' — that is two places, so split it or leave it out.\n"
        f"- Each suggestion must be SMALL: a few lines in one function — a "
        f"guard clause, a missing check, a default, a docstring, an early "
        f"return. Nothing that redesigns, restructures, parallelises, or "
        f"needs a new library.\n"
        f"- Name the exact function, written as name(), in every line. A "
        f"line with no function name will be discarded.\n"
        f"- Only code files from the listing, path in square brackets.\n"
        f"- No code, no diffs, no edit blocks — descriptions only.\n"
        f"- One line per suggestion, nothing between the markers but the "
        f"numbered lines.")


def parse_menu(reply: str, known=()) -> list[Change]:
    """The candidates a reply offers, tied to files that actually exist.

    Reads the whole reply rather than only between the markers, because a
    model that produces perfect numbered lines and forgets the fence is
    offering the same menu.  What makes it a menu is the *shape* — several
    numbered, bracketed-file lines — which does not occur in prose by
    accident; requiring at least two keeps one stray bracket from
    counting.
    """
    lookup = _known_by_basename(known)
    if not lookup:
        return []
    found: list[Change] = []
    seen: set = set()
    for match in MENU_LINE.finditer(reply or ""):
        body = match.group("body").strip().rstrip(".")
        name = _file_in(f"[{match.group('file')}]", lookup) or _file_in(
            match.group("file"), lookup)
        if not name or len(body) < MIN_BODY:
            continue
        if PROSE_FILE.search(name):
            # "Document the error scenarios in the README" is advice about
            # writing, not a code change this pipeline can carry safely.
            continue
        key = (name, body.lower())
        if key in seen:
            continue
        seen.add(key)
        found.append(Change(description=body, file=name))
        if len(found) >= MAX_CANDIDATES:
            break
    # Candidates that name their function first: they are the ones the
    # second pass lands, and the menu's order is a recommendation.
    found.sort(key=lambda c: 0 if NAMES_FUNCTION.search(c.description) else 1)
    return found if len(found) >= MIN_CANDIDATES else []


# Files where "editing" means writing prose, which is not what a picked
# candidate should quietly turn into.
PROSE_FILE = re.compile(r"\.(?:md|rst|txt)$", re.IGNORECASE)

# `name()` — the anchor the second pass needs to know where to look.
NAMES_FUNCTION = re.compile(r"\b(\w+)\(\)")


def verify(candidates, texts: dict) -> tuple[list, list]:
    """(kept, dropped) — candidates whose named functions actually exist.

    The index in the prompt reduces invention; this removes what slips
    through, because a menu is only worth offering if picking from it is
    safe.  A candidate naming no function at all is dropped too — the rules
    demanded one, and without it the second pass is back to guessing where
    to look, which is the failure this whole feature replaces.
    """
    from .edits import defined_names

    known: dict = {}
    kept, dropped = [], []
    for candidate in candidates:
        if candidate.file not in known:
            known[candidate.file] = set(
                defined_names(texts.get(candidate.file, "")))
        names = [m.group(1)
                 for m in NAMES_FUNCTION.finditer(candidate.description)]
        if not names:
            dropped.append((candidate, "names no function"))
            continue
        missing = [name for name in names
                   if name not in known[candidate.file]]
        if missing:
            dropped.append((candidate,
                            f"{missing[0]}() does not exist in "
                            f"{candidate.file}"))
            continue
        kept.append(candidate)
    return kept, dropped


def looks_multi_location(description: str) -> bool:
    """Does a candidate smuggle in the two-place shape the menu forbids?

    The model is told not to; this is for when it does anyway.  Flagged
    rather than dropped — the user can still pick it, but the label warns
    that this shape breaks more often.
    """
    return bool(re.search(
        r"\band\s+(?:use|call|apply|reference)\b"
        r"|\buse\s+it\b"
        r"|\b(?:everywhere|throughout|all\s+(?:call|place|use|occurrence))",
        description or "", re.IGNORECASE))


def pick_menu_files(ask: str, texts: dict, budget_tokens: int,
                    max_files: int = 12) -> list:
    """The files a menu should be built from: code, as much as fits.

    Question-keyword retrieval is the wrong scan for a broad-improvement ask.
    "I want performance enhancements" matches the *documentation* — the
    README and session notes discuss performance in those words — while the
    engines the improvements would land in match nothing and stay home.  A
    menu built from files it cannot even propose edits for is a menu of
    guesses.

    So: code files only (prose files cannot be candidates anyway), relevance
    first for whatever the question does name, then the remaining budget
    spent on breadth — smallest files first, because a menu wants to have
    *seen* as much of the project as possible, and three small engines teach
    it more than one giant one.
    """
    from .retrieval import Chunk, select

    code = {name: text for name, text in texts.items()
            if not PROSE_FILE.search(name)
            and not name.replace("\\", "/").rsplit("/", 1)[-1].startswith(".")}
    chunks = [Chunk(name=name, text=text)
              for name, text in sorted(code.items())]
    chosen = select(ask, chunks, budget_tokens=budget_tokens,
                    max_files=max_files)
    have = {chunk.name for chunk in chosen}
    spent = sum(chunk.tokens for chunk in chosen)
    for chunk in sorted(chunks, key=lambda c: c.tokens):
        if len(chosen) >= max_files:
            break
        if chunk.name in have or spent + chunk.tokens > budget_tokens:
            continue
        chosen.append(chunk)
        have.add(chunk.name)
        spent += chunk.tokens
    return chosen


def _first_line_of_docstring(text: str, name: str = "") -> str:
    """The first *informative* line of a module docstring, or "".

    The author's own description of what the file is for — already written,
    already accurate, and impossible to hallucinate.  Many docstrings open
    with the filename itself as a title line; that says nothing a map needs,
    so the first line that is not just the file's own name wins.
    """
    import ast

    try:
        doc = ast.get_docstring(ast.parse(text or ""), clean=True) or ""
    except (SyntaxError, ValueError, RecursionError):
        return ""
    bare = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for line in doc.strip().splitlines():
        stripped = line.strip().strip("-=# ").strip()
        if not stripped:
            continue
        if bare and stripped.lower().split()[0].strip(":,") == bare:
            stripped = stripped[len(bare):].strip(" -—:,")
            if not stripped:
                continue
        return stripped
    return ""


def _leading_comment(text: str, name: str = "") -> str:
    """The first informative comment line of a file with no docstring.

    Header blocks conventionally open with the filename itself — a title,
    not a description — so the first comment line that says something the
    name does not is the one the map wants.
    """
    bare = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    kept: list[str] = []
    for line in (text or "").splitlines()[:10]:
        stripped = line.strip()
        if stripped.startswith(("#", "//")):
            cleaned = stripped.lstrip("#/ ").strip()
            if not cleaned or cleaned.startswith(("-*-", "!")):
                continue
            if bare and cleaned.lower().split()[0].strip(":,") == bare:
                cleaned = cleaned[len(bare):].strip(" -—:,")
                if not cleaned:
                    continue
            # Changelog lines — "v1.2.0: acknowledgement, AI Review…" — are
            # history, not description.  Matched anywhere in the line, since
            # a changelog entry wraps and its continuation carries the next
            # "v1.3.0:" mid-line.
            if re.match(r"v?\d+\.\d+", cleaned) or re.search(
                    r"v\d+\.\d+(?:\.\d+)?:", cleaned):
                continue
            if len(cleaned) > 8:
                kept.append(cleaned)
                # The title line names the file; the line after it usually
                # says what the thing is for.  Two is a description; more is
                # the whole header block.
                if len(kept) == 2:
                    break
        elif stripped and not stripped.startswith(('"""', "'''")):
            break
    return " — ".join(kept)


def project_map(texts: dict) -> str:
    """One line per code file: what it is for, in the author's own words.

    A menu proposing changes to fourteen files can carry the full text of
    two or three; for the rest the model has only names and a function
    index, which say what a file *defines* but not what it is *for*.  The
    map fills that in from what the code already carries — the module
    docstring, a leading comment, or failing both, the first few things the
    file defines.  Deterministic on purpose: a map is only worth trusting
    if it cannot be wrong.
    """
    from .edits import defined_names

    lines = []
    for name in sorted(texts):
        bare = name.replace("\\", "/").rsplit("/", 1)[-1]
        if PROSE_FILE.search(name) or bare.startswith("."):
            continue
        text = texts[name]
        about = (_first_line_of_docstring(text, name)
                 if name.endswith((".py", ".pyw")) else "") or _leading_comment(
                     text, name)
        if not about:
            found = defined_names(text)
            if not found:
                continue
            about = "defines " + ", ".join(f"{fn}()" for fn in found[:5])
        lines.append(f"{name} — {about[:160]}")
    return "\n".join(lines)
