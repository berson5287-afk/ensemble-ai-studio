"""Asking a model to plan a big job, then working the plan one step at a time.

"Take a look at my app and make improvements, and give me a complete list of
what you changed" is not one question. Sent as one, it becomes a single
enormous generation that either rambles or stops halfway, and there is no way
to tell which until it finishes.

Split into a plan, it becomes several small answers with a visible checklist
in front of them. Each step is cheap enough to finish, you can see which one
is running, and a step that goes wrong is one step rather than the whole
reply. The cost is more round trips, which on a large local model is not
free — so this is a mode you choose, not something that happens to you.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

MAX_STEPS = 6
MIN_STEPS = 2
MAX_STEP_CHARS = 110

# Same shape as the list prefixes elsewhere: strip "1. ", "- ", "* " but never
# a bare leading digit, or "0-60 acceleration" loses its subject.
LIST_PREFIX = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s+")

# Models like to introduce themselves before doing as they were told.
# Applied twice, because "Sure! Here's the plan:" is two preambles stacked.
# The interjection branch deliberately stops at the punctuation rather than
# running on — matching "sure!" plus the next forty characters swallowed the
# beginning of step one.
PREAMBLE = re.compile(
    r"^(here(?:'s| is)[^\n:]{0,60}:|plan:|steps:|sure[,!.]|ok(?:ay)?[,!.])\s*",
    re.IGNORECASE)

# Lines that are commentary rather than a step.
NOT_A_STEP = re.compile(
    r"^(let me know|i hope|feel free|note:|caveat|in summary|summary:|"
    r"that'?s (?:it|all)|would you like)",
    re.IGNORECASE)

# Wording that signals a job with several distinct parts rather than a question
COMPLEX_HINTS = (
    "and also", "then ", "step by step", "one by one", "complete list",
    "full list", "go through", "review", "audit", "refactor", "improve",
    "improvements", "upgrade", "upgrades", "rewrite", "modernise", "modernize",
    "clean up", "test coverage", "each file", "every file", "all the files",
)


def looks_complex(question: str, attachments: Sequence[dict] = ()) -> bool:
    """Is this the kind of request worth planning before answering?

    Deliberately conservative: planning costs extra model calls, so a plain
    question should never trigger it.
    """
    text = (question or "").strip().lower()
    if not text:
        return False
    if len(text.split()) >= 25:
        return True
    if attachments and any(hint in text for hint in COMPLEX_HINTS):
        return True
    return sum(hint in text for hint in COMPLEX_HINTS) >= 2


# What a model has to be told when the request is "change my files".
#
# It has no file handle, no shell and no test runner.  Left alone it plans
# "1. Open the file … 5. Save the modified file … 6. Test it", then works
# that plan by *narrating* each step — and narration is indistinguishable
# from work when you are watching a checklist tick.  The user ends up with
# six green ticks, a paragraph saying the file was successfully modified,
# and a file on disk that nobody touched.
NO_HANDS = (
    "This request changes files, so before you plan: you cannot open, save, "
    "run or test anything yourself. The app has already put the files in "
    "front of you, and the app is what writes changes to disk — after the "
    "user has seen them and approved them. So 'open the file', 'save the "
    "file', 'apply the changes' and 'run the tests' are not work you can do, "
    "and must not appear in the plan. Plan the decisions instead: which "
    "file, which existing lines in it, and what those lines become.")

# The backstop for when it ignores the above, which it will.  Deliberately
# narrow — it drops theatre, not work.  "Write the new helper" is a real
# step; "Write the changes to the file" is not.
IMPOSSIBLE = re.compile(
    r"^(?:open|save|close|re-?open|reload|download|upload)\b"
    r"|^(?:write|apply|commit|persist|push)\s+(?:the\s+|these\s+|your\s+)?"
    r"(?:changes?|edits?|modifications?|updates?|files?|code|it)\b"
    r"|^(?:run|execute|launch|deploy)\b"
    r"|^test\b",
    re.IGNORECASE)


def drop_impossible_steps(steps: Sequence[str]) -> list[str]:
    """Strike the steps that are theatre rather than work.

    If striking them would leave nothing worth calling a plan, the plan is
    left alone — a visibly silly plan beats an empty one, and the caller
    falls back to answering in one go when a plan is too short to use.
    """
    kept = [step for step in steps if not IMPOSSIBLE.match(step.strip())]
    return kept if len(kept) >= MIN_STEPS else list(steps)


def build_plan_prompt(question: str, material: str = "",
                      editing: bool = False) -> str:
    return (
        "Break this request into a short checklist of steps you will work "
        "through one at a time.\n\n"
        f"{MIN_STEPS}-{MAX_STEPS} steps. One line each, numbered. Each step "
        "should be a concrete piece of work that produces part of the final "
        "answer — not 'understand the request' or 'gather information'.\n"
        "No preamble, no commentary, nothing after the list.\n\n"
        + (f"{NO_HANDS}\n\n" if editing else "")
        + (f"What you have to work with:\n{material}\n\n" if material else "")
        + f"Request: {question}\n\nSteps:")


def parse_plan(reply: str, limit: int = MAX_STEPS) -> list[str]:
    """Pull the steps out of whatever the model actually wrote."""
    text = (reply or "").strip()
    for _ in range(2):
        text = PREAMBLE.sub("", text, count=1).lstrip()
    steps: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        stripped = LIST_PREFIX.sub("", line).strip()
        # a heading or a sentence of commentary is not a step
        if not stripped or NOT_A_STEP.match(stripped):
            continue
        if stripped.startswith("#") or stripped.endswith(":") and len(stripped) < 30:
            continue
        if len(stripped) > MAX_STEP_CHARS:
            stripped = stripped[:MAX_STEP_CHARS].rsplit(" ", 1)[0] + "…"
        if stripped not in steps:
            steps.append(stripped)
        if len(steps) >= limit:
            break
    return steps


STEP_HAS_NO_HANDS = (
    "You still cannot open or save anything in this step. The file's real "
    "text is in the conversation above — work from that, and quote from it "
    "rather than from memory. Do not say you have opened, changed or saved "
    "a file; you have not.")

SUMMARY_MUST_PROPOSE = (
    "This request changes files, and only one thing reaches the disk: an "
    "edit block in the format you were given. Prose describing a change "
    "changes nothing. A fenced code snippet changes nothing. So write an "
    "edit block for every change you decided on above, quoting the existing "
    "lines exactly as they appear in the file you were shown — character for "
    "character, including indentation.\n"
    "Do not claim any file has been modified or saved. Nothing has been. The "
    "user will be shown your blocks as a diff and will decide whether to "
    "apply them.")


def build_step_prompt(question: str, steps: Sequence[str], index: int,
                      done: Sequence[str] = (), editing: bool = False) -> str:
    """Ask for one step's worth of work, with what is already done in view."""
    numbered = "\n".join(f"{n}. {s}" for n, s in enumerate(steps, 1))
    parts = [
        f"You are working through this request one step at a time.\n\n"
        f"Request: {question}\n\nYour plan:\n{numbered}",
    ]
    if done:
        recap = "\n\n".join(
            f"Step {n} ({steps[n - 1]}) produced:\n{text.strip()[:1200]}"
            for n, text in enumerate(done, 1))
        parts.append(f"What you have already done:\n\n{recap}")
    parts.append(
        f"Now do step {index + 1} only: {steps[index]}\n\n"
        f"Write just that step's work. Do not repeat earlier steps, do not "
        f"preview later ones, and do not add a summary — there will be one at "
        f"the end.")
    if editing:
        parts.append(STEP_HAS_NO_HANDS)
    return "\n\n".join(parts)


def build_summary_prompt(question: str, steps: Sequence[str],
                         done: Sequence[str], editing: bool = False) -> str:
    numbered = "\n".join(f"{n}. {s}" for n, s in enumerate(steps, 1))
    recap = "\n\n".join(f"Step {n}:\n{text.strip()[:1500]}"
                        for n, text in enumerate(done, 1))
    closing = (
        "Now write the final answer the user actually asked for. Pull the "
        "work together into one piece — do not describe your process, do not "
        "restate the steps as headings unless that genuinely is the clearest "
        "structure, and make sure anything the request explicitly asked for "
        "(a list, a summary, specific files) is there in full.")
    if editing:
        closing = f"{closing}\n\n{SUMMARY_MUST_PROPOSE}"
    return (
        f"You have finished working through this request:\n\n{question}\n\n"
        f"The plan was:\n{numbered}\n\nWhat each step produced:\n\n{recap}\n\n"
        f"{closing}")
