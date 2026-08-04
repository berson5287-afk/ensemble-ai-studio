"""Folding an old conversation down instead of throwing it away.

`session.build_context` keeps a chat inside its token budget by dropping the
oldest messages whole. That is safe and cheap, and it is also amnesia: the
model stops knowing what you agreed forty turns ago, with no sign that
anything went missing.

Compaction is the alternative. The oldest stretch of the conversation is
handed to a model and comes back as a digest — decisions, facts, open threads,
in a few hundred tokens instead of several thousand — and that digest takes
its place in the history. Recent turns stay verbatim, because the detail that
matters most is usually the detail you just discussed.

Everything here is pure. The model call lives in the UI layer; this module
decides *what* to fold, builds the prompt, and splices the result back in.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .session import attachment_details, carries_attachment, estimate_tokens

# Compaction starts costing more than it saves below a few thousand tokens,
# and folding a chat that is barely started throws away detail for nothing.
MIN_TOKENS_TO_COMPACT = 1_500

# Fraction of the budget at which we start suggesting a compaction.
COMPACT_AT = 0.75

# Turns kept word for word at the end of the conversation.  Four is two
# full exchanges — enough that "do that again but in Python" still resolves.
KEEP_RECENT = 4

# The digest is marked so we can find it again: a second compaction has to
# fold the previous summary into the new one rather than stacking digests of
# digests, which is how a summary slowly turns into noise.
SUMMARY_PREFIX = "[Earlier conversation, summarised]"
SUMMARY_RE = re.compile(re.escape(SUMMARY_PREFIX))

MAX_SOURCE_CHARS = 24_000      # cap on what we hand the summariser at once


def is_summary(message: dict) -> bool:
    return str(message.get("content", "")).startswith(SUMMARY_PREFIX)


@dataclass(frozen=True)
class CompactionPlan:
    """Which messages get folded, which stay, and what it is worth."""

    fold: list[dict]
    keep: list[dict]
    previous_summary: str = ""
    pinned: list[dict] = field(default_factory=list)

    @property
    def folded_count(self) -> int:
        return len(self.fold)

    @property
    def tokens_before(self) -> int:
        return sum(estimate_tokens(m.get("content", ""))
                   for m in self.fold + self.keep + self.pinned)

    @property
    def worth_doing(self) -> bool:
        return bool(self.fold) and sum(
            estimate_tokens(m.get("content", "")) for m in self.fold) > 400


def pinned_tokens(messages: Sequence[dict]) -> int:
    return sum(estimate_tokens(m.get("content", ""))
               for m in plan(messages).pinned)


def remedy(messages: Sequence[dict], budget: int) -> str:
    """The short version, for the one line under the message box.

    Shares its reasoning with `explain_no_compaction` so the label and the
    dialog can never disagree — which is the whole failure being fixed here.
    """
    plan_result = plan(messages)
    if plan_result.worth_doing:
        return "🗜 Compact to free space"
    total = usage(messages)
    pinned = sum(estimate_tokens(m.get("content", ""))
                 for m in plan_result.pinned)
    if pinned > total - pinned:
        return "mostly attachments — a new chat is the way to free it"
    return "raise the context budget in ⚙ Settings"


def explain_no_compaction(messages: Sequence[dict], budget: int) -> str:
    """Why folding will not help here — and, more usefully, what will.

    "These conversations are still short enough" was the only answer this
    ever gave, and for the case that matters most it is simply false.  A chat
    holding a 50,000-token folder is not short; it is over the ceiling and
    unfoldable, because attachments are set aside rather than summarised on
    purpose.  Telling someone their conversation is too short while the line
    above says the oldest turns are being dropped reads as the app
    contradicting itself, and the reason it is wrong is that it never
    measured — it inferred one cause from one failed check.
    """
    plan_result = plan(messages)
    total = usage(messages)
    pinned = sum(estimate_tokens(m.get("content", ""))
                 for m in plan_result.pinned)
    conversation = total - pinned

    if total < budget * COMPACT_AT:
        return ("This conversation is still short enough that folding it "
                "would throw away detail without buying any room.")

    if pinned > conversation:
        names = []
        for message in plan_result.pinned:
            names.extend(n for n, _ in attachment_details(
                message.get("content", "")))
        what = ", ".join(dict.fromkeys(names)) or "the attached files"
        return (
            f"There is nothing here that can be folded. Of the {total:,} "
            f"tokens in this conversation, {pinned:,} are {what} — kept whole "
            f"on purpose, because a four-line précis of your source tree "
            f"cannot be turned back into your source tree. That leaves only "
            f"{conversation:,} tokens of actual conversation, which is not "
            f"worth summarising.\n\n"
            f"What does help: start a new chat and attach a smaller slice of "
            f"the folder, raise the context budget in ⚙ Settings if the model "
            f"and your graphics card can take it, or carry on and accept that "
            f"the oldest turns are being dropped — the attachment itself is "
            f"never dropped, so the model keeps the material even when it "
            f"loses the early discussion.")

    return (f"There is nothing old enough to fold yet — the {conversation:,} "
            f"tokens of conversation here are all within the most recent few "
            f"turns, which are always kept. Raising the context budget in "
            f"⚙ Settings is the lever that helps.")


def usage(messages: Sequence[dict]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def should_compact(messages: Sequence[dict], budget: int,
                   at: float = COMPACT_AT) -> bool:
    """Is this conversation close enough to the ceiling to be worth folding?"""
    used = usage(messages)
    if used < MIN_TOKENS_TO_COMPACT:
        return False
    return used >= max(1, int(budget * at))


def plan(messages: Sequence[dict], keep_recent: int = KEEP_RECENT) -> CompactionPlan:
    """Split a conversation into the part to fold and the part to keep.

    Two kinds of message are set aside rather than folded.

    An existing digest is pulled out so the summariser can be told to carry it
    forward — otherwise each compaction summarises the previous summary and
    detail decays a little more every time.

    Attached files and folders are pulled out because summarising them is
    destructive in a way nothing else here is: the folder you attached is not
    a turn of conversation to be recapped, it is the material the conversation
    is about, and a four-line précis of your source tree cannot be turned back
    into your source tree.  `build_context` already refuses to trim them; it
    would be a poor joke for the Compact button to delete them instead.
    """
    messages = list(messages)
    previous = ""
    pinned: list[dict] = []
    body: list[dict] = []
    for message in messages:
        if is_summary(message) and not previous:
            previous = str(message.get("content", ""))
        elif (message.get("pinned")
                or carries_attachment(message.get("content", ""))):
            pinned.append(message)
        else:
            body.append(message)

    keep_recent = max(0, int(keep_recent))
    if keep_recent:
        fold, keep = body[:-keep_recent], body[-keep_recent:]
    else:
        fold, keep = body, []
    return CompactionPlan(fold=fold, keep=keep, previous_summary=previous,
                          pinned=pinned)


def transcript_for(messages: Sequence[dict],
                   max_chars: int = MAX_SOURCE_CHARS) -> str:
    """Render messages as a readable transcript, newest kept if it must cut."""
    lines = []
    for message in messages:
        role = message.get("role", "")
        who = {"user": "User", "assistant": "Assistant"}.get(role, "System")
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{who}: {content}")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        # keep the end: the most recent of the folded turns matter most
        text = "…[earlier turns omitted]…\n\n" + text[-max_chars:]
    return text


def build_prompt(plan_result: CompactionPlan) -> str:
    """Ask for a digest that a model could actually continue a chat from."""
    parts = [
        "Summarise this conversation so it can be continued later by someone "
        "who was not present.\n"
        "Write it as compact notes, not prose. Keep, in this order:\n"
        "- What the user is trying to do, and any constraints they gave\n"
        "- Decisions made and conclusions reached, with the reasoning behind "
        "them\n"
        "- Specific facts that would be expensive to work out again: names, "
        "numbers, file paths, versions, settings\n"
        "- Anything still open or unresolved\n\n"
        "Leave out pleasantries, restatements and anything superseded later. "
        "Do not add advice or commentary of your own. Aim for under 400 "
        "words.",
    ]
    if plan_result.previous_summary:
        parts.append(
            "This is a summary of even earlier turns. Carry anything still "
            "relevant into your new summary, and drop what has since been "
            "settled:\n\n"
            + plan_result.previous_summary[len(SUMMARY_PREFIX):].strip())
    parts.append("Conversation to summarise:\n\n"
                 + transcript_for(plan_result.fold))
    parts.append("Summary:")
    return "\n\n".join(parts)


def clean_summary(text: str) -> str:
    """Strip the preamble models like to put in front of a summary."""
    body = (text or "").strip()
    body = re.sub(r"^(here(?:'s| is)[^\n:]{0,40}:|summary:)\s*", "", body,
                  flags=re.IGNORECASE)
    return SUMMARY_RE.sub("", body).strip()


def build_message(summary: str, folded_count: int) -> dict:
    """The digest, as a message that can sit in the history like any other."""
    body = clean_summary(summary)
    return {
        "role": "system",
        "content": (f"{SUMMARY_PREFIX} — {folded_count} earlier message(s) "
                    f"condensed. Treat this as established context you already "
                    f"know.\n\n{body}"),
    }


def apply(messages: Sequence[dict], summary: str,
          keep_recent: int = KEEP_RECENT) -> list[dict]:
    """Return the conversation with its old turns replaced by one digest.

    Attachments come first, then the digest, then the recent turns — which is
    roughly chronological, because an attachment is nearly always the thing
    that started the conversation.
    """
    plan_result = plan(messages, keep_recent=keep_recent)
    if not plan_result.fold:
        return list(messages)
    folded = plan_result.folded_count
    if plan_result.previous_summary:
        # the previous digest was folded in too, so say so honestly
        folded += 1
    return [*plan_result.pinned, build_message(summary, folded),
            *plan_result.keep]


def saving(before: Sequence[dict], after: Sequence[dict]) -> tuple[int, int]:
    """(tokens before, tokens after) — for telling the user what it bought."""
    return usage(before), usage(after)
