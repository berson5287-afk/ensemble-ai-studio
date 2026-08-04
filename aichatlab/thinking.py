"""Reasoning models, and the two ways they break an app that ignores them.

Qwen3, DeepSeek-R1 and GPT-OSS reason before they answer, and Ollama turns
that on by default.  The reasoning does not arrive in `message.content` — it
comes in a separate `message.thinking` field.  An app that reads only
`content`, as this one did, gets two problems for the price of one:

1. **A long silence.**  Nothing streams while the model reasons, so a reply
   that is going perfectly well is indistinguishable from one that has hung.
   That is the exact complaint this app spent a version fixing for slow
   models, reintroduced by a model choice.

2. **A truncated answer.**  Reasoning tokens are generated tokens, so they
   come out of the same `num_predict` allowance as the reply.  At the Quick
   end of the speed slider that allowance is 192 tokens — comfortably less
   than a reasoning model spends thinking.  The answer arrives empty or cut
   off mid-word, and the cut-off detector blames the slider without knowing
   that nothing was left by the time the model started writing.

The other trap is that `think` cannot simply be sent to everything.  Ollama
rejects it outright on a model that does not reason, so it has to be sent only
where it is understood — and GPT-OSS understands only the level strings, not
the booleans everything else takes.
"""

from __future__ import annotations

import re

# Families that reason.  Matched against the whole tag, because the marker can
# be in either half: "qwen3:8b" but also "deepseek-r1:7b" and "magistral".
FAMILIES = (
    "qwen3",
    "deepseek-r1",
    "deepseek-v3.1",
    "gpt-oss",
    "magistral",
    "phi4-reasoning",
    "exaone-deep",
    "smallthinker",
    "qwq",
)

# GPT-OSS takes "low"/"medium"/"high" and ignores true/false, so a boolean
# would silently leave it reasoning at whatever the default is.
LEVELS_ONLY = ("gpt-oss",)

DEFAULT_LEVEL = "medium"
OFF_LEVEL = "low"

# Reasoning is not free, and it is spent before the first word of the answer.
# This is the floor the reply allowance is raised to when thinking is on.
MIN_ALLOWANCE = 1_500

# What reasoning typically costs, on top of whatever the reply itself needs.
THINKING_ALLOWANCE = 1_200

# Older builds and the /api/generate fallback path put the reasoning inline in
# the content instead of in its own field.
INLINE = re.compile(
    r"<(think|thinking|reasoning)>(?P<body>.*?)</\1>\s*",
    re.DOTALL | re.IGNORECASE)

# A reply cut off mid-reasoning never closes its tag.
UNCLOSED = re.compile(r"<(think|thinking|reasoning)>(?P<body>.*)\Z",
                      re.DOTALL | re.IGNORECASE)


# `think` was introduced in Ollama v0.9.0 (May 2025).  Older servers ignore
# the field — Go discards unknown JSON keys — so sending it is harmless, but
# knowing the version is what lets the app explain why reasoning is missing
# rather than leaving the user to wonder.
THINK_SINCE = (0, 9, 0)


def parse_version(text: str) -> tuple[int, ...]:
    """"0.6.2" -> (0, 6, 2).  Returns () for anything unparseable."""
    cleaned = (text or "").strip().lstrip("vV").split("-")[0].split("+")[0]
    parts = cleaned.split(".")
    if not parts or not parts[0].isdigit():
        return ()
    numbers = []
    for part in parts:
        if not part.isdigit():
            break
        numbers.append(int(part))
    return tuple(numbers)


def supports_think_api(version: str) -> bool:
    """Is this server new enough to understand `think`?

    An unknown version is treated as new enough: an old server ignores the
    field, so guessing wrong in this direction costs nothing, while guessing
    wrong the other way would silently disable reasoning on a capable server.
    """
    parsed = parse_version(version)
    if not parsed:
        return True
    return parsed >= THINK_SINCE


# The runner's own words when a model's architecture predates the server.
UNSUPPORTED = "not supported by your version"


def outdated_server_note(message: str, model: str, version: str = "") -> str:
    """Turn Ollama's terse 500 into something that says what to do.

    "llama runner process has terminated: this model is not supported by your
    version of Ollama" is accurate and useless — it does not say which version
    you have, which one you need, or that `ollama pull` will not fix it
    because the *server* is what is out of date, not the model.
    """
    if UNSUPPORTED not in (message or "").lower():
        return ""
    running = f"This server is running Ollama {version}. " if version else ""
    return (f"{running}{model} needs a newer Ollama than this machine has — "
            f"the model downloaded fine, but the server cannot run its "
            f"architecture. Updating Ollama itself is the fix; re-pulling the "
            f"model will not help. Download the current installer from "
            f"ollama.com and run it over the top of the existing install: it "
            f"keeps every model you have already got.")


def supports_thinking(model: str) -> bool:
    """Does this model reason?  Sending `think` to one that doesn't is an error."""
    name = (model or "").lower()
    return any(family in name for family in FAMILIES)


def takes_levels(model: str) -> bool:
    name = (model or "").lower()
    return any(family in name for family in LEVELS_ONLY)


def think_value(model: str, enabled: bool, level: str = DEFAULT_LEVEL):
    """What to put in the request's `think` field, or None to leave it out.

    None matters: omitting the field is the only correct thing to do for a
    model that does not reason, because Ollama refuses the request rather
    than ignoring the parameter.
    """
    if not supports_thinking(model):
        return None
    if takes_levels(model):
        # There is no "off" for GPT-OSS, only less of it.
        return level if enabled else OFF_LEVEL
    return bool(enabled)


def allowance(num_predict: int, model: str, enabled: bool) -> int:
    """Raise the reply allowance so reasoning does not eat the answer.

    Without this, "Quick" on the speed slider means 192 tokens total, and a
    reasoning model spends all of them before it starts writing.
    """
    if not enabled or not supports_thinking(model) or num_predict <= 0:
        return num_predict
    return max(num_predict + THINKING_ALLOWANCE, MIN_ALLOWANCE)


def split_inline(text: str) -> tuple[str, str]:
    """Separate `<think>…</think>` from the answer, for models that inline it.

    Returns (reasoning, answer).  Native Ollama thinking arrives in its own
    field and never reaches here; this covers the /api/generate fallback and
    servers old enough to predate the `thinking` field.
    """
    if not text:
        return "", text
    thoughts: list[str] = []

    def take(match: re.Match) -> str:
        thoughts.append(match.group("body").strip())
        return ""

    answer = INLINE.sub(take, text)
    # A stream cut off mid-thought leaves the tag open; everything after it is
    # reasoning, not answer, and pasting it into the reply reads as gibberish.
    unclosed = UNCLOSED.search(answer)
    if unclosed:
        thoughts.append(unclosed.group("body").strip())
        answer = answer[:unclosed.start()]
    return "\n\n".join(t for t in thoughts if t), answer.strip()


def summarise(reasoning: str, limit: int = 90) -> str:
    """A one-line gist of the reasoning, for the collapsed header."""
    flat = " ".join((reasoning or "").split())
    if not flat:
        return ""
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s"
