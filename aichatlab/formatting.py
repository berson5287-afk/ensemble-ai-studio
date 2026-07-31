"""Turning model output into something that reads like a person wrote it.

Two jobs, both deliberately free of any UI imports so they can be tested
without a display:

* `friendly_model_name` — "qwen2.5:32b-instruct-q4_K_M" is a filename, not a
  name you'd use in conversation.  This makes it "Qwen2.5 32B".
* `render_markdown` — models emit markdown whether you ask for it or not.
  Left alone it shows up as literal asterisks, which is the single thing that
  makes a chat look most like a machine dump.  This turns it into styled runs
  the chat view can apply Tk tags to.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------- model names

# Families whose casing people actually write a particular way.
KNOWN_FAMILIES = {
    "gpt": "GPT",
    "gpt-oss": "GPT-OSS",
    "qwen": "Qwen",
    "llama": "Llama",
    "gemma": "Gemma",
    "phi": "Phi",
    "mistral": "Mistral",
    "mixtral": "Mixtral",
    "deepseek": "DeepSeek",
    "codellama": "CodeLlama",
    "minicpm": "MiniCPM",
    "nomic": "Nomic",
    "starcoder": "StarCoder",
    "tinyllama": "TinyLlama",
}

SIZE_RE = re.compile(r"^\d+(?:\.\d+)?x?\d*\.?\d*[bm]$", re.IGNORECASE)
# noise people don't say out loud: quantisation, tuning and precision suffixes
NOISE_RE = re.compile(
    r"^(instruct|chat|it|text|base|latest|fp16|bf16|f16|f32|q\d.*|k_?[sml]?|"
    r"\d+bit|uncensored|tools|vision)$",
    re.IGNORECASE)


def _prettify_word(word: str) -> str:
    """One hyphen-separated chunk: "llama3.2" → "Llama3.2", "gpt" → "GPT"."""
    match = re.match(r"^([a-z]+)([\d.]*)$", word, re.IGNORECASE)
    if not match:
        return word[:1].upper() + word[1:]
    stem, version = match.group(1), match.group(2)
    pretty = KNOWN_FAMILIES.get(stem.lower(), stem[:1].upper() + stem[1:])
    return f"{pretty}{version}"


def _prettify_family(family: str) -> str:
    if family.lower() in KNOWN_FAMILIES:
        return KNOWN_FAMILIES[family.lower()]
    return "-".join(_prettify_word(part) for part in family.split("-") if part)


# capability words that read better as a suffix than inside the name
CAPABILITIES = (("vision", "Vision"), ("coder", "Coder"), ("code", "Coder"))


def friendly_model_name(model: str) -> str:
    """A name you could say out loud: qwen2.5:32b-instruct-q4_K_M → Qwen2.5 32B."""
    if not model:
        return ""
    family, _, variant = model.partition(":")

    parts = family.split("-")
    extras = []
    for marker, label in CAPABILITIES:
        if any(part.lower() == marker for part in parts):
            parts = [part for part in parts if part.lower() != marker]
            if label not in extras:
                extras.append(label)
    for marker, label in CAPABILITIES:
        if re.search(rf"(?:^|[-_]){marker}(?:$|[-_])", variant, re.IGNORECASE):
            if label not in extras:
                extras.append(label)

    name = _prettify_family("-".join(parts)) if parts else _prettify_family(family)

    size = ""
    for part in re.split(r"[-_]", variant):
        if SIZE_RE.match(part):
            size = part.upper()
            break

    return " ".join(part for part in [name, size, *extras] if part)


def initials(model: str) -> str:
    """Two-letter badge for a model, e.g. "Qwen2.5 32B" → "Qw"."""
    name = friendly_model_name(model) or model
    letters = re.sub(r"[^A-Za-z]", "", name)
    return (letters[:2] or "AI").capitalize()


# ------------------------------------------------------------------ markdown

Run = "tuple[str, tuple[str, ...]]"

INLINE_RE = re.compile(
    r"(?P<code>`[^`\n]+`)"
    r"|(?P<bolditalic>\*\*\*[^*\n]+?\*\*\*)"
    r"|(?P<bold>\*\*[^*\n]+?\*\*|__[^_\n]+?__)"
    r"|(?P<italic>(?<![\w*])\*[^*\n]+?\*(?!\*))"
    r"|(?P<link>\[[^\]\n]+\]\([^)\s]+\))"
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET_RE = re.compile(r"^([-*+])\s+(.*)$")
NUMBER_RE = re.compile(r"^(\d+[.)])\s+(.*)$")
RULE_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")


def parse_inline(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """Split one line into (text, tags) runs for bold / italic / code / links."""
    runs: list[tuple[str, tuple[str, ...]]] = []
    position = 0

    for match in INLINE_RE.finditer(text):
        if match.start() > position:
            runs.append((text[position:match.start()], ()))
        kind = match.lastgroup
        body = match.group()

        if kind == "code":
            runs.append((body[1:-1], ("md_code",)))
        elif kind == "bolditalic":
            runs.append((body[3:-3], ("md_bold", "md_italic")))
        elif kind == "bold":
            runs.append((body[2:-2], ("md_bold",)))
        elif kind == "italic":
            runs.append((body[1:-1], ("md_italic",)))
        elif kind == "link":
            label, _, target = body[1:].partition("](")
            runs.append((label, ("md_link",)))
            runs.append((f" ({target.rstrip(')')})", ("md_muted",)))
        position = match.end()

    if position < len(text):
        runs.append((text[position:], ()))
    return [run for run in runs if run[0]]


def render_markdown(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """Turn markdown into (text, tags) runs, newlines included."""
    runs: list[tuple[str, tuple[str, ...]]] = []
    in_code_block = False

    lines = text.split("\n")
    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            runs.append((line + "\n", ("md_code_block",)))
            continue
        if RULE_RE.match(stripped):
            runs.append(("─" * 30 + "\n", ("md_muted",)))
            continue

        block: tuple[str, ...] = ()
        content = line

        heading = HEADING_RE.match(stripped)
        bullet = BULLET_RE.match(stripped)
        numbered = NUMBER_RE.match(stripped)

        if heading:
            block, content = ("md_heading",), heading.group(2)
        elif bullet:
            indent = " " * (len(line) - len(line.lstrip()))
            block, content = ("md_bullet",), f"{indent}•  {bullet.group(2)}"
        elif numbered:
            indent = " " * (len(line) - len(line.lstrip()))
            block = ("md_bullet",)
            content = f"{indent}{numbered.group(1)}  {numbered.group(2)}"

        for run_text, run_tags in parse_inline(content):
            runs.append((run_text, block + run_tags))

        if index < len(lines) - 1:
            runs.append(("\n", block))

    return runs


MARKDOWN_HINT = re.compile(r"(\*\*|`|^\s*[-*+]\s+|^\s*#{1,6}\s+|\[[^\]]+\]\()",
                           re.MULTILINE)


def has_markdown(text: str) -> bool:
    """Cheap check so plain replies skip the re-render entirely."""
    return bool(text) and bool(MARKDOWN_HINT.search(text))


def tidy(text: str) -> str:
    """Trim the padding models like to leave around a reply."""
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip())
