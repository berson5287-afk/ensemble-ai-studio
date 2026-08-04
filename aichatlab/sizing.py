"""Whether what we are about to ask for will actually fit on the machine.

Asking Ollama for a large `num_ctx` is what stops it silently truncating a
prompt — but the context window is not free. Its key/value cache is allocated
alongside the weights, and on a big model a big window can be several
gigabytes on top of an already large model. If the pair does not fit in VRAM,
Ollama does not refuse: it moves layers to the CPU and carries on, perhaps
twenty times slower, with no error and nothing on screen to explain it.

That is the failure this module exists to name. It makes no attempt to compute
VRAM precisely — layer counts and attention shapes vary per architecture and
guessing them from a filename would be a fiction. It flags the combination
that is worth a second look and tells the user the one command that answers
the question properly.
"""

from __future__ import annotations

import re

# "qwen2.5:32b-instruct-q4_K_M" -> 32.0, "llama3.2:3b" -> 3.0, "phi3:mini" -> None
SIZE = re.compile(r"[:\-_/]?(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)

# Above this, weights alone are already a large share of a consumer card.
BIG_MODEL_B = 20.0

# Above this, the key/value cache stops being a rounding error.
BIG_WINDOW = 16_000

CHECK = "ollama ps"


def parse_size(model: str) -> float | None:
    """Billions of parameters from a model name, or None if it doesn't say."""
    name = (model or "").lower()
    # A version like "qwen2.5" must not be read as 2.5 billion parameters, so
    # only look at the part after the tag separator when there is one.
    tail = name.split(":", 1)[1] if ":" in name else name
    match = SIZE.search(tail)
    if not match:
        return None
    try:
        size = float(match.group(1))
    except ValueError:
        return None
    return size if 0.1 <= size <= 2000 else None


def heavy_request(model: str, num_ctx: int) -> str:
    """A warning about a model-plus-window pair, or "" when it looks fine."""
    size = parse_size(model)
    if size is None or num_ctx < BIG_WINDOW:
        return ""
    if size < BIG_MODEL_B:
        return ""
    return (f"A {size:.0f}B model with a {num_ctx:,}-token window needs a lot "
            f"of VRAM for the context alone. If it does not fit, Ollama moves "
            f"layers to the CPU rather than refusing — same answers, many "
            f"times slower, and nothing says so. Run `{CHECK}` while this is "
            f"working: anything other than 100% GPU is why it is slow.")


def gpu_share(entry: dict) -> float | None:
    """Fraction of a loaded model that is resident on the GPU, 0.0 to 1.0."""
    try:
        total = float(entry.get("size") or 0)
        on_gpu = float(entry.get("size_vram") or 0)
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    return max(0.0, min(1.0, on_gpu / total))


def find_loaded(models, name: str) -> dict | None:
    """The `/api/ps` entry for a model, matching how Ollama names them."""
    wanted = (name or "").strip().lower()
    if not wanted:
        return None
    for entry in models or []:
        for key in ("name", "model"):
            value = str(entry.get(key, "")).strip().lower()
            if value == wanted or value.split(":")[0] == wanted.split(":")[0]:
                return entry
    return None


def placement_note(name: str, entry: dict) -> str:
    """Say where the model is running, but only when that is bad news."""
    share = gpu_share(entry)
    if share is None or share > 0.95:
        return ""
    size_gb = float(entry.get("size") or 0) / 1e9
    if share <= 0.01:
        where = "entirely on the CPU"
    else:
        where = f"only {share * 100:.0f}% on the GPU"
    return (f"🐌 Ollama has {name} loaded {where} ({size_gb:.0f} GB). That is "
            f"why this is slow — it did not refuse to run, it just moved the "
            f"work off the graphics card. A smaller model, or a smaller "
            f"context window in ⚙ Settings, will fit and run many times "
            f"faster.")


def human_gb(size_bytes: float) -> str:
    gb = float(size_bytes or 0) / 1e9
    return f"{gb:.1f} GB" if gb < 10 else f"{gb:.0f} GB"


def observed_vram(entry: dict) -> int:
    """A lower bound on the graphics card, learned from a real load.

    Ollama reports how much of a model it managed to put on the GPU.  Whatever
    that was, the card holds at least that much — which is all that is needed
    to say whether a 23 GB model stands a chance.
    """
    try:
        return int(entry.get("size_vram") or 0)
    except (TypeError, ValueError):
        return 0


# Ollama needs room for the context on top of the weights, so a model exactly
# the size of the card does not fit.
FIT_MARGIN = 0.85


def fits(model_bytes: int, vram_bytes: int) -> bool | None:
    """Will this model run on the card?  None when we have not seen enough."""
    if not vram_bytes or not model_bytes:
        return None
    return model_bytes <= vram_bytes * FIT_MARGIN


def suggest_after_timeout(model: str, prompt_tokens: int,
                          seconds: int) -> str:
    """What to actually try, once waiting has already failed once."""
    size = parse_size(model)
    rate = prompt_tokens / seconds if seconds else 0
    lines = [
        f"That is {rate:.0f} tokens per second of prompt reading, which is "
        f"CPU speed rather than GPU speed — the model is very likely not "
        f"fitting in VRAM alongside a window this large."]
    if size and size >= BIG_MODEL_B:
        lines.append(
            f"Worth trying, in order: a smaller model on the same folder "
            f"(a 7B or 14B will read {prompt_tokens:,} tokens in a fraction "
            f"of the time), or the same {size:.0f}B model on fewer files.")
    else:
        lines.append("Worth trying: fewer files in the folder, or a lower "
                     "token ceiling when attaching it.")
    lines.append(f"`{CHECK}` will tell you how much of it is on the GPU.")
    return " ".join(lines)
