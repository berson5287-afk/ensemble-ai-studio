"""What graphics cards are actually here, and whether Ollama will use them all.

Everything else in this app has to infer the hardware from how badly things
went — `sizing.observed_vram` learns a lower bound on the card by watching how
much of a model Ollama managed to load.  That is a reasonable guess and it is
still only a guess.  `nvidia-smi` knows the answer, ships with every NVIDIA
driver, and answers in about fifty milliseconds.

The specific thing worth knowing is card *count*, because two 11 GB cards are
not one 22 GB card.  A model has to fit on one card unless Ollama splits it,
and Ollama's scheduler prefers to pack a model onto the fewest cards that will
hold it — which means a model that fits nowhere ends up partly on the CPU even
though the machine has plenty of total VRAM sitting idle in the second slot.
`OLLAMA_SCHED_SPREAD` is the documented switch for that: "Always schedule
model across all GPUs".

The awkward part is that the switch is an environment variable read by the
Ollama *server* when it starts.  Nothing this app can send over HTTP will
change it, and on a remote server it cannot be changed from here at all.  So
the honest thing is to detect the situation, say plainly what to set, and —
where the server is on this machine — offer to set it and explain that Ollama
has to be restarted before it takes effect.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass

SPREAD_VAR = "OLLAMA_SCHED_SPREAD"

FIELDS = ("--query-gpu=index,name,memory.total,memory.used",
          "--format=csv,noheader,nounits")

# `nvidia-smi` is normally on PATH because the driver puts it in System32,
# but that is not guaranteed — some driver installs leave it only in the
# NVSMI folder, and a GUI app launched from a shortcut does not always
# inherit the PATH a command prompt has.  A tool that silently finds nothing
# is indistinguishable from a machine with no cards, so look in the places it
# is actually installed before giving up.
WINDOWS_PATHS = (
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)

UNIX_PATHS = ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi")


def candidates() -> tuple[str, ...]:
    """Every place worth trying, best first."""
    found = shutil.which("nvidia-smi")
    paths = [found] if found else []
    extra = WINDOWS_PATHS if platform.system() == "Windows" else UNIX_PATHS
    paths.extend(path for path in extra if os.path.exists(path))
    return tuple(dict.fromkeys(paths))


QUERY = ("nvidia-smi", *FIELDS)

TRUTHY = {"1", "true", "yes", "on", "t", "y"}

# nvidia-smi reports mebibytes; everything else in this app is bytes.
MIB = 1024 * 1024
GIB = 1024 * MIB


def human_vram(size_bytes: float) -> str:
    """Card capacity the way the card is named.

    Model sizes come from Ollama in decimal gigabytes, but graphics cards are
    sold and known in binary ones: a GTX 1080 Ti reports 11,264 MiB, which is
    11.8 decimal GB.  Printing "12 GB GTX 1080 Ti" would look plainly wrong
    and cost the reader their trust in everything else in the message, so
    capacity is shown in the units the card is named in.  The arithmetic that
    decides what fits is done in bytes and is unaffected either way.
    """
    gib = float(size_bytes or 0) / GIB
    return f"{gib:.1f} GB" if gib < 10 else f"{gib:.0f} GB"


@dataclass(frozen=True)
class Card:
    index: int
    name: str
    total: int          # bytes
    used: int = 0       # bytes

    @property
    def free(self) -> int:
        return max(0, self.total - self.used)


def _run(command) -> str:
    """Run a read-only probe, returning "" rather than raising, ever."""
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=8,
            check=False,
            # Stop a console window flashing up on Windows.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def parse_cards(output: str) -> list[Card]:
    """Turn nvidia-smi's CSV into cards, skipping anything malformed."""
    cards: list[Card] = []
    for line in (output or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            total = int(float(parts[2])) * MIB
        except ValueError:
            continue
        try:
            used = int(float(parts[3])) * MIB if len(parts) > 3 else 0
        except ValueError:
            used = 0
        cards.append(Card(index=index, name=parts[1], total=total, used=used))
    return cards


def probe(runner=_run) -> list[Card]:
    """Every NVIDIA card on this machine, or [] if we cannot tell."""
    # The bare name first, so an injected runner in a test sees the same
    # command it always did.
    cards = parse_cards(runner(QUERY))
    if cards:
        return cards
    for path in candidates():
        if path == "nvidia-smi":
            continue
        cards = parse_cards(runner((path, *FIELDS)))
        if cards:
            return cards
    return []


def total_vram(cards) -> int:
    return sum(card.total for card in cards or [])


def largest_card(cards) -> int:
    """The biggest single card — the number that decides what fits unsplit."""
    return max((card.total for card in cards or []), default=0)


# Vendor prefixes eat half the width of a narrow sidebar and tell you nothing
# you did not already know from the model name that follows them.
NOISE = ("NVIDIA ", "GeForce ", "Corporation ")


def short_name(name: str) -> str:
    """"NVIDIA GeForce GTX 1080 Ti" -> "GTX 1080 Ti"."""
    trimmed = (name or "").strip()
    for prefix in NOISE:
        while trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix):]
    return trimmed or name


def short_describe(cards) -> str:
    """A sidebar-width summary: "2 × GTX 1080 Ti · 22 GB"."""
    if not cards:
        return ""
    names = {short_name(card.name) for card in cards}
    what = names.pop() if len(names) == 1 else f"{len(cards)} cards"
    if len(cards) == 1:
        return f"{what} · {human_vram(cards[0].total)}"
    return f"{len(cards)} × {what} · {human_vram(total_vram(cards))}"


# -- what is actually free ------------------------------------------------
def free_vram(cards) -> int:
    return sum(card.free for card in cards or [])


def largest_free(cards) -> int:
    """The most a single model could get without being split."""
    return max((card.free for card in cards or []), default=0)


def used_vram(cards) -> int:
    return sum(card.used for card in cards or [])


# Below this share of the card, something else is clearly resident.
BUSY_SHARE = 0.15


def looks_occupied(cards) -> bool:
    """Is a meaningful amount of this machine's VRAM already spoken for?"""
    total = total_vram(cards)
    return bool(total) and used_vram(cards) > total * BUSY_SHARE


def free_note(cards) -> str:
    """The line that answers "why did it go to the CPU?" without asking.

    Total capacity is the number everyone quotes and the wrong one to plan
    with: what decides where a model runs is what is free *right now*, on a
    single card.  Two Ollamas on one machine, or a game left running, and the
    22 GB on the box is 5 GB in practice — which looks from the outside like
    the scheduler making an inexplicable choice.
    """
    if not cards or not looks_occupied(cards):
        return ""
    free = free_vram(cards)
    biggest = largest_free(cards)
    line = (f"⚠ {human_vram(used_vram(cards))} of this machine's "
            f"{human_vram(total_vram(cards))} is already in use — "
            f"{human_vram(free)} free")
    if len(cards) > 1:
        line += f", at most {human_vram(biggest)} of it on one card"
    return (line + ". Anything larger than that goes to the processor. "
            "Another Ollama holding models open is the usual cause; "
            "quitting it, or setting OLLAMA_KEEP_ALIVE=0, releases them.")


def describe(cards) -> str:
    """"2 × GTX 1080 Ti (11 GB each, 22 GB total)"."""
    if not cards:
        return ""
    names = {card.name for card in cards}
    if len(cards) == 1:
        return f"{cards[0].name} ({human_vram(cards[0].total)})"
    what = names.pop() if len(names) == 1 else f"{len(cards)} cards"
    return (f"{len(cards)} × {what} ({human_vram(largest_card(cards))} each, "
            f"{human_vram(total_vram(cards))} total)")


# -- the spread setting ----------------------------------------------------
def _windows_user_env(name: str) -> str:
    """Read a persisted user variable, which os.environ may predate."""
    try:
        import winreg
    except ImportError:
        return ""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _kind = winreg.QueryValueEx(key, name)
            return str(value)
    except OSError:
        return ""


def spread_setting(environ=None) -> str:
    """The current value of OLLAMA_SCHED_SPREAD, or "" if it is unset.

    Checked in the persisted user environment as well as this process's own,
    because a variable set after this app started is invisible to `os.environ`
    and reporting it as unset would send the user round the loop again.
    """
    environ = os.environ if environ is None else environ
    value = str(environ.get(SPREAD_VAR, "") or "")
    if not value and platform.system() == "Windows":
        value = _windows_user_env(SPREAD_VAR)
    return value.strip()


def spread_enabled(environ=None) -> bool:
    return spread_setting(environ).lower() in TRUTHY


def worth_offering(cards, environ=None) -> bool:
    """Only worth raising with someone who has more than one card."""
    return len(cards or []) > 1 and not spread_enabled(environ)


def offer_note(cards) -> str:
    """Why this matters, in terms of the machine actually in front of them."""
    each = human_vram(largest_card(cards))
    total = human_vram(total_vram(cards))
    return (f"🎛 This machine has {describe(cards)}, but Ollama is not set to "
            f"spread a model across both. Its scheduler packs a model onto "
            f"the fewest cards that will hold it, so anything over {each} "
            f"can end up partly on the processor while the second card sits "
            f"idle — even though there is {total} between them. Setting "
            f"{SPREAD_VAR}=1 tells it to always use every card.")


def enable_spread(runner=None) -> tuple[bool, str]:
    """Persist OLLAMA_SCHED_SPREAD=1 for this user.

    Returns (worked, what to tell them).  This cannot take effect in a server
    that is already running: the variable is read at start-up, so Ollama has
    to be restarted, and saying so is part of the job.
    """
    if platform.system() != "Windows":
        return False, (
            f"Set {SPREAD_VAR}=1 in the environment Ollama starts with — for "
            f"a systemd service that is `systemctl edit ollama.service` and "
            f"an `Environment=\"{SPREAD_VAR}=1\"` line — then restart it.")
    runner = runner or _run
    # setx writes to the user environment permanently; it does not affect
    # processes that are already running, including Ollama.
    output = runner(("setx", SPREAD_VAR, "1"))
    if not output:
        return False, (
            f"Could not set it automatically. Add {SPREAD_VAR}=1 to your user "
            f"environment variables by hand, then restart Ollama.")
    os.environ[SPREAD_VAR] = "1"
    return True, (
        f"{SPREAD_VAR}=1 is set. Ollama reads it when it starts, so quit it "
        f"from the system tray and open it again before this takes effect — "
        f"then a model too big for one card will use both instead of falling "
        f"back to the processor.")


def placement_advice(cards, model_bytes: int) -> str:
    """Whether a specific model will fit, given the cards that are here."""
    if not cards or not model_bytes:
        return ""
    biggest = largest_card(cards)
    total = total_vram(cards)
    size_gb = model_bytes / 1e9
    # Free space first: a card that is full is not a card you can use, and
    # saying "this fits" because it fits the *empty* card is the mistake the
    # user then spends an hour trying to explain.
    if looks_occupied(cards) and model_bytes > largest_free(cards):
        if model_bytes <= free_vram(cards) * 0.9 and len(cards) > 1:
            return (f"This model is {size_gb:.0f} GB and no single card has "
                    f"that free ({human_vram(largest_free(cards))} is the "
                    f"most on one), though there is "
                    f"{human_vram(free_vram(cards))} between them.")
        return (f"This model is {size_gb:.0f} GB but only "
                f"{human_vram(free_vram(cards))} of video memory is free, so "
                f"it will run on the processor. Something else is holding "
                f"{human_vram(used_vram(cards))} — another Ollama with models "
                f"still loaded is the usual reason.")
    if model_bytes <= biggest * 0.85:
        return ""
    if len(cards) > 1 and model_bytes <= total * 0.85:
        if spread_enabled():
            return ""
        return (f"This model is {size_gb:.0f} GB, which will not fit on one "
                f"{human_vram(biggest)} card but would fit across both. "
                f"Ollama will not use both unless {SPREAD_VAR}=1 is set.")
    return (f"This model is {size_gb:.0f} GB and there is "
            f"{human_vram(total)} of video memory in this machine, so part "
            f"of it will run on the processor however it is scheduled — many "
            f"times slower. A smaller model, or a smaller quantisation of "
            f"this one, is the fix.")
