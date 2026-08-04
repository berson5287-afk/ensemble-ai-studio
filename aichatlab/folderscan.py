"""Reading a whole folder into a prompt — safely, and with the cost shown up front.

Attaching a folder is where a chat app quietly breaks. A project directory is
easily a million characters; the context budget is a few thousand tokens. The
failure mode is not an error, it is silent truncation of exactly the files you
cared about, followed by a confident answer based on whatever survived.

So nothing here reads a byte until the user has seen what would be sent and
what it costs. `survey` walks and measures (stat only — no file is opened),
`select` applies the limits, `plan_budget` prices the result against the
context budget, and only `build_block` actually reads. The dialog on top of
this module is a view; every decision it presents is computed here, which is
why the interesting behaviour is testable without a display.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .documents import BinaryFileError, ExtractionError, extract
from .session import CHARS_PER_TOKEN

# Build output, dependency trees and VCS internals: thousands of files that
# nobody means when they say "look at my project".  Skipped by default, and
# the dialog says so rather than hiding it.
NOISE_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", "site-packages", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", "dist", "build", ".next", ".nuxt", "target", ".idea", ".vscode",
    ".gradle", "obj", ".terraform", "vendor", ".cache", "coverage",
    ".eggs", ".parcel-cache", "bower_components",
})

CODE_SUFFIXES = frozenset({
    ".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cc", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts",
    ".lua", ".pl", ".r", ".scala", ".sh", ".bash", ".zsh", ".ps1", ".bat",
    ".cmd", ".sql", ".css", ".scss", ".less", ".vue", ".svelte", ".m", ".vb",
})

TEXT_SUFFIXES = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".log", ".json",
    ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".html", ".htm", ".properties", ".gitignore", ".dockerfile", "",
})

DOC_SUFFIXES = frozenset({".pdf", ".docx", ".pptx", ".xlsx", ".xlsm"})

KIND_LABELS = {"code": "code", "text": "text & data", "docs": "documents"}

# Extracted text is a fraction of a document's bytes — a 4 MB PowerPoint is
# mostly images.  Estimating from raw size without this over-states the cost
# by an order of magnitude and scares people off a folder that would fit fine.
TEXT_YIELD = {".pdf": 0.12, ".docx": 0.08, ".pptx": 0.04,
              ".xlsx": 0.25, ".xlsm": 0.25}

DEFAULT_MAX_FILES = 60
DEFAULT_MAX_FILE_BYTES = 256 * 1024
DEFAULT_MAX_TOKENS = 40_000
DEFAULT_MAX_DEPTH = 4

# A survey stops here rather than walking a million-file drive for a progress
# number nobody reads.
SURVEY_LIMIT = 20_000


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" or size >= 10 else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def kind_of(name: str) -> str | None:
    """Which group a filename belongs to, or None if we can't read it."""
    suffix = Path(name).suffix.lower()
    if suffix in DOC_SUFFIXES:
        return "docs"
    if suffix in CODE_SUFFIXES:
        return "code"
    # Dotfiles like ".gitignore" have the whole name as the suffix
    if suffix in TEXT_SUFFIXES or Path(name).name.lower() in TEXT_SUFFIXES:
        return "text"
    return None


@dataclass(frozen=True)
class FileEntry:
    path: Path
    relative: str
    size: int
    kind: str
    depth: int

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def tokens(self) -> int:
        """Estimated prompt cost, allowing for how much text a format yields."""
        chars = self.size * TEXT_YIELD.get(self.suffix, 1.0)
        return max(1, int(chars / CHARS_PER_TOKEN))


@dataclass
class Survey:
    """What is in a folder, measured without opening anything."""

    root: Path
    entries: list[FileEntry] = field(default_factory=list)
    skipped_dirs: list[str] = field(default_factory=list)
    unreadable: int = 0          # files with no extractor (images, binaries…)
    truncated: bool = False      # we stopped walking at SURVEY_LIMIT

    @property
    def total_bytes(self) -> int:
        return sum(entry.size for entry in self.entries)

    @property
    def max_depth(self) -> int:
        return max((entry.depth for entry in self.entries), default=0)

    def kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.kind] = counts.get(entry.kind, 0) + 1
        return counts

    def describe(self) -> str:
        if not self.entries:
            return "No readable files found in this folder."
        bits = [f"{len(self.entries):,} readable file"
                f"{'s' if len(self.entries) != 1 else ''}",
                human_size(self.total_bytes)]
        counts = self.kind_counts()
        detail = ", ".join(f"{counts[kind]} {KIND_LABELS[kind]}"
                           for kind in ("code", "text", "docs")
                           if counts.get(kind))
        if detail:
            bits.append(detail)
        return " · ".join(bits)


@dataclass(frozen=True)
class Limits:
    """Everything the user can turn down, in one immutable bundle."""

    max_files: int = DEFAULT_MAX_FILES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_depth: int = DEFAULT_MAX_DEPTH
    kinds: frozenset = frozenset({"code", "text", "docs"})

    @classmethod
    def everything(cls) -> Limits:
        """"Give it all access" — no caps, but still only readable files.

        Noise folders are handled separately (they are a property of the walk,
        not of the selection) so that "all access" can still mean "all of my
        actual work" rather than "all 40,000 files in node_modules".
        """
        return cls(max_files=SURVEY_LIMIT, max_file_bytes=20_000_000,
                   max_tokens=100_000_000, max_depth=64)


@dataclass
class Selection:
    """The files that would actually be sent, and why the rest were not.

    `readable` is the wider set: every file a model *could* be shown —
    right kind, sane size, not buried too deep — before the token budget
    decides how many fit one prompt.  The distinction is "what the app
    holds" versus "what one message can carry", and conflating them is how
    attaching a 14-file folder used to mean the app itself only knew about
    6 of them.
    """

    chosen: list[FileEntry] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)
    readable: list[FileEntry] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(entry.tokens for entry in self.chosen)

    @property
    def total_bytes(self) -> int:
        return sum(entry.size for entry in self.chosen)

    @property
    def dropped_count(self) -> int:
        return sum(self.dropped.values())

    def summary(self) -> str:
        if not self.chosen:
            return "Nothing would be sent — the limits exclude every file."
        text = (f"{len(self.chosen):,} file"
                f"{'s' if len(self.chosen) != 1 else ''} · "
                f"≈{self.tokens:,} tokens")
        if self.dropped_count:
            text += f" · {self.dropped_count:,} left out"
        return text

    def dropped_detail(self) -> str:
        labels = {"kind": "wrong type", "depth": "too deeply nested",
                  "size": "too large", "cap": "over the file limit",
                  "budget": "over the token limit"}
        parts = [f"{count:,} {labels.get(reason, reason)}"
                 for reason, count in sorted(self.dropped.items())
                 if count]
        return ", ".join(parts)


def survey(root, skip_noise: bool = True, skip_hidden: bool = True,
           limit: int = SURVEY_LIMIT) -> Survey:
    """Walk a folder and measure it.  Opens nothing, so it stays fast."""
    root = Path(root)
    result = Survey(root=root)
    if not root.is_dir():
        return result

    root_str = str(root)
    for current, dirnames, filenames in os.walk(root_str):
        keep = []
        for name in sorted(dirnames):
            if skip_hidden and name.startswith(".") and name not in NOISE_DIRS:
                result.skipped_dirs.append(name)
                continue
            if skip_noise and name.lower() in NOISE_DIRS:
                result.skipped_dirs.append(name)
                continue
            keep.append(name)
        dirnames[:] = keep

        relative_dir = os.path.relpath(current, root_str)
        depth = 0 if relative_dir == "." else relative_dir.count(os.sep) + 1

        for name in sorted(filenames):
            if skip_hidden and name.startswith("."):
                # a dotfile with a known name (.gitignore) is still content
                if kind_of(name) is None:
                    continue
            kind = kind_of(name)
            if kind is None:
                result.unreadable += 1
                continue
            path = Path(current) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            relative = os.path.relpath(str(path), root_str).replace(os.sep, "/")
            result.entries.append(
                FileEntry(path=path, relative=relative, size=size, kind=kind,
                          depth=depth))
            if len(result.entries) >= limit:
                result.truncated = True
                return result
    return result


def _sort_key(entry: FileEntry) -> tuple:
    """Shallow files first, README first of all — the orientation you'd want."""
    stem = Path(entry.relative).stem.lower()
    lead = 0 if stem in ("readme", "index") else 1
    return (lead, entry.depth, entry.relative.lower())


def select(survey_result: Survey, limits: Limits) -> Selection:
    """Apply the limits, in the order a person would explain them."""
    selection = Selection()
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    candidates = []
    for entry in survey_result.entries:
        if entry.kind not in limits.kinds:
            drop("kind")
        elif entry.depth > limits.max_depth:
            drop("depth")
        elif entry.size > limits.max_file_bytes:
            drop("size")
        else:
            candidates.append(entry)
    # The kind/depth/size drops are judgements about the file itself and
    # apply everywhere.  The cap and budget drops below are only about what
    # fits in one prompt, so they must not shrink this list.
    selection.readable = sorted(candidates, key=_sort_key)

    used = 0
    for entry in sorted(candidates, key=_sort_key):
        if len(selection.chosen) >= limits.max_files:
            drop("cap")
            continue
        if used + entry.tokens > limits.max_tokens and selection.chosen:
            drop("budget")
            continue
        selection.chosen.append(entry)
        used += entry.tokens

    selection.dropped = dropped
    return selection


# ------------------------------------------------------------ token budgeting

# Room left for the question, the reply and the rest of the conversation, so
# attaching a folder doesn't leave the model with nothing to say.
BUDGET_HEADROOM = 2_000


@dataclass(frozen=True)
class BudgetPlan:
    needed: int          # tokens the folder itself will cost
    current: int         # the context budget as configured today
    recommended: int     # what it would have to become
    fits: bool
    ceiling: int = 0     # the largest window we are willing to ask a model for
    over_ceiling: bool = False

    def message(self) -> str:
        if self.over_ceiling:
            return (f"≈{self.needed:,} tokens — more than the {self.ceiling:,}-"
                    f"token window this app will ask a model for. Most of this "
                    f"folder would be dropped before the model ever saw it.")
        if self.fits:
            return (f"≈{self.needed:,} tokens — fits inside your "
                    f"{self.current:,}-token context budget.")
        return (f"≈{self.needed:,} tokens, but your context budget is "
                f"{self.current:,}. Older messages would be dropped unless the "
                f"budget is raised to {self.recommended:,}.")


def plan_budget(tokens_needed: int, current_budget: int,
                headroom: int = BUDGET_HEADROOM,
                ceiling: int | None = None) -> BudgetPlan:
    """Decide whether the context budget has to go up, and to what.

    The ceiling matters more than it looks.  Raising our own trimming budget
    to 96,000 tokens while still asking Ollama for a 32,768-token window is
    strictly worse than not raising it at all: we stop trimming, and the
    server drops the oldest two thirds of the prompt instead — silently, and
    without choosing sensibly about what to lose.  So the recommendation is
    never allowed above the window we would actually request.
    """
    current = max(0, int(current_budget))
    target = int(tokens_needed) + headroom
    if target <= current:
        return BudgetPlan(int(tokens_needed), current, current, True,
                          ceiling=int(ceiling or 0))
    step = 1_000 if target < 20_000 else 4_000
    recommended = int((target + step - 1) // step * step)
    if ceiling and recommended > int(ceiling):
        return BudgetPlan(int(tokens_needed), current, int(ceiling), False,
                          ceiling=int(ceiling), over_ceiling=True)
    return BudgetPlan(int(tokens_needed), current, recommended, False,
                      ceiling=int(ceiling or 0))


# ------------------------------------------------------------------- reading

def worth_holding(selection: Selection, cap_bytes: int) -> list[FileEntry]:
    """The files to read and keep: sent ones first, then the rest that fit.

    The chosen files are first so that, under the cap, the files actually in
    the prompt are never the ones sacrificed.  Everything else readable
    follows until the byte cap says stop — a cap sized for the pathological
    folder, which ordinary projects never reach.
    """
    holding = list(selection.chosen)
    names = {entry.relative for entry in holding}
    held = sum(entry.size for entry in holding)
    for entry in selection.readable:
        if entry.relative in names:
            continue
        if held + entry.size > cap_bytes:
            continue
        holding.append(entry)
        names.add(entry.relative)
        held += entry.size
    return holding


def read_texts(entries: Sequence[FileEntry],
               read_fn: Callable[[Path], str] | None = None,
               progress: Callable[[int, int], None] | None = None) -> dict:
    """{relative name: text} for every readable file.

    Kept separate from `build_block` because retrieval needs the contents as
    data — to summarise, embed and rank — long before it knows which of them
    will end up in a prompt.  An unreadable file is skipped rather than
    raising: one bad file in two hundred should not cost the other one
    hundred and ninety-nine.
    """
    read = read_fn or extract
    tell = progress or (lambda _done, _total: None)
    entries = list(entries)
    texts: dict = {}
    for index, entry in enumerate(entries, start=1):
        try:
            text = read(entry.path)
        except Exception:
            text = ""
        if text and text.strip():
            texts[entry.relative] = text
        tell(index, len(entries))
    return texts


def build_block(root, entries: Sequence[FileEntry], total_files: int,
                read_fn: Callable[[Path], str] | None = None,
                progress: Callable[[int, int], None] | None = None,
                char_budget: int | None = None) -> str:
    """Read the chosen files and format them as one attachable block.

    A file that cannot be read becomes a one-line note rather than an
    exception: one unreadable file in a folder of two hundred should not throw
    away the other hundred and ninety-nine.
    """
    root = Path(root)
    read = read_fn or extract
    tell = progress or (lambda _done, _total: None)
    entries = list(entries)

    manifest = []
    sections = []
    truncated_files = 0

    # `char_budget` is a promise about the whole block, so the manifest, the
    # per-file headings and the fences all have to be paid for out of it —
    # counting only the file text overshoots by exactly enough to push the
    # finished block back over the caller's limit.
    scaffolding = sum(len(entry.relative) * 2 + 80 for entry in entries) + 400
    used = scaffolding if char_budget is not None else 0

    for index, entry in enumerate(entries, 1):
        tell(index, len(entries))
        try:
            text = read(entry.path)
        except BinaryFileError:
            text = "[skipped — this file is binary, not text]"
        except (ExtractionError, OSError, ValueError) as exc:
            text = f"[could not read this file: {exc}]"

        if char_budget is not None:
            remaining = char_budget - used
            if remaining <= 0:
                truncated_files += len(entries) - index + 1
                break
            if len(text) > remaining:
                text = text[:remaining] + "\n…[truncated to fit the budget]…"
                truncated_files += 1
        used += len(text)

        manifest.append(f"  {entry.relative} ({human_size(entry.size)})")
        sections.append(f"--- {entry.relative} ---\n```\n{text}\n```")

    name = root.name or root.resolve().name or str(root)
    header = [f"[Folder: {name} — {len(sections)} of {total_files:,} "
              f"file{'s' if total_files != 1 else ''} included]",
              "Files included:", *manifest]
    left_out = total_files - len(sections)
    if left_out > 0:
        header.append(f"\n{left_out:,} other file(s) in this folder were not "
                      f"included, so say so plainly if the answer depends on "
                      f"something you cannot see here.")
    if truncated_files:
        header.append(f"{truncated_files} file(s) were shortened to fit.")

    return ("\n".join(header) + "\n\n" + "\n\n".join(sections)
            + "\n\n[End of folder contents.]")
