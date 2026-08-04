"""The "you dropped a folder on me" conversation.

The whole point of this dialog is that the user sees the cost before paying
it. Every control recomputes the file count, the token estimate and the effect
on the context budget immediately, so "give it everything" and "just the code,
top two levels" are a click apart and you can see what each one means.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

from ..folderscan import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_FILES,
    KIND_LABELS,
    Limits,
    Survey,
    human_size,
    plan_budget,
    select,
)
from .theme import (
    BORDER,
    ERROR,
    FONT,
    FONT_BOLD,
    FONT_SMALL,
    MUTED,
    OK,
    PANEL,
    SUBTLE,
    flat_button,
    separator,
)

FILE_CHOICES = (20, 60, 150, 400, 1000)
SIZE_CHOICES = ((64, "64 KB"), (256, "256 KB"), (1024, "1 MB"), (8192, "8 MB"))
DEPTH_CHOICES = ((1, "Top level only"), (2, "2 levels"),
                 (4, "4 levels"), (64, "Every subfolder"))
TOKEN_CHOICES = ((8_000, "8k"), (24_000, "24k"), (60_000, "60k"),
                 (150_000, "150k"))


class FolderScanDialog(tk.Toplevel):
    """Ask how much of a folder to read, then hand back Limits + a budget."""

    def __init__(self, master, survey_result: Survey, context_budget: int,
                 on_accept: Callable[[Limits, int], None],
                 max_window: int = 32768):
        super().__init__(master)
        self.survey = survey_result
        self.context_budget = int(context_budget)
        self.max_window = int(max_window)
        self.on_accept = on_accept
        self.plan = None

        self.title(f"Add folder — {survey_result.root.name}")
        self.configure(bg=PANEL, padx=18, pady=14)
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        try:
            self.geometry(f"+{master.winfo_rootx() + 120}+{master.winfo_rooty() + 70}")
        except tk.TclError:
            pass

        self.access = tk.StringVar(value="limited")
        self.max_files = tk.IntVar(value=DEFAULT_MAX_FILES)
        self.max_bytes = tk.IntVar(value=DEFAULT_MAX_FILE_BYTES // 1024)
        self.max_depth = tk.IntVar(value=DEFAULT_MAX_DEPTH)
        self.max_tokens = tk.IntVar(value=self._default_token_ceiling())
        self.raise_budget = tk.BooleanVar(value=True)
        self.kind_vars = {kind: tk.BooleanVar(value=True)
                          for kind in ("code", "text", "docs")}

        self._build()
        self._recompute()

    def _default_token_ceiling(self) -> int:
        """Open on a setting that actually fits, not one that warns on sight.

        Defaulting to a fixed 40k while the model window is 32k means every
        folder shows a red warning before the user has touched anything, which
        trains people to ignore the warning — the opposite of the point.
        """
        room = max(4_000, int(self.max_window * 0.85))
        fitting = [value for value, _label in TOKEN_CHOICES if value <= room]
        return max(fitting) if fitting else TOKEN_CHOICES[0][0]

    # ------------------------------------------------------------------ view
    def _heading(self, text: str) -> None:
        tk.Label(self, text=text, font=FONT_BOLD, bg=PANEL, fg="#1f2430",
                 anchor="w").pack(fill="x", pady=(12, 2))

    def _build(self) -> None:
        tk.Label(self, text=f"📁 {self.survey.root}", font=FONT_BOLD, bg=PANEL,
                 fg="#1f2430", anchor="w", wraplength=520,
                 justify="left").pack(fill="x")
        tk.Label(self, text=self.survey.describe(), font=FONT_SMALL, bg=PANEL,
                 fg=MUTED, anchor="w").pack(fill="x", pady=(2, 0))

        notes = []
        if self.survey.skipped_dirs:
            unique = sorted(set(self.survey.skipped_dirs))
            shown = ", ".join(unique[:4])
            more = f" and {len(unique) - 4} more" if len(unique) > 4 else ""
            notes.append(f"Skipping build and version-control folders: {shown}{more}")
        if self.survey.unreadable:
            notes.append(f"{self.survey.unreadable:,} file(s) have no text to "
                         f"read (images, binaries) and are ignored.")
        if self.survey.truncated:
            notes.append("This folder is enormous — only the first "
                         "20,000 files were measured.")
        for note in notes:
            tk.Label(self, text=note, font=FONT_SMALL, bg=PANEL, fg=MUTED,
                     anchor="w", wraplength=520, justify="left").pack(fill="x")

        separator(self).pack(fill="x", pady=10)
        self._heading("How much of it should the models see?")

        tk.Radiobutton(
            self, text="Everything in this folder", variable=self.access,
            value="all", font=FONT, bg=PANEL, activebackground=PANEL,
            highlightthickness=0, bd=0, cursor="hand2", anchor="w",
            command=self._recompute).pack(fill="x")
        tk.Label(self, text="    Every readable file, however large the result.",
                 font=FONT_SMALL, bg=PANEL, fg=MUTED, anchor="w").pack(fill="x")

        tk.Radiobutton(
            self, text="Limit what gets read", variable=self.access,
            value="limited", font=FONT, bg=PANEL, activebackground=PANEL,
            highlightthickness=0, bd=0, cursor="hand2", anchor="w",
            command=self._recompute).pack(fill="x", pady=(6, 0))

        self.limit_box = tk.Frame(self, bg=SUBTLE, padx=12, pady=10,
                                  highlightbackground=BORDER, highlightthickness=1)
        self.limit_box.pack(fill="x", pady=(4, 0))
        self._build_limits(self.limit_box)

        separator(self).pack(fill="x", pady=12)

        self.summary_label = tk.Label(self, text="", font=FONT_BOLD, bg=PANEL,
                                      fg="#1f2430", anchor="w")
        self.summary_label.pack(fill="x")
        self.dropped_label = tk.Label(self, text="", font=FONT_SMALL, bg=PANEL,
                                      fg=MUTED, anchor="w", wraplength=520,
                                      justify="left")
        self.dropped_label.pack(fill="x")

        self.budget_label = tk.Label(self, text="", font=FONT_SMALL, bg=PANEL,
                                     fg=MUTED, anchor="w", wraplength=520,
                                     justify="left")
        self.budget_label.pack(fill="x", pady=(8, 0))
        self.budget_check = tk.Checkbutton(
            self, text="", variable=self.raise_budget, font=FONT_SMALL,
            bg=PANEL, activebackground=PANEL, highlightthickness=0, bd=0,
            cursor="hand2", anchor="w", command=self._sync_warning)
        self.budget_check.pack(fill="x")
        self.warning_label = tk.Label(self, text="", font=FONT_SMALL, bg=PANEL,
                                      fg=ERROR, anchor="w", wraplength=520,
                                      justify="left")
        self.warning_label.pack(fill="x")

        buttons = tk.Frame(self, bg=PANEL)
        buttons.pack(fill="x", pady=(14, 0))
        self.add_button = flat_button(buttons, "Add folder", self._accept,
                                      primary=True)
        self.add_button.config(padx=18, pady=6)
        self.add_button.pack(side="right")
        flat_button(buttons, "Cancel", self.destroy).pack(side="right", padx=(0, 8))

    def _build_limits(self, box: tk.Frame) -> None:
        def row(label: str) -> tk.Frame:
            line = tk.Frame(box, bg=SUBTLE)
            line.pack(fill="x", pady=3)
            tk.Label(line, text=label, font=FONT_SMALL, bg=SUBTLE, fg="#1f2430",
                     width=16, anchor="w").pack(side="left")
            return line

        line = row("File types")
        for kind, var in self.kind_vars.items():
            tk.Checkbutton(line, text=KIND_LABELS[kind], variable=var,
                           font=FONT_SMALL, bg=SUBTLE, activebackground=SUBTLE,
                           highlightthickness=0, bd=0, cursor="hand2",
                           command=self._recompute).pack(side="left", padx=(0, 10))

        line = row("Subfolders")
        for value, label in DEPTH_CHOICES:
            tk.Radiobutton(line, text=label, variable=self.max_depth,
                           value=value, font=FONT_SMALL, bg=SUBTLE,
                           activebackground=SUBTLE, highlightthickness=0, bd=0,
                           cursor="hand2", command=self._recompute
                           ).pack(side="left", padx=(0, 8))

        line = row("Most files")
        for value in FILE_CHOICES:
            tk.Radiobutton(line, text=f"{value:,}", variable=self.max_files,
                           value=value, font=FONT_SMALL, bg=SUBTLE,
                           activebackground=SUBTLE, highlightthickness=0, bd=0,
                           cursor="hand2", command=self._recompute
                           ).pack(side="left", padx=(0, 8))

        line = row("Skip files over")
        for value, label in SIZE_CHOICES:
            tk.Radiobutton(line, text=label, variable=self.max_bytes,
                           value=value, font=FONT_SMALL, bg=SUBTLE,
                           activebackground=SUBTLE, highlightthickness=0, bd=0,
                           cursor="hand2", command=self._recompute
                           ).pack(side="left", padx=(0, 8))

        line = row("Token ceiling")
        for value, label in TOKEN_CHOICES:
            tk.Radiobutton(line, text=label, variable=self.max_tokens,
                           value=value, font=FONT_SMALL, bg=SUBTLE,
                           activebackground=SUBTLE, highlightthickness=0, bd=0,
                           cursor="hand2", command=self._recompute
                           ).pack(side="left", padx=(0, 8))

    # ------------------------------------------------------------- behaviour
    def limits(self) -> Limits:
        if self.access.get() == "all":
            return Limits.everything()
        kinds = frozenset(kind for kind, var in self.kind_vars.items()
                          if var.get())
        return Limits(max_files=int(self.max_files.get()),
                      max_file_bytes=int(self.max_bytes.get()) * 1024,
                      max_tokens=int(self.max_tokens.get()),
                      max_depth=int(self.max_depth.get()),
                      kinds=kinds)

    def _recompute(self, *_args) -> None:
        limited = self.access.get() == "limited"
        for child in self.limit_box.winfo_children():
            for widget in child.winfo_children():
                try:
                    widget.config(state="normal" if limited else "disabled")
                except tk.TclError:
                    pass

        self.selection = select(self.survey, self.limits())
        self.plan = plan_budget(self.selection.tokens, self.context_budget,
                                ceiling=self.max_window)

        self.summary_label.config(text=self.selection.summary())
        detail = self.selection.dropped_detail()
        self.dropped_label.config(
            text=f"Left out: {detail}." if detail else
            ("Everything readable in this folder is included."
             if self.selection.chosen else ""))

        self.budget_label.config(
            text=self.plan.message(),
            fg=ERROR if self.plan.over_ceiling
            else (MUTED if self.plan.fits else "#1f2430"))
        self.budget_check.config(
            text=f"Raise the context budget to {self.plan.recommended:,} tokens",
            state="disabled" if self.plan.fits else "normal")
        self.add_button.config(
            state="normal" if self.selection.chosen else "disabled")
        self._sync_warning()

    def _sync_warning(self) -> None:
        if self.plan is None:
            return
        if self.plan.over_ceiling:
            self.warning_label.config(
                text="⚠ Narrow the selection — fewer files, shallower "
                     "subfolders or a lower token ceiling — or raise the "
                     "context-window cap in ⚙ Settings if your models can "
                     "take it.", fg=ERROR)
        elif self.plan.fits:
            self.warning_label.config(text="", fg=OK)
        elif not self.raise_budget.get():
            self.warning_label.config(
                text="⚠ Without raising it, the oldest messages — possibly "
                     "including this folder — will be trimmed away before the "
                     "model sees them.", fg=ERROR)
        else:
            self.warning_label.config(
                text=f"The models will be given a {self.plan.recommended:,}-"
                     f"token window to match. Smaller models may still "
                     f"truncate.", fg=MUTED)

    def _accept(self) -> None:
        if not self.selection.chosen:
            return
        budget = (self.plan.recommended
                  if self.raise_budget.get() and not self.plan.fits
                  else self.context_budget)
        limits = self.limits()
        self.destroy()
        self.on_accept(limits, budget)


def describe_choice(selection, plan) -> str:
    """One line for the status bar after the dialog closes."""
    text = f"{len(selection.chosen)} file(s), ≈{selection.tokens:,} tokens"
    if not plan.fits:
        text += f" · budget raised to {plan.recommended:,}"
    return text


__all__ = ["FolderScanDialog", "describe_choice", "human_size"]
