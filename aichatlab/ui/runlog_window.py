"""Every request this session made, and what became of it.

The transcript answers "what did the models say". This answers the questions
you only ask when something has gone wrong: which call took four minutes, how
many replies were cut off at the same cap, and whether the time went on
answering or on summarising.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..activity import human_duration
from ..runlog import RunLog, write_csv
from .theme import (
    ERROR,
    FONT,
    FONT_BOLD,
    FONT_SMALL,
    FONT_TITLE,
    MUTED,
    PANEL,
    WARN,
    flat_button,
)

COLUMNS = (
    ("at", "Time", 80),
    ("model", "Model", 170),
    ("server", "Where", 65),
    ("purpose", "For", 95),
    ("duration", "Took", 75),
    ("prompt", "Prompt", 80),
    ("reply", "Reply", 75),
    ("rate", "Tok/s", 60),
    ("outcome", "Outcome", 240),
)

ROW_COLOURS = {"error": ERROR, "length": WARN, "cancelled": MUTED}


class RunLogWindow(tk.Toplevel):
    def __init__(self, master, log: RunLog):
        super().__init__(master)
        self.log = log

        self.title("Run log")
        self.configure(bg=PANEL)
        self.geometry("980x520")
        self.minsize(720, 360)
        self.transient(master)

        self._build()
        self.refresh()

    def _build(self) -> None:
        header = tk.Frame(self, bg=PANEL, padx=16, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="📜 Run log", font=FONT_TITLE, bg=PANEL,
                 fg="#1f2430").pack(side="left")
        self.summary = tk.Label(header, text="", font=FONT_SMALL, bg=PANEL,
                                fg=MUTED)
        self.summary.pack(side="left", padx=12)

        flat_button(header, "Close", self.destroy).pack(side="right")
        flat_button(header, "Clear", self._clear).pack(side="right", padx=6)
        flat_button(header, "Export CSV", self._export).pack(side="right")

        body = tk.Frame(self, bg=PANEL, padx=16)
        body.pack(fill="both", expand=True, pady=(0, 12))

        style = ttk.Style(self)
        style.configure("Runlog.Treeview", font=FONT, rowheight=24,
                        background="white", fieldbackground="white")
        style.configure("Runlog.Treeview.Heading", font=FONT_BOLD)

        self.tree = ttk.Treeview(body, columns=[c[0] for c in COLUMNS],
                                 show="headings", style="Runlog.Treeview")
        for key, title, width in COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width,
                             anchor="w" if key in ("model", "outcome") else "center")
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for outcome, colour in ROW_COLOURS.items():
            self.tree.tag_configure(outcome, foreground=colour)

        self.hint = tk.Label(self, text="", font=FONT_SMALL, bg=PANEL,
                             fg=MUTED, anchor="w", padx=16, wraplength=940,
                             justify="left")
        self.hint.pack(fill="x", pady=(0, 10))

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for entry in reversed(self.log.entries):        # newest at the top
            self.tree.insert("", "end", values=entry.row(),
                             tags=(entry.outcome,))
        self.summary.config(text=self.log.summary())
        self.hint.config(text=self._hint())

    def _hint(self) -> str:
        """Say the useful thing rather than making them read the table."""
        if not self.log.entries:
            return ""
        notes = []
        cut_off = self.log.counts().get("length", 0)
        if cut_off >= 2:
            notes.append(
                f"{cut_off} replies were cut off at the token cap — the "
                f"speed slider sets that limit, so move it towards Quality "
                f"if answers keep stopping mid-sentence.")
        slowest = self.log.slowest(1)
        if slowest and slowest[0].duration_s > 60:
            entry = slowest[0]
            notes.append(f"Longest single request: {entry.model} "
                         f"({entry.purpose}) at "
                         f"{human_duration(entry.duration_s)}.")
        return "  ".join(notes)

    def _clear(self) -> None:
        self.log.clear()
        self.refresh()

    def _export(self) -> None:
        if not self.log.entries:
            messagebox.showinfo("Nothing to export",
                                "No requests have run yet this session.")
            return
        path = filedialog.asksaveasfilename(
            title="Export run log", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            write_csv(path, self.log.entries)
        except OSError as exc:
            messagebox.showerror("Error", f"Could not write the file:\n{exc}")
