"""A window for reviewing what Learning mode has remembered.

A knowledge base the user can't inspect is a black box that quietly shapes
every answer, so everything it holds is listed here, filterable, and
deletable one row at a time.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

from ..knowledge import KnowledgeBase
from .theme import FONT, FONT_BOLD, FONT_SMALL, FONT_TITLE, MUTED, PANEL, flat_button

COLUMNS = (
    ("topic", "Topic", 130),
    ("text", "Lesson", 520),
    ("uses", "Used", 55),
    ("created", "Learned", 110),
)


class KnowledgeWindow(tk.Toplevel):
    def __init__(self, master, knowledge: KnowledgeBase,
                 on_change: Callable[[], None] | None = None):
        super().__init__(master)
        self.knowledge = knowledge
        self.on_change = on_change or (lambda: None)

        self.title("What the models have learned")
        self.configure(bg=PANEL)
        self.geometry("880x520")
        self.minsize(640, 380)
        self.transient(master)

        self._build()
        self.refresh()

    def _build(self) -> None:
        header = tk.Frame(self, bg=PANEL, padx=16, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="🧠 Learned knowledge", font=FONT_TITLE,
                 bg=PANEL, fg="#1f2430").pack(side="left")
        self.summary = tk.Label(header, text="", font=FONT_SMALL, bg=PANEL,
                                fg=MUTED)
        self.summary.pack(side="left", padx=10)

        controls = tk.Frame(self, bg=PANEL, padx=16)
        controls.pack(fill="x")
        tk.Label(controls, text="Filter:", font=FONT, bg=PANEL,
                 fg="#1f2430").pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self.refresh())
        tk.Entry(controls, textvariable=self.filter_var, font=FONT, width=28,
                 relief="solid", bd=1).pack(side="left", padx=(6, 16))

        flat_button(controls, "🗑 Forget selected", self._forget).pack(side="left")
        flat_button(controls, "Clear all", self._clear).pack(side="left", padx=6)

        table = tk.Frame(self, bg=PANEL, padx=16, pady=10)
        table.pack(fill="both", expand=True)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Know.Treeview", font=FONT, rowheight=24,
                        fieldbackground="white")
        style.configure("Know.Treeview.Heading", font=FONT_BOLD)

        self.tree = ttk.Treeview(table, columns=[c[0] for c in COLUMNS],
                                 show="headings", style="Know.Treeview")
        for key, title, width in COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width,
                             anchor="e" if key == "uses" else "w")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        tk.Label(self, text=f"Stored in {self.knowledge.path}", font=FONT_SMALL,
                 bg=PANEL, fg=MUTED, anchor="w", padx=16, pady=6).pack(fill="x")

    # -- data --------------------------------------------------------------
    def refresh(self) -> None:
        needle = self.filter_var.get().strip().lower()
        self.tree.delete(*self.tree.get_children())

        shown = 0
        for lesson in reversed(self.knowledge.lessons):
            if needle and needle not in lesson.haystack.lower():
                continue
            self.tree.insert("", "end", iid=lesson.id, values=(
                lesson.topic,
                lesson.text,
                lesson.uses or "",
                lesson.created_at[:10],
            ))
            shown += 1

        total = len(self.knowledge)
        topics = len(self.knowledge.topics())
        suffix = f" ({shown} shown)" if needle else ""
        self.summary.config(
            text=f"{total} lesson(s) across {topics} topic(s){suffix}")

    # -- actions -----------------------------------------------------------
    def _forget(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Nothing selected",
                                "Pick a lesson to forget first.", parent=self)
            return
        for lesson_id in selection:
            self.knowledge.remove(lesson_id)
        self.refresh()
        self.on_change()

    def _clear(self) -> None:
        if not len(self.knowledge):
            return
        if messagebox.askyesno(
                "Forget everything?",
                f"This permanently deletes all {len(self.knowledge)} learned "
                f"lesson(s). Continue?", parent=self):
            self.knowledge.clear()
            self.refresh()
            self.on_change()
