"""What the edit check decided, and why.

The run log answers "what did the models do". This answers the one question
that log cannot: why a reply full of confident description of changes produced
no diff to approve. Those failures are invisible by design — the pipeline
refuses quietly so it cannot destroy a file — and invisible failure is exactly
what makes people conclude the feature does not work.

Every row is one run of the check. The verdict column is the answer; the
detail pane below is the working. Copying the report out is the point of the
button, because the next thing anyone does with it is paste it somewhere.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from ..editdebug import TRACE_PATH, VERDICTS, EditDebug, shape_note
from .theme import (
    BORDER,
    CODE_BG,
    CODE_FG,
    ERROR,
    FONT,
    FONT_BOLD,
    FONT_MONO_SMALL,
    FONT_SMALL,
    FONT_TITLE,
    MUTED,
    OK,
    PANEL,
    WARN,
    flat_button,
)

COLUMNS = (
    ("at", "Time", 80),
    ("verdict", "Verdict", 140),
    ("what", "What happened", 380),
    ("shape", "Reply shape", 300),
)

# Green means a diff reached the user. Red means the model tried and the app
# refused. Amber means the model never produced anything to refuse — which
# looks like the app's fault and is not.
ROW_COLOURS = {
    "offered": OK,
    "auto-applied": OK,
    "all-rejected": ERROR,
    "prose-only": WARN,
    "prose-unmatched": WARN,
    "prose-offered": WARN,
    "no-change": MUTED,
    "no-folder": MUTED,
    "edits-off": MUTED,
    "no-target": MUTED,
}

# The fix, per verdict. A diagnosis nobody can act on is just a nicer way of
# saying it broke.
ADVICE = {
    "no-folder": "Attach a folder first — Attach folder, then ask again. "
                 "Without one there is nowhere for a change to go.",
    "edits-off": "Editing is switched off. Turn it back on in Settings.",
    "all-rejected": "The model produced edit blocks and every one was "
                    "refused — the reasons are in the detail below. Almost "
                    "always it quoted lines from memory instead of copying "
                    "them; the “Try again with the real file” button hands "
                    "the file back and usually fixes it.",
    "no-change": "The changes were already on disk. Either they were applied "
                 "earlier, or the model was shown the new version and "
                 "proposed it again.",
    "prose-offered": "The model described the changes instead of writing "
                     "them as edit blocks. Press “Make these changes” to ask "
                     "for each one properly.",
    "prose-unmatched": "The model described changes but named no file the "
                       "app could match — check the filenames it used "
                       "against the folder listing. A smaller model often "
                       "needs the folder re-attached so it can see the real "
                       "paths.",
    "no-target": "There was no model recorded to go back to. Send the "
                 "request again with a model selected.",
}


class EditDebugWindow(tk.Toplevel):
    def __init__(self, master, debug: EditDebug):
        super().__init__(master)
        self.debug = debug

        self.title("Why nothing was written")
        self.configure(bg=PANEL)
        self.geometry("1060x640")
        self.minsize(820, 460)
        self.transient(master)

        self._build()
        self.refresh()

    def _build(self) -> None:
        header = tk.Frame(self, bg=PANEL, padx=16, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="✏ Edit diagnostics", font=FONT_TITLE, bg=PANEL,
                 fg="#1f2430").pack(side="left")
        self.summary = tk.Label(header, text="", font=FONT_SMALL, bg=PANEL,
                                fg=MUTED)
        self.summary.pack(side="left", padx=12)

        flat_button(header, "Close", self.destroy).pack(side="right")
        flat_button(header, "Clear", self._clear).pack(side="right", padx=6)
        flat_button(header, "Copy report", self._copy).pack(side="right")

        body = tk.Frame(self, bg=PANEL, padx=16)
        body.pack(fill="both", expand=True)

        style = ttk.Style(self)
        style.configure("EditDebug.Treeview", font=FONT, rowheight=24,
                        background="white", fieldbackground="white")
        style.configure("EditDebug.Treeview.Heading", font=FONT_BOLD)

        self.tree = ttk.Treeview(body, columns=[c[0] for c in COLUMNS],
                                 show="headings", height=9,
                                 style="EditDebug.Treeview")
        for key, title, width in COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width,
                             anchor="center" if key == "at" else "w")
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for verdict, colour in ROW_COLOURS.items():
            self.tree.tag_configure(verdict, foreground=colour)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._show_detail())

        self.advice = tk.Label(self, text="", font=FONT_SMALL, bg=PANEL,
                               fg="#1f2430", anchor="w", padx=16,
                               wraplength=1010, justify="left")
        self.advice.pack(fill="x", pady=(8, 4))

        detail = tk.Frame(self, bg=PANEL, padx=16)
        detail.pack(fill="both", expand=True, pady=(0, 12))
        self.detail = tk.Text(detail, height=12, wrap="word", bd=0,
                              font=FONT_MONO_SMALL, bg=CODE_BG, fg=CODE_FG,
                              highlightthickness=1, highlightbackground=BORDER)
        self.detail.pack(fill="both", expand=True)
        self.detail.config(state="disabled")

        tk.Label(self, text=f"Also written to {TRACE_PATH}", font=FONT_SMALL,
                 bg=PANEL, fg=MUTED, anchor="w", padx=16).pack(
            fill="x", pady=(0, 10))

    # -- filling it in -----------------------------------------------------
    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for index, trace in reversed(list(enumerate(self.debug.traces))):
            self.tree.insert(
                "", "end", iid=str(index), tags=(trace.verdict,),
                values=(trace.at, trace.verdict,
                        VERDICTS.get(trace.verdict, trace.verdict),
                        shape_note(trace.reply_shape) or "—"))
        self.summary.config(text=self.debug.summary())
        rows = self.tree.get_children()
        if rows:
            self.tree.selection_set(rows[0])
        else:
            self._write("The edit check has not run yet this session.")
            self.advice.config(text="")

    def _selected(self):
        picked = self.tree.selection()
        if not picked:
            return None
        try:
            return self.debug.traces[int(picked[0])]
        except (ValueError, IndexError):
            return None

    def _show_detail(self) -> None:
        trace = self._selected()
        if trace is None:
            return
        self.advice.config(text=ADVICE.get(trace.verdict, ""))
        text = trace.explain()
        if trace.reply_head:
            text += ("\n\n--- the reply it was reading "
                     "-------------------------------\n" + trace.reply_head)
        self._write(text)

    def _write(self, text: str) -> None:
        self.detail.config(state="normal")
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", text)
        self.detail.config(state="disabled")

    # -- actions -----------------------------------------------------------
    def _copy(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.debug.report())
        messagebox.showinfo("Copied",
                            "The full report is on the clipboard.")

    def _clear(self) -> None:
        self.debug.clear()
        self.refresh()
