"""The window that stands between a model's suggestion and your files.

This exists because the rest of the app is read-only and this part is not.
Everything about it is arranged so that the default outcome of confusion,
impatience or a mis-click is *nothing written*: Cancel is the default, files
are listed with their diffs rather than their names alone, and a file has to
be ticked to be written.

The "don't ask again" option is deliberately worded as being about this chat.
A permanent setting for "write to my disk without asking" is not something
that should be one checkbox away, and a per-chat scope means the worst case
of forgetting it is bounded by starting a new chat.

"Apply all" is a different promise from that checkbox and the two are kept
visibly apart.  It applies the changes on this screen — the ones whose diffs
are right there to be scrolled through — and nothing else.  The checkbox gives
away the next decision and the one after that, sight unseen.  Wanting the
first is the normal case; wanting the second should take a moment's thought.
"""

from __future__ import annotations

import tkinter as tk

from .. import edits as edit_tools
from .theme import (
    BORDER,
    CHAT_BG,
    CODE_BG,
    CODE_FG,
    ERROR,
    FONT_BOLD,
    FONT_MONO_SMALL,
    FONT_SMALL,
    MUTED,
    OK,
    PANEL,
    flat_button,
)

ADDED = "#1f7a3d"
REMOVED = "#b02a2a"


class EditReview(tk.Toplevel):
    """Show the diffs, let the user choose, write only on Apply."""

    def __init__(self, master, root, proposed, rejected=None,
                 on_apply=None, allow_always: bool = True,
                 flaws=None) -> None:
        super().__init__(master)
        self.title("Review changes")
        self.configure(bg=PANEL)
        self.geometry("900x680")
        self.transient(master)

        self.root_path = root
        self.proposed = list(proposed)
        # {filename: [Problem]} — what the change would break. Shown above the
        # diff rather than beside it, because a diff you have already read
        # looks fine and the warning has to arrive first.
        self.flaws = dict(flaws or {})
        self.on_apply = on_apply or (lambda _chosen, _always: None)
        self.always = tk.BooleanVar(value=False)
        self.picks: dict = {}

        head = tk.Frame(self, bg=PANEL)
        head.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(head, text=f"{edit_tools.summarise(root, self.proposed)}",
                 font=FONT_BOLD, bg=PANEL, fg="#1f2430").pack(side="left")
        tk.Label(head, text=f"  in {root}", font=FONT_SMALL, bg=PANEL,
                 fg=MUTED).pack(side="left")

        for item in rejected or []:
            tk.Label(self, text=f"⚠ {item.name} — {item.reason}",
                     font=FONT_SMALL, bg=PANEL, fg=ERROR, justify="left",
                     wraplength=860, anchor="w").pack(fill="x", padx=14)

        body = tk.Frame(self, bg=PANEL, highlightbackground=BORDER,
                        highlightthickness=1)
        body.pack(fill="both", expand=True, padx=14, pady=8)

        canvas = tk.Canvas(body, bg=CHAT_BG, highlightthickness=0)
        scroll = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=CHAT_BG)
        inner.bind("<Configure>", lambda _e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for edit in self.proposed:
            self._add_file(inner, edit)

        feet = tk.Frame(self, bg=PANEL)
        feet.pack(fill="x", padx=14, pady=(0, 12))
        if allow_always:
            tk.Checkbutton(
                feet,
                text="Apply future changes in this chat without asking me",
                variable=self.always, font=FONT_SMALL, bg=PANEL, fg=MUTED,
                activebackground=PANEL, highlightthickness=0, bd=0,
                cursor="hand2").pack(side="left")

        flat_button(feet, "Cancel", self._cancel).pack(side="right")
        flat_button(feet, "Apply selected", self._apply,
                    primary=True).pack(side="right", padx=(0, 8))
        # Ticking six boxes to say yes to six changes you have just read is
        # busywork, and busywork is what pushes people towards the checkbox
        # that stops asking altogether.  This is the safer shortcut: it still
        # only covers what is on this screen, and it will not sweep up a file
        # that has been shown not to run — that one has to be ticked by hand,
        # which is the whole point of knowing about it.
        runnable = self._runnable()
        label = f"Apply all {len(runnable)}"
        if len(runnable) != len(self.proposed):
            label += " that run"
        flat_button(feet, label, self._apply_all).pack(side="right",
                                                       padx=(0, 8))

        # Cancel is the default outcome of every accident: Escape, the window
        # close button, or simply walking away.
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _e: self._cancel())
        self.result: list = []
        self.grab_set()

    def _add_file(self, parent, edit) -> None:
        added, removed = edit_tools.counts(self.root_path, edit)
        exists = bool(edit_tools.read_current(self.root_path, edit.name))
        # A file that would also invent the folder around it does not arrive
        # ticked.  The diff of a brand-new file is all additions and every
        # line of it reads fine, so nothing in the part you actually look at
        # says "this belongs to a different project" — only the path says
        # that, and a pre-ticked box is easy to scroll straight past.
        odd = edit_tools.novelty(self.root_path, edit)
        # A change that stops the file working does not arrive ticked either.
        # This is the failure the diff cannot show: every line of it is
        # plausible, and the file only breaks when something calls it.
        broken = self.flaws.get(edit.name) or []
        chosen = tk.BooleanVar(value=not (odd or broken))
        self.picks[edit.name] = chosen

        row = tk.Frame(parent, bg=CHAT_BG)
        row.pack(fill="x", padx=8, pady=(10, 2))
        tk.Checkbutton(row, variable=chosen, bg=CHAT_BG, activebackground=CHAT_BG,
                       highlightthickness=0, bd=0, cursor="hand2").pack(side="left")
        tk.Label(row, text=edit.name, font=FONT_BOLD, bg=CHAT_BG,
                 fg="#1f2430").pack(side="left")
        tk.Label(row, text=f"  +{added}", font=FONT_SMALL, bg=CHAT_BG,
                 fg=ADDED).pack(side="left")
        tk.Label(row, text=f"−{removed}", font=FONT_SMALL, bg=CHAT_BG,
                 fg=REMOVED).pack(side="left", padx=(4, 0))
        if not exists:
            tk.Label(row, text="  new file", font=FONT_SMALL, bg=CHAT_BG,
                     fg=OK).pack(side="left")
        if broken:
            tk.Label(row, text="  will not run", font=FONT_SMALL, bg=CHAT_BG,
                     fg=ERROR).pack(side="left")
        for problem in broken:
            tk.Label(parent, text=f"✖ {problem.describe()}",
                     font=FONT_BOLD, bg=CHAT_BG, fg=ERROR, justify="left",
                     wraplength=820, anchor="w").pack(fill="x", padx=(30, 8))
        if odd:
            tk.Label(parent, text=f"⚠ {odd}", font=FONT_SMALL, bg=CHAT_BG,
                     fg=ERROR, justify="left", wraplength=820,
                     anchor="w").pack(fill="x", padx=(30, 8))

        diff = edit_tools.diff_for(self.root_path, edit)
        lines = diff.splitlines()
        shown = lines[:400]
        view = tk.Text(parent, height=min(24, max(3, len(shown))),
                       wrap="none", font=FONT_MONO_SMALL, bg=CODE_BG,
                       fg=CODE_FG, highlightthickness=1,
                       highlightbackground=BORDER, bd=0)
        view.pack(fill="x", padx=8, pady=(0, 4))
        view.tag_config("add", foreground=ADDED)
        view.tag_config("del", foreground=REMOVED)
        view.tag_config("meta", foreground=MUTED)
        for line in shown:
            tag = ("add" if line.startswith("+") else
                   "del" if line.startswith("-") else
                   "meta" if line.startswith(("@@", "#", "diff")) else "")
            view.insert(tk.END, line + "\n", tag)
        if len(lines) > len(shown):
            view.insert(tk.END, f"…{len(lines) - len(shown)} more lines\n",
                        "meta")
        view.config(state="disabled")

    def _apply(self) -> None:
        self._finish([e for e in self.proposed if self.picks[e.name].get()])

    def _runnable(self) -> list:
        """The changes that do not break the file they land in."""
        return [e for e in self.proposed if not self.flaws.get(e.name)]

    def _apply_all(self) -> None:
        """Every change on this screen that still runs.

        Files the app merely finds *surprising* are included: a genuinely new
        file in a genuinely new folder is the ordinary case for overriding
        that, and the warning is on screen while the button is pressed. A file
        that provably will not run is different in kind — that is not a
        judgement call the user is overriding, it is a fact — so it is left
        for a deliberate tick rather than swept along with the rest.
        """
        self._finish(self._runnable())

    def _finish(self, chosen) -> None:
        self.result = chosen
        always = bool(self.always.get())
        self.grab_release()
        self.destroy()
        self.on_apply(self.result, always)

    def _cancel(self) -> None:
        self.result = []
        self.grab_release()
        self.destroy()
        self.on_apply([], False)
