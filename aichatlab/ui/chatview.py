"""The chat transcript: bubbles, live streaming, and copy affordances.

Streaming is done with Tk text marks.  Each in-flight reply owns a mark that
sits at the end of its bubble; tokens are inserted at that mark, which then
slides along.  Because every message has its own mark, several models can
stream into the transcript at the same time without their words interleaving.
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import scrolledtext

from ..formatting import has_markdown, render_markdown, tidy
from ..thinking import human_duration
from .theme import (
    ACCENT,
    Tooltip,
    ACCENT_DARK,
    BOT_BG,
    BOT_FG,
    CHAT_BG,
    CODE_BG,
    CODE_FG,
    ERROR,
    FONT_CHAT,
    FONT_CHAT_BOLD,
    FONT_CHAT_HEADING,
    FONT_CHAT_ITALIC,
    FONT_MONO_SMALL,
    FONT_SMALL,
    MUTED,
    OK,
    SELECTION,
    USER_BG,
    USER_FG,
    flat_button,
)

TYPING_FRAMES = ("•  ", "•• ", "•••", " ••", "  •")


class _Holder:
    """Mutable text box for a copy chip whose message is still streaming."""

    def __init__(self, text: str = "") -> None:
        self.text = text


class ChatView(tk.Frame):
    def __init__(self, master, on_status=None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_status = on_status or (lambda _msg: None)
        self._streams: dict[int, dict] = {}
        # Reasoning blocks outlive their stream: the block stays on screen,
        # collapsed and clickable, long after the reply has finished.
        self._thoughts: dict[int, dict] = {}
        self._notes: dict[str, tuple[str, str]] = {}
        self._actions: dict[str, tuple[str, str]] = {}
        self._last_speaker: str | None = None
        self._replay_ids = -1

        self.text = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, state="disabled", font=FONT_CHAT,
            bg=CHAT_BG, fg=BOT_FG, relief="flat", padx=16, pady=14,
            cursor="xterm", insertbackground=BOT_FG,
            selectbackground=SELECTION, selectforeground=BOT_FG,
            # keep the highlight visible after focus moves to a copy button
            inactiveselectbackground=SELECTION,
            takefocus=True)
        self.text.pack(fill="both", expand=True)

        self._configure_tags()
        self._bind_copy_handlers()

    # -- appearance --------------------------------------------------------
    def _configure_tags(self) -> None:
        tag = self.text.tag_configure
        tag("user_name", font=FONT_SMALL, foreground=MUTED, justify="right",
            rmargin=14, spacing1=6)
        tag("user_msg", font=FONT_CHAT, foreground=USER_FG, background=USER_BG,
            justify="right", lmargin1=160, lmargin2=160, rmargin=12,
            spacing1=2, spacing3=2)
        tag("bot_name", font=FONT_SMALL, foreground=ACCENT_DARK, lmargin1=12,
            spacing1=6)
        tag("bot_msg", font=FONT_CHAT, foreground=BOT_FG, background=BOT_BG,
            lmargin1=12, lmargin2=12, rmargin=160, spacing1=2, spacing3=2)
        tag("meta", font=FONT_SMALL, foreground=MUTED, lmargin1=12)
        tag("sys_msg", font=FONT_SMALL, foreground=MUTED, justify="center",
            spacing1=4, spacing3=4)
        # A checklist is a list, so it is left-aligned and indented rather
        # than centred like the one-line notes.
        tag("checklist", font=FONT_MONO_SMALL, foreground=BOT_FG,
            background=CODE_BG, lmargin1=26, lmargin2=40, rmargin=26,
            spacing1=3, spacing3=3)
        tag("sys_err", font=FONT_SMALL, foreground=ERROR, justify="center",
            spacing1=4, spacing3=4)

        # Markdown styling.  These are configured after the bubble tags so
        # their fonts win where the two disagree.
        tag("md_bold", font=FONT_CHAT_BOLD)
        tag("md_italic", font=FONT_CHAT_ITALIC)
        tag("md_code", font=FONT_MONO_SMALL, background=CODE_BG,
            foreground=CODE_FG)
        tag("md_code_block", font=FONT_MONO_SMALL, background=CODE_BG,
            foreground=CODE_FG, lmargin1=26, lmargin2=26, rmargin=160)
        tag("md_heading", font=FONT_CHAT_HEADING, spacing1=6, spacing3=2)
        tag("md_bullet", lmargin1=26, lmargin2=40)
        tag("md_link", foreground=ACCENT_DARK, underline=True)
        tag("md_muted", foreground=MUTED)
        tag("md_typing", foreground=MUTED, font=FONT_CHAT)

        # Tk gives later-created tags higher priority than the built-in "sel"
        # tag, so a bubble's own background paints straight over the selection
        # highlight — the text looked unselectable when it was merely
        # invisible.  Raise "sel" above everything and colour it explicitly,
        # because the widget-level -selectbackground loses to a raised tag.
        tag("sel", background=SELECTION, foreground=BOT_FG)
        self.text.tag_raise("sel")

    def _bind_copy_handlers(self) -> None:
        # A disabled Text never takes focus by itself, so Ctrl+C would be sent
        # to whatever had focus instead.  Claim it on click.
        self.text.bind("<Button-1>", lambda e: self.text.focus_set(), add="+")
        for sequence in ("<Control-c>", "<Control-C>"):
            self.text.bind(sequence, self.copy_selection)
        for sequence in ("<Control-a>", "<Control-A>"):
            self.text.bind(sequence, self.select_all)
        self.menu = tk.Menu(self, tearoff=0, font=FONT_SMALL)
        self.menu.add_command(label="Copy selection", command=self.copy_selection)
        self.menu.add_command(label="Select all", command=self.select_all)
        self.menu.add_separator()
        self.menu.add_command(label="Copy entire conversation",
                              command=self.copy_all)
        self.text.bind("<Button-3>", self._popup)
        self.text.bind("<Button-2>", self._popup)

    def _popup(self, event):
        self.text.focus_set()
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
        return "break"

    # -- clipboard ---------------------------------------------------------
    def copy(self, text: str, note: str = "Copied to clipboard ✓") -> None:
        if not text:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update_idletasks()
            self.on_status(note)
        except tk.TclError as exc:
            self.on_status(f"Could not copy: {exc}")

    def copy_selection(self, event=None):
        try:
            selected = self.text.get("sel.first", "sel.last")
        except tk.TclError:
            self.on_status("Nothing selected — drag over some text first")
            return "break"
        self.copy(selected, "Copied selection ✓")
        return "break"

    def select_all(self, event=None):
        self.text.tag_add("sel", "1.0", "end-1c")
        self.text.focus_set()
        return "break"

    def copy_all(self):
        self.copy(self.text.get("1.0", "end-1c"),
                  "Copied the whole conversation ✓")

    def transcript(self) -> str:
        return self.text.get("1.0", "end-1c")

    # -- internals ---------------------------------------------------------
    def _at_bottom(self) -> bool:
        try:
            return self.text.yview()[1] > 0.999
        except tk.TclError:
            return True

    def _autoscroll(self, was_at_bottom: bool) -> None:
        """Only follow the conversation if the user had not scrolled away."""
        if was_at_bottom:
            self.text.see(tk.END)

    def _chip(self, holder: _Holder) -> tk.Label:
        chip = tk.Label(self.text, text="📋", font=FONT_SMALL, bg=CHAT_BG,
                        fg=MUTED, cursor="hand2", padx=3)

        def restore():
            try:
                chip.config(text="📋", fg=MUTED)
            except tk.TclError:
                pass

        def do_copy(_event=None):
            self.copy(holder.text, "Copied message ✓")
            try:
                chip.config(text="✓ copied", fg=OK)
            except tk.TclError:
                return
            chip.after(1300, restore)

        chip.bind("<Button-1>", do_copy)
        chip.bind("<Enter>",
                  lambda e: chip.config(fg=ACCENT) if chip["text"] == "📋" else None)
        chip.bind("<Leave>",
                  lambda e: chip.config(fg=MUTED) if chip["text"] == "📋" else None)
        return chip

    def _insert_chip(self, holder: _Holder, tag: str) -> None:
        start = self.text.index("end-1c")
        self.text.window_create(tk.END, window=self._chip(holder))
        self.text.tag_add(tag, start, "end-1c")

    def _now(self) -> str:
        return datetime.now().strftime("%H:%M")

    # -- public writing API ------------------------------------------------
    def clear(self) -> None:
        self._streams.clear()
        self._thoughts.clear()
        self._notes.clear()
        self._actions.clear()
        self._last_speaker = None
        self.text.config(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.config(state="disabled")

    def add_user(self, text: str, footer: str = "") -> None:
        bottom = self._at_bottom()
        self._last_speaker = None          # a new question breaks any grouping
        self.text.config(state="normal")
        self.text.insert(tk.END, f"You  {self._now()} ", "user_name")
        self._insert_chip(_Holder(text), "user_name")
        self.text.insert(tk.END, "\n", "user_name")
        body = text + (f"\n{footer}" if footer else "")
        self.text.insert(tk.END, f" {body} \n", "user_msg")
        self.text.insert(tk.END, "\n")
        self.text.config(state="disabled")
        self._autoscroll(bottom)

    def add_note(self, text: str, note_id: str | None = None,
                 tag: str = "sys_msg") -> None:
        """A quiet centred line.  Passing the same note_id updates in place,
        which keeps multi-step progress (searching, reading, done) to one line
        instead of stacking four."""
        bottom = self._at_bottom()
        self.text.config(state="normal")

        if note_id and note_id in self._notes:
            start, end = self._notes[note_id]
            self.text.delete(start, end)
            self.text.insert(start, text, tag)
        else:
            start_anchor = self.text.index("end-1c")
            self.text.insert(tk.END, text, tag)
            end_anchor = self.text.index("end-1c")
            self.text.insert(tk.END, "\n\n", "sys_msg")
            if note_id:
                start_mark, end_mark = f"note{note_id}a", f"note{note_id}b"
                self.text.mark_set(start_mark, start_anchor)
                self.text.mark_gravity(start_mark, "left")
                self.text.mark_set(end_mark, end_anchor)
                self.text.mark_gravity(end_mark, "right")
                self._notes[note_id] = (start_mark, end_mark)

        self.text.config(state="disabled")
        self._autoscroll(bottom)

    def add_checklist(self, text: str, list_id: str) -> None:
        """A multi-line checklist that redraws in place as steps tick off."""
        self.add_note(text, note_id=f"list{list_id}", tag="checklist")

    def close_checklist(self, list_id: str) -> None:
        self.close_note(f"list{list_id}")

    def close_note(self, note_id: str) -> None:
        """Stop updating a note, so the next one starts a fresh line."""
        self._notes.pop(note_id, None)

    def add_action(self, text: str, options, action_id: str) -> str:
        """A note the user can answer — real buttons, embedded in the transcript.

        Used where a modal would be wrong: the app has noticed something about
        the message you just wrote and needs a decision, but interrupting with
        a dialog box over the conversation is heavier than the question
        deserves.  Answering replaces the whole block with a one-line record of
        what you chose, so the transcript still reads as a history afterwards.
        """
        bottom = self._at_bottom()
        self.text.config(state="normal")

        start_anchor = self.text.index("end-1c")
        self.text.insert(tk.END, f"{text}\n", "sys_msg")

        row = tk.Frame(self.text, bg=CHAT_BG)
        for option in options:
            # Three or four elements: (label, callback, primary[, tooltip]).
            # The tooltip is for buttons whose label is necessarily terse —
            # a menu numbered 1..6 says nothing about what 4 *is*, and the
            # full description belongs under the pointer, not in the label.
            label, callback, primary = option[0], option[1], option[2]
            tip = option[3] if len(option) > 3 else ""
            # Guard against a double-click firing the action twice: the first
            # click resolves the block, so the second finds nothing to do.
            def fire(_event=None, callback=callback):
                if action_id in self._actions:
                    callback()
            button = flat_button(row, label, fire, primary=primary)
            button.config(font=FONT_SMALL, padx=10, pady=3)
            button.pack(side="left", padx=4)
            if tip:
                Tooltip(button, tip)
        self.text.window_create(tk.END, window=row)

        self.text.tag_add("sys_msg", start_anchor, "end-1c")
        end_anchor = self.text.index("end-1c")
        self.text.insert(tk.END, "\n\n", "sys_msg")

        start_mark, end_mark = f"act{action_id}a", f"act{action_id}b"
        self.text.mark_set(start_mark, start_anchor)
        self.text.mark_gravity(start_mark, "left")
        self.text.mark_set(end_mark, end_anchor)
        self.text.mark_gravity(end_mark, "right")
        self._actions[action_id] = (start_mark, end_mark)

        self.text.config(state="disabled")
        self._autoscroll(bottom)
        return action_id

    def resolve_action(self, action_id: str, text: str = "") -> None:
        """Answer a pending action, replacing its buttons with the outcome."""
        marks = self._actions.pop(action_id, None)
        if marks is None:
            return
        start_mark, end_mark = marks
        bottom = self._at_bottom()
        self.text.config(state="normal")
        try:
            self.text.delete(start_mark, end_mark)
            if text:
                self.text.insert(start_mark, text, "sys_msg")
        except tk.TclError:
            pass
        for mark in marks:
            try:
                self.text.mark_unset(mark)
            except tk.TclError:
                pass
        self.text.config(state="disabled")
        self._autoscroll(bottom)

    def has_action(self, action_id: str) -> bool:
        return action_id in self._actions

    def add_error(self, text: str) -> None:
        bottom = self._at_bottom()
        self.text.config(state="normal")
        self.text.insert(tk.END, "⚠  ", "sys_err")
        self._insert_chip(_Holder(text), "sys_err")
        self.text.insert(tk.END, f"  {text}\n\n", "sys_err")
        self.text.config(state="disabled")
        self._autoscroll(bottom)

    def add_assistant(self, heading: str, text: str) -> None:
        """Add a finished reply in one go (used when redrawing a loaded chat)."""
        stream_id = self._replay_ids
        self._replay_ids -= 1
        self.start_stream(stream_id, heading)
        self.append_stream(stream_id, text)
        self.end_stream(stream_id, text)

    # -- streaming ---------------------------------------------------------
    def start_stream(self, stream_id: int, heading: str,
                     speaker: str | None = None) -> int:
        bottom = self._at_bottom()
        holder = _Holder()
        self.text.config(state="normal")

        # Consecutive replies from the same model read as one person talking,
        # so only print the header when the speaker actually changes.
        speaker = speaker or heading
        grouped = speaker == self._last_speaker
        head_anchor = None

        if grouped:
            self.text.insert(tk.END, " ", "bot_name")
            self._insert_chip(holder, "bot_name")
            head_anchor = self.text.index("end-1c")
            self.text.insert(tk.END, "\n", "bot_name")
        else:
            # Anchors are captured *before* the padding is inserted and the
            # marks set *afterwards*.  A right-gravity mark set first would be
            # pushed to the end of the widget — which is how two models
            # streaming at once ended up interleaved in a single bubble.
            self.text.insert(tk.END, f"{heading} ", "bot_name")
            self._insert_chip(holder, "bot_name")
            head_anchor = self.text.index("end-1c")
            self.text.insert(tk.END, "\n", "bot_name")
        self._last_speaker = speaker

        # A reasoning block, when there is one, goes here — above the answer
        # and outside the start..body range that `end_stream` redraws, so a
        # markdown re-render cannot wipe it.
        thought_anchor = self.text.index("end-1c")

        self.text.insert(tk.END, " ", "bot_msg")
        body_anchor = self.text.index("end-1c")
        self.text.insert(tk.END, " \n\n")

        head_mark, body_mark = f"s{stream_id}head", f"s{stream_id}body"
        start_mark = f"s{stream_id}start"
        thought_mark = f"s{stream_id}think"
        self.text.mark_set(thought_mark, thought_anchor)
        self.text.mark_gravity(thought_mark, "right")
        for mark, anchor in ((head_mark, head_anchor), (body_mark, body_anchor)):
            self.text.mark_set(mark, anchor)
            self.text.mark_gravity(mark, "right")
        self.text.mark_set(start_mark, body_anchor)
        self.text.mark_gravity(start_mark, "left")

        self.text.config(state="disabled")
        tail_mark = f"s{stream_id}tail"
        self.text.mark_set(tail_mark, body_anchor)
        self.text.mark_gravity(tail_mark, "left")

        self._streams[stream_id] = {
            "holder": holder, "body": body_mark, "head": head_mark,
            "start": start_mark, "tail": tail_mark, "chunks": [],
            "pending": "", "typing": False, "thought": thought_mark,
        }
        self._start_typing(stream_id)
        self._autoscroll(bottom)
        return stream_id

    # -- typing indicator --------------------------------------------------
    def _start_typing(self, stream_id: int) -> None:
        stream = self._streams.get(stream_id)
        if not stream:
            return
        self.text.config(state="normal")
        first = self.text.index(stream["body"])
        self.text.insert(stream["body"], TYPING_FRAMES[0], "md_typing")
        last = self.text.index(stream["body"])
        self.text.config(state="disabled")

        marks = (f"s{stream_id}typea", f"s{stream_id}typeb")
        self.text.mark_set(marks[0], first)
        self.text.mark_gravity(marks[0], "left")
        self.text.mark_set(marks[1], last)
        self.text.mark_gravity(marks[1], "right")
        stream.update({"typing": True, "type_marks": marks, "frame": 0})
        self._typing_tick(stream_id)

    def _typing_tick(self, stream_id: int) -> None:
        stream = self._streams.get(stream_id)
        if not stream or not stream.get("typing"):
            return
        stream["frame"] = (stream["frame"] + 1) % len(TYPING_FRAMES)
        start, end = stream["type_marks"]
        try:
            self.text.config(state="normal")
            self.text.delete(start, end)
            self.text.insert(start, TYPING_FRAMES[stream["frame"]], "md_typing")
            self.text.config(state="disabled")
        except tk.TclError:
            return
        self.after(360, lambda: self._typing_tick(stream_id))

    def _stop_typing(self, stream: dict) -> None:
        if not stream.get("typing"):
            return
        stream["typing"] = False
        start, end = stream.get("type_marks", (None, None))
        try:
            self.text.config(state="normal")
            if start:
                self.text.delete(start, end)
                self.text.mark_unset(start)
                self.text.mark_unset(end)
            self.text.config(state="disabled")
        except tk.TclError:
            pass

    # -- reasoning ---------------------------------------------------------
    def append_thought(self, stream_id: int, chunk: str) -> None:
        """Stream a reasoning model's working, above the answer it precedes.

        Qwen3 and friends can reason for a minute before writing a word.
        Dropping that — which is what reading only `message.content` does —
        turns a model that is working perfectly into one that appears to have
        hung, which is the single complaint this app has spent the most
        effort on.  So the reasoning is shown live, dimmed and indented so it
        never reads as the answer, and folded away once the answer arrives.
        """
        stream = self._streams.get(stream_id)
        if not stream or not chunk:
            return
        block = self._thoughts.get(stream_id) or self._open_thought(stream_id)
        if block is None:
            return
        bottom = self._at_bottom()
        self.text.config(state="normal")
        try:
            self.text.insert(stream["thought"], chunk, block["tag"])
        except tk.TclError:
            self.text.config(state="disabled")
            return
        block["chunks"].append(chunk)
        self.text.config(state="disabled")
        self._autoscroll(bottom)

    def _open_thought(self, stream_id: int) -> dict | None:
        stream = self._streams.get(stream_id)
        if not stream:
            return None
        tag, head_tag = f"thought{stream_id}", f"thoughthead{stream_id}"
        self.text.tag_config(tag, font=FONT_CHAT_ITALIC, foreground=MUTED,
                             lmargin1=34, lmargin2=34)
        self.text.tag_config(head_tag, font=FONT_SMALL, foreground=MUTED,
                             lmargin1=18, lmargin2=18)
        self.text.tag_bind(head_tag, "<Button-1>",
                           lambda _e, sid=stream_id: self.toggle_thought(sid))
        self.text.tag_bind(head_tag, "<Enter>",
                           lambda _e: self.text.config(cursor="hand2"))
        self.text.tag_bind(head_tag, "<Leave>",
                           lambda _e: self.text.config(cursor=""))
        # Keep the reasoning behind the answer, whatever order the tags were
        # created in — an italic grey trace must never outrank the reply.
        self.text.tag_lower(tag, "bot_msg")

        mark = stream["thought"]
        self.text.config(state="normal")
        head_anchor = self.text.index(mark)
        self.text.insert(mark, "💭 Thinking…\n", head_tag)
        self.text.config(state="disabled")

        head_mark = f"s{stream_id}thead"
        self.text.mark_set(head_mark, head_anchor)
        self.text.mark_gravity(head_mark, "left")

        block = {"tag": tag, "head_tag": head_tag, "head": head_mark,
                 "chunks": [], "collapsed": False, "seconds": 0.0}
        self._thoughts[stream_id] = block
        return block

    def _set_thought_header(self, block: dict, text: str) -> None:
        """Rewrite the one-line header in place."""
        head = block["head"]
        self.text.config(state="normal")
        try:
            self.text.delete(head, f"{head} lineend")
            self.text.insert(head, text, block["head_tag"])
        except tk.TclError:
            pass
        self.text.config(state="disabled")

    def toggle_thought(self, stream_id: int) -> None:
        block = self._thoughts.get(stream_id)
        if not block:
            return
        self.collapse_thought(stream_id, not block["collapsed"])

    def collapse_thought(self, stream_id: int, collapsed: bool = True) -> None:
        block = self._thoughts.get(stream_id)
        if not block:
            return
        block["collapsed"] = collapsed
        try:
            self.text.tag_config(block["tag"], elide=collapsed)
        except tk.TclError:
            return
        # "Thought for 0s" is worse than saying nothing about the time.
        took = (human_duration(block["seconds"])
                if block["seconds"] >= 1 else "")
        if collapsed:
            label = f"💭 Thought for {took}" if took else "💭 Reasoning"
            self._set_thought_header(block, f"▸ {label} — click to show")
        else:
            label = f"💭 Thought for {took}" if took else "💭 Thinking…"
            self._set_thought_header(block, f"▾ {label}")

    def close_thought(self, stream_id: int, seconds: float = 0.0) -> None:
        """The answer has started, so fold the working away."""
        block = self._thoughts.get(stream_id)
        if not block:
            return
        block["seconds"] = seconds
        if not "".join(block["chunks"]).strip():
            # Nothing was actually reasoned; leave no empty block behind.
            self._drop_thought(stream_id)
            return
        self.collapse_thought(stream_id, True)

    def _drop_thought(self, stream_id: int) -> None:
        block = self._thoughts.pop(stream_id, None)
        if not block:
            return
        head = block["head"]
        self.text.config(state="normal")
        try:
            self.text.delete(head, f"{head} lineend + 1c")
        except tk.TclError:
            pass
        self.text.config(state="disabled")
        try:
            self.text.mark_unset(head)
        except tk.TclError:
            pass

    def thought_text(self, stream_id: int) -> str:
        block = self._thoughts.get(stream_id)
        return "".join(block["chunks"]) if block else ""

    # -- streaming body ----------------------------------------------------
    def append_stream(self, stream_id: int, chunk: str) -> None:
        stream = self._streams.get(stream_id)
        if not stream or not chunk:
            return
        bottom = self._at_bottom()
        self._stop_typing(stream)
        block = self._thoughts.get(stream_id)
        if block is not None and not block["collapsed"]:
            # The answer has started, so the working is no longer the news.
            self.collapse_thought(stream_id, True)
        self.text.config(state="normal")
        self.text.insert(stream["body"], chunk, "bot_msg")
        stream["chunks"].append(chunk)
        stream["pending"] += chunk
        if "\n" in stream["pending"]:
            self._flush_lines(stream)
        self.text.config(state="disabled")
        self._autoscroll(bottom)

    def _flush_lines(self, stream: dict) -> None:
        """Style each line the moment it is finished.

        Markdown can't be parsed mid-line — a lone "**" could be an opening or
        a closing marker — but once the newline arrives the line is settled and
        can be restyled immediately.  So only the line still being typed ever
        shows raw markers, and only for a moment.
        """
        complete, separator, remainder = stream["pending"].rpartition("\n")
        if not separator:
            return
        complete += "\n"

        try:
            if has_markdown(complete):
                self.text.delete(stream["tail"], stream["body"])
                for run_text, run_tags in render_markdown(complete):
                    self.text.insert(stream["body"], run_text,
                                     ("bot_msg", *run_tags))
                self.text.mark_set(stream["tail"],
                                   self.text.index(stream["body"]))
                if remainder:
                    self.text.insert(stream["body"], remainder, "bot_msg")
            else:
                # nothing to restyle; move the tail past the settled text
                self.text.mark_set(
                    stream["tail"],
                    self.text.index(f"{stream['body']} - {len(remainder)}c"))
            self.text.mark_gravity(stream["tail"], "left")
        except tk.TclError:
            return
        stream["pending"] = remainder

    def end_stream(self, stream_id: int, text: str | None = None,
                   meta: str = "", thought_s: float = 0.0) -> None:
        stream = self._streams.pop(stream_id, None)
        if not stream:
            return
        self._stop_typing(stream)
        self.close_thought(stream_id, thought_s)

        final = text if text is not None else "".join(stream["chunks"])
        final = tidy(final)
        stream["holder"].text = final

        bottom = self._at_bottom()
        self.text.config(state="normal")
        # Markdown is re-rendered once at the end: parsing it mid-stream would
        # mean guessing whether a lone "**" is an opening or closing marker.
        # We also redraw whenever the caller hands back text that differs from
        # what streamed — that's how control markers like [END] get removed.
        streamed = tidy("".join(stream["chunks"]))
        if final and (has_markdown(final) or final != streamed):
            self.text.delete(stream["start"], stream["body"])
            for run_text, run_tags in render_markdown(final):
                self.text.insert(stream["body"], run_text,
                                 ("bot_msg", *run_tags))
        if meta:
            self.text.insert(stream["head"], f"  {meta}", "meta")
        self.text.config(state="disabled")
        self._autoscroll(bottom)

        for key in ("body", "head", "start", "tail", "thought"):
            try:
                self.text.mark_unset(stream[key])
            except tk.TclError:
                pass

    def fail_stream(self, stream_id: int, message: str) -> None:
        stream = self._streams.pop(stream_id, None)
        if stream:
            self._stop_typing(stream)
            self.close_thought(stream_id)
            stream["holder"].text = message
            self.text.config(state="normal")
            self.text.insert(stream["body"], f"⚠ {message}", "sys_err")
            self.text.config(state="disabled")
            self._last_speaker = None
        else:
            self.add_error(message)
