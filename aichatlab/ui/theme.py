"""Colours, fonts and a couple of shared widget helpers."""

from __future__ import annotations

import tkinter as tk

BG = "#e9edf5"
PANEL = "#ffffff"
CHAT_BG = "#f4f7fb"
INPUT_BG = "#ffffff"
ACCENT = "#4f8ef7"
ACCENT_DARK = "#3b76d9"
USER_BG = "#4f8ef7"
USER_FG = "#ffffff"
BOT_BG = "#ffffff"
BOT_FG = "#1f2430"
MUTED = "#8a92a6"
ERROR = "#d64545"
OK = "#2e9e5b"
BORDER = "#d7dce7"
CHIP_BG = "#e3ebfb"
SUBTLE = "#f0f3f9"
SELECTION = "#a8ccf7"      # text-selection highlight
SLIDER = "#aeb7c9"         # slider handle at rest (blue on hover)
SLIDER_TROUGH = "#d9dfea"  # slider track

FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_CHAT = ("Segoe UI", 11)
FONT_CHAT_BOLD = ("Segoe UI", 11, "bold")
FONT_CHAT_ITALIC = ("Segoe UI", 11, "italic")
FONT_CHAT_HEADING = ("Segoe UI", 12, "bold")
FONT_MONO = ("Consolas", 10)
FONT_MONO_SMALL = ("Consolas", 10)

CODE_BG = "#eef1f7"
CODE_FG = "#25405e"


def flat_button(parent, text, command, primary=False, **kwargs):
    """A borderless button that looks the same on every platform."""
    return tk.Button(
        parent, text=text, command=command,
        font=FONT_BOLD if primary else FONT,
        bg=ACCENT if primary else SUBTLE,
        fg="white" if primary else BOT_FG,
        activebackground=ACCENT_DARK if primary else "#e2e8f3",
        activeforeground="white" if primary else BOT_FG,
        relief="flat", cursor="hand2", padx=12, pady=4, bd=0,
        highlightthickness=0, **kwargs)


def link_button(parent, text, command, **kwargs):
    return tk.Button(parent, text=text, command=command, font=FONT_SMALL,
                     bd=0, bg=PANEL, fg=ACCENT, activebackground=PANEL,
                     activeforeground=ACCENT_DARK, cursor="hand2",
                     highlightthickness=0, **kwargs)


def separator(parent, **kwargs):
    return tk.Frame(parent, bg=BORDER, height=1, **kwargs)
