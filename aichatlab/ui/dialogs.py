"""Settings dialog: server addresses with a live Connect test, plus sampling."""

from __future__ import annotations

import threading
import tkinter as tk
from typing import Callable

from ..client import OllamaClient
from ..config import Settings
from .theme import (
    ACCENT,
    ACCENT_DARK,
    BOT_BG,
    ERROR,
    FONT,
    FONT_BOLD,
    FONT_SMALL,
    MUTED,
    OK,
    PANEL,
    flat_button,
)

SECTIONS = (
    ("local", "💻 Local server",
     "Ollama running on this computer (usually 127.0.0.1:11434)"),
    ("host", "🌐 Host server",
     "Another computer running Ollama — enter its IP and press Connect"),
)


class SettingsDialog(tk.Toplevel):
    def __init__(self, master, settings: Settings,
                 on_saved: Callable[[], None] | None = None):
        super().__init__(master)
        self.settings = settings
        self.on_saved = on_saved or (lambda: None)
        self.entries: dict[str, tuple] = {}

        self.title("Settings")
        self.configure(bg=PANEL, padx=18, pady=14)
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.geometry(f"+{master.winfo_rootx() + 150}+{master.winfo_rooty() + 90}")

        row = 0
        for server, title, hint in SECTIONS:
            row = self._server_section(row, server, title, hint)
        row = self._searxng_section(row)
        row = self._model_section(row)
        self._buttons(row)

    # -- sections ----------------------------------------------------------
    def _label(self, row, column, text, font=FONT, fg=BOT_BG, **grid):
        widget = tk.Label(self, text=text, font=font, bg=PANEL, fg=fg)
        widget.grid(row=row, column=column, **grid)
        return widget

    def _server_section(self, row: int, server: str, title: str, hint: str) -> int:
        tk.Label(self, text=title, font=FONT_BOLD, bg=PANEL, fg="#1f2430").grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(12 if row else 0, 2))
        tk.Label(self, text=hint, font=FONT_SMALL, bg=PANEL, fg=MUTED).grid(
            row=row + 1, column=0, columnspan=4, sticky="w")

        tk.Label(self, text="IP address:", font=FONT, bg=PANEL,
                 fg="#1f2430").grid(row=row + 2, column=0, sticky="w", pady=6)
        ip_var = tk.StringVar(value=str(self.settings.get(f"{server}_ip", "")))
        tk.Entry(self, textvariable=ip_var, font=FONT, width=20, relief="solid",
                 bd=1).grid(row=row + 2, column=1, sticky="w", padx=(6, 12))

        tk.Label(self, text="Port:", font=FONT, bg=PANEL, fg="#1f2430").grid(
            row=row + 2, column=2, sticky="w")
        port_var = tk.StringVar(value=str(self.settings.get(f"{server}_port", "11434")))
        tk.Entry(self, textvariable=port_var, font=FONT, width=7, relief="solid",
                 bd=1).grid(row=row + 2, column=3, sticky="w", padx=6)

        status = tk.Label(self, text="", font=FONT_SMALL, bg=PANEL, fg=MUTED)
        status.grid(row=row + 3, column=1, columnspan=3, sticky="w")

        tk.Button(self, text="Connect", font=FONT_BOLD, bg=ACCENT, fg="white",
                  activebackground=ACCENT_DARK, activeforeground="white",
                  relief="flat", cursor="hand2", padx=14, pady=2, bd=0,
                  command=lambda: self._connect(server, ip_var, port_var, status)
                  ).grid(row=row + 3, column=0, sticky="w", pady=(4, 2))

        self.entries[server] = (ip_var, port_var)
        return row + 4

    def _searxng_section(self, row: int) -> int:
        tk.Label(self, text="🔍 Web research (SearXNG)", font=FONT_BOLD,
                 bg=PANEL, fg="#1f2430").grid(row=row, column=0, columnspan=4,
                                              sticky="w", pady=(14, 2))
        tk.Label(self,
                 text="Your self-hosted SearXNG instance, e.g. http://192.168.1.20:8080",
                 font=FONT_SMALL, bg=PANEL, fg=MUTED).grid(
            row=row + 1, column=0, columnspan=4, sticky="w")

        tk.Label(self, text="URL:", font=FONT, bg=PANEL, fg="#1f2430").grid(
            row=row + 2, column=0, sticky="w", pady=6)
        self.searxng_var = tk.StringVar(
            value=str(self.settings.get("searxng_url", "")))
        tk.Entry(self, textvariable=self.searxng_var, font=FONT, width=32,
                 relief="solid", bd=1).grid(row=row + 2, column=1, columnspan=3,
                                            sticky="w", padx=(6, 0))

        status = tk.Label(self, text="", font=FONT_SMALL, bg=PANEL, fg=MUTED)
        status.grid(row=row + 3, column=1, columnspan=3, sticky="w")

        tk.Button(self, text="Connect", font=FONT_BOLD, bg=ACCENT, fg="white",
                  activebackground=ACCENT_DARK, activeforeground="white",
                  relief="flat", cursor="hand2", padx=14, pady=2, bd=0,
                  command=lambda: self._connect_searxng(status)
                  ).grid(row=row + 3, column=0, sticky="w", pady=(4, 2))

        self.research_cache = tk.BooleanVar(
            value=bool(self.settings.get("research_cache", True)))
        tk.Checkbutton(self, text="Reuse recent research instead of searching "
                                  "again", variable=self.research_cache,
                       font=FONT, bg=PANEL, activebackground=PANEL,
                       fg="#1f2430", highlightthickness=0, bd=0,
                       cursor="hand2").grid(row=row + 4, column=0,
                                            columnspan=4, sticky="w",
                                            pady=(6, 0))

        tk.Label(self, text="Keep research for (minutes):", font=FONT, bg=PANEL,
                 fg="#1f2430").grid(row=row + 5, column=0, sticky="w", pady=4)
        self.cache_ttl = tk.StringVar(
            value=str(self.settings.get("research_cache_ttl_minutes", 360)))
        tk.Entry(self, textvariable=self.cache_ttl, font=FONT, width=9,
                 relief="solid", bd=1).grid(row=row + 5, column=1, sticky="w",
                                            padx=(6, 12))
        tk.Label(self, text="0 = never expire · weather and news always refresh "
                            "every 15 min", font=FONT_SMALL, bg=PANEL,
                 fg=MUTED).grid(row=row + 5, column=2, columnspan=2, sticky="w")

        self.assess_first = tk.BooleanVar(
            value=bool(self.settings.get("assess_before_search", True)))
        tk.Checkbutton(self, text="Read attached files first, and only search "
                                  "if they don't answer it",
                       variable=self.assess_first, font=FONT, bg=PANEL,
                       activebackground=PANEL, fg="#1f2430",
                       highlightthickness=0, bd=0, cursor="hand2").grid(
            row=row + 6, column=0, columnspan=4, sticky="w", pady=(6, 0))
        return row + 7

    def _connect_searxng(self, status) -> None:
        url = self.searxng_var.get().strip()
        if not url:
            status.config(text="Enter the SearXNG URL first", fg=ERROR)
            return
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
            self.searxng_var.set(url)
        status.config(text="Connecting…", fg=MUTED)

        def worker():
            from ..research import ResearchError, SearxngClient
            try:
                count = SearxngClient(url, timeout=10).ping()
            except ResearchError as exc:
                self.after(0, lambda error=exc: self._safe(
                    status, f"✗ {str(error)[:160]}", ERROR))
                return

            def ok():
                self.settings["searxng_url"] = url
                self.settings.save()
                self._safe(status,
                           f"✓ Connected — test search returned {count} results",
                           OK)

            self.after(0, ok)

        threading.Thread(target=worker, daemon=True).start()

    def _model_section(self, row: int) -> int:
        tk.Label(self, text="🎛 Generation", font=FONT_BOLD, bg=PANEL,
                 fg="#1f2430").grid(row=row, column=0, columnspan=4, sticky="w",
                                    pady=(14, 2))

        tk.Label(self, text="Temperature:", font=FONT, bg=PANEL,
                 fg="#1f2430").grid(row=row + 1, column=0, sticky="w", pady=4)
        self.temperature = tk.StringVar(value=str(self.settings.get("temperature", 0.7)))
        tk.Entry(self, textvariable=self.temperature, font=FONT, width=7,
                 relief="solid", bd=1).grid(row=row + 1, column=1, sticky="w",
                                            padx=(6, 12))

        tk.Label(self, text="Context budget (tokens):", font=FONT, bg=PANEL,
                 fg="#1f2430").grid(row=row + 2, column=0, sticky="w", pady=4)
        self.budget = tk.StringVar(
            value=str(self.settings.get("context_budget_tokens", 6000)))
        tk.Entry(self, textvariable=self.budget, font=FONT, width=9,
                 relief="solid", bd=1).grid(row=row + 2, column=1, sticky="w",
                                            padx=(6, 12))
        tk.Label(self, text="older turns are trimmed past this", font=FONT_SMALL,
                 bg=PANEL, fg=MUTED).grid(row=row + 2, column=2, columnspan=2,
                                          sticky="w")

        tk.Label(self, text="Largest model window:", font=FONT, bg=PANEL,
                 fg="#1f2430").grid(row=row + 3, column=0, sticky="w", pady=4)
        self.max_window = tk.StringVar(
            value=str(self.settings.get("max_context_window", 32768)))
        tk.Entry(self, textvariable=self.max_window, font=FONT, width=9,
                 relief="solid", bd=1).grid(row=row + 3, column=1, sticky="w",
                                            padx=(6, 12))
        tk.Label(self, text="cap on the num_ctx we ask for — bigger costs VRAM",
                 font=FONT_SMALL, bg=PANEL, fg=MUTED).grid(
            row=row + 3, column=2, columnspan=2, sticky="w")

        tk.Label(self, text="Give up after (seconds):", font=FONT, bg=PANEL,
                 fg="#1f2430").grid(row=row + 4, column=0, sticky="w", pady=4)
        self.timeout = tk.StringVar(
            value=str(self.settings.get("request_timeout", 1800)))
        tk.Entry(self, textvariable=self.timeout, font=FONT, width=9,
                 relief="solid", bd=1).grid(row=row + 4, column=1, sticky="w",
                                            padx=(6, 12))
        tk.Label(self, text="silence allowed before giving up — a big prompt "
                            "needs more", font=FONT_SMALL, bg=PANEL,
                 fg=MUTED).grid(row=row + 4, column=2, columnspan=2, sticky="w")

        self.conversational = tk.BooleanVar(
            value=bool(self.settings.get("conversational", True)))
        tk.Checkbutton(self, text="Talk like a person (no restating the "
                                  "question, no lecturing)",
                       variable=self.conversational, font=FONT, bg=PANEL,
                       activebackground=PANEL, fg="#1f2430",
                       highlightthickness=0, bd=0, cursor="hand2").grid(
            row=row + 5, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.smart_followups = tk.BooleanVar(
            value=bool(self.settings.get("smart_followups", True)))
        tk.Checkbutton(self, text="Understand follow-ups (\"what's its 0-60?\" "
                                  "searches the thing you were discussing)",
                       variable=self.smart_followups, font=FONT, bg=PANEL,
                       activebackground=PANEL, fg="#1f2430",
                       highlightthickness=0, bd=0, cursor="hand2").grid(
            row=row + 6, column=0, columnspan=4, sticky="w")

        self.auto_compact = tk.BooleanVar(
            value=bool(self.settings.get("auto_compact", True)))
        tk.Checkbutton(self, text="Summarise old turns as the context fills, "
                                  "instead of dropping them",
                       variable=self.auto_compact, font=FONT, bg=PANEL,
                       activebackground=PANEL, fg="#1f2430",
                       highlightthickness=0, bd=0, cursor="hand2").grid(
            row=row + 7, column=0, columnspan=4, sticky="w")

        # Off, this is the old behaviour: a row in the chat with a button on
        # it.  That row is easy to scroll past when the model has just written
        # three paragraphs above it, which is how "it listed the changes and
        # did nothing" gets said about a run that worked.
        self.review_popup = tk.BooleanVar(
            value=bool(self.settings.get("review_popup", True)))
        tk.Checkbutton(self, text="Open the review window as soon as changes "
                                  "are proposed (nothing is written until you "
                                  "approve it)",
                       variable=self.review_popup, font=FONT, bg=PANEL,
                       activebackground=PANEL, fg="#1f2430",
                       highlightthickness=0, bd=0, cursor="hand2").grid(
            row=row + 8, column=0, columnspan=4, sticky="w")

        tk.Label(self, text="System prompt:", font=FONT, bg=PANEL,
                 fg="#1f2430").grid(row=row + 9, column=0, sticky="nw", pady=4)
        self.system_prompt = tk.Text(self, height=3, width=42, font=FONT,
                                     relief="solid", bd=1, wrap="word")
        self.system_prompt.insert("1.0", str(self.settings.get("system_prompt", "")))
        self.system_prompt.grid(row=row + 9, column=1, columnspan=3, sticky="w",
                                padx=(6, 0), pady=4)
        return row + 10

    def _buttons(self, row: int) -> None:
        bar = tk.Frame(self, bg=PANEL)
        bar.grid(row=row, column=0, columnspan=4, sticky="e", pady=(16, 0))
        flat_button(bar, "Cancel", self.destroy).pack(side="right", padx=(8, 0))
        flat_button(bar, "Save & Close", self._save, primary=True).pack(side="right")

    # -- actions -----------------------------------------------------------
    def _connect(self, server, ip_var, port_var, status) -> None:
        ip = ip_var.get().strip()
        port = port_var.get().strip() or "11434"
        if not ip:
            status.config(text="Enter an IP address first", fg=ERROR)
            return
        status.config(text="Connecting…", fg=MUTED)

        def worker():
            client = OllamaClient(f"http://{ip}:{port}", server=server)
            try:
                models = client.list_models(timeout=8)
            except Exception as exc:
                # bind now: Python clears `exc` when the except block ends
                self.after(0, lambda error=exc: self._safe(
                    status, f"✗ {str(error)[:90]}", ERROR))
                return

            def ok():
                self.settings[f"{server}_ip"] = ip
                self.settings[f"{server}_port"] = port
                self.settings.save()
                self._safe(status, f"✓ Connected — {len(models)} model(s)", OK)
                self.on_saved()

            self.after(0, ok)

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _safe(widget, text, colour) -> None:
        try:
            widget.config(text=text, fg=colour)
        except tk.TclError:
            pass

    def _save(self) -> None:
        for server, (ip_var, port_var) in self.entries.items():
            self.settings[f"{server}_ip"] = ip_var.get().strip()
            self.settings[f"{server}_port"] = port_var.get().strip() or "11434"
        try:
            self.settings["temperature"] = float(self.temperature.get())
        except ValueError:
            self.settings["temperature"] = 0.7
        try:
            self.settings["context_budget_tokens"] = max(500, int(self.budget.get()))
        except ValueError:
            self.settings["context_budget_tokens"] = 6000
        try:
            self.settings["max_context_window"] = max(2048,
                                                      int(self.max_window.get()))
        except ValueError:
            self.settings["max_context_window"] = 32768
        try:
            self.settings["request_timeout"] = max(30, int(self.timeout.get()))
        except ValueError:
            self.settings["request_timeout"] = 1800
        self.settings["system_prompt"] = self.system_prompt.get("1.0", "end").strip()
        self.settings["searxng_url"] = self.searxng_var.get().strip()
        self.settings["conversational"] = bool(self.conversational.get())
        self.settings["smart_followups"] = bool(self.smart_followups.get())
        self.settings["research_cache"] = bool(self.research_cache.get())
        self.settings["auto_compact"] = bool(self.auto_compact.get())
        self.settings["review_popup"] = bool(self.review_popup.get())
        self.settings["assess_before_search"] = bool(self.assess_first.get())
        try:
            # 0 means "never expire" — don't clamp it up to 1
            self.settings["research_cache_ttl_minutes"] = max(
                0, int(self.cache_ttl.get()))
        except ValueError:
            self.settings["research_cache_ttl_minutes"] = 360
        self.settings.save()
        self.destroy()
        self.on_saved()
