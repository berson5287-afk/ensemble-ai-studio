"""The main window: model columns, orchestration modes and the chat panel."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .. import __version__
from ..benchmark import Target as _Target  # noqa: F401  (re-export convenience)
from ..cache import ResearchCache
from ..client import OllamaClient
from ..config import CONVERSATIONAL_PROMPT, Settings
from ..documents import SUPPORTED_HINT, BinaryFileError, ExtractionError, extract
from ..formatting import friendly_model_name
from ..knowledge import (
    KnowledgeBase,
    build_extraction_prompt,
    format_for_prompt,
    parse_lessons,
    worth_learning_from,
)
from ..orchestrator import MODES, Orchestrator, Target, parse_relay
from ..research import ResearchError, SearxngClient, contextual_query, gather
from ..session import Session, make_key, short_model, split_key
from .benchmark_window import BenchmarkWindow
from .chatview import ChatView
from .dialogs import SettingsDialog
from .knowledge_window import KnowledgeWindow
from .theme import (
    ACCENT,
    BG,
    BORDER,
    CHIP_BG,
    ERROR,
    FONT,
    FONT_BOLD,
    FONT_CHAT,
    FONT_SMALL,
    FONT_TITLE,
    MUTED,
    OK,
    PANEL,
    SLIDER,
    SLIDER_TROUGH,
    SUBTLE,
    flat_button,
    link_button,
    separator,
)

try:  # optional drag & drop
    from tkinterdnd2 import DND_FILES, TkinterDnD
    BaseTk = TkinterDnD.Tk
    DND_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the user's install
    BaseTk = tk.Tk
    DND_AVAILABLE = False

SERVER_TITLES = {"local": "💻 Local models", "host": "🌐 Host models"}
PLACEHOLDER = "Type a message…  (Enter to send, Shift+Enter for a new line)"
MODE_ORDER = ("chat", "debate", "critique", "judge")
FILE_TYPES = [
    ("All supported", "*.txt *.md *.py *.json *.csv *.log *.xml *.html "
                      "*.pdf *.docx *.pptx *.xlsx *.xlsm"),
    ("Documents", "*.pdf *.docx *.pptx *.xlsx *.xlsm"),
    ("Text files", "*.txt *.md *.csv *.log *.json *.xml *.html"),
    ("All files", "*.*"),
]


class ChatLabApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.settings = Settings.load()
        self.session = Session()
        self.model_vars: dict[tuple, tk.BooleanVar] = {}
        self.columns: dict[str, dict] = {}
        self.attachments: list[dict] = []
        self.events: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.placeholder_showing = False
        self.knowledge = KnowledgeBase()
        self.cache = ResearchCache(
            base_ttl_minutes=int(self.settings.get(
                "research_cache_ttl_minutes", 360)))

        root.title(f"AI Chat Lab {__version__}")
        root.geometry("1320x840")
        root.minsize(1060, 680)
        root.configure(bg=BG)
        root.bind("<Escape>", lambda e: self.stop())

        self._build_header()
        self._build_body()
        self._build_status()
        self._enable_dnd()

        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._show_placeholder()
        self.chat.add_note(
            "👋 Welcome! Tick models on the left, pick a mode, and press Send.\n"
            "Select any text to copy it, or click 📋 to copy a whole message.")
        self.refresh_models()
        self._pump()

    # ------------------------------------------------------------------ UI
    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=PANEL, padx=14, pady=10)
        header.pack(fill="x")
        separator(self.root).pack(fill="x")

        tk.Label(header, text="🤖 AI Chat Lab", font=FONT_TITLE, bg=PANEL,
                 fg="#1f2430").pack(side="left")

        for text, command in (
            ("⚙ Settings", self.open_settings),
            ("📊 Benchmark", self.open_benchmark),
            ("⏹ Stop (Esc)", self.stop),
            ("📋 Copy all", lambda: self.chat.copy_all()),
            ("💾 Save", self.save_chat),
            ("📂 Open", self.load_chat),
            ("✨ New chat", self.new_chat),
        ):
            flat_button(header, text, command).pack(side="right", padx=(6, 0))

    def _build_body(self) -> None:
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=12)

        self._build_sidebar(body)

        panel = tk.Frame(body, bg=PANEL, highlightbackground=BORDER,
                         highlightthickness=1)
        panel.pack(side="left", fill="both", expand=True)

        self.chat = ChatView(panel, on_status=self.set_status, bg=PANEL)
        self.chat.pack(fill="both", expand=True, padx=2, pady=(2, 0))

        self._build_mode_bar(panel)
        self.attach_bar = tk.Frame(panel, bg=PANEL)
        self._build_input(panel)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        sidebar = tk.Frame(parent, bg=PANEL, width=390,
                           highlightbackground=BORDER, highlightthickness=1)
        sidebar.pack(side="left", fill="y", padx=(0, 12))
        sidebar.pack_propagate(False)

        head = tk.Frame(sidebar, bg=PANEL)
        head.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(head, text="Models", font=FONT_BOLD, bg=PANEL,
                 fg="#1f2430").pack(side="left")
        flat_button(head, "↻ Refresh",
                    lambda: self.refresh_models(interactive=True)).pack(side="right")

        grid = tk.Frame(sidebar, bg=PANEL)
        grid.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        grid.grid_columnconfigure(0, weight=1, uniform="col")
        grid.grid_columnconfigure(1, weight=1, uniform="col")
        grid.grid_rowconfigure(0, weight=1)
        for index, server in enumerate(("local", "host")):
            self._build_column(grid, index, server)

        tk.Label(sidebar, text=("Tip: relay with  gemma2 ask llama3 <question>\n"
                                "Debate and Judge need two or more models."),
                 font=FONT_SMALL, fg=MUTED, bg=PANEL, justify="left").pack(
            anchor="w", padx=12, pady=(0, 10))

    def _build_column(self, parent: tk.Frame, column: int, server: str) -> None:
        frame = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER,
                         highlightthickness=1)
        frame.grid(row=0, column=column, sticky="nsew", padx=4)

        head = tk.Frame(frame, bg=SUBTLE)
        head.pack(fill="x")
        tk.Label(head, text=SERVER_TITLES[server], font=FONT_BOLD, bg=SUBTLE,
                 fg="#1f2430").pack(anchor="w", padx=8, pady=(6, 0))
        status = tk.Label(head, text="●", font=FONT_SMALL, bg=SUBTLE, fg=MUTED)
        status.pack(anchor="w", padx=8, pady=(0, 4))

        tools = tk.Frame(frame, bg=PANEL)
        tools.pack(fill="x", padx=6, pady=2)
        link_button(tools, "All", lambda: self._set_all(server, True)).pack(side="left")
        link_button(tools, "None", lambda: self._set_all(server, False)).pack(
            side="left", padx=(8, 0))
        link_button(tools, "↻",
                    lambda: self.refresh_models(server, interactive=True)).pack(
            side="right")

        canvas = tk.Canvas(frame, highlightthickness=0, bg=PANEL)
        scroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)

        def wheel(event):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        for widget in (canvas, inner):
            widget.bind("<MouseWheel>", wheel)
            widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        canvas.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        scroll.pack(side="right", fill="y")
        self.columns[server] = {"inner": inner, "status": status}

    def _build_mode_bar(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=PANEL)
        bar.pack(fill="x", padx=14, pady=(8, 0))

        tk.Label(bar, text="Mode", font=FONT_SMALL, bg=PANEL, fg=MUTED).pack(
            side="left", padx=(0, 6))
        self.mode = tk.StringVar(value=MODES["chat"])
        self.mode_box = ttk.Combobox(
            bar, textvariable=self.mode, state="readonly", width=38,
            values=[MODES[key] for key in MODE_ORDER], font=FONT_SMALL)
        self.mode_box.pack(side="left")
        self.mode_box.bind("<<ComboboxSelected>>", lambda e: self._sync_mode_bar())

        self.rounds_label = tk.Label(bar, text="Rounds", font=FONT_SMALL,
                                     bg=PANEL, fg=MUTED)
        self.rounds = tk.StringVar(value="2")
        self.rounds_box = tk.Spinbox(bar, from_=1, to=6, width=3, font=FONT_SMALL,
                                     textvariable=self.rounds)

        self.judge_label = tk.Label(bar, text="Judge", font=FONT_SMALL, bg=PANEL,
                                    fg=MUTED)
        self.judge = tk.StringVar(value="(first selected)")
        self.judge_box = ttk.Combobox(bar, textvariable=self.judge,
                                      state="readonly", width=24, font=FONT_SMALL)

        self.context_label = tk.Label(bar, text="", font=FONT_SMALL, bg=PANEL,
                                      fg=MUTED)
        self.context_label.pack(side="right")

        # Second row: web research toggle + the speed↔quality slider.
        row2 = tk.Frame(parent, bg=PANEL)
        row2.pack(fill="x", padx=14, pady=(4, 0))

        self.research_var = tk.BooleanVar(value=False)
        self.research_check = tk.Checkbutton(
            row2, text="🔍 Web research", variable=self.research_var,
            font=FONT_SMALL, bg=PANEL, activebackground=PANEL, fg="#1f2430",
            highlightthickness=0, bd=0, cursor="hand2",
            command=self._on_research_toggle)
        self.research_check.pack(side="left")

        self.learning_var = tk.BooleanVar(
            value=bool(self.settings.get("learning", False)))
        self.learning_check = tk.Checkbutton(
            row2, text="🧠 Learning", variable=self.learning_var,
            font=FONT_SMALL, bg=PANEL, activebackground=PANEL, fg="#1f2430",
            highlightthickness=0, bd=0, cursor="hand2",
            command=self._on_learning_toggle)
        self.learning_check.pack(side="left", padx=(12, 0))

        self.knowledge_link = link_button(row2, "", self.open_knowledge)
        self.knowledge_link.pack(side="left", padx=(2, 0))
        self._refresh_knowledge_link()

        # packed right-to-left, so on screen this reads:
        #   ⚡ Quick  [slider]  Quality 🔬  <profile name>
        self.profile_label = tk.Label(row2, text="", font=FONT_SMALL, bg=PANEL,
                                      fg=ACCENT, width=11, anchor="e")
        self.profile_label.pack(side="right", padx=(6, 0))
        tk.Label(row2, text="Quality 🔬", font=FONT_SMALL, bg=PANEL,
                 fg=MUTED).pack(side="right", padx=(6, 0))
        # tk.Scale draws its handle in the widget's own background colour, so
        # a white bg made the handle invisible.  Grey at rest, blue on hover.
        self.speed_scale = tk.Scale(
            row2, from_=0, to=100, orient="horizontal", showvalue=False,
            length=170, bg=SLIDER, highlightthickness=0, bd=0,
            troughcolor=SLIDER_TROUGH, activebackground=ACCENT,
            sliderrelief="raised", sliderlength=22, width=12,
            command=self._on_speed_change)
        self.speed_scale.set(int(self.settings.get("speed_quality", 50)))
        self.speed_scale.pack(side="right", pady=2)
        self.speed_scale.bind(
            "<Enter>", lambda e: self.speed_scale.config(bg=ACCENT))
        self.speed_scale.bind(
            "<Leave>", lambda e: self.speed_scale.config(bg=SLIDER))
        tk.Label(row2, text="⚡ Quick", font=FONT_SMALL, bg=PANEL,
                 fg=MUTED).pack(side="right", padx=(0, 6))

        self._on_speed_change(self.speed_scale.get())
        self._sync_mode_bar()

    def _on_speed_change(self, value) -> None:
        try:
            self.settings["speed_quality"] = int(float(value))
        except (TypeError, ValueError):
            return
        label, num_predict, _ = self.settings.profile()
        cap = "uncapped" if num_predict < 0 else f"≤{num_predict} tok"
        self.profile_label.config(text=f"{label}")
        if hasattr(self, "status"):     # the status bar is built after this bar
            self.set_status(f"Answer style: {label} ({cap})")
        self.settings.save()

    def _on_learning_toggle(self) -> None:
        self.settings["learning"] = bool(self.learning_var.get())
        self.settings.save()
        if self.learning_var.get():
            self.set_status(
                f"Learning on — durable lessons are saved and reused "
                f"({len(self.knowledge)} so far)")
        else:
            self.set_status("Learning off — nothing new will be remembered")

    def _refresh_knowledge_link(self) -> None:
        count = len(self.knowledge)
        self.knowledge_link.config(
            text=f"({count} learned)" if count else "(none yet)")

    def _on_research_toggle(self) -> None:
        if self.research_var.get() and not str(
                self.settings.get("searxng_url", "")).strip():
            self.research_var.set(False)
            messagebox.showinfo(
                "Set up SearXNG first",
                "Web research uses your SearXNG instance.\n\n"
                "Enter its URL in ⚙ Settings (e.g. http://192.168.1.20:8080) "
                "and press Connect, then switch this on.")
            self.open_settings()

    def _build_input(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=14, pady=12)

        flat_button(row, "📎", self.import_files).pack(side="left", padx=(0, 8),
                                                       anchor="s")
        self.send_button = flat_button(row, "Send  ➤", self.send, primary=True)
        self.send_button.config(pady=8, padx=18)
        self.send_button.pack(side="right", padx=(10, 0), anchor="s")

        wrap = tk.Frame(row, bg=PANEL, highlightbackground=BORDER,
                        highlightthickness=1)
        wrap.pack(side="left", fill="both", expand=True)
        self.input = tk.Text(wrap, height=3, width=10, wrap=tk.WORD,
                             font=FONT_CHAT, bg=PANEL, fg=MUTED, relief="flat",
                             padx=10, pady=8)
        self.input.pack(fill="both", expand=True)
        self.input.bind("<Return>", self._on_return)
        self.input.bind("<FocusIn>", lambda e: self._hide_placeholder())
        self.input.bind("<FocusOut>", lambda e: self._restore_placeholder())
        for sequence in ("<Control-a>", "<Control-A>"):
            self.input.bind(sequence, self._select_all_input)

    def _build_status(self) -> None:
        separator(self.root).pack(fill="x", side="bottom")
        bar = tk.Frame(self.root, bg=PANEL)
        bar.pack(fill="x", side="bottom")
        self.status = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self.status, anchor="w", font=FONT_SMALL,
                 bg=PANEL, fg=MUTED, padx=12, pady=4).pack(side="left")
        if not DND_AVAILABLE:
            tk.Label(bar, text="Drag & drop off — run:  py -m pip install tkinterdnd2",
                     anchor="e", font=FONT_SMALL, bg=PANEL, fg=MUTED,
                     padx=12).pack(side="right")

    def _enable_dnd(self) -> None:
        if not DND_AVAILABLE:
            return
        try:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass

    # --------------------------------------------------------------- state
    def set_status(self, text: str) -> None:
        self.status.set(text)

    def clients(self) -> dict[str, OllamaClient]:
        timeout = int(self.settings.get("request_timeout", 600))
        return {server: OllamaClient(self.settings.base_url(server),
                                     server=server, timeout=timeout)
                for server in ("local", "host")}

    def available_targets(self) -> list[Target]:
        return [Target(server, model) for (server, model) in self.model_vars]

    def selected_targets(self) -> list[Target]:
        return [Target(server, model)
                for (server, model), var in self.model_vars.items() if var.get()]

    def current_mode(self) -> str:
        label = self.mode.get()
        for key, text in MODES.items():
            if text == label:
                return key
        return "chat"

    def _sync_mode_bar(self) -> None:
        mode = self.current_mode()
        for widget in (self.rounds_label, self.rounds_box, self.judge_label,
                       self.judge_box):
            widget.pack_forget()
        if mode == "debate":
            self.rounds_label.pack(side="left", padx=(14, 4))
            self.rounds_box.pack(side="left")
        if mode in ("debate", "judge"):
            self.judge_label.pack(side="left", padx=(14, 4))
            self.judge_box.pack(side="left")
        self._sync_judges()

    def _sync_judges(self) -> None:
        labels = ["(first selected)"] + [t.label for t in self.available_targets()]
        self.judge_box.config(values=labels)
        if self.judge.get() not in labels:
            self.judge.set(labels[0])

    def _judge_target(self) -> Target | None:
        label = self.judge.get()
        for target in self.available_targets():
            if target.label == label:
                return target
        return None

    # -------------------------------------------------------------- models
    def refresh_models(self, only: str | None = None,
                       interactive: bool = False) -> None:
        servers = (only,) if only else ("local", "host")
        for server in servers:
            if not self.settings.base_url(server):
                self._set_column_status(server, "● not configured", MUTED)
                self._populate(server, [], hint="Set the IP in ⚙ Settings,\nthen Connect")
                if interactive and only == server:
                    self.open_settings()
                continue
            self._set_column_status(server, "● connecting…", MUTED)
            threading.Thread(target=self._fetch_models,
                             args=(server, interactive), daemon=True).start()

    def _fetch_models(self, server: str, interactive: bool) -> None:
        client = self.clients()[server]
        try:
            names = client.list_models()
        except Exception as exc:
            # bind now: Python clears `exc` when the except block ends
            self.root.after(0, lambda error=exc: self._models_failed(
                server, error, interactive))
            return
        self.root.after(0, lambda: self._models_loaded(server, names))

    def _models_failed(self, server: str, exc: Exception, interactive: bool) -> None:
        self._set_column_status(server, "● offline", ERROR)
        self.set_status(f"Could not reach the {server} server: {str(exc)[:120]}")
        if interactive:
            messagebox.showerror(
                "Connection failed",
                f"Could not fetch models from the {server} server:\n\n{str(exc)[:300]}")

    def _models_loaded(self, server: str, names: list[str]) -> None:
        self._populate(server, names)
        self._set_column_status(server, f"● {len(names)} model(s)", OK)
        self.set_status(f"Loaded {len(names)} {server} model(s)")
        self._sync_judges()

    def _populate(self, server: str, names: list[str], hint: str = "") -> None:
        inner = self.columns[server]["inner"]
        previously = {model for (srv, model), var in self.model_vars.items()
                      if srv == server and var.get()}
        for widget in inner.winfo_children():
            widget.destroy()
        for pair in [p for p in self.model_vars if p[0] == server]:
            del self.model_vars[pair]

        if not names:
            tk.Label(inner, text=hint or "No models found", font=FONT_SMALL,
                     fg=MUTED, bg=PANEL, justify="left").pack(anchor="w", padx=6,
                                                              pady=6)
            return

        for name in names:
            var = tk.BooleanVar(value=(name in previously))
            tk.Checkbutton(inner, text=friendly_model_name(name) or name,
                           variable=var, anchor="w",
                           justify="left", font=FONT, bg=PANEL,
                           activebackground=PANEL, highlightthickness=0, bd=0,
                           cursor="hand2", wraplength=155,
                           command=self._update_context_label).pack(anchor="w",
                                                                    fill="x", padx=2)
            self.model_vars[(server, name)] = var

    def _set_column_status(self, server: str, text: str, colour: str) -> None:
        self.columns[server]["status"].config(text=text, fg=colour)

    def _set_all(self, server: str, value: bool) -> None:
        for (srv, _model), var in self.model_vars.items():
            if srv == server:
                var.set(value)
        self._update_context_label()

    def _update_context_label(self) -> None:
        targets = self.selected_targets()
        if not targets:
            self.context_label.config(text="")
            return
        used = max((self.session.total_tokens(t.key) for t in targets), default=0)
        budget = int(self.settings.get("context_budget_tokens", 6000))
        self.context_label.config(
            text=f"context ≈ {used:,}/{budget:,} tokens",
            fg=ERROR if used > budget else MUTED)

    # --------------------------------------------------------- attachments
    def import_files(self) -> None:
        for path in filedialog.askopenfilenames(title="Attach files",
                                                filetypes=FILE_TYPES):
            self.add_attachment(path)

    def _on_drop(self, event) -> None:
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        for path in paths:
            self.add_attachment(path)

    def add_attachment(self, path) -> None:
        path = Path(path)
        if not path.is_file():
            return
        if any(a["name"] == path.name for a in self.attachments):
            self.set_status(f"'{path.name}' is already attached")
            return
        try:
            content = extract(path)
        except BinaryFileError:
            messagebox.showwarning(
                "Can't read this file",
                f"'{path.name}' looks like a binary file, so there is no text "
                f"to send.\n\nAttachments work with {SUPPORTED_HINT}.")
            return
        except (ExtractionError, OSError) as exc:
            messagebox.showerror("Could not read file", f"{path.name}\n\n{exc}")
            return

        self.attachments.append({"name": path.name, "content": content})
        self._rebuild_chips()
        note = f"{len(content):,} characters"
        if len(content) > 20000:
            note += " — large, smaller models may struggle"
        self.set_status(f"Attached '{path.name}' ({note})")

    def _rebuild_chips(self) -> None:
        for widget in self.attach_bar.winfo_children():
            widget.destroy()
        if not self.attachments:
            self.attach_bar.pack_forget()
            return
        self.attach_bar.pack(fill="x", padx=14, pady=(6, 0))
        for attachment in self.attachments:
            chip = tk.Frame(self.attach_bar, bg=CHIP_BG, padx=6, pady=2)
            chip.pack(side="left", padx=(0, 6), pady=2)
            tk.Label(chip, text=f"📎 {attachment['name']}", font=FONT_SMALL,
                     bg=CHIP_BG, fg="#1f2430").pack(side="left")
            tk.Button(chip, text="✕", font=("Segoe UI", 8), bd=0, bg=CHIP_BG,
                      fg=MUTED, activebackground=CHIP_BG, cursor="hand2",
                      command=lambda n=attachment["name"]: self._remove_chip(n)
                      ).pack(side="left", padx=(4, 0))

    def _remove_chip(self, name: str) -> None:
        self.attachments = [a for a in self.attachments if a["name"] != name]
        self._rebuild_chips()

    # --------------------------------------------------------------- input
    def _show_placeholder(self) -> None:
        self.input.delete("1.0", tk.END)
        self.input.insert("1.0", PLACEHOLDER)
        self.input.config(fg=MUTED)
        self.placeholder_showing = True

    def _hide_placeholder(self) -> None:
        if self.placeholder_showing:
            self.input.delete("1.0", tk.END)
            self.input.config(fg="#1f2430")
            self.placeholder_showing = False

    def _restore_placeholder(self) -> None:
        if not self.input.get("1.0", tk.END).strip():
            self._show_placeholder()

    def _input_text(self) -> str:
        return "" if self.placeholder_showing else self.input.get("1.0", tk.END).strip()

    def _select_all_input(self, event=None):
        self.input.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _on_return(self, event):
        if event.state & 0x0001:      # Shift+Enter makes a new line
            return None
        self.send()
        return "break"

    # -------------------------------------------------------------- sending
    def send(self) -> None:
        if self.worker and self.worker.is_alive():
            self.set_status("Still working — press Stop first")
            return

        text = self._input_text()
        if not text and not self.attachments:
            return

        prompt = self._compose(text)
        shown = text or "(sent attached files)"
        footer = "  ".join(f"📎 {a['name']}" for a in self.attachments)

        targets = self.selected_targets()
        mode = self.current_mode()
        relay = parse_relay(text, self.available_targets()) if text else None

        if not relay and not targets:
            messagebox.showwarning(
                "No models selected",
                "Tick at least one model, or use a relay command like:\n\n"
                "    gemma2 ask llama3 what do you think?")
            return
        if not relay and mode != "chat" and len(targets) < 2:
            messagebox.showwarning(
                "Need more models",
                f"{MODES[mode].split(' — ')[0]} needs at least two models "
                f"selected so they have someone to work with.")
            return

        self._hide_placeholder()
        self.input.delete("1.0", tk.END)
        self.chat.add_user(shown, footer)
        self.attachments = []
        self._rebuild_chips()

        orchestrator = self._make_orchestrator(text)
        judge = self._judge_target()
        rounds = self._rounds()
        research = (self.research_var.get() and bool(text) and not relay)

        history = list(self.session.history(targets[0].key)) if targets else []

        learning = self.learning_var.get() and bool(text) and not relay
        primary = targets[0] if targets else None

        def job():
            final_prompt = prompt
            if research:
                block = self._research(text, primary, history)
                if block is None and self.cancel.is_set():
                    return
                if block:
                    final_prompt = f"{block}\n\n{prompt}"
            if relay:
                source, target, question = relay
                orchestrator.relay(source, target, question)
                return
            if mode == "debate":
                orchestrator.debate(targets, final_prompt, rounds, judge)
            elif mode == "critique":
                orchestrator.critique_chain(targets, final_prompt)
            elif mode == "judge":
                orchestrator.judge_panel(targets, final_prompt, judge)
            else:
                results = orchestrator.broadcast(targets, final_prompt)
                if learning and results and not self.cancel.is_set():
                    self._learn(text, results[0].text, primary)

        self._run(job)

    # ------------------------------------------------------------ learning
    def _learn(self, question: str, answer: str, target: Target | None) -> None:
        """Mine one exchange for durable lessons (worker thread)."""
        if target is None or not worth_learning_from(question, answer):
            return
        client = self.clients().get(target.server)
        if client is None or not client.base_url:
            return

        def note(text: str) -> None:
            self.events.put(("note", {"text": text, "note_id": "learning"}))

        try:
            reply = client.chat(
                target.model,
                [{"role": "user",
                  "content": build_extraction_prompt(question, answer)}],
                options={"temperature": 0.2, "num_predict": 200},
                cancel=self.cancel)
        except Exception:
            return                      # learning must never break a chat
        if self.cancel.is_set():
            return

        kept = self.knowledge.add_many(parse_lessons(reply.text), source=question)
        if kept:
            topics = ", ".join(sorted({lesson.topic for lesson in kept}))
            note(f"🧠 Learned {len(kept)} new thing(s) about {topics}")
            self.events.put(("note_close", {"note_id": "learning"}))
        self.events.put(("knowledge_changed", {}))

    def _research(self, question: str, target: Target | None,
                  history: list[dict]) -> str | None:
        """Run the SearXNG research step (worker thread). None on failure.

        Follow-ups are resolved against the conversation first, so "what is
        the 0-60mph?" searches for the car being discussed rather than for the
        definition of the term.
        """
        def note(text: str) -> None:
            self.events.put(("note", {"text": text, "note_id": "research"}))

        query = question
        if self.settings.get("smart_followups", True) and history:
            query = contextual_query(question, history,
                                     self._make_rewriter(target))
            if query != question:
                note(f"🔍 Searching for “{query}”")

        # Asking the same thing twice shouldn't cost another round of searching
        # and page fetches — but weather cached this morning is wrong tonight,
        # so the cache decides staleness per question.
        if self.settings.get("research_cache", True):
            hit = self.cache.get(query)
            if hit is not None:
                how = "" if hit.exact else f" for “{hit.query}”"
                note(f"🔍 Reusing research{how} from {hit.age_phrase}")
                self.events.put(("note_close", {"note_id": "research"}))
                return hit.block

        client = SearxngClient(str(self.settings.get("searxng_url", "")))
        try:
            block = gather(client, query, emit=note)
        except ResearchError as exc:
            note(f"⚠ Web research skipped — {exc}")
            self.events.put(("note_close", {"note_id": "research"}))
            return None

        if self.settings.get("research_cache", True):
            self.cache.put(query, block)
        self.events.put(("note_close", {"note_id": "research"}))
        return block

    def _make_rewriter(self, target: Target | None):
        """A tiny, cheap model call that turns a follow-up into a real query."""
        if target is None:
            return None

        def rewrite(prompt: str) -> str:
            client = self.clients().get(target.server)
            if client is None or not client.base_url:
                return ""
            result = client.chat(
                target.model, [{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 48},
                cancel=self.cancel)
            return result.text

        return rewrite

    def _compose(self, text: str) -> str:
        if not self.attachments:
            return text
        parts = [f"[Attached file: {a['name']}]\n```\n{a['content']}\n```"
                 for a in self.attachments]
        if text:
            parts.append(text)
        return "\n\n".join(parts)

    def _rounds(self) -> int:
        try:
            return max(1, min(6, int(self.rounds.get())))
        except ValueError:
            return 2

    def _make_orchestrator(self, question: str = "") -> Orchestrator:
        self.cancel = threading.Event()
        return Orchestrator(
            clients=self.clients(),
            session=self.session,
            emit=lambda kind, **payload: self.events.put((kind, payload)),
            cancel=self.cancel,
            options=self.settings.sampling_options(),
            budget_tokens=int(self.settings.get("context_budget_tokens", 6000)),
            system_prompt=self._system_prompt(question))

    def _system_prompt(self, question: str = "") -> str:
        """Persona, recalled lessons, the user's own prompt, then the slider."""
        parts = []
        if self.settings.get("conversational", True):
            parts.append(CONVERSATIONAL_PROMPT)
        if question and self.learning_var.get():
            parts.append(format_for_prompt(self.knowledge.recall(question)))
        parts.append(str(self.settings.get("system_prompt", "")).strip())
        parts.append(self.settings.style_directive())
        return "\n\n".join(part for part in parts if part)

    def _run(self, job) -> None:
        self.send_button.config(state="disabled")
        self.set_status("Working…")

        def worker():
            try:
                job()
            except Exception as exc:                     # last-resort guard
                self.events.put(("fatal", {"message": str(exc)}))
            finally:
                self.events.put(("finished", {}))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.cancel.set()
        self.set_status("Stopping — closing open connections…")
        self.chat.add_note("⏹ Stop requested.")

    # ---------------------------------------------------------- event pump
    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self.root.after(40, self._pump)

    def _handle(self, kind: str, payload: dict) -> None:
        if kind == "turn_start":
            target = payload["target"]
            self.chat.start_stream(payload["stream_id"], payload["heading"],
                                   speaker=target.key)
            self.set_status(f"{friendly_model_name(target.model)} is typing…")
        elif kind == "token":
            self.chat.append_stream(payload["stream_id"], payload["text"])
        elif kind == "turn_end":
            result = payload["result"]
            self.chat.end_stream(payload["stream_id"], result.text,
                                 meta=self._meta(result))
        elif kind == "turn_error":
            self.chat.fail_stream(payload["stream_id"],
                                  f"{payload['target'].label}: {payload['message']}")
        elif kind == "note":
            self.chat.add_note(payload["text"], payload.get("note_id"))
        elif kind == "note_close":
            self.chat.close_note(payload["note_id"])
        elif kind == "knowledge_changed":
            self._refresh_knowledge_link()
        elif kind == "fatal":
            self.chat.add_error(payload["message"])
        elif kind == "finished":
            self.send_button.config(state="normal")
            self.set_status("Ready")
            self._update_context_label()

    @staticmethod
    def _meta(result) -> str:
        bits = []
        speed = result.tokens_per_second
        if speed:
            bits.append(f"{speed:.0f} tok/s")
        if result.elapsed_s:
            bits.append(f"{result.elapsed_s:.1f}s")
        if result.cancelled:
            bits.append("stopped")
        return " · ".join(bits)

    # --------------------------------------------------------- chat files
    def new_chat(self) -> None:
        self.cancel.set()
        self.session = Session()
        self.chat.clear()
        self.chat.add_note("✨ New chat started.")
        self.set_status("New chat started")
        self._update_context_label()

    def save_chat(self) -> None:
        if self.session.is_empty():
            messagebox.showwarning("Nothing to save",
                                   "There is no chat history to save yet.")
            return
        path = self.session.path or filedialog.asksaveasfilename(
            title="Save chat", defaultextension=".json",
            filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            selected = [t.key for t in self.selected_targets()]
            saved = Path(path)
            saved.write_text(__import__("json").dumps(
                self.session.to_dict(selected), indent=2, ensure_ascii=False),
                encoding="utf-8")
            self.session.path = saved
            self.set_status(f"Saved: {saved}")
        except OSError as exc:
            messagebox.showerror("Error", f"Failed to save chat:\n{exc}")

    def load_chat(self) -> None:
        path = filedialog.askopenfilename(title="Open chat",
                                          filetypes=[("JSON files", "*.json")])
        if not path:
            return
        try:
            session, selected = Session.load(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Error", f"Failed to load chat:\n{exc}")
            return

        self.session = session
        for pair, var in self.model_vars.items():
            var.set(make_key(*pair) in set(selected))
        self._redraw()
        self.set_status(f"Loaded: {path}")
        self._update_context_label()

    def _redraw(self) -> None:
        self.chat.clear()
        for key, messages in self.session.conversations.items():
            server, model = split_key(key)
            self.chat.add_note(f"───  {model} ({server})  ───")
            for message in messages:
                role = message.get("role")
                content = message.get("content", "")
                if role == "user":
                    self.chat.add_user(content)
                elif role == "assistant":
                    self.chat.add_assistant(
                        friendly_model_name(model) or short_model(model),
                        content)
                else:
                    self.chat.add_note(content)

    # ------------------------------------------------------------- windows
    def open_settings(self) -> None:
        SettingsDialog(self.root, self.settings, on_saved=self._after_settings)

    def _after_settings(self) -> None:
        self.refresh_models()
        self._update_context_label()

    def open_knowledge(self) -> None:
        KnowledgeWindow(self.root, self.knowledge,
                        on_change=self._refresh_knowledge_link)

    def on_close(self) -> None:
        """Throwaway chats shouldn't leave cached research behind."""
        self.cancel.set()
        try:
            if self.session.path is not None:
                self.cache.keep_session()
            else:
                self.cache.discard_session()
        except Exception:
            pass
        self.root.destroy()

    def open_benchmark(self) -> None:
        targets = self.selected_targets() or self.available_targets()
        BenchmarkWindow(self.root, self.clients(), targets,
                        options=self.settings.sampling_options())


def main() -> None:
    root = BaseTk()
    ChatLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
