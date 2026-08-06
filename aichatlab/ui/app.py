"""The main window: model columns, orchestration modes and the chat panel."""

from __future__ import annotations

import queue
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from .. import __version__, gpu, intent, projectindex, projectmemory, recovery
from .. import edits as edit_tools
from .. import bridge, codetree, editdebug, suggest, testrun, validate
from ..activity import (
    BIG_PROMPT_TOKENS,
    Tracker,
    describe,
    human_duration,
    looks_urgent,
    stall_note,
)
from ..autopilot import Limits as AutoLimits
from ..autopilot import Run as AutoRun
from ..benchmark import Target as _Target  # noqa: F401  (re-export convenience)
from ..cache import ResearchCache
from ..checklist import Checklist, pipeline_for
from ..client import OllamaClient
from ..compaction import (
    COMPACT_AT,
    explain_no_compaction,
    should_compact,
    usage,
)
from ..compaction import apply as apply_compaction
from ..compaction import build_prompt as build_compaction_prompt
from ..compaction import plan as plan_compaction
from ..compaction import (
    remedy as compaction_remedy,
)
from ..config import CONVERSATIONAL_PROMPT, Settings
from ..documents import SUPPORTED_HINT, BinaryFileError, ExtractionError, extract
from ..folderscan import build_block, read_texts, select, survey, worth_holding
from ..formatting import friendly_model_name
from ..knowledge import (
    KnowledgeBase,
    build_extraction_prompt,
    format_for_prompt,
    parse_lessons,
    worth_learning_from,
)
from ..orchestrator import END_MARKER, MODES, Orchestrator, Target, parse_relay
from ..pulling import Progress, human_bytes, looks_like_model_name
from ..research import (
    ResearchError,
    SearxngClient,
    contextual_query,
    fetch_page_text,
    gather,
    looks_time_sensitive,
    wants_web_search,
)
from ..retrieval import Chunk, describe_selection
from ..retrieval import worth_retrieving as retrieval_worth
from ..retrieval import select as select_chunks
from ..runlog import RunLog
from ..session import (
    DENSE_CHARS_PER_TOKEN,
    Session,
    attachment_allowance,
    attachment_details,
    carries_attachment,
    describe_attachments,
    estimate_tokens,
    make_key,
    short_model,
    split_key,
)
from ..sizing import (
    find_loaded,
    fits,
    heavy_request,
    human_gb,
    observed_vram,
    placement_note,
    suggest_after_timeout,
)
from ..thinking import supports_think_api, think_value
from ..triage import Triage, describe_material, worth_triaging
from ..triage import build_prompt as build_triage_prompt
from ..triage import parse as parse_triage
from .benchmark_window import BenchmarkWindow
from .chatview import ChatView
from .dialogs import SettingsDialog
from .edit_debug_window import EditDebugWindow
from .edit_review import EditReview
from .folder_dialog import FolderScanDialog
from .knowledge_window import KnowledgeWindow
from .runlog_window import RunLogWindow
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
    WARN,
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
# How much of an attached folder the app will hold in memory beyond what one
# prompt can carry.  Text files, so 20 MB is hundreds of source files; the
# cap exists for the pathological folder, not the ordinary one.
KEEP_ON_HAND_BYTES = 20_000_000
MODE_ORDER = ("chat", "plan", "research", "converse", "debate",
              "critique", "judge")
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
        self.interjections: queue.Queue = queue.Queue()
        self.conversing = False
        self.skip_search_prompt = False
        self.skip_carry_prompt = False
        # Which Ollama each server runs.  A model that downloads but will not
        # start is nearly always a server too old for its architecture.
        self.server_versions: dict[str, str] = {}
        self.warned_old_servers: set[str] = set()
        self.auto = AutoRun(limits=AutoLimits(
            continues=int(self.settings.get("auto_continues", 20)),
            hours=float(self.settings.get("auto_hours", 8.0))))
        self.cards: list = []
        self.pending_auto = None
        self.listed_models: dict[str, tuple[list[str], str]] = {}
        self.pulling = ""
        self.pull_cancel = threading.Event()
        self.cards_checked_at = 0.0
        self.last_vram_note = ""
        # The folder this chat is about, if any — what makes project memory
        # and a hand-off note meaningful.
        self.project: Path | None = None
        # Staleness prompts are per-phrase, once per chat: helpful the first
        # time you ask about "the latest" something, nagging by the third.
        self.stale_asked: set[str] = set()
        # The project's files, kept as data so each question can be answered
        # with the handful that matter rather than the whole folder.
        self.project_texts: dict[str, str] = {}
        # Every readable file the scan found, not just the ones whose
        # contents fit the budget.  The difference matters: project_texts is
        # "what the model has read", project_files is "what exists" — and
        # only the second is allowed to answer existence questions.
        self.project_files: list[str] = []
        # Files whose *contents* have been in a prompt this chat.  Distinct
        # from project_texts (what the app holds) — this is what the model
        # has actually read, which is the honest basis for "it was working
        # from the name alone" messages and for tie-breaking bare filenames.
        self.sent_files: set[str] = set()
        self.project_index = None
        # Consent is per chat: proposed edits wait for a decision, and the
        # "stop asking" choice dies with the conversation that made it.
        self.pending_edits: dict = {}
        # What each pending set would break, keyed the same way, so the
        # review window can put it above the diff.
        self.pending_flaws: dict = {}
        # Changes the model described in prose, waiting on the user to say
        # whether they are worth going back and asking for properly.
        self.pending_changes: dict = {}
        self.last_edit_target = None
        self.auto_apply_edits = False
        self.warned_edit_room = False
        self.warned_no_folder = False
        self.last_question = ""
        # Changes waiting their turn: one is asked for, decided,
        # and only then is the next one requested.
        self.change_queue: list = []
        self.change_target = None
        # The rest of an offered menu, waiting for "want another?".
        self.menu_left: list = []
        self.menu_target = None
        self.menu_action_id = ""
        # The order models were ticked in, oldest first — the critique
        # chain reads roles from it.
        self.model_tick_order: list = []
        # Why the edit check reached the answer it did.  Survives a new chat
        # deliberately: "it stopped working at some point this afternoon" is
        # a question about the whole session, not the current conversation.
        self.edit_debug = editdebug.EditDebug()
        # Started at the end of __init__, once there is a window to drive and
        # a transcript to announce it in.  Off unless asked for: see bridge.py.
        self.bridge = None
        self.action_seq = 0
        self.pending_selection: set[str] = set()
        self.pump_id: str | None = None
        self.tick_id: str | None = None
        self.activity = Tracker()
        self.runlog = RunLog()
        self.queued_message = ""
        self.cutoff_streak: dict[str, int] = {}
        self.stopping_since: float | None = None
        self.checklist: Checklist | None = None
        self.said_stalls: set[str] = set()
        self.model_sizes: dict[str, int] = {}
        self.vram_seen = 0
        self.attachment_names: list[str] = []
        self.last_autosave = 0.0
        self.cache = ResearchCache(
            base_ttl_minutes=int(self.settings.get(
                "research_cache_ttl_minutes", 360)))

        root.title(f"Ensemble AI Studio {__version__}")
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
        self._tick()
        self._offer_recovery()
        self._check_cards()
        self._start_bridge()

    def _start_bridge(self) -> None:
        """Let a script drive this window, when the user has asked for that."""
        if not bridge.enabled(self.settings):
            return
        try:
            self.bridge = bridge.Bridge(self)
            port = self.bridge.start()
        except OSError as exc:
            self.bridge = None
            self.chat.add_note(f"⚠ Control bridge could not start: {exc}")
            return
        self.chat.add_note(
            f"🔌 Control bridge listening on 127.0.0.1:{port} — this window "
            f"can be driven from a script while it is open. Every command "
            f"goes through the same buttons you would press, so nothing is "
            f"written without the usual review.")
        self._hold_bridge_handshake()

    def _hold_bridge_handshake(self) -> None:
        """Keep the handshake pointing here while this window is the live one.

        A second window takes the file over, and if it is killed rather than
        closed it leaves the file naming a dead port — so this one becomes
        unreachable despite being open and fine.  Re-checking on a slow timer
        makes that repair itself.
        """
        if self.bridge is None:
            return
        try:
            self.bridge.reassert()
        except Exception:                                  # noqa: BLE001
            pass
        try:
            self.root.after(5000, self._hold_bridge_handshake)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ UI
    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=PANEL, padx=14, pady=10)
        header.pack(fill="x")
        separator(self.root).pack(fill="x")

        tk.Label(header, text="🎼 Ensemble AI Studio", font=FONT_TITLE, bg=PANEL,
                 fg="#1f2430").pack(side="left")

        for text, command in (
            ("⚙ Settings", self.open_settings),
            ("📊 Benchmark", self.open_benchmark),
            ("📜 Run log", self.open_runlog),
            ("✏ Edit log", self.open_edit_debug),
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

        # The hardware line.  A setting that only ever surfaced as a one-time
        # prompt is a setting with no state you can check — the first question
        # anyone asks about it is "is it on?", and there was nowhere to look.
        hardware = tk.Frame(sidebar, bg=PANEL)
        hardware.pack(fill="x", padx=12, pady=(0, 2))
        self.gpu_label = tk.Label(hardware, text="🎛 Looking for graphics "
                                                 "cards…", font=FONT_SMALL,
                                  fg=MUTED, bg=PANEL, justify="left",
                                  anchor="w")
        self.gpu_label.pack(anchor="w")
        # Second row: the sidebar is 390px and one long line loses its own
        # ending, taking the link with it.
        spread_row = tk.Frame(hardware, bg=PANEL)
        spread_row.pack(anchor="w", fill="x")
        self.spread_label = tk.Label(spread_row, text="", font=FONT_SMALL,
                                     fg=MUTED, bg=PANEL, anchor="w")
        self.spread_label.pack(side="left")
        self.spread_link = link_button(spread_row, "", self._spread_clicked)
        self.spread_link.pack(side="left", padx=(6, 0))
        self.spread_link.pack_forget()

        tk.Label(sidebar, text=("Tip: relay with  gemma2 ask llama3 <question>\n"
                                "Conversation, Debate and Judge need 2+ models."),
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

        # Its own row: four links across a 175px column clips the last one,
        # and the one that gets clipped is the only one nobody expects to
        # find here.  The pull happens on the server, so this works from a
        # machine whose own outbound access is blocked and for a container
        # with no shell.
        adder = tk.Frame(frame, bg=PANEL)
        adder.pack(fill="x", padx=6)
        link_button(adder, "＋ Add a model",
                    lambda: self.add_model(server)).pack(side="left")

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
            bar, textvariable=self.mode, state="readonly", width=46,
            values=[MODES[key] for key in MODE_ORDER], font=FONT_SMALL)
        self.mode_box.pack(side="left")
        self.mode_box.bind("<<ComboboxSelected>>", lambda e: self._sync_mode_bar())
        # Critique chain gives each model a role by tick order — this is
        # where the screen says so, one pick at a time.
        self.chain_hint = tk.Label(bar, text="", font=FONT_SMALL, bg=PANEL,
                                   fg=MUTED)
        self.chain_hint.pack(side="left", padx=(10, 0))
        self._last_mode = "chat"

        self.rounds_label = tk.Label(bar, text="Rounds", font=FONT_SMALL,
                                     bg=PANEL, fg=MUTED)
        self.rounds = tk.StringVar(value="2")
        self.rounds_box = tk.Spinbox(bar, from_=1, to=6, width=3, font=FONT_SMALL,
                                     textvariable=self.rounds)

        self.turns_label = tk.Label(bar, text="Max turns", font=FONT_SMALL,
                                    bg=PANEL, fg=MUTED)
        self.turns = tk.StringVar(value="12")
        self.turns_box = tk.Spinbox(bar, from_=2, to=60, width=4,
                                    font=FONT_SMALL, textvariable=self.turns)

        self.judge_label = tk.Label(bar, text="Judge", font=FONT_SMALL, bg=PANEL,
                                    fg=MUTED)
        self.judge = tk.StringVar(value="(first selected)")
        self.judge_box = ttk.Combobox(bar, textvariable=self.judge,
                                      state="readonly", width=24, font=FONT_SMALL)

        self.context_label = tk.Label(bar, text="", font=FONT_SMALL, bg=PANEL,
                                      fg=MUTED)
        self.context_label.pack(side="right")
        self.compact_button = link_button(bar, "🗜 Compact", self.compact_chat)
        self.compact_button.pack(side="right", padx=(0, 8))

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
        self.learning_wanted = bool(self.settings.get("learning", False))
        self.learning_check = tk.Checkbutton(
            row2, text="🧠 Learning", variable=self.learning_var,
            font=FONT_SMALL, bg=PANEL, activebackground=PANEL, fg="#1f2430",
            highlightthickness=0, bd=0, cursor="hand2",
            command=self._on_learning_toggle)
        self.learning_check.pack(side="left", padx=(12, 0))

        # Reasoning models are on by default because that is what they are
        # for; the toggle exists because reasoning is slow and sometimes you
        # only want the answer.
        self.thinking_var = tk.BooleanVar(
            value=bool(self.settings.get("thinking", True)))
        self.thinking_check = tk.Checkbutton(
            row2, text="💭 Reasoning", variable=self.thinking_var,
            font=FONT_SMALL, bg=PANEL, activebackground=PANEL, fg="#1f2430",
            highlightthickness=0, bd=0, cursor="hand2",
            command=self._on_thinking_toggle)
        self.thinking_check.pack(side="left", padx=(12, 0))

        # Unattended running.  Off by default: it presses buttons on the
        # user's behalf, and that should always be something they asked for.
        self.auto_var = tk.BooleanVar(value=False)
        self.auto_check = tk.Checkbutton(
            row2, text="🌙 Auto", variable=self.auto_var,
            font=FONT_SMALL, bg=PANEL, activebackground=PANEL, fg="#1f2430",
            highlightthickness=0, bd=0, cursor="hand2",
            command=self._on_auto_toggle)
        self.auto_check.pack(side="left", padx=(12, 0))

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
        # A plain bool the worker can read.  `_learn` runs off the main
        # thread, and touching a Tk variable from there is a Tcl call off the
        # main thread — the thing that used to abort the interpreter.
        self.learning_wanted = bool(self.learning_var.get())
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

        flat_button(row, "📎", self.import_files).pack(side="left", padx=(0, 4),
                                                       anchor="s")
        flat_button(row, "📁", self.import_folder).pack(side="left", padx=(0, 8),
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

    # ---------------------------------------------------- adding models
    def add_model(self, server: str) -> None:
        """Ask for a name and have the server fetch it."""
        if not self.settings.base_url(server):
            messagebox.showinfo(
                "Set the address first",
                f"The {server} server has no address configured. Enter it in "
                f"⚙ Settings and press Connect, then try again.")
            self.open_settings()
            return
        if self.pulling:
            self.set_status(f"Already downloading {self.pulling}")
            return

        name = simpledialog.askstring(
            "Add a model",
            "Model to download, as Ollama names it:\n\n"
            "    qwen3:8b          a small reasoning model\n"
            "    qwen2.5:14b       q4, fits an 11 GB card\n"
            "    gemma3:12b\n\n"
            "The download runs on the server, not on this machine.",
            parent=self.root, initialvalue="qwen3:8b")
        if name is None:
            return
        name = name.strip()
        if not looks_like_model_name(name):
            messagebox.showwarning(
                "That does not look like a model name",
                "Use the name and tag Ollama uses, for example qwen3:8b.")
            return

        self.pulling = name
        self.pull_cancel = threading.Event()
        self.chat.add_note(f"⬇ Asking the {server} server to download "
                           f"{name}…", note_id="pull")
        self.set_status(f"Downloading {name} on the {server} server…")
        threading.Thread(target=self._pull_worker,
                         args=(server, name), daemon=True).start()

    def _pull_worker(self, server: str, name: str) -> None:
        """Worker thread — progress goes back through the queue, never Tk."""
        client = self.clients().get(server)
        if client is None:
            self.events.put(("pull_done", {"server": server, "model": name,
                                           "error": "no such server"}))
            return

        progress = Progress()
        last = [0.0]

        def on_progress(payload: dict) -> None:
            now = time.monotonic()
            progress.update(payload, now)
            # Ollama emits these faster than anything can usefully be read.
            if now - last[0] < 0.4 and not progress.done:
                return
            last[0] = now
            self.events.put(("pull_progress", {
                "text": progress.describe(name, now)}))

        error = client.pull(name, on_progress=on_progress,
                            cancel=self.pull_cancel)
        self.events.put(("pull_done", {"server": server, "model": name,
                                       "error": error,
                                       "size": progress.total}))

    def _pull_progress(self, text: str) -> None:
        self.chat.add_note(text, note_id="pull")
        self.set_status(text.replace("⬇ ", ""))

    def _pull_done(self, server: str, model: str, error: str,
                   size: int = 0) -> None:
        self.pulling = ""
        if error == "cancelled":
            self.chat.add_note(f"⏹ Download of {model} stopped.",
                               note_id="pull")
        elif error:
            self.chat.add_note(f"⚠ Could not download {model}: {error}",
                               note_id="pull")
            self.set_status(f"Download failed: {error[:120]}")
        else:
            self.chat.add_note(
                f"✅ {model} downloaded to the {server} server"
                + (f" ({human_bytes(size)})." if size else "."),
                note_id="pull")
            self.set_status(f"{model} is ready")
        self.chat.close_note("pull")
        # The list is stale either way: a partial pull leaves nothing behind,
        # a finished one leaves a model the sidebar has never heard of.
        self.refresh_models(server)

    # ------------------------------------------------ graphics cards
    def _check_cards(self) -> None:
        """Find out what is actually in the machine, off the main thread."""
        def probe() -> None:
            try:
                cards = gpu.probe()
            except Exception:
                cards = []
            self.events.put(("cards", {"cards": cards}))

        threading.Thread(target=probe, daemon=True).start()

    def _refresh_gpu_label(self) -> None:
        """Always say what was found and whether spreading is on.

        Including when the answer is "nothing" — a silent probe that finds no
        cards is indistinguishable from one that never ran, and the difference
        matters when someone is looking for a setting that never appeared.
        """
        if not self.cards:
            self.gpu_label.config(
                text="🎛 No NVIDIA cards detected", fg=MUTED)
            self.spread_label.config(text="nvidia-smi not found", fg=MUTED)
            self.spread_link.pack_forget()
            return

        free = gpu.free_vram(self.cards)
        summary = gpu.short_describe(self.cards)
        if gpu.looks_occupied(self.cards):
            # Total capacity is the number everyone quotes and the wrong one
            # to plan with.  What decides where a model runs is what is free.
            summary += f" · {gpu.human_vram(free)} free"
        self.gpu_label.config(
            text=f"🎛 {summary}",
            fg=WARN if gpu.looks_occupied(self.cards) else MUTED)
        if len(self.cards) < 2:
            self.spread_label.config(text="", fg=MUTED)
            self.spread_link.pack_forget()
            return

        on = gpu.spread_enabled()
        self.spread_label.config(
            text=f"spread across both: {'on' if on else 'OFF'}",
            fg=OK if on else WARN)
        if on:
            self.spread_link.pack_forget()
        else:
            self.spread_link.config(text="turn on")
            self.spread_link.pack(side="left", padx=(6, 0))

    def _spread_clicked(self) -> None:
        worked, message = gpu.enable_spread()
        self.chat.add_note(f"{'🎛' if worked else '⚠'} {message}")
        self._refresh_gpu_label()

    def _refresh_cards_soon(self) -> None:
        """Re-read the cards, at most now and then.

        Free VRAM is not a property of the machine — it is whatever some
        other process happened to leave behind, and it changes between one
        message and the next.
        """
        now = time.monotonic()
        if now - self.cards_checked_at < 20:
            return
        self.cards_checked_at = now
        self._check_cards()

    def _cards_found(self, cards: list) -> None:
        self.cards = cards
        self._refresh_gpu_label()
        if not cards:
            return
        # Real numbers beat the lower bound `sizing.observed_vram` infers from
        # watching a load go badly.
        before = self.vram_seen
        self.vram_seen = max(self.vram_seen, gpu.largest_card(cards))
        if self.vram_seen != before:
            # The "will it fit" colours beside each model were guesses until
            # now; repaint them with the real capacity.
            for server, (names, hint) in list(self.listed_models.items()):
                if names or hint:
                    self._populate(server, names, hint)
            self._apply_selection()
        if not gpu.worth_offering(cards):
            return
        # Somebody else is holding the cards, and that is a better
        # explanation of a slow reply than anything else the app can say.
        note = gpu.free_note(cards)
        if note and note != self.last_vram_note:
            self.last_vram_note = note
            self.chat.add_note(note, note_id="vram")

        if self.settings.get("skip_spread_prompt"):
            return

        self.action_seq += 1
        action_id = f"spread{self.action_seq}"
        self.chat.add_action(
            gpu.offer_note(cards),
            [("Turn it on", lambda: self._enable_spread(action_id), True),
             ("Leave it alone", lambda: self._decline_spread(action_id), False)],
            action_id=action_id)

    def _enable_spread(self, action_id: str) -> None:
        worked, message = gpu.enable_spread()
        self.chat.resolve_action(
            action_id, f"{'🎛' if worked else '⚠'} {message}")
        self._refresh_gpu_label()

    def _decline_spread(self, action_id: str) -> None:
        self.settings["skip_spread_prompt"] = True
        self.settings.save()
        self.chat.resolve_action(
            action_id,
            f"🎛 Left alone — set {gpu.SPREAD_VAR}=1 yourself if you change "
            f"your mind. This will not be asked again.")

    def _on_auto_toggle(self) -> None:
        """Auto mode is a promise to keep going, so say what it will do."""
        if not self.auto_var.get():
            self._end_auto_run()
            self.set_status("Auto mode off — cut-off replies will ask you")
            return
        self.auto.begin(time.monotonic())
        limits = self.auto.limits
        self.chat.add_note(
            f"🌙 Auto mode on. Cut-off replies will be continued without "
            f"asking, up to {limits.continues} times per message, and the "
            f"questions this app normally stops on will be answered with "
            f"their sensible default and logged. It stops on its own after "
            f"{limits.hours:g} hours, if a continuation stops adding "
            f"anything new, if the model starts repeating itself, or if the "
            f"context fills with nothing left to compact.")
        self.set_status("Auto mode on — it will keep going without you")

    def _end_auto_run(self) -> None:
        """Report what happened while nobody was watching."""
        summary = self.auto.summary(time.monotonic())
        if summary:
            self.chat.add_note(summary)
        self.auto.begin(time.monotonic())

    def _on_thinking_toggle(self) -> None:
        self.settings["thinking"] = bool(self.thinking_var.get())
        self.settings.save()
        self.set_status("Reasoning shown and allowed for"
                        if self.thinking_var.get() else
                        "Reasoning off — models will answer directly")

    def _options(self, num_predict: int, prompt_tokens: int = 0) -> dict:
        """Sampling options for a side call — same window as the main one.

        num_ctx is part of the model's runner configuration, so a helper call
        that omits it makes Ollama reload the model with its default window,
        and the next real request reloads it back.  On a 32B that is twenty
        gigabytes off disk, twice, for a sixty-token question.
        """
        options = dict(self.settings.sampling_options(prompt_tokens))
        options["num_predict"] = num_predict
        return options

    def clients(self, prompt_tokens: int = 0) -> dict[str, OllamaClient]:
        # How long the server may say nothing.  It says nothing for the whole
        # time it is reading the prompt, so the allowance has to cover that.
        timeout = self.settings.timeout_for(prompt_tokens,
                                            self.runlog.prompt_rate())
        return {server: OllamaClient(self.settings.base_url(server),
                                     server=server, timeout=timeout,
                                     server_version=self.server_versions.get(
                                         server, ""))
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
        # Entering critique chain with models already ticked: their roles
        # would be whatever order they happen to sit in the list, which
        # nothing on screen says.  Offer a clean slate so the order is
        # chosen, not inherited.
        if (mode == "critique" and self._last_mode != "critique"
                and self.selected_targets()):
            if messagebox.askyesno(
                    "Set up the critique chain",
                    "Critique chain assigns roles by the order you tick "
                    "models: first drafts, second reviews, third (optional) "
                    "rewrites.\n\nUncheck everything so you can pick the "
                    "roles in order?"):
                for key, var in self.model_vars.items():
                    var.set(False)
                self.model_tick_order = []
                self._update_context_label()
        self._last_mode = mode
        self._update_chain_hint()
        for widget in (self.rounds_label, self.rounds_box, self.judge_label,
                       self.judge_box, self.turns_label, self.turns_box):
            widget.pack_forget()
        if mode == "converse":
            self.turns_label.pack(side="left", padx=(14, 4))
            self.turns_box.pack(side="left")
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
        # Free VRAM changes far more often than the model list does, and
        # Refresh is what someone presses when they want the sidebar to tell
        # them the truth about right now.
        self.cards_checked_at = 0.0
        self._refresh_cards_soon()
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
        """Worker thread — results go back through the queue, never via Tk.

        `root.after` looks like a safe way to hop back to the main thread, but
        it registers a Tcl command as it goes, so calling it off-thread is a
        Tcl call off-thread.  It usually gets away with it and occasionally
        aborts the interpreter outright.  Everything here goes through
        `self.events` instead, which `_pump` drains on the main thread.
        """
        client = self.clients()[server]
        try:
            names = client.list_models()
        except Exception as exc:
            self.events.put(("models_failed", {
                "server": server, "message": str(exc),
                "interactive": interactive}))
            return
        self.events.put(("models_loaded", {
            "server": server, "names": names,
            "sizes": client.model_sizes(),
            "version": client.version()}))

    def _models_failed(self, server: str, message: str, interactive: bool) -> None:
        self._set_column_status(server, "● offline", ERROR)
        self.set_status(f"Could not reach the {server} server: {message[:120]}")
        if interactive:
            messagebox.showerror(
                "Connection failed",
                f"Could not fetch models from the {server} server:\n\n{message[:300]}")

    def _models_loaded(self, server: str, names: list[str],
                       version: str = "") -> None:
        self._populate(server, names)
        self._apply_selection()
        if version:
            self.server_versions[server] = version
        label = f"● {len(names)} model(s)"
        if version:
            label += f"  ·  Ollama {version}"
        self._set_column_status(server, label, OK)
        self.set_status(f"Loaded {len(names)} {server} model(s)")
        self._sync_judges()

    def _populate(self, server: str, names: list[str], hint: str = "") -> None:
        # `model_vars` is keyed by (server, name), so a repeated name would
        # build two rows sharing one variable — the second silently orphaning
        # the first.  Ollama should never send duplicates; this costs nothing
        # and means it does not matter if one does.
        names = list(dict.fromkeys(names))
        self.listed_models[server] = (list(names), hint)
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
            row = tk.Frame(inner, bg=PANEL)
            row.pack(anchor="w", fill="x", padx=2)
            tk.Checkbutton(row, text=friendly_model_name(name) or name,
                           variable=var, anchor="w",
                           justify="left", font=FONT, bg=PANEL,
                           activebackground=PANEL, highlightthickness=0, bd=0,
                           cursor="hand2", wraplength=118,
                           command=lambda key=(server, name):
                               self._on_model_tick(key)).pack(side="left")
            size = self.model_sizes.get(name, 0)
            if size:
                # Size is the number that decides whether this runs on the
                # graphics card or crawls along on the processor, and it was
                # sitting unused in /api/tags all along.
                verdict = fits(size, self.vram_seen)
                tk.Label(row, text=human_gb(size), font=FONT_SMALL, bg=PANEL,
                         fg=MUTED if verdict is None else (OK if verdict else WARN)
                         ).pack(side="right", padx=(2, 4))
            self.model_vars[(server, name)] = var

    def _on_model_tick(self, key) -> None:
        """One checkbox changed — keep the tick-order and the hint honest.

        The order matters now: in a critique chain the first model ticked
        drafts, the second reviews, the third (if any) rewrites.  Before this
        the roles came from list position, which nothing on screen said and
        nobody would guess.
        """
        var = self.model_vars.get(key)
        if var is not None and var.get():
            if key not in self.model_tick_order:
                self.model_tick_order.append(key)
        else:
            try:
                self.model_tick_order.remove(key)
            except ValueError:
                pass
        self._update_context_label()
        self._update_chain_hint()

    def _ordered_targets(self, targets) -> list:
        """Selected targets, in the order they were ticked.

        Anything ticked before the tracking existed (a restored session)
        keeps its list position, after the tracked ones.
        """
        order = {key: n for n, key in enumerate(self.model_tick_order)}
        return sorted(targets, key=lambda t: order.get(
            (t.server, t.model), len(order) + 1))

    def _update_chain_hint(self) -> None:
        """Walk the user through the roles, one tick at a time."""
        if self.current_mode() != "critique":
            self.chain_hint.config(text="")
            return
        picked = self._ordered_targets(self.selected_targets())
        if not picked:
            text, colour = "1st pick = drafter", MUTED
        elif len(picked) == 1:
            text = f"{picked[0].short} drafts — now pick the reviewer"
            colour = WARN
        elif len(picked) == 2:
            text = f"Ready: {picked[0].short} drafts, {picked[1].short} reviews"
            colour = OK
        elif len(picked) == 3:
            text = (f"Ready: {picked[0].short} drafts, {picked[1].short} "
                    f"reviews, {picked[2].short} rewrites")
            colour = OK
        else:
            text = (f"Only the first three are used — "
                    f"{len(picked) - 3} ticked model(s) would sit idle")
            colour = WARN
        self.chain_hint.config(text=text, fg=colour)

    def _set_column_status(self, server: str, text: str, colour: str) -> None:
        self.columns[server]["status"].config(text=text, fg=colour)

    def _set_all(self, server: str, value: bool) -> None:
        for (srv, model), var in self.model_vars.items():
            if srv == server:
                var.set(value)
                if not value:
                    try:
                        self.model_tick_order.remove((srv, model))
                    except ValueError:
                        pass
                elif (srv, model) not in self.model_tick_order:
                    self.model_tick_order.append((srv, model))
        self._update_context_label()
        self._update_chain_hint()

    def _update_context_label(self) -> None:
        """Say how full the context is *and* what happens when it fills.

        A bare "24,259/28,000" is a number, not a warning — it does not tell
        you that the next long answer starts silently deleting the beginning
        of the conversation.  So the label changes colour as it fills and
        states the consequence.

        It also names the model when only one is selected, because the count
        is per-model and the transcript is not.  Reopen a chat and the screen
        shows every model's conversation while this counts one of them; "≈ 85
        tokens" underneath a screen full of source code reads as a bug until
        you know whose 85 tokens they are.
        """
        targets = self.selected_targets()
        if not targets:
            self.context_label.config(text="")
            return
        used = max((self.session.total_tokens(t.key) for t in targets), default=0)
        budget = self.settings.effective_budget()
        share = used / budget if budget else 0

        whose = ""
        if len(targets) == 1:
            name = (friendly_model_name(targets[0].model)
                    or short_model(targets[0].model))
            whose = f"{name}: "
        text = f"{whose}context ≈ {used:,}/{budget:,} tokens"
        colour = MUTED
        # Only recommend compacting when compacting would actually do
        # something.  Advising it for a conversation that is one big pinned
        # attachment sends the user to a button that refuses, which reads as
        # the app arguing with itself.
        advice = compaction_remedy(self.session.history(targets[0].key),
                                   budget)
        if share >= 1:
            text += f" — oldest turns are being dropped, {advice}"
            colour = ERROR
        elif share >= 0.9:
            text += f" — nearly full, {advice}"
            colour = ERROR
        elif share >= COMPACT_AT:
            text += f" — filling up, {advice}"
            colour = WARN
        self.context_label.config(text=text, fg=colour)

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

    def import_folder(self) -> None:
        path = filedialog.askdirectory(title="Add a folder", mustexist=True)
        if path:
            self.add_folder(path)

    def add_attachment(self, path) -> None:
        path = Path(path)
        # Dropping a folder on the window used to do nothing at all — no
        # attachment, no error, no status line.
        if path.is_dir():
            self.add_folder(path)
            return
        if not path.exists():
            self.set_status(f"Could not find '{path.name}'")
            return
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
        self._adopt_project(path.parent)
        self._rebuild_chips()
        note = f"{len(content):,} characters"
        if len(content) > 20000:
            note += " — large, smaller models may struggle"
        self.set_status(f"Attached '{path.name}' ({note})")

    # ------------------------------------------------------- editing files
    def _adopt_project(self, folder) -> None:
        """Make attached files editable, without widening what may be written.

        Editing used to arm only for a whole folder, so dragging in three
        files and asking for changes produced prose — correct, and baffling.
        The folder those files came from becomes the boundary: the same rule
        as an attached folder, reached a different way.
        """
        folder = Path(folder)
        if self.project is not None:
            return                      # a real project already wins
        self.project = folder
        self.chat.add_note(
            f"✏ Files from {folder.name} can be edited — changes will be "
            f"shown as a diff for you to approve before anything is written.")

    # -- proposing and applying -----------------------------------------
    def _check_for_edits(self, reply: str, truncated: bool = False,
                         target=None, second_pass: bool = False) -> None:
        """Offer to write what the model proposed — never write it silently.

        `truncated` comes from the server: it is the difference between a
        model that forgot a closing marker and one that was interrupted, and
        the two look identical in the text.

        `second_pass` marks a reply that came back from asking for one change
        on its own.  It stops the fallback offering to go round again, which
        would otherwise be a loop with a button on it.

        Every path out of here records a trace.  The one complaint this
        feature attracts — "it said it would change the file and nothing
        happened" — covers about six different failures that look identical
        from the chat window, and the trace is what tells them apart.
        """
        trace = editdebug.begin(
            reply, project=self.project, truncated=truncated,
            second_pass=second_pass,
            had_target=(target or self.last_edit_target) is not None)
        if self.project is None:
            self.edit_debug.record(trace.done("no-folder"))
            return
        if not self.settings.get("allow_edits", True):
            self.edit_debug.record(trace.done("edits-off"))
            return
        # A menu before anything else: a reply shaped as candidates is an
        # answer to "what could be improved", and reading it for edit blocks
        # would find nothing and file it as a failure.
        if not second_pass:
            menu = suggest.parse_menu(
                reply, self.project_files or list(self.project_texts))
            kept, dropped = suggest.verify(menu, self.project_texts)
            if kept:
                for candidate, why in dropped:
                    self.chat.add_note(
                        f"✂ Dropped a suggestion for {candidate.file} — "
                        f"{why}.", tag="checklist")
                self._offer_suggestions(
                    kept, target or self.last_edit_target)
                self.edit_debug.record(trace.done("menu-offered"))
                return

        proposed, rejected = edit_tools.parse(reply, truncated=truncated)
        trace.step("whole-file blocks parsed", proposed)
        allowed, refused = edit_tools.check(self.project, proposed)
        trace.step("survived the safety check", allowed)
        rejected = list(rejected) + list(refused)

        # Anchored edits: the model wrote only the lines that change, and the
        # app works out what the file becomes.  This is the form that works
        # on a 14B, because reproducing a 200-line file is not asked of it.
        patches, bad = edit_tools.parse_patches(reply, truncated=truncated)
        trace.step("anchored patches parsed", patches,
                   detail=f"{len(bad)} malformed" if bad else "")
        # `known` is what the model has actually read this chat.  Falling
        # back to the loaded set keeps the tiebreak working for chats where
        # nothing has tracked sends (an adopted project, an old session).
        seen = sorted(self.sent_files) or list(self.project_texts) or None
        patched, unmatched = edit_tools.resolve_patches(
            self.project, patches, known=seen)
        trace.step("patches anchored to a real file", patched,
                   detail=f"{len(seen or [])} file(s) the model had read")
        rejected += list(bad) + list(unmatched)
        known = {e.name for e in allowed}
        allowed += [e for e in patched if e.name not in known]
        before_identical = list(allowed)
        # A file the model "changed" into exactly what it already is is not a
        # change, and offering it wastes the one decision this feature asks
        # the user to make.
        allowed = [e for e in allowed
                   if not edit_tools.unchanged(self.project, e)]
        identical = [e for e in before_identical if e not in allowed]
        trace.step("changes that differ from disk", allowed,
                   detail=(f"{len(identical)} were already applied"
                           if identical else ""))
        trace.refused(rejected)
        for item in rejected:
            self.chat.add_note(f"⚠ {item.name} — {item.reason}", tag="checklist")
        # A refused edit is worth a second attempt whether or not some other
        # edit in the same reply happened to land.  Gating this on "nothing
        # at all got through" meant one lucky block — often a brand-new file,
        # the easiest kind to produce and the least likely to be wanted —
        # silently cancelled the retry for every edit that failed beside it.
        if rejected and not second_pass:
            self._offer_to_retry(rejected, target)
        if not allowed:
            if not rejected:
                trace.claimed_to_write = self._contradict_pretend(reply)
            fallback = ""
            if not second_pass:
                fallback = self._offer_to_make_changes(reply, target)
            # Order matters: a refusal is the most actionable thing that can
            # have happened, and a change that is already on disk is not a
            # failure at all.  Only when neither applies does how the prose
            # fallback went become the story.
            if rejected:
                verdict = "all-rejected"
            elif identical:
                verdict = "no-change"
            elif fallback == "offered":
                verdict = "prose-offered"
            elif fallback == "no-target":
                verdict = "no-target"
            else:
                verdict = "prose-unmatched"
            self.edit_debug.record(trace.done(verdict))
            if second_pass:
                # This one produced no diff to decide on, so nothing else is
                # going to advance the queue. Stopping here would strand every
                # change behind a request that quietly failed.
                self._advance_change_queue()
            return

        # Does the file still work afterwards?  The review window can show
        # that a change is the one you asked for; it cannot show that the
        # result runs, and a five-line patch that reads perfectly is exactly
        # how a broken file gets approved.
        flaws = {}
        for edit in allowed:
            found = validate.introduced(
                edit.name, edit_tools.read_current(self.project, edit.name),
                edit.new_text)
            if found:
                flaws[edit.name] = found
                trace.broke(edit.name, found)

        summary = edit_tools.summarise(self.project, allowed)
        # "Stop asking me in this chat" was permission to skip a review, not
        # permission to write something that does not compile.  A file that
        # would break goes back in front of the user however tired of clicking
        # they are.
        if self.auto_apply_edits and not flaws:
            self.edit_debug.record(trace.done("auto-applied"))
            self._apply_edits(allowed, always=True)
            return
        if self.auto_apply_edits and flaws:
            self.chat.add_note(
                f"⚠ Asking anyway: {validate.summarise(next(iter(flaws.values())))} "
                f"— automatic applying is paused for this one.")

        self.action_seq += 1
        action_id = f"edit{self.action_seq}"
        names = ", ".join(e.name for e in allowed[:6])
        if len(allowed) > 6:
            names += f", and {len(allowed) - 6} more"
        self.pending_edits[action_id] = allowed
        self.pending_flaws[action_id] = flaws
        broken = ""
        if flaws:
            # Say it on the row, not only inside the window.  This is the one
            # thing that should change whether the window gets opened at all.
            broken = ("\n⚠ " + "; ".join(
                f"{name} — {validate.summarise(found)}"
                for name, found in list(flaws.items())[:3]))
        self.chat.add_action(
            f"✏ Proposed changes to {Path(self.project).name}: {summary} "
            f"— {names}. Nothing has been written yet.{broken}",
            [("Review the changes", lambda: self._review_edits(action_id), True),
             ("Discard", lambda: self._discard_edits(action_id), False)],
            action_id=action_id)
        self.set_status(f"Waiting on you — {summary}")
        self.edit_debug.record(trace.done("offered"))
        # The row in the chat is the record; the window is the request.  A
        # button that has to be found and clicked before anything can be
        # approved is the difference between "it showed me the changes" and
        # "it listed them and did nothing", which is the same sentence users
        # reach for when the pipeline fails outright.
        if self.settings.get("review_popup", True):
            self._review_edits(action_id, from_row=False)

    def _contradict_pretend(self, reply: str) -> str:
        """Say so when the model announced a save it cannot perform.

        This is the failure that costs the user the most, because it does not
        look like a failure.  The model says it opened the file, says it
        replaced the function, says the file has been successfully modified —
        and the app, which is the only thing here that can actually write,
        says nothing at all.  The user goes and looks at a file that has not
        changed and concludes the editing feature is broken.

        The app knows better than the model does, and this is where it says
        so.  Only when nothing was proposed and nothing was rejected: if
        there is a real diff waiting, or a real reason it was refused, that
        is the more useful thing to be reading.
        """
        claim = edit_tools.pretended_to_write(
            reply, names=list(self.project_texts))
        if not claim:
            return ""
        self.chat.add_note(
            f"⚠ Nothing was written — your files are untouched. The model "
            f"said “{claim}”, but it cannot open or save anything; the only "
            f"way a change reaches disk is an edit block in the reply, which "
            f"this one had none of.")
        return claim

    def _offer_to_retry(self, rejected, target=None) -> None:
        """Offer another go at edits that were refused for a fixable reason.

        Almost every refusal here is the same one: the model quoted lines
        from memory instead of copying them.  That is worth retrying rather
        than reporting, because the app is not guessing about the fix — it
        has the file, and the second request hands the real text back with
        the one change restated.  Reporting it and stopping leaves the user
        to paste the file into the chat by hand, which is the app's job.
        """
        target = target or self.last_edit_target
        if target is None:
            return
        # Retryable means "the real file is on disk to hand back" — loaded
        # or not.  A file that never fit the context budget is the *most*
        # likely to be here, since the model was working from its name
        # alone; refusing it a retry would refuse exactly the file that
        # needs one.
        retryable = [item for item in rejected
                     if item.name in self.project_texts
                     or edit_tools.read_current(self.project, item.name)]
        if not retryable:
            return
        changes = [intent.Change(
            description=f"the change you proposed for {item.name}",
            file=item.name) for item in retryable]
        self.action_seq += 1
        action_id = f"retry{self.action_seq}"
        self.pending_changes[action_id] = (changes, target)
        names = ", ".join(sorted({item.name for item in retryable}))
        self.chat.add_action(
            f"✏ Nothing was written for {names}. The lines it quoted are not "
            f"in the file — but the file is right here, so I can hand it back "
            f"and ask for the same change again with the real text to anchor "
            f"to.",
            [("Try again with the real file", lambda: self._realise_changes(action_id), True),
             ("Leave it", lambda: self._drop_changes(action_id), False)],
            action_id=action_id)

    # -- describe, approve, then fetch the actual change -------------------
    def _offer_to_make_changes(self, reply: str, target=None) -> str:
        """Treat a described change as the first half of an edit, not a failure.

        Returns which of the three things happened — "offered", "no-target" or
        "unmatched" — because the trace needs to tell them apart and a bare
        False makes "there was no model to ask" and "the model named no file
        we recognise" into the same answer. They have different fixes.

        Asking one model to explain a change *and* quote the exact lines it
        touches is two jobs competing for the same attention, and on a 14B
        the explanation wins every time.  But the explanation is not nothing:
        it was written with the files in view, and it is precisely what a
        person wants to say yes or no to.

        So the list becomes an offer.  Say yes and the app goes back to the
        same model once per item — one file, one change, the whole allowance,
        prose explicitly refused — which is a far easier question than the
        one that just failed.  What comes back is a real diff, and the review
        window still stands in front of it.
        """
        target = target or self.last_edit_target
        if target is None:
            return "no-target"
        # Matched against everything the scan found, not just what fit the
        # budget — a described change to an unloaded file is the case this
        # flow exists for, since that model never had the contents to quote.
        known = self.project_files or list(self.project_texts)
        changes = intent.parse_changes(reply, known)
        if not changes and not suggest.is_broad(self.last_question):
            # The reply named no file, but the question very often did —
            # "fix the retry in mail_engine.py" answered with four paragraphs
            # that never say the filename again.  Without this the app has a
            # described change it cannot place and offers nothing at all,
            # which is the failure that looks most like a dead button.  A
            # *broad* question is excluded: turning "make it faster" into a
            # single scoped change is exactly the shape that breaks, and the
            # menu path exists so it never has to.
            changes = intent.from_question(self.last_question, reply, known)
        if not changes:
            return "unmatched"

        self.action_seq += 1
        action_id = f"intent{self.action_seq}"
        self.pending_changes[action_id] = (changes, target)
        name = friendly_model_name(target.model) or short_model(target.model)
        listing = "\n".join(f"    {n}. {c.summary}  ({c.file})"
                            for n, c in enumerate(changes, 1))
        # Asking permission to *ask again* is a click that buys nothing.  The
        # second pass writes nothing — it fetches a diff, and the review
        # window still stands between that diff and the disk — so the only
        # thing the click protects is a minute of the model's time.  Meanwhile
        # the request the user actually made sits unanswered behind a button
        # they have to notice, which is the whole complaint this feature keeps
        # attracting: it described the changes and did nothing.
        # Only when the user actually asked for a change. Firing a second
        # round of model calls off the back of an ordinary question would
        # spend minutes of their machine answering something they did not
        # ask, and the button is the right shape for a maybe.
        if (self.settings.get("auto_realise", True)
                and edit_tools.wants_to_edit(self.last_question)):
            self.chat.add_note(
                f"✏ {intent.describe(changes)} described as prose rather than "
                f"as something the app can apply.\n{listing}\n\nGoing back to "
                f"{name} for each one on its own — the diff will be shown "
                f"before anything is written.")
            self._realise_changes(action_id)
            return "offered"
        self.chat.add_action(
            f"✏ {intent.describe(changes)} described, but written as prose "
            f"rather than as something the app can apply — so nothing has "
            f"been written.\n{listing}\n\nI can go back to {name} and ask for "
            f"each one on its own, then show you the diff before anything is "
            f"written.",
            [("Make these changes", lambda: self._realise_changes(action_id), True),
             ("Not now", lambda: self._drop_changes(action_id), False)],
            action_id=action_id)
        self.set_status(f"Waiting on you — {intent.describe(changes)} proposed")
        return "offered"

    def _drop_changes(self, action_id: str) -> None:
        self.pending_changes.pop(action_id, None)
        self.chat.resolve_action(
            action_id, "✏ Left alone — nothing was written.")

    def _realise_changes(self, action_id: str) -> None:
        """Go and get the real edits for changes the user has agreed to.

        One at a time, and the next one is not asked for until this one has
        been decided.  Fetching all of them first meant waiting several
        minutes per change before seeing anything, then being handed a pile
        of diffs to judge together — and paying for every one of them even
        when the first made it obvious the run was going nowhere.
        """
        pending = self.pending_changes.pop(action_id, None)
        if not pending:
            return
        changes, target = pending
        self.chat.resolve_action(
            action_id,
            f"✏ Asking for {intent.describe(changes)}, one at a time…")
        if self.settings.get("one_at_a_time", True):
            self.change_queue = list(changes)
            self.change_target = target
            self._next_change()
            return
        self._run(lambda: self._fetch_changes(changes, target))

    def _fetch_menu(self, ask: str, target) -> None:
        """Ask for the menu as a scoped request (worker thread).

        Low temperature, no style directive, the relevant files and the
        symbol index — the shape that produced a clean, fully-grounded menu
        on every measured run, where the same request inside the
        conversation produced an essay.
        """
        client = self.clients().get(target.server)
        if client is None or not client.base_url:
            self.events.put(("note", {
                "text": "⚠ That model is not reachable, so no suggestions "
                        "could be fetched."}))
            return
        # Not question-keyword retrieval: "performance enhancements" matches
        # the *documentation* — the README and notes discuss performance in
        # those words — while the engines the improvements would land in
        # match nothing and stay home.  A menu is built from code, as much
        # of it as fits, with the symbol index covering whatever does not.
        chosen = suggest.pick_menu_files(
            ask, self.project_texts,
            attachment_allowance(self.settings.effective_budget()))
        self.events.put(("note", {
            "text": f"📂 Scanning {len(chosen)} code file(s) in full for "
                    f"places to improve: "
                    f"{', '.join(c.name for c in chosen)} — plus the "
                    f"function index of everything else."}))
        block = "\n\n".join(
            f"--- {chunk.name} ---\n{chunk.text}\n--- end of {chunk.name} ---"
            for chunk in chosen)
        self.sent_files.update(chunk.name for chunk in chosen)
        user = f"{block}\n\n" + suggest.wrap_ask(
            ask, suggest.symbol_index(self.project_texts),
            file_map=suggest.project_map(self.project_texts))
        try:
            reply = client.chat(
                target.model,
                [{"role": "system", "content": CONVERSATIONAL_PROMPT},
                 {"role": "user", "content": user}],
                options={**self._options(edit_tools.MIN_EDIT_TOKENS),
                         "temperature": 0.3},
                think=think_value(target.model, False),
                cancel=self.cancel)
        except Exception as exc:                            # noqa: BLE001
            self.events.put(("note", {
                "text": f"⚠ Could not fetch suggestions — {str(exc)[:140]}"}))
            return
        self.runlog.record(reply, purpose="suggesting")
        if self.cancel.is_set():
            # Never silently: a fetch that vanishes reads as a dead button.
            self.events.put(("note", {
                "text": "✏ Stopped — no suggestions were fetched."}))
            return
        self.events.put(("menu_ready", {"reply": reply.text,
                                        "target": target}))

    def _offer_suggestions(self, candidates, target) -> None:
        """Show the menu and let the user pick exactly one.

        One, not several: the whole point is that a single scoped change is
        the shape that lands, and the rest of the menu waits its turn rather
        than being fetched on spec.
        """
        self.menu_left = list(candidates)
        self.menu_target = target
        self._show_menu("✏ Here's what could be improved — pick one and "
                        "I'll fetch that change for you to review. Nothing "
                        "is written without your approval.")

    def _show_menu(self, headline: str) -> None:
        candidates = self.menu_left
        if not candidates:
            return
        self.action_seq += 1
        action_id = f"menu{self.action_seq}"
        self.menu_action_id = action_id
        # The full description, not the 90-character summary.  These lines
        # are the entire basis of the decision; an ellipsis exactly where the
        # sentence says what the change *does* turns the pick into a guess.
        # Flags earn their place by being rare: "may already exist" is the
        # redundancy that went five-for-five in live testing, said before a
        # pick costs a model call rather than after.
        flags = {}
        for c in candidates:
            note = suggest.may_already_exist(c, self.project_texts)
            if note:
                flags[id(c)] = note
            elif suggest.looks_multi_location(c.description):
                flags[id(c)] = "touches more than one place"
        listing = "\n".join(
            f"    {n}. {c.description}  ({c.file})"
            + (f"  ⚠ {flags[id(c)]}" if id(c) in flags else "")
            for n, c in enumerate(candidates, 1))
        # The buttons say "1".."6", which is nothing — so each one carries
        # the whole candidate under the pointer.
        buttons = [(str(n), self._picker(action_id, n - 1), n == 1,
                    f"{c.file}\n\n{c.description}"
                    + (f"\n\n⚠ {flags[id(c)]}" if id(c) in flags else ""))
                   for n, c in enumerate(candidates, 1)]
        buttons.append(("None of these", lambda: self._drop_menu(action_id),
                        False))
        self.chat.add_action(f"{headline}\n{listing}", buttons,
                             action_id=action_id)
        self.set_status("Waiting on you — pick a change from the list")

    def _picker(self, action_id: str, index: int):
        return lambda: self._pick_suggestion(action_id, index)

    def _pick_suggestion(self, action_id: str, index: int) -> None:
        if index >= len(self.menu_left):
            return
        chosen = self.menu_left.pop(index)
        self.chat.resolve_action(
            action_id, f"✏ {chosen.summary}  ({chosen.file}) — fetching the "
                       f"change…")
        self.change_queue = [chosen]
        self.change_target = self.menu_target or self.last_edit_target
        if self.change_target is None:
            self.chat.add_note(
                "⚠ There is no model to ask — send a message with a model "
                "selected first.")
            return
        self._next_change()

    def _drop_menu(self, action_id: str) -> None:
        self.menu_left = []
        self.chat.resolve_action(
            action_id, "✏ Left alone — nothing was fetched or written.")

    def _next_change(self) -> None:
        """Ask for the next queued change, if the user has not stopped."""
        if not self.change_queue or self.cancel.is_set():
            return
        change = self.change_queue.pop(0)
        left = len(self.change_queue)
        target = self.change_target
        self.chat.add_note(
            f"✏ {change.summary}  ({change.file})"
            + (f" — {left} more after this one." if left else
               " — the last one."))
        self._run(lambda: self._fetch_changes([change], target))

    def _advance_change_queue(self) -> None:
        """Called once a change has been decided, however it was decided."""
        if not self.change_queue:
            if self.menu_left and not self.cancel.is_set():
                # The user's flow, as designed: one change, decided, then
                # "any more, or move on?"  The menu they already read comes
                # back minus the one they just dealt with.
                self._show_menu(
                    "✏ Done with that one. Want another from the list, or "
                    "ask for something else?")
                return
            if getattr(self, "change_target", None) is not None:
                self.change_target = None
            return
        if self.cancel.is_set():
            self.chat.add_note(
                f"✏ Stopped — {len(self.change_queue)} change(s) were not "
                f"asked for. Nothing further was written.")
            self.change_queue = []
            return
        self.root.after(50, self._next_change)

    def _fetch_changes(self, changes, target) -> None:
        """One scoped request per change (worker thread).

        Sequential rather than parallel on purpose: these are large local
        models on two consumer cards, and firing three at once means three
        models resident at once or three that swap.  Slower and finishing
        beats faster and thrashing.
        """
        client = self.clients().get(target.server)
        if client is None or not client.base_url:
            self.events.put(("note", {
                "text": "⚠ That model is no longer reachable, so the changes "
                        "could not be fetched. Nothing was written."}))
            return

        instructions = edit_tools.instructions(detailed=True)
        replies: list[str] = []
        declined: list[str] = []
        for index, change in enumerate(changes, 1):
            if self.cancel.is_set():
                break
            # Loaded text if we have it, disk if we don't.  This is the one
            # place an unloaded file gets its contents in front of a model,
            # so falling back to disk is the entire point — skipping here
            # made "did not fit the budget" mean "can never be edited".
            current = self.project_texts.get(change.file)
            if current is None:
                current = edit_tools.read_current(self.project, change.file)
            if not current:
                self.events.put(("note", {
                    "text": f"⚠ {change.file} could not be read from the "
                            f"attached folder, so that change was skipped."}))
                continue
            self.events.put(("note", {
                "text": f"✏ ({index}/{len(changes)}) {change.summary}",
                "note_id": "realise"}))
            # A file bigger than the window was uneditable by every path —
            # nothing could carry it, and Ollama answers an overfull prompt
            # by throwing away its front.  The tree knows where the target
            # function lives, so the request carries that region instead.
            excerpt_of, file_tree = None, ""
            if estimate_tokens(current) > codetree.FITS_WHOLE_TOKENS:
                total = current.count("\n") + 1
                named = suggest.NAMES_FUNCTION.search(change.description or "")
                found = (codetree.region(current, named.group(1))
                         if named else None) or codetree.region_by_terms(
                             current, suggest.claim_terms(change.description))
                if found:
                    excerpt, first, last = found
                    excerpt_of = (first, last, total)
                    file_tree = codetree.render(
                        codetree.tree(current), change.file, total)
                    current = excerpt
                    self.events.put(("note", {
                        "text": f"🌳 {change.file} is too big to send whole "
                                f"— sending lines {first}–{last}, where the "
                                f"change lives."}))
            try:
                reply = client.chat(
                    target.model,
                    [{"role": "user", "content": intent.build_change_prompt(
                        change, current, instructions,
                        excerpt_of=excerpt_of, file_tree=file_tree,
                        evidence=suggest.existing_evidence(
                            change, self.project_texts))}],
                    options={**self._options(edit_tools.MIN_EDIT_TOKENS * 2),
                             "temperature": 0.1},
                    think=think_value(target.model, False),
                    cancel=self.cancel)
            except Exception as exc:
                self.events.put(("note", {
                    "text": f"⚠ {change.file} — {str(exc)[:140]}"}))
                continue
            self.runlog.record(reply, purpose="editing")
            self.sent_files.add(change.file)
            if intent.said_no_change(reply.text):
                # It was given a way to say "nothing to do here" and took it.
                # That is an answer, and reporting it as one is the difference
                # between a model that declined and a feature that broke.
                declined.append(change.file)
                continue
            replies.append(reply.text)

        self.events.put(("note_close", {"note_id": "realise"}))
        if declined and not self.cancel.is_set():
            names = ", ".join(sorted(set(declined)))
            self.events.put(("note", {
                "text": f"✏ {names} — the model had no specific change to "
                        f"make here and said so rather than inventing one. "
                        f"Naming the change you want ('replace the 0.85 "
                        f"default with a constant') gets further than asking "
                        f"it to find something."}))
        if replies and not self.cancel.is_set():
            self.events.put(("changes_ready", {"reply": "\n\n".join(replies),
                                               "target": target}))
        elif not self.cancel.is_set():
            self.events.put(("note", {
                "text": "⚠ Nothing usable came back, so nothing was written. "
                        "Your files are untouched."}))

    def _review_edits(self, action_id: str, from_row: bool = True) -> None:
        """Open the review window on a set of changes that are waiting.

        `from_row` is the user having pressed Review, which spends the row in
        the transcript.  The window that opens by itself leaves the row alone
        until a decision is actually made, because a window that appears
        without being asked for is also one that gets dismissed without being
        read — and the diffs have to still be there when that happens.
        """
        proposed = self.pending_edits.get(action_id) or []
        if not proposed:
            return
        flaws = self.pending_flaws.get(action_id) or {}
        if from_row:
            self.pending_edits.pop(action_id, None)
            self.chat.resolve_action(action_id, "✏ Reviewing the changes…")

        def done(chosen, always: bool) -> None:
            if always:
                self.auto_apply_edits = True
                self.chat.add_note(
                    "✏ Future changes in this chat will be applied without "
                    "asking. Starting a new chat resets that.")
            if chosen:
                # Spend the row now, whichever way the window was opened.
                self.pending_edits.pop(action_id, None)
                self.pending_flaws.pop(action_id, None)
                self.chat.resolve_action(action_id, "✏ Reviewing the changes…")
                self._apply_edits(chosen, always=False)
                self._advance_change_queue()
            elif from_row:
                # The row is gone and the changes are not discarded, so they
                # need somewhere to live.
                self._reoffer_edits(proposed, flaws)
            else:
                self.chat.add_note(
                    "✏ Nothing was written. The changes are still on the row "
                    "above if you want another look.")
                self._advance_change_queue()

        EditReview(self.root, self.project, proposed, on_apply=done,
                   flaws=flaws)

    def _reoffer_edits(self, proposed, flaws=None) -> None:
        """Put declined-but-not-discarded changes back within reach."""
        self.action_seq += 1
        action_id = f"edit{self.action_seq}"
        self.pending_edits[action_id] = proposed
        self.pending_flaws[action_id] = flaws or {}
        summary = edit_tools.summarise(self.project, proposed)
        self.chat.add_action(
            f"✏ Nothing was written. The proposed changes are still here — "
            f"{summary} — if you want another look.",
            [("Review the changes", lambda: self._review_edits(action_id), True),
             ("Discard", lambda: self._discard_edits(action_id), False)],
            action_id=action_id)

    def _discard_edits(self, action_id: str) -> None:
        self.pending_edits.pop(action_id, None)
        self.pending_flaws.pop(action_id, None)
        self.chat.resolve_action(
            action_id, "✏ Discarded — nothing was written.")
        self._advance_change_queue()

    def _apply_edits(self, chosen, always: bool) -> None:
        results = edit_tools.apply(self.project, chosen)
        self.chat.add_note(edit_tools.describe_result(results))
        written = [r.name for r in results if r.ok]
        if written:
            # The files on disk have moved on, so anything cached about them
            # is now describing the wrong version.
            for name in written:
                target = edit_tools.resolve(self.project, name)
                if target is not None and target.is_file():
                    try:
                        self.project_texts[name] = target.read_text(
                            encoding="utf-8", errors="replace")
                    except OSError:
                        self.project_texts.pop(name, None)
                if name not in self.project_files:
                    self.project_files.append(name)
            self.set_status(f"Wrote {len(written)} file(s)")
            snapshot = next(iter(edit_tools.snapshots(self.project)), "")
            if snapshot:
                self.action_seq += 1
                undo_id = f"undo{self.action_seq}"
                self.chat.add_action(
                    "↩ If that was not what you wanted, this puts every one "
                    "of those files back exactly as it was.",
                    [("Undo these changes",
                      lambda: self._undo_edits(snapshot, undo_id), False),
                     ("Keep them", lambda: self.chat.resolve_action(
                         undo_id, "✓ Changes kept."), True)],
                    action_id=undo_id)
                self._test_after_apply(snapshot)
        if always:
            self.chat.add_note(
                "✏ Applied without asking, as you chose earlier in this chat.")

    def _test_after_apply(self, stamp: str) -> None:
        """Run the project's own tests against what was just written.

        Green is proof no diff can give.  Red rolls the apply back to the
        snapshot it just made and shows the failure — the edit that reached
        this user's disk broken passed every static gate on the way, and
        running the code is the only judge that cannot be fooled by a
        plausible-looking diff.
        """
        if not self.settings.get("test_after_apply", True):
            return
        runner = testrun.find_runner(self.project)
        if runner is None:
            self.chat.add_note(
                "🧪 No tests found to run for this project — the change "
                "stands on the diff review alone.")
            return
        self.chat.add_note(
            f"🧪 Running {runner.label} against what was just written…")
        project = self.project
        timeout_s = int(self.settings.get("test_timeout", 180) or 180)

        def work():
            outcome = testrun.run(project, runner, timeout_s=timeout_s,
                                  cancel=self.cancel)
            self.events.put(("tests_done", {"outcome": outcome,
                                            "stamp": stamp}))

        threading.Thread(target=work, daemon=True).start()

    def _tests_done(self, outcome, stamp: str) -> None:
        """The verdict, and the rollback when it is red (main thread)."""
        if not outcome.ran:
            self.chat.add_note(f"🧪 {outcome.summary()}.")
            return
        if outcome.ok:
            self.chat.add_note(f"✅ {outcome.summary()} — the change holds.")
            self.set_status(f"Tests passed ({outcome.seconds:.0f}s)")
            return
        results = edit_tools.restore(self.project, stamp)
        restored = all(r.ok for r in results) and bool(results)
        for result in results:
            if not result.ok:
                continue
            target = edit_tools.resolve(self.project, result.name)
            if target is not None and target.is_file():
                try:
                    self.project_texts[result.name] = target.read_text(
                        encoding="utf-8", errors="replace")
                except OSError:
                    pass
        tail = "\n".join(outcome.tail.splitlines()[-12:])
        self.chat.add_note(
            f"❌ {outcome.summary()} — "
            + (f"the change was rolled back automatically; every file is "
               f"exactly as it was before the apply."
               if restored else
               f"and the automatic rollback ALSO failed — restore snapshot "
               f"{stamp} from the backups folder by hand.")
            + f"\n\nWhat the tests said:\n{tail}")
        self.set_status("Tests failed — change rolled back"
                        if restored else "Tests failed — MANUAL RESTORE NEEDED")

    def _undo_edits(self, stamp: str, action_id: str) -> None:
        results = edit_tools.restore(self.project, stamp)
        if all(r.ok for r in results):
            self.chat.resolve_action(
                action_id,
                f"↩ Put {len(results)} file(s) back as they were.")
            for result in results:
                target = edit_tools.resolve(self.project, result.name)
                if target is not None and target.is_file():
                    try:
                        self.project_texts[result.name] = target.read_text(
                            encoding="utf-8", errors="replace")
                    except OSError:
                        pass
        else:
            problems = "; ".join(r.error for r in results if not r.ok)
            self.chat.resolve_action(action_id, f"⚠ Could not undo: {problems}")

    # ------------------------------------------------------ project memory
    def _pin_project_memory(self) -> None:
        """Give a new chat what the last one worked out.

        The folder is re-read from disk every time, so the *code* is never
        stale — what a fresh chat loses is the thinking about it.  This puts
        that back, and only that: a few hundred tokens of brief, position and
        decisions rather than a transcript nobody wants to re-read.
        """
        if self.project is None:
            return
        memory = projectmemory.load(self.project)
        if memory.empty:
            return
        block = memory.as_block()
        if not block:
            return
        for target in self.selected_targets() or []:
            history = self.session.history(target.key)
            if any(projectmemory.HEADER.split("—")[0].strip()
                   in str(m.get("content", "")) for m in history):
                continue
            self.session.add(target.key, "user", block, pinned=True)
        self.chat.add_note(
            f"🧠 Picked up where we left off with {Path(self.project).name} — "
            f"the brief, the decisions log and the last progress note.")
        self._update_context_label()

    def _write_handoff(self, then=None) -> bool:
        """Ask the model to leave a note for the next chat (worker thread)."""
        if self.project is None or not self.settings.get("project_memory", True):
            return False
        targets = self.selected_targets()
        history = self.session.history(targets[0].key) if targets else []
        if not projectmemory.worth_a_handoff(history):
            return False
        client = self.clients().get(targets[0].server) if targets else None
        if client is None or not client.base_url:
            return False

        transcript = "\n\n".join(
            f"{m.get('role')}: {m.get('content', '')}"
            for m in history
            if not carries_attachment(str(m.get("content", ""))))
        try:
            reply = client.chat(
                targets[0].model,
                [{"role": "user",
                  "content": projectmemory.build_handoff_prompt(transcript)}],
                options={**self._options(400), "temperature": 0.2},
                think=think_value(targets[0].model, False),
                cancel=self.cancel)
        except Exception:
            return False              # a hand-off must never block anything
        progress, decision = projectmemory.parse_handoff(reply.text)
        wrote = projectmemory.save_progress(self.project, progress)
        if projectmemory.append_decision(self.project, decision):
            wrote = True
        if wrote:
            self.events.put(("note", {
                "text": f"🧠 Wrote a hand-off note into "
                        f"{Path(self.project).name}/.aichatlab — the next "
                        f"chat on this folder will start with it."}))
        return wrote

    # -------------------------------------------------------------- folders
    def add_folder(self, path) -> None:
        """Measure a folder, ask how much of it to read, then read that much.

        Split across three steps and two threads because each has a different
        cost: walking is cheap but not instant, the decision is the user's, and
        the reading is the expensive part nobody should pay for before saying
        yes to it.
        """
        path = Path(path)
        if not path.is_dir():
            return
        if any(a["name"] == f"{path.name}/" for a in self.attachments):
            self.set_status(f"'{path.name}/' is already attached")
            return

        self.set_status(f"Scanning '{path.name}'…")

        def scan():
            try:
                result = survey(path)
            except OSError as exc:
                self.events.put(("folder_failed",
                                 {"path": path, "message": str(exc)}))
                return
            self.events.put(("folder_scanned", {"path": path, "survey": result}))

        threading.Thread(target=scan, daemon=True).start()

    def _folder_scanned(self, path: Path, result) -> None:
        self.set_status("Ready")
        if not result.entries:
            messagebox.showinfo(
                "Nothing to read here",
                f"'{path.name}' has no files the models can read.\n\n"
                f"Folders are scanned for {SUPPORTED_HINT}; images, videos and "
                f"binaries are skipped.")
            return
        FolderScanDialog(
            self.root, result,
            self.settings.effective_budget(),
            on_accept=lambda limits, budget: self._folder_accepted(
                path, result, limits, budget),
            max_window=int(self.settings.get("max_context_window", 32768)))

    def _folder_accepted(self, path: Path, result, limits, budget: int) -> None:
        selection = select(result, limits)
        if not selection.chosen:
            # Returning quietly here leaves the folder unattached with nothing
            # on screen saying so — and every later message then behaves as if
            # no folder was ever chosen, including refusing to edit. The user
            # pressed a button and the app did nothing, which is never an
            # acceptable answer.
            self.chat.add_note(
                f"⚠ Nothing from {path.name} was attached — the limits chosen "
                f"in the scan window left no files. Attach it again and raise "
                f"the file or token limit. Nothing else has changed, and no "
                f"folder is attached, so the models cannot edit anything.")
            self.set_status("Nothing attached — no files fitted the limits")
            return

        # This chat is now about a project, which is what makes a hand-off
        # worth writing when it ends.
        self.project = Path(path)
        self.project_files = [entry.relative for entry in result.entries]
        self._pin_project_memory()

        current = int(self.settings.get("context_budget_tokens", 6000))
        if int(budget) > current:
            self.settings["context_budget_tokens"] = int(budget)
            self.settings.save()
            self.chat.add_note(
                f"📈 Context budget raised from {current:,} to {int(budget):,} "
                f"tokens so this folder fits.")
            self._update_context_label()

        self.set_status(f"Reading {len(selection.chosen)} file(s)…")
        total = len(result.entries)

        def read():
            def progress(done: int, count: int) -> None:
                if done % 10 == 0 or done == count:
                    self.events.put(("status", {
                        "text": f"Reading {done}/{count} files from "
                                f"'{path.name}'…"}))

            # Cap on what we will actually build, not just what was asked
            # for.  If the folder is bigger than the model's window, something
            # is going to be dropped either way — far better that we do it,
            # deliberately and with a note in the text, than that the server
            # silently discards the front of the prompt (which is where the
            # manifest and the README live).
            ceiling = min(limits.max_tokens,
                          attachment_allowance(self.settings.effective_budget()))
            # Read every readable file, not just the ones that fit the send
            # budget.  The budget is about what one prompt can carry; the
            # app holding a file costs nothing but memory, and every editing
            # path — retrieval, the retry, the second pass — works from what
            # the app holds.  Capped by bytes so a pathological folder
            # cannot swallow the machine.
            texts = read_texts(worth_holding(selection, KEEP_ON_HAND_BYTES),
                               progress=progress)
            block = build_block(path, selection.chosen, total,
                                char_budget=ceiling * DENSE_CHARS_PER_TOKEN)
            self.events.put(("folder_read", {"path": path, "block": block,
                                             "selection": selection,
                                             "texts": texts}))

        threading.Thread(target=read, daemon=True).start()

    # ------------------------------------------------------------ retrieval
    def _retrieval_wanted(self) -> bool:
        """Only worth it once a folder is too big to simply send."""
        if not self.settings.get("folder_retrieval", True):
            return False
        return len(self.project_texts) >= int(
            self.settings.get("retrieval_min_files", 12))

    def _embedding_model(self) -> str:
        """An installed embedding model, or "" — retrieval works without one."""
        wanted = str(self.settings.get("embedding_model", "nomic-embed-text"))
        if not wanted:
            return ""
        for (_server, model) in self.model_vars:
            if model.split(":")[0] == wanted.split(":")[0]:
                return model
        return ""

    def _index_project(self) -> None:
        """Summarise and embed each file once per version of it (worker)."""
        if self.project is None or not self.project_texts:
            return
        index = projectindex.load(self.project)
        index.forget_missing(self.project_texts.keys())
        files = list(self.project_texts.items())

        model = self._embedding_model()
        server = next((srv for (srv, mdl) in self.model_vars
                       if mdl == model), "local")
        client = self.clients().get(server)
        outstanding = index.needs_embedding(files, model) if (
            model and client and client.base_url) else []
        if outstanding:
            self.events.put(("note", {
                "text": f"🧮 Indexing {len(outstanding)} file(s) with {model} "
                        f"so questions can pull only what they need…",
                "note_id": "indexing"}))
            # In batches: one request per file is a round-trip per file, and
            # one request for four hundred files is a payload no server wants.
            batch = 16
            for start in range(0, len(outstanding), batch):
                if self.cancel.is_set():
                    break
                chunk = outstanding[start:start + batch]
                vectors = client.embed(model, [
                    projectindex.embedding_text(
                        name, index.summary_for(
                            name, projectindex.content_hash(text)), text)
                    for name, text in chunk])
                if len(vectors) != len(chunk):
                    self.events.put(("note", {
                        "text": "🧮 The embedding model did not answer — "
                                "falling back to keyword search, which is "
                                "worse at paraphrase and fine at identifiers.",
                        "note_id": "indexing"}))
                    break
                for (name, text), vector in zip(chunk, vectors):
                    index.record_vector(name, text, vector, model)
                self.events.put(("status", {
                    "text": f"Indexed {min(start + batch, len(outstanding))}"
                            f"/{len(outstanding)} files…"}))
            projectindex.save(self.project, index)
            self.events.put(("note_close", {"note_id": "indexing"}))
        self.project_index = index

    def _chunks(self):
        index = self.project_index or projectindex.Index()
        model = self._embedding_model()
        chunks = []
        for name, text in self.project_texts.items():
            digest = projectindex.content_hash(text)
            chunks.append(Chunk(name=name, text=text,
                                summary=index.summary_for(name, digest),
                                vector=index.vector_for(name, digest, model)))
        return chunks

    def _retrieval_block(self, question: str) -> str:
        """The files this particular question needs, and nothing else."""
        chunks = self._chunks()
        if not chunks:
            return ""
        if not retrieval_worth(question, chunks):
            # "say hello" does not need three source files in front of it —
            # sent anyway, the model answers the code, because the code is
            # almost all of what it was given.
            self.events.put(("note", {
                "text": "📂 That doesn't look like a question about the "
                        "folder, so no files were sent with it."}))
            return ""
        model = self._embedding_model()
        question_vector = []
        if model and any(c.vector for c in chunks):
            server = next((srv for (srv, mdl) in self.model_vars
                           if mdl == model), "local")
            client = self.clients().get(server)
            if client is not None and client.base_url:
                vectors = client.embed(model, [question])
                question_vector = vectors[0] if vectors else []

        budget = attachment_allowance(self.settings.effective_budget())
        chosen = select_chunks(question, chunks, budget_tokens=budget,
                        question_vector=question_vector,
                        max_files=int(self.settings.get("retrieval_files", 10)))
        self.sent_files.update(chunk.name for chunk in chosen)
        self.events.put(("note", {
            "text": describe_selection(chosen, len(chunks))}))
        if not chosen:
            return ""
        parts = [f"[Folder: {Path(self.project).name} — "
                 f"{len(chosen)} of {len(chunks)} files, chosen for this "
                 f"question]"]
        for chunk in chosen:
            parts.append(f"--- {chunk.name} ---\n```\n{chunk.text}\n```")
        parts.append("[End of folder contents.]")
        return "\n\n".join(parts)

    def _folder_read(self, path: Path, block: str, selection,
                     texts: dict | None = None) -> None:
        self.project_texts = dict(texts or {})
        self.sent_files = set()
        self.project_index = None
        if self._retrieval_wanted():
            # Big folder: keep the contents as data and choose per question.
            # The chip still says the folder is attached, because it is —
            # what changes is that a question gets the files it needs rather
            # than the first thirty thousand tokens of everything.
            self.attachments = [a for a in self.attachments
                                if a["name"] != f"{path.name}/"]
            self._rebuild_chips()
            self.chat.add_note(
                f"📁 Attached {path.name}/ — all {len(self.project_texts)} "
                f"files read and kept on hand. Each question will be sent "
                f"only the files it needs, rather than a fixed slice of the "
                f"whole folder. Ask for changes and they will be shown as a "
                f"diff to approve before anything is written.")
            self.set_status(f"Attached '{path.name}/' — indexing…")
            threading.Thread(target=self._index_project, daemon=True).start()
            return
        name = f"{path.name}/"
        self.sent_files = {entry.relative for entry in selection.chosen}
        self.attachments = [a for a in self.attachments if a["name"] != name]
        self.attachments.append({"name": name, "content": block})
        self._rebuild_chips()

        # Report what was actually built, not what was estimated.  Announcing
        # "94,000 tokens" after trimming the block to 31,000 would be the same
        # class of quiet lie this whole feature exists to avoid.
        actual = estimate_tokens(block)
        total = len(self.project_files) or len(selection.chosen)
        counted = (f"{len(selection.chosen)} of {total} files"
                   if total > len(selection.chosen)
                   else f"{len(selection.chosen)} file"
                        f"{'s' if len(selection.chosen) != 1 else ''}")
        detail = f"{counted} · ≈{actual:,} tokens"
        if actual < selection.tokens * 0.95:
            detail += f" (trimmed from ≈{selection.tokens:,} to fit)"
        self.chat.add_note(f"📁 Attached {name} — {detail}")
        if total > len(selection.chosen):
            on_hand = len(self.project_texts) - len(selection.chosen)
            self.chat.add_note(
                f"📁 {total - len(selection.chosen)} file(s) were not sent up "
                f"front — together the folder is bigger than the model's "
                f"context window. {'All of them are' if on_hand >= total - len(selection.chosen) else f'{max(on_hand, 0)} of them are'} "
                f"still read and held by the app, so they can be edited: an "
                f"approved change sends the real file on its own.")
        self.set_status(f"Attached '{name}' — {detail}")

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
            # While the models are talking amongst themselves, a message from
            # you joins the conversation instead of being refused.
            if self.conversing:
                self._interject()
            else:
                self._offer_interrupt(self._input_text())
            return

        text = self._input_text()
        if not text and not self.attachments:
            return
        self._warn_if_nothing_to_edit(text)
        # Kept because the *question* often names the file when the answer
        # does not: "fix the retry in mail_engine.py" followed by three
        # paragraphs that never say the filename again.  Without it the app
        # has a described change it cannot place, and offers nothing.
        self.last_question = text
        # A broad wish — "make it faster", "more robust" — is not sent as an
        # instruction to edit, because measured on this app's own pipeline
        # that shape breaks things four times out of four.  It becomes a
        # request for a menu of small, one-place candidates instead, and the
        # one the user picks goes through the scoped path that works.
        menu_ask = (self.project is not None and bool(text)
                    and bool(self.project_texts)
                    and self.settings.get("allow_edits", True)
                    and self.settings.get("suggest_menu", True)
                    and self.current_mode() == "chat"
                    # A message that names models is an orchestration
                    # instruction, not a broad wish.  "14B suggest the list,
                    # THEN the 32B coder review it" was being hijacked into a
                    # single-model menu — the first model got a pick-list, the
                    # second was never called, and the handoff the user wrote
                    # out was silently thrown away.
                    and not self._text_names_a_model(text)
                    and suggest.is_broad(text))

        prompt = self._compose(text)
        shown = text or "(sent attached files)"
        footer = "  ".join(f"📎 {a['name']}" for a in self.attachments)

        targets = self.selected_targets()
        mode = self.current_mode()
        if mode == "critique":
            # First ticked drafts, second reviews, third rewrites — the
            # order the hint has been promising all along.
            targets = self._ordered_targets(targets)
        relay = parse_relay(text, self.available_targets()) if text else None

        if not relay and not targets:
            messagebox.showwarning(
                "No models selected",
                "Tick at least one model, or use a relay command like:\n\n"
                "    gemma2 ask llama3 what do you think?")
            return
        if not relay and mode == "research" and not str(
                self.settings.get("searxng_url", "")).strip():
            messagebox.showinfo(
                "Set up SearXNG first",
                "Research mode searches the web through your SearXNG "
                "instance.\n\nEnter its URL in ⚙ Settings and press Connect.")
            self.open_settings()
            return
        if not relay and mode not in ("chat", "research", "plan") and len(targets) < 2:
            messagebox.showwarning(
                "Need more models",
                f"{MODES[mode].split(' — ')[0]} needs at least two models "
                f"selected so they have someone to work with.")
            return
        if self._maybe_ask_to_search(text, mode, relay):
            return          # the message stays in the box until you decide
        if self._maybe_offer_history(text, mode, relay, targets):
            return

        if menu_ask and not relay:
            # Its own scoped request, exactly as the second pass is.  Riding
            # inside the conversation put the format rules up against the
            # style directive and the chat temperature, and the model wrote
            # an essay; scoped, with a low temperature and no directive, it
            # produced a clean menu on every measured run.
            target = targets[0]
            self.last_edit_target = target
            self._hide_placeholder()
            self.input.delete("1.0", tk.END)
            self.chat.add_user(shown, footer)
            self.chat.add_note(
                "✏ That's a broad one — asking for a menu of specific, "
                "small changes to pick from…")
            # The ordinary path gets a fresh cancel token inside
            # _make_orchestrator; this path returns before reaching it, and
            # a stale token from New chat silently killed the fetch.
            self.cancel = threading.Event()
            self._run(lambda: self._fetch_menu(text, target))
            return

        self._warn_about_edit_room()
        self._warn_about_prompt_size(prompt, targets)
        self._warn_about_old_server(targets)
        self._refresh_cards_soon()

        self._hide_placeholder()
        self.input.delete("1.0", tk.END)
        self.chat.add_user(shown, footer)
        # The triage step needs to know what was attached, but the chips are
        # cleared as soon as the message is sent, so take the inventory first.
        attached = list(self.attachments)
        self.attachment_names = [a["name"] for a in attached]
        self.attachments = []
        self._rebuild_chips()

        # Decided here, on the main thread, so the worker never has to reason
        # about which conversations exist — and before the checklist, which
        # has to list it.
        budget = self.settings.effective_budget()
        # `should_compact` only knows the conversation is large; whether any
        # of it can actually be folded is a different question — a chat that
        # is one big pinned attachment is over the threshold with nothing to
        # summarise.  Ask both, so the checklist never lists work that was
        # never going to happen.
        to_compact = ([t2 for t2 in targets
                       if should_compact(self.session.history(t2.key), budget)
                       and plan_compaction(self.session.history(t2.key)).worth_doing]
                      if self.settings.get("auto_compact", True) and not relay
                      else [])

        model_names = [friendly_model_name(t2.model) or short_model(t2.model)
                       for t2 in targets]
        self.checklist = pipeline_for(
            attachments=attached,
            searching=self.research_var.get() and bool(text) and not relay,
            models=[] if mode == "plan" else model_names,
            learning=self.learning_var.get() and bool(text) and not relay,
            compacting=bool(to_compact))
        self.said_stalls.clear()
        if attached:
            self.checklist.finish(
                "attach", detail=f"{len(attached)} attachment(s), "
                                 f"≈{sum(estimate_tokens(a['content']) for a in attached):,} tokens")

        orchestrator = self._make_orchestrator(
            text, prompt_tokens=estimate_tokens(prompt) + max(
                (self.session.total_tokens(t2.key) for t2 in targets),
                default=0))
        judge = self._judge_target()
        rounds = self._rounds()
        turns = self._turns()
        if mode == "converse":
            self.conversing = True
            while not self.interjections.empty():   # clear anything stale
                self.interjections.get_nowait()
        research = (self.research_var.get() and bool(text) and not relay)

        history = list(self.session.history(targets[0].key)) if targets else []

        learning = self.learning_var.get() and bool(text) and not relay
        primary = targets[0] if targets else None

        def job():
            if to_compact and not self.cancel.is_set():
                self._compact(to_compact)
            final_prompt = prompt
            # The files this question needs, chosen now rather than fixed
            # when the folder was attached.  Read fresh from what was loaded,
            # so a big project costs a small prompt.
            if self.project_texts and self._retrieval_wanted() and text:
                picked = self._retrieval_block(text)
                if picked:
                    final_prompt = f"{picked}\n\n{final_prompt}"
            if research:
                block = self._research(text, primary, history, attached)
                if block is None and self.cancel.is_set():
                    return
                if block:
                    final_prompt = f"{block}\n\n{prompt}"
            if relay:
                source, target, question = relay
                orchestrator.relay(source, target, question)
                return
            if mode == "plan":
                orchestrator.plan_and_work(
                    primary, final_prompt,
                    material=describe_material(attached, len(history)))
            elif mode == "research":
                self._deep_research(orchestrator, primary, text)
            elif mode == "converse":
                orchestrator.converse(targets, final_prompt, turns,
                                      pending=self._pending_interjections)
            elif mode == "debate":
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

    def _text_names_a_model(self, text: str) -> bool:
        """Does this message address any installed model by name?

        Matched loosely on the distinctive pieces of each name — "QWEN2.5
        32B CODER" has to count as naming `qwen2.5-coder:32b`, whatever the
        capitalisation and however the size and family are ordered.
        """
        said = (text or "").lower()
        if not said:
            return False
        for target in self.available_targets():
            friendly = (friendly_model_name(target.model)
                        or short_model(target.model)).lower()
            base = target.model.split(":")[0].lower()
            if friendly and friendly in said:
                return True
            if base and len(base) > 3 and base in said:
                return True
            # The family name alone — "qwen", "gemma", "llama" — is how
            # people actually type it mid-sentence.
            family = re.split(r"[\d:._-]", base, 1)[0]
            if len(family) >= 4 and family in said:
                return True
        return False

    def _warn_if_nothing_to_edit(self, text: str) -> None:
        """Say up front when a request to change files has nowhere to land.

        Without a folder the app cannot write, and it does not even tell the
        model that editing is possible — so the model asks which files are
        meant. From the outside that reads as the app refusing to do the work,
        and every part of the window stays quiet about the one fact that
        explains it. Said once per chat: a warning repeated on every message
        is one nobody reads.
        """
        if self.project is not None or self.warned_no_folder:
            return
        if not edit_tools.wants_to_edit(text):
            return
        self.warned_no_folder = True
        self.chat.add_note(
            "⚠ No folder is attached, so nothing can be written to disk and "
            "the model has not been told it may edit anything — it can only "
            "talk about code it cannot see, which is why it asks which files "
            "you mean. Press “Attach folder” and ask again.")

    def _warn_about_edit_room(self) -> None:
        """Say once when the answer length makes editing impossible."""
        if self.project is None or not self.settings.get("allow_edits", True):
            return
        _label, num_predict, _directive = self.settings.profile()
        if not edit_tools.cramped(num_predict) or self.warned_edit_room:
            return
        self.warned_edit_room = True
        self.chat.add_note(
            f"✏ The answer length is {num_predict} tokens, which is not "
            f"enough for the model to finish an edit block — every attempt "
            f"would be cut off part way through and nothing could be "
            f"applied. Raising it to {edit_tools.MIN_EDIT_TOKENS} for edits "
            f"in this chat; move the speed slider towards Quality if you "
            f"want longer answers generally.")

    def _warn_about_old_server(self, targets: list) -> None:
        """Say once when reasoning is asked for but the server cannot do it.

        Ollama gained `think` in 0.9.0.  An older server ignores the field
        rather than complaining, so reasoning simply never appears and there
        is nothing anywhere to explain why.
        """
        if not self.thinking_var.get():
            return
        for target in targets:
            version = self.server_versions.get(target.server, "")
            if not version or supports_think_api(version):
                continue
            if target.server in self.warned_old_servers:
                continue
            self.warned_old_servers.add(target.server)
            name = friendly_model_name(target.model) or short_model(target.model)
            self.chat.add_note(
                f"💭 {name} can reason, but the {target.server} server is "
                f"running Ollama {version} and reasoning arrived in 0.9.0 — "
                f"so it will answer without showing its working. Updating "
                f"Ollama from ollama.com switches it on; every model you "
                f"already have is kept.")
            return

    def _warn_about_prompt_size(self, prompt: str, targets: list) -> None:
        """Set expectations before a long silence, not during it.

        Nothing comes back while a model reads its prompt, so a big one looks
        identical to a hang.  Where the run log has seen this hardware read a
        prompt before, the estimate is measured rather than guessed.
        """
        tokens = estimate_tokens(prompt) + max(
            (self.session.total_tokens(t.key) for t in targets), default=0)
        if tokens < BIG_PROMPT_TOKENS or not targets:
            return
        rate = self.runlog.prompt_rate()
        estimate = ""
        if rate:
            estimate = (f" Based on this machine, reading it should take "
                        f"about {human_duration(tokens / rate)}.")
        model = targets[0].model
        window = self.settings.context_window(tokens)
        self.chat.add_note(
            f"⏳ Sending ≈{tokens:,} tokens to "
            f"{friendly_model_name(model) or short_model(model)}, asking for a "
            f"{window:,}-token window. A large model spends a while reading a "
            f"prompt this size before the first word appears.{estimate}")
        heavy = heavy_request(model, window)
        if heavy:
            self.chat.add_note(f"⚠ {heavy}")

    # --------------------------------------- switching to a different model
    def _richest_conversation(self, exclude: str):
        """The conversation with the most in it, other than this model's."""
        best, best_key = None, ""
        for key, messages in self.session.conversations.items():
            if key == exclude or not messages:
                continue
            weight = sum(estimate_tokens(m.get("content", "")) for m in messages)
            if best is None or weight > best:
                best, best_key = weight, key
        return best_key or None

    def _maybe_offer_history(self, text: str, mode: str, relay,
                             targets: list) -> bool:
        """Every model keeps its own history — say so before one answers blind.

        Tick a different model half way through a conversation and it starts
        with nothing, while the transcript on screen still shows everything
        that was said.  The reply that comes back — "I'll need some specifics
        about your app" — looks like the model being obtuse rather than the
        model being new.
        """
        if self.skip_carry_prompt:
            self.skip_carry_prompt = False
            return False
        # Only for a single model.  A broadcast to several is *expected* to
        # reach models that have not been part of the conversation, and
        # offering to seed one of them with another's history would be a
        # strange thing to do half way through a comparison.
        if relay or len(targets) != 1 or mode not in ("chat", "plan", "research"):
            return False
        target = targets[0]
        if self.session.history(target.key):
            return False
        donor = self._richest_conversation(target.key)
        if donor is None:
            return False

        messages = self.session.conversations[donor]
        _server, model = split_key(donor)
        name = friendly_model_name(target.model) or short_model(target.model)
        other = friendly_model_name(model) or short_model(model)
        attached = list(dict.fromkeys(
            n for m in messages
            for n, _ in attachment_details(m.get("content", ""))))
        count = len(messages)
        what = f"{count} message" if count == 1 else f"{count} messages"
        if attached:
            what += f", including {', '.join(attached)}"

        if self.auto_var.get():
            # Carrying it across is the choice that makes the next answer
            # useful; starting blind at 3am wastes the whole night.
            self.session.conversations[target.key] = [
                dict(message) for message in messages]
            self._update_context_label()
            self.auto.note(f"carried {other}'s conversation across to {name}")
            self.chat.add_note(
                f"🌙 {name} had not seen any of this, so {other}'s "
                f"conversation ({what}) was carried across automatically.")
            return False

        self.action_seq += 1
        action_id = f"carry{self.action_seq}"
        self.chat.add_action(
            f"↪ {name} has not seen any of this. Every model keeps its own "
            f"history, and this one is empty — what is on screen belongs to "
            f"{other} ({what}).",
            [(f"Bring it across to {name}",
              lambda: self._carry_history(donor, target, action_id), True),
             ("Start fresh with this model",
              lambda: self._start_fresh(action_id), False)],
            action_id=action_id)
        self.set_status(f"{name} starts with an empty history — carry it over?")
        return True

    def _carry_history(self, donor: str, target: Target, action_id: str) -> None:
        copied = [dict(message)
                  for message in self.session.conversations.get(donor, [])]
        self.session.conversations[target.key] = copied
        name = friendly_model_name(target.model) or short_model(target.model)
        plural = "message" if len(copied) == 1 else "messages"
        self.chat.resolve_action(
            action_id, f"↪ Carried {len(copied)} {plural} over to {name}.")
        self._update_context_label()
        self.skip_carry_prompt = True
        self.send()

    def _start_fresh(self, action_id: str) -> None:
        self.chat.resolve_action(
            action_id, "Starting this model with a clean history.")
        self.skip_carry_prompt = True
        self.send()

    # ------------------------------------------------ typing while it works
    def _offer_interrupt(self, text: str) -> None:
        """A message sent mid-run is a decision to make, not an error.

        "Still working — press Stop first" throws away what you typed and
        makes you the scheduler.  Keep the message instead, and ask the only
        question that matters: does this replace what is running, or follow
        it?
        """
        if not text:
            self.set_status("Still working — type a message to queue it, "
                            "or press Stop")
            return

        self.input.delete("1.0", tk.END)
        self.queued_message = text
        urgent = looks_urgent(text)
        quoted = text if len(text) <= 90 else text[:90].rstrip() + "…"

        self.action_seq += 1
        action_id = f"queue{self.action_seq}"
        stop_first = ("⏹ Stop that and do this now",
                      lambda: self._interrupt_with_queued(action_id), True)
        wait = ("⏳ Wait and do it next",
                lambda: self._queue_after(action_id), True)
        options = [stop_first, wait] if urgent else [wait, stop_first]
        # only one primary button, whichever is the recommendation
        options = [(label, callback, index == 0)
                   for index, (label, callback, _p) in enumerate(options)]

        self.chat.add_action(
            f"Something is still running. Your message — “{quoted}” — is held "
            f"until you say which:",
            options, action_id=action_id)

    def _queue_after(self, action_id: str) -> None:
        self.chat.resolve_action(
            action_id, "⏳ Queued — it will be sent as soon as this finishes.")
        if not (self.worker and self.worker.is_alive()):
            self._send_queued()

    def _interrupt_with_queued(self, action_id: str) -> None:
        self.chat.resolve_action(
            action_id, "⏹ Stopping the current task to run your message.")
        self.stop()

    def _send_queued(self) -> None:
        """Send the held message once the worker has actually let go."""
        if not self.queued_message:
            return
        if self.worker and self.worker.is_alive():
            self.root.after(150, self._send_queued)
            return
        text, self.queued_message = self.queued_message, ""
        self._hide_placeholder()
        self.input.delete("1.0", tk.END)
        self.input.insert("1.0", text)
        self.send()

    # --------------------------------------------------------- compaction
    def compact_chat(self) -> None:
        """Fold the old part of the conversation into a digest, on demand."""
        if self.worker and self.worker.is_alive():
            self.set_status("Still working — press Stop first")
            return
        targets = self.selected_targets()
        if not targets:
            messagebox.showinfo(
                "Nothing to compact",
                "Tick the model whose conversation you want to shorten.")
            return
        work = [t for t in targets
                if plan_compaction(self.session.history(t.key)).worth_doing]
        if not work:
            budget = self.settings.effective_budget()
            messagebox.showinfo(
                "Nothing to compact",
                explain_no_compaction(
                    self.session.history(targets[0].key), budget))
            return
        self._run(lambda: self._compact(work))

    def _compact(self, targets: list[Target]) -> bool:
        """Worker thread: replace old turns with a summary of them.

        Mutates the session directly, as the orchestrator already does — only
        the notes go through the queue, because only they touch Tk.
        """
        if not targets:
            return False
        def tick(state: str, detail: str = "") -> None:
            self.events.put(("check", {"key": "compact", "state": state,
                                       "detail": detail}))

        writer = targets[0]
        client = self.clients().get(writer.server)
        if client is None or not client.base_url:
            tick("skipped", "no model available to summarise with")
            return False

        compacted = False
        for target in targets:
            history = list(self.session.history(target.key))
            plan_result = plan_compaction(history)
            if not plan_result.worth_doing or self.cancel.is_set():
                tick("skipped", "nothing worth folding")
                continue

            note_id = f"compact{target.key}"
            name = friendly_model_name(target.model) or short_model(target.model)

            def note(text: str, _id: str = note_id) -> None:
                self.events.put(("note", {"text": text, "note_id": _id}))

            note(f"🗜 Compacting the {name} thread…")
            self._step(note_id, "compacting", name,
                       detail=f"{plan_result.folded_count} messages")
            tick("running")
            try:
                reply = client.chat(
                    writer.model,
                    [{"role": "user",
                      "content": build_compaction_prompt(plan_result)}],
                    options={**self._options(700), "temperature": 0.2},
                    think=think_value(writer.model, False),
                    cancel=self.cancel)
            except Exception as exc:
                self._step_done(note_id)
                tick("failed", str(exc)[:80])
                self.runlog.record_failure(writer.model, writer.server,
                                           "compaction", str(exc))
                note(f"⚠ Could not compact — {str(exc)[:120]}")
                self.events.put(("note_close", {"note_id": note_id}))
                continue
            self._step_done(note_id)
            self.runlog.record(reply, purpose="compaction")

            summary = reply.text.strip()
            if self.cancel.is_set() or not summary:
                note("⚠ The summary came back empty — history left as it was.")
                self.events.put(("note_close", {"note_id": note_id}))
                continue

            before = usage(history)
            self.session.conversations[target.key] = apply_compaction(
                history, summary)
            after = usage(self.session.conversations[target.key])
            note(f"🗜 {name}: folded {plan_result.folded_count} earlier "
                 f"message(s) — {before:,} → {after:,} tokens")
            tick("done", f"{before:,} → {after:,} tokens")
            self.events.put(("note_close", {"note_id": note_id}))
            self.events.put(("context_changed", {}))
            compacted = True
        return compacted

    # -------------------------------------------------- replies cut off short
    def _auto_continue(self, target: Target, result) -> bool:
        """Press Continue for the user, or explain why it stopped pressing."""
        if not self.auto_var.get():
            return False
        budget = self.settings.effective_budget()
        history = self.session.history(target.key)
        # Full, and nothing left to fold: continuing from here means deleting
        # the start of the work to make room for the end of it.
        exhausted = (self.session.total_tokens(target.key) >= budget
                     and not should_compact(history, budget))
        decision = self.auto.consider(result.text, time.monotonic(),
                                      context_exhausted=exhausted)
        if not decision:
            self.chat.add_note(
                f"🌙 Auto mode stopped here: {decision.reason}.",
                tag="checklist")
            self._end_auto_run()
            return False

        name = friendly_model_name(target.model) or short_model(target.model)
        self.chat.add_note(
            f"🌙 {name} was cut off — carrying on automatically "
            f"({self.auto.continues}/{self.auto.limits.continues}).")
        # Not fired here: the worker that ran this turn is still alive, and
        # `_continue_reply` refuses to start on top of a running one.  The
        # "finished" event is the first safe moment.
        self.pending_auto = target
        return True

    def _offer_continue(self, target: Target, result) -> None:
        """A capped reply looks identical to a finished one — so say which.

        Continuing an answer that is capped at 192 tokens produces another 192
        tokens that stop just as abruptly.  Offer Continue first once; after
        that, stop pretending the loop terminates and put the real fix — a
        bigger cap — in front.
        """
        label, num_predict, _directive = self.settings.profile()
        cap = ("the length cap" if num_predict < 0
               else f"the {num_predict}-token cap ({label})")
        name = friendly_model_name(target.model) or short_model(target.model)
        streak = self.cutoff_streak.get(target.key, 1)

        self.action_seq += 1
        action_id = f"cut{self.action_seq}"
        keep_going = ("▶ Continue the answer",
                      lambda: self._continue_reply(target, action_id), True)
        raise_it = ("⚙ Raise the limit & ask again",
                    lambda: self._raise_and_retry(target, action_id), True)

        if num_predict < 0:
            options, message = [keep_going], (
                f"⚠ {name} stopped at {cap} — it had not finished.")
        elif streak >= 2:
            options = [raise_it, ("▶ Continue anyway",
                                  lambda: self._continue_reply(target, action_id),
                                  False)]
            message = (
                f"⚠ {name} has now been cut off {streak} times in a row, all "
                f"at {cap}. Continuing just buys another {num_predict} tokens "
                f"that stop the same way — the speed slider is what sets this "
                f"limit.")
        else:
            options, message = [keep_going, (raise_it[0], raise_it[1], False)], (
                f"⚠ {name} stopped at {cap} — it had not finished.")

        self.chat.add_action(message, options, action_id=action_id)

    def _continue_reply(self, target: Target, action_id: str) -> None:
        if self.worker and self.worker.is_alive():
            self.set_status("Still working — press Stop first")
            return
        self.chat.resolve_action(action_id, "▶ Continuing where it left off…")
        prompt = ("Carry on from exactly where you stopped — your last reply "
                  "was cut off mid-answer. Do not repeat any of it and do not "
                  "start again; just continue from the next word.")
        orchestrator = self._make_orchestrator()
        self._run(lambda: orchestrator.broadcast([target], prompt))

    def _raise_and_retry(self, target: Target, action_id: str) -> None:
        if self.worker and self.worker.is_alive():
            self.set_status("Still working — press Stop first")
            return
        history = self.session.history(target.key)
        question = next((m.get("content", "") for m in reversed(history)
                         if m.get("role") == "user"), "")
        if not question:
            self.chat.resolve_action(action_id, "⚠ Nothing to ask again.")
            return

        self.cutoff_streak.pop(target.key, None)
        current = int(self.settings.get("speed_quality", 50))
        raised = min(100, max(current + 25, 60))
        self.speed_scale.set(raised)
        # Don't rely on the Scale's own callback firing — it doesn't, reliably,
        # on a window that has never been mapped.
        self._on_speed_change(raised)
        self.chat.resolve_action(
            action_id,
            f"⚙ Answer style raised to {self.settings.profile()[0]} — asking again.")
        orchestrator = self._make_orchestrator(question)
        self._run(lambda: orchestrator.broadcast([target], question))

    # ------------------------------------------------- asking before guessing
    def _maybe_ask_to_search(self, text: str, mode: str, relay) -> bool:
        """Pause a send that asks for a lookup while search is switched off.

        Answering "what's the latest on X" from a frozen snapshot of the world,
        in the same confident voice as everything else, is the worst thing this
        app can do — the user has no way to tell it happened.  So when the
        message explicitly asks for a lookup, the send stops and asks, rather
        than either silently ignoring the request or silently going online.
        """
        if self.skip_search_prompt:
            self.skip_search_prompt = False
            return False
        if relay or mode != "chat" or not text or self.research_var.get():
            return False
        phrase = wants_web_search(text)
        asked = bool(phrase)
        if not phrase:
            # The more common and more dangerous case: a question that needs
            # a lookup and does not say so.  Answered from training data it
            # comes back confident and possibly years out of date, in exactly
            # the same voice as everything else.
            phrase = looks_time_sensitive(text)
            if phrase and phrase.lower() in self.stale_asked:
                return False
            if phrase:
                self.stale_asked.add(phrase.lower())
        if not phrase:
            return False

        if self.auto_var.get():
            # Nothing may sit waiting for a person who is asleep.
            configured = bool(str(self.settings.get("searxng_url", "")).strip())
            if configured:
                self.research_var.set(True)
                self.auto.note(f"switched web research on to look up “{phrase}”")
                self.chat.add_note(
                    f"🌙 You asked me to “{phrase}” — web research switched "
                    f"on automatically.")
            else:
                self.auto.note(f"answered “{phrase}” from memory "
                               f"(no SearXNG configured)")
                self.chat.add_note(
                    f"🌙 You asked me to “{phrase}”, but no SearXNG is "
                    f"configured — answering from memory instead.")
            return False

        self.action_seq += 1
        action_id = f"search{self.action_seq}"
        message = (
            f"🔍 You asked me to “{phrase}”, but web research is switched off "
            f"— the models would answer from memory without saying so."
            if asked else
            f"🕰 This asks about something that changes (“{phrase}”), and web "
            f"research is off — so the answer will come from the model's "
            f"training data, which stopped somewhere in the past. It will "
            f"sound just as certain either way.")
        self.chat.add_action(
            message,
            [("🔍 Enable web research & send",
              lambda: self._enable_and_send(action_id), True),
             ("Answer without searching",
              lambda: self._send_without_search(action_id), False)],
            action_id=action_id)
        self.set_status("Waiting on you — search, or answer from memory?")
        return True

    def _enable_and_send(self, action_id: str) -> None:
        if not str(self.settings.get("searxng_url", "")).strip():
            self.chat.resolve_action(
                action_id, "🔍 Web research needs a SearXNG URL first.")
            messagebox.showinfo(
                "Set up SearXNG first",
                "Web research uses your own SearXNG instance.\n\n"
                "Enter its URL in ⚙ Settings (e.g. http://192.168.1.20:8080) "
                "and press Connect, then send the message again.")
            self.open_settings()
            return
        self.research_var.set(True)
        self.chat.resolve_action(action_id, "🔍 Web research switched on.")
        self.skip_search_prompt = True
        self.send()

    def _send_without_search(self, action_id: str) -> None:
        self.chat.resolve_action(
            action_id, "💬 Answering from memory — no search was made.")
        self.skip_search_prompt = True
        self.send()

    # ------------------------------------------------------------ learning
    def _learn(self, question: str, answer: str, target: Target | None) -> None:
        """Mine one exchange for durable lessons (worker thread).

        Learning is the one toggle that can still be obeyed after Send,
        because it runs *after* the answer rather than shaping the request.
        Web research and reasoning are already part of an HTTP request in
        flight by the time a reply is on screen; this has not happened yet.

        Only the cancelling direction is honoured.  Switching it on midway
        would run a step the checklist never listed, which reads as the app
        doing something behind your back.
        """
        def tick(state: str, detail: str = "") -> None:
            self.events.put(("check", {"key": "learn", "state": state,
                                       "detail": detail}))

        if not self.learning_wanted:
            tick("skipped", "switched off while the answer was being written")
            return
        if target is None or not worth_learning_from(question, answer):
            tick("skipped", "nothing durable in this exchange")
            return
        client = self.clients().get(target.server)
        if client is None or not client.base_url:
            tick("skipped", "no model available")
            return
        tick("running")

        def note(text: str) -> None:
            self.events.put(("note", {"text": text, "note_id": "learning"}))

        name = friendly_model_name(target.model) or short_model(target.model)
        self._step("learning", "learning", name)
        try:
            reply = client.chat(
                target.model,
                [{"role": "user",
                  "content": build_extraction_prompt(question, answer)}],
                options={**self._options(200), "temperature": 0.2},
                think=think_value(target.model, False),
                cancel=self.cancel)
        except Exception:
            self._step_done("learning")
            tick("failed", "the model could not be reached")
            return                      # learning must never break a chat
        self._step_done("learning")
        self.runlog.record(reply, purpose="learning")
        if self.cancel.is_set():
            return

        kept = self.knowledge.add_many(parse_lessons(reply.text),
                                       source=question,
                                       project=self._project_name())
        tick("done", f"learned {len(kept)}" if kept else "nothing new")
        if kept:
            topics = ", ".join(sorted({lesson.topic for lesson in kept}))
            note(f"🧠 Learned {len(kept)} new thing(s) about {topics}")
            self.events.put(("note_close", {"note_id": "learning"}))
        self.events.put(("knowledge_changed", {}))

    def _triage(self, question: str, attachments: list[dict],
                history: list[dict], target: Target | None) -> Triage:
        """Ask a model whether this actually needs the web (worker thread).

        On any failure we fall back to searching, because the triage step is
        an optimisation — it should never be the reason a genuine lookup does
        not happen.
        """
        client = self.clients().get(target.server) if target else None
        if client is None or not client.base_url:
            return Triage(True, query=question, reason="no model to ask")
        material = describe_material(attachments, len(history))
        name = friendly_model_name(target.model) or short_model(target.model)
        self._step("triage", "reading", name)
        try:
            reply = client.chat(
                target.model,
                [{"role": "user",
                  "content": build_triage_prompt(question, material)}],
                options={**self._options(64), "temperature": 0.1},
                think=think_value(target.model, False),
                cancel=self.cancel)
        except Exception as exc:
            self._step_done("triage")
            self.runlog.record_failure(target.model, target.server, "triage",
                                       str(exc))
            return Triage(True, query=question, reason="triage call failed")
        self._step_done("triage")
        self.runlog.record(reply, purpose="triage")
        return parse_triage(reply.text, fallback_query=question)

    def _research(self, question: str, target: Target | None,
                  history: list[dict],
                  attachments: list[dict] | None = None) -> str | None:
        """Run the SearXNG research step (worker thread). None on failure.

        Follow-ups are resolved against the conversation first, so "what is
        the 0-60mph?" searches for the car being discussed rather than for the
        definition of the term.
        """
        def note(text: str) -> None:
            self.events.put(("note", {"text": text, "note_id": "research"}))

        def tick(state: str, detail: str = "") -> None:
            self.events.put(("check", {"key": "search", "state": state,
                                       "detail": detail}))

        tick("running")

        # Look at what we already have before reaching for the web.  Without
        # this, "take a look at my app and give me some tips" with a folder
        # attached goes to the search engine as those literal words.
        attachments = list(attachments or [])
        # Files are attached and nothing in the message asks to look anything
        # up: that is not a judgement call, so do not spend a model round trip
        # making it.  Asking a 32B model to decide cost thirty seconds and it
        # still went and read two SEO pages about debugging.
        assess = self.settings.get("assess_before_search", True)
        if assess and attachments and not wants_web_search(question):
            names = ", ".join(str(a.get("name", "")) for a in attachments)
            note(f"📄 Working from {names} — no web search needed")
            tick("skipped", "not needed — the files answer it")
            self.events.put(("note_close", {"note_id": "research"}))
            return None

        if assess and worth_triaging(question, attachments, history):
            note("📄 Reading what you've given me…")
            verdict = self._triage(question, attachments, history, target)
            if self.cancel.is_set():
                return None
            note(verdict.note)
            if not verdict.needs_search:
                tick("skipped", "not needed")
                self.events.put(("note_close", {"note_id": "research"}))
                return None
            question = verdict.query or question
            history = []          # the query is already standalone

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
                tick("done", f"reused earlier research from {hit.age_phrase}")
                self.events.put(("note_close", {"note_id": "research"}))
                return hit.block

        client = SearxngClient(str(self.settings.get("searxng_url", "")))
        try:
            block = gather(client, query, emit=note)
        except ResearchError as exc:
            note(f"⚠ Web research skipped — {exc}")
            tick("failed", str(exc)[:80])
            self.events.put(("note_close", {"note_id": "research"}))
            return None

        if self.settings.get("research_cache", True):
            self.cache.put(query, block)
        tick("done", f"searched for “{query[:50]}”")
        self.events.put(("note_close", {"note_id": "research"}))
        return block

    def _deep_research(self, orchestrator, target: Target | None,
                       question: str) -> None:
        """Run the full research pipeline on one model (worker thread)."""
        if target is None:
            return
        client = SearxngClient(str(self.settings.get("searxng_url", "")))

        def search(query: str) -> list[dict]:
            return client.search(query, max_results=5)

        try:
            orchestrator.research(target, question, search_fn=search,
                                  fetch_fn=fetch_page_text)
        except ResearchError as exc:
            self.events.put(("note", {"text": f"⚠ Research stopped — {exc}",
                                      "note_id": "research"}))
            self.events.put(("note_close", {"note_id": "research"}))

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
                options={**self._options(48), "temperature": 0.1},
                think=think_value(target.model, False),
                cancel=self.cancel)
            return result.text

        return rewrite

    def _interject(self) -> None:
        """Hand a mid-conversation message to whoever speaks next."""
        text = self._input_text()
        if not text:
            self.set_status("Type something to join the conversation")
            return
        self.input.delete("1.0", tk.END)
        self.chat.add_user(text)
        self.interjections.put(text)
        self.set_status("Added — the next model to speak will see it")

    def _pending_interjections(self) -> list[str]:
        """Drain anything typed since the last turn (called from the worker)."""
        messages = []
        while True:
            try:
                messages.append(self.interjections.get_nowait())
            except queue.Empty:
                return messages

    def _compose(self, text: str) -> str:
        if not self.attachments:
            return text
        # A folder block already carries its own manifest and per-file fences,
        # so wrapping it in another one just nests code blocks.
        parts = [a["content"] if a["name"].endswith("/")
                 else f"[Attached file: {a['name']}]\n```\n{a['content']}\n```"
                 for a in self.attachments]
        if text:
            parts.append(text)
        return "\n\n".join(parts)

    def _rounds(self) -> int:
        try:
            return max(1, min(6, int(self.rounds.get())))
        except ValueError:
            return 2

    def _turns(self) -> int:
        try:
            return max(2, min(60, int(self.turns.get())))
        except ValueError:
            return 12

    def _make_orchestrator(self, question: str = "",
                           prompt_tokens: int = 0) -> Orchestrator:
        self.cancel = threading.Event()
        return Orchestrator(
            clients=self.clients(prompt_tokens),
            session=self.session,
            emit=lambda kind, **payload: self.events.put((kind, payload)),
            cancel=self.cancel,
            options=self.settings.sampling_options(prompt_tokens),
            budget_tokens=self.settings.effective_budget(),
            system_prompt=self._system_prompt(question),
            think=bool(self.thinking_var.get()),
            compactor=(self._compact
                       if self.settings.get("auto_compact", True) else None),
            editing=(self.project is not None
                     and bool(self.settings.get("allow_edits", True))))

    def _project_name(self) -> str:
        """What to file a lesson under, so it comes back in the right place.

        The folder's name rather than its full path: the same project moved
        or checked out somewhere else is still the same project, and a path
        that changes turns every earlier lesson into one nobody can reach.
        """
        return Path(self.project).name if self.project is not None else ""

    def _system_prompt(self, question: str = "") -> str:
        """Persona, recalled lessons, the user's own prompt, then the slider."""
        parts = []
        if self.settings.get("conversational", True):
            parts.append(CONVERSATIONAL_PROMPT)
        if question and self.learning_var.get():
            parts.append(format_for_prompt(self.knowledge.recall(
                question, project=self._project_name())))
        if self.project is not None and self.settings.get("allow_edits", True):
            parts.append(edit_tools.instructions())
        parts.append(str(self.settings.get("system_prompt", "")).strip())
        parts.append(self.settings.style_directive())
        return "\n\n".join(part for part in parts if part)

    def _run(self, job) -> None:
        if not self.conversing:
            self.send_button.config(state="disabled")
        self.set_status("Talking…" if self.conversing else "Working…")

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
        """Ask for a stop, and be honest about what that can and cannot do.

        Cancellation is checked between the lines a server streams back, so a
        server that has gone quiet cannot be interrupted until it says
        something.  Claiming otherwise is how "Stop" ends up looking broken.
        """
        # A download runs on its own thread, not `self.worker`, so this has
        # to come before the "nothing is running" guard — otherwise Stop is
        # inert for the one job most likely to need stopping.
        if self.pulling:
            self.pull_cancel.set()
            self.set_status(f"Stopping the download of {self.pulling}…")
            return
        # A queue of changes waiting on *you* has no worker running, so this
        # has to come before the "nothing is running" guard — otherwise Stop
        # is inert for a run that is only paused for a decision, which is
        # exactly when someone reaches for it.
        if self.change_queue or self.menu_left:
            left = len(self.change_queue) + len(self.menu_left)
            self.change_queue = []
            self.change_target = None
            self.menu_left = []
            self.menu_target = None
            self.chat.add_note(
                f"⏹ Stopped — the remaining {left} change(s) were not asked "
                f"for. Nothing further was written.")
            self.set_status(f"Stopped — {left} change(s) abandoned")
            if not (self.worker and self.worker.is_alive()):
                return
        if not (self.worker and self.worker.is_alive()):
            self.set_status("Nothing is running")
            return
        self.cancel.set()
        if self.auto_var.get():
            # Pressing Stop during an unattended run means stop, not pause.
            self.auto_var.set(False)
            self._end_auto_run()
        self.pending_auto = None
        self.stopping_since = time.monotonic()
        self.activity.clear()
        self.set_status("Stopping — closing open connections…")
        self.chat.add_note("⏹ Stop requested — it takes effect as soon as the "
                           "server sends its next chunk.")

    # ------------------------------------------------------- live progress
    def _tick(self) -> None:
        """Keep the status line honest while work is in flight.

        "Working…" tells you nothing: not which model, not what step, not how
        long, and — the part that actually matters when you are staring at a
        spinner — not whether anything is still arriving.
        """
        now = time.monotonic()
        if self.activity.busy():
            extra = max(0, len(self.activity.items) - 1)
            self.status.set(describe(self.activity.current(), now, extra))
            self._redraw_checklist(now)
            self._update_context_label()
            # A crash halfway through a ten-minute answer should still leave
            # the question behind, not just the last finished turn.
            if now - self.last_autosave > 10:
                self.last_autosave = now
                self._autosave()
            for key, activity in self.activity.stalled(now):
                activity.warned = True
                text = stall_note(activity, now)
                # Two steps of the same run are often the same model, so the
                # same sentence twice reads as a glitch rather than as news.
                if text in self.said_stalls:
                    continue
                self.said_stalls.add(text)
                self.chat.add_note(text, note_id=f"stall{key}")
                self._check_placement(activity.label)
        elif self.stopping_since is not None:
            waited = time.monotonic() - self.stopping_since
            if waited > 3:
                self.status.set(
                    f"Stopping — waiting for the server to answer "
                    f"({int(waited)}s). It cannot be interrupted mid-request.")
        self.tick_id = self.root.after(500, self._tick)

    def _show_plan(self, steps: list) -> None:
        """Replace the pipeline checklist with the model's own plan."""
        self.checklist = Checklist(title="The plan")
        for index, step in enumerate(steps):
            self.checklist.add(f"step{index}", step)
        self.checklist.add("summary", "Pull it together into the final answer")
        self._redraw_checklist(time.monotonic())

    def _check_placement(self, label: str) -> None:
        """Ask Ollama whether this model is actually on the GPU.

        `ollama ps` answers this in one line and almost nobody knows to run
        it.  When a model will not fit in VRAM Ollama does not refuse — it
        loads the whole thing into system memory and runs it on the CPU,
        perhaps twenty times slower, with nothing anywhere saying so.
        """
        targets = self.selected_targets()
        if not targets:
            return

        def look():
            clients = self.clients()
            for target in targets:
                client = clients.get(target.server)
                if client is None or not client.base_url:
                    continue
                entry = find_loaded(client.running_models(), target.model)
                if entry is None:
                    continue
                seen = observed_vram(entry)
                if seen > self.vram_seen:
                    self.events.put(("vram_seen", {"bytes": seen}))
                note = placement_note(
                    friendly_model_name(target.model) or short_model(target.model),
                    entry)
                if note:
                    self.events.put(("note", {"text": note,
                                              "note_id": f"where{target.key}"}))

        threading.Thread(target=look, daemon=True).start()

    def _redraw_checklist(self, now: float) -> None:
        if self.checklist and self.checklist.steps:
            self.chat.add_checklist(self.checklist.render(now), "run")

    def _tick_step(self, key: str, state: str, detail: str = "") -> None:
        """Mark one checklist item, from the main thread."""
        if self.checklist is None:
            return
        now = time.monotonic()
        if state == "running":
            self.checklist.start(key, now=now, detail=detail)
        else:
            self.checklist.finish(key, now=now, detail=detail, state=state)
        self._redraw_checklist(now)

    def _step(self, key: str, step: str, label: str = "",
              detail: str = "") -> None:
        """Announce a step from a worker thread (queued, never touching Tk)."""
        self.events.put(("step", {"key": key, "step": step, "label": label,
                                  "detail": detail}))

    def _step_done(self, key: str) -> None:
        self.events.put(("step_done", {"key": key}))

    # ---------------------------------------------------------- event pump
    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self.pump_id = self.root.after(40, self._pump)

    def _handle(self, kind: str, payload: dict) -> None:
        if kind == "turn_start":
            target = payload["target"]
            self.chat.start_stream(payload["stream_id"], payload["heading"],
                                   speaker=target.key)
            name = friendly_model_name(target.model) or short_model(target.model)
            activity = self.activity.start(
                payload["stream_id"], "thinking", name, now=time.monotonic())
            # How much prompt this model has to chew before its first word.
            activity.prompt_tokens = self.session.total_tokens(target.key)
            self._tick_step(f"model:{name}", "running")
        elif kind == "thought":
            # Reasoning is progress, so it counts as activity — otherwise a
            # model thinking hard for two minutes trips the stall warning.
            self.chat.append_thought(payload["stream_id"], payload["text"])
            self.activity.touch(payload["stream_id"], time.monotonic())
            self.activity.set_step(payload["stream_id"], "reasoning")
        elif kind == "token":
            self.chat.append_stream(payload["stream_id"], payload["text"])
            # Tokens arrive as text chunks, not counted tokens; roughly one
            # word each is close enough for a progress readout.
            self.activity.touch(payload["stream_id"], time.monotonic())
        elif kind == "turn_end":
            result = payload["result"]
            # [END] is how a model votes to finish a conversation; it's a
            # signal to the orchestrator, not something to show the user.
            spoken = END_MARKER.sub("", result.text).strip()
            self.chat.end_stream(payload["stream_id"], spoken,
                                 meta=self._meta(result),
                                 thought_s=result.thought_s)
            self.activity.finish(payload["stream_id"])
            self.chat.close_note(f"stall{payload['stream_id']}")
            self.runlog.record(result, purpose=self.current_mode())
            self._tick_step(
                f"model:{friendly_model_name(payload['target'].model) or short_model(payload['target'].model)}",
                "done", "cut off early" if result.truncated else "")
            self._check_for_edits(result.text, truncated=result.truncated,
                                  target=payload["target"])
            key = payload["target"].key
            if result.truncated and not self.conversing:
                self.cutoff_streak[key] = self.cutoff_streak.get(key, 0) + 1
                if not self._auto_continue(payload["target"], result):
                    self._offer_continue(payload["target"], result)
            elif not result.cancelled:
                self.cutoff_streak.pop(key, None)
        elif kind == "turn_error":
            self.chat.fail_stream(payload["stream_id"],
                                  f"{payload['target'].label}: {payload['message']}")
            self.activity.finish(payload["stream_id"])
            self.chat.close_note(f"stall{payload['stream_id']}")
            target = payload["target"]
            self.runlog.record_failure(target.model, target.server,
                                       self.current_mode(), payload["message"])
            self._tick_step(
                f"model:{friendly_model_name(target.model) or short_model(target.model)}",
                "failed", str(payload["message"])[:60])
            if "timed out" in str(payload["message"]).lower():
                self.chat.add_note("💡 " + suggest_after_timeout(
                    target.model,
                    self.session.total_tokens(target.key),
                    int(self.settings.timeout_for(
                        self.session.total_tokens(target.key),
                        self.runlog.prompt_rate()))))
        elif kind == "note":
            self.chat.add_note(payload["text"], payload.get("note_id"))
        elif kind == "note_close":
            self.chat.close_note(payload["note_id"])
        elif kind == "status":
            self.set_status(payload["text"])
        elif kind == "context_changed":
            self._update_context_label()
        elif kind == "vram_seen":
            self.vram_seen = max(self.vram_seen, int(payload["bytes"]))
            for server in ("local", "host"):
                names = [m for (s, m) in self.model_vars if s == server]
                if names:
                    self._populate(server, sorted(names))
            self._apply_selection()
        elif kind == "step":
            self.activity.start(payload["key"], payload["step"],
                                payload.get("label", ""),
                                now=time.monotonic(),
                                detail=payload.get("detail", ""))
        elif kind == "step_done":
            self.activity.finish(payload["key"])
            self.chat.close_note(f"stall{payload['key']}")
        elif kind == "check":
            self._tick_step(payload["key"], payload["state"],
                            payload.get("detail", ""))
        elif kind == "plan":
            self._show_plan(payload["steps"])
        elif kind == "models_loaded":
            self.model_sizes.update(payload.get("sizes") or {})
            self._models_loaded(payload["server"], payload["names"],
                                payload.get("version", ""))
        elif kind == "pull_progress":
            self._pull_progress(payload["text"])
        elif kind == "pull_done":
            self._pull_done(payload["server"], payload["model"],
                            payload.get("error", ""), payload.get("size", 0))
        elif kind == "cards":
            self._cards_found(payload["cards"])
        elif kind == "models_failed":
            self._models_failed(payload["server"], payload["message"],
                                payload["interactive"])
        elif kind == "folder_scanned":
            self._folder_scanned(payload["path"], payload["survey"])
        elif kind == "folder_read":
            self._folder_read(payload["path"], payload["block"],
                              payload["selection"], payload.get("texts"))
        elif kind == "folder_failed":
            self.set_status("Ready")
            messagebox.showerror("Could not read folder",
                                 f"{payload['path']}\n\n{payload['message']}")
        elif kind == "report":
            # the finished write-up: answer, source list, open questions
            self.chat.add_assistant(
                f"{payload['target'].short} · research report",
                payload["text"])
        elif kind == "tests_done":
            self._tests_done(payload["outcome"], payload["stamp"])
        elif kind == "menu_ready":
            # Routed through the ordinary check so the menu parse, the
            # verification and the trace all run exactly as they would for
            # any other reply.
            self._check_for_edits(payload["reply"], target=payload["target"])
        elif kind == "changes_ready":
            # The scoped replies, run back through the ordinary path: the
            # diff review is the same window whether a block arrived on the
            # first try or the second.  `second_pass` stops it offering to
            # go round again, which would be a loop with a button on it.
            self._check_for_edits(payload["reply"], target=payload["target"],
                                  second_pass=True)
        elif kind == "knowledge_changed":
            self._refresh_knowledge_link()
        elif kind == "fatal":
            self.chat.add_error(payload["message"])
        elif kind == "finished":
            self._autosave()
            if self.checklist:
                self.checklist.skip_remaining("not reached")
                self._redraw_checklist(time.monotonic())
                self.checklist = None
                self.chat.close_checklist("run")
            self.conversing = False
            self.activity.clear()
            self.stopping_since = None
            self.send_button.config(state="normal")
            self.set_status("Ready")
            self._update_context_label()
            if self.queued_message:
                self.root.after(120, self._send_queued)
            elif self.pending_auto is not None:
                target, self.pending_auto = self.pending_auto, None
                if self.auto_var.get() and not self.cancel.is_set():
                    self.root.after(120, lambda t=target: self._continue_reply(
                        t, f"auto{self.auto.continues}"))
                else:
                    # Stop was pressed, or auto was switched off mid-run.
                    self._end_auto_run()

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

    # ------------------------------------------------------------ recovery
    def _autosave(self) -> None:
        """Keep the unfinished conversation, quietly and without complaint."""
        try:
            recovery.write(self.session,
                           selected=[t.key for t in self.selected_targets()],
                           mode=self.current_mode(),
                           attachments=self.attachment_names)
        except Exception:
            pass          # insurance must never be the thing that breaks

    def _offer_recovery(self) -> None:
        """Put back whatever the last session was in the middle of.

        No question asked.  Being greeted by a dialog about a file you have
        never heard of is a worse start than simply finding your work where
        you left it — and ✨ New chat is right there if it was not wanted.
        """
        try:
            found = recovery.read()
        except Exception:
            return
        if found is not None:
            self._restore(found)

    def _restore(self, found) -> None:
        self.session = found.session
        self.pending_selection = set(found.selected)
        self._apply_selection()
        for key, label in MODES.items():
            if key == found.mode:
                self.mode.set(label)
                self._sync_mode_bar()
                break
        self._redraw()
        self.chat.add_note(
            f"🔄 Picked up where you left off — {found.messages} message(s) "
            f"from {found.when()}. 💾 Save to keep it properly, or ✨ New chat "
            f"to start over.")
        self.set_status(f"Restored your unfinished chat from {found.when()}")
        self._update_context_label()

    # --------------------------------------------------------- chat files
    def new_chat(self) -> None:
        if self.project is not None and not self.session.is_empty():
            self._start_handoff()
        self.cancel.set()
        self.project = None
        self.pending_edits = {}
        self.pending_changes = {}
        self.project_files = []
        self.sent_files = set()
        self.last_edit_target = None
        self.auto_apply_edits = False
        self.pending_flaws = {}
        self.change_queue = []
        self.change_target = None
        self.menu_left = []
        self.menu_target = None
        # Said once per chat, so a new chat is allowed to say it again — the
        # folder that was attached to the old one is not attached to this.
        self.warned_no_folder = False
        self.pending_selection = set()
        self.attachment_names = []
        recovery.discard()
        self.session = Session()
        self.chat.clear()
        self.chat.add_note("✨ New chat started.")
        self.set_status("New chat started")
        self._update_context_label()

    def _start_handoff(self) -> None:
        """Run the hand-off on a worker so the UI is never held up by it."""
        if self.worker and self.worker.is_alive():
            return
        project = self.project
        self.set_status("Leaving a note for the next chat…")

        def work() -> None:
            try:
                self._write_handoff()
            finally:
                self.events.put(("status", {"text": "Ready"}))

        # Not through `_run`: this outlives the chat it describes, and must
        # not be cancelled by the New chat that triggered it.
        self.cancel = threading.Event()
        self.project = project
        threading.Thread(target=work, daemon=True).start()

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
            saved = self.session.save(path, selected)
            # There is a real file with this conversation in it now, so a
            # second copy in the temp directory is clutter, not insurance.
            recovery.discard()
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
        # Remembered rather than just applied: the model lists arrive from the
        # servers a moment later, and opening a chat before they land used to
        # rebuild the checkboxes with nothing ticked, quietly losing the
        # selection the file had recorded.
        self.pending_selection = set(selected)
        self._apply_selection()
        self._redraw()
        self.set_status(f"Loaded: {path}")
        self._update_context_label()

    def _apply_selection(self) -> None:
        if not self.pending_selection:
            return
        for pair, var in self.model_vars.items():
            var.set(make_key(*pair) in self.pending_selection)

    def _redraw(self) -> None:
        self.chat.clear()
        for key, messages in self.session.conversations.items():
            server, model = split_key(key)
            self.chat.add_note(f"───  {model} ({server})  ───")
            for message in messages:
                role = message.get("role")
                content = message.get("content", "")
                if role == "user":
                    # Show attachments as the chips they were sent as, not as
                    # a hundred thousand characters of pasted source code.
                    names, typed = describe_attachments(content)
                    if names:
                        details = attachment_details(content)
                        self.chat.add_user(
                            typed or "(sent attached files)",
                            "  ".join(f"📎 {name} · ≈{tokens:,} tokens"
                                      for name, tokens in details) or
                            "  ".join(f"📎 {name}" for name in names))
                    else:
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

    def open_runlog(self) -> None:
        RunLogWindow(self.root, self.runlog)

    def open_edit_debug(self) -> None:
        EditDebugWindow(self.root, self.edit_debug)

    def open_knowledge(self) -> None:
        KnowledgeWindow(self.root, self.knowledge,
                        on_change=self._refresh_knowledge_link)

    def on_close(self) -> None:
        """Throwaway chats shouldn't leave cached research behind."""
        self.cancel.set()
        if self.bridge is not None:
            # Takes the handshake file with it, so nothing is left pointing at
            # a port that has gone.
            self.bridge.stop()
            self.bridge = None
        for attribute in ("pump_id", "tick_id"):
            handle = getattr(self, attribute, None)
            if handle is not None:
                try:
                    self.root.after_cancel(handle)
                except tk.TclError:
                    pass
                setattr(self, attribute, None)
        try:
            if self.session.path is not None:
                self.cache.keep_session()
                recovery.discard()
            else:
                self.cache.discard_session()
                # No prompt on the way out — closing stays cheap — but an
                # unsaved conversation is kept so the next launch can offer it.
                self._autosave()
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
