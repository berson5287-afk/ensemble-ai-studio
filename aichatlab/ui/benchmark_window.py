"""Benchmark window: one prompt, many models, side-by-side timings."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Sequence
from tkinter import filedialog, messagebox, ttk

from ..benchmark import BenchmarkRow, run_benchmark, summarise, write_csv
from ..client import OllamaClient
from ..orchestrator import Target
from .theme import FONT, FONT_BOLD, FONT_SMALL, FONT_TITLE, MUTED, PANEL, flat_button

COLUMNS = (
    ("model", "Model", 190),
    ("server", "Server", 70),
    ("runs", "Runs", 55),
    ("median_s", "Median s", 90),
    ("speed", "Tokens/s", 90),
    ("failures", "Failed", 65),
)

DEFAULT_PROMPT = ("Explain what a hash table is and when you would use one, "
                  "in about 100 words.")


class BenchmarkWindow(tk.Toplevel):
    def __init__(self, master, clients: dict[str, OllamaClient],
                 targets: Sequence[Target], options: dict | None = None):
        super().__init__(master)
        self.clients = clients
        self.targets = list(targets)
        self.options = options or {}
        self.rows: list[BenchmarkRow] = []
        self.events: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None

        self.title("Benchmark")
        self.configure(bg=PANEL)
        self.geometry("760x520")
        self.minsize(620, 420)
        self.transient(master)

        self._build()
        self._pump()
        self.protocol("WM_DELETE_WINDOW", self._close)

    # -- layout ------------------------------------------------------------
    def _build(self) -> None:
        header = tk.Frame(self, bg=PANEL, padx=16, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="📊 Benchmark", font=FONT_TITLE, bg=PANEL,
                 fg="#1f2430").pack(side="left")
        tk.Label(header, text=f"{len(self.targets)} model(s) selected",
                 font=FONT_SMALL, bg=PANEL, fg=MUTED).pack(side="left", padx=10)

        body = tk.Frame(self, bg=PANEL, padx=16)
        body.pack(fill="x")

        tk.Label(body, text="Prompt", font=FONT_BOLD, bg=PANEL,
                 fg="#1f2430").pack(anchor="w")
        self.prompt = tk.Text(body, height=3, font=FONT, wrap="word",
                              relief="solid", bd=1)
        self.prompt.insert("1.0", DEFAULT_PROMPT)
        self.prompt.pack(fill="x", pady=(2, 8))

        controls = tk.Frame(body, bg=PANEL)
        controls.pack(fill="x", pady=(0, 8))
        tk.Label(controls, text="Runs per model:", font=FONT, bg=PANEL,
                 fg="#1f2430").pack(side="left")
        self.repeats = tk.StringVar(value="1")
        tk.Spinbox(controls, from_=1, to=10, width=4, textvariable=self.repeats,
                   font=FONT).pack(side="left", padx=(6, 16))

        self.run_button = flat_button(controls, "▶ Run", self._start, primary=True)
        self.run_button.pack(side="left")
        self.stop_button = flat_button(controls, "⏹ Stop", self._stop)
        self.stop_button.pack(side="left", padx=6)
        self.stop_button.config(state="disabled")
        self.export_button = flat_button(controls, "⬇ Export CSV", self._export)
        self.export_button.pack(side="right")
        self.export_button.config(state="disabled")

        table_frame = tk.Frame(self, bg=PANEL, padx=16, pady=8)
        table_frame.pack(fill="both", expand=True)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Bench.Treeview", font=FONT, rowheight=24,
                        fieldbackground="white")
        style.configure("Bench.Treeview.Heading", font=FONT_BOLD)

        self.tree = ttk.Treeview(
            table_frame, columns=[c[0] for c in COLUMNS], show="headings",
            style="Bench.Treeview")
        for key, title, width in COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width,
                             anchor="w" if key in ("model", "server") else "e")
        scroll = ttk.Scrollbar(table_frame, orient="vertical",
                               command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=16, pady=(0, 4))

        self.status = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status, anchor="w", font=FONT_SMALL,
                 bg=PANEL, fg=MUTED, padx=16, pady=6).pack(fill="x")

    # -- running -----------------------------------------------------------
    def _start(self) -> None:
        if not self.targets:
            messagebox.showwarning(
                "No models", "Tick some models in the main window first.",
                parent=self)
            return
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            return

        try:
            repeats = max(1, int(self.repeats.get()))
        except ValueError:
            repeats = 1

        self.rows = []
        self.tree.delete(*self.tree.get_children())
        self.cancel = threading.Event()
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.export_button.config(state="disabled")
        self.progress.config(value=0, maximum=len(self.targets) * repeats)

        def worker():
            run_benchmark(
                self.clients, self.targets, prompt, repeats=repeats,
                emit=lambda kind, **payload: self.events.put((kind, payload)),
                cancel=self.cancel, options=self.options)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.cancel.set()
        self.status.set("Stopping…")

    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self.after(80, self._pump)

    def _handle(self, kind: str, payload: dict) -> None:
        if kind == "progress":
            self.status.set(payload.get("text", ""))
        elif kind == "row":
            self.rows.append(payload["row"])
            self.progress.config(value=payload["done"])
            self._refresh_table()
        elif kind == "finished":
            self.run_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.export_button.config(state="normal" if self.rows else "disabled")
            failed = sum(1 for row in self.rows if not row.ok)
            self.status.set(
                f"Done — {len(self.rows)} run(s)"
                + (f", {failed} failed" if failed else ""))

    def _refresh_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for summary in summarise(self.rows):
            self.tree.insert("", "end", values=(
                summary.model,
                summary.server,
                summary.runs,
                f"{summary.median_seconds:.2f}" if summary.median_seconds else "—",
                f"{summary.median_speed:.1f}" if summary.median_speed else "—",
                summary.failures or "",
            ))

    # -- export ------------------------------------------------------------
    def _export(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export benchmark results", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")], parent=self)
        if not path:
            return
        try:
            write_csv(self.rows, path)
            self.status.set(f"Exported to {path}")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)

    def _close(self) -> None:
        self.cancel.set()
        self.destroy()
