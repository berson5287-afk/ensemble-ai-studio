"""Multi-agent orchestration patterns.

This module is deliberately free of any UI code: it takes clients, a session
and an `emit` callback, and runs blocking work that the caller is expected to
put on a worker thread.  That separation is what makes the patterns below
testable without a display.

Patterns implemented, with the names they usually go by:

* broadcast      — fan one prompt out to N models in parallel
* relay          — one model puts a question to another
* debate         — N models argue over R rounds, a judge synthesises consensus
                   (the "mixture-of-agents" shape)
* critique chain — draft, then critique, then revise
* judge panel    — N models answer blind, a separate model scores them
                   ("LLM-as-judge")
"""

from __future__ import annotations

import itertools
import re
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .client import ChatResult, OllamaClient, OllamaError
from .formatting import friendly_model_name
from .session import Session, make_key, short_model

Emit = Callable[..., None]

MAX_PARALLEL = 8


@dataclass(frozen=True)
class Target:
    """One model on one server."""

    server: str
    model: str

    @property
    def key(self) -> str:
        return make_key(self.server, self.model)

    @property
    def label(self) -> str:
        """Readable name, e.g. "Qwen2.5 32B · host"."""
        name = friendly_model_name(self.model) or self.model
        return f"{name} · host" if self.server == "host" else name

    @property
    def short(self) -> str:
        return friendly_model_name(self.model) or short_model(self.model)

    @property
    def raw_label(self) -> str:
        """The exact tag, for places where precision beats readability."""
        return f"{self.model} ({self.server})" if self.server == "host" else self.model


MODES = {
    "chat": "Chat — send to every selected model",
    "debate": "Debate — models discuss, a judge sums up",
    "critique": "Critique chain — draft, critique, revise",
    "judge": "Judge panel — models answer, a judge scores",
}


RELAY_PATTERN = re.compile(r"^\s*([\w.\-:]+)\s+ask\s+([\w.\-:]+)\s+(.+)$",
                           re.IGNORECASE | re.DOTALL)


def find_target(fragment: str, available: Sequence[Target]) -> Target | None:
    """Resolve a typed model name, preferring exact matches then prefixes."""
    fragment = fragment.strip().lower()
    if not fragment:
        return None
    for target in available:
        if target.model.lower() == fragment:
            return target
    for target in available:
        if target.model.lower().startswith(fragment):
            return target
    return None


def parse_relay(text: str, available: Sequence[Target]):
    """Parse "gemma2 ask llama3 <question>" into (source, target, question)."""
    match = RELAY_PATTERN.match(text or "")
    if not match:
        return None
    source = find_target(match.group(1), available)
    target = find_target(match.group(2), available)
    question = match.group(3).strip()
    if not source or not target or not question or source == target:
        return None
    return source, target, question


class Orchestrator:
    """Runs one orchestration pattern to completion."""

    def __init__(self, clients: dict[str, OllamaClient], session: Session,
                 emit: Emit, cancel: threading.Event | None = None,
                 options: dict | None = None, budget_tokens: int = 6000,
                 system_prompt: str = "") -> None:
        self.clients = clients
        self.session = session
        self.emit = emit
        self.cancel = cancel or threading.Event()
        self.options = options or {}
        self.budget_tokens = budget_tokens
        self.system_prompt = system_prompt
        self._stream_ids = itertools.count(1)

    # -- helpers -----------------------------------------------------------
    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

    def _client(self, target: Target) -> OllamaClient:
        client = self.clients.get(target.server)
        if client is None or not client.base_url:
            raise OllamaError(f"The {target.server} server is not configured.")
        return client

    def _turn(self, target: Target, messages: list[dict],
              heading: str | None = None) -> ChatResult | None:
        """Run one model turn, streaming tokens out through `emit`."""
        if self.cancelled:
            return None

        stream_id = next(self._stream_ids)
        self.emit("turn_start", stream_id=stream_id, target=target,
                  heading=heading or target.label)
        try:
            result = self._client(target).chat(
                target.model, messages, options=self.options,
                on_token=lambda piece: self.emit("token", stream_id=stream_id,
                                                 text=piece),
                cancel=self.cancel)
        except Exception as exc:
            self.emit("turn_error", stream_id=stream_id, target=target,
                      message=str(exc))
            return None

        result.server = target.server
        self.emit("turn_end", stream_id=stream_id, target=target, result=result)
        return result

    def _context(self, target: Target) -> list[dict]:
        return self.session.build_context(
            target.key, self.budget_tokens, self.system_prompt)

    # -- patterns ----------------------------------------------------------
    def broadcast(self, targets: Sequence[Target], prompt: str) -> list[ChatResult]:
        """Send one prompt to every target at once, streaming all replies."""
        for target in targets:
            self.session.add(target.key, "user", prompt)

        results: list[ChatResult] = []

        def run(target: Target) -> ChatResult | None:
            result = self._turn(target, self._context(target))
            if result and result.text:
                self.session.add(target.key, "assistant", result.text)
            return result

        if len(targets) == 1:
            outcome = run(targets[0])
            return [outcome] if outcome else []

        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(targets))) as pool:
            for outcome in pool.map(run, targets):
                if outcome:
                    results.append(outcome)
        return results

    def relay(self, source: Target, target: Target, question: str) -> ChatResult | None:
        """Have `source` put a question to `target` and file the answer back."""
        self.session.add(source.key, "user", f"Ask {target.model}: {question}")
        prompt = (
            f"You are taking part in a discussion between AI models.\n"
            f"The model '{source.model}' is asking for your opinion.\n\n"
            f"Question: {question}\n\n"
            f"Answer clearly and directly.")
        self.session.add(target.key, "user", prompt)

        self.emit("note", text=f"Relaying from {source.short} to {target.short}…")
        result = self._turn(target, self._context(target),
                            heading=f"{target.short} (asked by {source.short})")
        if result and result.text:
            self.session.add(target.key, "assistant", result.text)
            self.session.add(
                source.key, "system",
                f"Received opinion from {target.model}: {result.text}")
        return result

    def debate(self, participants: Sequence[Target], topic: str, rounds: int = 2,
               judge: Target | None = None) -> ChatResult | None:
        """N models discuss for R rounds; an optional judge sums up.

        Each speaker sees the running transcript, so later turns can actually
        respond to earlier ones instead of restating the prompt.
        """
        if len(participants) < 2:
            self.emit("note", text="Debate needs at least two models selected.")
            return None

        transcript: list[str] = []
        self.emit("note", text=(f"Debate: {', '.join(p.short for p in participants)} "
                                f"over {rounds} round(s)"))

        for round_number in range(1, rounds + 1):
            if self.cancelled:
                break
            self.emit("note", text=f"── Round {round_number} ──")
            for speaker in participants:
                if self.cancelled:
                    break
                messages = [
                    {"role": "system", "content": self._debate_brief(speaker, topic,
                                                                     round_number)},
                    {"role": "user", "content": self._debate_prompt(topic, transcript,
                                                                    round_number)},
                ]
                result = self._turn(
                    messages=messages, target=speaker,
                    heading=f"{speaker.short} · round {round_number}")
                if result and result.text:
                    transcript.append(f"{speaker.model} said:\n{result.text.strip()}")
                    self.session.add(speaker.key, "user",
                                     f"[Debate round {round_number}] {topic}")
                    self.session.add(speaker.key, "assistant", result.text)

        if self.cancelled or not transcript:
            return None

        judge = judge or participants[0]
        self.emit("note", text=f"{judge.short} is summarising the debate…")
        verdict = self._turn(
            target=judge,
            messages=[
                {"role": "system",
                 "content": ("You are moderating a debate between AI models. "
                             "Be even-handed and concrete.")},
                {"role": "user",
                 "content": (f"Topic: {topic}\n\n"
                             f"Here is the full debate:\n\n" + "\n\n".join(transcript) +
                             "\n\nSummarise where the models agreed, where they "
                             "disagreed, and give your own conclusion. "
                             "Keep it under 250 words.")},
            ],
            heading=f"{judge.short} · consensus")
        if verdict and verdict.text:
            self.session.add(judge.key, "assistant",
                             f"[Debate consensus] {verdict.text}")
        return verdict

    @staticmethod
    def _debate_brief(speaker: Target, topic: str, round_number: int) -> str:
        stance = ("Open with your position." if round_number == 1
                  else "Respond to what the others have said. "
                       "Concede good points and push back on weak ones.")
        return (f"You are '{speaker.model}', one of several AI models debating a "
                f"topic. {stance} Be concise — under 150 words. "
                f"Do not repeat the topic back.")

    @staticmethod
    def _debate_prompt(topic: str, transcript: list[str], round_number: int) -> str:
        if not transcript:
            return f"Topic: {topic}\n\nGive your opening position."
        return (f"Topic: {topic}\n\nThe debate so far:\n\n"
                + "\n\n".join(transcript)
                + f"\n\nIt is your turn in round {round_number}.")

    def critique_chain(self, targets: Sequence[Target], prompt: str) -> ChatResult | None:
        """Draft, critique, revise — with the reviser seeing both prior turns."""
        if len(targets) < 2:
            self.emit("note", text="A critique chain needs at least two models.")
            return None

        drafter, critic = targets[0], targets[1]
        reviser = targets[2] if len(targets) > 2 else drafter

        self.emit("note", text=f"{drafter.short} is drafting…")
        draft = self._turn(
            target=drafter,
            messages=[{"role": "user", "content": prompt}],
            heading=f"{drafter.short} · draft")
        if not draft or not draft.text or self.cancelled:
            return None
        self.session.add(drafter.key, "user", prompt)
        self.session.add(drafter.key, "assistant", draft.text)

        self.emit("note", text=f"{critic.short} is critiquing…")
        critique = self._turn(
            target=critic,
            messages=[
                {"role": "system",
                 "content": ("You are a demanding but fair reviewer. Point out "
                             "concrete problems; do not rewrite the work.")},
                {"role": "user",
                 "content": (f"Original request:\n{prompt}\n\n"
                             f"Draft answer:\n{draft.text}\n\n"
                             "List the most important weaknesses, inaccuracies or "
                             "omissions as short bullet points.")},
            ],
            heading=f"{critic.short} · critique")
        if not critique or not critique.text or self.cancelled:
            return None
        self.session.add(critic.key, "assistant", f"[Critique] {critique.text}")

        self.emit("note", text=f"{reviser.short} is revising…")
        final = self._turn(
            target=reviser,
            messages=[
                {"role": "user",
                 "content": (f"Original request:\n{prompt}\n\n"
                             f"First draft:\n{draft.text}\n\n"
                             f"Reviewer feedback:\n{critique.text}\n\n"
                             "Write the improved final version. Address the "
                             "feedback; do not mention the review process.")},
            ],
            heading=f"{reviser.short} · final")
        if final and final.text:
            self.session.add(reviser.key, "assistant", f"[Revised] {final.text}")
        return final

    def judge_panel(self, participants: Sequence[Target], prompt: str,
                    judge: Target | None = None) -> ChatResult | None:
        """Everyone answers, then a judge scores the answers blind."""
        if len(participants) < 2:
            self.emit("note", text="A judge panel needs at least two models.")
            return None

        answers = self.broadcast(participants, prompt)
        answers = [a for a in answers if a and a.text]
        if self.cancelled or len(answers) < 2:
            return None

        judge = judge or participants[0]
        # Anonymised so the judge scores the writing, not the brand name.
        blind = "\n\n".join(
            f"Answer {index}:\n{answer.text.strip()}"
            for index, answer in enumerate(answers, 1))
        key = ", ".join(f"Answer {index} = {answer.model}"
                        for index, answer in enumerate(answers, 1))

        self.emit("note", text=f"{judge.short} is scoring {len(answers)} answers…")
        verdict = self._turn(
            target=judge,
            messages=[
                {"role": "system",
                 "content": ("You are an impartial judge. The answers are "
                             "anonymous. Score fairly on accuracy, completeness "
                             "and clarity.")},
                {"role": "user",
                 "content": (f"Question:\n{prompt}\n\n{blind}\n\n"
                             "Score each answer out of 10 with one sentence of "
                             "reasoning, then name the best answer.")},
            ],
            heading=f"{judge.short} · verdict")
        if verdict and verdict.text:
            self.emit("note", text=f"Key: {key}")
            self.session.add(judge.key, "assistant",
                             f"[Judge verdict] {verdict.text}\n\n{key}")
        return verdict
