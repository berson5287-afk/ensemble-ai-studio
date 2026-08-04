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
from .compaction import should_compact
from .deepresearch import (
    MAX_PER_DOMAIN,
    MAX_QUERIES,
    MAX_SOURCES,
    Source,
    Verification,
    build_plan_prompt,
    build_revision_prompt,
    build_synthesis_prompt,
    build_verification_prompt,
    check_citations,
    dedupe_sources,
    format_report,
    parse_queries,
    parse_verification,
)
from .edits import allowance_for_edits
from .formatting import friendly_model_name
from .planning import (
    MIN_STEPS,
    build_step_prompt,
    build_summary_prompt,
    drop_impossible_steps,
    parse_plan,
)
from .planning import build_plan_prompt as build_work_plan_prompt
from .session import Session, make_key, short_model
from .thinking import allowance, think_value

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
    "plan": "Plan & work — the model writes a checklist, then does each step",
    "research": "Research — plan, search, verify, then write it up",
    "converse": "Conversation — models talk, you can join in",
    "debate": "Debate — models discuss, a judge sums up",
    "critique": "Critique chain — draft, critique, revise",
    "judge": "Judge panel — models answer, a judge scores",
}

# How a model signals it thinks the conversation is finished.
END_MARKER = re.compile(r"\[\s*END\s*\]", re.IGNORECASE)


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
                 system_prompt: str = "", think: bool = True,
                 compactor: Callable[[list], bool] | None = None,
                 editing: bool = False) -> None:
        self.clients = clients
        self.session = session
        self.emit = emit
        self.cancel = cancel or threading.Event()
        self.options = options or {}
        self.budget_tokens = budget_tokens
        self.system_prompt = system_prompt
        self.think = think
        self.editing = editing
        # Called before a turn when the conversation has outgrown the budget.
        # Without it, compaction is decided once when Send is pressed — and a
        # pattern like plan-and-work runs a dozen turns inside that one send,
        # so the conversation sails past the ceiling and `build_context`
        # quietly drops the oldest turns for the rest of the run.
        self.compactor = compactor
        self._compact_tried: set[str] = set()
        self._stream_ids = itertools.count(1)

    # -- helpers -----------------------------------------------------------
    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

    def _thinking(self, model: str, enabled: bool | None = None):
        """The `think` value for a model, or None when it does not reason."""
        want = self.think if enabled is None else enabled
        return think_value(model, want)

    def _client(self, target: Target) -> OllamaClient:
        client = self.clients.get(target.server)
        if client is None or not client.base_url:
            raise OllamaError(f"The {target.server} server is not configured.")
        return client

    def _maybe_compact(self, target: Target) -> None:
        """Fold the conversation before a turn, if it has outgrown the budget.

        This is the "pause, tidy up, carry on" that a long run needs.  Trying
        once per target per run is deliberate: a conversation that is one
        enormous pinned attachment cannot be folded at all, and retrying
        before every step would spend a summarising call each time to be told
        the same thing.
        """
        if self.compactor is None or target.key in self._compact_tried:
            return
        history = self.session.history(target.key)
        if not should_compact(history, self.budget_tokens):
            return
        self._compact_tried.add(target.key)
        self.emit("note",
                  text=(f"🗜 {target.short}'s conversation has outgrown the "
                        f"context budget — folding the older turns before "
                        f"carrying on, so nothing is silently dropped."),
                  note_id=f"autocompact{target.key}")
        try:
            self.compactor([target])
        except Exception as exc:                 # never break a run over this
            self.emit("note", text=f"⚠ Could not compact — {str(exc)[:120]}",
                      note_id=f"autocompact{target.key}")
        self.emit("note_close", note_id=f"autocompact{target.key}")

    def _turn(self, target: Target, messages: list[dict],
              heading: str | None = None) -> ChatResult | None:
        """Run one model turn, streaming tokens out through `emit`."""
        if self.cancelled:
            return None
        self._maybe_compact(target)
        if self.cancelled:
            return None

        stream_id = next(self._stream_ids)
        self.emit("turn_start", stream_id=stream_id, target=target,
                  heading=heading or target.label)
        # Reasoning tokens come out of the same allowance as the reply, so a
        # reasoning model on the Quick end of the slider would spend all 192
        # of them thinking and hand back an empty answer.
        options = dict(self.options)
        options["num_predict"] = allowance(
            int(options.get("num_predict") or 0), target.model, self.think)
        if self.editing:
            # A reply that has to quote code needs room to finish the block it
            # started.  Cut off part way through, it produces nothing the app
            # can apply, and the user sees a model that will not edit.
            options["num_predict"] = allowance_for_edits(
                int(options.get("num_predict") or 0))
        try:
            result = self._client(target).chat(
                target.model, messages, options=options,
                on_token=lambda piece: self.emit("token", stream_id=stream_id,
                                                 text=piece),
                on_thought=lambda piece: self.emit("thought",
                                                   stream_id=stream_id,
                                                   text=piece),
                think=self._thinking(target.model),
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

    def plan_and_work(self, target: Target, question: str,
                      material: str = "") -> str | None:
        """Ask for a plan, work each step, then pull it together.

        One enormous generation either rambles or stops halfway and you cannot
        tell which until it ends.  Several small ones finish, and the
        checklist in front of them says which is running.  The cost is more
        round trips, which is why this is a mode you pick.
        """
        def tick(key: str, state: str, detail: str = "") -> None:
            self.emit("check", key=key, state=state, detail=detail)

        self.session.add(target.key, "user", question)

        self.emit("note", text="🗂 Working out a plan…", note_id="plan")
        plan_reply = self._quiet(
            target,
            build_work_plan_prompt(question, material, editing=self.editing),
            num_predict=400)
        if plan_reply is None or self.cancel.is_set():
            return None
        steps = parse_plan(plan_reply)
        if self.editing:
            # "Open the file", "Save the modified file", "Test it" — steps a
            # model without hands can only act out.  Left in the plan they
            # get ticked off one by one, which reads as work being done.
            steps = drop_impossible_steps(steps)
        if len(steps) < MIN_STEPS:
            # No usable plan — answering directly beats inventing one.
            self.emit("note", text="🗂 No clear plan came back — answering "
                                   "this in one go instead.", note_id="plan")
            self.emit("note_close", note_id="plan")
            result = self._turn(target, self._context(target))
            if result and result.text:
                self.session.add(target.key, "assistant", result.text)
            return result.text if result else None

        self.emit("note_close", note_id="plan")
        self.emit("plan", steps=steps, target=target)

        done: list[str] = []
        for index in range(len(steps)):
            if self.cancel.is_set():
                tick(f"step{index}", "skipped", "stopped")
                continue
            tick(f"step{index}", "running")
            heading = f"{friendly_model_name(target.model) or target.short} · step {index + 1}"
            result = self._turn(
                target,
                self._framed(target,
                             build_step_prompt(question, steps, index, done,
                                               editing=self.editing)),
                heading=heading)
            if result is None or not result.text.strip():
                tick(f"step{index}", "failed", "no answer")
                continue
            done.append(result.text)
            tick(f"step{index}", "done")

        if not done or self.cancel.is_set():
            return None

        tick("summary", "running")
        final = self._turn(
            target,
            self._framed(target, build_summary_prompt(
                question, steps, done, editing=self.editing)),
            heading=f"{friendly_model_name(target.model) or target.short} · the answer")
        tick("summary", "done" if final and final.text else "failed")
        text = final.text if final and final.text else "\n\n".join(done)
        self.session.add(target.key, "assistant", text)
        return text

    def _framed(self, target: Target, prompt: str) -> list[dict]:
        """A one-off instruction, sent with the framing it needs to be doable.

        A plan step used to go out as a bare user message.  That is the whole
        point of the mode — a small, cheap, self-contained call — but it also
        dropped the system prompt, which is where the editing instructions
        live, and the attached files with them.  So a step told to "replace
        the evaluate_and_schedule function" had neither the format for
        proposing the change nor the text of the function.  It produced both
        from memory, fluently, and every one was refused downstream as lines
        that are not in the file.

        The system prompt therefore always travels.  The files travel too,
        but only when there are edits to make, because that is the one job
        where an isolated step cannot possibly be right and the extra tokens
        buy something.
        """
        context = self._context(target)
        if self.editing:
            return context + [{"role": "user", "content": prompt}]
        system = [m for m in context if m.get("role") == "system"]
        return system + [{"role": "user", "content": prompt}]

    def _quiet(self, target: Target, prompt: str,
               num_predict: int = 300) -> str | None:
        """A model call whose output is machinery, not something to show."""
        client = self.clients.get(target.server)
        if client is None or not client.base_url:
            return None
        options = dict(self.options)
        options["num_predict"] = num_predict
        try:
            result = client.chat(target.model,
                                 [{"role": "user", "content": prompt}],
                                 options=options,
                                 think=self._thinking(target.model, False),
                                 cancel=self.cancel)
        except Exception as exc:
            self.emit("note", text=f"⚠ {str(exc)[:140]}", note_id="plan")
            return None
        return result.text

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

    # -- deep research -----------------------------------------------------
    def _quiet_turn(self, target: Target, prompt: str,
                    max_tokens: int = 400) -> str:
        """A short model call whose output is machinery, not conversation."""
        if self.cancelled:
            return ""
        options = dict(self.options)
        options.update({"temperature": 0.2, "num_predict": max_tokens})
        try:
            result = self._client(target).chat(
                target.model, [{"role": "user", "content": prompt}],
                options=options,
                think=self._thinking(target.model, False),
                cancel=self.cancel)
        except Exception:
            return ""
        return result.text or ""

    def research(self, target: Target, question: str,
                 search_fn: Callable[[str], list[dict]],
                 fetch_fn: Callable[[str], str],
                 max_queries: int = MAX_QUERIES,
                 max_sources: int = MAX_SOURCES,
                 fetch_pages: int = 4,
                 follow_up: bool = True) -> str | None:
        """Plan, gather, draft, verify, then revise — a full research pass.

        The point of the extra passes is that single-shot retrieval has two
        failure modes no amount of prompting fixes: one query only ever finds
        one facet of a question, and nothing checks that the citations in the
        answer point at sources that exist.
        """
        def note(text: str) -> None:
            """Transient progress — this line rewrites itself each stage."""
            self.emit("note", text=text, note_id="research")

        def record(text: str) -> None:
            """A finding worth keeping in the transcript afterwards."""
            self.emit("note", text=text)

        # 1. Plan — several distinct searches rather than one.
        note("🔬 Planning the research…")
        queries = parse_queries(
            self._quiet_turn(target, build_plan_prompt(question, max_queries), 200),
            fallback=question, limit=max_queries)
        if self.cancelled:
            return None
        record("🔬 Searching " + " · ".join(f"“{q}”" for q in queries))

        # 2. Gather — pool every result, then dedupe across domains.
        pooled: list[dict] = []
        for query in queries:
            if self.cancelled:
                return None
            try:
                pooled.extend(search_fn(query))
            except Exception as exc:
                note(f"⚠ Search failed for “{query}” — {exc}")
        sources = dedupe_sources(pooled, limit=max_sources)
        if not sources:
            note("⚠ Research found nothing to read.")
            self.emit("note_close", note_id="research")
            return None

        self._read_pages(sources, fetch_fn, fetch_pages, note)
        if self.cancelled:
            return None

        # 3. Draft — a cited answer from what was actually found.
        note(f"🔬 Reading complete — {len(sources)} sources. Writing the answer…")
        draft = self._turn(
            target=target,
            messages=[{"role": "user",
                       "content": build_synthesis_prompt(question, sources)}],
            heading=f"{target.short} · draft")
        if not draft or not draft.text.strip() or self.cancelled:
            self.emit("note_close", note_id="research")
            return None
        answer = draft.text.strip()

        # 4. Verify — a deterministic citation check plus a sceptical read.
        note("🔬 Fact-checking the answer against the sources…")
        citations = check_citations(answer, sources)
        verification = parse_verification(self._quiet_turn(
            target, build_verification_prompt(question, answer, sources), 400))
        if self.cancelled:
            return None

        findings = len(verification.unsupported) + len(verification.gaps)
        record(f"🔎 Fact-check — {citations.summary}"
               + (f"; {findings} issue(s) to fix" if findings
                  else "; nothing unsupported"))

        # 5. Follow up — search the gaps the check surfaced, then revise.
        if follow_up and verification.follow_ups and citations is not None:
            record("🔬 Filling gaps — searching " +
                   " · ".join(f"“{q}”" for q in verification.follow_ups))
            extra: list[dict] = []
            for query in verification.follow_ups:
                if self.cancelled:
                    return None
                queries.append(query)
                try:
                    extra.extend(search_fn(query))
                except Exception:
                    continue
            combined = dedupe_sources(
                [{"url": s.url, "title": s.title, "snippet": s.snippet}
                 for s in sources] + extra,
                limit=max_sources + len(verification.follow_ups) * 2,
                per_domain=MAX_PER_DOMAIN + 1)
            by_url = {s.url: s for s in sources}
            fresh = [s for s in combined if s.url not in by_url]
            for source in combined:
                if source.url in by_url:
                    source.text = by_url[source.url].text
            self._read_pages(fresh, fetch_fn, 2, note)
            sources = combined

        needs_revision = (verification.unsupported or verification.gaps
                          or not citations.clean or verification.follow_ups)
        if needs_revision and not self.cancelled:
            note("🔬 Revising with the corrections and new sources…")
            revised = self._turn(
                target=target,
                messages=[{"role": "user",
                           "content": build_revision_prompt(
                               question, answer, sources, verification,
                               citations)}],
                heading=f"{target.short} · final")
            if revised and revised.text.strip():
                answer = revised.text.strip()
                citations = check_citations(answer, sources)
                if verification.follow_ups:
                    # we went and looked; the rewrite states what it still
                    # couldn't confirm, so stale gap notes would contradict it
                    verification = Verification()

        # 6. Report — the answer, its sources, and what stayed unconfirmed.
        report = format_report(answer, sources, verification, citations,
                               queries=queries)
        self.session.add(target.key, "user", f"[Research] {question}")
        self.session.add(target.key, "assistant", report)
        note(f"🔬 Done — {len(sources)} sources, "
             f"{len(citations.cited)} cited.")
        self.emit("note_close", note_id="research")
        self.emit("report", text=report, target=target)
        return report

    def _read_pages(self, sources: Sequence[Source],
                    fetch_fn: Callable[[str], str], limit: int,
                    note: Callable[[str], None]) -> None:
        """Download the top few pages, leaving the rest as snippets."""
        read = 0
        for source in sources:
            if read >= limit or self.cancelled:
                return
            note(f"🔬 Reading {source.domain}…")
            try:
                text = fetch_fn(source.url)
            except Exception:
                text = ""
            if text:
                source.text = text
                read += 1

    def converse(self, participants: Sequence[Target], topic: str,
                 max_turns: int = 12,
                 pending: Callable[[], list[str]] | None = None) -> list[ChatResult]:
        """An open-ended conversation the models run themselves.

        Unlike `debate`, there is no fixed number of rounds and no judge: the
        models take turns until they decide they're finished, the turn cap is
        reached, or you stop them.  Anything the user types while it's running
        is handed to the next speaker, so you can steer without interrupting.
        """
        if len(participants) < 2:
            self.emit("note", text="Conversation mode needs at least two models "
                                   "so they have someone to talk to.")
            return []

        pending = pending or (lambda: [])
        transcript: list[str] = []
        results: list[ChatResult] = []

        self.emit("note", text=(
            f"💬 {' and '.join(p.short for p in participants)} are talking. "
            f"Type any time to join in — Stop ends it."))

        turn = 0
        ended_by = None
        while turn < max_turns and not self.cancelled:
            for message in pending():
                transcript.append(f"[The person listening said] {message}")
                self.emit("note", text="💬 You joined in")

            speaker = participants[turn % len(participants)]
            result = self._turn(
                target=speaker,
                messages=[
                    {"role": "system",
                     "content": self._conversation_brief(speaker, participants,
                                                         topic)},
                    {"role": "user",
                     "content": self._conversation_prompt(topic, transcript)},
                ],
                heading=f"{speaker.short} · turn {turn + 1}")
            turn += 1

            if not result or not result.text.strip():
                continue
            results.append(result)

            spoken = result.text.strip()
            # A model can only call time once everyone has had a say, so an
            # eager [END] on the opening turn doesn't cut things short.
            wrapping_up = bool(END_MARKER.search(spoken)) and turn >= len(participants)
            spoken = END_MARKER.sub("", spoken).strip()

            transcript.append(f"{speaker.model} said:\n{spoken}")
            self.session.add(speaker.key, "user", f"[Conversation] {topic}")
            self.session.add(speaker.key, "assistant", spoken)

            if wrapping_up:
                ended_by = speaker
                break

        if self.cancelled:
            self.emit("note", text=f"💬 Stopped after {turn} turn(s).")
        elif ended_by is not None:
            self.emit("note", text=(f"💬 {ended_by.short} wrapped it up after "
                                    f"{turn} turns."))
        elif turn >= max_turns:
            self.emit("note", text=(f"💬 Reached the {max_turns}-turn limit. "
                                    f"Send a message to keep it going."))
        return results

    @staticmethod
    def _conversation_brief(speaker: Target, participants: Sequence[Target],
                            topic: str) -> str:
        others = " and ".join(p.model for p in participants if p != speaker)
        return (
            f"You are '{speaker.model}', in a live conversation with {others}. "
            f"A person is listening and may join in at any point.\n\n"
            f"The subject is: {topic}\n\n"
            "- Talk the way people talk in a group, not the way an essay reads.\n"
            "- Keep each turn to two or three sentences.\n"
            "- Respond to what was just said. Disagree if you disagree, and ask "
            "the others questions.\n"
            "- If the person listening says something, answer them directly.\n"
            "- Never repeat a point that has already been made.\n"
            "- Don't narrate the conversation or summarise it.\n"
            "- When it has genuinely run its course, end your message with "
            "[END] on its own line.")

    @staticmethod
    def _conversation_prompt(topic: str, transcript: list[str]) -> str:
        if not transcript:
            return f"Open the conversation about: {topic}"
        return ("The conversation so far:\n\n" + "\n\n".join(transcript)
                + "\n\nYour turn — keep it short.")

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
