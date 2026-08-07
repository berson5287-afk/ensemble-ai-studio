"""App-level wiring: long-lived objects, prompt assembly and exit behaviour.

Needs a display, so it skips itself on headless machines (CI uses xvfb).
"""

from __future__ import annotations

import json
import time

import pytest

from aichatlab import editdebug, edits
from aichatlab import intent as intent_module

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A real ChatLabApp with no servers configured, so nothing hits the network."""
    from aichatlab import cache as cache_module
    from aichatlab import config as config_module
    from aichatlab import knowledge as knowledge_module
    from aichatlab import recovery as recovery_module

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"local_ip": "", "host_ip": ""}),
                             encoding="utf-8")
    monkeypatch.setattr(config_module, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(cache_module, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(knowledge_module, "KNOWLEDGE_PATH", tmp_path / "know.json")
    # Without this the app restores whatever unfinished chat happens to be in
    # the real temp directory, and the tests quietly depend on the machine.
    monkeypatch.setattr(recovery_module, "RECOVERY_PATH",
                        tmp_path / "unfinished.json")
    # The card probe shells out to nvidia-smi on a background thread.  Left
    # running, it outlives the test's root window and Tk variables then get
    # finalised with a non-main thread alive — "main thread is not in main
    # loop".  Tests that care about cards call `_cards_found` directly.
    from aichatlab import gpu as gpu_module
    monkeypatch.setattr(gpu_module, "probe", lambda *a, **k: [])

    # The edit-check trace appends to a file in the user's home directory.
    # A test run must not add to the record of what the real app did.
    from aichatlab import editdebug as editdebug_module
    monkeypatch.setattr(editdebug_module, "TRACE_PATH",
                        tmp_path / "edit_trace.jsonl")

    from aichatlab.ui.app import ChatLabApp

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    instance = ChatLabApp(root)
    yield instance
    # Stop the event pump before the window goes, or its pending `after`
    # callback fires into a destroyed interpreter and prints Tcl errors.
    if instance.pump_id is not None:
        try:
            root.after_cancel(instance.pump_id)
        except tk.TclError:
            pass
    try:
        root.destroy()
    except tk.TclError:
        pass



def test_cache_and_knowledge_survive_typing(app):
    """Regression: both were rebuilt on every keystroke.

    A stray edit put the constructors inside `_hide_placeholder`, which runs
    each time the user starts typing.  The cache lost track of what this
    session had added, so nothing was ever cleaned up on exit.
    """
    cache_before, knowledge_before = app.cache, app.knowledge

    app._hide_placeholder()
    app._show_placeholder()
    app._hide_placeholder()

    assert app.cache is cache_before
    assert app.knowledge is knowledge_before


def test_session_added_entries_are_tracked_for_cleanup(app):
    app.cache.put("how to sweat a copper pipe", "BLOCK")
    assert app.cache.session_keys       # remembered, so exit can clean it up


def test_closing_an_unsaved_chat_drops_its_cached_research(app):
    app.cache.put("how to sweat a copper pipe", "BLOCK")
    app.session.path = None

    app.on_close()

    from aichatlab.cache import ResearchCache
    assert len(ResearchCache(app.cache.path)) == 0


def test_closing_a_saved_chat_keeps_its_cached_research(app, tmp_path):
    app.cache.put("how to sweat a copper pipe", "BLOCK")
    app.session.path = tmp_path / "chat.json"

    app.on_close()

    from aichatlab.cache import ResearchCache
    assert ResearchCache(app.cache.path).get("how to sweat a copper pipe")


def test_learned_lessons_reach_the_system_prompt(app):
    app.knowledge.add("Plumbing", "Shut the main valve before opening a P-trap "
                                  "or the standing water empties onto the floor.")
    app.learning_var.set(True)

    prompt = app._system_prompt("my P-trap is leaking, plumbing advice")

    assert "main valve" in prompt
    assert "earlier conversations" in prompt


def test_lessons_stay_out_of_the_prompt_when_learning_is_off(app):
    app.knowledge.add("Plumbing", "Shut the main valve before opening a P-trap "
                                  "or the standing water empties onto the floor.")
    app.learning_var.set(False)

    assert "main valve" not in app._system_prompt("P-trap plumbing leak")


def test_persona_is_included_by_default(app):
    assert "restate" in app._system_prompt("anything")


def test_knowledge_link_shows_the_count(app):
    app.knowledge.add("Wiring", "Kill the breaker and test the line before "
                                "touching any conductor in the panel.")
    app._refresh_knowledge_link()

    assert "1" in app.knowledge_link["text"]


# ------------------------------------------- asking before answering blind

def pending_action(app):
    return next(iter(app.chat._actions), None)


def action_buttons(app):
    """The labels of the buttons embedded in the transcript, in order.

    They are real Tk widgets inside the Text, so they are invisible to
    `transcript()` — which reads characters, not windows.
    """
    labels = []
    for name in app.chat.text.window_names():
        widget = app.chat.text.nametowidget(name)
        labels.extend(child["text"] for child in widget.winfo_children()
                      if isinstance(child, tk.Button))
    return labels


def test_a_lookup_request_with_search_off_pauses_the_send(app):
    app.research_var.set(False)

    paused = app._maybe_ask_to_search("look up the latest firmware", "chat", None)

    assert paused
    assert pending_action(app)
    assert "look up" in app.chat.transcript()


def test_an_ordinary_question_is_sent_straight_through(app):
    app.research_var.set(False)

    assert not app._maybe_ask_to_search("explain how a heat pump works",
                                        "chat", None)
    assert pending_action(app) is None


def test_no_prompt_when_web_research_is_already_on(app):
    app.research_var.set(True)

    assert not app._maybe_ask_to_search("look up the latest firmware",
                                        "chat", None)


def test_no_prompt_in_research_mode_which_always_searches(app):
    app.research_var.set(False)

    assert not app._maybe_ask_to_search("look up the latest firmware",
                                        "research", None)


def test_no_prompt_for_a_relay_command(app):
    app.research_var.set(False)

    assert not app._maybe_ask_to_search("alpha ask beta to look up the specs",
                                        "chat", ("a", "b", "c"))


def test_answering_without_searching_leaves_the_toggle_alone(app, monkeypatch):
    app.research_var.set(False)
    app._maybe_ask_to_search("look up the latest firmware", "chat", None)
    action_id = pending_action(app)
    monkeypatch.setattr(app, "send", lambda: None)

    app._send_without_search(action_id)

    assert not app.research_var.get()
    assert app.skip_search_prompt          # the resend must not ask again
    assert "no search was made" in app.chat.transcript()


def test_enabling_search_flips_the_toggle_and_resends(app, monkeypatch):
    app.settings["searxng_url"] = "http://searx.local"
    app.research_var.set(False)
    app._maybe_ask_to_search("look up the latest firmware", "chat", None)
    action_id = pending_action(app)
    sent = []
    monkeypatch.setattr(app, "send", lambda: sent.append(True))

    app._enable_and_send(action_id)

    assert app.research_var.get()
    assert sent == [True]
    assert pending_action(app) is None


def test_enabling_search_without_a_searxng_url_asks_for_one_instead(app, monkeypatch):
    app.settings["searxng_url"] = ""
    app.research_var.set(False)
    app._maybe_ask_to_search("look up the latest firmware", "chat", None)
    action_id = pending_action(app)
    opened, sent = [], []
    monkeypatch.setattr(app, "open_settings", lambda: opened.append(True))
    monkeypatch.setattr(app, "send", lambda: sent.append(True))
    monkeypatch.setattr("aichatlab.ui.app.messagebox.showinfo",
                        lambda *a, **k: None)

    app._enable_and_send(action_id)

    assert opened == [True]
    assert sent == []                      # nothing was sent blind
    assert not app.research_var.get()


def test_the_prompt_is_not_repeated_after_a_decision(app):
    app.skip_search_prompt = True

    assert not app._maybe_ask_to_search("look up the latest firmware",
                                        "chat", None)
    assert not app.skip_search_prompt      # …but only once


# -------------------------------------------------------- folder attachments

def test_a_folder_block_is_not_double_fenced(app):
    app.attachments = [{"name": "project/", "content": "[Folder: project]\nBODY"}]

    composed = app._compose("what does this do?")

    assert "[Attached file:" not in composed
    assert composed.count("```") == 0
    assert "BODY" in composed


def test_a_file_attachment_is_still_fenced(app):
    app.attachments = [{"name": "notes.txt", "content": "BODY"}]

    assert "[Attached file: notes.txt]" in app._compose("summarise")


def test_accepting_a_folder_raises_the_context_budget_and_says_so(app, tmp_path):
    from aichatlab.folderscan import Limits, survey

    (tmp_path / "a.txt").write_text("x" * 40_000, encoding="utf-8")
    app.settings["context_budget_tokens"] = 6000

    app._folder_accepted(tmp_path, survey(tmp_path), Limits(), 20_000)

    assert app.settings["context_budget_tokens"] == 20_000
    assert "Context budget raised" in app.chat.transcript()


def test_a_folder_that_fits_leaves_the_budget_alone(app, tmp_path):
    from aichatlab.folderscan import Limits, survey

    (tmp_path / "a.txt").write_text("small", encoding="utf-8")
    app.settings["context_budget_tokens"] = 6000

    app._folder_accepted(tmp_path, survey(tmp_path), Limits(), 6000)

    assert app.settings["context_budget_tokens"] == 6000
    assert "Context budget raised" not in app.chat.transcript()


def test_a_read_folder_becomes_one_attachment_chip(app, tmp_path):
    from aichatlab.folderscan import Limits, select, survey

    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    result = survey(tmp_path)

    app._folder_read(tmp_path, "BLOCK", select(result, Limits()))

    assert [a["name"] for a in app.attachments] == [f"{tmp_path.name}/"]
    assert app.attachments[0]["content"] == "BLOCK"


def test_re_adding_the_same_folder_replaces_it_rather_than_duplicating(app, tmp_path):
    from aichatlab.folderscan import Limits, select, survey

    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    selection = select(survey(tmp_path), Limits())

    app._folder_read(tmp_path, "FIRST", selection)
    app._folder_read(tmp_path, "SECOND", selection)

    assert len(app.attachments) == 1
    assert app.attachments[0]["content"] == "SECOND"


def test_dropping_a_folder_no_longer_does_nothing(app, tmp_path, monkeypatch):
    """Regression: add_attachment returned silently for any non-file path."""
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    seen = []
    monkeypatch.setattr(app, "add_folder", lambda path: seen.append(path))

    app.add_attachment(tmp_path)

    assert seen == [tmp_path]


def test_dropping_a_path_that_does_not_exist_says_so(app, tmp_path):
    app.add_attachment(tmp_path / "ghost.txt")

    assert "Could not find" in app.status.get()


def test_the_attach_note_reports_what_was_sent_not_what_was_estimated(app, tmp_path):
    """Announcing 94,000 tokens after trimming the block to 31,000 would be
    the same quiet lie the whole feature exists to avoid."""
    from aichatlab.folderscan import Limits, select, survey

    (tmp_path / "big.txt").write_text("x" * 200_000, encoding="utf-8")
    selection = select(survey(tmp_path), Limits(max_file_bytes=10_000_000))

    app._folder_read(tmp_path, "short block", selection)

    note = app.chat.transcript()
    assert "trimmed from" in note
    assert "≈50,000 tokens" not in note        # the estimate is not the claim


def test_a_folder_that_was_not_trimmed_says_nothing_about_trimming(app, tmp_path):
    from aichatlab.folderscan import Limits, build_block, select, survey

    (tmp_path / "a.txt").write_text("hello there", encoding="utf-8")
    result = survey(tmp_path)
    selection = select(result, Limits())
    block = build_block(tmp_path, selection.chosen, len(result.entries))

    app._folder_read(tmp_path, block, selection)

    assert "trimmed from" not in app.chat.transcript()


# ----------------------------------------------------------- context warnings

def context_text(app):
    app._update_context_label()
    return app.context_label["text"]


def _select(app, server="local", model="alpha:1b"):
    """Register a ticked model — the test app has no servers to discover any."""
    from aichatlab.session import make_key
    app.model_vars[(server, model)] = tk.BooleanVar(value=True)
    return make_key(server, model)


def _fill(app, tokens, server="local", model="alpha:1b", turns=10):
    """Give a selected model a conversation of roughly `tokens` tokens.

    Spread over several turns rather than one enormous message, because a
    single message is not foldable — compaction needs turns older than the
    recent few — and a helper that produces an uncompactable conversation
    makes every test about compaction quietly meaningless.
    """
    key = _select(app, server, model)
    each = max(1, (tokens * 4) // turns)
    app.session.conversations[key] = [
        {"role": "user" if index % 2 == 0 else "assistant",
         "content": "x" * each}
        for index in range(turns)]
    return key


def test_the_context_label_stays_quiet_while_there_is_room(app):
    app.settings["context_budget_tokens"] = 28_000
    _fill(app, 2_000)

    text = context_text(app)
    assert "context ≈" in text
    assert "—" not in text
    assert app.context_label["fg"] != "#d64545"


def test_the_context_label_warns_as_it_fills(app):
    app.settings["context_budget_tokens"] = 28_000
    _fill(app, int(app.settings.effective_budget() * 0.8))

    assert "Compact" in context_text(app)


def test_the_context_label_says_what_is_being_lost_once_it_overflows(app):
    app.settings["context_budget_tokens"] = 28_000
    _fill(app, app.settings.effective_budget() + 5_000)

    assert "oldest turns are being dropped" in context_text(app)


def test_no_context_label_without_a_model_selected(app):
    _select(app)
    for var in app.model_vars.values():
        var.set(False)

    assert context_text(app) == ""


# ------------------------------------------------------------ cut-off replies

def test_a_truncated_reply_offers_to_continue(app):
    from aichatlab.client import ChatResult
    from aichatlab.orchestrator import Target

    app.settings["speed_quality"] = 10             # Fastest — a 192-token cap
    result = ChatResult(model="gpt-oss:20b", server="host", done_reason="length")

    app._offer_continue(Target("host", "gpt-oss:20b"), result)

    transcript = app.chat.transcript()
    assert "192-token cap" in transcript
    assert "had not finished" in transcript
    assert app.chat._actions


def test_an_uncapped_profile_offers_continue_but_not_raise_the_limit(app):
    from aichatlab.client import ChatResult
    from aichatlab.orchestrator import Target

    app.settings["speed_quality"] = 100            # Max quality — uncapped
    app._offer_continue(Target("host", "m"), ChatResult(model="m"))

    assert "the length cap" in app.chat.transcript()


def test_continuing_asks_the_model_to_pick_up_where_it_stopped(app, monkeypatch):
    from aichatlab.orchestrator import Target

    target = Target("host", "gpt-oss:20b")
    app._offer_continue(target, type("R", (), {})())
    action_id = next(iter(app.chat._actions))
    jobs = []
    monkeypatch.setattr(app, "_run", lambda job: jobs.append(job))

    app._continue_reply(target, action_id)

    assert jobs
    assert "Continuing where it left off" in app.chat.transcript()
    assert not app.chat._actions


def test_raising_the_limit_moves_the_slider_and_reasks(app, monkeypatch):
    from aichatlab.orchestrator import Target
    from aichatlab.session import make_key

    target = Target("host", "gpt-oss:20b")
    app.session.conversations[make_key("host", "gpt-oss:20b")] = [
        {"role": "user", "content": "give me tips"},
        {"role": "assistant", "content": "1. Centralise"}]
    app.settings["speed_quality"] = 10
    app._offer_continue(target, type("R", (), {})())
    action_id = next(iter(app.chat._actions))
    jobs = []
    monkeypatch.setattr(app, "_run", lambda job: jobs.append(job))

    app._raise_and_retry(target, action_id)

    assert app.settings["speed_quality"] > 10
    assert app.settings.profile()[1] > 192          # a bigger reply allowance
    assert jobs


def test_raising_the_limit_with_no_question_to_repeat_does_nothing(app, monkeypatch):
    from aichatlab.orchestrator import Target

    target = Target("host", "unused:1b")
    app._offer_continue(target, type("R", (), {})())
    action_id = next(iter(app.chat._actions))
    jobs = []
    monkeypatch.setattr(app, "_run", lambda job: jobs.append(job))

    app._raise_and_retry(target, action_id)

    assert jobs == []
    assert "Nothing to ask again" in app.chat.transcript()


# -------------------------------------------------------------- compaction

def test_compacting_a_short_chat_is_declined_politely(app, monkeypatch):
    said = []
    monkeypatch.setattr("aichatlab.ui.app.messagebox.showinfo",
                        lambda title, message: said.append(title))
    _select(app)

    app.compact_chat()

    assert said == ["Nothing to compact"]


def test_compaction_replaces_old_turns_and_reports_the_saving(app):
    from aichatlab.compaction import is_summary
    from aichatlab.orchestrator import Target
    from aichatlab.session import make_key
    from tests.conftest import FakeClient

    key = make_key("local", "alpha:1b")
    app.session.conversations[key] = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"m{i} " + "x" * 2000} for i in range(20)]
    before = len(app.session.conversations[key])
    summariser = FakeClient(server="local", reply="They chose Postgres.")
    app.clients = lambda: {"local": summariser, "host": summariser}

    assert app._compact([Target("local", "alpha:1b")])

    history = app.session.conversations[key]
    assert len(history) < before
    assert is_summary(history[0])
    assert "Postgres" in history[0]["content"]
    while not app.events.empty():
        kind, payload = app.events.get_nowait()
        if kind == "note" and "folded" in payload.get("text", ""):
            assert "→" in payload["text"]
            break
    else:
        raise AssertionError("no saving was reported")


def test_a_failed_compaction_leaves_the_history_untouched(app):
    from aichatlab.orchestrator import Target
    from aichatlab.session import make_key
    from tests.conftest import FakeClient

    key = make_key("local", "alpha:1b")
    original = [{"role": "user", "content": f"m{i} " + "x" * 2000}
                for i in range(20)]
    app.session.conversations[key] = list(original)
    app.clients = lambda: {"local": FakeClient(server="local", fail_on="alpha"),
                           "host": FakeClient()}

    assert not app._compact([Target("local", "alpha:1b")])
    assert app.session.conversations[key] == original


# ------------------------------------------------------- assess before search

def test_a_question_about_attached_files_never_reaches_the_search_engine(app):
    """Not a judgement call, so it does not cost a model round trip either.
    Asking a 32B model to decide took thirty seconds and it still went and
    read two SEO pages about debugging."""
    from aichatlab.orchestrator import Target
    from tests.conftest import FakeClient

    decider = FakeClient(server="local", reply="SEARCH: debugging tips")
    app.clients = lambda: {"local": decider, "host": FakeClient()}
    attached = [{"name": "udbg-phase1/", "content": "x" * 4000}]

    block = app._research("take a look at my app and give me tips",
                          Target("local", "alpha:1b"),
                          [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "hello"}],
                          attached)

    assert block is None                       # no search was made
    assert decider.calls == []                 # and no model was asked either
    assert "no web search needed" in _notes(app)


def test_triage_searches_for_its_own_query_not_the_raw_message(app, monkeypatch):
    from aichatlab.orchestrator import Target
    from tests.conftest import FakeClient

    app.settings["searxng_url"] = "http://searx.local"
    app.clients = lambda: {
        "local": FakeClient(server="local",
                            reply="SEARCH: Ollama num_ctx default"),
        "host": FakeClient()}
    searched = []

    def fake_gather(client, query, **kwargs):
        searched.append(query)
        return "BLOCK"

    monkeypatch.setattr("aichatlab.ui.app.gather", fake_gather)

    # explicit lookup wording, so the deterministic guard steps aside and the
    # model gets to choose the query
    app._research("look up the default for that setting",
                  Target("local", "alpha:1b"),
                  [{"role": "user", "content": "tell me about num_ctx"},
                   {"role": "assistant", "content": "it sets the window"}],
                  [{"name": "notes.txt", "content": "x"}])

    assert searched == ["Ollama num_ctx default"]


def test_triage_is_skipped_when_switched_off(app, monkeypatch):
    from aichatlab.orchestrator import Target
    from tests.conftest import FakeClient

    app.settings["assess_before_search"] = False
    app.settings["searxng_url"] = "http://searx.local"
    app.settings["smart_followups"] = False
    app.clients = lambda: {"local": FakeClient(server="local", reply="ANSWER"),
                           "host": FakeClient()}
    searched = []
    monkeypatch.setattr("aichatlab.ui.app.gather",
                        lambda client, query, **kw: searched.append(query) or "B")

    app._research("take a look at my app", Target("local", "alpha:1b"),
                  [{"role": "user", "content": "hi"}],
                  [{"name": "app/", "content": "x"}])

    assert searched == ["take a look at my app"]


def _notes(app):
    texts = []
    while not app.events.empty():
        kind, payload = app.events.get_nowait()
        if kind == "note":
            texts.append(payload.get("text", ""))
    return "\n".join(texts)


# ------------------------------------------------- reopening a saved chat

FOLDER_MSG = ("[Folder: udbg-phase1 — 10 of 24 files included]\n"
              "Files included:\n  udbg/cli.py (17 KB)\n\n"
              "--- udbg/cli.py ---\n```\n" + "source line\n" * 3000 +
              "```\n\n[End of folder contents.]\n\ngive me some tips")


def saved_chat(tmp_path, selected="local::alpha:1b"):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps({
        "schema": 2,
        "conversations": {selected: [
            {"role": "user", "content": FOLDER_MSG},
            {"role": "assistant", "content": "Here are a few ideas."}]},
        "selected_models": [selected],
    }), encoding="utf-8")
    return path


def load(app, path):
    from aichatlab.session import Session
    session, selected = Session.load(path)
    app.session = session
    app.pending_selection = set(selected)
    app._apply_selection()
    app._redraw()


def test_a_reloaded_folder_shows_as_a_chip_not_a_wall_of_source(app, tmp_path):
    """Redrawing used to paste the whole folder into the transcript as if the
    user had typed a hundred thousand characters of source code."""
    load(app, saved_chat(tmp_path))

    transcript = app.chat.transcript()
    assert "📎 udbg-phase1/" in transcript
    assert "give me some tips" in transcript
    assert "source line" not in transcript
    assert len(transcript) < 2_000


def test_a_reloaded_folder_still_reaches_the_model(app, tmp_path):
    load(app, saved_chat(tmp_path))
    app.session.add("local::alpha:1b", "user", "and what about the tests?")

    context = app.session.build_context("local::alpha:1b",
                                        app.settings.effective_budget())

    assert any("[Folder:" in m["content"] for m in context)


def test_the_saved_selection_is_applied_when_the_models_arrive_late(app, tmp_path):
    """Opening a chat before the servers answer used to lose the selection:
    the checkboxes were rebuilt from a model list that did not exist yet."""
    app.model_vars.clear()                       # servers have not replied
    load(app, saved_chat(tmp_path))
    assert not app.model_vars

    app._models_loaded("local", ["alpha:1b", "beta:2b"])

    assert [k for k, v in app.model_vars.items() if v.get()] == [
        ("local", "alpha:1b")]


def test_the_selection_is_applied_immediately_when_models_are_already_there(app, tmp_path):
    app.model_vars[("local", "alpha:1b")] = tk.BooleanVar(value=False)
    app.model_vars[("local", "beta:2b")] = tk.BooleanVar(value=True)

    load(app, saved_chat(tmp_path))

    assert app.model_vars[("local", "alpha:1b")].get()
    assert not app.model_vars[("local", "beta:2b")].get()


def test_starting_a_new_chat_forgets_the_loaded_selection(app, tmp_path):
    load(app, saved_chat(tmp_path))

    app.new_chat()
    app._models_loaded("local", ["alpha:1b"])

    assert not any(v.get() for v in app.model_vars.values())


def test_a_folder_block_is_built_smaller_than_the_context_budget(app, tmp_path):
    """Sized to the whole budget, the folder is evicted on the very next turn."""
    from aichatlab.folderscan import Limits, survey
    from aichatlab.session import attachment_allowance

    (tmp_path / "big.txt").write_text("x" * 400_000, encoding="utf-8")
    app.settings["context_budget_tokens"] = 28_000
    result = survey(tmp_path)

    app._folder_accepted(tmp_path, result, Limits(max_file_bytes=10_000_000),
                         28_000)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.root.update()
        if app.attachments:
            break
        time.sleep(0.01)
    assert app.attachments, "the folder was never read"

    block_tokens = len(app.attachments[0]["content"]) // 4
    assert block_tokens <= attachment_allowance(app.settings.effective_budget())
    assert block_tokens < app.settings.effective_budget()


def test_closing_stops_the_event_pump(app):
    """Its `after` callback used to keep firing into a destroyed window."""
    assert app.pump_id is not None

    app.on_close()

    assert app.pump_id is None


# ------------------------------------------------ typing while it is working

class _Busy:
    """Stands in for a live worker thread."""

    def is_alive(self):
        return True


def test_a_message_typed_mid_run_is_held_rather_than_thrown_away(app):
    """"Still working — press Stop first" discarded what you typed and made
    you the scheduler."""
    app.worker = _Busy()
    app._hide_placeholder()
    app.input.insert("1.0", "what about the tests folder?")

    app.send()

    assert app.queued_message == "what about the tests folder?"
    assert app.input.get("1.0", "end").strip() == ""
    assert "is held until you say which" in app.chat.transcript()
    assert app.chat._actions


def test_an_urgent_message_leads_with_stopping(app):
    app.worker = _Busy()
    app._hide_placeholder()
    app.input.insert("1.0", "stop, that's the wrong file")

    app.send()

    assert action_buttons(app)[0].startswith("⏹ Stop that")


def test_an_ordinary_follow_up_leads_with_waiting(app):
    app.worker = _Busy()
    app._hide_placeholder()
    app.input.insert("1.0", "what does the cli module do?")

    app.send()

    assert action_buttons(app)[0].startswith("⏳ Wait")


def test_choosing_to_wait_keeps_the_message_queued(app):
    app.worker = _Busy()
    app._hide_placeholder()
    app.input.insert("1.0", "what about the tests?")
    app.send()
    action_id = pending_action(app)

    app._queue_after(action_id)

    assert app.queued_message == "what about the tests?"
    assert "Queued" in app.chat.transcript()


def test_choosing_to_stop_cancels_the_running_task(app, monkeypatch):
    app.worker = _Busy()
    app._hide_placeholder()
    app.input.insert("1.0", "stop and do this instead")
    app.send()
    action_id = pending_action(app)

    app._interrupt_with_queued(action_id)

    assert app.cancel.is_set()
    assert app.queued_message == "stop and do this instead"
    assert "Stopping the current task" in app.chat.transcript()


def test_the_queued_message_is_sent_once_the_worker_lets_go(app, monkeypatch):
    app.queued_message = "the held question"
    app.worker = None
    sent = []
    monkeypatch.setattr(app, "send", lambda: sent.append(app._input_text()))

    app._send_queued()

    assert sent == ["the held question"]
    assert app.queued_message == ""


def test_sending_nothing_mid_run_just_says_so(app):
    app.worker = _Busy()

    app.send()

    assert "queue it" in app.status.get()
    assert not app.chat._actions


# ------------------------------------------------------ the cut-off treadmill

def _cut_off(app, target, times):
    from aichatlab.client import ChatResult
    for _ in range(times):
        app.cutoff_streak[target.key] = app.cutoff_streak.get(target.key, 0) + 1
        app._offer_continue(target, ChatResult(model=target.model,
                                               done_reason="length"))


def test_the_first_cut_off_simply_offers_to_continue(app):
    from aichatlab.orchestrator import Target

    app.settings["speed_quality"] = 10
    _cut_off(app, Target("host", "gpt-oss:20b"), 1)

    assert action_buttons(app)[0] == "▶ Continue the answer"
    assert "times in a row" not in app.chat.transcript()


def test_a_repeat_cut_off_stops_pretending_continuing_will_finish(app):
    """Continuing a 192-token cap buys another 192 tokens that stop the same
    way — your screenshot showed that loop twice over."""
    from aichatlab.orchestrator import Target

    app.settings["speed_quality"] = 10
    _cut_off(app, Target("host", "gpt-oss:20b"), 2)

    transcript = app.chat.transcript()
    assert "cut off 2 times in a row" in transcript
    assert "speed slider is what sets this limit" in transcript
    # "Continue anyway" only ever appears on the demoted second offer
    assert "▶ Continue anyway" in action_buttons(app)


def test_an_uncapped_profile_never_blames_the_slider(app):
    from aichatlab.orchestrator import Target

    app.settings["speed_quality"] = 100
    _cut_off(app, Target("host", "gpt-oss:20b"), 3)

    assert "speed slider" not in app.chat.transcript()


def test_raising_the_limit_resets_the_streak(app, monkeypatch):
    from aichatlab.orchestrator import Target
    from aichatlab.session import make_key

    target = Target("host", "gpt-oss:20b")
    app.session.conversations[make_key("host", "gpt-oss:20b")] = [
        {"role": "user", "content": "tips please"}]
    app.settings["speed_quality"] = 10
    _cut_off(app, target, 2)
    monkeypatch.setattr(app, "_run", lambda job: None)

    app._raise_and_retry(target, pending_action(app))

    assert target.key not in app.cutoff_streak


# ------------------------------------------------------------ live progress

def test_the_status_line_names_the_model_and_counts_tokens(app):
    from aichatlab.orchestrator import Target

    app._handle("turn_start", {"stream_id": 1, "heading": "GPT-OSS 20B",
                               "target": Target("host", "gpt-oss:20b")})
    for _ in range(5):
        app._handle("token", {"stream_id": 1, "text": "word "})
    app._tick()

    line = app.status.get()
    assert "GPT-OSS 20B" in line
    assert "5 tokens" in line


def test_a_finished_turn_clears_the_activity_and_logs_it(app):
    from aichatlab.client import ChatResult
    from aichatlab.orchestrator import Target

    target = Target("host", "gpt-oss:20b")
    app._handle("turn_start", {"stream_id": 1, "heading": "h", "target": target})
    app._handle("turn_end", {"stream_id": 1, "target": target,
                             "result": ChatResult(model="gpt-oss:20b",
                                                  server="host", text="hi",
                                                  elapsed_s=2.0,
                                                  done_reason="stop")})

    assert not app.activity.busy()
    assert len(app.runlog) == 1
    assert app.runlog.entries[0].outcome == "finished"


def test_a_failed_turn_is_logged_with_its_error(app):
    from aichatlab.orchestrator import Target

    target = Target("host", "gpt-oss:20b")
    app._handle("turn_start", {"stream_id": 1, "heading": "h", "target": target})
    app._handle("turn_error", {"stream_id": 1, "target": target,
                               "message": "HTTP 500"})

    assert not app.activity.busy()
    assert app.runlog.entries[0].outcome == "error"
    assert "HTTP 500" in app.runlog.entries[0].detail


def test_a_stalled_step_says_so_once(app):
    import time as time_module

    app._handle("step", {"key": "compact", "step": "compacting",
                         "label": "Gemma2 9B", "detail": "14 messages"})
    app.activity.items["compact"].started -= 120
    app.activity.items["compact"].last_at -= 120
    app._tick()
    first = app.chat.transcript()
    app._tick()

    assert "Nothing from Gemma2 9B" in first or "sent nothing" in first
    assert first.count("Gemma2 9B has sent nothing") <= 1
    assert time_module is not None


def test_a_worker_step_shows_in_the_status_line(app):
    app._handle("step", {"key": "compact", "step": "compacting",
                         "label": "Gemma2 9B", "detail": "14 messages"})
    app._tick()

    line = app.status.get()
    assert "summarising the older turns" in line
    assert "14 messages" in line


def test_stopping_with_nothing_running_says_so(app):
    app.worker = None

    app.stop()

    assert "Nothing is running" in app.status.get()


def test_stopping_admits_it_cannot_interrupt_a_silent_server(app):
    app.worker = _Busy()

    app.stop()

    assert app.cancel.is_set()
    assert "as soon as the server sends" in app.chat.transcript()


# ------------------------------------------------------------- the checklist

def checklist_text(app):
    for line in app.chat.transcript().splitlines():
        if "This request" in line or "The plan" in line:
            start = app.chat.transcript().index(line)
            return app.chat.transcript()[start:start + 500]
    return ""


def test_the_checklist_lists_the_steps_this_send_will_take(app):
    from aichatlab.checklist import pipeline_for

    app.checklist = pipeline_for(attachments=[{"name": "udbg-phase1/"}],
                                 models=["Qwen2.5 32B Coder"], learning=True)
    app._redraw_checklist(0.0)

    text = checklist_text(app)
    assert "Read udbg-phase1/" in text
    assert "Ask Qwen2.5 32B Coder" in text
    assert "Save anything worth remembering" in text
    assert "(0/3)" in text


def test_a_model_ticks_over_as_it_starts_and_finishes(app):
    from aichatlab.checklist import DONE, RUNNING, pipeline_for
    from aichatlab.client import ChatResult
    from aichatlab.orchestrator import Target

    target = Target("host", "gpt-oss:20b")
    app.checklist = pipeline_for(models=["GPT-OSS 20B"])

    app._handle("turn_start", {"stream_id": 1, "heading": "h", "target": target})
    assert app.checklist.find("model:GPT-OSS 20B").state == RUNNING

    app._handle("turn_end", {"stream_id": 1, "target": target,
                             "result": ChatResult(model="gpt-oss:20b",
                                                  server="host", text="hi",
                                                  done_reason="stop")})
    assert app.checklist.find("model:GPT-OSS 20B").state == DONE


def test_a_run_that_stops_early_does_not_leave_steps_looking_pending(app):
    from aichatlab.checklist import PENDING, pipeline_for

    app.checklist = pipeline_for(models=["A", "B"], learning=True)

    app._handle("finished", {})

    assert app.checklist is None
    assert PENDING not in app.chat.transcript()
    assert "not reached" in app.chat.transcript()


def test_the_models_own_plan_replaces_the_pipeline_checklist(app):
    app._show_plan(["Audit error handling", "Check test coverage"])

    text = checklist_text(app)
    assert "The plan" in text
    assert "Audit error handling" in text
    assert "Pull it together" in text


def test_a_plan_step_ticks_from_the_orchestrators_event(app):
    from aichatlab.checklist import DONE

    app._show_plan(["Audit error handling", "Check test coverage"])

    app._handle("check", {"key": "step0", "state": "running", "detail": ""})
    assert "▸ Audit error handling" in app.chat.transcript()

    app._handle("check", {"key": "step0", "state": "done", "detail": ""})
    assert app.checklist.find("step0").state == DONE


# ------------------------------------------------- the long-prompt warning

def test_a_big_prompt_warns_before_the_silence_starts(app):
    from aichatlab.orchestrator import Target
    from aichatlab.session import estimate_tokens

    prompt = "x" * 100_000
    app._warn_about_prompt_size(prompt, [Target("host", "gpt-oss:20b")])

    note = app.chat.transcript()
    assert f"≈{estimate_tokens(prompt):,} tokens" in note
    assert "before the first word" in note


def test_a_small_prompt_says_nothing(app):
    from aichatlab.orchestrator import Target

    app._warn_about_prompt_size("hello", [Target("host", "gpt-oss:20b")])

    assert "tokens to" not in app.chat.transcript()


def test_the_estimate_is_grounded_in_this_machine_once_it_has_run_once(app):
    from aichatlab.client import ChatResult
    from aichatlab.orchestrator import Target

    app.runlog.record(ChatResult(model="gpt-oss:20b", prompt_tokens=24_000,
                                 raw={"prompt_eval_duration": 600 * 10**9}))
    app._warn_about_prompt_size("x" * 100_000, [Target("host", "gpt-oss:20b")])

    assert "should take about" in app.chat.transcript()


def test_no_estimate_is_invented_before_anything_has_run(app):
    from aichatlab.orchestrator import Target

    app._warn_about_prompt_size("x" * 100_000, [Target("host", "gpt-oss:20b")])

    assert "should take about" not in app.chat.transcript()


# --------------------------------------------- silence while reading a prompt

def test_silence_before_the_first_word_blames_the_prompt_not_the_server(app):
    """Nothing comes back while a model reads 23,845 tokens. Calling that a
    stall is simply the wrong story."""
    from aichatlab.orchestrator import Target
    from aichatlab.session import make_key

    key = make_key("host", "gpt-oss:20b")
    app.session.conversations[key] = [
        {"role": "user", "content": "x" * 95_000}]
    app._handle("turn_start", {"stream_id": 1, "heading": "h",
                               "target": Target("host", "gpt-oss:20b")})
    app.activity.items[1].started -= 300
    app.activity.items[1].last_at -= 300
    app._tick()

    from aichatlab.session import estimate_tokens

    transcript = app.chat.transcript()
    assert "still reading" in transcript
    assert f"{estimate_tokens('x' * 95_000):,}-token prompt" in transcript
    assert "check the server is reachable" not in transcript


def test_the_same_stall_sentence_is_never_shown_twice(app):
    app._handle("step", {"key": "a", "step": "compacting", "label": "Qwen3"})
    app._handle("step", {"key": "b", "step": "compacting", "label": "Qwen3"})
    for key in ("a", "b"):
        app.activity.items[key].started -= 300
        app.activity.items[key].last_at -= 300

    app._tick()

    assert app.chat.transcript().count("Qwen3 has sent nothing") <= 1


# ---------------------------------------------------- picking up where it left off

def test_an_unfinished_chat_is_restored_without_being_asked_about(tmp_path, monkeypatch):
    """Being greeted by a dialog about a file you have never heard of is a
    worse start than simply finding your work where you left it."""
    from aichatlab import recovery as recovery_module
    from aichatlab.session import Session

    path = tmp_path / "unfinished.json"
    monkeypatch.setattr(recovery_module, "RECOVERY_PATH", path)
    earlier = Session()
    earlier.add("local::alpha:1b", "user", "take a look at my app")
    earlier.add("local::alpha:1b", "assistant", "Here are a few ideas.")
    recovery_module.write(earlier, selected=["local::alpha:1b"], mode="plan",
                          path=path)

    app = _fresh_app(tmp_path, monkeypatch)
    try:
        assert app.session.conversations["local::alpha:1b"]
        assert "Picked up where you left off" in app.chat.transcript()
        assert "take a look at my app" in app.chat.transcript()
        assert not app.chat._actions          # nothing to click, nothing asked
    finally:
        _shut(app)


def test_nothing_is_restored_when_there_is_nothing_to_restore(tmp_path, monkeypatch):
    from aichatlab import recovery as recovery_module

    monkeypatch.setattr(recovery_module, "RECOVERY_PATH",
                        tmp_path / "absent.json")

    app = _fresh_app(tmp_path, monkeypatch)
    try:
        assert app.session.is_empty()
        assert "Picked up where" not in app.chat.transcript()
    finally:
        _shut(app)


def test_a_finished_turn_is_written_out(app, tmp_path, monkeypatch):
    from aichatlab import recovery as recovery_module

    path = tmp_path / "unfinished.json"
    monkeypatch.setattr(recovery_module, "RECOVERY_PATH", path)
    app.session.add("local::alpha:1b", "user", "hello")

    app._handle("finished", {})

    assert recovery_module.read(path).messages == 1


def test_closing_an_unsaved_chat_keeps_it_for_next_time(app, tmp_path, monkeypatch):
    from aichatlab import recovery as recovery_module

    path = tmp_path / "unfinished.json"
    monkeypatch.setattr(recovery_module, "RECOVERY_PATH", path)
    app.session.add("local::alpha:1b", "user", "an unfinished thought")
    app.session.path = None

    app.on_close()

    assert recovery_module.read(path) is not None


def test_closing_a_saved_chat_leaves_no_duplicate_behind(app, tmp_path, monkeypatch):
    """There is a real file with this conversation in it now."""
    from aichatlab import recovery as recovery_module

    path = tmp_path / "unfinished.json"
    monkeypatch.setattr(recovery_module, "RECOVERY_PATH", path)
    recovery_module.write(app.session, path=path)
    app.session.add("local::alpha:1b", "user", "hello")
    app.session.path = tmp_path / "chat.json"

    app.on_close()

    assert recovery_module.read(path) is None


def test_starting_a_new_chat_throws_the_recovery_file_away(app, tmp_path, monkeypatch):
    from aichatlab import recovery as recovery_module

    path = tmp_path / "unfinished.json"
    monkeypatch.setattr(recovery_module, "RECOVERY_PATH", path)
    app.session.add("local::alpha:1b", "user", "hello")
    recovery_module.write(app.session, path=path)

    app.new_chat()

    assert recovery_module.read(path) is None


def test_autosave_never_breaks_a_chat_when_it_cannot_write(app, monkeypatch):
    from aichatlab import recovery as recovery_module

    monkeypatch.setattr(recovery_module, "write",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    app.session.add("local::alpha:1b", "user", "hello")

    app._autosave()                       # must not raise


def _fresh_app(tmp_path, monkeypatch):
    """A second app instance, for the things that only happen at startup."""
    import json as json_module

    from aichatlab import cache as cache_module
    from aichatlab import config as config_module
    from aichatlab import knowledge as knowledge_module
    from aichatlab.ui.app import ChatLabApp

    settings = tmp_path / "settings2.json"
    settings.write_text(json_module.dumps({"local_ip": "", "host_ip": ""}),
                        encoding="utf-8")
    monkeypatch.setattr(config_module, "SETTINGS_PATH", settings)
    monkeypatch.setattr(cache_module, "CACHE_PATH", tmp_path / "cache2.json")
    monkeypatch.setattr(knowledge_module, "KNOWLEDGE_PATH", tmp_path / "k2.json")
    root = tk.Tk()
    root.withdraw()
    return ChatLabApp(root)


def _shut(app):
    for attribute in ("pump_id", "tick_id"):
        handle = getattr(app, attribute, None)
        if handle is not None:
            try:
                app.root.after_cancel(handle)
            except tk.TclError:
                pass
    try:
        app.root.destroy()
    except tk.TclError:
        pass


# ------------------------------- the checklist has to reflect what happened

def test_reading_the_attachments_is_ticked_because_it_already_happened(app):
    """It said "not reached" for work the transcript above it proved was
    done — a checklist that reports its own steps wrongly is worse than none."""
    from aichatlab.checklist import DONE, pipeline_for

    app.checklist = pipeline_for(attachments=[{"name": "udbg/"}],
                                 models=["Qwen2.5 32B Coder"])
    app.checklist.finish("attach", detail="1 attachment(s), ≈23,843 tokens")

    assert app.checklist.find("attach").state == DONE
    assert "23,843 tokens" in app.checklist.render()


def test_deciding_not_to_search_ticks_the_search_step(app):
    from aichatlab.checklist import SKIPPED, pipeline_for
    from aichatlab.orchestrator import Target
    from tests.conftest import FakeClient

    app.checklist = pipeline_for(searching=True, models=["A"])
    app.clients = lambda *a, **k: {"local": FakeClient(server="local"),
                                   "host": FakeClient()}

    app._research("take a look at my app", Target("local", "alpha:1b"), [],
                  [{"name": "udbg/", "content": "x" * 4000}])
    _drain(app)

    assert app.checklist.find("search").state == SKIPPED
    assert "not needed" in app.checklist.find("search").detail


def test_a_failed_search_is_ticked_as_failed(app, monkeypatch):
    from aichatlab.checklist import FAILED, pipeline_for
    from aichatlab.orchestrator import Target
    from aichatlab.research import ResearchError
    from tests.conftest import FakeClient

    app.checklist = pipeline_for(searching=True, models=["A"])
    app.settings["searxng_url"] = "http://searx.local"
    app.settings["assess_before_search"] = False
    app.clients = lambda *a, **k: {"local": FakeClient(server="local"),
                                   "host": FakeClient()}

    def boom(*a, **k):
        raise ResearchError("SearXNG refused the JSON request")

    monkeypatch.setattr("aichatlab.ui.app.gather", boom)

    app._research("what is the latest", Target("local", "alpha:1b"), [], [])
    _drain(app)

    assert app.checklist.find("search").state == FAILED


def test_nothing_worth_learning_ticks_learn_as_skipped(app):
    from aichatlab.checklist import SKIPPED, pipeline_for
    from aichatlab.orchestrator import Target

    app.checklist = pipeline_for(models=["A"], learning=True)

    app._learn("hi", "hello", Target("local", "alpha:1b"))
    _drain(app)

    assert app.checklist.find("learn").state == SKIPPED


def test_the_timeout_grows_with_the_prompt_the_client_is_given(app):
    app.settings["request_timeout"] = 600

    small = app.clients(0)["local"].timeout
    large = app.clients(24_000)["local"].timeout

    assert large > small


def _drain(app):
    """Run the queued events through the handler, as the pump would."""
    while not app.events.empty():
        kind, payload = app.events.get_nowait()
        app._handle(kind, payload)


def test_compaction_is_not_listed_when_there_is_nothing_to_fold(app):
    """A chat that is one big pinned attachment is over the threshold with
    nothing summarisable in it — listing the step promises work that will
    never happen."""
    from aichatlab.compaction import plan as plan_compaction
    from aichatlab.compaction import should_compact
    from aichatlab.session import make_key

    key = make_key("local", "alpha:1b")
    app.session.conversations[key] = [
        {"role": "user", "pinned": True,
         "content": "[Folder: udbg — 10 of 10 files included]\n" + "x" * 95_000
                    + "\n\n[End of folder contents.]\n\ngive me the list"}]
    history = app.session.history(key)

    assert should_compact(history, 28_000)          # it is certainly large
    assert not plan_compaction(history).worth_doing  # and nothing can be folded


def test_a_compaction_that_turns_out_not_to_be_worth_it_is_ticked_as_skipped(app):
    from aichatlab.checklist import SKIPPED, pipeline_for
    from aichatlab.orchestrator import Target
    from aichatlab.session import make_key
    from tests.conftest import FakeClient

    app.checklist = pipeline_for(models=["A"], compacting=True)
    app.session.conversations[make_key("local", "alpha:1b")] = [
        {"role": "user", "content": "short"}]
    app.clients = lambda *a, **k: {"local": FakeClient(server="local"),
                                   "host": FakeClient()}

    app._compact([Target("local", "alpha:1b")])
    _drain(app)

    assert app.checklist.find("compact").state == SKIPPED
    assert "nothing worth folding" in app.checklist.find("compact").detail


# --------------------------------------- switching model mid-conversation
def _seed(app, key, text="[Folder: udbg]\nsource\n[End of folder contents.]"):
    from aichatlab.session import make_key
    app.session.add(make_key(*key), "user", text)
    app.session.add(make_key(*key), "assistant", "Here is what I found.")


def test_a_model_with_no_history_is_offered_the_other_ones(app):
    from aichatlab.orchestrator import Target

    _seed(app, ("local", "qwen2.5-coder:32b"))

    paused = app._maybe_offer_history(
        "carry on", "chat", None, [Target("local", "qwen2.5:14b")])

    assert paused
    assert pending_action(app)
    transcript = app.chat.transcript()
    assert "has not seen any of this" in transcript
    assert "udbg" in transcript          # says *what* it would carry across


def test_no_offer_when_the_model_already_has_the_conversation(app):
    from aichatlab.orchestrator import Target

    _seed(app, ("local", "qwen2.5:14b"))

    assert not app._maybe_offer_history(
        "carry on", "chat", None, [Target("local", "qwen2.5:14b")])
    assert pending_action(app) is None


def test_no_offer_on_the_very_first_message_of_a_chat(app):
    from aichatlab.orchestrator import Target

    assert not app._maybe_offer_history(
        "hello", "chat", None, [Target("local", "qwen2.5:14b")])


def test_no_offer_when_broadcasting_to_several_models(app):
    """A comparison is *meant* to reach models that were not part of it."""
    from aichatlab.orchestrator import Target

    _seed(app, ("local", "qwen2.5-coder:32b"))

    assert not app._maybe_offer_history(
        "compare", "chat", None,
        [Target("local", "alpha:1b"), Target("local", "beta:1b")])


def test_carrying_it_across_copies_the_messages_and_resends(app, monkeypatch):
    from aichatlab.orchestrator import Target
    from aichatlab.session import make_key

    _seed(app, ("local", "qwen2.5-coder:32b"))
    target = Target("local", "qwen2.5:14b")
    app._maybe_offer_history("carry on", "chat", None, [target])
    action_id = pending_action(app)
    sent = []
    monkeypatch.setattr(app, "send", lambda: sent.append(True))

    app._carry_history(make_key("local", "qwen2.5-coder:32b"), target, action_id)

    assert len(app.session.history(target.key)) == 2
    assert "udbg" in app.session.history(target.key)[0]["content"]
    assert sent
    assert app.skip_carry_prompt          # the resend must not ask again
    assert "Carried 2 messages" in app.chat.transcript()


def test_carrying_it_across_keeps_the_attachment_pinned(app):
    """A copy that loses `pinned` is a folder that gets trimmed away first."""
    from aichatlab.orchestrator import Target
    from aichatlab.session import make_key

    donor = make_key("local", "qwen2.5-coder:32b")
    app.session.add(donor, "user", "[Folder: udbg]\nsource\n[End of folder "
                                   "contents.]", pinned=True)
    target = Target("local", "qwen2.5:14b")

    app._carry_history(donor, target, "nope")

    assert app.session.history(target.key)[0].get("pinned")


def test_the_donor_conversation_is_left_alone(app):
    from aichatlab.orchestrator import Target
    from aichatlab.session import make_key

    donor = make_key("local", "qwen2.5-coder:32b")
    _seed(app, ("local", "qwen2.5-coder:32b"))
    target = Target("local", "qwen2.5:14b")

    app._carry_history(donor, target, "nope")
    app.session.add(target.key, "user", "something new")

    assert len(app.session.conversations[donor]) == 2


def test_starting_fresh_leaves_the_history_empty_and_resends(app, monkeypatch):
    from aichatlab.orchestrator import Target

    _seed(app, ("local", "qwen2.5-coder:32b"))
    target = Target("local", "qwen2.5:14b")
    app._maybe_offer_history("carry on", "chat", None, [target])
    action_id = pending_action(app)
    sent = []
    monkeypatch.setattr(app, "send", lambda: sent.append(True))

    app._start_fresh(action_id)

    assert not app.session.history(target.key)
    assert sent
    assert app.skip_carry_prompt


def test_the_richest_conversation_is_the_one_offered(app):
    from aichatlab.session import make_key

    thin = make_key("local", "alpha:1b")
    fat = make_key("local", "beta:1b")
    app.session.add(thin, "user", "hi")
    app.session.add(fat, "user", "x" * 8000)

    assert app._richest_conversation(make_key("local", "gamma:1b")) == fat


def test_the_context_label_names_the_model_it_is_counting(app):
    """Per-model counts under a transcript that shows every model read as a
    bug until the label says whose tokens they are."""
    from aichatlab.orchestrator import Target
    from aichatlab.session import make_key

    app.session.add(make_key("local", "qwen2.5:14b"), "user", "hello there")
    app.selected_targets = lambda: [Target("local", "qwen2.5:14b")]

    app._update_context_label()

    assert app.context_label.cget("text").startswith("Qwen2.5 14B: context ≈")


def test_the_context_label_stays_anonymous_for_several_models(app):
    from aichatlab.orchestrator import Target

    app.selected_targets = lambda: [Target("local", "alpha:1b"),
                                    Target("local", "beta:1b")]

    app._update_context_label()

    assert app.context_label.cget("text").startswith("context ≈")


# ------------------------------------------------------- reasoning models
def test_the_reasoning_toggle_is_remembered(app):
    app.thinking_var.set(False)
    app._on_thinking_toggle()

    assert app.settings.get("thinking") is False


def test_the_orchestrator_is_told_whether_reasoning_is_wanted(app):
    app.thinking_var.set(False)
    assert app._make_orchestrator().think is False

    app.thinking_var.set(True)
    assert app._make_orchestrator().think is True


def test_triage_never_asks_a_reasoning_model_to_reason(app):
    """Triage runs on 64 tokens.  Qwen3 would spend them all thinking and
    return nothing, so the app would search — or not — on a coin toss."""
    from aichatlab.orchestrator import Target
    from tests.conftest import FakeClient

    client = FakeClient(server="local", reply="ANSWER")
    app.clients = lambda *a, **k: {"local": client, "host": FakeClient()}

    app._triage("take a look at my app",
                [{"name": "udbg/", "content": "x" * 4000}], [],
                Target("local", "qwen3:8b"))
    _drain(app)

    assert client.last_think is False


def test_learning_never_asks_a_reasoning_model_to_reason(app):
    from aichatlab.orchestrator import Target
    from tests.conftest import FakeClient

    client = FakeClient(server="local", reply="- always check VRAM first")
    app.clients = lambda *a, **k: {"local": client, "host": FakeClient()}
    app.learning_wanted = True

    app._learn("why is it slow?",
               "Because Ollama put the model on the CPU rather than the "
               "graphics card, which is what it does silently whenever the "
               "weights and the context cache together will not fit in the "
               "video memory available on a single card.",
               Target("local", "qwen3:8b"))
    _drain(app)

    assert client.calls, "the learning call never happened"
    assert client.last_think is False


def test_a_reasoning_reply_streams_its_working_into_the_view(app):
    app._handle("turn_start", {"stream_id": 7, "target": _target("qwen3:8b"),
                               "heading": "Qwen3 8B"})
    app._handle("thought", {"stream_id": 7, "text": "weighing it up"})

    assert app.chat.thought_text(7) == "weighing it up"
    assert app.activity.items[7].step == "reasoning"


def _target(model: str):
    from aichatlab.orchestrator import Target
    return Target("local", model)


# ------------------------------------------------------------- auto mode
class _Cut:
    """A reply that stopped at the length cap."""

    def __init__(self, text: str):
        self.text = text
        self.truncated = True
        self.cancelled = False
        self.done_reason = "length"


BIG = ("The loader rebuilds the symbol table on every open, which is where "
       "most of the start-up cost on a large binary goes, and folding it into "
       "the relocation walk removes a whole traversal. ")


def test_auto_mode_continues_without_asking(app):
    app.auto_var.set(True)
    app.auto.begin(time.monotonic())
    target = _target("alpha:1b")

    handled = app._auto_continue(target, _Cut(BIG))

    assert handled, "auto mode should have taken this"
    assert pending_action(app) is None, "it must not stop to ask"
    assert app.pending_auto == target
    assert "carrying on automatically" in app.chat.transcript()


def test_the_continuation_waits_for_the_worker_to_finish(app):
    """`_continue_reply` refuses to start on top of a running worker, so
    firing it from turn_end would silently do nothing."""
    app.auto_var.set(True)
    app.auto.begin(time.monotonic())

    app._auto_continue(_target("alpha:1b"), _Cut(BIG))

    assert app.pending_auto is not None, "it must be queued, not fired"


def test_auto_mode_off_falls_back_to_asking(app):
    app.auto_var.set(False)

    assert not app._auto_continue(_target("alpha:1b"), _Cut(BIG))


def test_auto_mode_stops_and_says_why(app):
    from aichatlab.autopilot import Limits

    app.auto_var.set(True)
    app.auto.limits = Limits(continues=1)
    app.auto.begin(time.monotonic())
    app._auto_continue(_target("alpha:1b"), _Cut(BIG))
    app.pending_auto = None

    handled = app._auto_continue(_target("alpha:1b"), _Cut(BIG + BIG + "x" * 60))

    assert not handled, "it must hand back to the manual prompt"
    assert "Auto mode stopped here" in app.chat.transcript()


def test_pressing_stop_ends_the_overnight_run(app):
    """Stop during an unattended run means stop, not pause."""
    app.auto_var.set(True)
    app.auto.begin(time.monotonic())
    app.pending_auto = _target("alpha:1b")
    import threading as _threading
    import time as _time
    app.worker = _threading.Thread(target=lambda: _time.sleep(0.3))
    app.worker.start()

    app.stop()

    assert not app.auto_var.get()
    assert app.pending_auto is None
    app.worker.join()


def test_auto_mode_answers_the_carry_prompt_itself(app):
    """Nothing may sit waiting for a person who is asleep."""
    from aichatlab.session import make_key

    app.auto_var.set(True)
    app.auto.begin(time.monotonic())
    _seed(app, ("local", "qwen2.5-coder:32b"))
    target = _target("qwen3:8b")

    paused = app._maybe_offer_history("carry on", "chat", None, [target])

    assert not paused, "auto mode must not stop the send"
    assert pending_action(app) is None
    assert len(app.session.history(target.key)) == 2
    assert any("carried" in d for d in app.auto.decisions)
    assert make_key("local", "qwen2.5-coder:32b") in app.session.conversations


def test_auto_mode_answers_the_search_prompt_itself(app):
    app.auto_var.set(True)
    app.auto.begin(time.monotonic())
    app.research_var.set(False)
    app.settings["searxng_url"] = "http://searx.local"

    paused = app._maybe_ask_to_search("look up the latest firmware", "chat",
                                      None)

    assert not paused
    assert app.research_var.get(), "SearXNG is configured, so it should search"
    assert any("web research" in d for d in app.auto.decisions)


def test_without_searxng_it_answers_from_memory_rather_than_waiting(app):
    app.auto_var.set(True)
    app.auto.begin(time.monotonic())
    app.research_var.set(False)
    app.settings["searxng_url"] = ""

    paused = app._maybe_ask_to_search("look up the latest firmware", "chat",
                                      None)

    assert not paused
    assert not app.research_var.get()
    assert any("from memory" in d for d in app.auto.decisions)


def test_switching_auto_off_reports_what_it_did(app):
    app.auto_var.set(True)
    app.auto.begin(time.monotonic())
    app.auto.note("carried a conversation across")
    app.auto_var.set(False)

    app._on_auto_toggle()

    assert "Decisions made for you" in app.chat.transcript()


# --------------------------------------------------------- graphics cards
def test_two_cards_prompt_to_enable_spreading(app):
    from aichatlab.gpu import Card

    app._cards_found([Card(0, "GTX 1080 Ti", 11264 * 1024 * 1024),
                      Card(1, "GTX 1080 Ti", 11264 * 1024 * 1024)])

    assert pending_action(app)
    transcript = app.chat.transcript()
    assert "OLLAMA_SCHED_SPREAD" in transcript
    assert "11 GB each" in transcript


def test_one_card_is_left_alone(app):
    from aichatlab.gpu import Card

    app._cards_found([Card(0, "RTX 4090", 24564 * 1024 * 1024)])

    assert pending_action(app) is None


def test_declining_is_remembered_so_it_stops_asking(app):
    from aichatlab.gpu import Card

    cards = [Card(0, "GTX 1080 Ti", 11264 * 1024 * 1024),
             Card(1, "GTX 1080 Ti", 11264 * 1024 * 1024)]
    app._cards_found(cards)
    app._decline_spread(pending_action(app))
    app.chat.clear()

    app._cards_found(cards)

    assert pending_action(app) is None
    assert app.settings.get("skip_spread_prompt")


def test_real_card_capacity_replaces_the_guess(app):
    """`vram_seen` was a lower bound learned from watching a load go badly;
    nvidia-smi knows the answer."""
    from aichatlab.gpu import Card

    app.vram_seen = 5 * 10**9

    app._cards_found([Card(0, "GTX 1080 Ti", 11264 * 1024 * 1024)])

    assert app.vram_seen == 11264 * 1024 * 1024


def test_no_cards_found_changes_nothing(app):
    app.vram_seen = 7
    app._cards_found([])

    assert app.vram_seen == 7
    assert pending_action(app) is None


def test_the_sidebar_always_says_whether_spreading_is_on(app):
    """A setting that only ever surfaced as a one-time prompt has no state you
    can check, and "is it on?" is the first thing anyone asks."""
    from aichatlab.gpu import Card

    app._cards_found([Card(0, "GTX 1080 Ti", 11264 * 1024 * 1024),
                      Card(1, "GTX 1080 Ti", 11264 * 1024 * 1024)])

    assert "2 × GTX 1080 Ti" in app.gpu_label.cget("text")
    assert "22 GB" in app.gpu_label.cget("text")
    assert "OFF" in app.spread_label.cget("text")


def test_the_sidebar_says_so_when_spreading_is_already_on(app, monkeypatch):
    from aichatlab import gpu as gpu_module
    from aichatlab.gpu import Card

    monkeypatch.setattr(gpu_module, "spread_enabled", lambda *a, **k: True)
    app._cards_found([Card(0, "GTX 1080 Ti", 11264 * 1024 * 1024),
                      Card(1, "GTX 1080 Ti", 11264 * 1024 * 1024)])

    assert app.spread_label.cget("text").endswith("on")
    assert pending_action(app) is None, "nothing to offer when it is already on"


def test_finding_no_cards_says_so_rather_than_staying_blank(app):
    """Silence here is indistinguishable from a probe that never ran."""
    app._cards_found([])

    assert "No NVIDIA cards detected" in app.gpu_label.cget("text")
    assert "nvidia-smi" in app.spread_label.cget("text")


def test_a_single_card_needs_no_spread_state(app):
    from aichatlab.gpu import Card

    app._cards_found([Card(0, "RTX 4090", 24564 * 1024 * 1024)])

    assert "RTX 4090" in app.gpu_label.cget("text")
    assert app.spread_label.cget("text") == "", \
        "one card cannot be spread across, so there is no state to report"


# ------------------------------------------------------- downloading models
def test_adding_a_model_pulls_it_through_the_server(app, monkeypatch):
    """The pull runs server-side, which is the point: it works from a machine
    whose own outbound access is blocked, and for a container with no shell."""
    from aichatlab.ui import app as app_module

    calls = []

    class PullClient:
        base_url = "http://fake:11434"

        def pull(self, model, on_progress=None, cancel=None, timeout=60):
            calls.append(model)
            on_progress({"status": "pulling manifest"})
            on_progress({"status": "pulling", "digest": "a",
                         "total": 1000, "completed": 1000})
            on_progress({"status": "success"})
            return ""

    monkeypatch.setattr(app_module.simpledialog, "askstring",
                        lambda *a, **k: "qwen3:8b")
    monkeypatch.setattr(app, "clients", lambda *a, **k: {"host": PullClient()})
    monkeypatch.setattr(app, "refresh_models", lambda *a, **k: None)
    app.settings["host_ip"] = "10.0.0.5"

    app.add_model("host")
    for _ in range(200):
        _drain(app)
        if not app.pulling:
            break
        time.sleep(0.01)

    assert calls == ["qwen3:8b"]
    assert "downloaded to the host server" in app.chat.transcript()


def test_a_failed_download_says_why(app, monkeypatch):
    from aichatlab.ui import app as app_module

    class FailingClient:
        base_url = "http://fake:11434"

        def pull(self, model, on_progress=None, cancel=None, timeout=60):
            return "pull model manifest: file does not exist"

    monkeypatch.setattr(app_module.simpledialog, "askstring",
                        lambda *a, **k: "qwn3:8b")
    monkeypatch.setattr(app, "clients", lambda *a, **k: {"host": FailingClient()})
    monkeypatch.setattr(app, "refresh_models", lambda *a, **k: None)
    app.settings["host_ip"] = "10.0.0.5"

    app.add_model("host")
    for _ in range(200):
        _drain(app)
        if not app.pulling:
            break
        time.sleep(0.01)

    assert "Could not download" in app.chat.transcript()
    assert "does not exist" in app.chat.transcript()


def test_an_obvious_typo_never_reaches_the_server(app, monkeypatch):
    from aichatlab.ui import app as app_module

    monkeypatch.setattr(app_module.simpledialog, "askstring",
                        lambda *a, **k: "please get me qwen3")
    monkeypatch.setattr(app_module.messagebox, "showwarning",
                        lambda *a, **k: None)
    app.settings["host_ip"] = "10.0.0.5"

    app.add_model("host")

    assert not app.pulling


def test_cancelling_a_download_is_what_stop_does(app):
    app.pulling = "qwen3:8b"

    app.stop()

    assert app.pull_cancel.is_set()
    assert "Stopping the download" in app.status.get()


def test_two_downloads_at_once_are_refused(app, monkeypatch):
    from aichatlab.ui import app as app_module

    monkeypatch.setattr(app_module.simpledialog, "askstring",
                        lambda *a, **k: "qwen3:8b")
    app.settings["host_ip"] = "10.0.0.5"
    app.pulling = "gemma3:12b"

    app.add_model("host")

    assert app.pulling == "gemma3:12b"
    assert "Already downloading" in app.status.get()


def test_a_repeated_model_name_does_not_orphan_a_checkbox(app):
    """Two rows sharing one (server, name) key leaves the first unclickable."""
    app._populate("host", ["alpha:1b", "alpha:1b", "beta:2b"])

    assert len([k for k in app.model_vars if k[0] == "host"]) == 2


def test_the_sidebar_reports_free_vram_when_something_else_holds_it(app):
    """"22 GB" under a model that just ran on the CPU is the number that
    makes the whole thing inexplicable."""
    from aichatlab.gpu import Card

    gib = 1024 * 1024 * 1024
    app._cards_found([Card(0, "GTX 1080 Ti", 11 * gib, used=8 * gib),
                      Card(1, "GTX 1080 Ti", 11 * gib, used=8 * gib)])

    label = app.gpu_label.cget("text")
    assert "22 GB" in label
    assert "6.0 GB free" in label
    assert "already in use" in app.chat.transcript()


def test_idle_cards_are_reported_plainly(app):
    from aichatlab.gpu import Card

    gib = 1024 * 1024 * 1024
    app._cards_found([Card(0, "GTX 1080 Ti", 11 * gib, used=0),
                      Card(1, "GTX 1080 Ti", 11 * gib, used=0)])

    assert "free" not in app.gpu_label.cget("text")
    assert "already in use" not in app.chat.transcript()


def test_the_same_vram_warning_is_not_repeated(app):
    from aichatlab.gpu import Card

    gib = 1024 * 1024 * 1024
    cards = [Card(0, "GTX 1080 Ti", 11 * gib, used=8 * gib),
             Card(1, "GTX 1080 Ti", 11 * gib, used=8 * gib)]
    app._cards_found(cards)
    app._cards_found(cards)

    assert app.chat.transcript().count("already in use") == 1


def test_the_cards_are_re_read_but_not_on_every_message(app):
    """Free VRAM changes between messages; nvidia-smi is cheap but not free."""
    calls = []
    app._check_cards = lambda: calls.append(1)
    app.cards_checked_at = 0.0

    app._refresh_cards_soon()
    app._refresh_cards_soon()

    assert len(calls) == 1


def test_unticking_learning_mid_reply_actually_stops_it(app):
    """Learning runs *after* the answer, so unlike research and reasoning it
    has not happened yet when you change your mind."""
    from aichatlab.checklist import SKIPPED, pipeline_for
    from tests.conftest import FakeClient

    client = FakeClient(server="local", reply="- a lesson")
    app.clients = lambda *a, **k: {"local": client, "host": FakeClient()}
    app.checklist = pipeline_for(learning=True, models=["A"])
    app.learning_var.set(True)
    app._on_learning_toggle()

    # The reply is on screen; now the box is unticked.
    app.learning_var.set(False)
    app._on_learning_toggle()

    app._learn("why is it slow?",
               "Because Ollama put the model on the processor rather than "
               "the graphics card, which is what it does silently whenever "
               "the weights will not fit in the video memory available.",
               _target("alpha:1b"))
    _drain(app)

    assert not client.calls, "the learning call should never have happened"
    assert app.checklist.find("learn").state == SKIPPED
    assert "switched off" in app.checklist.find("learn").detail


def test_learning_still_runs_when_the_box_is_left_alone(app):
    from tests.conftest import FakeClient

    client = FakeClient(server="local", reply="- a lesson")
    app.clients = lambda *a, **k: {"local": client, "host": FakeClient()}
    app.learning_var.set(True)
    app._on_learning_toggle()

    app._learn("why is it slow?",
               "Because Ollama put the model on the processor rather than "
               "the graphics card, which is what it does silently whenever "
               "the weights will not fit in the video memory available.",
               _target("alpha:1b"))
    _drain(app)

    assert client.calls, "learning was requested and should have run"


def test_the_worker_never_reads_the_tk_variable_itself(app):
    """Reading a Tk variable off the main thread is a Tcl call off the main
    thread — the thing that used to abort the interpreter outright."""
    import inspect

    source = inspect.getsource(app._learn.__func__)

    assert "learning_var" not in source
    assert "learning_wanted" in source


# --------------------------------------- when compaction cannot help
def _attachment_chat(app, key):
    folder = ("[Folder: udbg-phase1 — 41 files]\n" + "x" * 200_000
              + "\n[End of folder contents.]")
    app.session.add(key, "user", folder, pinned=True)
    app.session.add(key, "assistant", "I have read all 41 files.")


def test_the_label_stops_recommending_a_button_that_will_refuse(app):
    """"Compact to free space" next to a Compact button that says there is
    nothing to compact is the app arguing with itself."""
    from aichatlab.orchestrator import Target

    target = Target("local", "qwen3:8b")
    _attachment_chat(app, target.key)
    app.selected_targets = lambda: [target]
    app.settings["context_budget_tokens"] = 28_000

    app._update_context_label()

    text = app.context_label.cget("text")
    assert "oldest turns are being dropped" in text
    assert "🗜 Compact" not in text


def test_the_label_still_recommends_compacting_when_it_would_work(app):
    from aichatlab.orchestrator import Target

    target = Target("local", "qwen3:8b")
    for _ in range(14):
        app.session.add(target.key, "user", "question " * 400)
        app.session.add(target.key, "assistant", "answer " * 400)
    app.selected_targets = lambda: [target]
    app.settings["context_budget_tokens"] = 28_000

    app._update_context_label()

    assert "🗜 Compact" in app.context_label.cget("text")


def test_the_compact_button_explains_the_real_reason(app, monkeypatch):
    from aichatlab.orchestrator import Target
    from aichatlab.ui import app as app_module

    shown = {}
    monkeypatch.setattr(app_module.messagebox, "showinfo",
                        lambda title, message, **k: shown.update(
                            title=title, message=message))
    target = Target("local", "qwen3:8b")
    _attachment_chat(app, target.key)
    app.selected_targets = lambda: [target]
    app.settings["context_budget_tokens"] = 28_000

    app.compact_chat()

    assert "short enough" not in shown["message"]
    assert "udbg-phase1" in shown["message"]
    assert "new chat" in shown["message"]


def test_refresh_also_re_reads_the_graphics_cards(app):
    """Free VRAM changes far more often than the model list does."""
    calls = []
    app._check_cards = lambda: calls.append(1)
    app.cards_checked_at = time.monotonic()      # just checked

    app.refresh_models()

    assert calls, "Refresh should have re-read the cards regardless"


# ------------------------------------------------------- project memory
def test_a_new_chat_on_a_known_project_starts_with_what_was_learned(app):
    import tempfile
    from pathlib import Path

    from aichatlab import projectmemory as pm

    project = Path(tempfile.mkdtemp())
    pm.save_brief(project, "udbg is a small ELF debugger.")
    pm.save_progress(project, "Halfway through folding the symbol table.")
    pm.append_decision(project, "Load lazily — start-up cost dominates.")

    target = _target("qwen3:8b")
    app.selected_targets = lambda: [target]
    app.project = project

    app._pin_project_memory()

    history = app.session.history(target.key)
    assert history, "the memory should have been pinned into the conversation"
    assert history[0].get("pinned"), "it must not be the first thing trimmed"
    assert "ELF debugger" in history[0]["content"]
    assert "symbol table" in history[0]["content"]
    assert "Picked up where we left off" in app.chat.transcript()


def test_an_unknown_project_pins_nothing(app, tmp_path):
    target = _target("qwen3:8b")
    app.selected_targets = lambda: [target]
    app.project = tmp_path

    app._pin_project_memory()

    assert not app.session.history(target.key)
    assert "Picked up where" not in app.chat.transcript()


def test_the_memory_is_not_pinned_twice(app, tmp_path):
    from aichatlab import projectmemory as pm

    pm.save_brief(tmp_path, "udbg is a small ELF debugger.")
    target = _target("qwen3:8b")
    app.selected_targets = lambda: [target]
    app.project = tmp_path

    app._pin_project_memory()
    app._pin_project_memory()

    assert len(app.session.history(target.key)) == 1


def test_ending_a_chat_writes_a_handoff(app, tmp_path):
    from aichatlab import projectmemory as pm
    from tests.conftest import FakeClient

    reply = ("PROGRESS:\nThe loader re-checks the gate at read time. Symbols "
             "still half done.\n\nDECISION:\nRe-classify at read time, because "
             "contents can change between passes.")
    client = FakeClient(server="local", reply=reply)
    app.clients = lambda *a, **k: {"local": client, "host": FakeClient()}
    target = _target("qwen3:8b")
    app.selected_targets = lambda: [target]
    app.project = tmp_path
    for _ in range(3):
        app.session.add(target.key, "user", "a real question " * 10)
        app.session.add(target.key, "assistant", "a real answer " * 10)

    assert app._write_handoff()

    memory = pm.load(tmp_path)
    assert "half done" in memory.progress
    assert "Re-classify at read time" in memory.decisions


def test_a_chat_where_nothing_happened_leaves_no_note(app, tmp_path):
    from tests.conftest import FakeClient

    client = FakeClient(server="local", reply="PROGRESS:\nstuff")
    app.clients = lambda *a, **k: {"local": client}
    target = _target("qwen3:8b")
    app.selected_targets = lambda: [target]
    app.project = tmp_path
    app.session.add(target.key, "user", "hi")

    assert not app._write_handoff()
    assert not client.calls, "it should not even have asked"


def test_no_project_means_no_handoff(app):
    app.project = None

    assert not app._write_handoff()


def test_the_attachment_is_kept_out_of_the_handoff_prompt(app, tmp_path):
    """Asking a model to summarise 24,000 tokens of source into a six-line
    progress note is a slow way to get a worse answer."""
    from tests.conftest import FakeClient

    client = FakeClient(server="local", reply="PROGRESS:\nfine\n\nDECISION:\nNONE")
    app.clients = lambda *a, **k: {"local": client}
    target = _target("qwen3:8b")
    app.selected_targets = lambda: [target]
    app.project = tmp_path
    app.session.add(target.key, "user",
                    "[Folder: udbg]\n" + "x" * 90_000 + "\n[End of folder "
                    "contents.]", pinned=True)
    for _ in range(3):
        app.session.add(target.key, "user", "a real question " * 10)
        app.session.add(target.key, "assistant", "a real answer " * 10)

    app._write_handoff()

    assert "xxxxxxxxxx" not in client.last_prompt


def test_a_model_that_refuses_the_shape_breaks_nothing(app, tmp_path):
    from tests.conftest import FakeClient

    client = FakeClient(server="local", reply="I'm not sure what you mean.")
    app.clients = lambda *a, **k: {"local": client}
    target = _target("qwen3:8b")
    app.selected_targets = lambda: [target]
    app.project = tmp_path
    for _ in range(3):
        app.session.add(target.key, "user", "a real question " * 10)
        app.session.add(target.key, "assistant", "a real answer " * 10)

    assert not app._write_handoff()      # nothing usable, nothing written


def test_a_stale_risk_question_pauses_even_without_a_search_word(app):
    """The dangerous case is not "look this up" — it is a question that needs
    a lookup and never says so."""
    app.research_var.set(False)

    paused = app._maybe_ask_to_search(
        "what is the 0-60 of the newest audi r8?", "chat", None)

    assert paused
    assert pending_action(app)
    transcript = app.chat.transcript()
    assert "something that changes" in transcript
    assert "newest" in transcript


def test_the_wording_differs_from_an_explicit_request(app):
    app.research_var.set(False)
    app._maybe_ask_to_search("look up the newest firmware", "chat", None)

    assert "You asked me to" in app.chat.transcript()


def test_the_same_staleness_prompt_is_not_repeated_all_chat(app):
    """Helpful the first time you ask about "the latest" something, nagging
    by the third."""
    app.research_var.set(False)

    assert app._maybe_ask_to_search("what is the newest R8?", "chat", None)
    assert not app._maybe_ask_to_search("and the newest M3?", "chat", None)


def test_a_timeless_question_is_still_sent_straight_through(app):
    app.research_var.set(False)

    assert not app._maybe_ask_to_search(
        "explain how a heat pump works", "chat", None)


# ------------------------------------------------------------- retrieval
LOADER = "int read_source_guarded(void) { verdict v = classify_file(); }"
SYMBOLS = "void rebuild_symbol_table(image *img) { hash_insert(); }"


def _project(app, tmp_path, count=20):
    app.project = tmp_path
    app.project_texts = {f"pad{i}.c": f"void pad{i}(void) {{}}"
                         for i in range(count)}
    app.project_texts["loader.c"] = LOADER
    app.project_texts["symbols.c"] = SYMBOLS
    app.project_index = None


def test_a_small_folder_is_still_sent_whole(app, tmp_path):
    """Retrieval earns its keep on a big folder; on five files it is just a
    way to leave things out."""
    app.project_texts = {"a.c": "x", "b.c": "y"}

    assert not app._retrieval_wanted()


def test_a_big_folder_switches_to_choosing_per_question(app, tmp_path):
    _project(app, tmp_path)

    assert app._retrieval_wanted()


def test_only_the_relevant_files_are_sent(app, tmp_path):
    _project(app, tmp_path)
    app.model_vars = {}                       # no embedding model installed

    block = app._retrieval_block("where is the symbol table rebuilt?")
    _drain(app)

    assert "symbols.c" in block
    assert "rebuild_symbol_table" in block
    assert "pad7.c" not in block, "irrelevant files must not be sent"


def test_what_was_sent_is_always_reported(app, tmp_path):
    _project(app, tmp_path)
    app.model_vars = {}

    app._retrieval_block("where is the symbol table rebuilt?")
    _drain(app)

    transcript = app.chat.transcript()
    assert "Sent" in transcript and "symbols.c" in transcript


def test_retrieval_works_with_no_embedding_model_at_all(app, tmp_path):
    """Keyword ranking is the fallback, not a broken state."""
    _project(app, tmp_path)
    app.model_vars = {}

    assert app._embedding_model() == ""
    assert app._retrieval_block("read_source_guarded")


def test_an_installed_embedding_model_is_found(app):
    import tkinter as tk

    app.model_vars = {("local", "nomic-embed-text:latest"): tk.BooleanVar()}

    assert app._embedding_model() == "nomic-embed-text:latest"


def test_the_selection_respects_the_attachment_budget(app, tmp_path):
    app.project = tmp_path
    app.project_texts = {f"big{i}.c": "classify_file " * 20_000
                         for i in range(20)}
    app.project_index = None
    app.model_vars = {}
    app.settings["context_budget_tokens"] = 28_000

    from aichatlab.session import estimate_tokens

    block = app._retrieval_block("classify_file")
    _drain(app)

    assert estimate_tokens(block) <= app.settings.effective_budget()


def test_indexing_a_project_without_a_model_is_harmless(app, tmp_path):
    _project(app, tmp_path)
    app.model_vars = {}

    app._index_project()          # must not raise

    assert app.project_index is not None


# ------------------------------------------------------------ editing files
REPLY_WITH_EDIT = """\
I'd simplify the loader.

=== FILE: loader.c ===
int read_source_guarded(void) { return 1; }
=== END FILE ===
"""


def test_a_proposed_edit_waits_for_consent(app, tmp_path):
    """Nothing this app does may write to someone's project unasked."""
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path

    app._check_for_edits(REPLY_WITH_EDIT)

    assert pending_action(app), "it must stop and ask"
    assert (tmp_path / "loader.c").read_text(encoding="utf-8") == "int old;\n"
    assert "Nothing has been written yet" in app.chat.transcript()


def test_discarding_writes_nothing(app, tmp_path):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app._check_for_edits(REPLY_WITH_EDIT)

    app._discard_edits(pending_action(app))

    assert (tmp_path / "loader.c").read_text(encoding="utf-8") == "int old;\n"
    assert not app.pending_edits


def test_applying_writes_and_keeps_a_backup(app, tmp_path):
    from aichatlab import edits as edit_tools

    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    proposed, _ = edit_tools.parse(REPLY_WITH_EDIT)

    app._apply_edits(proposed, always=False)

    assert "return 1" in (tmp_path / "loader.c").read_text(encoding="utf-8")
    assert list(edit_tools.backup_root(tmp_path).rglob("loader.c"))
    assert "Wrote 1 file" in app.chat.transcript()


def test_the_stop_asking_choice_applies_without_a_prompt(app, tmp_path):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.auto_apply_edits = True

    app._check_for_edits(REPLY_WITH_EDIT)

    transcript = app.chat.transcript()
    assert "Nothing has been written yet" not in transcript, "it must not ask"
    assert "return 1" in (tmp_path / "loader.c").read_text(encoding="utf-8")
    # An undo is still offered — especially here, since nobody looked at it.
    assert "puts every one of those files back" in transcript


def test_that_choice_dies_with_the_chat(app, tmp_path):
    """"Write to my disk without asking" must not outlive the conversation
    that granted it."""
    app.project = tmp_path
    app.auto_apply_edits = True

    app.new_chat()

    assert not app.auto_apply_edits
    assert not app.pending_edits


def test_no_project_means_edits_are_ignored_entirely(app):
    app.project = None

    app._check_for_edits(REPLY_WITH_EDIT)

    assert pending_action(app) is None


def test_a_cut_off_reply_is_reported_and_never_applied(app, tmp_path):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path

    app._check_for_edits("=== FILE: loader.c ===\nint half", truncated=True)

    assert pending_action(app) is None, "there is nothing safe to offer"
    assert "length cap" in app.chat.transcript()
    assert (tmp_path / "loader.c").read_text(encoding="utf-8") == "int old;\n"


def test_an_escaping_path_is_reported_not_offered(app, tmp_path):
    app.project = tmp_path

    app._check_for_edits(
        "=== FILE: ../../escaped.c ===\nint a;\n=== END FILE ===")

    assert pending_action(app) is None
    assert "outside the attached folder" in app.chat.transcript()


def test_an_edit_that_changes_nothing_is_not_offered(app, tmp_path):
    (tmp_path / "loader.c").write_text(
        "int read_source_guarded(void) { return 1; }\n", encoding="utf-8")
    app.project = tmp_path

    app._check_for_edits(REPLY_WITH_EDIT)

    assert pending_action(app) is None


def test_applying_refreshes_what_retrieval_knows(app, tmp_path):
    """The file on disk moved on, so the copy retrieval ranks over is now
    describing a version that no longer exists."""
    from aichatlab import edits as edit_tools

    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"loader.c": "int old;\n"}
    proposed, _ = edit_tools.parse(REPLY_WITH_EDIT)

    app._apply_edits(proposed, always=False)

    assert "return 1" in app.project_texts["loader.c"]


def test_an_undo_is_offered_after_writing(app, tmp_path):
    from aichatlab import edits as edit_tools

    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    proposed, _ = edit_tools.parse(REPLY_WITH_EDIT)
    app._apply_edits(proposed, always=False)

    assert pending_action(app), "there must be a way back"
    assert "puts every one of those files back" in app.chat.transcript()


def test_undoing_restores_the_file(app, tmp_path):
    from aichatlab import edits as edit_tools

    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    proposed, _ = edit_tools.parse(REPLY_WITH_EDIT)
    app._apply_edits(proposed, always=False)

    app._undo_edits(edit_tools.snapshots(tmp_path)[0], pending_action(app))

    assert (tmp_path / "loader.c").read_text(encoding="utf-8") == "int old;\n"
    assert "back as they were" in app.chat.transcript()


def test_attaching_a_single_file_makes_it_editable(app, tmp_path):
    """Dragging in three files and asking for changes used to produce prose
    — correct, and baffling."""
    source = tmp_path / "loader.c"
    source.write_text("int old;\n", encoding="utf-8")

    app.add_attachment(source)

    assert app.project == tmp_path
    assert "can be edited" in app.chat.transcript()


def test_an_attached_file_never_widens_an_existing_project(app, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "stray.c").write_text("int a;\n", encoding="utf-8")
    app.project = tmp_path

    app.add_attachment(other / "stray.c")

    assert app.project == tmp_path, "the real project keeps the boundary"


def test_a_reply_that_forgot_the_closing_marker_is_still_offered(app, tmp_path):
    """Four files were refused as "cut off" when the reply had finished fine
    — the model had simply left every closing marker off."""
    (tmp_path / "a.py").write_text("old = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("old = 2\n", encoding="utf-8")
    app.project = tmp_path

    app._check_for_edits(
        "=== FILE: a.py ===\nnew = 1\n\n=== FILE: b.py ===\nnew = 2\n",
        truncated=False)

    assert pending_action(app), "these were recoverable"
    assert "cut off" not in app.chat.transcript()


def test_a_genuinely_truncated_reply_says_what_to_do(app, tmp_path):
    (tmp_path / "a.py").write_text("old = 1\n", encoding="utf-8")
    app.project = tmp_path

    app._check_for_edits("=== FILE: a.py ===\nnew = 1", truncated=True)

    assert pending_action(app) is None
    transcript = app.chat.transcript()
    assert "length cap" in transcript
    assert "Continue" in transcript


def test_a_cramped_answer_length_is_raised_and_explained(app, tmp_path):
    app.project = tmp_path
    app.settings["speed_quality"] = 0          # Quick: the 192-token cap

    app._warn_about_edit_room()

    transcript = app.chat.transcript()
    assert "not enough for the model to finish an edit block" in transcript
    assert app.warned_edit_room


def test_that_warning_is_not_repeated(app, tmp_path):
    app.project = tmp_path
    app.settings["speed_quality"] = 0
    app._warn_about_edit_room()
    app._warn_about_edit_room()

    assert app.chat.transcript().count("finish an edit block") == 1


def test_the_orchestrator_is_told_when_editing_is_possible(app, tmp_path):
    app.project = tmp_path
    assert app._make_orchestrator().editing

    app.project = None
    assert not app._make_orchestrator().editing


# ----------------------------------- when the model says it saved and did not
NARRATED_SAVE = """\
Saving the modified file loader.c.

The loader.c file has been successfully modified to guard the read.
"""


def test_the_app_contradicts_a_save_that_never_happened(app, tmp_path):
    """The worst failure this feature has is the one that looks like success:
    a checklist ticking past "Save the modified file", a paragraph saying the
    file was successfully modified, and an untouched file on disk.  The app
    does the writing, so the app is the one thing that can say otherwise."""
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"loader.c": "int old;\n"}

    app._check_for_edits(NARRATED_SAVE)

    transcript = app.chat.transcript()
    assert "Nothing was written" in transcript
    assert "Saving the modified file loader.c" in transcript
    assert (tmp_path / "loader.c").read_text(encoding="utf-8") == "int old;\n"


def test_a_real_edit_is_not_accused_of_pretending(app, tmp_path):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"loader.c": "int old;\n"}

    app._check_for_edits("I have updated the loader.\n" + REPLY_WITH_EDIT)

    assert "Nothing was written" not in app.chat.transcript()
    assert pending_action(app)


# -- the edit-check trace ---------------------------------------------------
# The complaint this records against is always the same sentence — "it listed
# the changes and did nothing" — and these tests exist to keep the six things
# that sentence can mean from collapsing into one.
def test_a_successful_check_is_recorded_as_having_offered(app, tmp_path):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.settings["review_popup"] = False

    app._check_for_edits(REPLY_WITH_EDIT)

    trace = app.edit_debug.last()
    assert trace.verdict == "offered"
    assert trace.offered
    assert any(step.count for step in trace.steps)


def test_a_reply_with_no_blocks_is_recorded_as_prose(app, tmp_path):
    """The commonest failure, and the one that looks most like a bug."""
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"loader.c": "int old;\n"}
    app.last_edit_target = _target("alpha:1b")

    app._check_for_edits("I'll rewrite the loader to be simpler. Done!")

    trace = app.edit_debug.last()
    assert trace.verdict in ("prose-offered", "prose-unmatched")
    assert not trace.offered
    assert trace.reply_shape["file_openers"] == 0
    assert "prose only" in editdebug.shape_note(trace.reply_shape)


def test_no_model_to_go_back_to_is_told_apart_from_unrecognised_prose(
        app, tmp_path):
    """Two prose failures with two different fixes."""
    app.project = tmp_path
    app.last_edit_target = None

    app._check_for_edits("I'll rewrite the loader to be simpler.")

    assert app.edit_debug.last().verdict == "no-target"


def test_a_refused_block_is_recorded_with_its_reason(app, tmp_path):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"loader.c": "int old;\n"}

    app._check_for_edits(
        "=== EDIT: loader.c ===\n--- FIND\nint nonexistent;\n"
        "--- REPLACE\nint fresh;\n=== END EDIT ===\n")

    trace = app.edit_debug.last()
    assert trace.verdict == "all-rejected"
    assert trace.rejections
    assert trace.rejections[0]["reason"]


def test_no_folder_is_recorded_rather_than_ignored(app):
    """Silence here is what makes people think the feature is broken."""
    app.project = None

    app._check_for_edits(REPLY_WITH_EDIT)

    assert app.edit_debug.last().verdict == "no-folder"


def test_editing_switched_off_is_recorded(app, tmp_path):
    app.project = tmp_path
    app.settings["allow_edits"] = False

    app._check_for_edits(REPLY_WITH_EDIT)

    assert app.edit_debug.last().verdict == "edits-off"


def test_a_change_already_on_disk_is_told_apart_from_a_failure(app, tmp_path):
    (tmp_path / "loader.c").write_text(
        "int read_source_guarded(void) { return 1; }\n", encoding="utf-8")
    app.project = tmp_path

    app._check_for_edits(REPLY_WITH_EDIT)

    assert app.edit_debug.last().verdict == "no-change"


def test_a_model_claiming_to_have_saved_is_recorded(app, tmp_path):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"loader.c": "int old;\n"}

    app._check_for_edits(
        "I've applied the change to loader.c.\n"
        "The file has been successfully modified.")

    claim = app.edit_debug.last().claimed_to_write
    assert claim, "the trace must keep the model's own words"
    assert "claimed to have written" in app.edit_debug.last().explain()


def test_the_trace_survives_across_checks(app, tmp_path):
    app.project = tmp_path
    app.settings["review_popup"] = False

    app._check_for_edits("just talking")
    app._check_for_edits("still just talking")

    assert len(app.edit_debug) == 2
    assert "2 checks" in app.edit_debug.summary()
    assert "0 reached a diff" in app.edit_debug.summary()


# -- the change that reads fine and does not run ---------------------------
# The real one: an anchored patch added `nocat += 1` with no `nocat = 0`. The
# diff was five plausible lines, it was approved, and it broke the app.
BREAKING_EDIT = """\
=== EDIT: counter.py ===
--- FIND
    for row in rows:
        if row:
            n += 1
--- REPLACE
    for row in rows:
        if row:
            n += 1
        else:
            skipped += 1
=== END EDIT ===
"""

WORKING_SOURCE = """\
def tally(rows):
    n = 0
    for row in rows:
        if row:
            n += 1
    return n
"""


def _breakable(app, tmp_path):
    (tmp_path / "counter.py").write_text(WORKING_SOURCE, encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"counter.py": WORKING_SOURCE}
    app.sent_files = {"counter.py"}


def test_a_change_that_breaks_the_file_is_flagged(app, tmp_path, monkeypatch):
    _breakable(app, tmp_path)
    app.settings["review_popup"] = False

    app._check_for_edits(BREAKING_EDIT)

    trace = app.edit_debug.last()
    assert trace.verdict == "offered", "it is still offered, not refused"
    assert trace.flaws, "but the trace must record what it would break"
    assert "skipped" in trace.flaws[0]["problem"]
    assert "UnboundLocalError" in trace.flaws[0]["problem"]


def test_the_warning_reaches_the_transcript_row(app, tmp_path):
    _breakable(app, tmp_path)
    app.settings["review_popup"] = False

    app._check_for_edits(BREAKING_EDIT)

    assert "UnboundLocalError" in app.chat.transcript()


def test_a_breaking_change_arrives_unticked(app, tmp_path, monkeypatch):
    from aichatlab.ui.edit_review import EditReview

    _breakable(app, tmp_path)
    app.settings["review_popup"] = False
    app._check_for_edits(BREAKING_EDIT)
    action_id = next(iter(app.pending_edits))

    window = EditReview(app.root, tmp_path, app.pending_edits[action_id],
                        flaws=app.pending_flaws[action_id])
    try:
        assert not window.picks["counter.py"].get(), "must not be pre-ticked"
        assert window._runnable() == [], "and Apply all must skip it"
    finally:
        window.destroy()


def test_a_healthy_change_is_still_ticked_and_swept_up(app, tmp_path):
    from aichatlab.ui.edit_review import EditReview

    _breakable(app, tmp_path)
    app.settings["review_popup"] = False
    app._check_for_edits(
        "=== EDIT: counter.py ===\n--- FIND\n    n = 0\n--- REPLACE\n"
        "    n = 0\n    skipped = 0\n=== END EDIT ===\n")

    assert not app.edit_debug.last().flaws
    action_id = next(iter(app.pending_edits))
    window = EditReview(app.root, tmp_path, app.pending_edits[action_id],
                        flaws=app.pending_flaws[action_id])
    try:
        assert window.picks["counter.py"].get()
        assert len(window._runnable()) == 1
    finally:
        window.destroy()


def test_automatic_applying_pauses_for_a_change_that_would_break(app, tmp_path):
    """"Stop asking me" was never permission to write something broken."""
    _breakable(app, tmp_path)
    app.auto_apply_edits = True
    app.settings["review_popup"] = False

    app._check_for_edits(BREAKING_EDIT)

    assert app.edit_debug.last().verdict == "offered"
    assert app.pending_edits, "it must stop and ask instead"
    assert (tmp_path / "counter.py").read_text(
        encoding="utf-8") == WORKING_SOURCE
    assert "paused" in app.chat.transcript()


def test_automatic_applying_still_works_when_nothing_breaks(app, tmp_path):
    _breakable(app, tmp_path)
    app.auto_apply_edits = True

    app._check_for_edits(
        "=== EDIT: counter.py ===\n--- FIND\n    n = 0\n--- REPLACE\n"
        "    n = 0\n    skipped = 0\n=== END EDIT ===\n")

    assert app.edit_debug.last().verdict == "auto-applied"
    assert "skipped = 0" in (tmp_path / "counter.py").read_text(
        encoding="utf-8")


def test_a_markdown_file_is_not_second_guessed(app, tmp_path):
    """The checker only speaks about languages it can actually read."""
    (tmp_path / "NOTES.md").write_text("# Notes\n", encoding="utf-8")
    app.project = tmp_path
    app.settings["review_popup"] = False

    app._check_for_edits(
        "=== FILE: NOTES.md ===\n# Notes\n\nSomething (((\n=== END FILE ===\n")

    assert not app.edit_debug.last().flaws


# -- the review window opening by itself -----------------------------------
def test_the_review_window_opens_without_a_click(app, tmp_path, monkeypatch):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.settings["review_popup"] = True
    opened = []
    monkeypatch.setattr("aichatlab.ui.app.EditReview",
                        lambda *a, **k: opened.append(a))

    app._check_for_edits(REPLY_WITH_EDIT)

    assert opened, "the window must appear on its own"


def test_the_row_stays_until_the_window_is_answered(app, tmp_path, monkeypatch):
    """An unrequested window is one people dismiss without reading."""
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.settings["review_popup"] = True
    monkeypatch.setattr("aichatlab.ui.app.EditReview", lambda *a, **k: None)

    app._check_for_edits(REPLY_WITH_EDIT)

    assert pending_action(app), "the changes must still be reachable"
    assert app.pending_edits


def test_declining_the_automatic_window_keeps_the_changes(app, tmp_path,
                                                          monkeypatch):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.settings["review_popup"] = True
    captured = {}

    def fake_review(_master, _root, proposed, **kwargs):
        captured["on_apply"] = kwargs["on_apply"]

    monkeypatch.setattr("aichatlab.ui.app.EditReview", fake_review)
    app._check_for_edits(REPLY_WITH_EDIT)

    captured["on_apply"]([], False)          # Escape / Cancel

    assert app.pending_edits, "cancelling is not discarding"
    assert (tmp_path / "loader.c").read_text(encoding="utf-8") == "int old;\n"


def test_applying_from_the_automatic_window_spends_the_row(app, tmp_path,
                                                           monkeypatch):
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.settings["review_popup"] = True
    captured = {}

    def fake_review(_master, _root, proposed, **kwargs):
        captured["on_apply"] = kwargs["on_apply"]
        captured["proposed"] = proposed

    monkeypatch.setattr("aichatlab.ui.app.EditReview", fake_review)
    app._check_for_edits(REPLY_WITH_EDIT)

    captured["on_apply"](captured["proposed"], False)

    assert not app.pending_edits
    assert "read_source_guarded" in (tmp_path / "loader.c").read_text(
        encoding="utf-8")


def test_a_refusal_is_the_more_useful_thing_to_read(app, tmp_path):
    """When a block was there and was refused, the reason it was refused
    beats a general note about there being no block."""
    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"loader.c": "int old;\n"}

    app._check_for_edits(
        "I have updated the loader.\n\n"
        "=== EDIT: loader.c ===\n--- FIND\nint nonexistent;\n"
        "--- REPLACE\nint new;\n=== END EDIT ===\n")

    transcript = app.chat.transcript()
    assert "Nothing was written —" not in transcript
    assert "loader.c" in transcript


def test_ordinary_conversation_collects_no_edit_warning(app, tmp_path):
    app.project = tmp_path
    app.project_texts = {"loader.c": "int old;\n"}

    app._check_for_edits("Here is how I would restructure the loader.")

    assert "Nothing was written" not in app.chat.transcript()


# ------------------- describing a change, approving it, then getting the diff
DESCRIBED = """These changes have been applied to the specified files.

1. **Guard the read** in `loader.c` by checking the handle before use.
2. **Rebuild the table** in `symbols.c` after every load.
"""


def _wire_project(app, tmp_path):
    from aichatlab.orchestrator import Target

    (tmp_path / "loader.c").write_text("int old;\n", encoding="utf-8")
    (tmp_path / "symbols.c").write_text("int table;\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"loader.c": "int old;\n", "symbols.c": "int table;\n"}
    return Target("local", "alpha:1b")


def test_a_described_change_becomes_an_offer_rather_than_a_dead_end(app, tmp_path):
    """The prose was written with the files in view. It is the first half of
    an edit, not a failed one."""
    target = _wire_project(app, tmp_path)

    app._check_for_edits(DESCRIBED, target=target)

    assert pending_action(app)
    assert "Make these changes" in action_buttons(app)
    assert "2 changes across 2 files" in app.chat.transcript()


def test_approving_the_description_still_writes_nothing_by_itself(app, tmp_path):
    """The approval here is of the intent. A description is not a diff, and
    agreeing to one is not agreeing to the other — the review window still
    stands between the model and the filesystem."""
    target = _wire_project(app, tmp_path)
    app._check_for_edits(DESCRIBED, target=target)
    asked = []
    app._run = lambda job: asked.append(job)

    app._realise_changes(pending_action(app))

    assert asked, "it must go and ask for the real edits"
    assert (tmp_path / "loader.c").read_text(encoding="utf-8") == "int old;\n"
    assert (tmp_path / "symbols.c").read_text(encoding="utf-8") == "int table;\n"


def test_declining_leaves_everything_alone(app, tmp_path):
    target = _wire_project(app, tmp_path)
    app._check_for_edits(DESCRIBED, target=target)

    app._drop_changes(pending_action(app))

    assert not app.pending_changes
    assert "nothing was written" in app.chat.transcript()
    assert (tmp_path / "loader.c").read_text(encoding="utf-8") == "int old;\n"


def test_what_comes_back_from_the_second_pass_lands_in_the_review(app, tmp_path):
    _wire_project(app, tmp_path)

    app._handle("changes_ready", {
        "reply": "=== FILE: loader.c ===\nint guarded;\n=== END FILE ===",
        "target": _target("alpha:1b")})

    assert pending_action(app)
    assert "Nothing has been written yet" in app.chat.transcript()
    assert (tmp_path / "loader.c").read_text(encoding="utf-8") == "int old;\n"


def test_the_second_pass_does_not_offer_to_go_round_again(app, tmp_path):
    """Otherwise a model that answers prose with prose gives you a loop with
    a button on it."""
    target = _wire_project(app, tmp_path)

    app._check_for_edits(DESCRIBED, target=target, second_pass=True)

    assert not pending_action(app)
    assert not app.pending_changes


def test_a_real_edit_block_skips_the_middleman(app, tmp_path):
    """When the model does produce a block, the old path is still the fast
    one — describe-then-fetch is the fallback, not a toll booth."""
    target = _wire_project(app, tmp_path)

    app._check_for_edits(REPLY_WITH_EDIT, target=target)

    assert "Proposed changes" in app.chat.transcript()
    assert "Make these changes" not in action_buttons(app)


def test_a_new_chat_forgets_changes_nobody_answered(app, tmp_path):
    target = _wire_project(app, tmp_path)
    app._check_for_edits(DESCRIBED, target=target)
    assert app.pending_changes

    app.new_chat()

    assert not app.pending_changes
    assert app.last_edit_target is None


def test_a_refused_edit_is_offered_a_retry_even_when_another_one_landed(app, tmp_path):
    """One lucky block — often a brand-new file, the easiest kind to produce
    and the least likely to be wanted — used to silently cancel the retry for
    every edit that failed beside it."""
    target = _wire_project(app, tmp_path)

    app._check_for_edits(
        "=== FILE: brand_new.py ===\nnew = 1\n=== END FILE ===\n\n"
        "=== EDIT: loader.c ===\n--- FIND\nint nonexistent;\n"
        "--- REPLACE\nint fixed;\n=== END EDIT ===\n",
        target=target)

    assert any(action_id.startswith("retry")
               for action_id in app.pending_changes), app.pending_changes
    assert "Try again with the real file" in action_buttons(app)


def test_the_retry_hands_back_the_file_rather_than_reporting_the_problem(app, tmp_path):
    target = _wire_project(app, tmp_path)
    app._check_for_edits(
        "=== EDIT: loader.c ===\n--- FIND\nint nonexistent;\n"
        "--- REPLACE\nint fixed;\n=== END EDIT ===\n", target=target)
    action_id = next(a for a in app.pending_changes if a.startswith("retry"))
    changes, _target = app.pending_changes[action_id]

    assert [c.file for c in changes] == ["loader.c"]
    prompt = intent_module.build_change_prompt(
        changes[0], app.project_texts["loader.c"], "rules")
    assert "int old;" in prompt


def test_a_refusal_about_a_file_nobody_loaded_is_not_offered_a_retry(app, tmp_path):
    """Without the file there is nothing to hand back, so the retry would be
    the same question that just failed."""
    target = _wire_project(app, tmp_path)

    app._check_for_edits(
        "=== EDIT: never_loaded.c ===\n--- FIND\nint x;\n"
        "--- REPLACE\nint y;\n=== END EDIT ===\n", target=target)

    assert not any(a.startswith("retry") for a in app.pending_changes)


# --------------------- files that exist on disk but never fit the context
#
# The screenshot version of this: Explorer showing replypilot_classify_engine.py
# in the Replyit folder, next to the app insisting "there is no
# replypilot_classify_engine.py in the attached folder."  Both were right about
# different questions — it was on the disk and not in the prompt — and the app
# reported the prompt as if it were the disk.

def test_an_edit_to_an_unloaded_file_is_not_told_the_file_is_missing(app, tmp_path):
    target = _wire_project(app, tmp_path)
    (tmp_path / "unloaded_engine.py").write_text("threshold = 0.85\n",
                                                 encoding="utf-8")
    app.project_files = ["loader.c", "symbols.c", "unloaded_engine.py"]

    app._check_for_edits(
        "=== EDIT: unloaded_engine.py ===\n--- FIND\nthreshold = 0.99\n"
        "--- REPLACE\nthreshold = 0.90\n=== END EDIT ===\n", target=target)

    transcript = app.chat.transcript()
    assert "there is no unloaded_engine.py" not in transcript
    assert "did not fit the context budget" in transcript


def test_the_unloaded_file_gets_a_retry_that_reads_the_disk(app, tmp_path):
    """The file that never fit the budget is the file most in need of a
    retry — the model was working from its name alone."""
    target = _wire_project(app, tmp_path)
    (tmp_path / "unloaded_engine.py").write_text("threshold = 0.85\n",
                                                 encoding="utf-8")
    app.project_files = ["loader.c", "symbols.c", "unloaded_engine.py"]

    app._check_for_edits(
        "=== EDIT: unloaded_engine.py ===\n--- FIND\nthreshold = 0.99\n"
        "--- REPLACE\nthreshold = 0.90\n=== END EDIT ===\n", target=target)

    action_id = next(a for a in app.pending_changes if a.startswith("retry"))
    changes, _t = app.pending_changes[action_id]
    assert [c.file for c in changes] == ["unloaded_engine.py"]
    # and the second pass must hand back the real contents, from disk
    from aichatlab import edits as edit_tools
    current = app.project_texts.get(changes[0].file) or edit_tools.read_current(
        app.project, changes[0].file)
    assert current == "threshold = 0.85\n"


def test_a_correct_anchor_against_an_unloaded_file_simply_works(app, tmp_path):
    """A model that happens to quote the file correctly — say, from a retry —
    must not be blocked just because the first pass never loaded it."""
    target = _wire_project(app, tmp_path)
    (tmp_path / "unloaded_engine.py").write_text("threshold = 0.85\n",
                                                 encoding="utf-8")
    app.project_files = ["loader.c", "symbols.c", "unloaded_engine.py"]

    app._check_for_edits(
        "=== EDIT: unloaded_engine.py ===\n--- FIND\nthreshold = 0.85\n"
        "--- REPLACE\nthreshold = 0.90\n=== END EDIT ===\n", target=target)

    assert pending_action(app), "the review must be offered"
    assert (tmp_path / "unloaded_engine.py").read_text(
        encoding="utf-8") == "threshold = 0.85\n", "nothing written before approval"


def test_described_changes_to_unloaded_files_are_still_offered(app, tmp_path):
    target = _wire_project(app, tmp_path)
    (tmp_path / "unloaded_engine.py").write_text("threshold = 0.85\n",
                                                 encoding="utf-8")
    app.project_files = ["loader.c", "symbols.c", "unloaded_engine.py"]

    app._check_for_edits(
        "1. Raise the threshold in `unloaded_engine.py` from 0.85 to 0.90.",
        target=target)

    assert any(a.startswith("intent") for a in app.pending_changes)


# ------------------------ "if I'm giving permission to the whole folder,
#                           it should be 14/14"
#
# It should, and now it is.  The token budget decides which files travel in
# one prompt; it does not decide which files the app is allowed to know
# about.  These pin the seam between the two.

def _fourteen_file_selection(tmp_path):
    from aichatlab.folderscan import Limits, Survey
    from aichatlab.folderscan import select as fs_select

    result = Survey(root=tmp_path)
    for n in range(14):
        (tmp_path / f"engine{n:02d}.py").write_text(
            f"value_{n} = {n}\n" * 40, encoding="utf-8")
        result.entries.append(__import__("aichatlab.folderscan",
            fromlist=["FileEntry"]).FileEntry(
            path=tmp_path / f"engine{n:02d}.py",
            relative=f"engine{n:02d}.py",
            size=(tmp_path / f"engine{n:02d}.py").stat().st_size,
            kind="code", depth=0))
    selection = fs_select(result, Limits(max_tokens=1_000))
    assert 0 < len(selection.chosen) < 14, "the budget must bite"
    return selection


def test_the_whole_folder_ends_up_on_hand(app, tmp_path):
    from aichatlab.folderscan import read_texts, worth_holding

    selection = _fourteen_file_selection(tmp_path)
    app.project = tmp_path
    app.project_files = [e.relative for e in selection.readable]
    texts = read_texts(worth_holding(selection, 20_000_000))

    app._folder_read(tmp_path, "block", selection, texts)

    assert len(app.project_texts) == 14
    assert "all 14 files read and kept on hand" in app.chat.transcript()


def test_what_the_model_saw_is_tracked_separately_from_what_the_app_holds(app, tmp_path):
    from aichatlab.folderscan import read_texts, worth_holding

    selection = _fourteen_file_selection(tmp_path)
    app.project = tmp_path
    app.project_files = [e.relative for e in selection.readable]
    app.settings["folder_retrieval"] = False       # force the pinned path
    texts = read_texts(worth_holding(selection, 20_000_000))

    app._folder_read(tmp_path, "block", selection, texts)

    assert app.sent_files == {e.relative for e in selection.chosen}
    assert len(app.project_texts) == 14


# -- nowhere for the change to go ------------------------------------------
# Driven directly rather than through send(), which opens a modal warning
# when no models are ticked and would block a headless run forever.
def test_asking_to_edit_with_no_folder_says_so(app):
    """The failure that reads as the app refusing to do the work."""
    app.project = None

    app._warn_if_nothing_to_edit("detect deficiencies and upgrade my app")

    transcript = app.chat.transcript()
    assert "No folder is attached" in transcript
    assert "Attach folder" in transcript


def test_the_no_folder_warning_is_said_once_not_every_message(app):
    app.project = None

    app._warn_if_nothing_to_edit("please upgrade my app")
    app._warn_if_nothing_to_edit("upgrade the code again")

    assert app.chat.transcript().count("No folder is attached") == 1


def test_an_ordinary_question_gets_no_warning(app):
    app.project = None

    app._warn_if_nothing_to_edit("what is a decorator in python")

    assert "No folder is attached" not in app.chat.transcript()


def test_with_a_folder_attached_there_is_nothing_to_warn_about(app, tmp_path):
    app.project = tmp_path

    app._warn_if_nothing_to_edit("upgrade my app")

    assert "No folder is attached" not in app.chat.transcript()


def test_a_new_chat_lets_the_warning_be_said_again(app):
    app.project = None
    app._warn_if_nothing_to_edit("upgrade my app")

    app.new_chat()
    app.project = None
    app._warn_if_nothing_to_edit("upgrade my app")

    assert "No folder is attached" in app.chat.transcript()


# -- one change at a time --------------------------------------------------
# Fetching every change first meant minutes of waiting per change before
# anything appeared, then a pile of diffs to judge together — and paying for
# all of them even when the first showed the run was going nowhere.
def _queued(app, tmp_path, count=3):
    from aichatlab.intent import Change
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    app.project = tmp_path
    app.change_target = _target("alpha:1b")
    app.change_queue = [Change(description=f"change {n}", file="a.py")
                        for n in range(count)]


def test_only_one_change_is_asked_for_at_a_time(app, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(app, "_run", lambda job: asked.append(job))
    _queued(app, tmp_path, count=3)

    app._next_change()

    assert len(asked) == 1, "one request, not three"
    assert len(app.change_queue) == 2, "the rest wait their turn"


def test_the_note_says_how_many_are_left(app, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_run", lambda job: None)
    _queued(app, tmp_path, count=3)

    app._next_change()

    assert "2 more after this one" in app.chat.transcript()


def test_the_last_one_says_so(app, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_run", lambda job: None)
    _queued(app, tmp_path, count=1)

    app._next_change()

    assert "the last one" in app.chat.transcript()


def test_deciding_one_asks_for_the_next(app, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(app, "_next_change", lambda: started.append(1))
    monkeypatch.setattr(app.root, "after",
                        lambda _ms, fn=None: fn() if fn else None)
    _queued(app, tmp_path, count=2)

    app._advance_change_queue()

    assert started, "the next one is requested once this is decided"


def test_an_empty_queue_advances_to_nothing(app, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(app, "_next_change", lambda: started.append(1))
    app.change_queue = []

    app._advance_change_queue()

    assert not started


def test_stop_abandons_the_rest_even_with_no_worker_running(app, tmp_path):
    """The queue is paused on a decision, so nothing is running to cancel."""
    _queued(app, tmp_path, count=3)
    app.worker = None

    app.stop()

    assert app.change_queue == []
    assert "3 change(s) were not asked for" in app.chat.transcript()


def test_a_new_chat_forgets_the_queue(app, tmp_path):
    _queued(app, tmp_path, count=2)

    app.new_chat()

    assert app.change_queue == []
    assert app.change_target is None


def test_a_second_pass_that_produced_nothing_still_advances(app, tmp_path,
                                                            monkeypatch):
    """Otherwise a request that quietly failed strands every change behind it."""
    moved = []
    monkeypatch.setattr(app, "_advance_change_queue", lambda: moved.append(1))
    app.project = tmp_path
    app.last_edit_target = _target("alpha:1b")

    app._check_for_edits("just talking, no blocks here", second_pass=True)

    assert moved, "the queue must not stall on a failed change"


# -- the menu: a broad wish becomes a pick-one list -------------------------
MENU_REPLY = """\
=== SUGGESTIONS ===
1. [counter.py] Add a guard to tally() for an empty rows list
2. [counter.py] Add a docstring to tally() describing the return value
3. [other.py] Early return in load() when the path is missing
=== END SUGGESTIONS ===
"""


def _menu_project(app, tmp_path):
    (tmp_path / "counter.py").write_text(
        "def tally(rows):\n    return len(rows)\n", encoding="utf-8")
    (tmp_path / "other.py").write_text(
        "def load(path):\n    return path\n", encoding="utf-8")
    app.project = tmp_path
    app.project_files = ["counter.py", "other.py"]
    app.project_texts = {
        "counter.py": (tmp_path / "counter.py").read_text(encoding="utf-8"),
        "other.py": (tmp_path / "other.py").read_text(encoding="utf-8")}
    app.last_edit_target = _target("alpha:1b")


def test_a_menu_reply_becomes_a_pick_one_offer(app, tmp_path):
    _menu_project(app, tmp_path)

    app._check_for_edits(MENU_REPLY)

    assert app.edit_debug.last().verdict == "menu-offered"
    assert len(app.menu_left) == 3
    assert "pick one" in app.chat.transcript()


def test_an_invented_function_is_dropped_and_said(app, tmp_path):
    _menu_project(app, tmp_path)
    reply = MENU_REPLY.replace("load()", "imaginary()")

    app._check_for_edits(reply)

    assert len(app.menu_left) == 2
    assert "imaginary() does not exist" in app.chat.transcript()


def test_picking_one_queues_exactly_that_change(app, tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(app, "_run", lambda job: fetched.append(job))
    _menu_project(app, tmp_path)
    app._check_for_edits(MENU_REPLY)

    app._pick_suggestion(app.menu_action_id, 1)

    assert len(fetched) == 1
    assert len(app.menu_left) == 2, "the picked one leaves the menu"
    assert "docstring" not in " ".join(
        c.description for c in app.menu_left)


def test_after_a_decision_the_rest_of_the_menu_comes_back(app, tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(app, "_run", lambda job: None)
    _menu_project(app, tmp_path)
    app._check_for_edits(MENU_REPLY)
    app._pick_suggestion(app.menu_action_id, 0)

    app._advance_change_queue()

    assert "another from the list" in app.chat.transcript()
    assert len(app.menu_left) == 2


def test_none_of_these_closes_the_menu(app, tmp_path):
    _menu_project(app, tmp_path)
    app._check_for_edits(MENU_REPLY)

    app._drop_menu(app.menu_action_id)

    assert app.menu_left == []


def test_a_broad_ask_takes_the_menu_path(app, tmp_path, monkeypatch):
    """It becomes a scoped request for candidates, not a conversation turn."""
    _menu_project(app, tmp_path)
    ran = []
    monkeypatch.setattr(app, "_run", lambda job: ran.append(job))
    monkeypatch.setattr(app, "_input_text",
                        lambda: "I want performance enhancements in this app")
    monkeypatch.setattr(app, "selected_targets",
                        lambda: [_target("alpha:1b")])

    app.send()

    assert len(ran) == 1
    assert "broad one" in app.chat.transcript()
    assert app.last_edit_target is not None


def test_a_specific_ask_skips_the_menu_path(app, tmp_path, monkeypatch):
    _menu_project(app, tmp_path)
    ran = []
    monkeypatch.setattr(app, "_run", lambda job: ran.append(job))
    monkeypatch.setattr(app, "_input_text",
                        lambda: "add a docstring to tally() in counter.py")
    monkeypatch.setattr(app, "selected_targets",
                        lambda: [_target("alpha:1b")])

    app.send()

    assert "broad one" not in app.chat.transcript()


def test_a_failed_menu_does_not_become_a_broad_second_pass(app, tmp_path):
    """A model that answers the menu request with an essay must not have the
    broad wish auto-fed into the scoped path — that is the shape that breaks,
    and the menu exists so it never runs."""
    _menu_project(app, tmp_path)
    app.last_question = "I want performance enhancements in this app"

    app._check_for_edits("Here are my thoughts on performance in general…")

    assert app.change_queue == []
    assert app.edit_debug.last().verdict == "prose-unmatched"


def test_stop_clears_the_menu_too(app, tmp_path):
    _menu_project(app, tmp_path)
    app._check_for_edits(MENU_REPLY)
    app.worker = None

    app.stop()

    assert app.menu_left == []


# -- a message that names models is not a broad wish ------------------------
def test_naming_models_bypasses_the_menu(app, tmp_path, monkeypatch):
    """"14B suggest the list, THEN the 32B coder review it" was hijacked
    into a single-model menu: the first model got a pick-list, the second
    was never called, and the written-out handoff was thrown away."""
    from aichatlab.ui import app as app_module

    _menu_project(app, tmp_path)
    app.model_vars = {("host", "qwen2.5:14b-instruct-q8_0"): _FakeVar(True),
                      ("host", "qwen2.5-coder:32b"): _FakeVar(True)}
    monkeypatch.setattr(app, "_run", lambda job: None)
    monkeypatch.setattr(app_module.messagebox, "showwarning",
                        lambda *a, **k: None)
    monkeypatch.setattr(app, "_input_text", lambda: (
        "QWEN2.5 14B PLEASE LOOK THROUGH REPLYIT AND SUGGEST WHAT NEW "
        "PERFORMANCE ENHANCEMENTS WE CAN ADD. THEN I WANT QWEN2.5 32B "
        "CODER TO OVER LOOK THE LIST"))

    app.send()

    assert "broad one" not in app.chat.transcript(), \
        "the menu must not hijack a two-model instruction"


def test_a_broad_ask_without_model_names_still_menus(app, tmp_path,
                                                     monkeypatch):
    _menu_project(app, tmp_path)
    app.model_vars = {("host", "qwen2.5:14b-instruct-q8_0"): _FakeVar(True)}
    monkeypatch.setattr(app, "_run", lambda job: None)
    monkeypatch.setattr(app, "_input_text",
                        lambda: "I want performance enhancements in this app")

    app.send()

    assert "broad one" in app.chat.transcript()


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


# -- critique chain: roles come from tick order, and the screen says so -----
def _tick(app, server, model, on=True):
    app.model_vars[(server, model)] = _FakeVar(on)
    app._on_model_tick((server, model))


def test_roles_follow_tick_order_not_list_order(app):
    app.model_vars = {}
    app.mode.set("Critique chain — draft, critique, revise")
    # ticked in the *reverse* of list order
    _tick(app, "host", "zeta:1b")
    _tick(app, "host", "alpha:1b")

    ordered = app._ordered_targets(app.selected_targets())

    assert [t.model for t in ordered] == ["zeta:1b", "alpha:1b"]


def test_the_hint_walks_through_the_roles(app):
    app.model_vars = {}
    app.mode.set("Critique chain — draft, critique, revise")
    app._update_chain_hint()
    assert "drafter" in app.chain_hint.cget("text")

    _tick(app, "host", "alpha:1b")
    assert "pick the reviewer" in app.chain_hint.cget("text")

    _tick(app, "host", "beta:2b")
    hint = app.chain_hint.cget("text").lower()
    assert hint.startswith("ready:")
    assert hint.index("alpha") < hint.index("beta"), "drafter named first"


def test_a_third_model_is_the_rewriter_and_a_fourth_is_called_idle(app):
    app.model_vars = {}
    app.mode.set("Critique chain — draft, critique, revise")
    for model in ("a:1b", "b:1b", "c:1b"):
        _tick(app, "host", model)
    assert "rewrites" in app.chain_hint.cget("text")

    _tick(app, "host", "d:1b")
    assert "idle" in app.chain_hint.cget("text")


def test_unticking_forgets_the_order(app):
    app.model_vars = {}
    app.mode.set("Critique chain — draft, critique, revise")
    _tick(app, "host", "alpha:1b")
    _tick(app, "host", "beta:2b")
    _tick(app, "host", "alpha:1b", on=False)
    _tick(app, "host", "alpha:1b")      # re-ticked: now last, not first

    ordered = app._ordered_targets(app.selected_targets())

    assert [t.model for t in ordered] == ["beta:2b", "alpha:1b"]


def test_entering_critique_with_models_ticked_offers_a_clean_slate(
        app, monkeypatch):
    from aichatlab.ui import app as app_module

    app.model_vars = {("host", "a:1b"): _FakeVar(True),
                      ("host", "b:1b"): _FakeVar(True)}
    asked = []
    monkeypatch.setattr(app_module.messagebox, "askyesno",
                        lambda *a, **k: asked.append(a) or True)
    app._last_mode = "chat"
    app.mode.set("Critique chain — draft, critique, revise")

    app._sync_mode_bar()

    assert asked, "the offer must be made"
    assert not app.selected_targets(), "yes unchecks everything"
    assert app.model_tick_order == []


def test_the_offer_is_not_repeated_while_staying_in_critique(app, monkeypatch):
    from aichatlab.ui import app as app_module

    app.model_vars = {("host", "a:1b"): _FakeVar(True)}
    asked = []
    monkeypatch.setattr(app_module.messagebox, "askyesno",
                        lambda *a, **k: asked.append(a) or False)
    app._last_mode = "chat"
    app.mode.set("Critique chain — draft, critique, revise")
    app._sync_mode_bar()
    app._sync_mode_bar()        # re-sync without leaving the mode

    assert len(asked) == 1


def test_the_hint_clears_outside_critique(app):
    app.chain_hint.config(text="Ready: something")
    app.mode.set("Chat — send to every selected model")

    app._update_chain_hint()

    assert app.chain_hint.cget("text") == ""


# -- tests run after an apply, and red rolls it back ------------------------
def _tested_project(app, tmp_path):
    """A real tiny project whose self-test fails if engine.py is broken."""
    (tmp_path / "engine.py").write_text(
        "def tally(rows):\n    return len(rows)\n", encoding="utf-8")
    (tmp_path / "selftest.py").write_text(
        "from engine import tally\n"
        "assert tally([1, 2]) == 2, 'tally broke'\n"
        "print('selftest ok')\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {
        "engine.py": (tmp_path / "engine.py").read_text(encoding="utf-8")}


def _wait_tests_done(app, seconds=30):
    import time as _t
    deadline = _t.time() + seconds
    while _t.time() < deadline:
        _drain(app)
        t = app.chat.transcript()
        # The apply note itself contains a ✅ ("Wrote 1 file"), so the wait
        # keys on the runner's own phrasings, not on the emoji.
        if ("the change holds" in t or "What the tests said" in t
                or "No tests found" in t or "no tests were run" in t):
            return
        _t.sleep(0.3)


def test_a_green_suite_keeps_the_change_and_says_so(app, tmp_path):
    _tested_project(app, tmp_path)
    good = edits.Edit("engine.py",
                      "def tally(rows):\n    return len(rows or [])\n")

    app._apply_edits([good], always=False)
    _wait_tests_done(app)

    transcript = app.chat.transcript()
    assert "the change holds" in transcript and "passed" in transcript
    assert "len(rows or [])" in (tmp_path / "engine.py").read_text(
        encoding="utf-8"), "green means the change stays"


def test_a_red_suite_rolls_the_apply_back(app, tmp_path):
    """The whole point: running the code is the judge a diff cannot fool."""
    _tested_project(app, tmp_path)
    before = (tmp_path / "engine.py").read_text(encoding="utf-8")
    bad = edits.Edit("engine.py",
                     "def tally(rows):\n    return 7\n")   # parses fine

    app._apply_edits([bad], always=False)
    _wait_tests_done(app)

    transcript = app.chat.transcript()
    assert "❌" in transcript and "rolled back" in transcript
    assert "tally broke" in transcript, "the failure itself is shown"
    assert (tmp_path / "engine.py").read_text(
        encoding="utf-8") == before, "the file is exactly as it was"
    assert app.project_texts["engine.py"] == before


def test_a_project_with_no_tests_says_so_honestly(app, tmp_path):
    (tmp_path / "engine.py").write_text("x = 1\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"engine.py": "x = 1\n"}

    app._apply_edits([edits.Edit("engine.py", "x = 2\n")], always=False)
    _wait_tests_done(app)

    assert "No tests found" in app.chat.transcript()
    assert (tmp_path / "engine.py").read_text(encoding="utf-8") == "x = 2\n"


def test_the_runner_can_be_switched_off(app, tmp_path):
    _tested_project(app, tmp_path)
    app.settings["test_after_apply"] = False

    app._apply_edits([edits.Edit("engine.py", "def tally(rows):\n"
                                              "    return 7\n")],
                     always=False)
    _wait_tests_done(app, seconds=3)

    assert "🧪" not in app.chat.transcript()


# -- the import sandbox, in the pipeline ------------------------------------
def test_a_module_level_crash_arrives_flagged_not_clean(app, tmp_path):
    """Fatal, outside any function, invisible to every reading-based check —
    and now caught before the diff is even shown."""
    (tmp_path / "engine.py").write_text("X = 1\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"engine.py": "X = 1\n"}
    app.sent_files = {"engine.py"}
    app.settings["review_popup"] = False

    app._check_for_edits(
        "=== EDIT: engine.py ===\n--- FIND\nX = 1\n--- REPLACE\n"
        "X = compute_limit()\n=== END EDIT ===\n")

    trace = app.edit_debug.last()
    assert trace.verdict == "offered"
    assert trace.flaws and "compute_limit" in trace.flaws[0]["problem"]
    assert "importing the edited" in app.chat.transcript()


def test_a_clean_edit_is_not_slowed_into_a_flag(app, tmp_path):
    (tmp_path / "engine.py").write_text("X = 1\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"engine.py": "X = 1\n"}
    app.sent_files = {"engine.py"}
    app.settings["review_popup"] = False

    app._check_for_edits(
        "=== EDIT: engine.py ===\n--- FIND\nX = 1\n--- REPLACE\n"
        "X = 2\n=== END EDIT ===\n")

    assert not app.edit_debug.last().flaws


def test_the_sandbox_can_be_switched_off(app, tmp_path):
    (tmp_path / "engine.py").write_text("X = 1\n", encoding="utf-8")
    app.project = tmp_path
    app.project_texts = {"engine.py": "X = 1\n"}
    app.sent_files = {"engine.py"}
    app.settings["review_popup"] = False
    app.settings["sandbox_imports"] = False

    app._check_for_edits(
        "=== EDIT: engine.py ===\n--- FIND\nX = 1\n--- REPLACE\n"
        "X = compute_limit()\n=== END EDIT ===\n")

    assert not app.edit_debug.last().flaws
