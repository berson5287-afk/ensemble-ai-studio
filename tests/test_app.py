"""App-level wiring: long-lived objects, prompt assembly and exit behaviour.

Needs a display, so it skips itself on headless machines (CI uses xvfb).
"""

from __future__ import annotations

import json

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A real ChatLabApp with no servers configured, so nothing hits the network."""
    from aichatlab import cache as cache_module
    from aichatlab import config as config_module
    from aichatlab import knowledge as knowledge_module

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"local_ip": "", "host_ip": ""}),
                             encoding="utf-8")
    monkeypatch.setattr(config_module, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(cache_module, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(knowledge_module, "KNOWLEDGE_PATH", tmp_path / "know.json")

    from aichatlab.ui.app import ChatLabApp

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    instance = ChatLabApp(root)
    yield instance
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
