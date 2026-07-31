"""History, context budgeting and save/load round-trips."""

from __future__ import annotations

import json

import pytest

from aichatlab.session import Session, estimate_tokens, make_key, split_key


def test_split_key_handles_v1_files():
    assert split_key("host::llama3:8b") == ("host", "llama3:8b")
    assert split_key("local::gemma2:9b") == ("local", "gemma2:9b")
    # v1 saved bare model names, and model names contain colons
    assert split_key("llama3:8b") == ("local", "llama3:8b")


def test_add_and_history():
    session = Session()
    key = make_key("local", "m:1b")
    session.add(key, "user", "hello")
    session.add(key, "assistant", "hi")

    assert [m["role"] for m in session.history(key)] == ["user", "assistant"]
    assert session.is_empty() is False


def test_rejects_unknown_role():
    with pytest.raises(ValueError):
        Session().add("k", "wizard", "hello")


def test_build_context_keeps_recent_turns_within_budget():
    session = Session()
    key = "local::m"
    for _index in range(30):
        session.add(key, "user", "x" * 400)        # ~100 tokens each

    context = session.build_context(key, budget_tokens=500)
    kept = [m for m in context if m["role"] == "user"]

    assert 1 <= len(kept) <= 6                     # far fewer than 30
    assert context[0]["role"] == "system"
    assert "trimmed" in context[0]["content"]


def test_build_context_keeps_system_prompt():
    session = Session()
    key = "local::m"
    session.add(key, "user", "hello")

    context = session.build_context(key, budget_tokens=1000,
                                    system_prompt="You are terse.")

    assert context[0] == {"role": "system", "content": "You are terse."}
    assert context[-1]["content"] == "hello"


def test_build_context_always_keeps_at_least_the_latest_message():
    """A single huge attachment must not be trimmed away to nothing."""
    session = Session()
    key = "local::m"
    session.add(key, "user", "x" * 100_000)

    context = session.build_context(key, budget_tokens=100)
    assert any(m["role"] == "user" for m in context)


def test_save_load_round_trip(tmp_path):
    session = Session()
    key = make_key("host", "qwen:7b")
    session.add(key, "user", "say hello")
    session.add(key, "assistant", "Hello! 👋 café")

    path = session.save(tmp_path / "chat.json")
    loaded, _selected = Session.load(path)

    assert loaded.conversations == session.conversations
    assert "👋" in loaded.history(key)[1]["content"]


def test_load_normalises_legacy_files(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "conversations": {"llama3:8b": [{"role": "user", "content": "hi"}]},
        "selected_models": ["llama3:8b"],
    }), encoding="utf-8")

    session, selected = Session.load(path)

    assert "local::llama3:8b" in session.conversations
    assert selected == ["local::llama3:8b"]


def test_load_drops_malformed_messages(tmp_path):
    path = tmp_path / "messy.json"
    path.write_text(json.dumps({"conversations": {"local::m": [
        {"role": "user", "content": "good"},
        {"role": "user", "content": 12345},        # not a string
        "not a dict",
        {"role": "hacker", "content": "nope"},
    ]}}), encoding="utf-8")

    session, _ = Session.load(path)
    assert session.history("local::m") == [{"role": "user", "content": "good"}]


def test_load_rejects_nonsense(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"conversations": "not a dict"}), encoding="utf-8")

    with pytest.raises(ValueError):
        Session.load(path)


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("x" * 400) == 100
