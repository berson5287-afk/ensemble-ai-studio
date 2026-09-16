"""The hosted front end, driven through FastAPI's test client with fake models."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from aichatlab.web import server  # noqa: E402
from tests.conftest import FakeClient  # noqa: E402

SESSION = "browser-session-1"


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.delenv("DEMO_PASSPHRASE", raising=False)
    fake = FakeClient(server="cloud")
    app = server.create_app(clients={"cloud": fake})
    return TestClient(app), fake


def events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def run(client, **overrides):
    body = {"session": SESSION, "mode": "chat", "models": ["alpha:1b"],
            "prompt": "hello"}
    body.update(overrides)
    return client.post("/api/run", json=body)


def test_index_and_health(rig):
    client, _fake = rig
    assert client.get("/api/health").json()["ok"] is True
    page = client.get("/")
    assert page.status_code == 200
    assert "Ensemble AI Studio" in page.text


def test_models_lists_what_the_client_offers(rig):
    client, _fake = rig
    data = client.get("/api/models").json()
    assert [m["id"] for m in data["models"]] == ["alpha:1b", "beta:2b"]
    assert {m["id"] for m in data["modes"]} == set(server.WEB_MODES)
    assert data["private"] is False


def test_chat_streams_the_orchestrator_events(rig):
    client, fake = rig
    response = run(client, models=["alpha:1b", "beta:2b"])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    kinds = [e["kind"] for e in events(response)]
    assert kinds[0] == "run_start"
    assert kinds[-1] == "done"
    assert kinds.count("turn_start") == 2
    assert kinds.count("turn_end") == 2
    starts = [e for e in events(response) if e["kind"] == "turn_start"]
    assert {s["target"]["model"] for s in starts} == {"alpha:1b", "beta:2b"}
    ends = [e for e in events(response) if e["kind"] == "turn_end"]
    assert ends[0]["result"]["eval_tokens"] == 12
    assert len(fake.calls) == 2


def test_conversation_history_survives_between_runs(rig):
    client, fake = rig
    run(client, prompt="first")
    run(client, prompt="second")
    roles = [m["role"] for m in fake.calls[-1]["messages"]]
    assert roles[-3:] == ["user", "assistant", "user"]


def test_reset_forgets_the_conversation(rig):
    client, fake = rig
    run(client, prompt="first")
    client.post("/api/reset", json={"session": SESSION})
    run(client, prompt="second")
    assert [m["role"] for m in fake.calls[-1]["messages"]
            if m["role"] != "system"] == ["user"]


def test_quality_sets_the_reply_allowance_and_directive(rig):
    client, fake = rig
    run(client, quality="quick")
    assert fake.calls[-1]["options"]["num_predict"] == server.QUALITY["quick"][0]
    assert "concise" in fake.calls[-1]["messages"][0]["content"].lower()


def test_debate_and_judge_modes_dispatch(rig):
    client, fake = rig
    response = run(client, mode="debate", models=["alpha:1b", "beta:2b"],
                   rounds=1, judge="beta:2b")
    headings = [e["heading"] for e in events(response) if e["kind"] == "turn_start"]
    assert any("round 1" in h for h in headings)
    assert headings[-1].endswith("consensus")

    fake.calls.clear()
    response = run(client, mode="judge", models=["alpha:1b", "beta:2b"])
    headings = [e["heading"] for e in events(response) if e["kind"] == "turn_start"]
    assert headings[-1].endswith("verdict")


def test_plan_mode_uses_the_first_model(rig):
    client, fake = rig
    response = run(client, mode="plan", models=["beta:2b", "alpha:1b"])
    assert response.status_code == 200
    assert {c["model"] for c in fake.calls} == {"beta:2b"}


@pytest.mark.parametrize("bad, status", [
    ({"models": []}, 400),
    ({"models": ["nope:1b"]}, 400),
    ({"mode": "research"}, 400),
    ({"quality": "ludicrous"}, 400),
    ({"prompt": "x" * (server.MAX_PROMPT_CHARS + 1)}, 422),
    ({"rounds": 99}, 422),
    ({"session": "short"}, 422),
])
def test_bad_requests_are_refused(rig, bad, status):
    client, _fake = rig
    assert run(client, **bad).status_code == status


def test_too_many_models_is_refused(rig):
    client, _fake = rig
    fake_many = FakeClient(server="cloud")
    fake_many.list_models = lambda timeout=15: [f"m{i}:1b" for i in range(6)]
    app = server.create_app(clients={"cloud": fake_many})
    many = TestClient(app)
    response = many.post("/api/run", json={
        "session": SESSION, "mode": "chat", "prompt": "hi",
        "models": [f"m{i}:1b" for i in range(5)]})
    assert response.status_code == 400
    assert "At most" in response.json()["error"]


def test_rate_limit_per_caller(rig, monkeypatch):
    client, _fake = rig
    monkeypatch.setattr(server, "RUNS_PER_MINUTE", 2)
    assert run(client).status_code == 200
    assert run(client).status_code == 200
    assert run(client).status_code == 429
    # A different caller (as Cloud Run reports it) is not affected.
    other = client.post("/api/run", headers={"X-Forwarded-For": "203.0.113.9"},
                        json={"session": SESSION, "mode": "chat",
                              "models": ["alpha:1b"], "prompt": "hi"})
    assert other.status_code == 200


def test_passphrase_gates_every_api_route(monkeypatch):
    monkeypatch.setenv("DEMO_PASSPHRASE", "open-sesame")
    app = server.create_app(clients={"cloud": FakeClient(server="cloud")})
    client = TestClient(app)
    assert client.get("/").status_code == 200            # the page itself is public
    assert client.get("/api/models").status_code == 401
    assert client.get("/api/models",
                      headers={"X-Passphrase": "open-sesame"}).status_code == 200
    assert client.get("/api/models",
                      headers={"X-Passphrase": "open-sesame"}).json()["private"]


def test_stop_and_say_need_a_live_run(rig):
    client, _fake = rig
    assert client.post("/api/stop", json={"run_id": "gone"}).json() == {"stopped": False}
    assert client.post("/api/say", json={"run_id": "gone", "text": "hi"}).status_code == 404


def test_model_labels_read_well():
    assert server.model_label("gemini-3.8-flash") == "Gemini 3.8 Flash"
    assert server.model_label("qwen2.5:32b-instruct") == "Qwen2.5 32B"


def test_headings_and_notes_use_the_web_labels():
    from aichatlab.orchestrator import Target

    targets = [Target("cloud", "gemini-3.8-flash"), Target("cloud", "gemini-3.7-flash")]
    heading = server.polish({"heading": "Gemini-3.8-Flash · round 1"}, targets)
    assert heading["heading"] == "Gemini 3.8 Flash · round 1"
    note = server.polish({"text": "Debate: Gemini-3.8-Flash, Gemini-3.7-Flash over 2 round(s)"},
                         targets)
    assert note["text"] == "Debate: Gemini 3.8 Flash, Gemini 3.7 Flash over 2 round(s)"
    # Ollama-style tags are left exactly as the orchestrator wrote them.
    assert server.polish({"heading": "Qwen2.5 32B"}, [Target("local", "qwen2.5:32b")]) == {
        "heading": "Qwen2.5 32B"}


def test_payloads_are_json_safe():
    from aichatlab.client import ChatResult
    from aichatlab.orchestrator import Target

    encoded = json.loads(server.encode("turn_end", {
        "target": Target("cloud", "gemini-3.8-flash"),
        "result": ChatResult(model="gemini-3.8-flash", text="x", eval_tokens=4,
                             elapsed_s=2.0, raw={"eval_duration": 2_000_000_000}),
        "extra": {"nested": (1, 2)},
    }))
    assert encoded["kind"] == "turn_end"
    assert encoded["target"]["label"] == "Gemini 3.8 Flash"
    assert encoded["result"]["tokens_per_second"] == 2.0
    assert encoded["extra"]["nested"] == [1, 2]


def test_a_judge_outside_the_panel_is_labelled_too(rig):
    client, _fake = rig
    fake_g = FakeClient(server="cloud")
    fake_g.list_models = lambda timeout=15: ["gemini-3.8-flash", "gemini-3.5-flash-lite",
                                             "gemini-3.7-flash"]
    app = server.create_app(clients={"cloud": fake_g})
    response = TestClient(app).post("/api/run", json={
        "session": SESSION, "mode": "debate", "prompt": "topic", "rounds": 1,
        "models": ["gemini-3.8-flash", "gemini-3.5-flash-lite"], "judge": "gemini-3.7-flash"})
    headings = [e["heading"] for e in events(response) if e["kind"] == "turn_start"]
    assert headings[-1] == "Gemini 3.7 Flash · consensus"
