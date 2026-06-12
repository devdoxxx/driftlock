"""
Tests for the mission dashboard data API in examples/fastapi_app.py.

Populates a telemetry DB, points the app at it via DRIFTLOCK_DB_PATH, and
exercises the JSON routes with FastAPI's TestClient.
"""

import importlib.util
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

import driftlock
from driftlock import DriftlockClient, DriftlockConfig

_APP_PATH = Path(__file__).resolve().parent.parent / "examples" / "fastapi_app.py"


def _resp(model):
    r = MagicMock()
    r.model = model
    r.usage.prompt_tokens = 50
    r.usage.completion_tokens = 20
    r.usage.total_tokens = 70
    r.choices[0].message.content = "ok"
    return r


def _populate(db_path, monkeypatch):
    monkeypatch.setattr("driftlock.client.estimate_cost", lambda m, p, c: 0.1)
    with patch("driftlock.client.OpenAI") as M:
        M.return_value.chat.completions.create.side_effect = (
            lambda *a, **k: _resp(k.get("model"))
        )
        c = DriftlockClient(
            api_key="sk-test",
            config=DriftlockConfig(db_path=db_path, log_json=False),
        )
        with driftlock.mission("dash", budget_usd=10.0, expected_calls=5, mission_id="dash1"):
            for _ in range(3):
                c.chat.completions.create(
                    model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
                )


@pytest.fixture
def api(tmp_path, monkeypatch):
    db = str(tmp_path / "dash.db")
    _populate(db, monkeypatch)
    monkeypatch.setenv("DRIFTLOCK_DB_PATH", db)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("driftlock.client.OpenAI"), patch("driftlock.client.AsyncOpenAI"):
        spec = importlib.util.spec_from_file_location("fastapi_app_under_test", _APP_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    from fastapi.testclient import TestClient
    return TestClient(mod.app)


def test_list_missions(api):
    r = api.get("/missions")
    assert r.status_code == 200
    body = r.json()
    ids = {m["mission_id"] for m in body["missions"]}
    assert "dash1" in ids
    assert body["limit"] == 20 and body["offset"] == 0


def test_list_missions_pagination(api):
    r = api.get("/missions", params={"limit": 1, "offset": 0})
    assert r.status_code == 200
    assert len(r.json()["missions"]) <= 1


def test_mission_detail(api):
    r = api.get("/missions/dash1")
    assert r.status_code == 200
    body = r.json()
    assert body["calls"] == 3
    assert body["total_cost_usd"] == pytest.approx(0.3)
    assert body["status"] == "completed"


def test_mission_detail_404(api):
    r = api.get("/missions/nope")
    assert r.status_code == 404


def test_mission_call_graph(api):
    r = api.get("/missions/dash1/calls")
    assert r.status_code == 200
    body = r.json()
    assert body["mission_id"] == "dash1"
    assert len(body["call_graph"]) == 3  # flat (no parent links)


def test_metrics_summary(api):
    r = api.get("/metrics/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["today"]["calls"] >= 3
    assert body["today"]["spend_usd"] >= 0.3
    assert "this_month" in body


def test_burn_rate(api):
    r = api.get("/metrics/burn-rate", params={"hours": 24})
    assert r.status_code == 200
    body = r.json()
    assert body["hours"] == 24
    assert len(body["buckets"]) >= 1
    assert sum(b["calls"] for b in body["buckets"]) >= 3
