"""
Tests for examples/agent_demo.py — the multi-step research agent runs end to end
against a mocked backend and the mission guardrail fires mid-run.
"""

import importlib.util
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import driftlock
from driftlock import DriftlockClient, DriftlockConfig, MissionBudgetExceededError

# Load the example module by path (examples/ is not an installed package).
_DEMO_PATH = Path(__file__).resolve().parent.parent / "examples" / "agent_demo.py"
_spec = importlib.util.spec_from_file_location("agent_demo", _DEMO_PATH)
agent_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_demo)


def _mock_response(model):
    r = MagicMock()
    r.model = model
    r.usage.prompt_tokens = 50
    r.usage.completion_tokens = 20
    r.usage.total_tokens = 70
    r.choices[0].message.content = "finding"
    return r


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    def _factory(cost_per_call=0.001):
        config = DriftlockConfig(db_path=str(tmp_path / "demo.db"), log_json=False)
        monkeypatch.setattr("driftlock.client.estimate_cost", lambda m, p, c: cost_per_call)
        with (
            patch("driftlock.client.OpenAI") as MockOpenAI,
            patch("driftlock.client.AsyncOpenAI") as MockAsync,
        ):
            MockOpenAI.return_value.chat.completions.create.side_effect = (
                lambda *a, **k: _mock_response(k.get("model", "?"))
            )
            MockAsync.return_value.chat.completions.create = AsyncMock(
                side_effect=lambda *a, **k: _mock_response(k.get("model", "?"))
            )
            return DriftlockClient(api_key="sk-test", config=config)
    return _factory


def test_parse_subtasks_fallback():
    subs = agent_demo._parse_subtasks("", "interest rates")
    assert len(subs) >= 3
    assert all("interest rates" in s or len(s) > 3 for s in subs)


def test_agent_completes_under_budget(demo_client):
    client = demo_client(cost_per_call=0.001)
    with driftlock.mission(
        "demo", budget_usd=5.0, expected_calls=8, on_exceed="downgrade",
        downgrade_to="gpt-4o-mini", mission_id="d_ok",
    ) as m:
        result = agent_demo.run_research_agent("interest rates", client, m,
                                               model="gpt-4o", verbose=False)
    assert isinstance(result, str) and result
    assert m.status == "completed"
    assert m.call_count >= 5  # 1 plan + >=3 research + 1 synthesis


def test_agent_downgrades_mid_run(demo_client):
    client = demo_client(cost_per_call=0.10)
    with driftlock.mission(
        "demo", budget_usd=0.30, expected_calls=8, on_exceed="downgrade",
        downgrade_to="gpt-4o-mini", mission_id="d_deg",
    ) as m:
        agent_demo.run_research_agent("topic", client, m, model="gpt-4o", verbose=False)
    assert m.status == "degraded"
    stats = client.mission_stats("d_deg")
    assert stats["intervention_count"] >= 1
    # The final (synthesis) call ran on the downgraded model.
    assert "gpt-4o-mini" in {d["model"] for d in stats["model_distribution"]}


def test_agent_kill_halts_run(demo_client):
    client = demo_client(cost_per_call=0.10)
    raised = False
    with driftlock.mission(
        "demo", budget_usd=0.30, expected_calls=8, on_exceed="kill", mission_id="d_kill",
    ) as m:
        try:
            agent_demo.run_research_agent("topic", client, m, model="gpt-4o", verbose=False)
        except MissionBudgetExceededError:
            raised = True
    assert raised
    assert m.status == "killed"


# ------------------------------------------------------------------ #
# Mock mode — full pipeline, no API key
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_env(tmp_path, monkeypatch):
    """Isolated cwd (demo writes ./driftlock.sqlite) + zero simulated latency."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DRIFTLOCK_MOCK_LATENCY_SCALE", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


def _latest_mission(tmp_path):
    from driftlock.storage import SQLiteStorage
    storage = SQLiteStorage(str(tmp_path / "driftlock.sqlite"))
    rows = storage.list_missions(limit=1)
    return rows[0] if rows else None


def test_mock_is_default_without_api_key(mock_env, capsys):
    rc = agent_demo.main(["some topic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[MOCK]" in out


def test_mock_downgrade_run_end_to_end(mock_env, capsys):
    rc = agent_demo.main(["interest rates", "--mock"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARNING" in out                      # soft warning fired
    assert "status=degraded" in out              # intervention armed mid-run
    row = _latest_mission(mock_env)
    assert row["status"] == "degraded"
    assert row["calls"] == 7                     # plan + 4 research + fact-check + synth
    assert row["total_cost_usd"] == pytest.approx(0.0795, abs=2e-3)


def test_mock_kill_halts_at_fact_check(mock_env, capsys):
    rc = agent_demo.main(["interest rates", "--mock", "--kill"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "KILLED" in out
    row = _latest_mission(mock_env)
    assert row["status"] == "killed"
    assert row["calls"] == 5                     # halted before fact-check (call 6)


def test_mock_flag_with_api_key_present(mock_env, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-unused")
    rc = agent_demo.main(["topic", "--mock"])
    assert rc == 0
    assert "[MOCK]" in capsys.readouterr().out


def test_mock_restores_estimate_cost(mock_env):
    import driftlock.client as dl_client
    original = dl_client.estimate_cost
    agent_demo.main(["topic", "--mock"])
    assert dl_client.estimate_cost is original


def test_mock_provider_cost_schedule():
    provider = agent_demo.MockProvider(downgrade_model="gpt-4o-mini")
    # Primary-model costs match the profile; downgraded calls are ~5.5%.
    assert provider.estimate_cost("gpt-4o", 120, 60) == pytest.approx(0.0003)
    assert provider.estimate_cost("gpt-4o", 640, 180) == pytest.approx(0.0420)
    assert provider.estimate_cost("gpt-4o-mini", 640, 180) == pytest.approx(0.0420 * 0.055, abs=1e-6)
