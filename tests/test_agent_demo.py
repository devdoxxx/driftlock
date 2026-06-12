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


def test_demo_main_requires_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc = agent_demo.main(["some topic"])
    assert rc == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
