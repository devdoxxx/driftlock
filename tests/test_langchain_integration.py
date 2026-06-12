"""
Tests for the LangChain compatibility shim. These run WITHOUT langchain
installed — the handler is import-safe and uses a fallback base class — and
exercise the callback lifecycle with a fake ``LLMResult``.
"""

import pytest
from unittest.mock import patch

import driftlock
from driftlock import DriftlockClient, DriftlockConfig, MissionBudgetExceededError
from driftlock.integrations import DriftlockCallbackHandler


class _FakeLLMResult:
    def __init__(self, model="gpt-4o", prompt_tokens=100, completion_tokens=50):
        self.llm_output = {
            "model_name": model,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }


@pytest.fixture
def client(tmp_path):
    config = DriftlockConfig(db_path=str(tmp_path / "lc.db"), log_json=False)
    with patch("driftlock.client.OpenAI"), patch("driftlock.client.AsyncOpenAI"):
        return DriftlockClient(api_key="sk-test", config=config)


def test_handler_exported_from_package():
    assert hasattr(driftlock, "DriftlockCallbackHandler")


def test_handler_records_call_into_mission(client, monkeypatch):
    monkeypatch.setattr(
        "driftlock.integrations.langchain.estimate_cost", lambda m, p, c: 0.05
    )
    handler = DriftlockCallbackHandler(client=client)
    with driftlock.mission("lc", budget_usd=10.0, mission_id="lc1") as m:
        handler.on_llm_start({}, ["hi"])
        handler.on_llm_end(_FakeLLMResult(model="gpt-4o"))
        assert m.call_count == 1
        assert m.spent == pytest.approx(0.05)
    stats = client.mission_stats("lc1")
    assert stats["calls"] == 1
    assert {d["model"] for d in stats["model_distribution"]} == {"gpt-4o"}


def test_handler_kill_raises_on_start(client, monkeypatch):
    monkeypatch.setattr(
        "driftlock.integrations.langchain.estimate_cost", lambda m, p, c: 2.0
    )
    handler = DriftlockCallbackHandler(client=client)
    with driftlock.mission("lck", budget_usd=1.0, on_exceed="kill", mission_id="lc2") as m:
        handler.on_llm_start({}, ["hi"])
        handler.on_llm_end(_FakeLLMResult())  # spent 2.0 > 1.0 -> killed
        assert m.status == "killed"
        with pytest.raises(MissionBudgetExceededError):
            handler.on_llm_start({}, ["again"])


def test_handler_error_does_not_change_spend(client, monkeypatch):
    monkeypatch.setattr(
        "driftlock.integrations.langchain.estimate_cost", lambda m, p, c: 0.05
    )
    handler = DriftlockCallbackHandler(client=client)
    with driftlock.mission("lce", budget_usd=10.0) as m:
        handler.on_llm_start({}, ["hi"])
        handler.on_llm_error(RuntimeError("boom"))
        assert m.call_count == 0
        assert m.spent == 0.0


def test_handler_stamps_mission_id_in_metadata(client):
    handler = DriftlockCallbackHandler(client=client)
    metadata: dict = {}
    with driftlock.mission("lcm", budget_usd=10.0, mission_id="lc_meta"):
        handler.on_llm_start({}, ["hi"], metadata=metadata)
    assert metadata["driftlock_mission_id"] == "lc_meta"


def test_handler_noop_without_mission(client):
    handler = DriftlockCallbackHandler(client=client)
    # No active mission — must not raise.
    handler.on_llm_start({}, ["hi"])
    handler.on_llm_end(_FakeLLMResult())
