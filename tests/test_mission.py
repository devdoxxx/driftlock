"""
Tests for the Mission system — runtime financial guardrails for agents.

Uses a mocked OpenAI backend (no real API calls) and a controllable per-call
cost so budget/intervention behaviour is fully deterministic.

Phase 2 semantics under test:
  - No projection (and no projection-based intervention) before 3 calls.
  - EWMA (alpha=0.3) burn-rate projection.
  - Nested missions: dual attribution (innermost + propagation up the stack).
  - Async record path guarded by asyncio.Lock (no races under gather).
  - Mission lifecycle persisted to SQLite (running/completed/degraded/killed/failed).
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import driftlock
from driftlock import (
    DriftlockClient,
    DriftlockConfig,
    MissionBudgetExceededError,
    MissionContext,
)
from driftlock.mission import current_mission


def _mock_response(model, prompt_tokens=50, completion_tokens=20):
    response = MagicMock()
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    response.choices[0].message.content = "Hello!"
    return response


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    """
    Factory for a DriftlockClient whose per-call cost is controllable.

    ``cost_per_call`` sets a constant; ``costs`` (a list) is consumed one value
    per call, then falls back to the constant — handy for EWMA tests where
    recent calls must differ from earlier ones.
    """
    state = {"const": 0.5, "costs": None}

    def _cost(model, p, c):
        if state["costs"]:
            return state["costs"].pop(0)
        return state["const"]

    def _factory(cost_per_call=0.5, costs=None):
        state["const"] = cost_per_call
        state["costs"] = list(costs) if costs else None
        config = DriftlockConfig(db_path=str(tmp_path / "missions.db"), log_json=False)
        monkeypatch.setattr("driftlock.client.estimate_cost", _cost)
        with patch("driftlock.client.OpenAI") as MockOpenAI:
            mock_openai = MockOpenAI.return_value
            mock_openai.chat.completions.create.side_effect = (
                lambda *a, **k: _mock_response(k.get("model", "unknown"))
            )
            c = DriftlockClient(api_key="sk-test", config=config)
            return c, mock_openai

    return _factory


def _call(client, model="gpt-4o", **kwargs):
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )


# ================================================================== #
# Normal completion + basic attribution
# ================================================================== #

def test_normal_completion_under_budget(make_client):
    client, _ = make_client(cost_per_call=0.5)
    with driftlock.mission("ok", budget_usd=100.0, expected_calls=3) as m:
        for _ in range(3):
            _call(client)
        assert m.call_count == 3
        assert m.spent == pytest.approx(1.5)
        assert m.remaining == pytest.approx(98.5)
        assert m.status == "completed"
        assert m.events == []


def test_mission_id_propagated_to_labels(make_client):
    client, _ = make_client(cost_per_call=0.1)
    with driftlock.mission("tagged", budget_usd=10.0, mission_id="mission_fixed"):
        _call(client)
    recent = client.recent_calls(limit=1)
    assert recent[0]["mission_id"] == "mission_fixed"
    assert recent[0]["labels"]["mission_id"] == "mission_fixed"


# ================================================================== #
# Burn-rate projection hardening (Task 1)
# ================================================================== #

def test_no_projection_before_three_calls(make_client):
    client, _ = make_client(cost_per_call=0.10)
    with driftlock.mission(
        "early", budget_usd=0.45, expected_calls=5, on_exceed="kill"
    ) as m:
        _call(client)
        assert m.projected_final_cost is None
        assert m.status == "completed"
        _call(client)
        assert m.projected_final_cost is None
        assert m.status == "completed"
        # 3rd call: projection now possible — 0.30 + (5-3)*0.10 = 0.50 > 0.45
        _call(client)
        assert m.projected_final_cost == pytest.approx(0.50)
        assert m.status == "killed"


def test_actual_breach_arms_even_before_three_calls(make_client):
    # An actual overspend is a fact, not a projection — it arms immediately.
    client, _ = make_client(cost_per_call=1.5)
    with driftlock.mission("breach", budget_usd=1.0, on_exceed="kill") as m:
        _call(client)  # spent 1.5 > 1.0
        assert m.status == "killed"
        with pytest.raises(MissionBudgetExceededError):
            _call(client)


def test_projected_final_cost_with_expected_calls(make_client):
    client, _ = make_client(cost_per_call=0.2)
    with driftlock.mission("proj", budget_usd=100.0, expected_calls=5) as m:
        for _ in range(3):
            _call(client)
        # ewma 0.2, remaining 5-3=2 -> 0.6 + 0.4 = 1.0
        assert m.projected_final_cost == pytest.approx(1.0)
        assert m.estimated_calls_remaining == 2
        assert m.burn_rate == pytest.approx(0.2)
        assert m.projection_confidence == "low"


def test_projected_final_cost_default_remaining_no_history(make_client):
    client, _ = make_client(cost_per_call=0.3)
    with driftlock.mission("proj2", budget_usd=100.0) as m:
        for _ in range(3):
            _call(client)
        # no expected_calls, no completed missions -> assume 20 remaining
        # 0.9 + 20*0.3 = 6.9
        assert m.projected_final_cost == pytest.approx(6.9)
        assert m.estimated_calls_remaining is None


def test_ewma_weights_recent_calls(make_client):
    client, _ = make_client(costs=[0.1, 0.1, 1.0], cost_per_call=1.0)
    with driftlock.mission("ewma", budget_usd=100.0, expected_calls=4) as m:
        for _ in range(3):
            _call(client)
        # ewma: 0.1 -> 0.1 -> 0.3*1.0 + 0.7*0.1 = 0.37
        assert m.burn_rate == pytest.approx(0.37)
        # spent 1.2, remaining 1 -> 1.2 + 0.37 = 1.57
        assert m.projected_final_cost == pytest.approx(1.57)


def test_projection_confidence_levels(make_client):
    client, _ = make_client(cost_per_call=0.001)
    with driftlock.mission("conf", budget_usd=100.0, expected_calls=100) as m:
        for _ in range(4):
            _call(client)
        assert m.projection_confidence == "low"  # < 5
        for _ in range(3):
            _call(client)
        assert m.projection_confidence == "medium"  # 7
        for _ in range(10):
            _call(client)
        assert m.projection_confidence == "high"  # 17


def test_warning_fires_once(make_client):
    client, _ = make_client(cost_per_call=0.10)
    fired = []
    with driftlock.mission(
        "warn",
        budget_usd=0.50,
        expected_calls=6,
        on_exceed="callback",
        callback=lambda mi: "continue",
        on_warning=lambda mi: fired.append(mi.spent),
        warning_threshold=0.8,
    ):
        # call 3: spent 0.30, projected 0.30 + 3*0.10 = 0.60 >= 0.40 threshold
        for _ in range(5):
            _call(client)
    assert len(fired) == 1  # exactly once across the whole run


# ================================================================== #
# Intervention modes
# ================================================================== #

def _tight_mission(name, **kwargs):
    # Arms after exactly 3 calls: cost 0.10, budget 0.45, expected 5 ->
    # projected at call 3 = 0.30 + 2*0.10 = 0.50 > 0.45.
    return driftlock.mission(
        name, budget_usd=0.45, expected_calls=5, **kwargs
    )


def test_kill_intervention(make_client):
    client, _ = make_client(cost_per_call=0.10)
    with _tight_mission("kill", on_exceed="kill") as m:
        for _ in range(3):
            _call(client)
        assert m.status == "killed"
        with pytest.raises(MissionBudgetExceededError) as exc:
            _call(client)
    assert exc.value.mission_id == m.mission_id
    assert exc.value.decision.metadata["budget_usd"] == 0.45


def test_kill_is_policy_violation_subclass(make_client):
    from driftlock import PolicyViolationError
    client, _ = make_client(cost_per_call=1.5)
    with driftlock.mission("kill2", budget_usd=1.0, on_exceed="kill"):
        _call(client)  # actual breach
        with pytest.raises(PolicyViolationError):
            _call(client)


def test_pause_intervention(make_client):
    client, _ = make_client(cost_per_call=0.10)
    with _tight_mission("pause", on_exceed="pause") as m:
        for _ in range(3):
            _call(client)
        with pytest.raises(MissionBudgetExceededError):
            _call(client)
    assert m.status == "killed"
    assert any(e["action"] == "pause" for e in m.events)


def test_downgrade_intervention(make_client):
    client, mock_openai = make_client(cost_per_call=0.10)
    with _tight_mission("deg", on_exceed="downgrade", downgrade_to="gpt-4o-mini") as m:
        for _ in range(3):
            _call(client, model="gpt-4o")
        assert m.status == "degraded"
        _call(client, model="gpt-4o")  # 4th call swapped

    sent = [c.kwargs["model"] for c in mock_openai.chat.completions.create.call_args_list]
    assert sent[:3] == ["gpt-4o", "gpt-4o", "gpt-4o"]
    assert sent[3] == "gpt-4o-mini"
    assert any(e["action"] == "downgrade" for e in m.events)


def test_callback_kill(make_client):
    client, _ = make_client(cost_per_call=0.10)
    seen = []

    def decide(mi):
        seen.append(mi.spent)
        return "kill"

    with _tight_mission("cbk", on_exceed="callback", callback=decide) as m:
        for _ in range(3):
            _call(client)
        assert seen
        with pytest.raises(MissionBudgetExceededError):
            _call(client)
    assert m.status == "killed"


def test_callback_downgrade(make_client):
    client, mock_openai = make_client(cost_per_call=0.10)
    with _tight_mission(
        "cbd", on_exceed="callback", callback=lambda mi: "downgrade", downgrade_to="gpt-4o-mini"
    ) as m:
        for _ in range(3):
            _call(client, model="gpt-4o")
        assert m.status == "degraded"
        _call(client, model="gpt-4o")
    sent = [c.kwargs["model"] for c in mock_openai.chat.completions.create.call_args_list]
    assert sent[-1] == "gpt-4o-mini"


def test_callback_continue_never_kills(make_client):
    client, _ = make_client(cost_per_call=0.10)
    with _tight_mission("cbc", on_exceed="callback", callback=lambda mi: "continue") as m:
        for _ in range(6):
            _call(client)  # never raises despite projected overage
        assert m.call_count == 6
        assert m.status == "completed"


# ================================================================== #
# Nested missions — dual attribution (Task 2)
# ================================================================== #

def test_nested_dual_attribution(make_client):
    client, _ = make_client(cost_per_call=0.5)
    with driftlock.mission("outer", budget_usd=100.0, mission_id="m_outer") as outer:
        _call(client)  # direct to outer
        assert current_mission() is outer
        with driftlock.mission("inner", budget_usd=100.0, mission_id="m_inner") as inner:
            assert current_mission() is inner
            _call(client)
            _call(client)
            assert inner.call_count == 2
            assert inner.spent == pytest.approx(1.0)
            assert inner.nested_spend == pytest.approx(0.0)
        assert current_mission() is outer
        _call(client)  # direct to outer

    # Outer sees its 2 direct calls AND the 2 nested calls (dual attribution).
    assert outer.call_count == 4
    assert outer.spent == pytest.approx(2.0)
    assert outer.direct_spend == pytest.approx(1.0)
    assert outer.nested_spend == pytest.approx(1.0)


def test_nested_inner_exhausts_outer_continues(make_client):
    client, _ = make_client(cost_per_call=0.10)
    with driftlock.mission(
        "outer", budget_usd=100.0, expected_calls=50, on_exceed="kill", mission_id="o1"
    ) as outer:
        with _tight_mission("inner", on_exceed="kill", mission_id="i1") as inner:
            for _ in range(3):
                _call(client)
            assert inner.status == "killed"
            with pytest.raises(MissionBudgetExceededError):
                _call(client)  # inner is killed
        # Outer still has plenty of budget — it keeps running.
        assert outer.status != "killed"
        _call(client)
        assert outer.status == "completed"
        assert outer.call_count >= 4


def test_nested_outer_exhausts_mid_inner_run(make_client):
    client, _ = make_client(cost_per_call=0.10)
    with driftlock.mission(
        "outer", budget_usd=0.45, expected_calls=5, on_exceed="kill", mission_id="o2"
    ) as outer:
        _call(client)
        _call(client)
        with driftlock.mission("inner", budget_usd=100.0, on_exceed="kill", mission_id="i2") as inner:
            _call(client)  # 3rd outer-attributed call -> outer projects over budget
            assert outer.status == "killed"
            assert inner.status != "killed"
            with pytest.raises(MissionBudgetExceededError) as exc:
                _call(client)  # outer guardrail halts the run mid-inner
            assert exc.value.mission_id == "o2"


def test_three_levels_deep(make_client):
    client, _ = make_client(cost_per_call=0.5)
    with driftlock.mission("l1", budget_usd=100.0, mission_id="l1") as l1:
        with driftlock.mission("l2", budget_usd=100.0, mission_id="l2") as l2:
            with driftlock.mission("l3", budget_usd=100.0, mission_id="l3") as l3:
                _call(client)
    assert l3.spent == pytest.approx(0.5)
    assert l3.nested_spend == pytest.approx(0.0)
    assert l2.spent == pytest.approx(0.5)
    assert l2.nested_spend == pytest.approx(0.5)
    assert l1.spent == pytest.approx(0.5)
    assert l1.nested_spend == pytest.approx(0.5)
    assert l1.call_count == 1


def test_mission_stats_direct_vs_nested(make_client):
    client, _ = make_client(cost_per_call=0.5)
    with driftlock.mission("outer", budget_usd=100.0, mission_id="ms_outer"):
        _call(client)  # direct
        with driftlock.mission("inner", budget_usd=100.0, mission_id="ms_inner"):
            _call(client)
            _call(client)

    stats = client.mission_stats("ms_outer")
    assert stats["direct_spend"] == pytest.approx(0.5)
    assert stats["nested_spend"] == pytest.approx(1.0)
    assert stats["total_cost_usd"] == pytest.approx(1.5)


# ================================================================== #
# Mission stats + call graph
# ================================================================== #

def test_mission_stats_call_graph_and_distribution(make_client):
    client, _ = make_client(cost_per_call=0.25)
    with driftlock.mission("g", budget_usd=100.0, mission_id="m_graph"):
        _call(client, model="gpt-4o")
        root_id = client.recent_calls(limit=1)[0]["request_id"]
        _call(client, model="gpt-4o-mini", _dl_parent_call_id=root_id)

    stats = client.mission_stats("m_graph")
    assert stats["calls"] == 2
    assert stats["total_cost_usd"] == pytest.approx(0.5)
    models = {d["model"] for d in stats["model_distribution"]}
    assert models == {"gpt-4o", "gpt-4o-mini"}
    assert len(stats["call_graph"]) == 1
    root_node = stats["call_graph"][0]
    assert root_node["call_id"] == root_id
    assert len(root_node["children"]) == 1
    assert root_node["children"][0]["parent_call_id"] == root_id


def test_mission_stats_includes_interventions(make_client):
    client, _ = make_client(cost_per_call=0.10)
    with _tight_mission("s", on_exceed="kill", mission_id="m_kill"):
        for _ in range(3):
            _call(client)
        with pytest.raises(MissionBudgetExceededError):
            _call(client)

    stats = client.mission_stats("m_kill")
    assert stats["status"] == "killed"
    assert stats["intervention_count"] == 1
    assert stats["interventions"][0]["action"] == "kill"


def test_interventions_excluded_from_call_stats(make_client):
    client, _ = make_client(cost_per_call=1.5)
    with driftlock.mission("iso", budget_usd=1.0, on_exceed="kill"):
        _call(client)  # actual breach -> 1 real call + 1 intervention row

    stats = client.stats()
    assert stats["calls"] == 1  # intervention row not counted
    recent = client.recent_calls(limit=10)
    assert all(r["model"] != "(intervention)" for r in recent)


# ================================================================== #
# Persistence + recovery (Task 6)
# ================================================================== #

def test_mission_lifecycle_persisted(make_client):
    client, _ = make_client(cost_per_call=0.1)
    with driftlock.mission("life", budget_usd=10.0, expected_calls=5, mission_id="m_life"):
        _call(client)
    row = client._storage.get_mission("m_life")
    assert row["status"] == "completed"
    assert row["final_call_count"] == 1
    assert row["final_spent"] == pytest.approx(0.1)
    assert row["started_at"] and row["ended_at"]


def test_running_row_inserted_on_enter(make_client):
    client, _ = make_client(cost_per_call=0.1)
    with driftlock.mission("running", budget_usd=10.0, mission_id="m_running"):
        # While inside the block the row exists and is 'running'.
        row = client._storage.get_mission("m_running")
        assert row is not None
        assert row["status"] == "running"


def test_mission_failed_status_on_exception(make_client):
    client, _ = make_client(cost_per_call=0.1)
    with pytest.raises(RuntimeError):
        with driftlock.mission("boom", budget_usd=10.0, mission_id="m_boom"):
            _call(client)
            raise RuntimeError("agent crashed")
    row = client._storage.get_mission("m_boom")
    assert row["status"] == "failed"
    assert row["final_spent"] == pytest.approx(0.1)
    assert row["final_call_count"] == 1


def test_resume_mission(make_client):
    client, _ = make_client(cost_per_call=0.1)
    with driftlock.mission("done", budget_usd=10.0, mission_id="m_done"):
        _call(client)
        _call(client)
    summary = client.resume_mission("m_done")
    assert summary.mission_id == "m_done"
    assert summary.status == "completed"
    assert summary.call_count == 2
    assert summary.spent_usd == pytest.approx(0.2)
    assert summary.over_budget is False
    assert summary.stats["calls"] == 2
    assert client.resume_mission("does_not_exist") is None


def test_missions_listing(make_client):
    client, _ = make_client(cost_per_call=0.10)
    with _tight_mission("listed", on_exceed="kill", mission_id="m_listed"):
        for _ in range(3):
            _call(client)

    rows = {r["mission_id"]: r for r in client.missions()}
    row = rows["m_listed"]
    assert row["calls"] == 3
    assert row["status"] == "killed"
    assert row["interventions"] == 1
    assert row["total_cost_usd"] == pytest.approx(0.3)


def test_parent_mission_id_persisted(make_client):
    client, _ = make_client(cost_per_call=0.1)
    with driftlock.mission("outer", budget_usd=100.0, mission_id="pm_outer"):
        with driftlock.mission("inner", budget_usd=100.0, mission_id="pm_inner"):
            _call(client)
    inner_row = client._storage.get_mission("pm_inner")
    assert inner_row["parent_mission_id"] == "pm_outer"
    outer_row = client._storage.get_mission("pm_outer")
    assert outer_row["parent_mission_id"] is None


# ================================================================== #
# Async hardening (Task 3)
# ================================================================== #

def _async_client(tmp_path, monkeypatch, cost=0.1):
    config = DriftlockConfig(db_path=str(tmp_path / "async.db"), log_json=False)
    monkeypatch.setattr("driftlock.client.estimate_cost", lambda m, p, c: cost)
    with (
        patch("driftlock.client.OpenAI"),
        patch("driftlock.client.AsyncOpenAI") as MockAsync,
    ):
        mock_async = MockAsync.return_value
        mock_async.chat.completions.create = AsyncMock(
            side_effect=lambda *a, **k: _mock_response(k.get("model", "unknown"))
        )
        return DriftlockClient(api_key="sk-test", config=config), mock_async


async def _acall(client, model="gpt-4o", **kwargs):
    return await client.chat.completions.acreate(
        model=model, messages=[{"role": "user", "content": "hi"}], **kwargs
    )


@pytest.mark.asyncio
async def test_async_parallel_calls_no_race(tmp_path, monkeypatch):
    client, _ = _async_client(tmp_path, monkeypatch, cost=0.1)
    with driftlock.mission("par", budget_usd=100.0, mission_id="m_par") as m:
        await asyncio.gather(*[_acall(client) for _ in range(5)])
        # All 5 parallel calls accumulate exactly — no lost updates.
        assert m.call_count == 5
        assert m.spent == pytest.approx(0.5)
    assert client.mission_stats("m_par")["calls"] == 5


@pytest.mark.asyncio
async def test_async_parallel_attribution_correct(tmp_path, monkeypatch):
    client, _ = _async_client(tmp_path, monkeypatch, cost=0.1)
    with driftlock.mission("attr", budget_usd=100.0, mission_id="m_attr"):
        await asyncio.gather(*[_acall(client) for _ in range(5)])
    calls = client._storage.mission_calls("m_attr")
    assert len(calls) == 5
    assert all(c["mission_id"] == "m_attr" for c in calls)


@pytest.mark.asyncio
async def test_async_high_concurrency_no_lost_updates(tmp_path, monkeypatch):
    # Stress the asyncio.Lock: 50 parallel calls must accumulate exactly.
    client, _ = _async_client(tmp_path, monkeypatch, cost=0.01)
    with driftlock.mission("stress", budget_usd=1000.0, mission_id="m_stress") as m:
        await asyncio.gather(*[_acall(client) for _ in range(50)])
        assert m.call_count == 50
        assert m.spent == pytest.approx(0.5)
    assert client.mission_stats("m_stress")["calls"] == 50


@pytest.mark.asyncio
async def test_async_mission_kill(tmp_path, monkeypatch):
    client, _ = _async_client(tmp_path, monkeypatch, cost=0.10)
    with driftlock.mission(
        "ak", budget_usd=0.45, expected_calls=5, on_exceed="kill", mission_id="m_ak"
    ) as m:
        for _ in range(3):
            await _acall(client)
        assert m.status == "killed"
        with pytest.raises(MissionBudgetExceededError):
            await _acall(client)
    assert client.mission_stats("m_ak")["status"] == "killed"


@pytest.mark.asyncio
async def test_async_downgrade(tmp_path, monkeypatch):
    client, mock_async = _async_client(tmp_path, monkeypatch, cost=0.10)
    with driftlock.mission(
        "ad", budget_usd=0.45, expected_calls=5, on_exceed="downgrade",
        downgrade_to="gpt-4o-mini", mission_id="m_ad",
    ) as m:
        for _ in range(3):
            await _acall(client, model="gpt-4o")
        assert m.status == "degraded"
        await _acall(client, model="gpt-4o")
    sent = [c.kwargs["model"] for c in mock_async.chat.completions.create.call_args_list]
    assert sent[-1] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_async_callback(tmp_path, monkeypatch):
    client, _ = _async_client(tmp_path, monkeypatch, cost=0.10)
    decisions = []

    def cb(mi):
        decisions.append(mi.spent)
        return "kill"

    with driftlock.mission(
        "acb", budget_usd=0.45, expected_calls=5, on_exceed="callback",
        callback=cb, mission_id="m_acb",
    ) as m:
        for _ in range(3):
            await _acall(client)
        assert decisions
        assert m.status == "killed"


# ================================================================== #
# Misc API guarantees
# ================================================================== #

def test_dl_parent_call_id_not_forwarded(make_client):
    client, mock_openai = make_client(cost_per_call=0.1)
    with driftlock.mission("p", budget_usd=10.0):
        _call(client, _dl_parent_call_id="abc")
    sent_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert "_dl_parent_call_id" not in sent_kwargs


def test_invalid_on_exceed_rejected():
    with pytest.raises(ValueError):
        MissionContext("bad", budget_usd=1.0, on_exceed="explode")


def test_mission_without_active_client_is_noop():
    with driftlock.mission("empty", budget_usd=1.0) as m:
        assert m.call_count == 0
        assert m.spent == 0.0
    assert current_mission() is None


def test_kill_escaping_block_finalizes_as_killed(make_client):
    """
    A MissionBudgetExceededError that propagates OUT of the with-block is the
    guardrail working — the mission must finalize as 'killed', not 'failed'.
    (Integrations like the LangGraph middleware rely on this.)
    """
    client, _ = make_client(cost_per_call=1.5)
    with pytest.raises(MissionBudgetExceededError):
        with driftlock.mission(
            "escape", budget_usd=1.0, on_exceed="kill", mission_id="m_escape"
        ):
            _call(client)   # breach -> kill armed
            _call(client)   # raises, escapes the block
    row = client._storage.get_mission("m_escape")
    assert row["status"] == "killed"


def test_unrelated_exception_still_finalizes_as_failed(make_client):
    client, _ = make_client(cost_per_call=0.1)
    with pytest.raises(RuntimeError):
        with driftlock.mission("boom", budget_usd=10.0, mission_id="m_boom"):
            _call(client)
            raise RuntimeError("agent crashed")
    row = client._storage.get_mission("m_boom")
    assert row["status"] == "failed"
