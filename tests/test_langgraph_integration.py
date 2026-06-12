"""
Tests for driftlock/integrations/langgraph.py.

The middleware only requires an object with ``.invoke(input, config)``, so a
FakeGraph drives the injected DriftlockCallbackHandler exactly the way LangGraph
does — including the ``langgraph_node`` metadata used for node attribution —
without needing langgraph installed.
"""

import threading
import uuid

import pytest

from driftlock import DriftlockClient, DriftlockConfig, MissionBudgetExceededError
from driftlock.integrations.langgraph import DriftlockLangGraphMiddleware


# ------------------------------------------------------------------ #
# Fakes
# ------------------------------------------------------------------ #

class _FakeLLMResult:
    """Shaped like LangChain's LLMResult as far as _extract_usage reads it."""

    def __init__(self, model="gpt-4o", prompt_tokens=100, completion_tokens=50):
        self.llm_output = {
            "model_name": model,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }


class FakeGraph:
    """
    Simulates a compiled LangGraph: each step fires the callback pair the way
    LangGraph does, stamping the node name into metadata.
    """

    def __init__(self, plan):
        # plan: list of (node_name, model) steps executed in order
        self.plan = plan

    def invoke(self, input, config=None, **kwargs):
        handler = config["callbacks"][-1]
        for node, model in self.plan:
            run_id = uuid.uuid4()
            handler.on_llm_start(
                {}, ["prompt"], run_id=run_id, metadata={"langgraph_node": node}
            )
            handler.on_llm_end(_FakeLLMResult(model=model), run_id=run_id)
        return {"ok": True, "input": input}


class ParallelFakeGraph:
    """
    Fires N callback pairs concurrently — models LangGraph's Send fan-out.

    LangGraph runs parallel nodes in an executor and propagates contextvars to
    the workers (langchain-core submits via ``copy_context().run``), so the
    fake does the same — that's what carries the active mission into threads.
    """

    def __init__(self, node: str, model: str, count: int):
        self.node, self.model, self.count = node, model, count

    def invoke(self, input, config=None, **kwargs):
        import contextvars

        handler = config["callbacks"][-1]

        def _one():
            run_id = uuid.uuid4()
            handler.on_llm_start(
                {}, ["p"], run_id=run_id, metadata={"langgraph_node": self.node}
            )
            handler.on_llm_end(_FakeLLMResult(model=self.model), run_id=run_id)

        # One context copy per worker — a single Context can't be entered twice.
        threads = [
            threading.Thread(target=contextvars.copy_context().run, args=(_one,))
            for _ in range(self.count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return {"ok": True}


@pytest.fixture
def dl_client(tmp_path, monkeypatch):
    """Client used purely as the persistence target; cost fixed via pricing patch."""
    def _factory(cost_per_call=0.1):
        monkeypatch.setattr(
            "driftlock.integrations.langchain.estimate_cost",
            lambda model, p, c: cost_per_call,
        )
        from unittest.mock import patch
        with patch("driftlock.client.OpenAI"), patch("driftlock.client.AsyncOpenAI"):
            return DriftlockClient(
                api_key="sk-test",
                config=DriftlockConfig(db_path=str(tmp_path / "lg.db"), log_json=False),
            )
    return _factory


_PLAN = [
    ("planner", "gpt-4o"),
    ("researcher", "gpt-4o"),
    ("researcher", "gpt-4o"),
    ("researcher", "gpt-4o"),
    ("synthesizer", "gpt-4o"),
]


# ------------------------------------------------------------------ #
# Normal completion
# ------------------------------------------------------------------ #

def test_normal_completion_under_budget(dl_client):
    client = dl_client(cost_per_call=0.01)
    graph = DriftlockLangGraphMiddleware(
        FakeGraph(_PLAN), client=client, mission_budget_usd=5.0, expected_calls=5
    )
    result = graph.invoke({"topic": "rates"})
    assert result["ok"]
    assert graph.last_mission_id is not None
    assert graph.last_mission.status == "completed"
    stats = client.mission_stats(graph.last_mission_id)
    assert stats["calls"] == 5
    assert stats["total_cost_usd"] == pytest.approx(0.05)


def test_mission_row_persisted(dl_client):
    client = dl_client(cost_per_call=0.01)
    graph = DriftlockLangGraphMiddleware(
        FakeGraph(_PLAN), client=client, mission_budget_usd=5.0
    )
    graph.invoke({})
    row = client._storage.get_mission(graph.last_mission_id)
    assert row["status"] == "completed"
    assert row["final_call_count"] == 5


# ------------------------------------------------------------------ #
# Node attribution
# ------------------------------------------------------------------ #

def test_node_attribution(dl_client):
    client = dl_client(cost_per_call=0.01)
    graph = DriftlockLangGraphMiddleware(
        FakeGraph(_PLAN), client=client, mission_budget_usd=5.0
    )
    graph.invoke({})
    calls = client._storage.mission_calls(graph.last_mission_id)
    endpoints = [c["endpoint"] for c in calls]
    assert endpoints.count("researcher") == 3
    assert "planner" in endpoints and "synthesizer" in endpoints


def test_node_attribution_in_call_graph(dl_client):
    client = dl_client(cost_per_call=0.01)
    graph = DriftlockLangGraphMiddleware(
        FakeGraph(_PLAN), client=client, mission_budget_usd=5.0
    )
    graph.invoke({})
    stats = client.mission_stats(graph.last_mission_id)
    nodes = {c["endpoint"] for c in stats["call_graph"]}
    assert {"planner", "researcher", "synthesizer"} <= nodes


def test_fallback_endpoint_without_node_metadata(dl_client):
    client = dl_client(cost_per_call=0.01)

    class NoMetadataGraph:
        def invoke(self, input, config=None, **kwargs):
            handler = config["callbacks"][-1]
            rid = uuid.uuid4()
            handler.on_llm_start({}, ["p"], run_id=rid)   # no metadata at all
            handler.on_llm_end(_FakeLLMResult(), run_id=rid)
            return {}

    graph = DriftlockLangGraphMiddleware(
        NoMetadataGraph(), client=client, mission_budget_usd=5.0
    )
    graph.invoke({})
    calls = client._storage.mission_calls(graph.last_mission_id)
    assert calls[0]["endpoint"] == "langgraph"


# ------------------------------------------------------------------ #
# Downgrade mid-graph
# ------------------------------------------------------------------ #

def test_downgrade_mid_graph(dl_client):
    client = dl_client(cost_per_call=0.10)
    graph = DriftlockLangGraphMiddleware(
        FakeGraph(_PLAN),
        client=client,
        mission_budget_usd=0.30,
        expected_calls=5,
        on_exceed="downgrade",
        downgrade_to="gpt-4o-mini",
    )
    graph.invoke({})
    assert graph.last_mission.status == "degraded"
    stats = client.mission_stats(graph.last_mission_id)
    assert stats["intervention_count"] >= 1
    assert stats["interventions"][0]["action"] == "downgrade"


def test_current_model_flips_on_degrade(dl_client):
    client = dl_client(cost_per_call=0.10)
    seen_models = []

    class ModelAwareGraph:
        """Each node asks the middleware which model to use — the real pattern."""

        def __init__(self):
            self.middleware = None

        def invoke(self, input, config=None, **kwargs):
            handler = config["callbacks"][-1]
            for node in ["planner", "researcher", "researcher", "researcher", "synthesizer"]:
                model = self.middleware.current_model("gpt-4o")
                seen_models.append(model)
                rid = uuid.uuid4()
                handler.on_llm_start({}, ["p"], run_id=rid,
                                     metadata={"langgraph_node": node})
                handler.on_llm_end(_FakeLLMResult(model=model), run_id=rid)
            return {}

    inner = ModelAwareGraph()
    graph = DriftlockLangGraphMiddleware(
        inner, client=client, mission_budget_usd=0.30, expected_calls=5,
        on_exceed="downgrade", downgrade_to="gpt-4o-mini",
    )
    inner.middleware = graph
    graph.invoke({})
    # Degrades once the projection breaches (after call 3) — later nodes
    # transparently get the cheaper model.
    assert seen_models[0] == "gpt-4o"
    assert seen_models[-1] == "gpt-4o-mini"


def test_current_model_default_before_any_run():
    graph = DriftlockLangGraphMiddleware(
        FakeGraph([]), mission_budget_usd=1.0, downgrade_to="gpt-4o-mini"
    )
    assert graph.current_model("gpt-4o") == "gpt-4o"


# ------------------------------------------------------------------ #
# Kill mid-graph
# ------------------------------------------------------------------ #

def test_kill_mid_graph(dl_client):
    client = dl_client(cost_per_call=0.10)
    graph = DriftlockLangGraphMiddleware(
        FakeGraph(_PLAN), client=client, mission_budget_usd=0.30,
        expected_calls=5, on_exceed="kill",
    )
    with pytest.raises(MissionBudgetExceededError):
        graph.invoke({})
    # The mission row records the kill, not a generic failure.
    row = client._storage.get_mission(graph.last_mission_id)
    assert row["status"] == "killed"
    assert graph.last_mission.status == "killed"


def test_kill_halts_before_remaining_nodes(dl_client):
    client = dl_client(cost_per_call=0.10)
    graph = DriftlockLangGraphMiddleware(
        FakeGraph(_PLAN), client=client, mission_budget_usd=0.30,
        expected_calls=5, on_exceed="kill",
    )
    with pytest.raises(MissionBudgetExceededError):
        graph.invoke({})
    stats = client.mission_stats(graph.last_mission_id)
    assert stats["calls"] < len(_PLAN)   # later nodes never executed


# ------------------------------------------------------------------ #
# Parallel node spend accumulation (Send fan-out)
# ------------------------------------------------------------------ #

def test_parallel_node_spend_accumulation(dl_client):
    client = dl_client(cost_per_call=0.02)
    graph = DriftlockLangGraphMiddleware(
        ParallelFakeGraph("researcher", "gpt-4o", count=8),
        client=client, mission_budget_usd=5.0,
    )
    graph.invoke({})
    m = graph.last_mission
    assert m.call_count == 8                       # no lost updates
    assert m.spent == pytest.approx(0.16)          # 8 * 0.02, race-free
    calls = client._storage.mission_calls(graph.last_mission_id)
    assert all(c["endpoint"] == "researcher" for c in calls)


# ------------------------------------------------------------------ #
# Config handling / misc
# ------------------------------------------------------------------ #

def test_existing_callbacks_preserved(dl_client):
    client = dl_client(cost_per_call=0.01)
    sentinel = object()

    class AssertingGraph:
        def invoke(self, input, config=None, **kwargs):
            assert config["callbacks"][0] is sentinel       # user callback kept
            assert len(config["callbacks"]) == 2            # ours appended
            return {}

    graph = DriftlockLangGraphMiddleware(
        AssertingGraph(), client=client, mission_budget_usd=1.0
    )
    graph.invoke({}, config={"callbacks": [sentinel]})


def test_warning_callback_fires(dl_client):
    client = dl_client(cost_per_call=0.10)
    fired = []
    graph = DriftlockLangGraphMiddleware(
        FakeGraph(_PLAN), client=client, mission_budget_usd=0.60,
        expected_calls=5, on_warning=lambda m: fired.append(m.spent),
        on_exceed="downgrade", downgrade_to="gpt-4o-mini",
    )
    graph.invoke({})
    assert len(fired) == 1


def test_verbose_prints_node_lines(dl_client, capsys):
    client = dl_client(cost_per_call=0.01)
    graph = DriftlockLangGraphMiddleware(
        FakeGraph(_PLAN), client=client, mission_budget_usd=5.0, verbose=True
    )
    graph.invoke({})
    out = capsys.readouterr().out
    assert "[Node: planner]" in out
    assert "[Node: researcher]" in out
    assert "Mission: $" in out


def test_import_safe_without_langgraph():
    # The module (and package export) must import without langgraph installed.
    import driftlock.integrations as integrations
    assert hasattr(integrations, "DriftlockLangGraphMiddleware")
