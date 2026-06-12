"""
LangGraph compatibility shim.

``DriftlockLangGraphMiddleware`` wraps a compiled LangGraph graph so the whole
graph invocation runs inside a Driftlock mission — with node-level spend
attribution and mid-execution intervention::

    from driftlock.integrations.langgraph import DriftlockLangGraphMiddleware

    graph = DriftlockLangGraphMiddleware(
        compiled_graph,
        mission_budget_usd=1.00,
        on_exceed="downgrade",
        downgrade_to="gpt-4o-mini",
    )

    result = graph.invoke({"topic": "impact of interest rates on tech stocks"})
    print(graph.last_mission_id)   # query stats afterwards

How it works:

  - ``invoke()`` opens a ``driftlock.mission()`` around the graph run and
    injects a :class:`DriftlockCallbackHandler` into the LangChain ``config``
    so every LLM call inside any node reports into the mission.
  - LangGraph stamps the executing node's name into callback metadata as
    ``langgraph_node``; the handler records it as the call's ``_dl_endpoint``,
    so spend is attributable per node.
  - When the mission is killed, the next LLM call raises
    :class:`~driftlock.mission.MissionBudgetExceededError` from inside the
    graph; the error propagates out of ``invoke()`` and the mission is
    finalized as ``killed`` (not ``failed``).

Model downgrades and callbacks: LangChain's callback API cannot rewrite an
in-flight request, so the middleware cannot silently swap the model the way
``DriftlockClient`` does. Instead, nodes ask the middleware which model to use::

    llm = ChatOpenAI(model=graph.current_model("gpt-4o"))

``current_model()`` returns ``downgrade_to`` once the mission has degraded,
the default otherwise. This one-line pattern is what makes the downgrade
intervention real in a LangGraph agent.

This module is import-safe without ``langgraph`` installed — the middleware
only needs an object with ``.invoke(input, config)``.
"""

from __future__ import annotations

from typing import Any, Callable

from ..mission import MissionContext
from ..mission import mission as _mission
from .langchain import DriftlockCallbackHandler


class _VerboseHandler(DriftlockCallbackHandler):
    """Callback handler that prints a per-node spend line after each call."""

    def __init__(self, *, downgrade_to: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._downgrade_to = downgrade_to
        self._printed_downgrade = False

    def _on_recorded(self, metrics: Any, mission: Any) -> None:
        node = metrics.endpoint or "llm"
        cost = metrics.estimated_cost_usd or 0.0
        suffix = ""
        if self._downgrade_to and metrics.model == self._downgrade_to:
            suffix = "  ← cheaper model"
        print(
            f"[Node: {node}]".ljust(20)
            + f"${cost:.4f}  |  Mission: ${mission.spent:.4f} / ${mission.budget:.2f}"
            + suffix
        )
        if mission.status == "degraded" and not self._printed_downgrade:
            self._printed_downgrade = True
            print(f"🔀 Downgrade → {self._downgrade_to} (budget pressure)")


class DriftlockLangGraphMiddleware:
    """
    Wrap a compiled LangGraph graph with a Driftlock mission budget.

    Args:
        graph: A compiled LangGraph graph (anything with ``.invoke(input, config)``).
        mission_budget_usd: Hard spend ceiling for one ``invoke()``.
        client: Optional ``DriftlockClient`` whose storage persists the call
            records (so ``mission_stats``/CLI/dashboard see the run). Without
            it, guardrails still work in memory but nothing is persisted.
        mission_name: Label for the mission rows.
        on_exceed / downgrade_to / expected_calls / on_warning /
        warning_threshold / callback: Forwarded to :func:`driftlock.mission`.
        verbose: Print a per-node spend line after each LLM call (demo-style).
    """

    def __init__(
        self,
        graph: Any,
        *,
        mission_budget_usd: float,
        client: Any = None,
        mission_name: str = "langgraph_mission",
        on_exceed: str = "kill",
        downgrade_to: str | None = None,
        expected_calls: int | None = None,
        on_warning: Callable[[MissionContext], Any] | None = None,
        warning_threshold: float = 0.8,
        callback: Callable[[MissionContext], str] | None = None,
        verbose: bool = False,
    ) -> None:
        self._graph = graph
        self._client = client
        self._mission_name = mission_name
        self._budget = mission_budget_usd
        self._on_exceed = on_exceed
        self._downgrade_to = downgrade_to
        self._expected_calls = expected_calls
        self._on_warning = on_warning
        self._warning_threshold = warning_threshold
        self._callback = callback
        self._verbose = verbose

        self.last_mission: MissionContext | None = None
        self.last_mission_id: str | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def current_model(self, default: str) -> str:
        """
        The model a node should use right now: ``downgrade_to`` once the
        active mission has degraded, ``default`` otherwise.
        """
        m = self.last_mission
        if m is not None and m.status == "degraded" and self._downgrade_to:
            return self._downgrade_to
        return default

    def invoke(self, input: Any, config: dict | None = None, **kwargs: Any) -> Any:
        """
        Run the wrapped graph inside a budgeted mission.

        A killed mission raises ``MissionBudgetExceededError`` out of this
        call; the mission row is finalized as ``killed``.
        """
        handler = self._make_handler()
        config = dict(config or {})
        callbacks = list(config.get("callbacks") or [])
        callbacks.append(handler)
        config["callbacks"] = callbacks

        with _mission(
            self._mission_name,
            budget_usd=self._budget,
            on_exceed=self._on_exceed,
            downgrade_to=self._downgrade_to,
            expected_calls=self._expected_calls,
            on_warning=self._on_warning,
            warning_threshold=self._warning_threshold,
            callback=self._callback,
        ) as m:
            self.last_mission = m
            self.last_mission_id = m.mission_id
            return self._graph.invoke(input, config, **kwargs)

    def _make_handler(self) -> DriftlockCallbackHandler:
        if self._verbose:
            return _VerboseHandler(
                client=self._client,
                endpoint="langgraph",
                downgrade_to=self._downgrade_to,
            )
        return DriftlockCallbackHandler(client=self._client, endpoint="langgraph")
