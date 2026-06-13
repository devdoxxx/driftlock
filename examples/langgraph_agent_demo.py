"""
What:     A ReAct-style LangGraph research agent (planner → parallel researchers →
          synthesizer → critic loop) governed by a mission, with per-node spend
          attribution and mid-graph downgrade/kill.
Requires: pip install "driftlock[langgraph]"  +  OPENAI_API_KEY
Run:      OPENAI_API_KEY=sk-... python examples/langgraph_agent_demo.py "impact of interest rates on tech stocks"
          OPENAI_API_KEY=sk-... python examples/langgraph_agent_demo.py "..." --kill

----

Driftlock + LangGraph demo — a ReAct-style research agent with a critic loop,
governed by a mission budget.

The graph::

    planner ──> researcher (parallel, Send API) ──> synthesizer ──> critic
                     ▲                                                │
                     └──────────── needs more research? ─────────────┘

The critic→researcher loop is *intentional* — open-ended quality loops are
exactly where real agents burn budgets. The mission budget is tight ($0.30) so
Driftlock reliably intervenes before the loop runs away: with
``on_exceed="downgrade"`` the loop's extra research happens on the cheap model;
with ``--kill`` the next call after the breach raises and the run halts.

Each LLM call is attributed to its graph node (planner/researcher/synthesizer/
critic) via LangGraph's ``langgraph_node`` callback metadata, so per-node spend
shows up in ``driftlock mission <id>`` and the dashboard.

Model downgrade pattern: LangChain callbacks can't rewrite an in-flight request,
so each node asks the middleware which model to use::

    llm = ChatOpenAI(model=graph.current_model("gpt-4o"), ...)

Requires the optional extra::

    pip install "driftlock[langgraph]"

Run::

    OPENAI_API_KEY=sk-... python examples/langgraph_agent_demo.py "impact of interest rates on tech stocks"
    OPENAI_API_KEY=sk-... python examples/langgraph_agent_demo.py "..." --kill
"""

from __future__ import annotations

import argparse
import operator
import os
import sys
from typing import Annotated, TypedDict

from driftlock import DriftlockClient, DriftlockConfig, MissionBudgetExceededError
from driftlock.integrations.langgraph import DriftlockLangGraphMiddleware


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Driftlock + LangGraph research agent demo.")
    parser.add_argument("topic", nargs="?", default="impact of interest rates on tech stocks")
    parser.add_argument("--budget", type=float, default=0.30, help="Mission budget USD")
    parser.add_argument("--kill", action="store_true", help="on_exceed='kill' instead of 'downgrade'")
    args = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: set OPENAI_API_KEY to run the demo.", file=sys.stderr)
        return 1

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Send
    except ImportError:
        print(
            'LangGraph is not installed. Install the extra:\n'
            '  pip install "driftlock[langgraph]"',
            file=sys.stderr,
        )
        return 1

    # Persistence target so mission_stats / CLI / dashboard see the run.
    dl_client = DriftlockClient(
        api_key=os.environ["OPENAI_API_KEY"],
        config=DriftlockConfig(log_level="WARNING", log_json=False),
    )

    # ------------------------------------------------------------------ #
    # Graph state + nodes
    # ------------------------------------------------------------------ #

    class State(TypedDict):
        topic: str
        subtasks: list[str]
        findings: Annotated[list[str], operator.add]
        draft: str
        verdict: str
        rounds: int

    class ResearchTask(TypedDict):
        topic: str
        subtask: str

    # `graph` is assigned after compile; nodes close over it to pick the model
    # the mission currently allows (this is how the downgrade becomes real).
    graph: DriftlockLangGraphMiddleware | None = None

    def _llm(max_tokens: int) -> "ChatOpenAI":
        model = graph.current_model("gpt-4o") if graph else "gpt-4o"
        return ChatOpenAI(model=model, max_tokens=max_tokens)

    def planner(state: State) -> dict:
        resp = _llm(150).invoke(
            [
                SystemMessage(content=(
                    "Break the topic into exactly 3 concise research subtasks, "
                    "one per line, no numbering."
                )),
                HumanMessage(content=f"Topic: {state['topic']}"),
            ]
        )
        lines = [ln.strip("-•* \t") for ln in resp.content.splitlines() if len(ln.strip()) > 3]
        return {"subtasks": lines[:3] or [f"Overview of {state['topic']}"], "rounds": 0}

    def dispatch_research(state: State) -> list:
        # Send API: one researcher invocation per subtask, dispatched in parallel.
        return [
            Send("researcher", {"topic": state["topic"], "subtask": s})
            for s in state["subtasks"]
        ]

    def researcher(task: ResearchTask) -> dict:
        resp = _llm(220).invoke(
            [
                SystemMessage(content="You are a research analyst. Be concise and concrete."),
                HumanMessage(content=f"Research this for '{task['topic']}': {task['subtask']}"),
            ]
        )
        return {"findings": [resp.content]}

    def synthesizer(state: State) -> dict:
        joined = "\n\n".join(f"- {f}" for f in state["findings"])
        resp = _llm(280).invoke(
            [
                SystemMessage(content="Synthesize the findings into a 3-sentence brief."),
                HumanMessage(content=f"Topic: {state['topic']}\n\nFindings:\n{joined}"),
            ]
        )
        return {"draft": resp.content}

    def critic(state: State) -> dict:
        resp = _llm(60).invoke(
            [
                SystemMessage(content=(
                    "You are a harsh research critic. If the brief needs more depth, "
                    "reply exactly 'MORE RESEARCH: <one missing angle>'. "
                    "If it is solid, reply exactly 'ACCEPT'."
                )),
                HumanMessage(content=state["draft"]),
            ]
        )
        return {"verdict": resp.content, "rounds": state["rounds"] + 1}

    def route_after_critic(state: State):
        # The runaway loop: the critic can keep demanding more research.
        # (Capped at 5 rounds as a belt — the mission budget is the suspenders
        # that actually fires first.)
        if "MORE RESEARCH" in state.get("verdict", "") and state["rounds"] < 5:
            angle = state["verdict"].split(":", 1)[-1].strip() or state["topic"]
            return [Send("researcher", {"topic": state["topic"], "subtask": angle})]
        return END

    builder = StateGraph(State)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("critic", critic)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch_research, ["researcher"])
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("synthesizer", "critic")
    builder.add_conditional_edges("critic", route_after_critic, ["researcher", END])
    compiled = builder.compile()

    # ------------------------------------------------------------------ #
    # Wrap the graph in a mission
    # ------------------------------------------------------------------ #

    def on_warning(m):
        proj = m.projected_final_cost
        proj_s = f"${proj:.2f}" if proj is not None else "n/a"
        print(
            f"⚠  Warning: ${m.spent:.4f} spent — projecting {proj_s} final "
            f"(confidence: {m.projection_confidence})"
        )

    graph = DriftlockLangGraphMiddleware(
        compiled,
        client=dl_client,
        mission_name="langgraph_research",
        mission_budget_usd=args.budget,
        on_exceed="kill" if args.kill else "downgrade",
        downgrade_to="gpt-4o-mini",
        expected_calls=8,
        on_warning=on_warning,
        verbose=True,
    )

    print(f"\nLangGraph + Driftlock — topic: {args.topic!r}  budget=${args.budget:.2f}\n")
    result = None
    try:
        result = graph.invoke({"topic": args.topic, "findings": []})
    except MissionBudgetExceededError as exc:
        print(f"\n🛑 Mission KILLED mid-graph: {exc.rule_name}")

    m = graph.last_mission
    print(
        f"\nMission complete: ${m.spent:.4f} | {m.call_count} calls | status={m.status}"
    )
    if result and result.get("draft"):
        print(f"\nBrief:\n  {result['draft'].strip()}")

    print("\nPer-node spend:")
    stats = dl_client.mission_stats(graph.last_mission_id)
    by_node: dict[str, float] = {}
    for call in stats["call_graph"]:
        node = call.get("endpoint") or "?"
        by_node[node] = by_node.get(node, 0.0) + (call.get("cost_usd") or 0.0)
    for node, cost in sorted(by_node.items(), key=lambda kv: -kv[1]):
        print(f"  {node:<14} ${cost:.4f}")
    print(f"\nInspect:  driftlock mission {graph.last_mission_id}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
