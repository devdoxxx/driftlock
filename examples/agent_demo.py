"""
Driftlock agent demo — a real multi-step research agent under a mission budget.

This is the wedge in action: a runaway-prone agent (planning → parallel research
→ synthesis, ~6–8 LLM calls) wrapped in a single budgeted mission that intervenes
*mid-run* — downgrading the model (or killing the run) before the budget blows.

Run it against a real API key::

    OPENAI_API_KEY=sk-... python examples/agent_demo.py "impact of interest rates on tech stocks"

Flags::

    --budget 0.03          hard mission budget in USD (default: 0.03 — tight, to show the wedge)
    --model gpt-4o         primary model (default: gpt-4o)
    --downgrade-to gpt-4o-mini   cheaper fallback model
    --kill                 use on_exceed="kill" instead of "downgrade"
    --expected-calls 8     projection hint

The agent logic (`run_research_agent`) is import-safe and provider-agnostic — it
just takes a `DriftlockClient`, so it is exercised by the test suite with a mock
backend as well as by this CLI with a real key.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import driftlock
from driftlock import DriftlockClient, DriftlockConfig, MissionBudgetExceededError


# --------------------------------------------------------------------------- #
# Pretty receipts
# --------------------------------------------------------------------------- #

def _bar(spent: float, budget: float, width: int = 24) -> str:
    frac = 0.0 if budget <= 0 else min(1.0, spent / budget)
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {frac * 100:5.1f}%"


def _receipt(label: str, client: DriftlockClient, mission) -> None:
    recent = client.recent_calls(limit=1)
    model = recent[0]["model"] if recent else "?"
    cost = (recent[0]["estimated_cost_usd"] if recent else 0.0) or 0.0
    proj = mission.projected_final_cost
    proj_s = f"${proj:.4f}" if proj is not None else "n/a (need 3+ calls)"
    print(
        f"  {label:<22} model={model:<14} call=${cost:.4f}  "
        f"spent=${mission.spent:.4f}  {_bar(mission.spent, mission.budget)}"
    )
    print(f"  {'':<22} projected_final={proj_s}  status={mission.status}")


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

_PLAN_SYS = (
    "You are a research planner. Break the user's topic into 4 concise, distinct "
    "research subtasks. Reply with one subtask per line, no numbering."
)


def _parse_subtasks(content: str, topic: str, n: int = 4) -> list[str]:
    lines = [ln.strip("-•* \t") for ln in (content or "").splitlines()]
    subtasks = [ln for ln in lines if len(ln) > 3][:n]
    # Robust fallback so the agent always has work to do (and stays testable).
    while len(subtasks) < 3:
        subtasks.append(f"Key aspect {len(subtasks) + 1} of: {topic}")
    return subtasks[:n]


def run_research_agent(
    topic: str,
    client: DriftlockClient,
    mission,
    *,
    model: str = "gpt-4o",
    verbose: bool = True,
) -> str:
    """Plan → research (parallel) → synthesize. Returns the synthesized answer."""

    # 1) Planning — one call to decompose the topic.
    try:
        plan = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _PLAN_SYS},
                {"role": "user", "content": f"Topic: {topic}"},
            ],
            max_tokens=200,
            _dl_endpoint="plan",
        )
        plan_text = plan.choices[0].message.content
    except MissionBudgetExceededError:
        raise
    subtasks = _parse_subtasks(plan_text, topic)
    if verbose:
        _receipt("plan", client, mission)
        print(f"  └─ {len(subtasks)} subtasks: {', '.join(s[:32] for s in subtasks)}")

    # 2) Research — parallel calls, one per subtask. The mission contextvar
    #    propagates into each gathered task, so every call is attributed and
    #    the asyncio.Lock keeps spend accumulation race-free.
    async def _research_all() -> list[str]:
        async def _one(subtask: str) -> str:
            resp = await client.chat.completions.acreate(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a research analyst. Be concise."},
                    {"role": "user", "content": f"Research this for '{topic}': {subtask}"},
                ],
                max_tokens=250,
                _dl_endpoint="research",
            )
            return resp.choices[0].message.content

        return await asyncio.gather(*[_one(s) for s in subtasks])

    try:
        findings = asyncio.run(_research_all())
    except MissionBudgetExceededError:
        raise
    if verbose:
        _receipt("research (parallel)", client, mission)
        print(f"  └─ {len(findings)} findings gathered")

    # 3) Synthesis — one call to combine findings.
    joined = "\n\n".join(f"- {f}" for f in findings)
    synth = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Synthesize the findings into a 3-sentence brief."},
            {"role": "user", "content": f"Topic: {topic}\n\nFindings:\n{joined}"},
        ],
        max_tokens=300,
        _dl_endpoint="synthesize",
    )
    if verbose:
        _receipt("synthesize", client, mission)
    return synth.choices[0].message.content


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Driftlock budgeted research agent demo.")
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--budget", type=float, default=0.03, help="Mission budget USD")
    parser.add_argument("--model", default="gpt-4o", help="Primary model")
    parser.add_argument("--downgrade-to", default="gpt-4o-mini", help="Fallback model")
    parser.add_argument("--expected-calls", type=int, default=8, help="Projection hint")
    parser.add_argument("--kill", action="store_true", help="on_exceed='kill' instead of 'downgrade'")
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: set OPENAI_API_KEY to run the demo.", file=sys.stderr)
        return 1

    client = DriftlockClient(
        api_key=api_key,
        config=DriftlockConfig(log_level="WARNING", log_json=False),
    )

    on_exceed = "kill" if args.kill else "downgrade"
    print(f"\nDriftlock research agent — topic: {args.topic!r}")
    print(
        f"  budget=${args.budget:.4f}  on_exceed={on_exceed}  "
        f"model={args.model} → {args.downgrade_to}\n"
    )

    def on_warning(m):
        proj = m.projected_final_cost
        proj_s = f"${proj:.4f}" if proj is not None else "n/a (need 3+ calls)"
        print(
            f"  ⚠️  WARNING: ${m.spent:.4f} spent of ${m.budget:.4f}, "
            f"projecting {proj_s}\n"
        )

    result = None
    with driftlock.mission(
        "research_agent",
        budget_usd=args.budget,
        expected_calls=args.expected_calls,
        on_exceed=on_exceed,
        downgrade_to=args.downgrade_to,
        on_warning=on_warning,
    ) as mission:
        try:
            result = run_research_agent(args.topic, client, mission, model=args.model)
        except MissionBudgetExceededError as exc:
            print(f"\n  🛑 Mission KILLED mid-run: {exc.rule_name}")
            print(f"     {exc.decision.metadata}\n")

    print("\n" + "=" * 70)
    print(
        f"Mission complete: ${mission.spent:.4f} spent | {mission.call_count} calls | "
        f"status={mission.status}"
    )
    if result:
        print("\nResult:\n  " + result.strip().replace("\n", "\n  "))

    stats = client.mission_stats(mission.mission_id)
    print("\nMission stats:")
    print(f"  total=${stats['total_cost_usd']:.4f}  calls={stats['calls']}  "
          f"interventions={stats['intervention_count']}  status={stats['status']}")
    print("  model distribution:")
    for d in stats["model_distribution"]:
        print(f"    {d['model']:<16} {d['calls']} calls  ${d['total_cost_usd']:.4f}")
    if stats["interventions"]:
        print("  interventions:")
        for ev in stats["interventions"]:
            print(f"    {ev['action']}: {ev['reason']}")
    print(f"\n  Inspect later:  driftlock mission {mission.mission_id}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
