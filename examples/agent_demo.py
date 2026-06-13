"""
What:     A multi-step research agent (plan → parallel research → fact-check →
          synthesis, ~7 calls) under a mission budget that intervenes mid-run.
Requires: Nothing — runs in mock mode by default. Set OPENAI_API_KEY for a real run.
Run:      python examples/agent_demo.py "impact of interest rates on tech stocks"
          python examples/agent_demo.py "your topic" --kill     # hard stop instead of downgrade

----

Driftlock agent demo — a real multi-step research agent under a mission budget.

This is the wedge in action: a runaway-prone agent (planning → parallel research
→ fact-check → synthesis, ~7 LLM calls) wrapped in a single budgeted mission that
intervenes *mid-run* — downgrading the model (or killing the run) before the
budget blows.

Works out of the box with no API key (mock mode — simulated calls through the
full Driftlock pipeline)::

    python examples/agent_demo.py "impact of interest rates on tech stocks"

Or against a real API key::

    OPENAI_API_KEY=sk-... python examples/agent_demo.py "impact of interest rates on tech stocks"

Flags::

    --mock                 simulate the LLM calls (default when no OPENAI_API_KEY)
    --budget 0.03          hard mission budget in USD (default: 0.15 mock / 0.03 real)
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
import random
import threading
import time
from types import SimpleNamespace

import driftlock
import driftlock.client as _dl_client_module
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
    """Plan → research (parallel) → fact-check → synthesize. Returns the brief."""

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

    # 3) Fact-check — one call to vet the findings before they're combined.
    joined = "\n\n".join(f"- {f}" for f in findings)
    client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a fact checker. Flag dubious claims, briefly."},
            {"role": "user", "content": f"Verify these findings on '{topic}':\n{joined}"},
        ],
        max_tokens=200,
        _dl_endpoint="fact_check",
    )
    if verbose:
        _receipt("fact-check", client, mission)

    # 4) Synthesis — one call to combine findings.
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
# Mock mode — simulated calls through the *full* Driftlock pipeline
# --------------------------------------------------------------------------- #
#
# The MockProvider replaces only the HTTP backend inside DriftlockClient. The
# mission context, _before_call/_record_call hooks, policy surface, metrics, and
# SQLite persistence all run for real — the demo output is structurally
# identical to a real run.
#
# The cost profile is engineered against the EWMA projection (alpha=0.3,
# expected_calls=11) so that with the default $0.15 budget the warning fires
# during the research wave and the intervention arms right after it — the
# fact-check call is downgraded (or killed) before it goes out.

# (endpoint, cost_usd, latency_s [None = randomized 0.8–1.2s], prompt_tok, completion_tok)
_MOCK_PROFILE = [
    ("plan",       0.0003, 0.150, 120, 60),
    ("research",   0.0180, None,  420, 230),
    ("research",   0.0180, None,  410, 215),
    ("research",   0.0180, None,  450, 240),
    ("research",   0.0180, None,  430, 225),
    ("fact_check", 0.0420, 1.100, 640, 180),
    ("synthesize", 0.0890, 1.800, 920, 330),
]

# Cost lookup keyed by prompt_tokens — ties each response to its profile cost
# without any ordering assumptions between parallel calls.
_MOCK_COST_BY_PTOK = {p: cost for (_, cost, _, p, _) in _MOCK_PROFILE}

# Cheaper model costs ~5.5% of the primary (roughly gpt-4o → gpt-4o-mini).
_MOCK_MINI_FACTOR = 0.055

_MOCK_DEFAULT_BUDGET = 0.15
_MOCK_EXPECTED_CALLS = 11  # arms the EWMA guardrail right after the research wave


def _mock_topic(kwargs: dict) -> str:
    for m in reversed(kwargs.get("messages", [])):
        if m.get("role") == "user":
            text = str(m.get("content", ""))
            if "Topic:" in text:
                return text.split("Topic:", 1)[1].split("\n")[0].strip()
            return text[:60]
    return "the topic"


def _mock_content(stage: str, kwargs: dict) -> str:
    topic = _mock_topic(kwargs)
    if stage == "plan":
        return (
            f"Historical precedent and key drivers behind {topic}\n"
            f"Current data and leading indicators for {topic}\n"
            f"Sector winners and losers affected by {topic}\n"
            f"Expert forecasts and contrarian views on {topic}"
        )
    if stage == "research":
        return (
            "Finding: the relationship is strongly regime-dependent; the last "
            "three cycles show divergent outcomes depending on starting valuations."
        )
    if stage == "fact_check":
        return "Verified: findings are consistent; one figure adjusted for recency."
    return (
        f"In brief, {topic} shows a clear but conditional pattern. The effect is "
        "strongest when valuations are stretched and weakest mid-cycle. Watch the "
        "leading indicators rather than the headline number."
    )


class _MockResponse:
    """Shaped like an OpenAI ChatCompletion as far as the pipeline reads it."""

    def __init__(self, model: str, content: str, prompt_tokens: int, completion_tokens: int):
        self.model = model
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        self.choices = [
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(content=content, role="assistant"),
            )
        ]


class MockProvider:
    """
    Simulated LLM backend: sleeps for realistic latency and returns canned
    responses, while everything else in Driftlock runs unmodified.
    """

    def __init__(self, downgrade_model: str = "gpt-4o-mini", latency_scale: float = 1.0):
        self._downgrade_model = downgrade_model
        self._scale = max(0.0, latency_scale)
        self._i = 0
        self._lock = threading.Lock()
        self._orig_estimate = None
        self.sync_backend = SimpleNamespace(create=self._create)
        self.async_backend = SimpleNamespace(create=self._acreate)

    def _next(self) -> tuple[str, float, int, int]:
        with self._lock:
            idx = min(self._i, len(_MOCK_PROFILE) - 1)
            self._i += 1
        stage, _cost, latency, p, c = _MOCK_PROFILE[idx]
        if latency is None:
            latency = random.uniform(0.8, 1.2)
        return stage, latency, p, c

    def _build(self, stage: str, p: int, c: int, kwargs: dict) -> _MockResponse:
        return _MockResponse(
            model=kwargs.get("model", "gpt-4o"),
            content=_mock_content(stage, kwargs),
            prompt_tokens=p,
            completion_tokens=c,
        )

    def _create(self, *args, **kwargs) -> _MockResponse:
        stage, latency, p, c = self._next()
        time.sleep(latency * self._scale)
        return self._build(stage, p, c, kwargs)

    async def _acreate(self, *args, **kwargs) -> _MockResponse:
        stage, latency, p, c = self._next()
        await asyncio.sleep(latency * self._scale)
        return self._build(stage, p, c, kwargs)

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        base = _MOCK_COST_BY_PTOK.get(prompt_tokens, 0.001)
        if model == self._downgrade_model:
            base *= _MOCK_MINI_FACTOR
        return round(base, 6)

    def install(self, client: DriftlockClient) -> None:
        """Swap the client's HTTP backends and pin deterministic costs."""
        wrapper = client.chat.completions
        wrapper._sync = self.sync_backend
        wrapper._async = self.async_backend
        self._orig_estimate = _dl_client_module.estimate_cost
        _dl_client_module.estimate_cost = self.estimate_cost

    def uninstall(self) -> None:
        if self._orig_estimate is not None:
            _dl_client_module.estimate_cost = self._orig_estimate
            self._orig_estimate = None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Driftlock budgeted research agent demo.")
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--mock", action="store_true",
                        help="Simulate the LLM calls (default when no OPENAI_API_KEY)")
    parser.add_argument("--budget", type=float, default=None,
                        help="Mission budget USD (default: 0.15 mock / 0.03 real)")
    parser.add_argument("--model", default="gpt-4o", help="Primary model")
    parser.add_argument("--downgrade-to", default="gpt-4o-mini", help="Fallback model")
    parser.add_argument("--expected-calls", type=int, default=None, help="Projection hint")
    parser.add_argument("--kill", action="store_true", help="on_exceed='kill' instead of 'downgrade'")
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    mock = args.mock or not api_key
    budget = args.budget if args.budget is not None else (
        _MOCK_DEFAULT_BUDGET if mock else 0.03
    )
    expected_calls = args.expected_calls if args.expected_calls is not None else (
        _MOCK_EXPECTED_CALLS if mock else 8
    )

    client = DriftlockClient(
        api_key=api_key or "sk-mock",
        config=DriftlockConfig(log_level="WARNING", log_json=False),
    )

    provider: MockProvider | None = None
    if mock:
        provider = MockProvider(
            downgrade_model=args.downgrade_to,
            latency_scale=float(os.environ.get("DRIFTLOCK_MOCK_LATENCY_SCALE", "1")),
        )
        provider.install(client)

    on_exceed = "kill" if args.kill else "downgrade"
    tag = " [MOCK]" if mock else ""
    print(f"\nDriftlock research agent{tag} — topic: {args.topic!r}")
    print(
        f"  budget=${budget:.4f}  on_exceed={on_exceed}  "
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
    try:
        with driftlock.mission(
            "research_agent",
            budget_usd=budget,
            expected_calls=expected_calls,
            on_exceed=on_exceed,
            downgrade_to=args.downgrade_to,
            on_warning=on_warning,
        ) as mission:
            try:
                result = run_research_agent(args.topic, client, mission, model=args.model)
            except MissionBudgetExceededError as exc:
                print(f"\n  🛑 Mission KILLED mid-run: {exc.rule_name}")
                print(f"     {exc.decision.metadata}\n")
    finally:
        if provider is not None:
            provider.uninstall()

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
