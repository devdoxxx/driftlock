# Driftlock

AI agents are unpredictable, and a single runaway loop can burn hundreds of dollars before anyone notices — tools like LangSmith and Helicone only tell you after it already happened. Driftlock runs *inside* your agent's execution and intervenes **before** the breach: it projects spend from the live burn rate and downgrades models, pauses runs, or kills them outright when a mission budget is about to blow. It's a drop-in wrapper around the OpenAI and Anthropic clients with a policy engine, cost optimizer, and cache underneath.

```bash
pip install driftlock
```

---

## See it in 30 seconds

No API key needed — this runs the full pipeline (mission, guardrails, interventions, SQLite) with simulated LLM calls:

```bash
git clone https://github.com/maddox-214/driftlock && cd driftlock
pip install -e .
python examples/agent_demo.py "impact of interest rates on tech stocks"
```

```text
Driftlock research agent [MOCK] — topic: 'impact of interest rates on tech stocks'
  budget=$0.1500  on_exceed=downgrade  model=gpt-4o → gpt-4o-mini

  plan                   model=gpt-4o         call=$0.0003  spent=$0.0003  [------------------------]   0.2%
  ⚠️  WARNING: $0.0543 spent of $0.1500, projecting $0.1378
  research (parallel)    model=gpt-4o         call=$0.0180  spent=$0.0723  [############------------]  48.2%
                         projected_final=$0.1548  status=degraded
  fact-check             model=gpt-4o-mini    call=$0.0023  spent=$0.0746  [############------------]  49.7%
  synthesize             model=gpt-4o-mini    call=$0.0049  spent=$0.0795  [#############-----------]  53.0%
======================================================================
Mission complete: $0.0795 spent | 7 calls | status=degraded
  interventions:
    downgrade: projected_final_cost $0.154801 exceeds budget $0.150000
```

The agent never actually exceeded its budget. Driftlock **projected** the breach from the live burn rate and downgraded the model *before* the expensive calls went out — the run finished at 53% of budget instead of 135%. Add `--kill` for a hard stop instead.

---

## How it works

### Basic call tracking

`DriftlockClient` is a drop-in for `openai.OpenAI()`. Every call is costed, timed, and saved to local SQLite — no other code changes.

```python
from driftlock import DriftlockClient

client = DriftlockClient(api_key="sk-...")
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Policy engine

Rules run before every call and block, downgrade, or rate-limit a request the moment it violates a budget.

```python
from driftlock import DriftlockClient, PolicyEngine, MonthlyBudgetRule

policy = PolicyEngine(rules=[MonthlyBudgetRule(max_usd=100.0)])
client = DriftlockClient(api_key="sk-...", policy=policy)   # raises once $100/month is hit
```

### Mission budgets

A mission wraps a whole agent run and intervenes mid-execution when projected spend crosses the budget.

```python
import driftlock

with driftlock.mission("research", budget_usd=1.00,
                       on_exceed="downgrade", downgrade_to="gpt-4o-mini") as m:
    result = run_agent(topic, client)     # any number of tracked calls
    print(m.spent, m.status)
```

---

## Integrations

- **OpenAI** — drop-in `DriftlockClient` (sync, async, streaming). See [examples/basic_usage.py](examples/basic_usage.py).
- **Anthropic** — `AnthropicDriftlockClient` for the Messages API. See [examples/demo.py](examples/demo.py).
- **LangChain** — attach `DriftlockCallbackHandler` to any chat model. See [examples/langchain_agent_demo.py](examples/langchain_agent_demo.py).
- **LangGraph** — wrap a compiled graph in `DriftlockLangGraphMiddleware` for per-node attribution. See [examples/langgraph_agent_demo.py](examples/langgraph_agent_demo.py).

---

## How it compares

| Feature | Driftlock | LangSmith / Helicone |
|---|---|---|
| Observability (traces, cost logs) | ✅ | ✅ |
| Runtime intervention (before the next call) | ✅ | ❌ (post-hoc only) |
| Mission budgets for multi-call agent runs | ✅ | ❌ |
| Automatic model downgrade on budget pressure | ✅ | ❌ |
| Framework-agnostic (raw SDK, LangChain, LangGraph) | ✅ | Partial |

---

## Documentation

Full reference lives in [`docs/`](docs/) — [configuration](docs/configuration.md), [policy engine](docs/policy-engine.md), [missions](docs/missions.md), [optimization](docs/optimization.md), and the [CLI](docs/cli.md). Runnable examples are in [`examples/`](examples/).

---

## Roadmap

| Feature | Status |
|---|---|
| OpenAI chat wrapper (sync + async) | ✅ |
| Anthropic Messages wrapper (sync + async) | ✅ |
| Token tracking + cost estimation | ✅ |
| SQLite storage (auto-migrating) | ✅ |
| Structured JSON logging | ✅ |
| Policy engine (budget, velocity, model) | ✅ |
| Per-user / per-team budget caps | ✅ |
| Forecast-based budget blocking | ✅ |
| Velocity + cost circuit breakers | ✅ |
| Prompt optimization pipeline | ✅ |
| Exact in-memory response cache | ✅ |
| Streaming support | ✅ |
| Prompt drift detection | ✅ |
| Alert channels (Slack, Webhook, Log) | ✅ |
| Ambient tagging context manager | ✅ |
| CLI (stats, forecast, drift, top-users) | ✅ |
| **Mission budgets (runtime guardrails for agents)** | ✅ |
| **Mid-run intervention (downgrade / pause / kill / callback)** | ✅ |
| **EWMA burn-rate projection** | ✅ |
| **Nested missions with dual attribution** | ✅ |
| **Async-safe spend accounting (`asyncio.Lock`)** | ✅ |
| **Mission persistence + recovery (`resume_mission`)** | ✅ |
| **LangChain callback handler** | ✅ |
| **LangGraph middleware (per-node attribution)** | ✅ |
| **Mission dashboard data API** | ✅ |
| **Web dashboard (mission control UI)** | ✅ |
| **Zero-key mock demo (full pipeline, no API calls)** | ✅ |
| PyPI release | ✅ |
| Postgres / Redis storage backend | Next |
| OpenTelemetry export | Next |
| CrewAI / AutoGen integrations | Planned |
| Semantic (embedding-based) cache | Planned |
| Gemini adapter | Planned |

---

## License

MIT — see [LICENSE](LICENSE).
