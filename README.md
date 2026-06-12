# Driftlock

**Runtime financial guardrails for AI agents.**

An autonomous agent can make dozens of LLM calls per run. When one goes off the
rails — a retry loop, an exploding context, a runaway plan — you find out *after*
the bill lands. Driftlock gives an agent run a **budget** and intervenes
**mid-execution**: it stops, reroutes, or downgrades the model *before* the budget
blows, not after.

```python
import driftlock

with driftlock.mission(
    "research_agent",
    budget_usd=0.50,
    on_exceed="downgrade",          # downgrade | pause | kill | callback
    downgrade_to="gpt-4o-mini",
    on_warning=lambda m: print(f"⚠️  ${m.spent:.3f} spent, projecting ${m.projected_final_cost:.3f}"),
) as mission:
    result = run_agent(topic, client)   # any number of tracked LLM calls

print(f"${mission.spent:.4f} | {mission.call_count} calls | status={mission.status}")
```

Every call inside the block is attributed to the mission. After each call
completes, Driftlock recomputes the burn rate and projected final cost and fires
the intervention **before the next call is allowed through**.

Underneath the mission layer is a full cost-governance stack — policy engine,
optimization pipeline, response cache, telemetry, alerts — that works as a
drop-in wrapper around the OpenAI and Anthropic clients (see [Core Features](#core-features)).

---

## How it compares

| | LangSmith / Helicone / Portkey | **Driftlock** |
|---|---|---|
| Primary mode | Observability — log what happened | **Intervention** — change what happens next |
| When it acts | **After** the call (post-hoc traces) | **Before** the next call (pre-emptive) |
| Budget enforcement | Dashboards & alerts you read later | **Mission budget enforced at runtime** |
| Runaway agent | Visible in the trace, after the spend | **Killed / downgraded mid-run** |
| Cost projection | Reporting on past usage | **Live EWMA burn-rate projection per run** |

Observability tools tell you that you overspent. Driftlock stops you from
overspending. They are complementary — keep your tracing; add a guardrail.

---

## Install

```bash
pip install driftlock
```

With Anthropic support:

```bash
pip install "driftlock[anthropic]"
```

With LangChain, LangGraph, or FastAPI support:

```bash
pip install "driftlock[langchain]"
pip install "driftlock[langgraph]"
pip install "driftlock[fastapi]"
```

Requires Python ≥ 3.11.

---

## Try It in 30 Seconds (no API key needed)

```bash
git clone https://github.com/maddox-214/driftlock && cd driftlock
pip install -e .
python examples/agent_demo.py "impact of interest rates on tech stocks"
```

No `OPENAI_API_KEY` set? The demo runs in **mock mode**: a realistic 7-call
research agent (plan → 4 parallel research → fact-check → synthesis) flows
through the *full* Driftlock pipeline — mission context, guardrail hooks,
intervention engine, SQLite persistence — everything real except the HTTP
request. Here's the actual output:

```text
Driftlock research agent [MOCK] — topic: 'impact of interest rates on tech stocks'
  budget=$0.1500  on_exceed=downgrade  model=gpt-4o → gpt-4o-mini

  plan                   model=gpt-4o         call=$0.0003  spent=$0.0003  [------------------------]   0.2%
                         projected_final=n/a (need 3+ calls)  status=completed
  └─ 4 subtasks: Historical precedent and key dri, Current data and leading indicat, ...
  ⚠️  WARNING: $0.0543 spent of $0.1500, projecting $0.1378

  research (parallel)    model=gpt-4o         call=$0.0180  spent=$0.0723  [############------------]  48.2%
                         projected_final=$0.1548  status=degraded
  └─ 4 findings gathered
  fact-check             model=gpt-4o-mini    call=$0.0023  spent=$0.0746  [############------------]  49.7%
                         projected_final=$0.1262  status=degraded
  synthesize             model=gpt-4o-mini    call=$0.0049  spent=$0.0795  [#############-----------]  53.0%
                         projected_final=$0.1143  status=degraded

======================================================================
Mission complete: $0.0795 spent | 7 calls | status=degraded

Mission stats:
  total=$0.0795  calls=7  interventions=1  status=degraded
  model distribution:
    gpt-4o           5 calls  $0.0723
    gpt-4o-mini      2 calls  $0.0072
  interventions:
    downgrade: projected_final_cost $0.154801 exceeds budget $0.150000
```

Read that intervention line again: the agent never actually exceeded its budget.
Driftlock **projected** the breach from the live burn rate and downgraded the
model *before* the expensive fact-check call went out — the run finished at 53%
of budget instead of 135%.

```bash
python examples/agent_demo.py "any topic" --kill    # hard stop instead of downgrade
driftlock missions                                  # both runs recorded with correct status
# mission_02de67d7...   $0.072300   calls=5   interventions=yes  status=killed
# mission_96c1e9e8...   $0.079505   calls=7   interventions=yes  status=degraded
```

With `OPENAI_API_KEY` set, the same command runs against the real API (or force
simulation with `--mock`).

---

## Demos

Every demo is a single command from a fresh clone:

| Demo | Command | Needs a key? |
|---|---|---|
| **Budgeted agent run** (warn → downgrade mid-run) | `python examples/agent_demo.py "your topic"` | No (mock) |
| **Kill switch** (hard stop mid-run) | `python examples/agent_demo.py "your topic" --kill` | No (mock) |
| **Web dashboard** (mission control UI) | `uvicorn examples.fastapi_app:app` → open localhost:8000 | No |
| **CLI receipt** (one real call + cost receipt) | `driftlock demo` | Yes |
| **Feature tour** (tracking, cache, policies) | `python examples/demo.py` | No |
| **LangChain agent** under a mission | `python examples/langchain_agent_demo.py "topic"` | Yes |
| **LangGraph agent** with critic loop + intervention | `python examples/langgraph_agent_demo.py "topic"` | Yes |

Run the mock agent demo first, then start the dashboard — it will already have
mission data to show.

---

## 60-Second Quickstart (real API)

```bash
export OPENAI_API_KEY=sk-...       # or ANTHROPIC_API_KEY=sk-ant-...
driftlock demo
```

Driftlock makes one cheap request (`gpt-4o-mini` or `claude-haiku-4-5`) under a default policy and prints a receipt:

```
Driftlock demo  —  provider=openai  model=gpt-4o-mini

  ┌─ Receipt ──────────────────────────────────────────────┐
  │  provider  : openai                                    │
  │  model     : gpt-4o-mini                               │
  │  tokens    : 23 (15 in / 8 out)                        │
  │  cost      : $0.000007                                 │
  │  latency   : 412 ms                                    │
  │  db        : ./driftlock.sqlite                        │
  └────────────────────────────────────────────────────────┘

  Next steps:
    driftlock stats            # aggregate cost + token totals
    driftlock recent           # last 20 calls
    driftlock forecast         # projected monthly spend
```

---

## Core Features

The mission layer sits on top of a complete cost-governance stack. Everything
below works on its own (drop-in client wrapper) and composes with missions.

## Basic Integration — OpenAI

```python
from driftlock import DriftlockClient

# Replace openai.OpenAI() with DriftlockClient().
# All other arguments are forwarded to the OpenAI client unchanged.
client = DriftlockClient(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Every call is logged, costed, and saved to a local SQLite file.

```json
{
  "level": "INFO",
  "logger": "driftlock",
  "message": "model=gpt-4o-mini | tokens=157 | latency=421ms | cost=$0.000033",
  "metrics": {
    "timestamp": "2025-03-01T12:00:00+00:00",
    "model": "gpt-4o-mini",
    "prompt_tokens": 120,
    "completion_tokens": 37,
    "total_tokens": 157,
    "latency_ms": 421.3,
    "estimated_cost_usd": 0.0000330
  }
}
```

### Async

```python
response = await client.chat.completions.acreate(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Streaming

```python
for chunk in client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
# Metrics are logged and saved when the stream closes.
```

---

## Basic Integration — Anthropic

Requires `pip install -e ".[anthropic]"`.

```python
from driftlock import AnthropicDriftlockClient

client = AnthropicDriftlockClient(api_key="sk-ant-...")

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

`max_tokens` is required by Anthropic. The `system` parameter is a top-level kwarg, not a message role — same as the native SDK.

---

## Labelling Calls

Use `_dl_endpoint` and `_dl_labels` to annotate individual calls. These are stripped before the request reaches the provider.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    _dl_endpoint="summarise_article",        # logical function name
    _dl_labels={"user_id": "u_123", "team": "growth"},
)
```

`user_id` and `team_id` in labels are indexed in SQLite for fast per-user queries.

---

## Ambient Tagging

Attach labels to all calls within a scope without modifying every call site — useful in middleware:

```python
import driftlock

with driftlock.tag(request_id="req_abc", user_id="u_42", feature="chat"):
    response = client.chat.completions.create(...)
```

Tags from nested `driftlock.tag()` blocks merge; inner values override outer ones. Per-call `_dl_labels` always wins.

---

## Configuration

```python
from driftlock import DriftlockClient, DriftlockConfig

config = DriftlockConfig(
    log_json=True,                         # JSON logs (default). False = human-readable.
    log_level="INFO",
    storage_backend="sqlite",             # "sqlite" | "none"
    db_path="driftlock.sqlite",
    prompt_token_warning_threshold=4000,  # Warn if prompt > N tokens.
    cost_warning_threshold=0.10,          # Warn if a single call costs > $X.
    default_labels={"env": "prod"},       # Attached to every tracked call.
)

client = DriftlockClient(api_key="sk-...", config=config)
```

---

## Policy Engine

The policy engine enforces governance rules before every API call. Rules are evaluated in order; the first block raises `PolicyViolationError`.

```python
from driftlock import (
    DriftlockClient,
    PolicyEngine,
    MonthlyBudgetRule,
    MaxCostPerRequestRule,
    VelocityLimitRule,
    CostVelocityRule,
    PerUserBudgetRule,
    ForecastBudgetRule,
    RestrictModelRule,
    TagBasedModelDowngradeRule,
    PolicyViolationError,
)

policy = PolicyEngine(rules=[
    MonthlyBudgetRule(max_usd=100.0),                 # Block at $100/month workspace
    MaxCostPerRequestRule(max_usd=0.10),              # Block single calls > $0.10
    VelocityLimitRule(max_requests=60, window_seconds=60),  # 60 req/min circuit breaker
])

client = DriftlockClient(api_key="sk-...", policy=policy)

try:
    response = client.chat.completions.create(...)
except PolicyViolationError as e:
    print(f"Blocked by {e.rule_name}: {e.decision.metadata}")
```

### Available Rules

| Rule | What it does |
|---|---|
| `MonthlyBudgetRule(max_usd, scope="workspace"\|"user")` | Block once monthly spend cap is reached |
| `MaxCostPerRequestRule(max_usd)` | Block a single call if estimated cost exceeds the limit |
| `PerUserBudgetRule(user_budgets, default_max_usd)` | Per-user monthly caps from a dict |
| `ForecastBudgetRule(monthly_budget_usd, lookback_days=7)` | Block when projected 30-day spend will exceed budget |
| `VelocityLimitRule(max_requests, window_seconds, scope="workspace"\|"user")` | Circuit breaker on request rate |
| `CostVelocityRule(max_cost_usd, window_seconds)` | Circuit breaker on spend rate (e.g. $5/hour) |
| `RestrictModelRule(disallowed_models, condition=None)` | Block calls to specific models |
| `TagBasedModelDowngradeRule(condition, downgrade_to)` | Silently swap model based on labels |

### Per-User Budgets

```python
policy = PolicyEngine(rules=[
    PerUserBudgetRule(
        user_budgets={"power_user": 20.0, "free_tier": 1.0},
        default_max_usd=5.0,   # applied to any user_id not in the dict
    ),
])
# user_id is read from _dl_labels={"user_id": "..."} or ambient tags
```

### Forecast-Based Blocking

```python
policy = PolicyEngine(rules=[
    ForecastBudgetRule(monthly_budget_usd=50.0, lookback_days=7),
])
# Blocks before the budget is actually exhausted — proactive, not reactive
```

### Model Governance

```python
policy = PolicyEngine(rules=[
    # Block GPT-4o on free plan users
    RestrictModelRule(
        disallowed_models={"gpt-4o", "gpt-4"},
        condition=lambda ctx: ctx["labels"].get("plan") == "free",
    ),
    # Auto-downgrade free users to mini
    TagBasedModelDowngradeRule(
        condition=lambda ctx: ctx["labels"].get("plan") == "free",
        downgrade_to="gpt-4o-mini",
    ),
])
```

---

## Missions — Runtime Financial Guardrails for Agents

The policy engine evaluates one request at a time. A **mission** wraps a whole
agent run and intervenes **mid-execution** — stopping, rerouting, or downgrading
a running agent *before* budget damage compounds. This is the layer observability
tools (LangSmith, Helicone, Portkey) don't have: they log what happened; missions
change what happens next.

```python
import driftlock

with driftlock.mission(
    "research_task",
    budget_usd=1.00,
    on_exceed="downgrade",      # downgrade | pause | kill | callback
    downgrade_to="gpt-4o-mini",
) as m:
    result = agent.run("do the thing")   # any number of tracked calls
    print(m.spent, m.remaining, m.projected_final_cost)
```

Every tracked call inside the block is attributed to the mission (via the ambient
tag system). After each call completes, the mission re-evaluates burn rate and
projected final cost; if an overage is projected, the intervention fires **before
the next call is allowed through**.

**Intervention modes (`on_exceed`):**

- `"downgrade"` — transparently swap the model to `downgrade_to` for all
  subsequent calls in the mission.
- `"kill"` / `"pause"` — raise `MissionBudgetExceededError` on the next call.
- `"callback"` — hand the decision to your `callback(mission)`, which returns
  `"continue"`, `"downgrade"`, or `"kill"`.

**Live properties:** `m.spent`, `m.remaining`, `m.direct_spend`, `m.nested_spend`,
`m.call_count`, `m.burn_rate`, `m.projected_final_cost`, `m.projection_confidence`,
`m.status`.

**Burn-rate projection.** The projection uses an exponential weighted moving
average (EWMA, α=0.3) of per-call cost, so recent calls weigh more. It refuses to
project from fewer than 3 calls (`projected_final_cost is None` — early calls are
too noisy). With `expected_calls=N` the remaining-call count is exact; without it,
Driftlock uses the rolling average call count of your last 10 completed missions.
`projection_confidence` is `low` (<5 calls), `medium` (5–14), or `high` (15+).

> An *actual* budget breach always intervenes immediately — the 3-call minimum
> only gates *projection-based* pre-emption.

**Soft warnings:** pass `on_warning=callable` (fires **once** per mission when
spend or projection crosses `warning_threshold`, default `0.8` of the budget).

```python
def warn(m):
    print(f"⚠️  {m.spent:.2f}/{m.budget:.2f} spent, projecting {m.projected_final_cost}")

with driftlock.mission("batch", budget_usd=5.00, on_warning=warn, expected_calls=20):
    ...
```

**Nested missions (dual attribution).** Missions nest. Every call attributes to
its innermost mission *and* propagates its cost up the whole stack — so a parent
mission's budget accounts for all nested work — while each level evaluates its own
budget independently (an inner budget can be exhausted without killing the outer
run; an exhausted outer run halts everything).

**Async-safe.** The async record path is guarded by an `asyncio.Lock`, so parallel
`asyncio.gather` sub-calls inside one mission accumulate spend without races.

**Persistence & recovery.** Each mission is written to a `missions` table —
`running` on enter, then `completed` / `degraded` / `killed` / `failed` on exit —
so a crashed process leaves a recoverable record.

```python
summary = client.resume_mission(mission_id)   # read-only MissionSummary for post-run analysis
summary.status, summary.spent_usd, summary.nested_spent_usd, summary.over_budget
```

**Querying afterwards:**

```python
client.mission_stats(mission_id)   # spend, direct/nested split, call graph, model mix, interventions
client.missions(limit=20)          # recent missions with status
```

```bash
driftlock missions            # list recent missions (spend, calls, status)
driftlock mission <id>        # per-mission detail + call graph + interventions
```

**LangChain.** Already using LangChain? Attach `DriftlockCallbackHandler` to your
LLM and missions govern it the same way (`pip install "driftlock[langchain]"`):

```python
from driftlock.integrations import DriftlockCallbackHandler
llm = ChatOpenAI(model="gpt-4o", callbacks=[DriftlockCallbackHandler(client=dl_client)])
with driftlock.mission("lc_agent", budget_usd=0.50, on_exceed="kill"):
    agent.invoke(...)
```

**LangGraph.** Wrap a compiled graph in `DriftlockLangGraphMiddleware` and the
whole invocation runs as a mission, with spend attributed **per graph node**
(`pip install "driftlock[langgraph]"`):

```python
from driftlock.integrations.langgraph import DriftlockLangGraphMiddleware

graph = DriftlockLangGraphMiddleware(
    compiled_graph,
    client=dl_client,
    mission_budget_usd=0.30,
    on_exceed="downgrade",
    downgrade_to="gpt-4o-mini",
)
result = graph.invoke({"topic": "impact of interest rates on tech stocks"})
stats = dl_client.mission_stats(graph.last_mission_id)   # per-node spend, interventions
```

Nodes pick their model through the middleware — one line makes the downgrade
intervention real inside the graph:

```python
llm = ChatOpenAI(model=graph.current_model("gpt-4o"))
```

[examples/langgraph_agent_demo.py](examples/langgraph_agent_demo.py) builds a
ReAct-style agent with a **critic→researcher loop** — the classic runaway-cost
shape — and shows the mission stepping in before the loop burns the budget. A
killed graph finalizes its mission as `killed` (not `failed`), so post-mortems
stay accurate.

Mission calls and intervention events are persisted to SQLite as their own record
types, so the full run timeline is queryable later. Existing call analytics
(`stats`, `forecast`, velocity rules) ignore intervention rows.

---

## Alerts

Fire-and-forget notifications when policies trip or cost thresholds are crossed.

```python
from driftlock import DriftlockConfig, WebhookAlertChannel, SlackAlertChannel, LogAlertChannel

config = DriftlockConfig(
    alert_channels=[
        SlackAlertChannel(webhook_url="https://hooks.slack.com/services/..."),
        WebhookAlertChannel(url="https://example.com/hooks/driftlock"),
        LogAlertChannel(),   # logs to Python logging at WARNING level
    ]
)
```

Alert events: `policy_block`, `cost_warning`, `budget_threshold`, `velocity_trip`.

Delivery failures are logged at WARNING level and never propagate to the caller.

---

## Cost Reduction Engine

Enable the optimization pipeline to automatically trim prompts, cap output, and fall back to cheaper models:

```python
from driftlock import DriftlockClient, OptimizationConfig

client = DriftlockClient(
    api_key="sk-...",
    optimization=OptimizationConfig(
        max_prompt_tokens=3000,          # trim history if prompt exceeds this
        keep_last_n_messages=10,         # always keep the N most recent turns
        always_keep_system=True,         # never drop the system message
        default_max_output_tokens=512,   # cap output when caller omits max_tokens
        max_cost_per_request_usd=0.05,   # abort if estimated cost > $0.05
        budget_exceeded_action="fallback",
        fallback_model="gpt-4o-mini",
    ),
)
```

Every call logs an `optimization` block showing tokens and cost saved:

```json
{
  "optimization": {
    "original_prompt_tokens": 3840,
    "optimized_prompt_tokens": 142,
    "tokens_saved": 3698,
    "cost_saved_usd": 0.0005547,
    "optimizations_applied": ["prompt_trim", "output_cap"],
    "quality_risk": true
  }
}
```

---

## Response Cache

Exact in-memory cache (LRU + TTL). Returns stored responses for identical requests without hitting the API:

```python
from driftlock import DriftlockClient, CacheConfig

client = DriftlockClient(
    api_key="sk-...",
    cache=CacheConfig(
        ttl_seconds=600,    # entries expire after 10 minutes
        max_entries=500,    # LRU eviction above this
    ),
)
```

Cache hits report `cost=$0.00` and record tokens and dollars saved. Streaming responses are never cached.

```python
client.cache_stats()
# {"enabled": True, "size": 12, "hits": 48, "misses": 14, "hit_rate": 0.7742}
```

---

## Reading Metrics

```python
# Aggregate stats (all time)
client.stats()
# {'calls': 42, 'total_tokens': 18500, 'total_cost_usd': 0.003245, ...}

# Filter by endpoint, model, or time window
client.stats(endpoint="summarise_article")
client.stats(model="gpt-4o")
client.stats(since="2025-03-01T00:00:00+00:00")

# Recent calls
client.recent_calls(limit=10)

# Projected monthly spend
client.forecast(lookback_days=7)
# {'daily_avg_usd': 0.0004, 'projected_monthly_usd': 0.012, ...}

# Prompt drift detection (detect template changes by endpoint)
client.prompt_drift(endpoint="summarise_article")
```

---

## CLI

Inspect telemetry without writing code:

```bash
driftlock stats                          # aggregate totals
driftlock stats --since 7d              # last 7 days
driftlock stats --endpoint summarise    # filter by endpoint
driftlock recent --limit 20             # last 20 calls
driftlock forecast --lookback 7         # projected monthly spend
driftlock top-endpoints --since 7d      # most expensive endpoints
driftlock top-users --since 30d         # per-user spend
driftlock models                        # spend by model
driftlock drift summarise_article       # prompt change history
driftlock missions                      # recent agent missions + status
driftlock mission <id>                  # per-mission detail + call graph
driftlock --db /path/to/other.db stats  # point at a different db
```

Set `DRIFTLOCK_DB_PATH` to override the default `driftlock.sqlite` path.

---

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `DRIFTLOCK_ENABLED` | `true` | Set to `false` to pass through all calls with zero overhead |
| `DRIFTLOCK_TRACK_ONLY` | `false` | Track metrics but skip optimization and policy enforcement |
| `DRIFTLOCK_DB_PATH` | `driftlock.sqlite` | Override the SQLite file path for CLI commands |

---

## Web Dashboard

A single-page mission-control dashboard ships with the FastAPI example — vanilla
HTML/CSS/JS, no build step, no npm:

```bash
OPENAI_API_KEY=sk-... uvicorn examples.fastapi_app:app --reload
# open http://localhost:8000
```

(Run `python examples/agent_demo.py "any topic"` first — mock mode, no key — so
the dashboard has mission data to show.)

What you see: a dark-theme control panel with a stat bar (spend today / this
month, active missions, calls in the last hour), a live mission feed where each
row shows a status badge (green completed / amber degraded / red killed), a
spend-vs-budget progress bar, call count and duration — click a row and the call
graph expands inline, listing every call's node, model, cost, and latency, with
intervention events highlighted in amber (downgrade) or red (kill). The right
column carries an hourly burn-rate SVG bar chart for the last 24h with the
current hour highlighted, and a top-endpoints spend table. Auto-refreshes every
10 seconds.

The dashboard is pure frontend over the JSON data API (no auth, clean JSON):

```
GET /missions                      # recent missions, paginated
GET /missions/{mission_id}         # full mission stats (spend, direct/nested, model mix)
GET /missions/{mission_id}/calls   # parent/child call graph
GET /metrics/summary               # spend, calls, missions — today / this month
GET /metrics/burn-rate?hours=24    # hourly spend for the last N hours
GET /metrics/top-endpoints         # top endpoints by spend (calls, total, avg, latency)
```

[examples/fastapi_app.py](examples/fastapi_app.py) also shows the full client
integration: middleware tagging, optimization, and cache.

---

## Project Structure

```
driftlock/
├── __init__.py          # Public API
├── client.py            # DriftlockClient (OpenAI wrapper, sync + async)
├── anthropic_client.py  # AnthropicDriftlockClient (opt-in)
├── mission.py           # Mission system — runtime guardrails for agents
├── integrations/        # LangChain handler + LangGraph middleware (opt-in)
├── config.py            # DriftlockConfig
├── policy.py            # PolicyEngine + all rules
├── alerts.py            # AlertChannel, Webhook/Slack/Log implementations
├── metrics.py           # CallMetrics dataclass
├── pricing.py           # OpenAI + Anthropic pricing table
├── storage.py           # SQLiteStorage + NoopStorage (auto-migrating)
├── optimization.py      # OptimizationPipeline, OptimizationConfig
├── cache.py             # ResponseCache (LRU+TTL), CacheConfig
├── streaming.py         # StreamingInterceptor (deferred metrics)
├── drift.py             # Prompt hash + drift detection
├── cli.py               # driftlock CLI entry point
├── context.py           # driftlock.tag() context manager
├── logger.py            # Structured JSON logger
├── tokenizer.py         # tiktoken + char fallback
└── providers/           # NormalizedUsage, OpenAIProvider, AnthropicProvider

examples/
├── basic_usage.py
├── agent_demo.py            # research agent under a mission budget (mock mode built in)
├── langchain_agent_demo.py  # same, via the LangChain callback handler
├── langgraph_agent_demo.py  # ReAct agent with critic loop, via LangGraph middleware
├── fastapi_app.py           # mission dashboard (UI + data API)
├── static/dashboard.html    # single-page mission control (no build step)
└── dashboard_app.py

tests/                   # 235 tests
```

---

## Roadmap

| Feature | Status |
|---|---|
| OpenAI chat wrapper (sync + async) | ✅ |
| Anthropic Messages wrapper (sync + async) | ✅ |
| Token tracking + cost estimation | ✅ |
| Latency measurement | ✅ |
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

MIT
