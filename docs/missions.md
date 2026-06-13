# Mission Budgets

A **mission** wraps a whole agent run (many LLM calls) and intervenes
**mid-execution** — stopping, rerouting, or downgrading a running agent *before*
budget damage compounds. Where the [policy engine](policy-engine.md) judges one
request at a time, a mission budgets the entire run.

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

Every tracked call inside the block is attributed to the mission (via the
ambient tag system). After each call completes, the mission re-evaluates burn
rate and projected final cost; if an overage is projected, the intervention
fires **before the next call is allowed through**.

---

## Intervention modes (`on_exceed`)

| Mode | Behavior |
|---|---|
| `"downgrade"` | Transparently swap the model to `downgrade_to` for all subsequent calls |
| `"kill"` / `"pause"` | Raise `MissionBudgetExceededError` on the next call |
| `"callback"` | Call your `callback(mission)`, which returns `"continue"`, `"downgrade"`, or `"kill"` |

---

## Live properties

`m.spent`, `m.remaining`, `m.direct_spend`, `m.nested_spend`, `m.call_count`,
`m.burn_rate`, `m.projected_final_cost`, `m.projection_confidence`, `m.status`.

---

## Burn-rate projection

The projection uses an exponential weighted moving average (EWMA, α=0.3) of
per-call cost, so recent calls weigh more. It refuses to project from fewer than
3 calls (`projected_final_cost is None` — early calls are too noisy).

- With `expected_calls=N`, the remaining-call count is exact.
- Without it, Driftlock uses the rolling average call count of your last 10
  completed missions (defaulting to 20 when there is no history).

`projection_confidence` is `low` (<5 calls), `medium` (5–14), or `high` (15+).

> An **actual** budget breach always intervenes immediately — the 3-call minimum
> only gates *projection-based* pre-emption.

---

## Soft warnings

Pass `on_warning=callable` (fires **once** per mission when spend or projection
crosses `warning_threshold`, default `0.8` of the budget).

```python
def warn(m):
    print(f"⚠️  {m.spent:.2f}/{m.budget:.2f} spent, projecting {m.projected_final_cost}")

with driftlock.mission("batch", budget_usd=5.00, on_warning=warn, expected_calls=20):
    ...
```

---

## Nested missions (dual attribution)

Missions nest. Every call attributes to its innermost mission *and* propagates
its cost up the whole stack — so a parent mission's budget accounts for all
nested work — while each level evaluates its own budget independently. An inner
budget can be exhausted without killing the outer run; an exhausted outer run
halts everything.

`mission_stats()` reports the `direct_spend` / `nested_spend` split.

---

## Async-safe

The async record path is guarded by an `asyncio.Lock`, so parallel
`asyncio.gather` sub-calls inside one mission accumulate spend without races.

---

## Persistence & recovery

Each mission is written to a `missions` table — `running` on enter, then
`completed` / `degraded` / `killed` / `failed` on exit — so a crashed process
leaves a recoverable record.

```python
summary = client.resume_mission(mission_id)   # read-only MissionSummary
summary.status, summary.spent_usd, summary.nested_spent_usd, summary.over_budget
```

---

## Querying afterwards

```python
client.mission_stats(mission_id)   # spend, direct/nested split, call graph, model mix, interventions
client.missions(limit=20)          # recent missions with status
```

```bash
driftlock missions            # list recent missions (spend, calls, status)
driftlock mission <id>        # per-mission detail + call graph + interventions
```

Mission calls and intervention events are persisted to SQLite as their own
record types, so the full run timeline is queryable later. Existing call
analytics (`stats`, `forecast`, velocity rules) ignore intervention rows.

---

## Framework integrations

- **LangChain** — attach `DriftlockCallbackHandler`; see
  [examples/langchain_agent_demo.py](../examples/langchain_agent_demo.py).
- **LangGraph** — wrap a compiled graph in `DriftlockLangGraphMiddleware` for
  per-node attribution; see
  [examples/langgraph_agent_demo.py](../examples/langgraph_agent_demo.py).
