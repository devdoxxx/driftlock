# Policy Engine

The policy engine enforces governance rules before every API call. Rules are
evaluated in order; the first block raises `PolicyViolationError`.

Policies evaluate a **single request** in isolation. For budgeting a whole
multi-call agent run, see [missions.md](missions.md).

```python
from driftlock import (
    DriftlockClient,
    PolicyEngine,
    MonthlyBudgetRule,
    MaxCostPerRequestRule,
    VelocityLimitRule,
    PolicyViolationError,
)

policy = PolicyEngine(rules=[
    MonthlyBudgetRule(max_usd=100.0),                       # block at $100/month workspace
    MaxCostPerRequestRule(max_usd=0.10),                    # block single calls > $0.10
    VelocityLimitRule(max_requests=60, window_seconds=60),  # 60 req/min circuit breaker
])

client = DriftlockClient(api_key="sk-...", policy=policy)

try:
    response = client.chat.completions.create(...)
except PolicyViolationError as e:
    print(f"Blocked by {e.rule_name}: {e.decision.metadata}")
```

---

## Available rules

| Rule | What it does |
|---|---|
| `MonthlyBudgetRule(max_usd, scope="workspace"\|"user")` | Block once the monthly spend cap is reached |
| `MaxCostPerRequestRule(max_usd)` | Block a single call if estimated cost exceeds the limit |
| `PerUserBudgetRule(user_budgets, default_max_usd)` | Per-user monthly caps from a dict |
| `ForecastBudgetRule(monthly_budget_usd, lookback_days=7)` | Block when projected 30-day spend will exceed budget |
| `VelocityLimitRule(max_requests, window_seconds, scope)` | Circuit breaker on request rate |
| `CostVelocityRule(max_cost_usd, window_seconds)` | Circuit breaker on spend rate (e.g. $5/hour) |
| `RestrictModelRule(disallowed_models, condition=None)` | Block calls to specific models |
| `TagBasedModelDowngradeRule(condition, downgrade_to)` | Silently swap model based on labels |

---

## Per-user budgets

```python
from driftlock import PolicyEngine, PerUserBudgetRule

policy = PolicyEngine(rules=[
    PerUserBudgetRule(
        user_budgets={"power_user": 20.0, "free_tier": 1.0},
        default_max_usd=5.0,   # applied to any user_id not in the dict
    ),
])
# user_id is read from _dl_labels={"user_id": "..."} or ambient tags
```

---

## Forecast-based blocking

```python
from driftlock import PolicyEngine, ForecastBudgetRule

policy = PolicyEngine(rules=[
    ForecastBudgetRule(monthly_budget_usd=50.0, lookback_days=7),
])
# Blocks before the budget is actually exhausted — proactive, not reactive.
```

---

## Model governance

```python
from driftlock import PolicyEngine, RestrictModelRule, TagBasedModelDowngradeRule

policy = PolicyEngine(rules=[
    # Block GPT-4o for free-plan users
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

## Writing a custom rule

Subclass `BaseRule` and return a `RuleDecision`. The request context dict
contains `model`, `messages`, `kwargs`, `labels`, and `storage`.

```python
from driftlock import BaseRule, RuleDecision

class BlockWeekendsRule(BaseRule):
    def evaluate(self, ctx: dict) -> RuleDecision:
        import datetime
        if datetime.datetime.utcnow().weekday() >= 5:
            return RuleDecision(allow=False, action="block",
                                metadata={"reason": "no weekend spend"})
        return RuleDecision(allow=True)
```

Pass an instance into `PolicyEngine(rules=[...])`. See
[`driftlock/policy.py`](../driftlock/policy.py) for the built-in rules as
worked examples.
