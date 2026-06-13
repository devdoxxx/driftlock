# Optimization Pipeline

The optional optimization pipeline reduces spend *before* a call leaves your
process: it trims long prompt history, caps output length, and can fall back to
a cheaper model when an estimated cost crosses a per-request budget.

```python
from driftlock import DriftlockClient, OptimizationConfig

client = DriftlockClient(
    api_key="sk-...",
    optimization=OptimizationConfig(
        max_prompt_tokens=3000,          # trim history if the prompt exceeds this
        keep_last_n_messages=10,         # always keep the N most recent turns
        always_keep_system=True,         # never drop the system message
        default_max_output_tokens=512,   # cap output when the caller omits max_tokens
        max_cost_per_request_usd=0.05,   # abort/fallback if estimated cost > $0.05
        budget_exceeded_action="fallback",
        fallback_model="gpt-4o-mini",
    ),
)
```

---

## OptimizationConfig options

| Field | Effect |
|---|---|
| `max_prompt_tokens` | Trim older messages until the prompt fits this budget |
| `keep_last_n_messages` | Always preserve the N most recent turns when trimming |
| `always_keep_system` | Never drop the system message |
| `default_max_output_tokens` | Apply `max_tokens` when the caller omits it |
| `max_cost_per_request_usd` | Per-request cost ceiling |
| `budget_exceeded_action` | `"raise"` (raise `BudgetExceededError`) or `"fallback"` |
| `fallback_model` | Model to switch to when `budget_exceeded_action="fallback"` |
| `shadow_mode` | Compute savings without modifying the outgoing request |
| `sample_rate` / `sample_key` | Apply optimization to a deterministic fraction of traffic |

---

## What gets logged

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

## Shadow mode & sampling

- **Shadow mode** (`shadow_mode=True`) computes and logs the savings the
  pipeline *would* have produced without changing the request you actually send
  — useful for measuring impact before turning it on.
- **Sampling** (`sample_rate`, `sample_key`) applies optimization to a
  deterministic fraction of traffic keyed on a label (e.g. `user_id`), so a
  given key is consistently in or out of the experiment.

See [`driftlock/optimization.py`](../driftlock/optimization.py) for the
pipeline internals.
