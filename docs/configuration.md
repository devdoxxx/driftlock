# Configuration & Client Reference

Full reference for `DriftlockClient`, `DriftlockConfig`, labelling, ambient
tagging, alerts, cache, and reading metrics. For the policy engine see
[policy-engine.md](policy-engine.md); for the optimization pipeline see
[optimization.md](optimization.md).

---

## DriftlockConfig

```python
from driftlock import DriftlockClient, DriftlockConfig

config = DriftlockConfig(
    log_json=True,                         # JSON logs (default). False = human-readable.
    log_level="INFO",
    storage_backend="sqlite",              # "sqlite" | "none"
    db_path="driftlock.sqlite",
    prompt_token_warning_threshold=4000,   # warn if a prompt exceeds N tokens
    cost_warning_threshold=0.10,           # warn if a single call costs > $X
    default_labels={"env": "prod"},        # attached to every tracked call
)

client = DriftlockClient(api_key="sk-...", config=config)
```

| Field | Default | Effect |
|---|---|---|
| `log_json` | `True` | JSON structured logs; `False` for human-readable |
| `log_level` | `"INFO"` | Standard Python log level |
| `storage_backend` | `"sqlite"` | `"sqlite"` persists to disk; `"none"` disables storage |
| `db_path` | `"driftlock.sqlite"` | SQLite file path |
| `prompt_token_warning_threshold` | `4000` | Warn when a prompt exceeds this many tokens |
| `cost_warning_threshold` | `None` | Warn when a single call costs more than `$X` |
| `default_labels` | `{}` | Labels attached to every tracked call |
| `alert_channels` | `[]` | See [Alerts](#alerts) |

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `DRIFTLOCK_ENABLED` | `true` | Set to `false` to pass through all calls with zero overhead |
| `DRIFTLOCK_TRACK_ONLY` | `false` | Track metrics but skip optimization and policy enforcement |
| `DRIFTLOCK_DB_PATH` | `driftlock.sqlite` | Override the SQLite file path for CLI commands |

---

## Basic integration — OpenAI

```python
from driftlock import DriftlockClient

# Drop-in for openai.OpenAI(); all other kwargs are forwarded unchanged.
client = DriftlockClient(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Every call is logged, costed, and saved to local SQLite:

```json
{
  "level": "INFO",
  "logger": "driftlock",
  "message": "model=gpt-4o-mini | tokens=157 | latency=421ms | cost=$0.000033",
  "metrics": {
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

## Basic integration — Anthropic

Requires `pip install "driftlock[anthropic]"`.

```python
from driftlock import AnthropicDriftlockClient

client = AnthropicDriftlockClient(api_key="sk-ant-...")

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

`max_tokens` is required by Anthropic. The `system` parameter is a top-level
kwarg, not a message role — same as the native SDK.

---

## Labelling calls

Use `_dl_endpoint` and `_dl_labels` to annotate individual calls. Both are
stripped before the request reaches the provider.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    _dl_endpoint="summarise_article",                 # logical function name
    _dl_labels={"user_id": "u_123", "team": "growth"},
)
```

`user_id` and `team_id` in labels are indexed in SQLite for fast per-user queries.

---

## Ambient tagging

Attach labels to every call within a scope without touching each call site —
useful in middleware:

```python
import driftlock

with driftlock.tag(request_id="req_abc", user_id="u_42", feature="chat"):
    response = client.chat.completions.create(...)
```

Nested `driftlock.tag()` blocks merge; inner values override outer ones. Per-call
`_dl_labels` always wins.

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

## Response cache

Exact in-memory cache (LRU + TTL). Returns stored responses for identical
requests without hitting the API:

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

Cache hits report `cost=$0.00` and record tokens and dollars saved. Streaming
responses are never cached.

```python
client.cache_stats()
# {"enabled": True, "size": 12, "hits": 48, "misses": 14, "hit_rate": 0.7742}
```

---

## Reading metrics

```python
client.stats()                                   # aggregate, all time
client.stats(endpoint="summarise_article")       # filter by endpoint
client.stats(model="gpt-4o")
client.stats(since="2025-03-01T00:00:00+00:00")  # time window
client.recent_calls(limit=10)                    # most recent calls
client.forecast(lookback_days=7)                 # projected monthly spend
client.prompt_drift(endpoint="summarise_article")  # detect template changes
```

See [cli.md](cli.md) to read the same data from the command line.
