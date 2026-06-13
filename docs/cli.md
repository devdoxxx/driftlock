# CLI Reference

Inspect Driftlock telemetry from the command line — no code required. The CLI
reads the same SQLite database the client writes to.

```bash
driftlock <command> [options]
```

By default it reads `./driftlock.sqlite`. Override with `--db PATH` or the
`DRIFTLOCK_DB_PATH` environment variable.

---

## Commands

| Command | Description |
|---|---|
| `driftlock demo` | Make one cheap real call and print a cost receipt (needs an API key) |
| `driftlock stats [--since 7d] [--endpoint NAME] [--model NAME]` | Aggregate cost / token totals |
| `driftlock recent [--limit N]` | Most recent tracked calls |
| `driftlock forecast [--lookback DAYS]` | Project end-of-month spend |
| `driftlock top-endpoints [--limit N] [--since 7d]` | Most expensive endpoints |
| `driftlock top-users [--limit N] [--since 7d]` | Per-user spend breakdown |
| `driftlock models [--since 7d]` | Spend by model |
| `driftlock drift ENDPOINT [--limit N]` | Prompt-template change history |
| `driftlock missions [--limit N] [--since 7d]` | Recent agent missions + status |
| `driftlock mission <id>` | Per-mission detail + call graph + interventions |
| `driftlock --db PATH <command>` | Point at a different database |

---

## Examples

```bash
driftlock stats                          # aggregate totals
driftlock stats --since 7d               # last 7 days
driftlock stats --endpoint summarise     # filter by endpoint
driftlock recent --limit 20              # last 20 calls
driftlock forecast --lookback 7          # projected monthly spend
driftlock top-endpoints --since 7d       # most expensive endpoints
driftlock top-users --since 30d          # per-user spend
driftlock models                         # spend by model
driftlock drift summarise_article        # prompt change history
driftlock missions                       # recent agent missions
driftlock mission mission_134ee6f0       # one mission's full detail
driftlock --db /path/to/other.sqlite stats
```

---

## `--since` durations

`--since` accepts a human-readable duration or an ISO timestamp:

| Value | Meaning |
|---|---|
| `7d` | last 7 days |
| `24h` | last 24 hours |
| `30m` | last 30 minutes |
| `2025-03-01T00:00:00+00:00` | from an explicit ISO timestamp |
