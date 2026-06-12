"""
Driftlock CLI — inspect local telemetry without writing any code.

Usage:
    driftlock stats [--since 7d] [--endpoint NAME] [--model NAME]
    driftlock recent [--limit N]
    driftlock top-endpoints [--limit N] [--since 7d]
    driftlock top-users [--limit N] [--since 7d]
    driftlock models [--since 7d]
    driftlock forecast [--lookback N]
    driftlock drift ENDPOINT [--limit N]
    driftlock missions [--limit N] [--since 7d]
    driftlock mission MISSION_ID
    driftlock --db PATH <subcommand>

Environment:
    DRIFTLOCK_DB_PATH   Override the default driftlock.sqlite path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _db_path(args: argparse.Namespace) -> str:
    return args.db or os.environ.get("DRIFTLOCK_DB_PATH", "driftlock.sqlite")


def _since_str(since: str | None) -> str | None:
    """Convert a human duration like '7d', '24h', '1h' to an ISO timestamp."""
    if since is None:
        return None
    since = since.strip().lower()
    try:
        if since.endswith("d"):
            delta = timedelta(days=int(since[:-1]))
        elif since.endswith("h"):
            delta = timedelta(hours=int(since[:-1]))
        elif since.endswith("m"):
            delta = timedelta(minutes=int(since[:-1]))
        else:
            # Assume ISO timestamp passed directly
            return since
        return (datetime.now(timezone.utc) - delta).isoformat()
    except ValueError:
        return since  # pass through and let SQLite handle it


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _get_storage(db: str):
    if not Path(db).exists():
        print(f"Error: database not found at '{db}'", file=sys.stderr)
        sys.exit(1)
    from .storage import SQLiteStorage
    return SQLiteStorage(db_path=db)


# ------------------------------------------------------------------ #
# Subcommand handlers
# ------------------------------------------------------------------ #

def cmd_stats(args: argparse.Namespace) -> None:
    storage = _get_storage(_db_path(args))
    result = storage.aggregate(
        endpoint=args.endpoint or None,
        model=args.model or None,
        since=_since_str(args.since),
    )
    _print_json(result)


def cmd_recent(args: argparse.Namespace) -> None:
    storage = _get_storage(_db_path(args))
    rows = storage.recent(limit=args.limit)
    _print_json(rows)


def cmd_top_endpoints(args: argparse.Namespace) -> None:
    storage = _get_storage(_db_path(args))
    since = _since_str(args.since)
    # Re-use aggregate grouped by endpoint by reading recent + grouping in Python
    # (simpler than duplicating a new SQL query; fine for CLI use)
    rows = storage.recent(limit=10_000)
    if since:
        rows = [r for r in rows if r.get("timestamp", "") >= since]
    groups: dict[str, dict] = {}
    for r in rows:
        ep = r.get("endpoint") or "(none)"
        if ep not in groups:
            groups[ep] = {"endpoint": ep, "calls": 0, "total_cost_usd": 0.0}
        groups[ep]["calls"] += 1
        groups[ep]["total_cost_usd"] += r.get("estimated_cost_usd") or 0.0
    result = sorted(groups.values(), key=lambda x: x["total_cost_usd"], reverse=True)
    for r in result:
        r["total_cost_usd"] = round(r["total_cost_usd"], 6)
    _print_json(result[: args.limit])


def cmd_top_users(args: argparse.Namespace) -> None:
    storage = _get_storage(_db_path(args))
    result = storage.top_users(since=_since_str(args.since), limit=args.limit)
    _print_json(result)


def cmd_models(args: argparse.Namespace) -> None:
    storage = _get_storage(_db_path(args))
    result = storage.model_distribution(since=_since_str(args.since))
    _print_json(result)


def cmd_forecast(args: argparse.Namespace) -> None:
    storage = _get_storage(_db_path(args))
    trend = storage.daily_cost_trend(lookback_days=args.lookback)
    if not trend:
        _print_json({"error": "No data in lookback window", "lookback_days": args.lookback})
        return
    total = sum(d["cost_usd"] for d in trend)
    daily_avg = total / args.lookback
    _print_json(
        {
            "daily_avg_usd": round(daily_avg, 6),
            "projected_monthly_usd": round(daily_avg * 30, 4),
            "lookback_days": args.lookback,
            "data_points": len(trend),
            "daily_trend": trend,
        }
    )


def _resolve_demo_db_path() -> str:
    """Return a writable DB path: ./driftlock.sqlite or ~/.driftlock/driftlock.sqlite."""
    try:
        probe = Path(".driftlock_write_probe")
        probe.touch()
        probe.unlink()
        return str(Path("driftlock.sqlite"))
    except OSError:
        fallback = Path.home() / ".driftlock"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback / "driftlock.sqlite")


def _receipt_line(label: str, value: str, width: int = 43) -> str:
    padded = value[: width].ljust(width)
    return f"  │  {label:<10}: {padded}│"


def _print_demo_receipt(m: dict, db_path: str) -> None:
    cost = m.get("estimated_cost_usd") or 0.0
    total_tok = m.get("total_tokens") or 0
    in_tok = m.get("prompt_tokens") or 0
    out_tok = m.get("completion_tokens") or 0
    latency = m.get("latency_ms") or 0.0
    print("  ┌─ Receipt ──────────────────────────────────────────────┐")
    print(_receipt_line("provider", m.get("provider", "unknown")))
    print(_receipt_line("model", m.get("model", "unknown")))
    print(_receipt_line("tokens", f"{total_tok} ({in_tok} in / {out_tok} out)"))
    print(_receipt_line("cost", f"${cost:.6f}"))
    print(_receipt_line("latency", f"{latency:.0f} ms"))
    print(_receipt_line("db", db_path))
    print("  └────────────────────────────────────────────────────────┘")
    print()
    print("  Next steps:")
    print("    driftlock stats            # aggregate cost + token totals")
    print("    driftlock recent           # last 20 calls")
    print("    driftlock forecast         # projected monthly spend")
    print()


def cmd_demo(args: argparse.Namespace) -> None:
    """Zero-config demo: make one cheap request and print a cost receipt."""
    from .alerts import LogAlertChannel
    from .config import DriftlockConfig
    from .policy import (
        CircuitOpenError,
        MaxCostPerRequestRule,
        MonthlyBudgetRule,
        PolicyEngine,
        PolicyViolationError,
        VelocityLimitRule,
    )

    # --- Provider selection ---
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if openai_key:
        provider = "openai"
        model = "gpt-4o-mini"
    elif anthropic_key:
        provider = "anthropic"
        model = "claude-haiku-4-5"
    else:
        print(
            "Error: no API key found.\n"
            "  Set one of:\n"
            "    export OPENAI_API_KEY=sk-...\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  then re-run:  driftlock demo",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Zero-config DB path ---
    db_path = _resolve_demo_db_path()

    # --- Default policy + config ---
    config = DriftlockConfig(
        db_path=db_path,
        log_json=False,
        log_level="WARNING",
        alert_channels=[LogAlertChannel()],
    )
    policy = PolicyEngine([
        MonthlyBudgetRule(max_usd=50.0, scope="workspace"),
        MaxCostPerRequestRule(max_usd=0.05),
        VelocityLimitRule(max_requests=60, window_seconds=60, scope="workspace"),
    ])

    prompt = "Say hi in 5 words."
    messages = [{"role": "user", "content": prompt}]

    print(f"\nDriftlock demo  —  provider={provider}  model={model}")
    print(f"  db: {db_path}\n")

    try:
        if provider == "openai":
            from .client import DriftlockClient

            client = DriftlockClient(api_key=openai_key, config=config, policy=policy)
            client.chat.completions.create(
                model=model,
                messages=messages,
                _dl_endpoint="demo",
            )
            recent = client.recent_calls(limit=1)
        else:
            from .anthropic_client import AnthropicDriftlockClient

            client = AnthropicDriftlockClient(
                api_key=anthropic_key, config=config, policy=policy
            )
            client.messages.create(
                model=model,
                max_tokens=32,
                messages=messages,
                _dl_endpoint="demo",
            )
            recent = client.recent_calls(limit=1)

    except (PolicyViolationError, CircuitOpenError) as exc:
        print(f"Policy blocked the request: {exc.rule_name}")
        for k, v in exc.decision.metadata.items():
            print(f"  {k}: {v}")
        print()
        print("To loosen limits, initialise with a custom policy:")
        print("  MonthlyBudgetRule(max_usd=200, scope='workspace')")
        print("  MaxCostPerRequestRule(max_usd=0.10)")
        print("  VelocityLimitRule(max_requests=120, window_seconds=60)")
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if recent:
        _print_demo_receipt(recent[0], db_path)
    else:
        print("Request succeeded (no metrics stored).")


def cmd_missions(args: argparse.Namespace) -> None:
    storage = _get_storage(_db_path(args))
    rows = storage.list_missions(limit=args.limit, since=_since_str(args.since))
    if not rows:
        print("No missions recorded yet.")
        return
    for r in rows:
        intervened = "yes" if r["interventions"] else "no"
        print(
            f"{r['mission_id']:<28} "
            f"${r['total_cost_usd']:<10.6f} "
            f"calls={r['calls']:<5} "
            f"interventions={intervened:<4} "
            f"status={r['status']}"
        )


def cmd_mission(args: argparse.Namespace) -> None:
    storage = _get_storage(_db_path(args))
    from .mission import build_mission_stats
    stats = build_mission_stats(storage, args.mission_id)
    if stats["calls"] == 0 and stats["intervention_count"] == 0:
        print(f"No records found for mission '{args.mission_id}'.")
        return
    _print_json(stats)


def cmd_drift(args: argparse.Namespace) -> None:
    storage = _get_storage(_db_path(args))
    from .drift import detect_drift
    history = storage.prompt_hash_history(args.endpoint, limit=args.limit)
    changes = detect_drift(history)
    if not changes:
        print(f"No prompt changes detected for endpoint '{args.endpoint}'.")
    else:
        print(f"Detected {len(changes)} prompt change(s) for endpoint '{args.endpoint}':")
        _print_json(changes)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="driftlock",
        description="Inspect Driftlock telemetry from the command line.",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="Path to driftlock.sqlite (default: $DRIFTLOCK_DB_PATH or ./driftlock.sqlite)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # stats
    p_stats = sub.add_parser("stats", help="Aggregate cost / token stats")
    p_stats.add_argument("--since", metavar="DURATION", default=None,
                         help="e.g. 7d, 24h, 1h, or ISO timestamp")
    p_stats.add_argument("--endpoint", default=None)
    p_stats.add_argument("--model", default=None)

    # recent
    p_recent = sub.add_parser("recent", help="Show recent calls")
    p_recent.add_argument("--limit", type=int, default=20)

    # top-endpoints
    p_top = sub.add_parser("top-endpoints", help="Most expensive endpoints")
    p_top.add_argument("--limit", type=int, default=10)
    p_top.add_argument("--since", metavar="DURATION", default=None)

    # top-users
    p_users = sub.add_parser("top-users", help="Per-user cost breakdown")
    p_users.add_argument("--limit", type=int, default=20)
    p_users.add_argument("--since", metavar="DURATION", default=None)

    # models
    p_models = sub.add_parser("models", help="Per-model usage distribution")
    p_models.add_argument("--since", metavar="DURATION", default=None)

    # forecast
    p_forecast = sub.add_parser("forecast", help="Project end-of-month spend")
    p_forecast.add_argument("--lookback", type=int, default=7, metavar="DAYS")

    # drift
    p_drift = sub.add_parser("drift", help="Show prompt template change history")
    p_drift.add_argument("endpoint", help="Endpoint name to inspect")
    p_drift.add_argument("--limit", type=int, default=30)

    # missions
    p_missions = sub.add_parser("missions", help="List recent agent missions")
    p_missions.add_argument("--limit", type=int, default=20)
    p_missions.add_argument("--since", metavar="DURATION", default=None)

    # mission <id>
    p_mission = sub.add_parser("mission", help="Per-mission detail (spend, call graph, interventions)")
    p_mission.add_argument("mission_id", help="Mission id to inspect")

    # demo
    sub.add_parser(
        "demo",
        help="2-minute quickstart: make one cheap request and show a cost receipt",
    )

    args = parser.parse_args(argv)

    dispatch = {
        "stats": cmd_stats,
        "recent": cmd_recent,
        "top-endpoints": cmd_top_endpoints,
        "top-users": cmd_top_users,
        "models": cmd_models,
        "forecast": cmd_forecast,
        "drift": cmd_drift,
        "missions": cmd_missions,
        "mission": cmd_mission,
        "demo": cmd_demo,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
