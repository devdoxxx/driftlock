"""
Mission system — runtime financial guardrails for AI agents.

Where the policy engine evaluates a single request in isolation, a *mission*
wraps a whole agent run (potentially many calls) and intervenes **mid-execution**
when the run is on track to blow its budget — stopping, rerouting, or downgrading
a running agent *before* the damage compounds.

Usage::

    import driftlock

    with driftlock.mission(
        "research_task",
        budget_usd=2.00,
        on_exceed="downgrade",
        downgrade_to="gpt-4o-mini",
    ) as m:
        agent.run("do the thing")
        print(m.spent, m.remaining)

A ``MissionContext`` propagates its ``mission_id`` through the ambient tag system
(see :mod:`driftlock.context`) so every tracked call made inside the block is
attributed to the mission. After each call completes, the mission re-evaluates
burn rate and projected final cost and fires the configured intervention before
the next call is allowed through.

Phase 2 hardening:
  - Burn-rate projection uses an EWMA (alpha=0.3) and refuses to project from
    fewer than 3 completed calls (too noisy).
  - Nested missions get *dual attribution*: every call increments its innermost
    mission and propagates its cost up the whole mission stack; each level
    evaluates its own budget independently.
  - Async record path is guarded by an ``asyncio.Lock`` so parallel
    ``asyncio.gather`` sub-calls cannot race on ``_spent`` / ``_call_count``.
  - Mission lifecycle is persisted to SQLite so a crashed run is recoverable.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .context import push_tags, reset_tags
from .policy import PolicyViolationError, RuleDecision

# Stack of active missions for the current context (supports nesting).
_active_missions: ContextVar[tuple["MissionContext", ...]] = ContextVar(
    "driftlock_missions", default=()
)

# Most-recently-constructed client storage, used as the default persistence
# target for missions created via the bare ``driftlock.mission(...)`` factory
# (which has no client handle of its own). Best-effort; never required for the
# in-memory guardrail behaviour to be correct.
_default_storage: Any = None

_EWMA_ALPHA = 0.3
_MIN_CALLS_TO_PROJECT = 3
_DEFAULT_ASSUMED_REMAINING = 20


def register_default_storage(storage: Any) -> None:
    """Register the storage a bare mission should persist its lifecycle to."""
    global _default_storage
    _default_storage = storage


def current_mission() -> "MissionContext | None":
    """Return the innermost active mission in this context, or None."""
    stack = _active_missions.get()
    return stack[-1] if stack else None


def enforce_before_call(kwargs: dict) -> None:
    """
    Enforce every active mission's guardrail before a call goes out.

    Walks the stack outer→inner so an exhausted *outer* mission can kill a run
    even while an inner mission still has budget, and the innermost mission wins
    any model-downgrade swap.
    """
    for m in _active_missions.get():
        m._before_call(kwargs)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ #
# Errors
# ------------------------------------------------------------------ #

class MissionBudgetExceededError(PolicyViolationError):
    """
    Raised on the next call attempt after a mission's budget guardrail has
    tripped in ``kill``/``pause`` mode.

    Subclasses :class:`~driftlock.policy.PolicyViolationError` so existing
    handlers (and the CLI) that catch policy blocks treat it uniformly — it
    exposes ``rule_name`` and ``decision`` like any other policy violation.
    """

    def __init__(self, mission_id: str, metadata: dict | None = None) -> None:
        self.mission_id = mission_id
        decision = RuleDecision(allow=False, action="block", metadata=metadata or {})
        super().__init__(f"MissionBudget:{mission_id}", decision)


# ------------------------------------------------------------------ #
# Intervention status helpers (shared by storage / stats)
# ------------------------------------------------------------------ #

def _derive_status(interventions: list[dict]) -> str:
    actions = {e.get("action") for e in interventions}
    if actions & {"kill", "pause"}:
        return "killed"
    if "downgrade" in actions:
        return "degraded"
    return "completed"


# ------------------------------------------------------------------ #
# MissionContext
# ------------------------------------------------------------------ #

class MissionContext:
    """
    Context manager that turns an agent run into a budgeted mission.

    See the module docstring for usage. The live properties (:attr:`spent`,
    :attr:`remaining`, :attr:`call_count`, :attr:`projected_final_cost`) update
    as calls complete inside the block.
    """

    def __init__(
        self,
        name: str,
        *,
        budget_usd: float,
        on_exceed: str = "kill",
        on_warning: Callable[["MissionContext"], Any] | None = None,
        warning_threshold: float = 0.8,
        downgrade_to: str | None = None,
        expected_calls: int | None = None,
        callback: Callable[["MissionContext"], str] | None = None,
        mission_id: str | None = None,
    ) -> None:
        if on_exceed not in {"downgrade", "pause", "kill", "callback"}:
            raise ValueError(
                f"on_exceed must be one of downgrade/pause/kill/callback, got {on_exceed!r}"
            )
        self.name = name
        self.mission_id = mission_id or f"mission_{uuid.uuid4().hex[:16]}"
        self.budget = float(budget_usd)
        self._on_exceed = on_exceed
        self._warning_callback = on_warning
        self._warning_threshold = warning_threshold
        self._downgrade_to = downgrade_to
        self._expected_calls = expected_calls
        # ``callback`` is an alias-friendly way to supply the on_exceed handler;
        # on_exceed="callback" with no callable is a no-op (treated as "continue").
        self._callback = callback

        # Concurrency guards. ``_lock`` protects sync mutation (and nested
        # propagation); ``_async_lock`` serialises the async record path so
        # asyncio.gather sub-calls cannot interleave on shared counters.
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

        # Live state
        self._spent = 0.0
        self._nested_spent = 0.0
        self._call_count = 0
        self._ewma_cost: float | None = None
        self._calls: list[dict] = []
        self._events: list[dict] = []

        # Intervention flags
        self._warning_fired = False
        self._intervened = False
        self._degraded = False
        self._killed = False

        # Nesting / lifecycle
        self._parents: tuple["MissionContext", ...] = ()
        self._parent_mission_id: str | None = None
        self._storage: Any = None
        self._tag_token = None
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._final_status: str | None = None
        self._persisted_start = False

    # -------------------------------------------------------------- #
    # Context manager protocol
    # -------------------------------------------------------------- #

    def __enter__(self) -> "MissionContext":
        # Propagate mission_id through the ambient tag system so every call
        # inside the block is attributable to this mission.
        self._tag_token = push_tags(mission_id=self.mission_id)
        self._parents = _active_missions.get()
        self._parent_mission_id = (
            self._parents[-1].mission_id if self._parents else None
        )
        _active_missions.set(self._parents + (self,))
        self._started_at = _now_iso()
        if self._storage is None:
            self._storage = _default_storage
        self._persist_start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        stack = _active_missions.get()
        if stack and stack[-1] is self:
            _active_missions.set(stack[:-1])
        if self._tag_token is not None:
            reset_tags(self._tag_token)
            self._tag_token = None
        self._ended_at = _now_iso()
        if exc_type is None:
            status = self.status
        elif isinstance(exc, MissionBudgetExceededError) and self._killed:
            # Our own kill propagating out of the block is the guardrail working
            # as intended — finalize as killed, not failed.
            status = "killed"
        else:
            status = "failed"
        self._final_status = status
        self._persist_finalize(status)
        return False  # never suppress exceptions

    # -------------------------------------------------------------- #
    # Live properties
    # -------------------------------------------------------------- #

    @property
    def spent(self) -> float:
        return round(self._spent, 6)

    @property
    def remaining(self) -> float:
        return round(self.budget - self._spent, 6)

    @property
    def nested_spend(self) -> float:
        """Spend that flowed up from nested missions."""
        return round(self._nested_spent, 6)

    @property
    def direct_spend(self) -> float:
        """Spend from calls made directly in this mission (excludes nested)."""
        return round(self._spent - self._nested_spent, 6)

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def burn_rate(self) -> float:
        """EWMA cost per call (alpha=0.3); 0 until the first call completes."""
        return round(self._ewma_cost, 6) if self._ewma_cost is not None else 0.0

    @property
    def estimated_calls_remaining(self) -> int | None:
        if self._expected_calls is None:
            return None
        return max(0, self._expected_calls - self._call_count)

    @property
    def projection_confidence(self) -> str:
        """``low`` (<5 calls), ``medium`` (5–14), ``high`` (15+)."""
        n = self._call_count
        if n < 5:
            return "low"
        if n < 15:
            return "medium"
        return "high"

    @property
    def projected_final_cost(self) -> float | None:
        """
        Project the mission's final spend, or ``None`` if fewer than 3 calls
        have completed (early calls are too noisy to extrapolate from).

        Uses an EWMA of per-call cost (recent calls weigh more). With an
        ``expected_calls`` hint the remaining-call count is exact; otherwise it
        falls back to the rolling average call count of the last 10 completed
        missions (default 20 when there is no history).
        """
        if self._call_count < _MIN_CALLS_TO_PROJECT:
            return None
        ewma = self._ewma_cost or 0.0
        if self._expected_calls is not None:
            remaining = max(0, self._expected_calls - self._call_count)
        else:
            remaining = self._rolling_avg_remaining()
        return round(self._spent + remaining * ewma, 6)

    @property
    def status(self) -> str:
        if self._killed:
            return "killed"
        if self._degraded:
            return "degraded"
        return "completed"

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def _rolling_avg_remaining(self) -> float:
        avg = None
        if self._storage is not None:
            try:
                avg = self._storage.avg_calls_per_mission(limit=10)
            except Exception:  # pragma: no cover - storage failures are non-fatal
                avg = None
        return avg if avg else _DEFAULT_ASSUMED_REMAINING

    # -------------------------------------------------------------- #
    # Persistence (best-effort; never breaks the run)
    # -------------------------------------------------------------- #

    def _mission_record(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "name": self.name,
            "budget_usd": self.budget,
            "expected_calls": self._expected_calls,
            "on_exceed": self._on_exceed,
            "downgrade_to": self._downgrade_to,
            "parent_mission_id": self._parent_mission_id,
            "started_at": self._started_at,
        }

    def _persist_start(self) -> None:
        if self._storage is None or self._persisted_start:
            return
        try:
            self._storage.start_mission(self._mission_record())
            self._persisted_start = True
        except Exception:  # pragma: no cover - persistence is best-effort
            pass

    def _persist_finalize(self, status: str) -> None:
        if self._storage is None:
            self._storage = _default_storage
        if self._storage is None:
            return
        record = self._mission_record()
        record.update(
            {
                "ended_at": self._ended_at,
                "status": status,
                "final_spent": round(self._spent, 6),
                "final_call_count": self._call_count,
                "nested_spent_usd": round(self._nested_spent, 6),
            }
        )
        try:
            self._storage.finalize_mission(record)
        except Exception:  # pragma: no cover - persistence is best-effort
            pass

    # -------------------------------------------------------------- #
    # Pipeline hooks (called by the client wrappers)
    # -------------------------------------------------------------- #

    def _before_call(self, kwargs: dict) -> None:
        """
        Enforce the mission's current intervention state *before* a call is
        allowed through. Called at the very start of the tracked pipeline.
        """
        if self._killed:
            raise MissionBudgetExceededError(
                self.mission_id,
                metadata={
                    "spent_usd": self.spent,
                    "budget_usd": self.budget,
                    "call_count": self._call_count,
                    "action": self._on_exceed if self._on_exceed != "callback" else "kill",
                },
            )
        if self._degraded and self._downgrade_to:
            kwargs["model"] = self._downgrade_to

    def _record_call(self, metrics: Any, storage: Any = None) -> None:
        """Sync record path. Mutates under the threading lock."""
        with self._lock:
            self._apply_call(metrics, storage)

    async def _arecord_call(self, metrics: Any, storage: Any = None) -> None:
        """
        Async record path. The ``asyncio.Lock`` guarantees that parallel
        ``asyncio.gather`` sub-calls in one event loop serialise their mutation
        of ``_spent`` / ``_call_count`` / intervention flags — no races.
        """
        async with self._async_lock:
            self._apply_call(metrics, storage)

    def _apply_call(self, metrics: Any, storage: Any) -> None:
        """Shared record logic (no awaits → atomic on a single event loop)."""
        if storage is not None and self._storage is None:
            self._storage = storage
            self._persist_start()
        cost = metrics.estimated_cost_usd or 0.0
        self._spent += cost
        self._call_count += 1
        self._update_ewma(cost)
        self._calls.append(
            {
                "call_id": metrics.request_id,
                "parent_call_id": metrics.parent_call_id,
                "model": metrics.model,
                "cost_usd": cost,
            }
        )
        self._evaluate(metrics.request_id)
        # Dual attribution: propagate the cost up the full mission stack so a
        # parent mission's budget accounts for all nested work.
        for parent in self._parents:
            parent._absorb_nested(cost, metrics)

    def _absorb_nested(self, cost: float, metrics: Any) -> None:
        """Record a nested call's cost against this (ancestor) mission."""
        with self._lock:
            self._spent += cost
            self._nested_spent += cost
            self._call_count += 1
            self._update_ewma(cost)
            self._evaluate(metrics.request_id)

    def _update_ewma(self, cost: float) -> None:
        if self._ewma_cost is None:
            self._ewma_cost = cost
        else:
            self._ewma_cost = _EWMA_ALPHA * cost + (1 - _EWMA_ALPHA) * self._ewma_cost

    # -------------------------------------------------------------- #
    # Internal evaluation / intervention engine
    # -------------------------------------------------------------- #

    def _evaluate(self, call_id: str | None) -> None:
        projected = self.projected_final_cost  # None until 3+ calls

        # Soft warning — fires at most once per mission.
        if not self._warning_fired and self._warning_callback is not None:
            threshold = self.budget * self._warning_threshold
            crossed = self._spent >= threshold or (
                projected is not None and projected >= threshold
            )
            if crossed:
                self._warning_fired = True
                _safe_call(self._warning_callback, self)

        if self._intervened:
            return

        # An actual budget breach is a fact, not a projection — it arms at any
        # call count. Projection-based pre-emption requires 3+ calls.
        actual_breach = self._spent > self.budget
        projected_breach = projected is not None and projected > self.budget
        if not (actual_breach or projected_breach):
            return

        if projected_breach and not actual_breach:
            reason = (
                f"projected_final_cost ${projected:.6f} exceeds budget "
                f"${self.budget:.6f}"
            )
        else:
            reason = f"spent ${self._spent:.6f} exceeds budget ${self.budget:.6f}"
        self._trigger(reason, call_id)

    def _trigger(self, reason: str, call_id: str | None) -> None:
        mode = self._on_exceed
        if mode == "callback":
            decision = _safe_call(self._callback, self) if self._callback else "continue"
            action = decision if decision in {"continue", "downgrade", "kill"} else "continue"
            if action == "continue":
                # User opted to let it ride; re-evaluated after the next call.
                return
        else:
            action = mode

        self._apply_action(action, reason, call_id)

    def _apply_action(self, action: str, reason: str, call_id: str | None) -> None:
        self._intervened = True
        if action == "downgrade":
            self._degraded = True
        elif action in {"kill", "pause"}:
            self._killed = True

        event = {
            "action": action,
            "reason": reason,
            "call_id": call_id,
            "mission_id": self.mission_id,
            "spent_usd": round(self._spent, 6),
            "budget_usd": self.budget,
            "projected_final_cost_usd": self.projected_final_cost,
            "call_count": self._call_count,
            "downgrade_to": self._downgrade_to if action == "downgrade" else None,
        }
        self._events.append(event)
        if self._storage is not None:
            try:
                self._storage.save_intervention(self.mission_id, event)
            except Exception:  # pragma: no cover - persistence is best-effort
                pass


def _safe_call(fn: Callable | None, mission: "MissionContext"):
    """Invoke a user callback, swallowing exceptions (never break the run)."""
    if fn is None:
        return None
    try:
        return fn(mission)
    except Exception:  # pragma: no cover - user callback failures are non-fatal
        return None


# ------------------------------------------------------------------ #
# Read-only summary for post-run analysis
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class MissionSummary:
    """Read-only snapshot of a completed (or crashed) mission from storage."""

    mission_id: str
    name: str | None
    status: str
    budget_usd: float | None
    spent_usd: float | None
    call_count: int | None
    nested_spent_usd: float | None
    started_at: str | None
    ended_at: str | None
    stats: dict = field(default_factory=dict)

    @property
    def over_budget(self) -> bool:
        if self.budget_usd is None or self.spent_usd is None:
            return False
        return self.spent_usd > self.budget_usd


def resume_mission(storage: Any, mission_id: str) -> MissionSummary | None:
    """Build a :class:`MissionSummary` from persisted mission data, or None."""
    row = storage.get_mission(mission_id)
    stats = build_mission_stats(storage, mission_id)
    if row is None and stats["calls"] == 0:
        return None
    row = row or {}
    return MissionSummary(
        mission_id=mission_id,
        name=row.get("name"),
        status=row.get("status") or stats.get("status", "unknown"),
        budget_usd=row.get("budget_usd"),
        spent_usd=row.get("final_spent")
        if row.get("final_spent") is not None
        else stats.get("total_cost_usd"),
        call_count=row.get("final_call_count")
        if row.get("final_call_count") is not None
        else stats.get("calls"),
        nested_spent_usd=row.get("nested_spent_usd"),
        started_at=row.get("started_at"),
        ended_at=row.get("ended_at"),
        stats=stats,
    )


# ------------------------------------------------------------------ #
# Public factory
# ------------------------------------------------------------------ #

def mission(
    name: str,
    *,
    budget_usd: float,
    on_exceed: str = "kill",
    on_warning: Callable[["MissionContext"], Any] | None = None,
    warning_threshold: float = 0.8,
    downgrade_to: str | None = None,
    expected_calls: int | None = None,
    callback: Callable[["MissionContext"], str] | None = None,
    mission_id: str | None = None,
) -> MissionContext:
    """
    Create a :class:`MissionContext` (use as a context manager).

    Args:
        name: Human-readable label for the mission.
        budget_usd: Hard spend ceiling for the whole run.
        on_exceed: Intervention mode — ``"downgrade"`` (swap to ``downgrade_to``
            for subsequent calls), ``"kill"`` / ``"pause"`` (raise
            :class:`MissionBudgetExceededError` on the next call), or
            ``"callback"`` (delegate the decision to ``callback``).
        on_warning: Callable invoked once when spend/projection crosses
            ``warning_threshold`` of the budget. Receives the mission.
        warning_threshold: Fraction of the budget that triggers ``on_warning``
            (default 0.8).
        downgrade_to: Model string used when downgrading.
        expected_calls: Optional hint for how many calls the run will make,
            used to sharpen :attr:`MissionContext.projected_final_cost`.
        callback: Handler for ``on_exceed="callback"``; returns ``"continue"``,
            ``"downgrade"``, or ``"kill"``.
        mission_id: Override the generated mission id (mainly for testing).
    """
    return MissionContext(
        name,
        budget_usd=budget_usd,
        on_exceed=on_exceed,
        on_warning=on_warning,
        warning_threshold=warning_threshold,
        downgrade_to=downgrade_to,
        expected_calls=expected_calls,
        callback=callback,
        mission_id=mission_id,
    )


# ------------------------------------------------------------------ #
# Stats assembly (shared by client.mission_stats and the CLI)
# ------------------------------------------------------------------ #

def _build_call_tree(calls: list[dict]) -> list[dict]:
    """Assemble a parent/child call tree from flat call rows."""
    nodes: dict[str, dict] = {}
    for c in calls:
        cid = c.get("request_id") or c.get("call_id")
        if cid is None:
            continue
        nodes[cid] = {
            "call_id": cid,
            "parent_call_id": c.get("parent_call_id"),
            "model": c.get("model"),
            "cost_usd": c.get("estimated_cost_usd", c.get("cost_usd")),
            "endpoint": c.get("endpoint"),
            "latency_ms": c.get("latency_ms"),
            "timestamp": c.get("timestamp"),
            "children": [],
        }
    roots: list[dict] = []
    for node in nodes.values():
        pid = node["parent_call_id"]
        if pid and pid in nodes:
            nodes[pid]["children"].append(node)
        else:
            roots.append(node)
    return roots


def build_mission_stats(storage: Any, mission_id: str) -> dict:
    """
    Assemble an aggregate report for a mission from persisted records:
    total spend (incl. nested), direct vs nested split, call count, call graph,
    model distribution, and the list of intervention events.
    """
    calls = storage.mission_calls(mission_id)
    interventions = storage.mission_interventions(mission_id)
    try:
        row = storage.get_mission(mission_id)
    except Exception:  # pragma: no cover - older storage backends
        row = None

    direct_cost = sum(c.get("estimated_cost_usd") or 0.0 for c in calls)
    total_tokens = sum(c.get("total_tokens") or 0 for c in calls)

    if row is not None:
        nested = row.get("nested_spent_usd") or 0.0
        total = row.get("final_spent")
        if total is None:
            total = direct_cost + nested
        status = row.get("status") or _derive_status(interventions)
    else:
        nested = 0.0
        total = direct_cost
        status = _derive_status(interventions)

    dist: dict[str, dict] = {}
    for c in calls:
        m = c.get("model") or "unknown"
        d = dist.setdefault(m, {"model": m, "calls": 0, "total_cost_usd": 0.0})
        d["calls"] += 1
        d["total_cost_usd"] += c.get("estimated_cost_usd") or 0.0
    model_distribution = sorted(
        dist.values(), key=lambda x: x["total_cost_usd"], reverse=True
    )
    for d in model_distribution:
        d["total_cost_usd"] = round(d["total_cost_usd"], 6)

    return {
        "mission_id": mission_id,
        "name": row.get("name") if row else None,
        "calls": len(calls),
        "total_cost_usd": round(total, 6),
        "direct_spend": round(total - nested, 6),
        "nested_spend": round(nested, 6),
        "total_tokens": total_tokens,
        "budget_usd": row.get("budget_usd") if row else None,
        "status": status,
        "model_distribution": model_distribution,
        "interventions": interventions,
        "intervention_count": len(interventions),
        "call_graph": _build_call_tree(calls),
    }
