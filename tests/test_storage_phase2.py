"""
Direct tests for the Phase 2 storage additions: the missions lifecycle table,
projection helpers, and dashboard aggregation queries.
"""

from datetime import datetime, timezone

import pytest

from driftlock.storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(db_path=str(tmp_path / "s.db"))


def _start(storage, mid, **kw):
    rec = {"mission_id": mid, "name": kw.get("name", mid), "budget_usd": kw.get("budget", 1.0),
           "expected_calls": kw.get("expected_calls"), "on_exceed": kw.get("on_exceed", "kill"),
           "downgrade_to": kw.get("downgrade_to"), "parent_mission_id": kw.get("parent"),
           "started_at": kw.get("started_at", datetime.now(timezone.utc).isoformat())}
    storage.start_mission(rec)
    return rec


def _finalize(storage, rec, status, spent, calls, nested=0.0):
    rec = dict(rec)
    rec.update({"status": status, "ended_at": datetime.now(timezone.utc).isoformat(),
                "final_spent": spent, "final_call_count": calls, "nested_spent_usd": nested})
    storage.finalize_mission(rec)


def test_start_and_get_mission(storage):
    _start(storage, "m1")
    row = storage.get_mission("m1")
    assert row["status"] == "running"
    assert row["budget_usd"] == 1.0
    assert row["final_spent"] is None


def test_finalize_updates_row(storage):
    rec = _start(storage, "m2")
    _finalize(storage, rec, "completed", 0.42, 5)
    row = storage.get_mission("m2")
    assert row["status"] == "completed"
    assert row["final_spent"] == pytest.approx(0.42)
    assert row["final_call_count"] == 5


def test_finalize_without_start_inserts(storage):
    # finalize_mission must upsert if the start row was never written.
    rec = {"mission_id": "m3", "name": "x", "budget_usd": 2.0, "started_at": "t0"}
    _finalize(storage, rec, "failed", 1.1, 3)
    row = storage.get_mission("m3")
    assert row is not None
    assert row["status"] == "failed"
    assert row["final_spent"] == pytest.approx(1.1)


def test_avg_calls_per_mission(storage):
    assert storage.avg_calls_per_mission() is None  # no completed missions
    for i, calls in enumerate([4, 6, 8]):
        rec = _start(storage, f"a{i}")
        _finalize(storage, rec, "completed", 0.1, calls)
    assert storage.avg_calls_per_mission() == pytest.approx(6.0)


def test_avg_ignores_running(storage):
    rec = _start(storage, "done")
    _finalize(storage, rec, "completed", 0.1, 10)
    _start(storage, "still_running")  # excluded from the average
    assert storage.avg_calls_per_mission() == pytest.approx(10.0)


def test_list_missions_running_falls_back_to_calls(storage):
    _start(storage, "live")
    rows = {r["mission_id"]: r for r in storage.list_missions()}
    assert rows["live"]["status"] == "running"
    assert rows["live"]["calls"] == 0


def test_metrics_summary_empty(storage):
    summary = storage.metrics_summary()
    assert summary["today"]["calls"] == 0
    assert summary["this_month"]["spend_usd"] == 0


def test_hourly_burn_rate_empty(storage):
    assert storage.hourly_burn_rate(hours=24) == []


def test_missions_table_migration_safe(tmp_path):
    # Re-opening an existing DB must not error and must keep prior rows.
    db = str(tmp_path / "mig.db")
    s1 = SQLiteStorage(db_path=db)
    _finalize(s1, _start(s1, "keep"), "completed", 0.5, 3)
    s2 = SQLiteStorage(db_path=db)  # triggers migration path again
    assert s2.get_mission("keep")["final_call_count"] == 3
