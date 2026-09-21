from datetime import datetime, timedelta
import json
from pathlib import Path

from filelock import FileLock
import pytest
from sqlalchemy import func, select

import prediction_service.downside_watcher as watcher_module
from prediction_service.downside_shadow import prospective_report
from prediction_service.calendar import SHANGHAI
from prediction_service.models import Publication, ShadowRun
from prediction_service.downside_watcher import FixedDownsideWatcher
from test_prediction_service import _service, _features
from test_downside_shadow import calendar, clock, contract, record


@pytest.fixture
def context(tmp_path, monkeypatch):
    service = _service(tmp_path)
    calendar(service)
    now = datetime(2026, 1, 14, 18, 30, tzinfo=SHANGHAI)
    clock(monkeypatch, now)
    public = service.publish_from_features(_features(), source="test", raw_frames=None, actor="test")
    monkeypatch.setattr(watcher_module, "load_candidate", lambda bundle: contract())
    watcher = FixedDownsideWatcher(service, tmp_path / "bundle")
    yield service, watcher, now, public
    service.shutdown()


def forbid_runner(*args, **kwargs):
    raise AssertionError("Unexpected request or forecast calculation")


def test_watcher_records_once_then_settles_next_publication_without_public_changes(context, monkeypatch):
    service, watcher, now, public = context
    calls = []

    def execute(current, bundle, *, fetch, allow_backfill):
        assert current is service and fetch is True and allow_backfill is False
        calls.append(bundle)
        _, snapshot, control = service._load_active_context()
        from prediction_service.archive import read_features
        result = record(service, read_features(snapshot.features_path))
        return result, prospective_report(service, result.release_id, control.id)

    monkeypatch.setattr(watcher_module, "run_shadow", execute)
    first = watcher.run_once(now)
    assert first["status"] == "succeeded"
    assert first["attempts"] == 1 and len(calls) == 1
    assert first["report"]["prospective_paired_rows"] == 0
    assert service.read_published().csv_sha256 == public.csv_sha256
    assert watcher.run_once(now + timedelta(minutes=1))["status"] == "up_to_date"
    assert len(calls) == 1
    with service.database.session() as session:
        assert session.scalar(select(func.count()).select_from(ShadowRun)) == 1
        assert session.scalar(select(func.count()).select_from(Publication)) == 1
    next_time = now + timedelta(days=1)
    clock(monkeypatch, next_time)
    second_public = service.publish_from_features(_features(9), source="test", raw_frames=None, actor="test")
    second = watcher.run_once(next_time)
    assert second["status"] == "succeeded" and second["attempts"] == 1
    assert len(calls) == 2
    assert second["snapshot_id"] == second_public.snapshot_id
    assert second["report"]["prospective_paired_rows"] == 1
    assert second["report"]["promotion_allowed"] is False
    assert service.read_published().csv_sha256 == second_public.csv_sha256


@pytest.mark.parametrize("hour,minute,status", [(17, 59, "waiting_window"), (9, 30, "missed_window")])
def test_watcher_never_fetches_before_release_or_after_open(context, monkeypatch, hour, minute, status):
    _, watcher, now, _ = context
    monkeypatch.setattr(watcher_module, "run_shadow", forbid_runner)
    check = now.replace(hour=hour, minute=minute)
    if status == "missed_window":
        check += timedelta(days=1)
    result = watcher.run_once(check)
    assert result["status"] == status
    assert result["attempts"] == 0 and result["backfill_allowed"] is False


def test_failed_fetch_is_redacted_and_retry_limit_survives_restart(context, monkeypatch):
    service, watcher, now, public = context
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("SECRET_TOKEN")

    monkeypatch.setattr(watcher_module, "run_shadow", fail)
    monkeypatch.setattr(watcher_module, "MAX_ATTEMPTS", 2)
    assert watcher.run_once(now)["status"] == "failed"
    restarted = FixedDownsideWatcher(service, watcher.bundle)
    assert restarted.run_once(now + timedelta(minutes=9))["status"] == "retry_wait"
    assert len(calls) == 1
    assert restarted.run_once(now + timedelta(minutes=10))["status"] == "failed"
    assert restarted.run_once(now + timedelta(minutes=20))["status"] == "retry_exhausted"
    assert len(calls) == 2
    state = json.loads(watcher.state_path.read_text(encoding="utf-8"))
    assert state["attempts"] == 2 and state["error_type"] == "RuntimeError"
    assert "SECRET_TOKEN" not in watcher.state_path.read_text(encoding="utf-8")
    assert service.read_published().csv_sha256 == public.csv_sha256


def test_existing_cli_lock_does_not_consume_retry_or_fetch(context, monkeypatch):
    service, watcher, now, _ = context
    monkeypatch.setattr(watcher_module, "run_shadow", forbid_runner)
    with FileLock(str(service.settings.root_dir / "downside_shadow.lock"), timeout=0):
        state = watcher.run_once(now)
    assert state["status"] == "busy" and state["attempts"] == 0


def test_changed_candidate_source_blocks_requests_and_redacts_reason(context, monkeypatch):
    _, watcher, now, _ = context

    def changed(bundle):
        raise ValueError("private configuration SECRET_TOKEN")

    monkeypatch.setattr(watcher_module, "load_candidate", changed)
    monkeypatch.setattr(watcher_module, "run_shadow", forbid_runner)
    state = watcher.run_once(now)
    assert state["status"] == "blocked" and state["error_type"] == "ValueError"
    assert "SECRET_TOKEN" not in watcher.state_path.read_text(encoding="utf-8")


def test_corrupt_existing_archive_never_triggers_replacement(context, monkeypatch):
    service, watcher, now, public = context
    result = record(service, _features())
    Path(result.result_csv_path).write_bytes(b"invalid")
    monkeypatch.setattr(watcher_module, "run_shadow", forbid_runner)
    state = watcher.run_once(now)
    assert state["status"] == "failed" and state["attempts"] == 0
    assert state["error_type"] == "PredictionDriftError"
    assert service.read_published().csv_sha256 == public.csv_sha256


def test_naive_clock_is_rejected_before_any_requests(context):
    _, watcher, _, _ = context
    with pytest.raises(ValueError, match="timezone-aware"):
        watcher.run_once(datetime(2026, 1, 14, 18, 30))


def test_status_uses_live_process_lock_instead_of_stale_heartbeat(context):
    from tools.watch_downside_shadow import read_status

    service, watcher, now, _ = context
    watcher._save({}, now, "up_to_date")
    assert read_status(service.settings)["watcher_running"] is False
    with FileLock(str(service.settings.root_dir / "downside_watcher.lock"), timeout=0):
        assert read_status(service.settings)["watcher_running"] is True
    assert read_status(service.settings)["watcher_running"] is False
