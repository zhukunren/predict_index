"""Delayed providers must recover after the original six-attempt window."""

from datetime import datetime, timedelta, timezone

import pytest

from prediction_service.models import RefreshJob
from prediction_service.service import RefreshManager
from test_prediction_service import _service, _features


@pytest.mark.parametrize("attempts,cooldown", [(1, 10), (5, 10), (6, 20), (7, 40), (8, 60), (50, 60)])
def test_retry_backoff_is_bounded_and_never_exhausted(tmp_path, monkeypatch, attempts, cooldown):
    service = _service(tmp_path)
    manager = RefreshManager(service)
    now = datetime.now(timezone.utc)
    clock = [now]
    monkeypatch.setattr("prediction_service.service.utcnow", lambda: clock[0])
    submitted = []
    monkeypatch.setattr(manager._executor, "submit", lambda _callback, job_id: submitted.append(job_id))
    try:
        with service.database.session() as session:
            for i in range(attempts):
                timestamp = now - timedelta(hours=attempts - i)
                session.add(RefreshJob(id=f"attempt-{i}", trigger="scheduled:20261106", status="failed",
                                       created_at=timestamp, finished_at=timestamp))
            session.flush()
            session.get(RefreshJob, f"attempt-{attempts - 1}").finished_at = now - timedelta(minutes=cooldown - 1)
        last = f"attempt-{attempts - 1}"
        assert manager.submit(trigger="scheduled", actor=None, idempotency_key="20261106") == (last, False)
        assert submitted == []
        clock[0] += timedelta(minutes=1)
        job_id, created = manager.submit(trigger="scheduled", actor=None, idempotency_key="20261106")
        assert created and submitted == [job_id]
        assert manager.submit(trigger="scheduled", actor=None, idempotency_key="20261106") == (job_id, False)
    finally:
        manager.shutdown()
        service.shutdown()


def test_retry_stops_after_publication_has_caught_up(tmp_path, monkeypatch):
    service = _service(tmp_path)
    artifact = service.publish_from_features(_features(), source="test", raw_frames=None, actor="test")
    manager = RefreshManager(service)
    monkeypatch.setattr(manager._executor, "submit", lambda *_: pytest.fail("already published"))
    try:
        with service.database.session() as session:
            session.add(RefreshJob(id="old-attempt", trigger=f"scheduled:{artifact.data_as_of}", status="failed",
                                   finished_at=datetime.now(timezone.utc) - timedelta(days=1)))
        assert manager.submit(trigger="scheduled", actor=None, idempotency_key=artifact.data_as_of) == ("old-attempt", False)
    finally:
        manager.shutdown()
        service.shutdown()
