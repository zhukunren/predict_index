from __future__ import annotations

import io
import json
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from prediction_service.archive import frame_to_csv_bytes, read_features
from prediction_service.calendar import SHANGHAI
from prediction_service.engine import canonicalize_features, merge_append_only_features, HistoricalMarketDataDriftError
from prediction_service.metrics import statistics, clean_records
from prediction_service.models import RefreshJob, ShadowRun
from prediction_service.models import OutcomeResolution, PredictionLedger
from sqlalchemy import select
from prediction_service.service import PredictionService, RefreshManager
from prediction_service.scheduler import DailyRefreshScheduler
from prediction_service.web import create_app
from test_prediction_service import _service, _features, _fake_calculation


def login(client):
    page = client.get("/admin/login")
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
    response = client.post("/admin/login", data={"csrf_token": token, "username": "admin", "password": "test-password"}, follow_redirects=False)
    assert response.status_code == 303


def test_float_round_trip_preserves_frozen_large_values(tmp_path):
    frame = _features()
    frame["amount"] = 724153675.7008
    serialized = frame_to_csv_bytes(frame)
    parsed = pd.read_csv(io.BytesIO(serialized))
    merged = merge_append_only_features(parsed, frame)
    assert merged["amount"].equals(parsed["amount"])
    changed = frame.copy()
    changed.loc[2, "amount"] += 0.01
    with pytest.raises(HistoricalMarketDataDriftError):
        merge_append_only_features(parsed, changed)
    path = tmp_path / "features.csv"
    path.write_bytes(serialized)
    assert read_features(path)["amount"].iloc[0] == frame["amount"].iloc[0]


def test_compact_dates_and_invalid_market_rows():
    frame = _features(3)
    frame["trade_date"] = [20260915, 20260916, 20260917]
    assert canonicalize_features(frame)["trade_date"].tolist() == ["2026-09-15", "2026-09-16", "2026-09-17"]
    frame.loc[1, "trade_date"] = 20260915
    with pytest.raises(ValueError, match="重复"):
        canonicalize_features(frame)
    frame.loc[1, "trade_date"] = 20260230
    with pytest.raises(ValueError, match="解析"):
        canonicalize_features(frame)


def test_public_download_never_computes_and_supports_conditional_get(tmp_path, monkeypatch):
    service = _service(tmp_path)
    artifact = service.publish_from_features(_features(), source="test", raw_frames=None, actor="test")
    monkeypatch.setattr(service, "calculation_function", lambda *args: pytest.fail("GET ran the model"))
    service.settings = replace(service.settings, signal_engine="volatility_rule")
    with TestClient(create_app(service.settings, service=service, bootstrap=False)) as client:
        result = client.get("/api/v1/sh000001/latest.csv")
        assert result.status_code == 200 and result.content == artifact.csv_bytes
        conditional = client.get("/api/v1/sh000001/latest.csv", headers={"If-None-Match": "W/" + result.headers["etag"]})
        assert conditional.status_code == 304 and not conditional.content
        head = client.head("/api/v1/sh000001/latest.csv")
        assert head.status_code == 200 and not head.content
        assert int(head.headers["content-length"]) == len(artifact.csv_bytes)
        artifact.archive.results_path.write_bytes(b"corrupt")
        assert client.get("/api/v1/sh000001/latest.csv", headers={"If-None-Match": result.headers["etag"]}).status_code == 503
        health = client.get("/healthz")
        assert health.status_code == 503 and health.json()["csv_available"] is False
        assert client.get("/livez").status_code == 200


def test_balanced_accuracy_uses_actual_classes_and_ignores_pending():
    frame = pd.DataFrame({
        "信号日期": list(range(7)),
        "预测方向": ["上涨"] * 4 + ["下跌", "下跌", "上涨"],
        "预测次日涨跌幅": [0.01] * 4 + [-0.01, -0.01, 0.01],
        "次日实际涨跌幅": [0.01] * 5 + [-0.01, None],
        "置信度": [0.6] * 7,
    })
    result = statistics(frame)
    assert result["rows"] == 6
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert result["balanced_accuracy"] == pytest.approx(0.9)
    assert statistics(frame, 1)["balanced_accuracy"] is None


def test_custom_window_uses_ledger_and_persists_without_changing_csv(tmp_path):
    service = _service(tmp_path)
    for size in range(8, 12):
        artifact = service.publish_from_features(_features(size), source="test", raw_frames=None, actor="test")
    assert service.dashboard_data(1)["metrics"]["rows"] == 1
    assert service.dashboard_data(4)["metrics"]["rows"] == 4
    assert service.dashboard_data(20)["metrics"]["available_rows"] == 5
    assert len(service.dashboard_data(20)["rows"]) == 5
    assert service.dashboard_data(1)["latest"]["次日实际涨跌幅"] is None
    with TestClient(create_app(service.settings, service=service, bootstrap=False)) as client:
        login(client)
        page = client.get("/admin/?days=4")
        assert page.status_code == 200
        records = json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', page.text, re.S).group(1))
        assert len(records) == 4
        retained = client.get("/admin/?view=history")
        assert 'value="4" aria-label="自定义统计交易日数"' in retained.text
        assert "+nan%" not in page.text
        assert client.get("/admin/?days=0").status_code == 422
        assert client.get("/admin/?days=5001").status_code == 422
        assert client.get("/api/v1/sh000001/latest.csv").content == artifact.csv_bytes


def test_calendar_handles_holidays_freshness_and_missing_coverage(tmp_path):
    service = _service(tmp_path)
    dates = pd.date_range("2026-08-01", "2026-10-10")
    calendar = pd.DataFrame({"cal_date": dates.strftime("%Y%m%d"), "is_open": [int(day.weekday() < 5 and not ("20261001" <= day.strftime("%Y%m%d") <= "20261007")) for day in dates]})
    service.calendar.store(calendar)
    assert service.calendar.next_session("20260930") == "20261008"
    assert service.calendar.next_session("20261010") is None
    assert service.calendar.expected_as_of(datetime(2026, 10, 6, 19, tzinfo=SHANGHAI), 18, 15) == "20260930"
    assert service.calendar.expected_as_of(datetime(2026, 9, 17, 9, tzinfo=SHANGHAI), 18, 15) == "20260916"
    service.publish_from_features(_features(), source="test", raw_frames=None, actor="test")
    assert service.health(datetime(2026, 9, 17, 19, tzinfo=SHANGHAI))["status"] == "stale"
    assert service.health(datetime(2027, 1, 1, 19, tzinfo=SHANGHAI))["status"] == "unknown"
    service.shutdown()


def test_restart_releases_orphan_jobs_and_retry_is_bounded(tmp_path, monkeypatch):
    service = _service(tmp_path)
    with service.database.session() as session:
        session.add(RefreshJob(id="orphan", trigger="manual", status="running"))
    service.recover_interrupted_jobs()
    manager = RefreshManager(service)
    submitted = []
    monkeypatch.setattr(manager._executor, "submit", lambda callback, job_id: submitted.append(job_id))
    assert manager.get("orphan").status == "interrupted"
    first, created = manager.submit(trigger="scheduled", actor=None, idempotency_key="20260917")
    assert created
    assert manager.submit(trigger="scheduled", actor=None, idempotency_key="20260917") == (first, False)
    with service.database.session() as session:
        job = session.get(RefreshJob, first)
        job.status = "skipped"
        job.finished_at = datetime.now(timezone.utc) - timedelta(minutes=11)
    second, created = manager.submit(trigger="scheduled", actor=None, idempotency_key="20260917")
    assert created and second != first and len(submitted) == 2
    manager.shutdown()
    service.shutdown()


def test_scheduler_can_catch_up_on_weekends(tmp_path):
    service = _service(tmp_path)
    class Manager:
        def submit(self, **kwargs):
            assert kwargs["idempotency_key"] == "20260918"
            return "job", True
    scheduler = DailyRefreshScheduler(service.settings, Manager(), expected_date=lambda now: "20260918")
    assert scheduler.run_once(datetime(2026, 9, 19, 10, tzinfo=SHANGHAI))
    service.shutdown()


def test_pending_prediction_is_settled_after_long_downtime(tmp_path):
    service = _service(tmp_path)
    initial = service.publish_from_features(_features(8), source="test", raw_frames=None, actor="test")
    pending_date = str(int(initial.result.iloc[-1]["trade_date"]))
    service.publish_from_features(_features(20), source="test", raw_frames=None, actor="test")
    with service.database.session() as session:
        prediction = session.scalar(select(PredictionLedger).where(PredictionLedger.signal_date == pending_date))
        outcome = session.get(OutcomeResolution, prediction.id)
        assert outcome is not None and outcome.actual_return > 0
    assert service.dashboard_data(100)["metrics"]["rows"] == 5
    service.shutdown()


def test_calendar_updates_do_not_change_archived_export_metadata(tmp_path):
    service = _service(tmp_path)
    first = service.publish_from_features(_features(), source="test", raw_frames=None, actor="test")
    dates = pd.date_range("2026-01-01", "2026-02-01")
    service.calendar.store(pd.DataFrame({"cal_date": dates.strftime("%Y%m%d"), "is_open": (dates.weekday < 5).astype(int)}))
    assert service.recompute_active().csv_bytes == first.csv_bytes
    service.shutdown()


def test_display_records_normalize_missing_and_nonfinite_values():
    frame = pd.DataFrame({"return": [float("inf"), float("nan"), 0.01], "correct": [None, False, True]})
    records = clean_records(frame)
    assert records == [{"return": None, "correct": None}, {"return": None, "correct": False}, {"return": 0.01, "correct": True}]
    json.dumps(records, allow_nan=False)
