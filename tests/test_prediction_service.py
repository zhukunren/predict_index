from __future__ import annotations

import re
import importlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from prediction_service.config import Settings
from prediction_service.engine import HistoricalMarketDataDriftError
from prediction_service.models import ModelRelease, PredictionLedger
from prediction_service.scheduler import DailyRefreshScheduler
from prediction_service.service import (
    NoNewMarketDataError,
    PredictionDriftError,
    PredictionService,
)
from prediction_service.web import create_app


def _features(rows: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=rows)
    close = 100.0 + pd.Series(range(rows), dtype=float) * 0.5
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close * 0.998,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "pre_close": close.shift(1).fillna(close.iloc[0] - 0.5),
            "vol": 1_000_000.0,
            "amount": close * 1_000_000.0,
            "target_next_return": close.shift(-1) / close - 1.0,
            "target_next_direction": (close.shift(-1) > close).astype(float),
        }
    )


def _fake_calculation(features: pd.DataFrame, _settings: Settings) -> pd.DataFrame:
    source = features.copy()
    source["trade_date"] = pd.to_datetime(source["trade_date"])
    source = source.sort_values("trade_date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for index in range(max(0, len(source) - 3), len(source)):
        close = float(source.loc[index, "close"])
        signal_date = int(source.loc[index, "trade_date"].strftime("%Y%m%d"))
        predicted_return = 0.001 if index % 2 == 0 else -0.001
        has_outcome = index + 1 < len(source)
        realized = (
            float(source.loc[index + 1, "close"] / close - 1.0)
            if has_outcome
            else None
        )
        rows.append(
            {
                "trade_date": signal_date,
                "predicted_pct_change": predicted_return,
                "predicted_label": int(predicted_return > 0),
                "predicted_close": close * (1.0 + predicted_return),
                "confidence": 0.61,
                "calibrated_confidence": 0.62,
                "real_pct_change": realized,
                "correct": (
                    bool((predicted_return > 0) == (realized > 0))
                    if realized is not None
                    else None
                ),
                "confidence_calibration_window": 300,
                "confidence_calibration_method": "platt",
                "confidence_calibration_rows": 300,
                "confidence_calibration_fallback": 0,
                "return_calibration_scale": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _service(tmp_path: Path) -> PredictionService:
    settings = Settings.for_test(tmp_path, validation_days=2)
    service = PredictionService(settings, calculation_function=_fake_calculation)
    service.initialize(bootstrap=False)
    return service


def _fake_calculation_with_diagnostics(
    features: pd.DataFrame,
    settings: Settings,
    diagnostics_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = _fake_calculation(features, settings)
    diagnostics = result.copy()
    diagnostics["veto_applied"] = [0, 1, 1]
    diagnostics["base_predicted_label"] = [
        int(label) for label in 1 - diagnostics["predicted_label"]
    ]
    diagnostics["base_predicted_return"] = -diagnostics["predicted_pct_change"]
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    return result, diagnostics


def test_publish_recomputes_from_archive_without_drift(tmp_path: Path):
    service = _service(tmp_path)
    first = service.publish_from_features(
        _features(), source="test", raw_frames=None, actor="tester"
    )

    verified = service.recompute_active()

    assert verified.csv_sha256 == first.csv_sha256
    assert verified.csv_bytes == first.csv_bytes
    assert verified.public_frame["结果类型"].tolist() == [
        "循环验证",
        "循环验证",
        "次日预测",
    ]
    assert first.archive is not None
    assert first.archive.manifest_path.exists()


def test_appended_day_preserves_prediction_and_only_adds_outcome(tmp_path: Path):
    service = _service(tmp_path)
    first = service.publish_from_features(
        _features(8), source="test", raw_frames=None, actor="tester"
    )
    previous_pending = first.result.iloc[-1]
    pending_date = f"{int(previous_pending['trade_date']):08d}"

    second = service.publish_from_features(
        _features(9), source="test", raw_frames=None, actor="tester"
    )
    completed = second.result.loc[
        second.result["trade_date"].eq(int(pending_date))
    ].iloc[0]

    assert completed["predicted_pct_change"] == previous_pending["predicted_pct_change"]
    assert completed["predicted_close"] == previous_pending["predicted_close"]
    assert completed["confidence"] == previous_pending["confidence"]
    assert pd.isna(previous_pending["real_pct_change"])
    assert pd.notna(completed["real_pct_change"])
    with service.database.session() as session:
        row = session.scalar(
            select(PredictionLedger).where(
                PredictionLedger.signal_date == pending_date
            )
        )
        assert row is not None
        assert row.predicted_return == previous_pending["predicted_pct_change"]


def test_historical_provider_revision_is_rejected_and_not_published(tmp_path: Path):
    service = _service(tmp_path)
    first = service.publish_from_features(
        _features(), source="test", raw_frames=None, actor="tester"
    )
    changed = _features(9)
    changed.loc[2, "close"] += 1.0

    with pytest.raises(HistoricalMarketDataDriftError):
        service.publish_from_features(
            changed, source="test", raw_frames=None, actor="tester"
        )

    verified = service.recompute_active()
    assert verified.snapshot_id == first.snapshot_id


def test_prediction_rows_cannot_be_updated_or_deleted(tmp_path: Path):
    service = _service(tmp_path)
    service.publish_from_features(_features(), source="test", raw_frames=None, actor="tester")

    with pytest.raises(IntegrityError):
        with service.database.session() as session:
            row = session.scalar(select(PredictionLedger))
            assert row is not None
            row.predicted_return = 0.9


def test_public_csv_is_unauthenticated_and_admin_uses_password(tmp_path: Path):
    service = _service(tmp_path)
    service.publish_from_features(_features(), source="test", raw_frames=None, actor="tester")
    app = create_app(service.settings, service=service, bootstrap=False)

    with TestClient(app) as client:
        public = client.get("/api/v1/sh000001/latest.csv")
        assert public.status_code == 200
        assert public.content.startswith(b"\xef\xbb\xbf")
        assert public.headers["x-snapshot-id"]

        denied = client.get("/admin/", follow_redirects=False)
        assert denied.status_code == 303
        login_page = client.get("/admin/login")
        token = re.search(
            r'name="csrf_token" value="([^"]+)"', login_page.text
        )
        assert token is not None
        logged_in = client.post(
            "/admin/login",
            data={
                "csrf_token": token.group(1),
                "username": "admin",
                "password": "test-password",
            },
            follow_redirects=False,
        )
        assert logged_in.status_code == 303
        dashboard = client.get("/admin/")
        assert dashboard.status_code == 200
        assert "运行概览" in dashboard.text


def test_public_api_fails_closed_when_recalculation_drifts(tmp_path: Path):
    state = {"drift": False}

    def changing_calculation(features: pd.DataFrame, settings: Settings) -> pd.DataFrame:
        frame = _fake_calculation(features, settings)
        if state["drift"]:
            frame.loc[0, "predicted_pct_change"] = 0.009
            frame.loc[0, "predicted_label"] = 1
        return frame

    settings = Settings.for_test(tmp_path, validation_days=2)
    service = PredictionService(settings, calculation_function=changing_calculation)
    service.initialize(bootstrap=False)
    service.publish_from_features(_features(), source="test", raw_frames=None, actor="tester")
    state["drift"] = True

    with pytest.raises(PredictionDriftError):
        service.recompute_active()

    app = create_app(settings, service=service, bootstrap=False)
    with TestClient(app) as client:
        response = client.get("/api/v1/sh000001/latest.csv")
    assert response.status_code == 503


def test_archive_feature_hash_mismatch_fails_closed(tmp_path: Path):
    service = _service(tmp_path)
    artifact = service.publish_from_features(
        _features(), source="test", raw_frames=None, actor="tester"
    )
    assert artifact.archive is not None
    with artifact.archive.features_path.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(PredictionDriftError, match="哈希"):
        service.recompute_active()


def test_refresh_skips_when_provider_has_no_new_market_day(tmp_path: Path, monkeypatch):
    service = _service(tmp_path)
    first = service.publish_from_features(
        _features(), source="test", raw_frames=None, actor="tester"
    )
    monkeypatch.setattr(service, "fetch_tushare_features", lambda: (_features(), {}))

    with pytest.raises(NoNewMarketDataError):
        service.refresh_from_tushare(actor="tester")

    assert service.recompute_active().snapshot_id == first.snapshot_id


class _RefreshManagerStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def submit(self, **kwargs):
        self.calls.append(kwargs)
        return "job-1", True


def test_scheduler_uses_trading_day_filter_and_daily_idempotency(tmp_path: Path):
    settings = Settings.for_test(
        tmp_path,
        scheduled_refresh_enabled=True,
        scheduled_refresh_hour=18,
        scheduled_refresh_minute=15,
    )
    manager = _RefreshManagerStub()
    scheduler = DailyRefreshScheduler(
        settings,
        manager,  # type: ignore[arg-type]
        is_trading_day=lambda day: day == "20260112",
    )
    timezone = ZoneInfo("Asia/Shanghai")

    assert not scheduler.run_once(datetime(2026, 1, 10, 18, 15, tzinfo=timezone))
    assert not scheduler.run_once(datetime(2026, 1, 9, 18, 15, tzinfo=timezone))
    assert scheduler.run_once(datetime(2026, 1, 12, 18, 15, tzinfo=timezone))
    assert not scheduler.run_once(datetime(2026, 1, 12, 19, 0, tzinfo=timezone))
    assert manager.calls == [
        {
            "trigger": "scheduled",
            "actor": None,
            "idempotency_key": "20260112",
        }
    ]


def test_veto_diagnostics_are_archived_and_reported_on_dashboard(tmp_path: Path):
    settings = Settings.for_test(tmp_path, validation_days=2)
    service = PredictionService(
        settings,
        calculation_function=_fake_calculation,
        diagnostics_calculation_function=_fake_calculation_with_diagnostics,
    )
    service.initialize(bootstrap=False)

    artifact = service.publish_from_features(
        _features(), source="test", raw_frames=None, actor="tester"
    )
    assert artifact.archive is not None
    assert artifact.archive.diagnostics_path is not None
    assert artifact.archive.diagnostics_path.exists()

    veto = service.dashboard_data()["veto"]
    assert veto is not None
    assert veto["all"]["rows"] == 2
    assert veto["all"]["trigger_rate"] == pytest.approx(2.0 / 3.0)
    assert veto["recent_20"]["directional_return_lift"] is not None


def test_english_script_aliases_export_legacy_apis():
    legacy_loop = importlib.import_module("循环验证脚本")
    legacy_predictor = importlib.import_module("预测脚本")
    legacy_fetcher = importlib.import_module("数据拉取脚本_tushare")

    assert importlib.import_module("loop_validation").loop_validate_prediction_results is legacy_loop.loop_validate_prediction_results
    assert importlib.import_module("predictor").save_prediction_csv is legacy_predictor.save_prediction_csv
    assert importlib.import_module("tushare_fetcher").fetch_all is legacy_fetcher.fetch_all


def test_compatible_release_promotion_preserves_old_ledger(tmp_path: Path):
    first_settings = Settings.for_test(tmp_path, validation_days=2)
    first_service = PredictionService(
        first_settings,
        calculation_function=_fake_calculation,
    )
    first_service.initialize(bootstrap=False)
    original = first_service.publish_from_features(
        _features(), source="test", raw_frames=None, actor="tester"
    )

    updated_settings = Settings.for_test(
        tmp_path,
        validation_days=2,
        signal_engine="volatility_rule",
    )
    updated_service = PredictionService(
        updated_settings,
        calculation_function=_fake_calculation,
    )
    updated_service.initialize(bootstrap=False)
    assert updated_service.release_upgrade_required()

    promoted = updated_service.promote_compatible_release(actor="tester")

    assert promoted is not None
    assert promoted.release_id != original.release_id
    assert promoted.snapshot_id != original.snapshot_id
    assert not updated_service.release_upgrade_required()
    with updated_service.database.session() as session:
        assert len(session.scalars(select(ModelRelease)).all()) == 2
        assert len(session.scalars(select(PredictionLedger)).all()) == 6


def test_dashboard_renders_veto_monitoring_panel(tmp_path: Path):
    settings = Settings.for_test(tmp_path, validation_days=2)
    service = PredictionService(
        settings,
        calculation_function=_fake_calculation,
        diagnostics_calculation_function=_fake_calculation_with_diagnostics,
    )
    service.initialize(bootstrap=False)
    service.publish_from_features(_features(), source="test", raw_frames=None, actor="tester")
    app = create_app(settings, service=service, bootstrap=False)

    with TestClient(app) as client:
        login_page = client.get("/admin/login")
        token = re.search(r'name="csrf_token" value="([^"]+)"', login_page.text)
        assert token is not None
        client.post(
            "/admin/login",
            data={
                "csrf_token": token.group(1),
                "username": "admin",
                "password": "test-password",
            },
        )
        dashboard = client.get("/admin/")

    assert dashboard.status_code == 200
    assert "低波动状态反转归因" in dashboard.text
