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

from prediction_service.config import ConfigurationError, Settings
from prediction_service.engine import (
    HistoricalMarketDataDriftError,
    release_id,
    shadow_settings,
)
from prediction_service.models import ModelRelease, PredictionLedger, RefreshJob, ShadowRun
from prediction_service.scheduler import DailyRefreshScheduler
from prediction_service.service import (
    NoNewMarketDataError,
    PredictionDriftError,
    PredictionService,
    RefreshManager,
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


def test_settings_loads_chinese_ini_and_resolves_relative_paths(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "配置目录"
    config_dir.mkdir()
    config_path = config_dir / "config.ini"
    config_path.write_text(
        """# 中文注释不应影响解析。
[服务]
监听地址 = 0.0.0.0
监听端口 = 8123
数据目录 = 运行数据
本地特征文件 = 行情/merged_features.csv
历史起始日期 = 20210104
循环验证天数 = 75
正式预测引擎 = state_veto_rule
启用定时刷新 = 开启
刷新小时 = 19
刷新分钟 = 20
Cookie仅HTTPS = 是

[管理员]
账号 = 管理员
密码 = 密码%不会插值

[Tushare]
令牌 = token%不会插值
重试次数 = 4
普通重试等待秒数 = 1.5
限流等待秒数 = 70
港股接口最小间隔秒数 = 7.5

[BiLSTM影子]
启用 = 否
循环验证天数 = 30
重训间隔交易日 = 6
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PREDICTION_SERVICE_PORT", "9999")

    settings = Settings.from_config(config_path)

    assert settings.host == "0.0.0.0"
    assert settings.port == 8123
    assert settings.root_dir == (config_dir / "运行数据").resolve()
    assert settings.local_feature_path == (config_dir / "行情/merged_features.csv").resolve()
    assert settings.validation_days == 75
    assert settings.admin_username == "管理员"
    assert settings.admin_password == "密码%不会插值"
    assert settings.tushare_token == "token%不会插值"
    assert settings.scheduled_refresh_enabled is True
    assert settings.cookie_secure is True
    assert settings.bilstm_shadow_enabled is False
    assert settings.bilstm_shadow_validation_days == 30
    assert settings.bilstm_shadow_refit_interval == 6
    assert settings.archive_dir.is_dir()
    assert settings.shadow_archive_dir.is_dir()
    assert settings.database_url.endswith("运行数据/prediction_service.db")


def test_settings_rejects_scheduled_refresh_without_configured_token(tmp_path: Path):
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        """[服务]
启用定时刷新 = 是
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Tushare"):
        Settings.from_config(config_path)


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


def test_tushare_fetch_and_calendar_use_configured_token(tmp_path: Path, monkeypatch):
    settings = Settings.for_test(tmp_path, tushare_token="ini-token")
    service = PredictionService(settings, calculation_function=_fake_calculation)
    fetcher = importlib.import_module("数据拉取脚本_tushare")
    captured: dict[str, object] = {}

    def fake_fetch_all(*_args, **kwargs):
        captured["fetch_token"] = kwargs["token"]
        return {"merged_features": _features(), "sh000001": pd.DataFrame()}

    class FakePro:
        def trade_cal(self, **kwargs):
            captured["calendar_fields"] = kwargs["fields"]
            return pd.DataFrame({"is_open": [1]})

    def fake_get_pro(token):
        captured["calendar_token"] = token
        return FakePro()

    monkeypatch.setattr(fetcher, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(fetcher, "get_pro", fake_get_pro)

    features, _ = service.fetch_tushare_features()

    assert not features.empty
    assert captured["fetch_token"] == "ini-token"
    assert service.is_sse_trading_day("20260112") is True
    assert captured["calendar_token"] == "ini-token"
    assert captured["calendar_fields"] == "cal_date,is_open"


def test_tushare_refresh_does_not_fall_back_to_environment_token(tmp_path: Path, monkeypatch):
    settings = Settings.for_test(tmp_path, tushare_token=None)
    service = PredictionService(settings, calculation_function=_fake_calculation)
    monkeypatch.setenv("TUSHARE_TOKEN", "environment-token")

    with pytest.raises(RuntimeError, match="config.ini"):
        service.fetch_tushare_features()


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


def test_bilstm_shadow_uses_separate_release_without_changing_publication(tmp_path: Path):
    settings = Settings.for_test(
        tmp_path,
        validation_days=2,
        bilstm_shadow_enabled=True,
        bilstm_shadow_validation_days=2,
        bilstm_shadow_refit_interval=5,
    )
    service = PredictionService(
        settings,
        calculation_function=_fake_calculation,
        shadow_calculation_function=_fake_calculation,
    )
    service.initialize(bootstrap=False)
    public = service.publish_from_features(
        _features(), source="test", raw_frames=None, actor="tester"
    )
    before_publication, _, before_release = service._load_active_context()

    run, created = service.request_bilstm_shadow(public.snapshot_id, actor="tester")
    completed = service.run_bilstm_shadow(run.id)
    after_publication, _, after_release = service._load_active_context()

    assert created is True
    assert completed.status == "succeeded"
    assert completed.engine == "bilstm_causal"
    assert completed.release_id != before_release.id
    assert after_publication.id == before_publication.id
    assert after_release.id == before_release.id
    assert service.recompute_active().csv_sha256 == public.csv_sha256
    shadow = service.shadow_dashboard_data()
    assert shadow["run"].id == completed.id
    assert shadow["metrics"]["rows"] == 2
    assert service.shadow_result_file(completed.id).exists()

    with service.database.session() as session:
        assert len(session.scalars(select(ModelRelease)).all()) == 2
        assert len(session.scalars(select(ShadowRun)).all()) == 1
        assert len(
            session.scalars(
                select(PredictionLedger).where(
                    PredictionLedger.release_id == completed.release_id
                )
            ).all()
        ) == 3


def test_shadow_request_is_idempotent_for_same_snapshot_and_release(tmp_path: Path):
    settings = Settings.for_test(
        tmp_path,
        validation_days=2,
        bilstm_shadow_enabled=True,
    )
    service = PredictionService(
        settings,
        calculation_function=_fake_calculation,
        shadow_calculation_function=_fake_calculation,
    )
    service.initialize(bootstrap=False)
    public = service.publish_from_features(
        _features(), source="test", raw_frames=None, actor="tester"
    )

    first, first_created = service.request_bilstm_shadow(public.snapshot_id, actor="tester")
    second, second_created = service.request_bilstm_shadow(public.snapshot_id, actor="tester")

    assert first_created is True
    assert second_created is False
    assert first.id == second.id


def test_shadow_release_fingerprint_includes_execution_parameters(tmp_path: Path):
    first = Settings.for_test(tmp_path / "first", bilstm_shadow_refit_interval=5)
    second = Settings.for_test(tmp_path / "second", bilstm_shadow_refit_interval=7)

    first_id, first_config, _, _ = release_id(shadow_settings(first))
    second_id, _, _, _ = release_id(shadow_settings(second))

    assert first_id != second_id
    assert first_config["algorithm_id"] == "bilstm_causal"
    assert first_config["loop_options"]["signal_engine"] == "bilstm_causal"
    assert first_config["loop_options"]["bilstm_refit_interval"] == 5
    assert first_config["loop_options"]["recent_failure_guard"] is False


def test_dashboard_shows_completed_bilstm_shadow_without_replacing_default(tmp_path: Path):
    settings = Settings.for_test(
        tmp_path,
        validation_days=2,
        bilstm_shadow_enabled=True,
        bilstm_shadow_validation_days=2,
    )
    service = PredictionService(
        settings,
        calculation_function=_fake_calculation,
        shadow_calculation_function=_fake_calculation,
    )
    service.initialize(bootstrap=False)
    public = service.publish_from_features(
        _features(), source="test", raw_frames=None, actor="tester"
    )
    run, _ = service.request_bilstm_shadow(public.snapshot_id, actor="tester")
    service.run_bilstm_shadow(run.id)
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
        shadow_csv = client.get(f"/admin/shadow-runs/{run.id}/results.csv")

    assert dashboard.status_code == 200
    assert "BiLSTM 影子运行" in dashboard.text
    assert shadow_csv.status_code == 200
    assert shadow_csv.content.startswith(b"\xef\xbb\xbf")


def test_successful_refresh_invokes_shadow_submission_callback(tmp_path: Path, monkeypatch):
    service = _service(tmp_path)
    artifact = service.publish_from_features(
        _features(), source="test", raw_frames=None, actor="tester"
    )
    observed: list[tuple[str, str | None]] = []
    manager = RefreshManager(
        service,
        on_success=lambda result, actor: observed.append((result.snapshot_id, actor)),
    )
    monkeypatch.setattr(service, "refresh_from_tushare", lambda *, actor: artifact)
    with service.database.session() as session:
        job = RefreshJob(
            id="refresh-test-job",
            trigger="manual",
            requested_by="tester",
            status="queued",
        )
        session.add(job)

    manager._run("refresh-test-job")

    assert observed == [(artifact.snapshot_id, "tester")]
    assert manager.get("refresh-test-job").status == "succeeded"
    manager.shutdown()
