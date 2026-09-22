from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from prediction_service import model_registry, portfolio
from prediction_service.forecast_models import MODELS, MODEL_KEYS, PRODUCTION_KEY
from prediction_service.model_comparison import comparison_data
from prediction_service.models import ModelRelease, PredictionLedger, Publication, ShadowRun
from prediction_service.service import ModelReleaseMismatchError, PredictionDriftError
from prediction_service.web import create_app
from test_prediction_service import _service, _features, _fake_calculation


@pytest.fixture
def portfolio_service(tmp_path, monkeypatch, request):
    service = _service(tmp_path)
    days = pd.date_range("2026-01-01", "2026-02-28")
    service.calendar.store(pd.DataFrame({"cal_date": days.strftime("%Y%m%d"), "is_open": (days.dayofweek < 5).astype(int)}))
    initial = _features(8)
    if getattr(request, "param", False):
        initial["hangseng_gap_lag1"] = [0.01] * 7 + [None]
    service.publish_from_features(initial, source="original", raw_frames=None, actor="test")
    service.settings = replace(service.settings, model_bundle_dir=tmp_path / "models")
    manifest = {"bundle_id": "test-bundle", "releases": {}}
    for model in MODELS:
        config = {"algorithm_id": "fixed_" + model.key, "model_key": model.key,
                  "model_name": model.name, "runtime": {"sources": {}}}
        manifest["releases"][model.key] = {"release_id": model.key + "_release", "configuration": config}

    class FakeBundle:
        def __init__(self, _directory):
            self.manifest = manifest

        def context(self):
            return {name: pd.DataFrame({"trade_date": [20260101]}) for name in portfolio.CONTEXT_NAMES}

        def calculate(self, features, context):
            return {key: _fake_calculation(features, service.settings) for key in MODEL_KEYS}

    monkeypatch.setattr(portfolio, "ModelBundle", FakeBundle)
    monkeypatch.setattr(model_registry, "ModelBundle", FakeBundle)
    yield service
    service.shutdown()


def activate(service, rows=8):
    return portfolio.publish_portfolio(service, _features(rows), source="test_activation", actor="test", activate=True)


def test_activation_is_atomic_idempotent_and_keeps_original_ledger(portfolio_service):
    service = portfolio_service
    old = service.read_published()
    with service.database.session() as session:
        previous = [(p.id, p.prediction_fingerprint, p.created_at) for p in session.scalars(select(PredictionLedger)).all()]
    result = activate(service)
    assert result.release_id == "option_moneyflow_release"
    assert set(result.public_frame["模型版本"]) == {result.release_id}
    with service.database.session() as session:
        assert session.scalar(select(func.count()).select_from(Publication).where(Publication.is_active.is_(True))) == 1
        assert session.scalar(select(func.count()).select_from(ShadowRun)) == 3
        for identifier, fingerprint, created_at in previous:
            record = session.get(PredictionLedger, identifier)
            assert (record.prediction_fingerprint, record.created_at) == (fingerprint, created_at)
        assert session.get(ModelRelease, old.release_id) is not None
    again = activate(service)
    assert again.snapshot_id == result.snapshot_id
    assert again.csv_bytes == result.csv_bytes


def test_model_drift_does_not_publish_partial_results(portfolio_service, monkeypatch):
    service = portfolio_service
    activate(service)
    old = service.read_published()
    original = portfolio.ModelBundle.calculate

    def broken(self, features, context):
        results = original(self, features, context)
        results["moneyflow"].loc[0, "predicted_pct_change"] += 0.1
        return results

    monkeypatch.setattr(portfolio.ModelBundle, "calculate", broken)
    with pytest.raises(PredictionDriftError):
        activate(service, 9)
    assert service.read_published().csv_bytes == old.csv_bytes
    with service.database.session() as session:
        assert session.scalar(select(func.count()).select_from(ShadowRun)) == 3


def test_all_models_settle_on_same_dates_and_recompute_exactly(portfolio_service):
    service = portfolio_service
    activate(service)
    activate(service, 9)
    data = comparison_data(service, 3)
    assert data["paired_rows"] == 3
    assert {m["metrics"]["rows"] for m in data["models"]} == {3}
    assert {row["target_date"] for row in data["rows"]} == {"20260113", "20260114", "20260115"}
    assert comparison_data(service, 3, "live")["paired_rows"] == 0
    before = service.read_published()
    assert service.recompute_active().csv_bytes == before.csv_bytes


def test_new_release_does_not_inherit_earlier_models_timestamps(portfolio_service, monkeypatch):
    service = portfolio_service
    monkeypatch.setattr(portfolio, "utcnow", lambda: datetime(2026, 1, 14, 13, tzinfo=timezone.utc))
    activate(service)
    data = comparison_data(service, 20)
    assert all(m["latest"]["记录来源"] == "live" for m in data["models"])
    assert all(m["metrics"]["live_rows"] == 0 for m in data["models"])
    monkeypatch.setattr(portfolio, "utcnow", lambda: datetime(2026, 1, 15, 13, tzinfo=timezone.utc))
    activate(service, 9)
    live = comparison_data(service, 20, "live")
    assert live["paired_rows"] == 1
    assert live["rows"][0]["target_date"] == "20260115"


def test_regular_refresh_cannot_implicitly_change_production(portfolio_service):
    service = portfolio_service
    with pytest.raises(ModelReleaseMismatchError):
        portfolio.publish_portfolio(service, _features(8), source="tushare", actor="test")


def test_admin_comparison_auth_windows_downloads_and_empty_live_view(portfolio_service):
    service = portfolio_service
    activate(service)
    app = create_app(service.settings, service=service, bootstrap=False)
    with TestClient(app) as client:
        for route in ("/admin/models", "/admin/api/models", "/admin/models/comparison.csv", "/admin/models/baseline/latest.csv"):
            assert client.get(route, follow_redirects=False).status_code == 303
        login = client.get("/admin/login")
        token = re.search(r'name="csrf_token" value="([^"]+)"', login.text).group(1)
        assert client.post("/admin/login", data={"username": "admin", "password": "test-password", "csrf_token": token}, follow_redirects=False).status_code == 303
        page = client.get("/admin/models?days=1")
        assert page.status_code == 200
        for model in MODELS:
            assert model.name in page.text
        data = client.get("/admin/api/models?days=1").json()
        assert data["paired_rows"] == 1
        assert client.get("/admin/models").status_code == 200
        empty = client.get("/admin/models?sample=live")
        assert "事前预测尚未形成共同结算结果" in empty.text
        csv = client.get("/admin/models/comparison.csv?days=1")
        assert len(pd.read_csv(io.BytesIO(csv.content), encoding="utf-8-sig")) == 1
        public = client.get("/api/v1/sh000001/latest.csv")
        shadow = client.get("/admin/models/baseline/latest.csv")
        assert public.headers["X-Model-Release"] == "option_moneyflow_release"
        assert shadow.content != public.content
        assert client.get("/admin/models/nonexistent/latest.csv").status_code == 404
        assert client.get("/admin/models?days=0").status_code == 422
        assert client.get("/admin/models?sample=bogus").status_code == 422


def test_health_reports_a_configured_portfolio_that_is_not_activated(portfolio_service):
    service = portfolio_service
    before = service.health()
    assert before["status"] == "model_mismatch"
    assert before["csv_available"] is True
    assert before["runtime_matches"] is False
    published = activate(service)
    after = service.health()
    assert after["runtime_matches"] is True
    assert after["model_release"] == published.release_id
    assert after["algorithm_id"] == "fixed_option_moneyflow"


@pytest.mark.parametrize("portfolio_service", [True], indirect=True)
def test_new_release_uses_its_own_optional_inputs_without_rewriting_old_snapshot(portfolio_service):
    service = portfolio_service
    original, snapshot = service._current_canonical_features()
    old_bytes = Path(snapshot.features_path).read_bytes()
    candidate = _features(8)
    candidate["hangseng_gap_lag1"] = [0.01] * 7 + [0.02]
    portfolio.publish_portfolio(service, candidate, source="test_activation", actor="test", activate=True)
    current, _ = service._current_canonical_features()
    assert current.hangseng_gap_lag1.iloc[-1] == 0.02
    assert pd.isna(original.hangseng_gap_lag1.iloc[-1])
    assert Path(snapshot.features_path).read_bytes() == old_bytes
    changed = _features(9)
    changed["hangseng_gap_lag1"] = [0.01] * 7 + [0.99, 0.03]
    from prediction_service.engine import HistoricalMarketDataDriftError
    with pytest.raises(HistoricalMarketDataDriftError):
        portfolio.publish_portfolio(service, changed, source="tushare", actor="test")
