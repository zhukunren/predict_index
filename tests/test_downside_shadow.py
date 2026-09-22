from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select, func, update
from sqlalchemy.exc import IntegrityError

import prediction_service.downside_shadow as shadow
import prediction_service.service as service_module
from prediction_service.calendar import SHANGHAI
from prediction_service.models import PredictionLedger, Publication, OutcomeResolution
from prediction_service.service import PredictionDriftError
from tools.fixed_downside_candidate import canonical_json, ALGORITHM
from prediction_service.archive import sha256_bytes
from test_prediction_service import _service, _features, _fake_calculation


def calendar(service):
    days = pd.date_range("2026-01-01", "2026-02-01")
    service.calendar.store(pd.DataFrame({"cal_date": days.strftime("%Y%m%d"), "is_open": (days.weekday < 5).astype(int)}))


def clock(monkeypatch, value):
    monkeypatch.setattr(shadow, "utcnow", lambda: value.astimezone(timezone.utc))
    monkeypatch.setattr(service_module, "utcnow", lambda: value.astimezone(timezone.utc))


def contract():
    config = {"algorithm_id": ALGORITHM, "sources": {"test.py": "fixture"}, "threshold": 0.55}
    return {"release_id": sha256_bytes(canonical_json(config)), "configuration": config}


def record(service, features, *, backfill=False):
    _, snapshot, control = service._load_active_context()
    baseline = _fake_calculation(features, service.settings)
    return shadow.record_result(service, snapshot, control, contract(), baseline.copy(), baseline,
                                {}, allow_backfill=backfill)


def test_prediction_window_requires_complete_calendar_and_real_clock(tmp_path):
    service = _service(tmp_path)
    try:
        calendar(service)
        before = datetime(2026, 1, 16, 17, 59, tzinfo=SHANGHAI)
        with pytest.raises(ValueError, match="18:00"):
            shadow.prediction_window(service.calendar, "20260116", before)
        after = before + timedelta(minutes=1)
        assert shadow.prediction_window(service.calendar, "20260116", after)[0] == "20260119"
        opening = datetime(2026, 1, 19, 9, 30, tzinfo=SHANGHAI)
        with pytest.raises(ValueError, match="already opened"):
            shadow.prediction_window(service.calendar, "20260116", opening)
        shadow.prediction_window(service.calendar, "20260116", opening, allow_backfill=True)
        with pytest.raises(ValueError, match="complete exchange calendar"):
            shadow.prediction_window(service.calendar, "20260202", opening)
    finally:
        service.shutdown()


def test_new_candidate_does_not_inherit_control_time_or_change_publication(tmp_path, monkeypatch):
    service = _service(tmp_path)
    try:
        calendar(service)
        features = _features()
        early = datetime(2026, 1, 14, 18, 30, tzinfo=SHANGHAI)
        clock(monkeypatch, early)
        public = service.publish_from_features(features, source="test", raw_frames=None, actor="test")
        later = early + timedelta(hours=1)
        clock(monkeypatch, later)
        run = record(service, features)
        with service.database.session() as session:
            forecasts = session.scalars(select(PredictionLedger).where(PredictionLedger.release_id == run.release_id)).all()
            assert len(forecasts) == 3
            assert all(row.created_at.replace(tzinfo=timezone.utc) == later.astimezone(timezone.utc) for row in forecasts)
            assert session.scalar(select(func.count()).select_from(Publication)) == 1
            assert session.scalar(select(Publication).where(Publication.is_active.is_(True))).snapshot_id == public.snapshot_id
        assert service.read_published().csv_sha256 == public.csv_sha256
        assert service.shadow_dashboard_data()["run"] is None
        assert record(service, features).id == run.id
        with service.database.session() as session:
            assert session.scalar(select(func.count()).select_from(PredictionLedger).where(PredictionLedger.release_id == run.release_id)) == 3
        with pytest.raises(IntegrityError):
            with service.database.session() as session:
                session.execute(update(PredictionLedger).where(PredictionLedger.release_id == run.release_id).values(predicted_label=0))
        content = Path(run.result_csv_path).read_bytes()
        Path(run.result_csv_path).write_bytes(content + b"\n")
        with pytest.raises(PredictionDriftError):
            shadow.verify_run(service, run)
    finally:
        service.shutdown()


def test_late_replay_is_excluded_and_next_snapshot_settles_original_prediction(tmp_path, monkeypatch):
    service = _service(tmp_path)
    try:
        calendar(service)
        features = _features()
        clock(monkeypatch, datetime(2026, 1, 14, 18, 30, tzinfo=SHANGHAI))
        service.publish_from_features(features, source="test", raw_frames=None, actor="test")
        _, _, control = service._load_active_context()
        clock(monkeypatch, datetime(2026, 1, 15, 10, tzinfo=SHANGHAI))
        with pytest.raises(ValueError, match="already opened"):
            record(service, features)
        first = record(service, features, backfill=True)
        next_features = _features(9)
        clock(monkeypatch, datetime(2026, 1, 15, 18, 30, tzinfo=SHANGHAI))
        service.publish_from_features(next_features, source="test", raw_frames=None, actor="test")
        record(service, next_features)
        assert shadow.prospective_report(service, first.release_id, control.id)["prospective_paired_rows"] == 0
        final_features = _features(10)
        clock(monkeypatch, datetime(2026, 1, 16, 18, 30, tzinfo=SHANGHAI))
        service.publish_from_features(final_features, source="test", raw_frames=None, actor="test")
        record(service, final_features)
        report = shadow.prospective_report(service, first.release_id, control.id)
        assert report["prospective_paired_rows"] == 1
        assert report["candidate"]["start_date"] == 20260115
        assert report["promotion_allowed"] is False
        with service.database.session() as session:
            row = session.scalar(select(PredictionLedger).where(PredictionLedger.release_id == first.release_id,
                                                                PredictionLedger.signal_date == "20260114"))
            assert row.created_at.hour == 2
            assert session.get(OutcomeResolution, row.id).target_trade_date == "20260115"
    finally:
        service.shutdown()


def test_invalid_outcomes_and_control_drift_are_rejected(tmp_path, monkeypatch):
    service = _service(tmp_path)
    try:
        calendar(service)
        features = _features()
        clock(monkeypatch, datetime(2026, 1, 14, 18, 30, tzinfo=SHANGHAI))
        service.publish_from_features(features, source="test", raw_frames=None, actor="test")
        _, snapshot, control = service._load_active_context()
        baseline = _fake_calculation(features, service.settings)
        bad = baseline.copy()
        bad.loc[0, "real_pct_change"] = 0.2
        with pytest.raises(ValueError, match="outcomes disagree"):
            shadow.record_result(service, snapshot, control, contract(), bad, baseline, {})
        bad = baseline.copy()
        bad.loc[2, "confidence"] = 0.8
        with pytest.raises(PredictionDriftError, match="Control replay"):
            shadow.record_result(service, snapshot, control, contract(), baseline, bad, {})
    finally:
        service.shutdown()


def test_bilstm_shadow_also_uses_its_own_generation_time(tmp_path, monkeypatch):
    service = _service(tmp_path)
    try:
        service.shadow_calculation_function = _fake_calculation
        calendar(service)
        early = datetime(2026, 1, 14, 18, 30, tzinfo=SHANGHAI)
        clock(monkeypatch, early)
        public = service.publish_from_features(_features(), source="test", raw_frames=None, actor="test")
        later = early + timedelta(hours=1)
        clock(monkeypatch, later)
        run, _ = service.request_bilstm_shadow(public.snapshot_id, actor="test")
        service.run_bilstm_shadow(run.id)
        with service.database.session() as session:
            rows = session.scalars(select(PredictionLedger).where(PredictionLedger.release_id == run.release_id)).all()
            assert all(row.created_at.replace(tzinfo=timezone.utc) == later.astimezone(timezone.utc) for row in rows)
    finally:
        service.shutdown()


def test_incremental_context_fetches_only_missing_sessions(monkeypatch):
    import tools.run_downside_shadow as runner
    from tools.moneyflow_features import AMOUNTS, aggregate_moneyflow
    from tools.breadth_features import aggregate_day

    def stocks(day):
        return pd.DataFrame({"ts_code": [f"{600000 + n}.SH" for n in range(1001)],
                             "trade_date": day, "pct_chg": 1., "amount": 100., "vol": 10.})

    def flows(day):
        return stocks(day).loc[:, ["ts_code", "trade_date"]].assign(**{name: 10. for name in AMOUNTS})

    class Client:
        calls = []

        def daily(self, **kwargs):
            day = int(kwargs["trade_date"])
            self.calls.append(("daily", day))
            return stocks(day)

        def query(self, name, **kwargs):
            day = int(kwargs["trade_date"])
            self.calls.append((name, day))
            return flows(day)

        def index_global(self, **kwargs):
            self.calls.append((kwargs["ts_code"], kwargs["start_date"], kwargs["end_date"]))
            return pd.DataFrame({"trade_date": ["20260106"], "close": [101.]})

    seed = {"breadth": pd.DataFrame([aggregate_day(stocks(20260105), 20260105)]),
            "moneyflow": pd.DataFrame([aggregate_moneyflow(flows(20260105), stocks(20260105), 20260105, price_context=True)]),
            "spx": pd.DataFrame({"trade_date": [20260105], "close": [100.]}),
            "nasdaq": pd.DataFrame({"trade_date": [20260105], "close": [100.]})}
    market = pd.DataFrame({"trade_date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])})
    client = Client()
    result, raw, receipts = runner.extend_context(seed, market, client)
    assert client.calls == [("daily", 20260106), ("daily", 20260107), ("moneyflow", 20260106),
                            ("SPX", "20260106", "20260106"), ("IXIC", "20260106", "20260106")]
    assert result["moneyflow"].trade_date.tolist() == [20260105, 20260106]
    assert len(seed["breadth"]) == 1
    assert len(raw) == len(receipts) == 5
    assert all(item["received_at"].endswith("+00:00") for item in receipts)
    for name in seed:
        pd.testing.assert_frame_equal(result[name].iloc[:1], seed[name], check_exact=True)


def test_frozen_version_rejects_changed_runtime_and_seed(tmp_path, monkeypatch):
    import tools.fixed_downside_candidate as fixed

    config = {"algorithm_id": ALGORITHM, "sources": {"a.py": "original"}}
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "baseline.csv").write_bytes(b"original")
    (tmp_path / "sources.zip").write_bytes(b"source")
    frozen_config = {**config, "seed_files": {"baseline.csv": sha256_bytes(b"original")}}
    frozen = {"configuration": frozen_config, "release_id": sha256_bytes(canonical_json(frozen_config)),
              "source_archive_sha256": sha256_bytes(b"source")}
    (tmp_path / "frozen.json").write_bytes(canonical_json(frozen))
    monkeypatch.setattr(fixed, "runtime_contract", lambda: config)
    assert fixed.load_candidate(tmp_path) == frozen
    monkeypatch.setattr(fixed, "runtime_contract", lambda: {**config, "threshold": 0.6})
    with pytest.raises(ValueError, match="parameters changed"):
        fixed.load_candidate(tmp_path)
    monkeypatch.setattr(fixed, "runtime_contract", lambda: config)
    (seed / "baseline.csv").write_bytes(b"altered")
    with pytest.raises(ValueError, match="seed changed"):
        fixed.load_candidate(tmp_path)


@pytest.mark.research_artifacts
def test_frozen_candidate_records_a_real_snapshot_without_publication(tmp_path, monkeypatch):
    from tools.fixed_downside_candidate import read_frame, predict

    seed = Path(__file__).resolve().parents[1] / "artifacts/evaluation/ensemble_moneyflow_price_downside_v1"
    if not (seed / "candidate_predictions.csv").is_file():
        pytest.fail("Restore the frozen research fixture before running -m research_artifacts.")
    market = read_frame(seed / "features.csv")
    market = market.loc[pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int).le(20260916)].reset_index(drop=True)
    baseline = read_frame(seed / "baseline.csv")
    baseline.loc[baseline.index[-1], "real_pct_change"] = float("nan")
    baseline["correct"] = baseline["correct"].astype(object)
    baseline.loc[baseline.index[-1], "correct"] = None
    context = {name: read_frame(seed / f"{name}.csv") for name in ("breadth", "moneyflow", "spx", "nasdaq")}
    settings = service_module.Settings.for_test(tmp_path, validation_days=len(baseline) - 1)
    service = service_module.PredictionService(settings, calculation_function=lambda _features, _settings: baseline.copy())
    service.initialize(bootstrap=False)
    try:
        days = pd.DatetimeIndex([*pd.to_datetime(market.trade_date), pd.Timestamp("2026-09-17")])
        service.calendar.store(pd.DataFrame({"cal_date": days.strftime("%Y%m%d"), "is_open": 1}))
        first_time = datetime(2026, 9, 16, 18, 30, tzinfo=SHANGHAI)
        clock(monkeypatch, first_time)
        public = service.publish_from_features(market, source="fixture", raw_frames=None, actor="test")
        candidate = predict(market, baseline, context)
        later = datetime(2026, 9, 16, 19, 0, tzinfo=SHANGHAI)
        clock(monkeypatch, later)
        frozen = contract()
        run = shadow.record_result(service, service._load_active_context()[1], service._load_active_context()[2], frozen,
                                   candidate, baseline, context, allow_backfill=False)
        assert run.engine == ALGORITHM
        assert run.status == "succeeded"
        assert service.read_published().csv_sha256 == public.csv_sha256
        with service.database.session() as session:
            rows = session.scalars(select(PredictionLedger).where(PredictionLedger.release_id == run.release_id)).all()
            assert len(rows) == len(baseline)
            assert all(row.created_at.replace(tzinfo=timezone.utc) == later.astimezone(timezone.utc) for row in rows)
    finally:
        service.shutdown()
