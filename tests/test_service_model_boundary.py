"""The service must preserve the model's decisions, including long up runs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prediction_service import engine
from prediction_service.config import Settings
from prediction_service.models import ModelRelease, PredictionLedger
from prediction_service.service import PredictionDriftError, PredictionService
from sqlalchemy import select
from test_prediction_service import _fake_calculation, _features


@pytest.mark.parametrize("labels", [[1] * 24, [1, 1, 1, 0] * 6])
@pytest.mark.parametrize("with_diagnostics", [False, True])
def test_service_preserves_forecasts_despite_direction_concentration(tmp_path, monkeypatch, labels, with_diagnostics):
    market = _features(len(labels))
    expected = pd.DataFrame({
        "trade_date": market.trade_date.dt.strftime("%Y%m%d").astype(int),
        "predicted_label": labels,
        "predicted_pct_change": np.where(labels, 0.002, -0.003),
        "confidence": np.linspace(0.58, 0.72, len(labels)),
        "calibrated_confidence": np.linspace(0.52, 0.61, len(labels)),
        "real_pct_change": market.target_next_return,
    })
    expected["predicted_close"] = market.close * (1 + expected.predicted_pct_change)
    expected["correct"] = expected.predicted_label.eq(expected.real_pct_change.gt(0)).astype("boolean")
    expected.loc[expected.real_pct_change.isna(), "correct"] = pd.NA
    expected_diagnostics = expected.copy()
    expected_diagnostics["model_evidence"] = "unchanged"
    settings = Settings.for_test(tmp_path, validation_days=len(labels) - 1)

    def model_result(frame, *, validation_days, signal_engine, progress, diagnostics_output_path=None):
        assert validation_days == settings.validation_days
        assert signal_engine == settings.signal_engine
        if diagnostics_output_path is not None:
            expected_diagnostics.to_csv(diagnostics_output_path, index=False, encoding="utf-8-sig")
        return expected.copy(deep=True)

    monkeypatch.setattr(engine.pipeline, "run_validation_and_prediction", model_result)
    if with_diagnostics:
        path = tmp_path / "diagnostics.csv"
        actual, diagnostics = engine.calculate_results_with_diagnostics(market, settings, path)
        pd.testing.assert_frame_equal(diagnostics, pd.read_csv(path, encoding="utf-8-sig"), check_exact=True)
        assert diagnostics.model_evidence.eq("unchanged").all()
    else:
        actual = engine.calculate_results(market, settings)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_restoring_original_release_checks_its_ledger_and_preserves_other_releases(tmp_path: Path):
    original = PredictionService(Settings.for_test(tmp_path, validation_days=2), calculation_function=_fake_calculation)
    rejected = PredictionService(Settings.for_test(tmp_path, validation_days=2, signal_engine="volatility_rule"), calculation_function=_fake_calculation)
    try:
        original.initialize(bootstrap=False)
        first = original.publish_from_features(_features(), source="test_original", raw_frames=None, actor="test")
        with rejected.database.session() as session:
            target = rejected._ensure_release_for_settings(session, rejected.settings)
        altered = rejected.publish_from_features(_features(), source="test_rejected", raw_frames=None,
                                                actor="test", release_id_override=target.id)
        with original.database.session() as session:
            fingerprints = {row.id: row.prediction_fingerprint for row in session.scalars(select(PredictionLedger)).all()}

        def corrupt(frame, settings):
            result = _fake_calculation(frame, settings)
            result.loc[result.index[0], "confidence"] = 0.9
            return result

        original.calculation_function = corrupt
        with pytest.raises(PredictionDriftError, match="预测漂移"):
            original.publish_from_features(_features(9), source="test_restore", raw_frames=None,
                                           actor="test", release_id_override=first.release_id)
        assert original._load_active_context()[0].snapshot_id == altered.snapshot_id

        original.calculation_function = _fake_calculation
        restored = original.publish_from_features(_features(9), source="test_restore", raw_frames=None,
                                                 actor="test", release_id_override=first.release_id)
        assert restored.release_id == first.release_id
        assert restored.data_as_of > first.data_as_of
        assert len(restored.public_frame) == 3
        assert original.recompute_active().csv_sha256 == restored.csv_sha256
        with original.database.session() as session:
            assert session.get(ModelRelease, altered.release_id) is not None
            for identifier, expected in fingerprints.items():
                assert session.get(PredictionLedger, identifier).prediction_fingerprint == expected
    finally:
        original.shutdown()
        rejected.shutdown()
