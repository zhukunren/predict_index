from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import 循环验证脚本 as core
from tools import directional_error_calibration as calibration
from tools.evaluate_selective_context import selective_predictions


def _frame(rows=170):
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, rows)
    actual = np.where(rng.random(rows) < 0.35, labels, 1 - labels)
    returns = np.where(actual == 1, 0.01, -0.01)
    raw = np.where(labels == 1, 0.003, -0.003)
    return pd.DataFrame({
        "trade_date": pd.bdate_range("2023-01-03", periods=rows).strftime("%Y%m%d").astype(int),
        "predicted_label": labels, "predicted_pct_change": raw,
        "uncalibrated_predicted_return": raw, "predicted_close": 100 * (1 + raw),
        "confidence": 0.55 + rng.uniform(0, 0.2, rows), "calibrated_confidence": 0.6,
        "real_pct_change": returns, "correct": labels == actual,
    })


def test_error_model_is_causal_and_preserves_incumbent(monkeypatch):
    monkeypatch.setattr(calibration, "COMMON", calibration.COMMON | {"min_train_rows": 60})
    frame = _frame()
    before = frame.copy(deep=True)
    result = calibration.corrected_predictions(core, frame, "directional_error_year")
    assert result.correction_selected.sum() > 5
    fitted = result.loc[result.error_model_training_rows.gt(0)]
    assert fitted.error_model_last_training_date.lt(fitted.trade_date).all()
    pd.testing.assert_frame_equal(frame, before)
    columns = ["predicted_label", "predicted_pct_change", "predicted_close", "calibrated_confidence",
               "correction_selected", "error_model_correctness_probability", "error_model_last_training_date"]
    for current in (80, 103):
        prefix = frame.iloc[:current + 1].copy()
        prefix.loc[current, "real_pct_change"] = np.nan
        prefix["correct"] = prefix["correct"].astype("boolean")
        prefix.loc[current, "correct"] = None
        observed = calibration.corrected_predictions(core, prefix, "directional_error_year")
        pd.testing.assert_frame_equal(result.loc[:current, columns], observed[columns])
    frame.loc[103:, "real_pct_change"] *= -1
    frame.loc[104:, "predicted_label"] = 1 - frame.loc[104:, "predicted_label"]
    changed = calibration.corrected_predictions(core, frame, "directional_error_year")
    pd.testing.assert_frame_equal(result.loc[:103, columns], changed.loc[:103, columns])


def test_side_features_only_count_settled_prior_results():
    frame = _frame(6)
    frame["predicted_label"] = [1, 1, 0, 1, 0, 1]
    frame["real_pct_change"] = [0.01, -0.01, np.nan, 0.01, -0.01, 0.01]
    features = calibration.error_features(frame)
    assert features.loc[0, "side_1_accuracy"] == 0.5
    assert features.loc[2, "side_1_accuracy"] == 0.5
    assert features.loc[3, "side_0_accuracy"] == 0.5
    assert features.loc[5, "side_1_accuracy"] == pytest.approx(7 / 13)
    assert features.loc[5, "side_0_accuracy"] == pytest.approx(6 / 11)


def test_selective_threshold_uses_opposite_direction_probability_symmetrically():
    champion = _frame(4)
    champion["predicted_label"] = [1, 1, 0, 0]
    candidate = champion.copy()
    candidate["predicted_label"] = [0, 0, 1, 1]
    candidate["residual_probability_up"] = [0.3, 0.49, 0.7, 0.51]
    result = selective_predictions(core, champion, candidate, 0.6)
    assert result.predicted_label.tolist() == [0, 1, 1, 0]
    assert result.correction_selected.tolist() == [True, False, True, False]
    candidate.loc[0, "trade_date"] += 1
    with pytest.raises(ValueError, match="identical signal dates"):
        selective_predictions(core, champion, candidate, 0.6)


def test_error_threshold_rebuilds_proposal_from_probabilities_not_previous_decisions():
    champion = _frame(4)
    champion["predicted_label"] = [1, 1, 0, 0]
    model = champion.copy()
    model["error_model_correctness_probability"] = [0.44, 0.49, 0.44, 0.49]
    model["error_model_training_rows"] = 120
    model["error_model_last_training_date"] = 20220101
    result = calibration.error_threshold_predictions(core, champion, model, 0.55)
    assert result.predicted_label.tolist() == [0, 1, 1, 0]
    unchanged = calibration.error_threshold_predictions(core, champion, model, 0.6)
    assert not unchanged.correction_selected.any()
    model.loc[0, "error_model_correctness_probability"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calibration.error_threshold_predictions(core, champion, model, 0.6)


def _auxiliary(frame):
    candidate = frame.copy()
    candidate["residual_probability_up"] = np.where(frame.predicted_label == 1, 0.3, 0.7)
    candidate["residual_training_rows"] = 120
    candidate["residual_last_training_signal"] = frame.trade_date - 1
    candidate.loc[:9, "residual_training_rows"] = 0
    return {"context": candidate}


@pytest.mark.parametrize("name", calibration.STACKED_CANDIDATES)
def test_stacked_predictions_are_causal_and_require_fitted_auxiliaries(name, monkeypatch):
    monkeypatch.setattr(calibration, "COMMON", calibration.COMMON | {"min_train_rows": 60})
    champion = _frame()
    auxiliary = _auxiliary(champion)
    result = calibration.corrected_predictions(core, champion, name, auxiliary=auxiliary)
    assert result.error_model_training_rows.iloc[:70].eq(0).all()
    assert result.error_model_training_rows.iloc[70] == 60
    assert result.correction_selected.any()
    current = 103
    prefix = champion.iloc[:current + 1].copy()
    prefix_aux = {key: value.iloc[:current + 1].copy() for key, value in auxiliary.items()}
    prefix.loc[current, "real_pct_change"] = np.nan
    prefix_aux["context"].loc[current, "real_pct_change"] = np.nan
    observed = calibration.corrected_predictions(core, prefix, name, auxiliary=prefix_aux)
    columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence", "error_model_correctness_probability"]
    pd.testing.assert_frame_equal(result.loc[:current, columns], observed[columns])
    champion.loc[current:, "real_pct_change"] *= -1
    auxiliary["context"].loc[current:, "real_pct_change"] *= -1
    auxiliary["context"].loc[current + 1:, "residual_probability_up"] = 0.99
    changed = calibration.corrected_predictions(core, champion, name, auxiliary=auxiliary)
    pd.testing.assert_frame_equal(result.loc[:current, columns], changed.loc[:current, columns])


def test_stacking_rejects_misalignment_and_noncausal_auxiliary_predictions():
    champion = _frame()
    auxiliary = _auxiliary(champion)
    auxiliary["context"].loc[50, "residual_last_training_signal"] = champion.trade_date.iloc[50]
    with pytest.raises(ValueError, match="precede"):
        calibration.corrected_predictions(core, champion, "stacked_error", auxiliary=auxiliary)
    auxiliary = _auxiliary(champion)
    auxiliary["context"].loc[50, "real_pct_change"] *= -1
    with pytest.raises(ValueError, match="identical targets"):
        calibration.corrected_predictions(core, champion, "stacked_error", auxiliary=auxiliary)
