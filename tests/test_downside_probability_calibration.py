from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.downside_probability_calibration import calibrate_downside_probabilities


def inputs(rows=320):
    rng = np.random.default_rng(264)
    dates = pd.bdate_range("2023-01-02", periods=rows).strftime("%Y%m%d").astype(int)
    signal = rng.uniform(0.1, 0.9, rows)
    champion = pd.DataFrame({"trade_date": dates, "predicted_label": (np.arange(rows) % 5 != 0).astype(int),
                             "real_pct_change": np.where(signal > 0.55, -0.01, 0.01)})
    probabilities = pd.DataFrame({"trade_date": dates, "downside_probability": signal * 0.5,
                                   "downside_training_rows": 120,
                                   "downside_last_training_date": np.r_[20221230, dates[:-1]]})
    return champion, probabilities


def test_calibration_corrects_understated_scores_without_current_outcomes():
    champion, probabilities = inputs()
    original = probabilities.copy(deep=True)
    full = calibrate_downside_probabilities(champion, probabilities)
    assert full.downside_calibration_applied.sum() > 100
    assert full.downside_probability.gt(0.8).sum() > 20
    assert full.loc[champion.predicted_label.eq(0), "downside_calibration_applied"].eq(False).all()
    applied = full.loc[full.downside_calibration_applied]
    assert applied.downside_calibration_last_date.lt(applied.trade_date).all()
    pd.testing.assert_frame_equal(probabilities, original)
    for position in (101, 267):
        prefix = champion.iloc[:position + 1].copy()
        prefix.loc[position, "real_pct_change"] = np.nan
        replay = calibrate_downside_probabilities(prefix, probabilities.iloc[:position + 1])
        pd.testing.assert_frame_equal(full.iloc[:position + 1], replay, check_exact=True)
    champion.loc[267:, "real_pct_change"] *= -1
    probabilities.loc[268:, "downside_probability"] = 0.99
    changed = calibrate_downside_probabilities(champion, probabilities)
    pd.testing.assert_frame_equal(full.iloc[:268], changed.iloc[:268], check_exact=True)


def test_unfitted_or_unresolved_predictions_and_original_down_signals_cannot_train():
    champion, probabilities = inputs()
    probabilities.loc[:19, "downside_training_rows"] = 0
    champion.loc[20:29, "real_pct_change"] = np.nan
    original = calibrate_downside_probabilities(champion, probabilities)
    excluded = champion.predicted_label.eq(0) | probabilities.downside_training_rows.eq(0) | champion.real_pct_change.isna()
    probabilities.loc[excluded, "downside_probability"] = 0.99
    changed = calibrate_downside_probabilities(champion, probabilities)
    pd.testing.assert_series_equal(original.loc[~excluded, "downside_probability"], changed.loc[~excluded, "downside_probability"])
    assert original.downside_calibration_rows.iloc[99] == int((~excluded).iloc[:99].sum())


def test_calibrated_selection_keeps_down_labels_and_replays_final_forecast_fields():
    import tushare_prediction_pipeline as pipeline
    from tools.downside_specialist import specialist_predictions

    champion, probabilities = inputs(180)
    raw = np.where(champion.predicted_label.eq(1), 0.003, -0.003)
    champion["predicted_pct_change"] = raw
    champion["uncalibrated_predicted_return"] = raw
    champion["predicted_close"] = 100 * (1 + raw)
    champion["confidence"] = 0.65
    champion["calibrated_confidence"] = 0.6
    champion["correct"] = champion.predicted_label.eq(champion.real_pct_change.gt(0))
    original = champion.copy(deep=True)
    full = specialist_predictions(pipeline.prediction_core, champion,
                                 calibrate_downside_probabilities(champion, probabilities), 0.6)
    assert full.correction_selected.sum() > 10
    assert full.loc[champion.predicted_label.eq(0), "predicted_label"].eq(0).all()
    current = 137
    prefix = champion.iloc[:current + 1].copy()
    prefix.loc[current, "real_pct_change"] = np.nan
    prefix["correct"] = prefix.correct.astype("boolean")
    prefix.loc[current, "correct"] = None
    replay = specialist_predictions(pipeline.prediction_core, prefix,
                                   calibrate_downside_probabilities(prefix, probabilities.iloc[:current + 1]), 0.6)
    columns = ["predicted_label", "predicted_pct_change", "predicted_close", "confidence", "calibrated_confidence",
               "downside_probability", "correction_selected", "downside_calibration_last_date"]
    pd.testing.assert_frame_equal(full.loc[:current, columns], replay[columns], check_exact=True)
    pd.testing.assert_frame_equal(champion, original)


@pytest.mark.parametrize("problem", ["misaligned", "future_training", "nonfinite", "out_of_range", "nonbinary"])
def test_invalid_probability_history_is_rejected(problem):
    champion, probabilities = inputs()
    if problem == "misaligned":
        probabilities.loc[0, "trade_date"] -= 1
    elif problem == "future_training":
        probabilities.loc[0, "downside_last_training_date"] = probabilities.trade_date.iloc[0]
    elif problem == "nonfinite":
        probabilities.loc[0, "downside_probability"] = np.nan
    elif problem == "out_of_range":
        probabilities.loc[0, "downside_probability"] = 1.1
    else:
        champion.loc[0, "predicted_label"] = 2
    with pytest.raises(ValueError):
        calibrate_downside_probabilities(champion, probabilities)
