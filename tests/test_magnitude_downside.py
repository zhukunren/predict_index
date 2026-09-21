import numpy as np
import pandas as pd
import pytest

import tools.magnitude_downside as magnitude
from tools.downside_specialist import downside_probabilities, specialist_predictions
import tushare_prediction_pipeline as pipeline
from test_downside_specialist import inputs


@pytest.mark.parametrize("name", ("downside_logistic", "downside_shallow"))
def test_weighted_models_are_causal_and_keep_original_down_signals(name, monkeypatch):
    monkeypatch.setattr(magnitude, "COMMON", magnitude.COMMON | {"min_train_rows": 40, "train_window": 100})
    frame, features = inputs(200)
    frame["real_pct_change"] *= np.linspace(.1, 2., len(frame))
    original = frame.copy(deep=True)
    scores = magnitude.magnitude_scores(frame, features, name)
    result = specialist_predictions(pipeline.prediction_core, frame, scores, .55)
    assert result.correction_selected.sum() > 5
    assert result.loc[frame.predicted_label.eq(0), "predicted_label"].eq(0).all()
    trained = scores.downside_training_rows.gt(0)
    assert (scores.loc[trained, "magnitude_effective_training_rows"] <= scores.loc[trained, "downside_training_rows"] + 1e-12).all()
    assert scores.loc[trained, "magnitude_largest_weight_share"].between(0, 1).all()
    columns = ["predicted_label", "predicted_pct_change", "predicted_close", "calibrated_confidence", *scores.columns.drop("trade_date")]
    for cutoff in (90, 135):
        prefix = frame.iloc[:cutoff + 1].copy()
        prefix.loc[cutoff, "real_pct_change"] = np.nan
        p = magnitude.magnitude_scores(prefix, features.iloc[:cutoff + 1], name)
        r = specialist_predictions(pipeline.prediction_core, prefix, p, .55)
        pd.testing.assert_frame_equal(result.loc[:cutoff, columns], r[columns], check_exact=True)
    altered = frame.copy()
    altered.loc[135:, "real_pct_change"] *= -100
    changed_features = features.copy()
    changed_features.loc[136:, "x"] *= -100
    changed = magnitude.magnitude_scores(altered, changed_features, name)
    pd.testing.assert_frame_equal(scores.iloc[:136], changed.iloc[:136], check_exact=True)
    pd.testing.assert_frame_equal(frame, original)


def test_magnitude_weighting_distinguishes_frequent_small_drops_from_large_rises(monkeypatch):
    monkeypatch.setattr(magnitude, "COMMON", magnitude.COMMON | {"min_train_rows": 40, "train_window": 100})
    frame, features = inputs(180)
    frame["predicted_label"] = 1
    frame["real_pct_change"] = np.where(np.arange(len(frame)) % 5 == 0, .1, -.01)
    features["x"] = 0.
    scores = magnitude.magnitude_scores(frame, features, "downside_logistic")
    reference = downside_probabilities(frame, features, "downside_logistic")
    assert reference.downside_probability.iloc[-1] > .7
    assert scores.downside_probability.iloc[-1] < .4
    scaled = frame.copy()
    scaled["real_pct_change"] *= 100
    result = magnitude.magnitude_scores(scaled, features, "downside_logistic")
    np.testing.assert_allclose(scores.downside_probability, result.downside_probability, rtol=0, atol=1e-12)


def test_zero_missing_and_unavailable_samples_have_no_training_weight(monkeypatch):
    monkeypatch.setattr(magnitude, "COMMON", magnitude.COMMON | {"min_train_rows": 40, "train_window": 100})
    frame, features = inputs(180)
    frame["predicted_label"] = 1
    frame.loc[0:4, "real_pct_change"] = 0.
    frame.loc[5:9, "real_pct_change"] = np.nan
    available = np.ones(len(frame), dtype=bool)
    available[10:15] = False
    features.loc[10:14, "x"] = np.nan
    scores = magnitude.magnitude_scores(frame, features, "downside_logistic", available=available)
    assert scores.downside_training_rows.iloc[:55].eq(0).all()
    assert scores.downside_training_rows.iloc[55] == 40
    assert scores.downside_last_training_date.iloc[55] == frame.trade_date.iloc[54]
    invalid = frame.copy()
    invalid.loc[0, "real_pct_change"] = np.inf
    with pytest.raises(ValueError, match="finite or unknown"):
        magnitude.magnitude_scores(invalid, features, "downside_logistic", available=available)
