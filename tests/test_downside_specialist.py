from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tushare_prediction_pipeline as pipeline
from tools import downside_specialist as specialist
from tools.breadth_features import GROUPS, MEASURES


def inputs(rows=200):
    rng = np.random.default_rng(42)
    x = rng.normal(size=rows)
    labels = (np.arange(rows) % 4 != 0).astype(int)
    actual = np.where(x > 0, -0.01, 0.01)
    raw = np.where(labels, 0.003, -0.003)
    frame = pd.DataFrame({
        "trade_date": pd.bdate_range("2023-01-02", periods=rows).strftime("%Y%m%d").astype(int),
        "predicted_label": labels, "predicted_pct_change": raw,
        "uncalibrated_predicted_return": raw, "predicted_close": 100 * (1 + raw),
        "confidence": 0.65, "calibrated_confidence": 0.6,
        "real_pct_change": actual, "correct": labels == (actual > 0),
    })
    features = pd.DataFrame({"trade_date": frame.trade_date, "x": x})
    return frame, features


@pytest.mark.parametrize("name", specialist.CANDIDATES)
@pytest.mark.parametrize("training_policy", specialist.TRAINING_POLICIES)
def test_real_model_is_conditional_causal_and_preserves_down_predictions(name, training_policy, monkeypatch):
    monkeypatch.setattr(specialist, "COMMON", specialist.COMMON | {"min_train_rows": 40, "train_window": 100})
    frame, features = inputs()
    original = frame.copy(deep=True)
    model = specialist.downside_probabilities(frame, features, name, training_policy=training_policy)
    result = specialist.specialist_predictions(pipeline.prediction_core, frame, model, 0.55)
    assert result.correction_selected.sum() > 5
    assert result.loc[frame.predicted_label.eq(0), "predicted_label"].eq(0).all()
    assert result.loc[frame.predicted_label.eq(0), "downside_training_rows"].eq(0).all()
    assert result.downside_training_rows.max() == 75
    fitted = result.loc[result.downside_training_rows.gt(0)]
    assert fitted.downside_last_training_date.lt(fitted.trade_date).all()
    pd.testing.assert_frame_equal(frame, original)
    columns = ["predicted_label", "predicted_pct_change", "predicted_close", "calibrated_confidence",
               "correction_selected", "downside_probability", "downside_last_training_date"]
    for current in (79, 130):
        prefix = frame.iloc[:current + 1].copy()
        prefix.loc[current, "real_pct_change"] = np.nan
        prefix["correct"] = prefix.correct.astype("boolean")
        prefix.loc[current, "correct"] = None
        probabilities = specialist.downside_probabilities(prefix, features.iloc[:current + 1], name,
                                                          training_policy=training_policy)
        replay = specialist.specialist_predictions(pipeline.prediction_core, prefix, probabilities, 0.55)
        pd.testing.assert_frame_equal(result.loc[:current, columns], replay[columns], check_exact=True)
    frame.loc[130:, "real_pct_change"] *= -1
    features.loc[131:, "x"] *= -100
    changed = specialist.downside_probabilities(frame, features, name, training_policy=training_policy)
    pd.testing.assert_frame_equal(model.iloc[:131], changed.iloc[:131], check_exact=True)
    frame = original.copy()
    features = inputs()[1]
    frame.loc[frame.predicted_label.eq(0), "real_pct_change"] *= -100
    features.loc[frame.predicted_label.eq(0), "x"] *= 100
    changed = specialist.downside_probabilities(frame, features, name, training_policy=training_policy)
    pd.testing.assert_frame_equal(model, changed, check_exact=True)


def test_recent_weights_use_signal_age_and_preserve_regularization_scale():
    history = np.array([0, 126, 252])
    weights = specialist.training_weights(history, 253, "recent")
    np.testing.assert_allclose(weights, np.array([1, 2, 4]) * 3 / 7)
    assert specialist.training_weights(history, 253, "uniform") is None
    frame, features = inputs()
    with pytest.raises(ValueError, match="training policy"):
        specialist.downside_probabilities(frame, features, "downside_logistic", training_policy="invalid")


def test_mean_probabilities_require_both_models_and_do_not_mutate_inputs():
    base = pd.DataFrame({
        "trade_date": [20230104, 20230105, 20230106],
        "downside_probability": [0.4, 0.6, 0.8],
        "downside_training_rows": [130, 0, 140],
        "downside_last_training_date": [20230103, 0, 20230104],
    })
    extended = base.copy()
    extended["downside_probability"] = [0.8, 0.9, 0.7]
    extended["downside_training_rows"] = [120, 120, 0]
    extended["downside_last_training_date"] = [20230102, 20230104, 0]
    extended.index = [10, 20, 30]
    original = base.copy(deep=True)
    result = specialist.mean_downside_probabilities(base, extended)
    np.testing.assert_allclose(result.downside_probability, [0.6, 0, 0])
    assert result.downside_training_rows.tolist() == [120, 0, 0]
    assert result.downside_last_training_date.tolist() == [20230103, 0, 0]
    pd.testing.assert_frame_equal(base, original)
    pd.testing.assert_frame_equal(result.iloc[:2], specialist.mean_downside_probabilities(base.iloc[:2], extended.iloc[:2]))
    extended.loc[10, "downside_last_training_date"] = 20230104
    with pytest.raises(ValueError, match="precede"):
        specialist.mean_downside_probabilities(base, extended)
    extended.loc[10, "trade_date"] = 20230103
    with pytest.raises(ValueError, match="identical"):
        specialist.mean_downside_probabilities(base, extended)


def test_threshold_respects_fitted_flag_and_incumbent_direction():
    frame, _ = inputs(5)
    model = pd.DataFrame({
        "trade_date": frame.trade_date, "downside_probability": [0.99, 0.65, 0.64, 0.99, 0.99],
        "downside_training_rows": [120, 120, 120, 0, 120],
        "downside_last_training_date": 20221230,
    })
    result = specialist.specialist_predictions(pipeline.prediction_core, frame, model, 0.65)
    assert result.predicted_label.tolist() == [0, 0, 1, 1, 0]
    assert result.correction_selected.tolist() == [False, True, False, False, False]
    model.loc[1, "downside_last_training_date"] = frame.trade_date.iloc[1]
    with pytest.raises(ValueError, match="precede"):
        specialist.specialist_predictions(pipeline.prediction_core, frame, model, 0.65)


@pytest.mark.parametrize("problem", ["misaligned", "nonfinite", "nonbinary", "unordered"])
def test_invalid_model_inputs_are_rejected(problem):
    frame, features = inputs()
    if problem == "misaligned":
        features.loc[0, "trade_date"] -= 1
    elif problem == "nonfinite":
        features.loc[0, "x"] = np.inf
    elif problem == "nonbinary":
        frame["predicted_label"] = frame.predicted_label.astype(float)
        frame.loc[0, "predicted_label"] = 0.5
    else:
        frame = frame.iloc[::-1]
    with pytest.raises(ValueError):
        specialist.downside_probabilities(frame, features, "downside_logistic")


def test_combined_features_have_prefix_parity_and_reject_missing_current_breadth():
    rows = 180
    dates = pd.bdate_range("2023-01-02", periods=rows)
    close = 100 * np.cumprod(1 + 0.005 * np.sin(np.arange(rows) * 0.37))
    market = pd.DataFrame({
        "trade_date": dates.strftime("%Y-%m-%d"), "open": close * 0.999,
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "pre_close": np.r_[100, close[:-1]], "vol": 10000, "amount": 1000000,
    })
    frame, _ = inputs(rows)
    frame = frame.iloc[40:].reset_index(drop=True)
    breadth = pd.DataFrame({"trade_date": dates.strftime("%Y%m%d").astype(int)})
    for group in GROUPS:
        for measure in MEASURES:
            breadth[f"{group}_{measure}"] = 0.3 if "fraction" in measure else 0.001
    overseas_dates = pd.bdate_range("2022-10-03", dates[-1])
    overseas = {name: pd.DataFrame({"trade_date": overseas_dates.strftime("%Y%m%d").astype(int),
                                    "close": 100 + np.arange(len(overseas_dates)) * 0.1})
                for name in ("spx", "nasdaq")}
    config = pipeline.prediction_core.DirectionPredictionConfig(lookback=10, min_train_sequences=30)
    full = specialist.specialist_features(pipeline.prediction_core, market, frame, config, breadth, overseas)
    prefix = specialist.specialist_features(pipeline.prediction_core, market.iloc[:131], frame.iloc[:91], config, breadth, overseas)
    pd.testing.assert_frame_equal(full.iloc[:91], prefix, check_exact=True)
    market.loc[131:, ["close", "open", "high", "low"]] *= 10
    breadth.loc[131:, "all_up_fraction"] = 0.9
    for asset in overseas.values():
        asset.loc[asset.trade_date.ge(frame.trade_date.iloc[90]), "close"] *= 3
    changed = specialist.specialist_features(pipeline.prediction_core, market, frame, config, breadth, overseas)
    pd.testing.assert_frame_equal(full.iloc[:91], changed.iloc[:91], check_exact=True)
    with pytest.raises(ValueError, match="Missing lagged breadth"):
        specialist.specialist_features(pipeline.prediction_core, market, frame, config, breadth.iloc[:-1], overseas)
