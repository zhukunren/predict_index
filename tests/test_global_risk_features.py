from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import 循环验证脚本 as core
from tools.directional_error_calibration import GLOBAL_CANDIDATES, corrected_predictions
from tools.global_risk_features import global_risk_features


def _assets(rows=410):
    rng = np.random.default_rng(503)
    dates = pd.bdate_range("2022-11-01", periods=rows).strftime("%Y%m%d").astype(int)
    return {name: pd.DataFrame({"trade_date": dates, "close": 100 * np.exp(rng.normal(0, 0.01, rows).cumsum())})
            for name in ("spx", "nasdaq")}


def test_same_day_us_close_is_excluded_and_future_changes_do_not_leak():
    assets = _assets(80)
    dates = assets["spx"].trade_date.iloc[30:65].reset_index(drop=True)
    result = global_risk_features(dates, assets)
    current = int(dates.iloc[20])
    changed = {name: frame.copy() for name, frame in assets.items()}
    for frame in changed.values():
        frame.loc[frame.trade_date.ge(current), "close"] *= 1.5
    observed = global_risk_features(dates, changed)
    pd.testing.assert_frame_equal(result.iloc[:21], observed.iloc[:21])
    prefix_assets = {name: frame.loc[frame.trade_date.lt(current)].copy() for name, frame in assets.items()}
    prefix = global_risk_features(dates.iloc[:21], prefix_assets)
    pd.testing.assert_frame_equal(result.iloc[:21], prefix)
    friday = assets["spx"].loc[assets["spx"].trade_date.lt(dates.iloc[0])].iloc[-1]
    assert result.spx_source_date.iloc[0] == friday.trade_date
    assert result.spx_source_date.lt(result.trade_date).all()


def test_missing_stale_duplicate_or_short_overseas_history_is_rejected():
    assets = _assets(80)
    dates = assets["spx"].trade_date.iloc[30:65].reset_index(drop=True)
    missing = {name: frame.copy() for name, frame in assets.items()}
    missing["spx"] = missing["spx"].drop(index=40)
    with pytest.raises(ValueError, match="different session coverage"):
        global_risk_features(dates, missing)
    stale = {name: frame.iloc[:40].copy() for name, frame in assets.items()}
    with pytest.raises(ValueError, match="unavailable or stale"):
        global_risk_features(dates, stale)
    duplicate = {name: pd.concat([frame, frame.iloc[[0]]]) for name, frame in assets.items()}
    with pytest.raises(ValueError, match="duplicate"):
        global_risk_features(dates, duplicate)
    with pytest.raises(ValueError, match="21 prior sessions"):
        global_risk_features(assets["spx"].trade_date.iloc[5:10], assets)


def test_signal_index_does_not_change_date_alignment():
    assets = _assets(80)
    dates = assets["spx"].trade_date.iloc[30:65]
    expected = global_risk_features(dates.reset_index(drop=True), assets)
    pd.testing.assert_frame_equal(global_risk_features(dates, assets), expected)


def _champion(assets):
    rng = np.random.default_rng(94)
    dates = assets["spx"].trade_date.iloc[50:380].reset_index(drop=True)
    labels = rng.integers(0, 2, len(dates))
    actual = np.where(rng.random(len(dates)) < 0.25, labels, 1 - labels)
    raw = np.where(labels == 1, 0.003, -0.003)
    return pd.DataFrame({
        "trade_date": dates, "predicted_label": labels, "predicted_pct_change": raw,
        "uncalibrated_predicted_return": raw, "predicted_close": 100 * (1 + raw),
        "confidence": 0.55 + rng.uniform(0, 0.2, len(dates)), "calibrated_confidence": 0.6,
        "real_pct_change": np.where(actual == 1, 0.01, -0.01), "correct": labels == actual,
    })


@pytest.mark.parametrize("name", GLOBAL_CANDIDATES)
def test_global_model_prefix_replay_and_current_outcome_isolation(name):
    assets = _assets()
    champion = _champion(assets)
    original = champion.copy(deep=True)
    context = global_risk_features(champion.trade_date, assets)
    result = corrected_predictions(core, champion, name, context=context)
    assert result.correction_selected.sum() >= 5
    assert not result.correction_selected.iloc[:252].any()
    fitted = result.loc[result.error_model_training_rows.gt(0)]
    assert fitted.error_model_last_training_date.lt(fitted.trade_date).all()
    pd.testing.assert_frame_equal(champion, original)
    current = 303
    columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence",
               "error_model_correctness_probability", "correction_selected"]
    prefix = champion.iloc[:current + 1].copy()
    prefix.loc[current, "real_pct_change"] = np.nan
    prefix["correct"] = prefix.correct.astype("boolean")
    prefix.loc[current, "correct"] = None
    replay = corrected_predictions(core, prefix, name, context=context.iloc[:current + 1])
    pd.testing.assert_frame_equal(result.loc[:current, columns], replay[columns], check_exact=True)
    champion.loc[current:, "real_pct_change"] *= -1
    changed_assets = {key: frame.copy() for key, frame in assets.items()}
    for frame in changed_assets.values():
        frame.loc[frame.trade_date.ge(champion.trade_date.iloc[current]), "close"] *= 1.5
    changed_context = global_risk_features(champion.trade_date, changed_assets)
    changed = corrected_predictions(core, champion, name, context=changed_context)
    pd.testing.assert_frame_equal(result.loc[:current, columns], changed.loc[:current, columns], check_exact=True)


def test_model_rejects_same_day_or_misaligned_context():
    assets = _assets()
    champion = _champion(assets)
    context = global_risk_features(champion.trade_date, assets)
    context.loc[10, "spx_source_date"] = context.trade_date.iloc[10]
    with pytest.raises(ValueError, match="precede"):
        corrected_predictions(core, champion, "global_risk_error", context=context)
    context = global_risk_features(champion.trade_date, assets)
    context.loc[10, "trade_date"] += 1
    with pytest.raises(ValueError, match="identical signal dates"):
        corrected_predictions(core, champion, "global_risk_error", context=context)
