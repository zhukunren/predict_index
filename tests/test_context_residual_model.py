from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import 循环验证脚本 as core
from tools import context_residual_model as residual


def _market(rows=160, phase=0.0):
    positions = np.arange(rows)
    returns = 0.006 * np.sin(positions * 0.37 + phase) + 0.003 * np.cos(positions * 0.11)
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "trade_date": pd.bdate_range("2024-01-02", periods=rows).strftime("%Y-%m-%d"),
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close,
        "pre_close": np.r_[100, close[:-1]], "vol": 10000 + positions * 13, "amount": 1000000 + positions * 29,
    })


def _inputs():
    data = _market()
    assets = {name: _market(phase=index * 0.7) for index, name in enumerate(residual.ASSETS)}
    return data, assets


def _champion(base):
    positions = np.arange(len(base))
    labels = (positions % 7 < 4).astype(int)
    predicted = np.where(labels, 0.003, -0.003)
    actual = base.close.shift(-1) / base.close - 1
    return pd.DataFrame({
        "trade_date": base.date.dt.strftime("%Y%m%d").astype(int), "predicted_label": labels,
        "predicted_pct_change": predicted, "uncalibrated_predicted_return": predicted,
        "predicted_close": base.close * (1 + predicted), "confidence": 0.62,
        "calibrated_confidence": 0.6, "real_pct_change": actual,
        "correct": pd.Series(labels == actual.gt(0)).where(actual.notna(), None),
    })


def test_context_features_do_not_read_future_prices_or_bridge_missing_dates():
    data, assets = _inputs()
    config = core.DirectionPredictionConfig(lookback=10, min_train_sequences=30)
    _, full = residual.context_features(core, data, config, assets)
    _, prefix = residual.context_features(core, data.iloc[:101], config, assets)
    pd.testing.assert_frame_equal(full.iloc[:101], prefix)
    for frame in [data, *assets.values()]:
        frame.loc[101:, ["open", "high", "low", "close"]] *= 1.7
    _, changed = residual.context_features(core, data, config, assets)
    pd.testing.assert_frame_equal(full.iloc[:101], changed.iloc[:101])
    assets["csi300"] = assets["csi300"].drop(index=20)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        residual.context_features(core, data, config, assets)


@pytest.mark.parametrize("name", ["context_residual_stumps", "context_error_long", "history_direction_balanced"])
def test_real_booster_has_causal_predictions_and_prefix_parity(name, monkeypatch):
    monkeypatch.setattr(residual, "COMMON", residual.COMMON | {
        "train_window": 90, "min_train_rows": 64, "num_boost_round": 20,
        "eta": 0.2, "min_child_weight": 1, "lambda": 1, "alpha": 0,
    })
    monkeypatch.setattr(residual, "HISTORY_CANDIDATES", {
        key: settings | {"train_window": 90} for key, settings in residual.HISTORY_CANDIDATES.items()
    })
    data, assets = _inputs()
    config = core.DirectionPredictionConfig(lookback=10, min_train_sequences=30)
    base, features = residual.context_features(core, data, config, assets)
    champion = _champion(base)
    before = champion.copy(deep=True)
    full = residual.residual_predictions(core, champion, base, features, name)
    assert full.residual_training_rows.max() == 90
    assert full.residual_probability_up.iloc[64:].nunique() > 5
    fitted = full.loc[full.residual_last_training_signal.gt(0)]
    assert fitted.residual_last_training_signal.lt(fitted.trade_date).all()
    pd.testing.assert_frame_equal(before, champion)
    columns = ["predicted_label", "predicted_pct_change", "predicted_close", "calibrated_confidence",
               "residual_probability_up", "residual_last_training_signal", "residual_training_rows"]
    for current in (64, 78, 120):
        prefix = champion.iloc[:current + 1].copy()
        prefix.loc[current, "real_pct_change"] = np.nan
        actual = residual.residual_predictions(core, prefix, base.iloc[:current + 1], features.iloc[:current + 1], name)
        pd.testing.assert_frame_equal(full.loc[:current, columns], actual[columns])
    champion.loc[120:, "real_pct_change"] = 0.4
    features.loc[121:] *= -500
    changed = residual.residual_predictions(core, champion, base, features, name)
    pd.testing.assert_frame_equal(full.loc[:120, columns], changed.loc[:120, columns])


@pytest.mark.parametrize("name", residual.BREADTH_CANDIDATES)
@pytest.mark.parametrize("lag_sessions", [0, 1])
def test_breadth_booster_uses_aligned_lagged_features_with_prefix_parity(name, lag_sessions, monkeypatch):
    from tools.breadth_features import GROUPS, MEASURES
    from tools.evaluate_breadth_direction import add_breadth

    monkeypatch.setattr(residual, "COMMON", residual.COMMON | {
        "train_window": 90, "min_train_rows": 64, "num_boost_round": 20,
        "eta": 0.2, "min_child_weight": 1, "lambda": 1, "alpha": 0,
    })
    data, assets = _inputs()
    config = core.DirectionPredictionConfig(lookback=10, min_train_sequences=30)
    base, _ = residual.context_features(core, data, config, assets)
    champion = _champion(base).iloc[20:].reset_index(drop=True)
    calendar = base.date.dt.strftime("%Y%m%d").astype(int)
    breadth = pd.DataFrame({"trade_date": calendar})
    for group in GROUPS:
        for measure in MEASURES:
            breadth[f"{group}_{measure}"] = 0.2 if "fraction" in measure else 0.001
        breadth[f"{group}_up_fraction"] = 0.5 + 0.3 * np.sin(np.arange(len(base)) * 0.27)
    timing = {"lag_sessions": lag_sessions, "publication_hour": 18}
    base, features = add_breadth(core, data, champion, config, assets, breadth, calendar, **timing)
    assert features.filter(like="breadth_").iloc[:20].isna().all().all()
    assert np.isfinite(features.iloc[20:].to_numpy()).all()
    original = champion.copy(deep=True)
    result = residual.residual_predictions(core, champion, base, features, name)
    pd.testing.assert_frame_equal(champion, original)
    assert result.residual_training_rows.max() == 90
    current = 100
    market_position = current + 20
    prefix_champion = champion.iloc[:current + 1].copy()
    prefix_champion.loc[current, "real_pct_change"] = np.nan
    prefix_base, prefix_features = add_breadth(
        core, data.iloc[:market_position + 1], prefix_champion, config, assets,
        breadth.iloc[:market_position + 1 - lag_sessions], calendar, **timing,
    )
    pd.testing.assert_frame_equal(features.iloc[:market_position + 1], prefix_features, check_exact=True)
    replay = residual.residual_predictions(core, prefix_champion, prefix_base, prefix_features, name)
    columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence", "residual_probability_up"]
    pd.testing.assert_frame_equal(result.loc[:current, columns], replay[columns], check_exact=True)
    champion.loc[current:, "real_pct_change"] = -0.05
    breadth.loc[market_position + 1 - lag_sessions:, "all_up_fraction"] = 0.99
    changed_base, changed_features = add_breadth(core, data, champion, config, assets, breadth, calendar, **timing)
    changed = residual.residual_predictions(core, champion, changed_base, changed_features, name)
    pd.testing.assert_frame_equal(result.loc[:current, columns], changed.loc[:current, columns], check_exact=True)
