from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import 循环验证脚本 as core
from tools.directional_error_calibration import corrected_predictions
from tools.rule_context_features import ranked_context, rule_context_features


OPTIONS = {
    "nested_rule_threshold_window": 80, "nested_rule_min_threshold_rows": 20,
    "nested_rule_calibration_window": 60, "nested_rule_min_calibration_rows": 30, "rule_top_k": 5,
}


def _market(rows=440):
    rng = np.random.default_rng(101)
    close = 100 * np.exp(rng.normal(0, 0.01, rows).cumsum())
    prior = np.r_[100, close[:-1]]
    open_ = prior * (1 + rng.normal(0, 0.003, rows))
    return pd.DataFrame({
        "trade_date": pd.bdate_range("2021-01-01", periods=rows).strftime("%Y-%m-%d"),
        "open": open_, "high": np.maximum(close, open_) * 1.005,
        "low": np.minimum(close, open_) * 0.995, "close": close, "pre_close": prior,
        "volume": rng.uniform(1e6, 2e6, rows), "amount": rng.uniform(1e8, 2e8, rows),
    })


def _cache(data, config):
    base = core._normalize_market_frame(data, config)
    features = core._clean_feature_frame(core._build_features(base, config))
    cache = core._build_nested_rule_cache(base=base, features=features, threshold_window=80, min_threshold_rows=20)
    return base, cache


def _streams(data, config):
    base, cache = _cache(data, config)
    positions = np.arange(80, len(data))
    labels = np.array([core._nested_volatility_rule_signal(
        base=base, cache=cache, idx=int(idx), calibration_window=60, min_calibration_rows=30, top_k=5,
    ).predicted_label for idx in positions])
    rng = np.random.default_rng(501)
    actual = np.where(rng.random(len(positions)) < 0.2, labels, 1 - labels)
    raw = np.where(labels == 1, 0.003, -0.003)
    champion = pd.DataFrame({
        "trade_date": base.date.iloc[positions].dt.strftime("%Y%m%d").astype(int).to_numpy(),
        "predicted_label": labels, "predicted_pct_change": raw, "uncalibrated_predicted_return": raw,
        "predicted_close": 100 * (1 + raw), "confidence": 0.6, "calibrated_confidence": 0.6,
        "real_pct_change": np.where(actual == 1, 0.01, -0.01), "correct": labels == actual,
    })
    diagnostics = champion[["trade_date", "predicted_label", "real_pct_change", "correct"]].copy()
    diagnostics["base_predicted_label"] = labels
    diagnostics["veto_applied"] = 0
    diagnostics["recent_failure_guard_applied"] = 0
    return champion, diagnostics


def test_reconstructed_votes_match_production_and_are_causal():
    data = _market(240)
    config = core.DirectionPredictionConfig(external_feature_mode="none", min_train_sequences=20)
    champion, diagnostics = _streams(data, config)
    result = rule_context_features(core, data, champion, diagnostics, config, OPTIONS)
    assert result.rule_last_calibration_date.lt(result.trade_date).all()
    assert result.rule_effective_experts.between(1, 5).all()
    assert result.rule_margin.between(0, 1).all()
    current = 103
    signal_position = current + 80
    prefix = rule_context_features(core, data.iloc[:signal_position + 1], champion.iloc[:current + 1],
                                   diagnostics.iloc[:current + 1], config, OPTIONS)
    pd.testing.assert_frame_equal(prefix, result.iloc[:current + 1], check_exact=True)
    changed = data.copy()
    changed.loc[signal_position + 1:, ["open", "high", "low", "close", "pre_close"]] *= 1.2
    changed_diagnostics = diagnostics.copy()
    changed_diagnostics["real_pct_change"] *= -1
    changed_diagnostics["correct"] = ~changed_diagnostics.correct
    altered = rule_context_features(core, changed, champion.iloc[:current + 1],
                                    changed_diagnostics.iloc[:current + 1], config, OPTIONS)
    pd.testing.assert_frame_equal(altered, result.iloc[:current + 1], check_exact=True)


def test_cache_current_label_and_future_predictions_do_not_change_context():
    data = _market(180)
    _, cache = _cache(data, core.DirectionPredictionConfig(external_feature_mode="none", min_train_sequences=20))
    kwargs = {"idx": 130, "calibration_window": 60, "min_calibration_rows": 30, "top_k": 5}
    result = ranked_context(cache, **kwargs)
    cache.labels[130:] = 1 - cache.labels[130:]
    cache.predictions[:, 131:] = 1 - cache.predictions[:, 131:]
    assert ranked_context(cache, **kwargs) == result


def test_alignment_and_reconstruction_mismatch_are_rejected():
    data = _market(120)
    config = core.DirectionPredictionConfig(external_feature_mode="none", min_train_sequences=20)
    champion, diagnostics = _streams(data, config)
    diagnostics.loc[10, "base_predicted_label"] = 1 - diagnostics.base_predicted_label.iloc[10]
    with pytest.raises(ValueError, match="Rebuilt ranked vote"):
        rule_context_features(core, data, champion, diagnostics, config, OPTIONS)
    diagnostics.loc[10, "trade_date"] += 1
    with pytest.raises(ValueError, match="identical incumbent dates"):
        rule_context_features(core, data, champion, diagnostics, config, OPTIONS)


def test_rule_error_model_is_causal_and_keeps_incumbent_immutable():
    data = _market()
    config = core.DirectionPredictionConfig(external_feature_mode="none", min_train_sequences=20)
    champion, diagnostics = _streams(data, config)
    context = rule_context_features(core, data, champion, diagnostics, config, OPTIONS)
    before = champion.copy(deep=True)
    result = corrected_predictions(core, champion, "rule_context_error", rule_context=context)
    pd.testing.assert_frame_equal(champion, before)
    assert not result.correction_selected.iloc[:252].any()
    assert result.correction_selected.sum() > 5
    current = 303
    prefix = champion.iloc[:current + 1].copy()
    prefix.loc[current, "real_pct_change"] = np.nan
    prefix["correct"] = prefix.correct.astype("boolean")
    prefix.loc[current, "correct"] = None
    observed = corrected_predictions(core, prefix, "rule_context_error", rule_context=context.iloc[:current + 1])
    columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence", "error_model_correctness_probability"]
    pd.testing.assert_frame_equal(result.loc[:current, columns], observed[columns], check_exact=True)
    champion.loc[current:, "real_pct_change"] *= -1
    context.loc[current + 1:, "rule_margin"] = 1 - context.loc[current + 1:, "rule_margin"]
    changed = corrected_predictions(core, champion, "rule_context_error", rule_context=context)
    pd.testing.assert_frame_equal(result.loc[:current, columns], changed.loc[:current, columns], check_exact=True)
    context.loc[5, "rule_last_calibration_date"] = champion.trade_date.iloc[5]
    with pytest.raises(ValueError, match="calibration must precede"):
        corrected_predictions(core, champion, "rule_context_error", rule_context=context)
