from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import downside_ensemble as ensemble


def inputs(rows=180):
    dates = pd.Series(pd.bdate_range("2023-01-02", periods=rows).strftime("%Y%m%d").astype(int))
    champion = pd.DataFrame({"trade_date": dates, "predicted_label": 1,
                             "real_pct_change": np.where(np.arange(rows) % 7 == 0, 0.01, -0.01)})
    left = pd.DataFrame({"trade_date": dates, "downside_probability": 0.8,
                        "downside_training_rows": 120, "downside_last_training_date": 20221230})
    right = left.copy()
    right["downside_probability"] = 0.2
    return champion, left, right


def test_weights_follow_prior_probabilistic_accuracy_and_match_causal_prefixes():
    champion, left, right = inputs()
    original = left.copy(deep=True)
    full = ensemble.adaptive_downside_probabilities(champion, left, right)
    assert not full.downside_weight_applied.iloc[:60].any()
    assert full.downside_left_weight.iloc[:60].eq(0.5).all()
    assert full.downside_left_weight.iloc[100] > 0.95
    fitted = full.loc[full.downside_weight_applied]
    assert fitted.downside_weight_last_training_date.lt(fitted.trade_date).all()
    prefix = champion.iloc[:101].copy()
    prefix.loc[100, "real_pct_change"] = np.nan
    replay = ensemble.adaptive_downside_probabilities(prefix, left.iloc[:101], right.iloc[:101])
    pd.testing.assert_frame_equal(full.iloc[:101], replay, check_exact=True)
    champion.loc[100:, "real_pct_change"] *= -1
    left.loc[101:, "downside_probability"] = 0.01
    right.loc[101:, "downside_probability"] = 0.99
    changed = ensemble.adaptive_downside_probabilities(champion, left, right)
    pd.testing.assert_frame_equal(full.iloc[:101], changed.iloc[:101], check_exact=True)
    pd.testing.assert_frame_equal(left.iloc[:101], original.iloc[:101])


def test_unfitted_unresolved_and_original_down_rows_do_not_train_weights(monkeypatch):
    monkeypatch.setattr(ensemble, "PARAMETERS", ensemble.PARAMETERS | {"window": 80, "minimum_rows": 20})
    champion, left, right = inputs()
    champion.loc[::5, "predicted_label"] = 0
    champion.loc[31:39, "real_pct_change"] = np.nan
    right.loc[70:79, "downside_training_rows"] = 0
    right.loc[70:79, "downside_last_training_date"] = 0
    full = ensemble.adaptive_downside_probabilities(champion, left, right)
    assert full.downside_probability.iloc[70:80].eq(0).all()
    assert not full.loc[champion.predicted_label.eq(0), "downside_weight_applied"].any()
    eligible = champion.predicted_label.eq(1) & champion.real_pct_change.notna() & right.downside_training_rows.gt(0)
    assert full.downside_weight_training_rows.iloc[-1] == int(eligible.iloc[-81:-1].sum())
    champion.loc[champion.predicted_label.eq(0), "real_pct_change"] *= -1
    champion.loc[70:79, "real_pct_change"] *= -1
    altered = ensemble.adaptive_downside_probabilities(champion, left, right)
    pd.testing.assert_frame_equal(full, altered, check_exact=True)


def test_equal_forecasts_retain_equal_weights():
    champion, left, _ = inputs()
    full = ensemble.adaptive_downside_probabilities(champion, left, left)
    assert full.downside_left_weight.eq(0.5).all()
    assert not full.downside_weight_applied.any()
    np.testing.assert_array_equal(full.downside_probability, left.downside_probability)


def test_invalid_dates_and_training_dates_are_rejected():
    champion, left, right = inputs()
    bad = champion.copy()
    bad.loc[0, "trade_date"] = 20221230
    with pytest.raises(ValueError, match="identical"):
        ensemble.adaptive_downside_probabilities(bad, left, right)
    right.loc[0, "downside_last_training_date"] = right.trade_date.iloc[0]
    with pytest.raises(ValueError, match="precede"):
        ensemble.adaptive_downside_probabilities(champion, left, right)
