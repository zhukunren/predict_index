from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.distribution_features import distribution_day, distribution_features


def stocks(date, returns):
    return pd.DataFrame({"ts_code": [f"{index:06d}.SH" for index in range(len(returns))],
                         "trade_date": date, "pct_chg": returns, "amount": 100.0, "vol": 10.0})


def test_transitions_use_only_matching_active_stocks_and_correct_units():
    previous = stocks(20230102, [1, -2, -3, 1])
    current = stocks(20230103, [-1, -2, 2, 3, -4])
    current.loc[3, ["amount", "vol"]] = 0
    current.loc[4, "ts_code"] = "999999.SH"
    result = distribution_day(current, previous, 20230103, 20230102)
    assert result["active_rows"] == 4
    assert result["matched_rows"] == 3
    for key in ("advance_to_decline", "decline_persistence", "decline_to_advance"):
        assert result[key] == pytest.approx(1 / 3)
    assert result["q10"] == pytest.approx(np.quantile([-0.01, -0.02, 0.02, -0.04], 0.1))
    assert result["amount_return_gap"] == pytest.approx(0)
    assert result["amount_top_decile"] == pytest.approx(0.25)


def inputs():
    dates = pd.Series(pd.bdate_range("2023-01-02", periods=40).strftime("%Y%m%d").astype(int))
    rng = np.random.default_rng(841)
    raw = [stocks(date, rng.normal(0, 2, 20)) for date in dates]
    daily = pd.DataFrame([distribution_day(raw[i], raw[i - 1], dates[i], dates[i - 1]) for i in range(1, len(dates))])
    return dates, daily


def test_distribution_prefix_and_future_data_independence():
    dates, daily = inputs()
    original = daily.copy(deep=True)
    full = distribution_features(dates.iloc[6:], daily, dates)
    prefix = distribution_features(dates.iloc[6:26], daily.iloc[:25], dates.iloc[:26])
    pd.testing.assert_frame_equal(full.iloc[:20], prefix, check_exact=True)
    pd.testing.assert_frame_equal(daily, original)
    daily.loc[25:, "q10"] -= 0.1
    changed = distribution_features(dates.iloc[6:], daily, dates)
    pd.testing.assert_frame_equal(full.iloc[:20], changed.iloc[:20], check_exact=True)


@pytest.mark.parametrize("problem", ["source_date", "missing", "duplicate", "early_publication", "nonfinite"])
def test_unavailable_or_misaligned_distribution_is_rejected(problem):
    dates, daily = inputs()
    hour = 18
    if problem == "source_date":
        daily.loc[10, "previous_source_date"] = dates.iloc[5]
    elif problem == "missing":
        daily = daily.drop(index=10)
    elif problem == "duplicate":
        daily = pd.concat([daily, daily.iloc[:1]])
    elif problem == "early_publication":
        hour = 15
    else:
        daily.loc[10, "q10"] = np.nan
    with pytest.raises(ValueError):
        distribution_features(dates.iloc[6:], daily, dates, publication_hour=hour)


def test_duplicate_stocks_and_future_transition_source_are_rejected():
    current, previous = stocks(20230103, [1, 2]), stocks(20230102, [-1, 2])
    with pytest.raises(ValueError):
        distribution_day(current, previous, 20230103, 20230103)
    with pytest.raises(ValueError):
        distribution_day(pd.concat([current, current.iloc[:1]]), previous, 20230103, 20230102)
