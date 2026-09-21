from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.hangseng_timing import CANDIDATES, timed_features


def _inputs():
    dates = pd.bdate_range("2024-01-02", periods=80)
    positions = np.arange(len(dates))
    close = 100 * np.cumprod(1 + 0.005 * np.sin(positions * 0.3))
    raw = pd.DataFrame({"trade_date": dates, "open": close * 0.999, "high": close * 1.01,
                        "low": close * 0.99, "close": close, "vol": 1000 + positions * 10})
    market = pd.DataFrame({"trade_date": dates, "close": 100, "hangseng_ret1_lag1": 0.1,
                           "hangseng_vol_chg_lag1": 0.2})
    return market, raw


@pytest.mark.parametrize("name", CANDIDATES)
def test_timed_features_have_causal_prefixes(name):
    market, raw = _inputs()
    before = market.copy(deep=True)
    full, availability = timed_features(market, raw, name, publication_hour=18)
    prefix, prefix_availability = timed_features(market.iloc[:51], raw.iloc[:51], name, publication_hour=18)
    pd.testing.assert_frame_equal(full.iloc[:51], prefix)
    pd.testing.assert_frame_equal(availability.iloc[:51], prefix_availability)
    raw.loc[51:, ["close", "open", "high", "low", "vol"]] *= 2
    changed, _ = timed_features(market, raw, name, publication_hour=18)
    pd.testing.assert_frame_equal(full.iloc[:51], changed.iloc[:51])
    pd.testing.assert_frame_equal(market, before)
    if name == "current_prices":
        raw.loc[50, "vol"] *= 5
        altered_volume, _ = timed_features(market, raw, name, publication_hour=18)
        pd.testing.assert_frame_equal(full.iloc[:51], altered_volume.iloc[:51])


def test_previous_available_handles_different_market_sessions_without_extra_lag():
    market, raw = _inputs()
    raw = raw.drop(index=50)
    prepared, availability = timed_features(market, raw, "previous_available", publication_hour=15)
    assert availability.price_source_date.iloc[50] == market.trade_date.iloc[49]
    assert availability.price_source_date.iloc[51] == market.trade_date.iloc[49]
    expected = raw.set_index("trade_date").close.pct_change().loc[market.trade_date.iloc[49]]
    assert prepared.hangseng_ret1_lag1.iloc[50] == pytest.approx(expected)
    with pytest.raises(ValueError, match="closing auction"):
        timed_features(market, raw, "current_prices", publication_hour=15)
