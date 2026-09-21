import numpy as np
import pandas as pd
import pytest

from tools.capital_close_features import PRICE_COLUMNS, PRICE_MEASURES, capital_price_day, capital_close_features
from tools.capital_features import COVERAGE_COLUMNS, FEATURE_COLUMNS, MEASURES, capital_features


def series():
    dates = pd.Series(pd.bdate_range("2023-01-02", periods=80).strftime("%Y%m%d").astype(int))
    lagged = pd.DataFrame({"trade_date": dates, **{name: 1. for name in COVERAGE_COLUMNS},
                           **{name: np.arange(len(dates)) / 100 for name in MEASURES}})
    prices = lagged.loc[:, ["trade_date", "basic_stock_coverage", *PRICE_MEASURES]].copy()
    return dates, lagged, prices


def test_current_prices_replace_only_price_features_and_keep_lagged_flows():
    dates, lagged, prices = series()
    baseline = capital_features(dates.iloc[20:], lagged, dates)
    current = capital_close_features(dates.iloc[20:], lagged, prices, dates)
    flows = [column for column in FEATURE_COLUMNS if column not in PRICE_COLUMNS]
    pd.testing.assert_frame_equal(current[flows], baseline[flows], check_exact=True)
    assert current.capital_source_date.iloc[0] == dates.iloc[19]
    assert current.capital_price_source_date.iloc[0] == dates.iloc[20]
    assert current.capital_cap_return.iloc[0] == .2
    assert current.capital_cap_return_mean_5.iloc[0] == pytest.approx(.18)
    assert current.capital_cap_up_change_1.iloc[0] == pytest.approx(.01)


def test_current_price_prefix_is_exact_and_future_inputs_do_not_change_history():
    dates, lagged, prices = series()
    complete = capital_close_features(dates.iloc[20:], lagged, prices, dates)
    prefix = capital_close_features(dates.iloc[20:51], lagged.iloc[:50], prices.iloc[:51], dates.iloc[:51])
    pd.testing.assert_frame_equal(complete.iloc[:31], prefix, check_exact=True)
    prices.loc[51:, "cap_return"] = -1.
    lagged.loc[50:, "cap_net_ratio"] = -1.
    changed = capital_close_features(dates.iloc[20:], lagged, prices, dates)
    pd.testing.assert_frame_equal(complete.iloc[:31], changed.iloc[:31], check_exact=True)


def test_missing_current_market_cap_is_unavailable_for_full_rolling_window():
    dates, lagged, prices = series()
    prices.loc[35, "basic_stock_coverage"] = .99
    result = capital_close_features(dates.iloc[20:], lagged, prices, dates)
    assert result.loc[~result.capital_available, "trade_date"].tolist() == dates.iloc[35:40].tolist()
    with pytest.raises(ValueError, match="Missing same-day"):
        capital_close_features(dates.iloc[20:], lagged, prices.iloc[:-1], dates)
    with pytest.raises(ValueError, match="18:00"):
        capital_close_features(dates.iloc[20:], lagged, prices, dates, publication_hour=17)


def test_current_prices_use_complete_historical_shanghai_universe_without_flows():
    daily = pd.DataFrame({"ts_code": ["600001.SH", "600002.SH", "000001.SZ", "900001.SH", "830001.BJ"],
                          "trade_date": 20230103, "pct_chg": [-1., 1., -50., -50., -50.], "amount": 100., "vol": 10.})
    basic = daily[["ts_code", "trade_date"]].assign(total_mv=[9., 1., 1e12, 1e12, 1e12])
    result = capital_price_day(basic, daily, 20230103)
    assert result["active_rows"] == 2
    assert result["cap_return"] == pytest.approx(-.008)
    assert result["cap_up_fraction"] == .1
    assert result["cap_vs_equal_up"] == -.4
    basic["total_mv"] *= 1e4
    assert capital_price_day(basic.iloc[::-1], daily.iloc[::-1], 20230103) == result
    missing = capital_price_day(basic.iloc[1:], daily, 20230103)
    assert missing["basic_stock_coverage"] == .5
    assert all(np.isnan(missing[name]) for name in PRICE_MEASURES)
    for changed in (pd.concat([basic, basic.iloc[:1]]), basic.assign(trade_date=20230104), basic.assign(total_mv=-1)):
        with pytest.raises(ValueError):
            capital_price_day(changed, daily, 20230103)
