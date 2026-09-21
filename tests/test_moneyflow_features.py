from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.moneyflow_features import AMOUNTS, MEASURES, PRICE_MEASURES, aggregate_moneyflow, moneyflow_features
from tools.fetch_moneyflow_context import MoneyflowRequests


def inputs():
    codes = ["000001.SH", "000002.SZ"]
    flow = pd.DataFrame({"ts_code": codes, "trade_date": 20230103, **{name: 10.0 for name in AMOUNTS}})
    flow.loc[0, ["buy_lg_amount", "sell_sm_amount"]] = 30
    flow.loc[1, ["sell_lg_amount", "buy_sm_amount"]] = 30
    daily = pd.DataFrame({"ts_code": codes, "trade_date": 20230103, "amount": 100.0, "vol": 10.0, "pct_chg": [1, -1]})
    return flow, daily


def test_signed_flows_and_weighted_selling_coverage():
    flow, daily = inputs()
    original = flow.copy(deep=True)
    result = aggregate_moneyflow(flow, daily, 20230103)
    assert result["large_net_ratio"] == pytest.approx(0)
    assert result["small_net_ratio"] == pytest.approx(0)
    assert result["large_selling_fraction"] == 0.5
    assert result["large_selling_amount_fraction"] == 0.5
    assert result["large_small_divergence"] == 0.5
    assert result["shanghai_relative"] == pytest.approx(20 / 120)
    pd.testing.assert_frame_equal(flow, original)


def test_missing_or_zero_flow_has_explicit_coverage_not_zero_net_flow():
    flow, daily = inputs()
    result = aggregate_moneyflow(flow.iloc[:1], daily, 20230103)
    assert result["stock_coverage"] == 0.5
    assert result["amount_coverage"] == 0.5
    assert np.isnan(result["large_net_ratio"])
    flow.loc[1, AMOUNTS] = 0
    result = aggregate_moneyflow(flow, daily, 20230103)
    assert result["matched_rows"] == 1
    assert np.isnan(result["large_net_ratio"])


def test_inactive_and_non_target_exchange_stocks_are_excluded():
    flow, daily = inputs()
    extra = flow.iloc[:1].assign(ts_code="000003.BJ")
    flow = pd.concat([flow, extra], ignore_index=True)
    daily.loc[1, ["vol", "amount"]] = 0
    result = aggregate_moneyflow(flow, daily, 20230103)
    assert result["active_rows"] == result["matched_rows"] == 1
    assert result["stock_coverage"] == 1
    assert result["large_net_ratio"] == pytest.approx(20 / 120)


def test_high_stock_coverage_cannot_hide_missing_high_turnover_stock():
    codes = [f"{index:06d}.SH" for index in range(100)]
    daily = pd.DataFrame({"ts_code": codes, "trade_date": 20230103, "amount": 1.0, "vol": 10.0, "pct_chg": 1.0})
    daily.loc[99, "amount"] = 100
    flow = pd.DataFrame({"ts_code": codes[:-1], "trade_date": 20230103, **{name: 10.0 for name in AMOUNTS}})
    result = aggregate_moneyflow(flow, daily, 20230103)
    assert result["stock_coverage"] == 0.99
    assert result["amount_coverage"] == pytest.approx(99 / 199)
    assert np.isnan(result["large_net_ratio"])


def test_price_conditioned_flows_and_turnover_groups_use_current_cross_section():
    flow, daily = inputs()
    daily["pct_chg"] = [-1, 1]
    daily["amount"] = [10, 100]
    result = aggregate_moneyflow(flow, daily, 20230103, price_context=True)
    assert result["up_sell_amount_fraction"] == 0.5
    assert result["down_buy_amount_fraction"] == 0.5
    assert result["high_turnover_large_ratio"] == pytest.approx(-20 / 120)
    assert result["other_turnover_large_ratio"] == pytest.approx(20 / 120)
    original = aggregate_moneyflow(flow, daily, 20230103)
    assert {key: value for key, value in result.items() if key not in PRICE_MEASURES} == original


@pytest.mark.parametrize("problem", ["duplicate", "wrong_date", "negative", "nonfinite"])
def test_invalid_source_rows_are_rejected(problem):
    flow, daily = inputs()
    if problem == "duplicate":
        flow = pd.concat([flow, flow.iloc[:1]])
    elif problem == "wrong_date":
        flow.loc[0, "trade_date"] = 20230104
    elif problem == "negative":
        flow.loc[0, "buy_lg_amount"] = -1
    else:
        flow.loc[0, "buy_lg_amount"] = np.nan
    with pytest.raises(ValueError):
        aggregate_moneyflow(flow, daily, 20230103)


def feature_inputs():
    dates = pd.Series(pd.bdate_range("2023-01-02", periods=90).strftime("%Y%m%d").astype(int))
    daily = pd.DataFrame({"trade_date": dates, "stock_coverage": 1.0, "amount_coverage": 1.0,
                          **{name: np.arange(90) / 1000 for name in MEASURES}})
    return dates, daily


def test_feature_prefix_lag_and_missing_coverage_propagation():
    dates, daily = feature_inputs()
    full = moneyflow_features(dates.iloc[20:], daily, dates)
    assert full.moneyflow_source_date.iloc[0] == dates.iloc[19]
    assert full.moneyflow_large_net_mean_20.iloc[0] == pytest.approx(0.0095)
    prefix = moneyflow_features(dates.iloc[20:51], daily.iloc[:50], dates.iloc[:51])
    pd.testing.assert_frame_equal(full.iloc[:31], prefix, check_exact=True)
    daily.loc[50:, "large_net_ratio"] *= -1
    changed = moneyflow_features(dates.iloc[20:], daily, dates)
    pd.testing.assert_frame_equal(full.iloc[:31], changed.iloc[:31], check_exact=True)
    daily.loc[35, "stock_coverage"] = 0.9
    unavailable = moneyflow_features(dates.iloc[20:], daily, dates)
    assert unavailable.loc[~unavailable.moneyflow_available, "trade_date"].tolist() == dates.iloc[36:56].tolist()


def test_entire_missing_session_is_rejected():
    dates, daily = feature_inputs()
    with pytest.raises(ValueError, match="Missing moneyflow"):
        moneyflow_features(dates.iloc[20:], daily.drop(index=35), dates)


def test_price_context_preserves_old_features_and_has_prefix_parity():
    dates, daily = feature_inputs()
    original = moneyflow_features(dates.iloc[20:], daily, dates)
    for name in PRICE_MEASURES:
        daily[name] = np.arange(len(daily)) / 1000
    full = moneyflow_features(dates.iloc[20:], daily, dates, price_context=True)
    pd.testing.assert_frame_equal(full.loc[:, original.columns], original, check_exact=True)
    assert full.moneyflow_up_sell_mean_5.iloc[0] == pytest.approx(0.017)
    prefix = moneyflow_features(dates.iloc[20:51], daily.iloc[:50], dates.iloc[:51], price_context=True)
    pd.testing.assert_frame_equal(full.iloc[:31], prefix, check_exact=True)
    daily.loc[50:, "up_sell_amount_fraction"] = 0.9
    changed = moneyflow_features(dates.iloc[20:], daily, dates, price_context=True)
    pd.testing.assert_frame_equal(full.iloc[:31], changed.iloc[:31], check_exact=True)


def normalized_inputs():
    dates = pd.Series(pd.bdate_range("2022-01-03", periods=210).strftime("%Y%m%d").astype(int))
    value = np.sin(np.arange(len(dates)) * 0.37) / 100 + np.arange(len(dates)) / 10000
    daily = pd.DataFrame({"trade_date": dates, "stock_coverage": 1.0, "amount_coverage": 1.0,
                          **{name: value.copy() for name in (*MEASURES, *PRICE_MEASURES)}})
    return dates, daily


def test_normalized_flow_uses_only_previous_source_values_and_clips_extremes():
    dates, daily = normalized_inputs()
    original = daily.copy(deep=True)
    full = moneyflow_features(dates.iloc[20:], daily, dates, price_context=True, normalize=True)
    history = daily.large_net_ratio.iloc[23:149]
    expected = (daily.large_net_ratio.iloc[149] - history.mean()) / history.std(ddof=0)
    assert full.loc[full.trade_date.eq(dates.iloc[150]), "moneyflow_large_net_ratio"].item() == pytest.approx(expected)
    assert full.loc[full.moneyflow_available, "trade_date"].iloc[0] == dates.iloc[146]
    assert not full.loc[full.trade_date.lt(dates.iloc[146]), "moneyflow_available"].any()
    pd.testing.assert_frame_equal(daily, original)
    prefix = moneyflow_features(dates.iloc[20:151], daily.iloc[:150], dates.iloc[:151], price_context=True, normalize=True)
    pd.testing.assert_frame_equal(full.iloc[:131], prefix, check_exact=True)
    daily.loc[150:, "large_net_ratio"] = 1000
    changed = moneyflow_features(dates.iloc[20:], daily, dates, price_context=True, normalize=True)
    pd.testing.assert_frame_equal(full.iloc[:131], changed.iloc[:131], check_exact=True)
    assert changed.loc[changed.trade_date.eq(dates.iloc[151]), "moneyflow_large_net_ratio"].item() == 5


def test_normalization_does_not_fill_missing_or_zero_scale_history():
    dates, daily = normalized_inputs()
    daily.loc[155, "stock_coverage"] = 0.9
    full = moneyflow_features(dates.iloc[20:], daily, dates, normalize=True)
    assert full.loc[full.trade_date.eq(dates.iloc[155]), "moneyflow_available"].item()
    assert not full.loc[full.trade_date.ge(dates.iloc[156]), "moneyflow_available"].any()
    dates, daily = normalized_inputs()
    daily["small_net_ratio"] = 0.25
    full = moneyflow_features(dates.iloc[20:], daily, dates, normalize=True)
    assert full.moneyflow_small_net_ratio.isna().all()
    assert not full.moneyflow_available.any()


def test_request_pagination_and_exception_sanitizing(monkeypatch):
    import tools.fetch_moneyflow_context as fetcher
    monkeypatch.setattr(fetcher, "PAGE_ROWS", 2)
    monkeypatch.setattr(fetcher, "REQUEST_INTERVAL", 0)
    flow, _ = inputs()
    calls = []
    class Client:
        def query(self, api, **kwargs):
            assert api == "moneyflow"
            calls.append(kwargs["offset"])
            return flow if kwargs["offset"] == 0 else flow.iloc[:1].assign(ts_code="000003.SH")
    result = MoneyflowRequests(Client()).fetch(20230103)
    assert calls == [0, 2]
    assert len(result) == 3
    class Failed:
        def query(self, *args, **kwargs):
            raise RuntimeError("SECRET_TOKEN")
    with pytest.raises(RuntimeError, match="credentials omitted") as failure:
        MoneyflowRequests(Failed()).fetch(20230103)
    assert "SECRET_TOKEN" not in str(failure.value)
