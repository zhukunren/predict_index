import numpy as np
import pandas as pd
import pytest

from tools.capital_features import capital_day, capital_features, MEASURES, COVERAGE_COLUMNS
from tools.fetch_capital_context import CapitalRequests
from tools.moneyflow_features import AMOUNTS


def inputs(count=2):
    codes = [f"{600000 + n}.SH" for n in range(count)]
    date = 20230103
    daily = pd.DataFrame({"ts_code": codes, "trade_date": date, "pct_chg": 1., "amount": 100., "vol": 10.})
    daily.loc[0, "pct_chg"] = -1
    basic = daily.loc[:, ["ts_code", "trade_date"]].assign(total_mv=1.)
    basic.loc[0, "total_mv"] = 9.
    flow = daily.loc[:, ["ts_code", "trade_date"]].assign(**{name: 10. for name in AMOUNTS})
    flow.loc[0, "sell_lg_amount"] = 30.
    flow.loc[1:, "buy_lg_amount"] = 30.
    return basic, daily, flow


def test_capitalization_exposes_large_stock_disagreement_and_preserves_sources():
    basic, daily, flow = inputs()
    originals = [frame.copy(deep=True) for frame in (basic, daily, flow)]
    result = capital_day(basic, daily, flow, 20230103)
    assert result["cap_up_fraction"] == pytest.approx(0.1)
    assert result["cap_vs_equal_up"] == pytest.approx(-0.4)
    assert result["cap_return"] == pytest.approx(-0.008)
    assert result["cap_net_ratio"] == pytest.approx(-0.16)
    assert result["cap_selling_fraction"] == pytest.approx(0.9)
    assert result["large_cap_net_ratio"] == pytest.approx(-0.2)
    assert result["small_cap_net_ratio"] == pytest.approx(0.2)
    for current, original in zip((basic, daily, flow), originals):
        pd.testing.assert_frame_equal(current, original)


def test_weights_are_scale_and_row_order_invariant_and_exclude_other_markets():
    basic, daily, flow = inputs()
    reference = capital_day(basic, daily, flow, 20230103)
    basic["total_mv"] *= 10000
    for code in ("000001.SZ", "900001.SH", "830001.BJ"):
        basic = pd.concat([basic, basic.iloc[:1].assign(ts_code=code, total_mv=1e12)], ignore_index=True)
        daily = pd.concat([daily, daily.iloc[:1].assign(ts_code=code, pct_chg=-50)], ignore_index=True)
        flow = pd.concat([flow, flow.iloc[:1].assign(ts_code=code)], ignore_index=True)
    assert capital_day(basic.iloc[::-1], daily.iloc[::-1], flow.iloc[::-1], 20230103) == reference


@pytest.mark.parametrize("bad", ["missing", "zero", "nan"])
def test_unknown_capitalization_is_not_imputed_from_high_stock_coverage(bad):
    basic, daily, flow = inputs(100)
    if bad == "missing":
        basic = basic.iloc[1:]
    else:
        basic.loc[0, "total_mv"] = 0 if bad == "zero" else np.nan
    result = capital_day(basic, daily, flow, 20230103)
    assert result["basic_stock_coverage"] == 0.99
    assert all(np.isnan(result[column]) for column in MEASURES)


def test_missing_large_stock_flow_fails_cap_coverage_despite_count_and_amount():
    basic, daily, flow = inputs(100)
    basic.loc[0, "total_mv"] = 100.
    result = capital_day(basic, daily, flow.iloc[1:], 20230103)
    assert result["flow_stock_coverage"] == result["flow_amount_coverage"] == 0.99
    assert result["flow_cap_coverage"] == pytest.approx(99 / 199)
    assert np.isnan(result["cap_net_ratio"])


@pytest.mark.parametrize("bad", ["duplicate", "future_date", "negative", "infinite"])
def test_invalid_capitalization_is_rejected(bad):
    basic, daily, flow = inputs()
    if bad == "duplicate":
        basic = pd.concat([basic, basic.iloc[:1]])
    elif bad == "future_date":
        basic.loc[0, "trade_date"] = 20230104
    else:
        basic.loc[0, "total_mv"] = -1 if bad == "negative" else np.inf
    with pytest.raises(ValueError):
        capital_day(basic, daily, flow, 20230103)


def test_lagged_prefix_and_missing_capital_coverage_propagation():
    dates = pd.Series(pd.bdate_range("2023-01-02", periods=80).strftime("%Y%m%d").astype(int))
    daily = pd.DataFrame({"trade_date": dates, **{name: 1. for name in COVERAGE_COLUMNS},
                          **{name: np.arange(len(dates)) / 100 for name in MEASURES}})
    reference = capital_features(dates.iloc[20:], daily, dates)
    assert reference.capital_source_date.iloc[0] == dates.iloc[19]
    assert reference.capital_cap_net_mean_5.iloc[0] == pytest.approx(.17)
    prefix = capital_features(dates.iloc[20:51], daily.iloc[:50], dates.iloc[:51])
    pd.testing.assert_frame_equal(reference.iloc[:31], prefix, check_exact=True)
    changed = daily.copy()
    changed.loc[50:, "cap_net_ratio"] *= -1
    pd.testing.assert_frame_equal(reference.iloc[:31], capital_features(dates.iloc[20:], changed, dates).iloc[:31], check_exact=True)
    changed = daily.copy()
    changed.loc[35, "flow_cap_coverage"] = .9
    result = capital_features(dates.iloc[20:], changed, dates)
    assert result.loc[~result.capital_available, "trade_date"].tolist() == dates.iloc[36:41].tolist()
    with pytest.raises(ValueError, match="Missing capital sessions"):
        capital_features(dates.iloc[20:], daily.drop(index=35), dates)


def test_fetch_errors_do_not_expose_credentials():
    class FailingClient:
        def query(self, *args, **kwargs):
            raise RuntimeError("SECRET_TOKEN")

    with pytest.raises(RuntimeError, match="credentials omitted") as caught:
        CapitalRequests(FailingClient()).fetch(20230103)
    assert "SECRET_TOKEN" not in str(caught.value)


def test_daily_basic_requests_page_instead_of_truncating():
    class Client:
        offsets = []

        def query(self, name, **kwargs):
            assert name == "daily_basic"
            self.offsets.append(kwargs["offset"])
            offset = kwargs["offset"]
            count = 6000 if not offset else 1
            return pd.DataFrame({"ts_code": [f"{n:06d}.SH" for n in range(offset, offset + count)],
                                 "trade_date": 20230103, "total_mv": 1.})

    client = Client()
    assert len(CapitalRequests(client).fetch(20230103)) == 6001
    assert client.offsets == [0, 6000]
