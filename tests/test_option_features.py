from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools.fetch_option_context import fetch_pages
from tools.option_features import UNDERLYING, aggregate_options, option_features


def inputs():
    dates = pd.bdate_range("2024-01-02", periods=50).strftime("%Y%m%d").astype(int)
    contracts = pd.DataFrame({"ts_code": ["p1.SH", "c1.SH", "p2.SH", "c2.SH"],
                              "opt_code": UNDERLYING, "call_put": ["P", "C", "P", "C"],
                              "list_date": [dates[0], dates[0], dates[25], dates[25]],
                              "delist_date": [dates[24], dates[24], dates[-1], dates[-1]]})
    rows = []
    for index, date in enumerate(dates):
        for right, volume in (("p", 100 + index), ("c", 300 - index)):
            rows.append({"trade_date": date, "ts_code": f"{right}{1 if index < 25 else 2}.SH",
                         "vol": volume, "amount": volume * 0.1, "oi": volume * 5})
    return pd.DataFrame(rows), contracts, dates


def test_historical_contracts_and_activity_are_preserved_through_expiration():
    raw, contracts, dates = inputs()
    before = raw.copy(deep=True)
    unrelated = raw.iloc[[0]].copy()
    unrelated["ts_code"] = "other.SH"
    daily = aggregate_options(pd.concat([raw, unrelated]), contracts, dates)
    assert daily.contract_rows.eq(2).all()
    assert daily.loc[0, "P_volume"] == 100
    assert daily.loc[0, "C_volume"] == 300
    assert daily.loc[25, "P_volume"] == 125
    assert daily.loc[25, "C_volume"] == 275
    features = option_features(dates[20:], daily, dates)
    assert features.loc[0, "option_put_volume_share"] == pytest.approx(119 / 400)
    assert features.loc[0, "option_put_volume_change_1"] == pytest.approx(1 / 400)
    assert features.loc[0, "option_put_volume_mean_20"] == pytest.approx(109.5 / 400)
    assert features.option_source_date.tolist() == list(dates[19:-1])
    pd.testing.assert_frame_equal(raw, before)


def test_option_features_have_prefix_parity_and_do_not_read_current_activity():
    raw, contracts, dates = inputs()
    daily = aggregate_options(raw, contracts, dates)
    full = option_features(dates[20:], daily, dates)
    prefix = option_features(dates[20:36], daily.iloc[:35], dates[:36])
    pd.testing.assert_frame_equal(full.iloc[:16], prefix, check_exact=True)
    changed = daily.copy()
    changed.loc[35:, ["P_volume", "P_amount", "P_interest"]] *= 10
    observed = option_features(dates[20:], changed, dates)
    pd.testing.assert_frame_equal(full.iloc[:16], observed.iloc[:16], check_exact=True)
    with pytest.raises(ValueError, match="Missing prior-session"):
        option_features(dates[20:], daily.drop(index=26), dates)


def test_same_day_options_require_evening_publication_and_ignore_later_sessions():
    raw, contracts, dates = inputs()
    daily = aggregate_options(raw, contracts, dates)
    with pytest.raises(ValueError, match="18:30"):
        option_features(dates[20:], daily, dates, lag_sessions=0)
    full = option_features(dates[20:], daily, dates, lag_sessions=0, publication_hour=18, publication_minute=30)
    assert full.option_source_date.tolist() == list(dates[20:])
    assert full.option_put_volume_share.iloc[0] == pytest.approx(120 / 400)
    prefix = option_features(dates[20:36], daily.iloc[:36], dates[:36], lag_sessions=0, publication_hour=18, publication_minute=30)
    pd.testing.assert_frame_equal(full.iloc[:16], prefix, check_exact=True)
    daily.loc[36:, ["P_volume", "P_amount", "P_interest"]] *= 10
    changed = option_features(dates[20:], daily, dates, lag_sessions=0, publication_hour=18, publication_minute=30)
    pd.testing.assert_frame_equal(full.iloc[:16], changed.iloc[:16], check_exact=True)
    with pytest.raises(ValueError, match="same-day"):
        option_features(dates[20:], daily.iloc[:-1], dates, lag_sessions=0, publication_hour=18, publication_minute=30)


@pytest.mark.parametrize("problem", ["missing", "expired_only", "future_only", "duplicate", "nonfinite", "negative", "wrong_underlying", "unknown_right"])
def test_invalid_or_incomplete_option_universe_is_rejected(problem):
    raw, contracts, dates = inputs()
    if problem == "missing":
        raw = raw.drop(index=10)
    elif problem == "expired_only":
        contracts = contracts.iloc[:2]
    elif problem == "future_only":
        contracts = contracts.iloc[2:]
    elif problem == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]])
    elif problem == "nonfinite":
        raw["vol"] = raw.vol.astype(float)
        raw.loc[0, "vol"] = np.nan
    elif problem == "negative":
        raw.loc[0, "vol"] = -1
    elif problem == "wrong_underlying":
        contracts.loc[0, "opt_code"] = "OP510300.SH"
    else:
        contracts.loc[0, "call_put"] = "unknown"
    with pytest.raises(ValueError):
        aggregate_options(raw, contracts, dates)


def test_zero_activity_contract_is_included_but_missing_activity_is_not_zero():
    raw, contracts, dates = inputs()
    extra = contracts.iloc[[0]].copy()
    extra["ts_code"] = "inactive.SH"
    extra["delist_date"] = dates[-1]
    contracts = pd.concat([contracts, extra])
    zero = pd.DataFrame({"ts_code": "inactive.SH", "trade_date": dates, "vol": 0, "amount": 0, "oi": 0})
    daily = aggregate_options(pd.concat([raw, zero]), contracts, dates)
    assert daily.contract_rows.eq(3).all()
    assert daily.P_contract_rows.eq(2).all()
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate_options(raw, contracts, dates)


def test_pagination_requires_distinct_complete_pages_and_redacts_errors(monkeypatch):
    monkeypatch.setattr("tools.fetch_option_context.time.sleep", lambda seconds: None)
    pages = [pd.DataFrame({"ts_code": ["a", "b"], "trade_date": [20240102, 20240102]}),
             pd.DataFrame({"ts_code": ["c"], "trade_date": [20240102]})]
    offsets = []
    def query(api, **kwargs):
        offsets.append(kwargs["offset"])
        return pages[len(offsets) - 1]
    params = {"start_date": "20240101", "end_date": "20240131"}
    result = fetch_pages(SimpleNamespace(query=query), "opt_daily", params, page_rows=2, maximum_rows=8)
    assert len(result) == 3
    assert offsets == [0, 2]
    with pytest.raises(ValueError, match="pagination exceeded"):
        fetch_pages(SimpleNamespace(query=lambda *args, **kwargs: pages[0]), "opt_daily", params, page_rows=2, maximum_rows=4)
    def failed(*args, **kwargs):
        raise RuntimeError("provider confidential details")
    with pytest.raises(RuntimeError, match="credentials omitted") as error:
        fetch_pages(SimpleNamespace(query=failed), "opt_daily", params, page_rows=2, maximum_rows=4)
    assert "confidential" not in str(error.value)
