from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools.fetch_futures_context import fetch_month
from tools.futures_features import PRODUCTS, aggregate_contracts, futures_features


def inputs(rows=45):
    calendar = pd.bdate_range("2024-01-02", periods=rows).strftime("%Y%m%d").astype(int)
    records = []
    for index, date in enumerate(calendar):
        for product in PRODUCTS:
            for month in (3, 4, 6, 9):
                close = 100 + month
                records.append({"trade_date": date, "ts_code": f"{product}24{month:02}.CFX",
                                "pre_close": close, "open": close, "close": close,
                                "vol": 1000 if month == (3 if index < 25 else 4) else 100,
                                "oi": 1000 + index * 10})
    spots = {name: pd.DataFrame({"trade_date": calendar, "close": 100.0, "pre_close": 100.0}) for name in PRODUCTS.values()}
    return pd.DataFrame(records), spots, calendar


def test_basis_and_open_interest_use_actual_contracts_without_roll_jump():
    raw, spots, _ = inputs()
    synthetic = raw.iloc[[0]].copy()
    synthetic["ts_code"] = "IF.CFX"
    synthetic["vol"] = 1000000
    daily = aggregate_contracts(pd.concat([raw, synthetic]), spots)
    assert daily.loc[24, "IF_basis"] == pytest.approx(0.03)
    assert daily.loc[25, "IF_basis"] == pytest.approx(0.04)
    assert daily.IF_basis_change.eq(0).all()
    assert daily.IF_excess_return.eq(0).all()
    assert daily.IF_curve_per_month.eq(0.01).all()
    assert daily.loc[0, "IF_open_interest"] == 4000
    assert daily.loc[0, "IF_volume"] == 1300
    assert daily.loc[0, "IF_weighted_basis"] == pytest.approx(0.055)


@pytest.mark.parametrize("lag_sessions", [0, 1])
def test_lagged_context_uses_declared_session_and_has_exact_prefix_parity(lag_sessions):
    raw, spots, calendar = inputs()
    daily = aggregate_contracts(raw, spots)
    timing = {"lag_sessions": lag_sessions, "publication_hour": 18}
    full = futures_features(calendar[20:], daily, calendar, **timing)
    assert full.futures_source_date.tolist() == list(calendar[20 - lag_sessions:len(calendar) - lag_sessions])
    prefix = futures_features(calendar[20:31], daily.iloc[:31 - lag_sessions], calendar[:31], **timing)
    pd.testing.assert_frame_equal(full.iloc[:11], prefix, check_exact=True)
    daily.loc[31 - lag_sessions:, "IF_basis"] *= -100
    changed = futures_features(calendar[20:], daily, calendar, **timing)
    pd.testing.assert_frame_equal(full.iloc[:11], changed.iloc[:11], check_exact=True)
    with pytest.raises(ValueError, match="Missing prior-session"):
        futures_features(calendar[20:], daily.drop(index=25), calendar, **timing)


def test_same_day_futures_requires_after_close_publication():
    raw, spots, calendar = inputs()
    daily = aggregate_contracts(raw, spots)
    with pytest.raises(ValueError, match="evening publication"):
        futures_features(calendar[20:], daily, calendar, lag_sessions=0, publication_hour=15)


def test_unneeded_previous_close_can_be_missing_but_main_return_cannot_be_imputed():
    raw, spots, _ = inputs()
    expected = aggregate_contracts(raw, spots)
    raw["pre_close"] = raw.pre_close.astype(float)
    raw.loc[raw.ts_code.eq("IF2409.CFX"), "pre_close"] = np.nan
    actual = aggregate_contracts(raw, spots)
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    raw.loc[0, "pre_close"] = np.nan
    with pytest.raises(ValueError, match="Main contract"):
        aggregate_contracts(raw, spots)


@pytest.mark.parametrize("problem", ["duplicate", "missing_contract", "nonfinite", "missing_spot", "invalid_month", "negative_volume"])
def test_invalid_contract_data_cannot_silently_enter_training(problem):
    raw, spots, _ = inputs()
    if problem == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]])
    elif problem == "missing_contract":
        raw = raw.drop(index=0)
    elif problem == "nonfinite":
        raw["close"] = raw.close.astype(float)
        raw.loc[0, "close"] = np.nan
    elif problem == "missing_spot":
        spots["csi300"] = spots["csi300"].iloc[1:]
    elif problem == "invalid_month":
        raw.loc[0, "ts_code"] = "IF2413.CFX"
    else:
        raw.loc[0, "vol"] = -1
    with pytest.raises(ValueError):
        aggregate_contracts(raw, spots)


def test_monthly_pagination_and_provider_error_redaction():
    pages = [pd.DataFrame({"trade_date": "20240102", "ts_code": [f"test{i}" for i in range(2000)]}),
             pd.DataFrame({"trade_date": ["20240103"], "ts_code": ["last"]})]
    offsets = []
    def query(api, **kwargs):
        offsets.append(kwargs["offset"])
        return pages[len(offsets) - 1]
    frame = fetch_month(SimpleNamespace(query=query), "20240101", "20240131")
    assert len(frame) == 2001
    assert offsets == [0, 2000]
    def failed(*args, **kwargs):
        raise RuntimeError("provider message with credentials")
    with pytest.raises(RuntimeError, match="credentials omitted") as error:
        fetch_month(SimpleNamespace(query=failed), "20240101", "20240131")
    assert "provider message" not in str(error.value)


def test_pagination_rejects_repeated_provider_pages():
    page = pd.DataFrame({"trade_date": "20240102", "ts_code": [f"test{i}" for i in range(2000)]})
    with pytest.raises(ValueError, match="pagination exceeded"):
        fetch_month(SimpleNamespace(query=lambda *args, **kwargs: page), "20240101", "20240131")
