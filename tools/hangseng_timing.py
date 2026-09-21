"""Causal Hong Kong close alignment for explicitly timed research forecasts."""

from __future__ import annotations

import pandas as pd

from 数据拉取脚本_tushare import _asset_feature_frame


CANDIDATES = ("previous_available", "current_prices", "augment_current_prices")


def timed_features(data, raw_hangseng, candidate, *, publication_hour):
    if candidate not in CANDIDATES:
        raise ValueError("Unknown Hong Kong timing candidate.")
    if not 0 <= publication_hour <= 23:
        raise ValueError("Publication hour must be in [0, 23].")
    if candidate != "previous_available" and publication_hour < 17:
        raise ValueError("Same-day Hong Kong prices require publication after its closing auction.")
    frame = data.copy().reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame.trade_date)
    if frame.empty or frame.trade_date.duplicated().any() or not frame.trade_date.is_monotonic_increasing:
        raise ValueError("Signal dates must be unique and chronological.")
    source = _asset_feature_frame("hangseng", raw_hangseng, lag=0)
    if source.empty:
        raise ValueError("Hong Kong prices are unavailable.")
    source = source.rename(columns={"trade_date": "hk_date"})
    previous = pd.merge_asof(frame[["trade_date"]], source, left_on="trade_date", right_on="hk_date", allow_exact_matches=False)
    prices = previous if candidate == "previous_available" else pd.merge_asof(
        frame[["trade_date"]], source.drop(columns="hangseng_vol_chg", errors="ignore"),
        left_on="trade_date", right_on="hk_date", allow_exact_matches=True,
    )
    original_columns = [column for column in frame if column.startswith("hangseng_")]
    if candidate != "augment_current_prices":
        frame = frame.drop(columns=original_columns)
    for column in prices:
        if not column.startswith("hangseng_"):
            continue
        name = column + "_lag1" if candidate == "previous_available" else column
        frame[name] = prices[column]
    if candidate == "current_prices" and "hangseng_vol_chg" in previous:
        frame["hangseng_vol_chg_lag1"] = previous.hangseng_vol_chg
    availability = pd.DataFrame({"trade_date": frame.trade_date, "price_source_date": prices.hk_date,
                                 "volume_source_date": previous.hk_date})
    known_price = availability.price_source_date.notna()
    if candidate == "previous_available":
        assert availability.loc[known_price, "price_source_date"].lt(availability.loc[known_price, "trade_date"]).all()
    else:
        assert availability.loc[known_price, "price_source_date"].le(availability.loc[known_price, "trade_date"]).all()
    return frame, availability
