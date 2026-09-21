"""Completed GMT-date offshore RMB quotes available before Shanghai forecasts."""

import numpy as np
import pandas as pd

from tools.liquidity_features import date_values


TS_CODE = "USDCNH.FXCM"
FIELDS = ("ts_code", "trade_date", "bid_open", "bid_high", "bid_low", "bid_close", "ask_close", "tick_qty")
FEATURE_COLUMNS = ("fx_cnh_return_1", "fx_cnh_return_5", "fx_cnh_return_20",
                   "fx_cnh_volatility_20", "fx_cnh_bid_range", "fx_cnh_close_spread")
MAX_AGE_DAYS = 7
POLICY = {
    "source": "Tushare fx_daily", "instrument": TS_CODE,
    "date_convention": "provider labels quote bars by GMT date; use source_date strictly earlier than Shanghai signal date",
    "features": "mid-close returns over 1/5/20 observed quote bars, 20-bar volatility, bid high-low range, relative closing ask-bid spread",
    "interpretation": "positive USDCNH return denotes RMB depreciation; quote count is not trading volume",
    "maximum_age_calendar_days": MAX_AGE_DAYS,
    "unused_open_field": "provider bid_open may lie outside the quoted range; retain and audit this anomaly, never use open in model features",
    "incomplete_data": "no synthetic quotes or filled feature values; decline stale, truncated or invalid histories",
    "documentation": "https://tushare.pro/document/2?doc_id=179",
    "historical_delivery_timestamps_available": False,
}


def normalize_quotes(frame):
    if frame.empty or not set(FIELDS).issubset(frame.columns):
        raise ValueError("FX history requires the declared quote fields.")
    result = frame.loc[:, FIELDS].copy()
    result["trade_date"] = date_values(result.trade_date).to_numpy()
    if result.trade_date.duplicated().any() or not result.ts_code.eq(TS_CODE).all():
        raise ValueError("Unexpected FX code or duplicate quote dates.")
    result = result.sort_values("trade_date").reset_index(drop=True)
    prices = [n for n in FIELDS if n not in ("ts_code", "trade_date", "tick_qty")]
    for name in (*prices, "tick_qty"):
        result[name] = pd.to_numeric(result[name], errors="raise")
    if not np.isfinite(result.loc[:, [*prices, "tick_qty"]].to_numpy(dtype=float)).all():
        raise ValueError("FX quotes must be finite.")
    if not result.loc[:, prices].gt(0).all().all() or result.tick_qty.le(0).any():
        raise ValueError("FX quotes and quote activity must be positive.")
    if (result.bid_high.lt(result.bid_low).any()
            or result.bid_close.lt(result.bid_low).any() or result.bid_close.gt(result.bid_high).any()
            or result.ask_close.lt(result.bid_close).any()):
        raise ValueError("Invalid FX bid range or crossed closing quotes.")
    result["unused_open_range_anomaly"] = result.bid_open.lt(result.bid_low) | result.bid_open.gt(result.bid_high)
    return result


def fx_features(signal_dates, quotes):
    signals = date_values(signal_dates)
    if signals.empty or signals.duplicated().any() or not signals.is_monotonic_increasing:
        raise ValueError("FX signal dates must be unique and chronological.")
    # Later quotes are not relevant to either validation or feature calculation.
    source = quotes.loc[date_values(quotes.trade_date).to_numpy() < int(signals.iloc[-1])]
    source = normalize_quotes(source)
    source_dates = pd.to_datetime(source.trade_date.astype(str), format="%Y%m%d")
    if source_dates.diff().dt.days.gt(MAX_AGE_DAYS).any():
        raise ValueError("FX history contains a long missing interval.")
    mid = (source.bid_close + source.ask_close) / 2
    daily_return = mid.pct_change(fill_method=None)
    daily = pd.DataFrame({
        "source_time": source_dates, "fx_source_date": source.trade_date,
        "fx_cnh_return_1": daily_return,
        "fx_cnh_return_5": mid.pct_change(5, fill_method=None),
        "fx_cnh_return_20": mid.pct_change(20, fill_method=None),
        "fx_cnh_volatility_20": daily_return.rolling(20).std(),
        "fx_cnh_bid_range": (source.bid_high - source.bid_low) / source.bid_close.shift(1),
        "fx_cnh_close_spread": (source.ask_close - source.bid_close) / mid,
    })
    signal_times = pd.to_datetime(signals.astype(str), format="%Y%m%d")
    aligned = pd.merge_asof(pd.DataFrame({"signal_time": signal_times}), daily,
                            left_on="signal_time", right_on="source_time", allow_exact_matches=False)
    age = (aligned.signal_time - aligned.source_time).dt.days
    if not age.between(1, MAX_AGE_DAYS).all():
        raise ValueError("FX observations are unavailable or stale for a signal date.")
    if not np.isfinite(aligned.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("FX features need at least 21 earlier observed quote bars.")
    result = aligned.loc[:, ["fx_source_date", *FEATURE_COLUMNS]].copy()
    result["fx_source_date"] = result.fx_source_date.astype(int)
    result.insert(0, "trade_date", signals.to_numpy())
    result.insert(2, "fx_source_age_days", age.to_numpy(dtype=int))
    return result
