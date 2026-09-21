"""Same-day Shanghai capitalization prices with strictly lagged money flow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.breadth_features import aggregate_day
from tools.capital_features import FIELDS, FEATURE_COLUMNS, capital_features
from tools.liquidity_features import date_values


PRICE_MEASURES = ("cap_up_fraction", "cap_return", "cap_vs_equal_up")
PRICE_COLUMNS = tuple(f"capital_{name}" for name in (*PRICE_MEASURES, "cap_return_mean_5", "cap_up_change_1"))


def capital_price_day(basic, daily, date):
    aggregate_day(daily, date)
    if basic.empty or not set(FIELDS).issubset(basic.columns):
        raise ValueError("Capital prices require complete daily_basic fields.")
    basic = basic.loc[:, FIELDS].copy()
    if (basic.ts_code.duplicated().any() or not basic.trade_date.astype(str).eq(str(date)).all()
            or not basic.ts_code.str.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)").all()):
        raise ValueError("Invalid capitalization codes, dates or duplicate rows.")
    basic["total_mv"] = pd.to_numeric(basic.total_mv, errors="raise")
    if basic.total_mv.lt(0).any() or np.isinf(basic.total_mv).any():
        raise ValueError("Capitalization must not be negative or infinite.")
    prices = daily.copy()
    for name in ("vol", "amount", "pct_chg"):
        prices[name] = pd.to_numeric(prices[name], errors="raise")
    active = prices.loc[prices.ts_code.str.fullmatch(r"6[0-9]{5}\.SH") & prices.vol.gt(0) & prices.amount.gt(0)].set_index("ts_code")
    if len(active) < 2:
        raise ValueError("Capital prices require at least two active Shanghai A-shares.")
    caps = basic.set_index("ts_code").total_mv.reindex(active.index)
    result = {"trade_date": int(date), "active_rows": len(active), "basic_stock_coverage": float(caps.gt(0).mean()),
              **{name: np.nan for name in PRICE_MEASURES}}
    if result["basic_stock_coverage"] == 1:
        up = active.pct_chg.gt(0)
        result.update(cap_up_fraction=float(np.average(up, weights=caps)),
                      cap_return=float(np.average(active.pct_chg / 100, weights=caps)),
                      cap_vs_equal_up=float(np.average(up, weights=caps) - up.mean()))
    return result


def capital_close_features(signal_dates, lagged_daily, price_daily, calendar, *, publication_hour=18):
    if not 18 <= publication_hour <= 23:
        raise ValueError("Same-day capital prices require publication at 18:00 or later.")
    result = capital_features(signal_dates, lagged_daily, calendar)
    signals, calendar = date_values(signal_dates), date_values(calendar)
    source = price_daily.copy()
    source["trade_date"] = date_values(source.trade_date).to_numpy()
    if source.trade_date.duplicated().any() or not source.trade_date.is_monotonic_increasing:
        raise ValueError("Capital price dates must be unique and chronological.")
    positions = pd.Index(calendar).get_indexer(signals)
    required = calendar.iloc[positions[0] - 20:positions[-1] + 1]
    if not pd.Index(required).isin(source.trade_date).all():
        raise ValueError("Missing same-day capital prices; no stale filling.")
    source = source.set_index("trade_date").reindex(calendar)
    if not source.basic_stock_coverage.dropna().between(0, 1).all():
        raise ValueError("Capital price coverage must be a fraction.")
    values = source.loc[:, PRICE_MEASURES].astype(float).where(source.basic_stock_coverage.eq(1), np.nan)
    if (np.isinf(values.to_numpy()).any()
            or not values.cap_up_fraction.dropna().between(0, 1).all()
            or not values.cap_vs_equal_up.dropna().between(-1, 1).all()):
        raise ValueError("Invalid capital price aggregates.")
    values["cap_return_mean_5"] = values.cap_return.rolling(5).mean()
    values["cap_up_change_1"] = values.cap_up_fraction.diff()
    selected = values.reindex(signals).reset_index(drop=True).add_prefix("capital_")
    result.loc[:, PRICE_COLUMNS] = selected.loc[:, PRICE_COLUMNS].to_numpy()
    result["capital_available"] &= np.isfinite(result.loc[:, FEATURE_COLUMNS].to_numpy()).all(axis=1)
    result.insert(2, "capital_price_source_date", signals.to_numpy())
    return result
