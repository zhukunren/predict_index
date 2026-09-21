"""Strictly lagged bank and exchange funding stress features."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUESTS = {
    "shibor": ("shibor", {}, ("date", "on", "1w", "3m")),
    "gc001": ("repo_daily", {"ts_code": "204001.SH"},
              ("trade_date", "ts_code", "close", "high", "low", "amount")),
    "gc007": ("repo_daily", {"ts_code": "204007.SH"},
              ("trade_date", "ts_code", "close", "high", "low", "amount")),
}
WARMUP = 20
FEATURE_COLUMNS = (
    "liquidity_overnight", "liquidity_overnight_change_1", "liquidity_overnight_change_5",
    "liquidity_overnight_vs_mean_20", "liquidity_bank_week_spread", "liquidity_bank_quarter_spread",
    "liquidity_exchange_bank_overnight_spread", "liquidity_exchange_bank_week_spread",
    "liquidity_exchange_week_spread", "liquidity_exchange_overnight_range",
    "liquidity_exchange_overnight_change_1", "liquidity_exchange_volume_ratio_20",
    "liquidity_exchange_short_volume_share",
)


def date_values(values):
    strings = pd.Series(values).astype(str).reset_index(drop=True)
    if not strings.str.fullmatch(r"\d{8}").all():
        raise ValueError("Liquidity dates must use YYYYMMDD.")
    pd.to_datetime(strings, format="%Y%m%d", errors="raise")
    return strings.astype(int)


def normalize_asset(name, frame):
    _, parameters, columns = REQUESTS[name]
    if frame.empty or not set(columns).issubset(frame.columns):
        raise ValueError(f"Missing liquidity observations or fields for {name}.")
    result = frame.loc[:, columns].copy().rename(columns={"date": "trade_date"})
    result["trade_date"] = date_values(result.trade_date).to_numpy()
    if result.trade_date.duplicated().any():
        raise ValueError(f"Duplicate liquidity dates for {name}.")
    if parameters and not result.ts_code.eq(parameters["ts_code"]).all():
        raise ValueError(f"Unexpected liquidity instrument for {name}.")
    numeric = [column for column in result if column not in ("trade_date", "ts_code")]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite liquidity observations for {name}.")
    rates = [column for column in numeric if column != "amount"]
    if not result[rates].ge(0).all().all() or not result[rates].le(100).all().all():
        raise ValueError(f"Invalid percentage rate for {name}.")
    if name != "shibor":
        if (result.amount.le(0).any() or result.high.lt(result.low).any()
                or result.close.lt(result.low).any() or result.close.gt(result.high).any()):
            raise ValueError(f"Invalid repo range or volume for {name}.")
    return result.sort_values("trade_date").reset_index(drop=True)


def liquidity_features(signal_dates, assets, calendar, *, allow_missing=False):
    signals, calendar = date_values(signal_dates), date_values(calendar)
    for dates in (signals, calendar):
        if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Liquidity signals and calendar must be unique and chronological.")
    if set(assets) != set(REQUESTS):
        raise ValueError("All three frozen liquidity assets are required.")
    positions = pd.Index(calendar).get_indexer(signals)
    if (positions < WARMUP).any():
        raise ValueError("Liquidity signals require 20 prior domestic sessions.")
    first, last = int(positions[0] - WARMUP), int(positions[-1])
    required = pd.Index(calendar.iloc[first:last])
    aligned = {}
    for name, raw in assets.items():
        values = normalize_asset(name, raw).set_index("trade_date")
        if not allow_missing and not required.isin(values.index).all():
            raise ValueError(f"Missing domestic-session liquidity data for {name}; no stale filling.")
        aligned[name] = values.reindex(required)
    bank, overnight, week = (aligned[name] for name in REQUESTS)
    bank_rate = bank["on"] / 100
    overnight_rate, week_rate = overnight.close / 100, week.close / 100
    values = pd.DataFrame({
        "liquidity_overnight": bank_rate,
        "liquidity_overnight_change_1": bank_rate.diff(),
        "liquidity_overnight_change_5": bank_rate.diff(5),
        "liquidity_overnight_vs_mean_20": bank_rate - bank_rate.rolling(20).mean(),
        "liquidity_bank_week_spread": (bank["1w"] - bank["on"]) / 100,
        "liquidity_bank_quarter_spread": (bank["3m"] - bank["on"]) / 100,
        "liquidity_exchange_bank_overnight_spread": overnight_rate - bank_rate,
        "liquidity_exchange_bank_week_spread": week_rate - bank["1w"] / 100,
        "liquidity_exchange_week_spread": week_rate - overnight_rate,
        "liquidity_exchange_overnight_range": (overnight.high - overnight.low) / 100,
        "liquidity_exchange_overnight_change_1": overnight_rate.diff(),
        "liquidity_exchange_volume_ratio_20": overnight.amount / overnight.amount.rolling(20).mean(),
        "liquidity_exchange_short_volume_share": overnight.amount / (overnight.amount + week.amount),
    })
    source_dates = calendar.iloc[positions - 1].to_numpy()
    result = values.reindex(source_dates).reset_index(drop=True)
    available = np.isfinite(result.to_numpy(dtype=float)).all(axis=1)
    if not allow_missing and not available.all():
        raise ValueError("Liquidity feature history is incomplete.")
    if allow_missing:
        result.insert(0, "liquidity_available", available)
    result.insert(0, "liquidity_source_date", source_dates)
    result.insert(0, "trade_date", signals.to_numpy())
    return result
