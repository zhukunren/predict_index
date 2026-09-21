"""Overseas risk features available before an after-close Shanghai forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd


ASSETS = {"spx": "SPX", "nasdaq": "IXIC"}
MAX_AGE_DAYS = 7


def global_risk_features(signal_dates, assets):
    dates = pd.to_datetime(pd.Series(signal_dates).reset_index(drop=True).astype(str), format="%Y%m%d")
    if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Signal dates must be unique and chronological.")
    result = pd.DataFrame({"trade_date": dates.dt.strftime("%Y%m%d").astype(int)})
    calendars = []
    for name in ASSETS:
        source = assets[name].copy()
        source["session_date"] = pd.to_datetime(source.trade_date.astype(str), format="%Y%m%d")
        source = source.sort_values("session_date").reset_index(drop=True)
        if source.empty or source.session_date.duplicated().any():
            raise ValueError(f"Missing or duplicate overseas sessions: {name}.")
        close = pd.to_numeric(source.close, errors="raise")
        if not (np.isfinite(close) & close.gt(0)).all():
            raise ValueError("Overseas closes must be finite and positive.")
        calendars.append(pd.DatetimeIndex(source.session_date))
        daily = close.pct_change(fill_method=None)
        risk = pd.DataFrame({
            "source_date": source.session_date,
            f"{name}_return_1": daily,
            f"{name}_return_5": close.pct_change(5, fill_method=None),
            f"{name}_volatility_20": daily.rolling(20).std(),
            f"{name}_drawdown_20": close / close.rolling(20).max() - 1,
        })
        # A US session labelled with today's date closes tomorrow in Shanghai.
        available = pd.merge_asof(
            pd.DataFrame({"signal_date": dates}), risk,
            left_on="signal_date", right_on="source_date", allow_exact_matches=False,
        )
        age = (available.signal_date - available.source_date).dt.days
        if available.source_date.isna().any() or not age.between(1, MAX_AGE_DAYS).all():
            raise ValueError(f"Overseas input is unavailable or stale: {name}.")
        result[f"{name}_source_date"] = available.source_date.dt.strftime("%Y%m%d").astype(int)
        columns = [column for column in risk if column != "source_date"]
        if not np.isfinite(available[columns].to_numpy()).all():
            raise ValueError(f"Overseas indicators require at least 21 prior sessions: {name}.")
        result[columns] = available[columns]
    relevant = [calendar[calendar < dates.max()] for calendar in calendars]
    if not relevant[0].equals(relevant[1]):
        raise ValueError("Overseas assets have different session coverage.")
    result["nasdaq_relative_return_1"] = result.nasdaq_return_1 - result.spx_return_1
    return result
