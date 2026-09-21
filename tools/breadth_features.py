"""Survivorship-free daily market breadth with explicit publication lag."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("ts_code", "trade_date", "pct_chg", "amount", "vol")
GROUPS = ("all", "shanghai")
MEASURES = ("up_fraction", "down_fraction", "median_return", "mean_return", "down_amount_fraction", "large_down_fraction")


def aggregate_day(frame, trade_date):
    if frame.empty or not set(REQUIRED_COLUMNS).issubset(frame.columns):
        raise ValueError("Daily breadth requires complete stock observations.")
    values = frame.loc[:, REQUIRED_COLUMNS].copy()
    if values.ts_code.duplicated().any() or not values.trade_date.astype(str).eq(str(trade_date)).all():
        raise ValueError("Daily breadth has duplicate stocks or unexpected dates.")
    if not values.ts_code.str.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)").all():
        raise ValueError("Daily breadth requires supported A-share codes.")
    source_rows = len(values)
    # BJ-coded series include pre-exchange OTC history. Use a stable SH/SZ universe.
    values = values.loc[values.ts_code.str.endswith((".SH", ".SZ"))].copy()
    for column in ("pct_chg", "amount", "vol"):
        values[column] = pd.to_numeric(values[column], errors="raise")
    if not np.isfinite(values[["pct_chg", "amount", "vol"]].to_numpy()).all():
        raise ValueError("Daily breadth requires finite observations.")
    if values.amount.lt(0).any() or values.vol.lt(0).any():
        raise ValueError("Daily turnover and volume must be nonnegative.")
    active = values.loc[values.vol.gt(0) & values.amount.gt(0)]
    result = {"trade_date": int(trade_date), "source_rows": source_rows,
              "excluded_exchange_rows": source_rows - len(values)}
    for name in GROUPS:
        group = active if name == "all" else active.loc[active.ts_code.str.endswith(".SH")]
        if group.empty:
            raise ValueError(f"No actively traded stocks in breadth group {name}.")
        returns = group.pct_chg / 100
        result.update({
            f"{name}_active_rows": len(group),
            f"{name}_up_fraction": float(returns.gt(0).mean()),
            f"{name}_down_fraction": float(returns.lt(0).mean()),
            f"{name}_median_return": float(returns.median()),
            f"{name}_mean_return": float(returns.mean()),
            f"{name}_down_amount_fraction": float(group.loc[returns.lt(0), "amount"].sum() / group.amount.sum()),
            f"{name}_large_down_fraction": float(returns.le(-0.02).mean()),
        })
    return result


def breadth_features(signal_dates, breadth, calendar, *, lag_sessions=1, publication_hour=18):
    if lag_sessions not in (0, 1):
        raise ValueError("Breadth lag must be zero or one domestic session.")
    if lag_sessions == 0 and not 17 <= publication_hour <= 23:
        raise ValueError("Same-day breadth requires publication after daily data ingestion.")
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(signal_dates).astype(str), format="%Y%m%d"))
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.Series(calendar).astype(str), format="%Y%m%d"))
    for label, index in (("signal", dates), ("calendar", calendar)):
        if index.empty or index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError(f"Breadth {label} dates must be unique and chronological.")
    if len(dates.difference(calendar)):
        raise ValueError("Signal dates must belong to the frozen domestic calendar.")
    source = breadth.copy()
    source["trade_date"] = pd.to_datetime(source.trade_date.astype(str), format="%Y%m%d")
    if source.trade_date.duplicated().any():
        raise ValueError("Duplicate market breadth dates.")
    source = source.set_index("trade_date").reindex(calendar)
    features = pd.DataFrame(index=calendar)
    for group in GROUPS:
        for measure in MEASURES:
            values = pd.to_numeric(source[f"{group}_{measure}"], errors="raise")
            if "fraction" in measure and not values.dropna().between(0, 1).all():
                raise ValueError("Breadth fractions must be in [0, 1].")
            features[f"breadth_{group}_{measure}"] = values
        advancing = source[f"{group}_up_fraction"]
        features[f"breadth_{group}_up_change_1"] = advancing.diff()
        features[f"breadth_{group}_up_mean_5"] = advancing.rolling(5, min_periods=5).mean()
        features[f"breadth_{group}_up_mean_20"] = advancing.rolling(20, min_periods=20).mean()
    features["breadth_shanghai_relative"] = source.shanghai_up_fraction - source.all_up_fraction
    lagged = features.shift(lag_sessions).reindex(dates)
    if not np.isfinite(lagged.to_numpy()).all():
        raise ValueError("Missing lagged breadth history; missing sessions cannot be filled.")
    source_dates = pd.Series(calendar, index=calendar).shift(lag_sessions).reindex(dates)
    available = source_dates.to_numpy() < dates.to_numpy() if lag_sessions else source_dates.to_numpy() == dates.to_numpy()
    if not available.all():
        raise ValueError("Breadth source dates disagree with the declared publication policy.")
    lagged["breadth_source_date"] = source_dates.dt.strftime("%Y%m%d").astype(int)
    return lagged.reset_index(drop=True)
