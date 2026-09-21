"""ETF share-count changes, with two domestic sessions of publication delay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.liquidity_features import date_values


ASSETS = {"sse50": "510050.SH", "csi300": "510300.SH", "csi500": "510500.SH", "chinext": "159915.SZ"}
FIELDS = ("ts_code", "trade_date", "fd_share")
LAG_SESSIONS = 2
WARMUP = 22
FEATURE_COLUMNS = tuple(f"etf_{name}_{measure}" for name in ASSETS
                        for measure in ("share_change_1", "share_change_5", "share_change_mean_20"))


def normalize_shares(name, frame):
    if frame.empty or not set(FIELDS).issubset(frame.columns):
        raise ValueError(f"Missing ETF share observations for {name}.")
    result = frame.loc[:, FIELDS].copy()
    result["trade_date"] = date_values(result.trade_date).to_numpy()
    if result.trade_date.duplicated().any() or not result.ts_code.eq(ASSETS[name]).all():
        raise ValueError(f"Duplicate ETF dates or unexpected instrument for {name}.")
    result["fd_share"] = pd.to_numeric(result.fd_share, errors="raise")
    if not (np.isfinite(result.fd_share) & result.fd_share.gt(0)).all():
        raise ValueError("ETF shares must be positive finite observations.")
    return result.sort_values("trade_date").reset_index(drop=True)


def etf_features(signal_dates, assets, calendar):
    signals, calendar = date_values(signal_dates), date_values(calendar)
    for dates in (signals, calendar):
        if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("ETF signals and calendar must be unique and chronological.")
    if set(assets) != set(ASSETS):
        raise ValueError("All four frozen ETF share series are required.")
    positions = pd.Index(calendar).get_indexer(signals)
    if (positions < WARMUP).any():
        raise ValueError("ETF signals require 22 prior domestic sessions.")
    source_dates = calendar.iloc[positions - LAG_SESSIONS].to_numpy()
    values = pd.DataFrame(index=pd.Index(calendar))
    for name in ASSETS:
        shares = normalize_shares(name, assets[name]).set_index("trade_date").fd_share.reindex(calendar)
        daily_change = shares.pct_change(fill_method=None)
        values[f"etf_{name}_share_change_1"] = daily_change
        values[f"etf_{name}_share_change_5"] = shares.pct_change(5, fill_method=None).where(shares.rolling(6).count().eq(6))
        values[f"etf_{name}_share_change_mean_20"] = daily_change.rolling(20).mean()
    result = values.reindex(source_dates).reset_index(drop=True)
    result.insert(0, "etf_available", np.isfinite(result.to_numpy(dtype=float)).all(axis=1))
    result.insert(0, "etf_source_date", source_dates)
    result.insert(0, "trade_date", signals.to_numpy())
    return result
