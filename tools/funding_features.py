"""Domestic financing and valuation features with a one-session publication lag."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ASSETS = ("index_basic", "margin")


def load_assets(directory: Path):
    return {name: pd.read_csv(directory / f"{name}.csv", float_precision="round_trip") for name in ASSETS}


def funding_features(signal_dates, assets):
    dates = pd.DatetimeIndex(pd.to_datetime(signal_dates))
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Funding features require unique chronological signal dates.")
    frames = {}
    for name in ASSETS:
        frame = assets[name].copy()
        frame["trade_date"] = pd.to_datetime(frame.trade_date.astype(str), format="%Y%m%d")
        if frame.trade_date.duplicated().any():
            raise ValueError(f"Duplicate funding dates for {name}.")
        frames[name] = frame.set_index("trade_date").sort_index()
    # Inserting every requested trading date prevents a missing observation
    # from silently turning a one-session lag into a longer lag.
    calendar = frames["index_basic"].index.union(dates).sort_values()
    basic = frames["index_basic"].reindex(calendar)
    margin = frames["margin"].reindex(calendar)
    features = pd.DataFrame(index=calendar)
    for column in ("turnover_rate", "turnover_rate_f"):
        values = pd.to_numeric(basic[column], errors="coerce")
        features[f"funding_{column}"] = values
        features[f"funding_{column}_change"] = values.pct_change(fill_method=None)
        features[f"funding_{column}_relative_20"] = values / values.rolling(20, min_periods=20).mean() - 1
    for column in ("pe_ttm", "pb"):
        values = pd.to_numeric(basic[column], errors="coerce")
        features[f"funding_{column}"] = values
        features[f"funding_{column}_z252"] = (values - values.rolling(252, min_periods=60).mean()) / values.rolling(252, min_periods=60).std().clip(lower=1e-12)
    balance = pd.to_numeric(margin.rzye, errors="coerce")
    buying = pd.to_numeric(margin.rzmre, errors="coerce")
    repayment = pd.to_numeric(margin.rzche, errors="coerce")
    features["funding_finance_change_1"] = balance.pct_change(fill_method=None)
    features["funding_finance_change_5"] = balance.pct_change(5, fill_method=None)
    features["funding_finance_net_ratio"] = (buying - repayment) / balance.shift(1)
    features["funding_finance_buy_ratio"] = buying / (buying + repayment)
    features["funding_finance_market_ratio"] = balance / pd.to_numeric(basic.float_mv, errors="coerce")
    features["funding_lending_share"] = pd.to_numeric(margin.rqye, errors="coerce") / pd.to_numeric(margin.rzrqye, errors="coerce")
    lagged = features.shift(1).reindex(dates).replace([np.inf, -np.inf], np.nan)
    if lagged.isna().any().any():
        bad_dates = lagged.index[lagged.isna().any(axis=1)].strftime("%Y%m%d").tolist()
        raise ValueError(f"Missing lagged funding inputs for {len(bad_dates)} signal dates: {bad_dates[:5]}.")
    source_date = pd.Series(calendar, index=calendar).shift(1).reindex(dates)
    if not (source_date.to_numpy() < dates.to_numpy()).all():
        raise ValueError("Funding input must precede the prediction signal day.")
    lagged["funding_source_date"] = source_date.dt.strftime("%Y%m%d").astype(int)
    return lagged.reset_index(drop=True)
