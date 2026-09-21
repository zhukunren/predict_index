"""Lagged stock-index futures context from actual dated contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRODUCTS = {"IF": "csi300", "IC": "csi500"}
FIELDS = ("ts_code", "trade_date", "pre_close", "open", "close", "vol", "oi")
MEASURES = ("basis", "basis_change", "weighted_basis", "curve_per_month", "turnover", "excess_return")
FEATURE_COLUMNS = tuple(f"futures_{product}_{name}" for product in PRODUCTS
                        for name in (*MEASURES, "volume_ratio_20", "interest_change_1"))


def aggregate_contracts(raw, spots):
    if not set(FIELDS).issubset(raw.columns):
        raise ValueError("Futures observations require all declared fields.")
    if raw.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate futures contract observations.")
    dates = pd.to_datetime(raw.trade_date.astype(str), format="%Y%m%d")
    contracts = raw.ts_code.str.extract(r"^(IF|IC)([0-9]{2})([0-9]{2})\.CFX$")
    keep = contracts[0].notna()
    frame = raw.loc[keep].copy()
    frame["date"] = dates.loc[keep]
    frame["product"] = contracts.loc[keep, 0]
    frame["contract_month"] = (2000 + contracts.loc[keep, 1].astype(int)) * 12 + contracts.loc[keep, 2].astype(int)
    if not contracts.loc[keep, 2].astype(int).between(1, 12).all():
        raise ValueError("Invalid futures delivery month.")
    if frame.contract_month.lt(frame.date.dt.year * 12 + frame.date.dt.month).any():
        raise ValueError("Futures contract month precedes its trading date.")
    numeric = list(FIELDS[2:])
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[["open", "close", "vol", "oi"]].to_numpy()).all():
        raise ValueError("Futures observations must be finite.")
    previous = frame.pre_close.dropna()
    if not (np.isfinite(previous) & previous.gt(0)).all():
        raise ValueError("Observed futures previous closes must be positive and finite.")
    if frame[["open", "close"]].le(0).any().any() or frame[["vol", "oi"]].lt(0).any().any():
        raise ValueError("Invalid futures prices, volume, or open interest.")
    outputs = []
    for product, spot_name in PRODUCTS.items():
        subset = frame.loc[frame["product"].eq(product)]
        if subset.empty:
            raise ValueError(f"No dated contracts for {product}.")
        spot = spots[spot_name].copy()
        spot["date"] = pd.to_datetime(spot.trade_date.astype(str), format="mixed")
        if spot.date.duplicated().any():
            raise ValueError("Duplicate spot dates.")
        spot = spot.set_index("date")
        rows = []
        for date, group in subset.groupby("date", sort=True):
            if len(group) != 4 or group.contract_month.duplicated().any():
                raise ValueError(f"Expected four distinct dated {product} contracts on {date.date()}.")
            if date not in spot.index:
                raise ValueError("Missing spot price for futures basis.")
            close, previous = float(spot.loc[date, "close"]), float(spot.loc[date, "pre_close"])
            if not np.isfinite([close, previous]).all() or min(close, previous) <= 0:
                raise ValueError("Spot prices must be finite and positive.")
            if group.oi.sum() <= 0 or group.vol.sum() <= 0:
                raise ValueError("Futures product must have positive total activity.")
            main = group.sort_values(["vol", "ts_code"], ascending=[False, True]).iloc[0]
            if not np.isfinite(main.pre_close) or main.pre_close <= 0:
                raise ValueError("Main contract requires its own observed previous close.")
            curve = group.sort_values("contract_month").iloc[:2]
            month_gap = int(curve.contract_month.iloc[1] - curve.contract_month.iloc[0])
            rows.append({
                "trade_date": int(date.strftime("%Y%m%d")),
                f"{product}_main_contract": main.ts_code,
                f"{product}_basis": float(main.close / close - 1),
                # pre_close belongs to the same dated contract, even on a roll day.
                f"{product}_basis_change": float(main.close / close - main.pre_close / previous),
                f"{product}_weighted_basis": float(np.average(group.close / close - 1, weights=group.oi)),
                f"{product}_curve_per_month": float((curve.close.iloc[1] - curve.close.iloc[0]) / close / month_gap),
                f"{product}_open_interest": float(group.oi.sum()), f"{product}_volume": float(group.vol.sum()),
                f"{product}_turnover": float(group.vol.sum() / group.oi.sum()),
                f"{product}_excess_return": float(main.close / main.pre_close - close / previous),
            })
        outputs.append(pd.DataFrame(rows).set_index("trade_date"))
    if not outputs[0].index.equals(outputs[1].index):
        raise ValueError("Futures products have different trading calendars.")
    return pd.concat(outputs, axis=1).reset_index()


def futures_features(signal_dates, daily, calendar, *, lag_sessions=1, publication_hour=18):
    if lag_sessions not in (0, 1):
        raise ValueError("Futures lag must be zero or one domestic session.")
    if lag_sessions == 0 and not 18 <= publication_hour <= 23:
        raise ValueError("Same-day futures context requires evening publication after close.")
    dates = pd.Index(pd.Series(signal_dates).to_numpy(dtype=int))
    sessions = pd.Index(pd.Series(calendar).to_numpy(dtype=int))
    for name, values in (("signal", dates), ("calendar", sessions)):
        if values.empty or values.has_duplicates or not values.is_monotonic_increasing:
            raise ValueError(f"Futures {name} dates must be unique and chronological.")
    if len(dates.difference(sessions)) or daily.trade_date.duplicated().any():
        raise ValueError("Missing calendar alignment or duplicate futures days.")
    source = daily.set_index("trade_date").reindex(sessions)
    features = pd.DataFrame(index=sessions)
    for product in PRODUCTS:
        for name in MEASURES:
            features[f"futures_{product}_{name}"] = source[f"{product}_{name}"]
        volume = source[f"{product}_volume"]
        interest = source[f"{product}_open_interest"]
        features[f"futures_{product}_volume_ratio_20"] = volume / volume.rolling(20).mean()
        features[f"futures_{product}_interest_change_1"] = interest.pct_change(fill_method=None)
    lagged = features.shift(lag_sessions).reindex(dates)
    if not np.isfinite(lagged.to_numpy()).all():
        raise ValueError("Missing prior-session futures context; no forward filling is allowed.")
    source_dates = pd.Series(sessions, index=sessions).shift(lag_sessions).reindex(dates)
    aligned_dates = pd.Series(dates, index=dates)
    available = source_dates.lt(aligned_dates) if lag_sessions else source_dates.eq(aligned_dates)
    if not available.all():
        raise ValueError("Futures source session disagrees with the publication policy.")
    lagged["futures_source_date"] = source_dates.astype(int)
    lagged.index.name = "trade_date"
    return lagged.reset_index()
