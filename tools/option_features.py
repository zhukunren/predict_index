"""Historical 50ETF put/call activity using dated contract membership."""

from __future__ import annotations

import numpy as np
import pandas as pd


UNDERLYING = "OP510050.SH"
CONTRACT_FIELDS = ("ts_code", "opt_code", "call_put", "list_date", "delist_date")
DAILY_FIELDS = ("ts_code", "trade_date", "vol", "amount", "oi")
MEASURES = ("volume", "amount", "interest")
FEATURE_COLUMNS = tuple(f"option_put_{name}_{statistic}" for name in MEASURES
                        for statistic in ("share", "change_1", "mean_20"))


def validate_contracts(contracts):
    if contracts.empty or not set(CONTRACT_FIELDS).issubset(contracts.columns):
        raise ValueError("Option contract history requires the declared fields.")
    frame = contracts.loc[:, CONTRACT_FIELDS].copy()
    if frame.isna().any().any() or frame.ts_code.duplicated().any():
        raise ValueError("Option contracts must be complete and unique.")
    if not frame.opt_code.eq(UNDERLYING).all() or not frame.call_put.isin(("P", "C")).all():
        raise ValueError("Option contracts require a fixed underlying and binary exercise rights.")
    for name in ("list_date", "delist_date"):
        frame[name] = pd.to_datetime(frame[name].astype(str), format="%Y%m%d").dt.strftime("%Y%m%d").astype(int)
    if frame.list_date.gt(frame.delist_date).any():
        raise ValueError("Option listing cannot follow delisting.")
    return frame.sort_values("ts_code").reset_index(drop=True)


def aggregate_options(raw, contracts, required_dates):
    contracts = validate_contracts(contracts)
    dates = pd.Index(pd.Series(required_dates).to_numpy(dtype=int))
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Option dates must be unique and chronological.")
    if raw.empty or not set(DAILY_FIELDS).issubset(raw.columns):
        raise ValueError("Option history requires all declared daily fields.")
    if raw.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate option daily observations.")
    frame = raw.loc[raw.ts_code.isin(contracts.ts_code), DAILY_FIELDS].copy()
    frame["trade_date"] = pd.to_datetime(frame.trade_date.astype(str), format="%Y%m%d").dt.strftime("%Y%m%d").astype(int)
    if len(pd.Index(frame.trade_date.unique()).difference(dates)):
        raise ValueError("Unexpected option trading dates.")
    for name in ("vol", "amount", "oi"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(frame[["vol", "amount", "oi"]].to_numpy()).all() or frame[["vol", "amount", "oi"]].lt(0).any().any():
        raise ValueError("Option activity must be finite and nonnegative.")
    frame = frame.merge(contracts[["ts_code", "call_put"]], on="ts_code", validate="many_to_one")
    groups = dict(tuple(frame.groupby("trade_date", sort=True)))
    rows = []
    for date in dates:
        expected = contracts.loc[contracts.list_date.le(date) & contracts.delist_date.ge(date)]
        group = groups.get(date)
        if group is None or len(expected) == 0 or set(group.ts_code) != set(expected.ts_code):
            raise ValueError(f"Incomplete in-market option contract history for {date}.")
        row = {"trade_date": int(date), "contract_rows": len(group)}
        for right in ("P", "C"):
            side = group.loc[group.call_put.eq(right)]
            if side.empty:
                raise ValueError("Both put and call contracts must be represented.")
            row[f"{right}_contract_rows"] = len(side)
            for name, column in zip(MEASURES, ("vol", "amount", "oi")):
                total = float(side[column].sum())
                if total <= 0:
                    raise ValueError("Aggregate option activity must be positive on each side.")
                row[f"{right}_{name}"] = total
        rows.append(row)
    return pd.DataFrame(rows)


def option_features(signal_dates, daily, calendar, *, lag_sessions=1, publication_hour=18, publication_minute=0):
    if (lag_sessions not in (0, 1) or not 0 <= publication_hour <= 23
            or not 0 <= publication_minute <= 59):
        raise ValueError("Invalid option timing policy.")
    if lag_sessions == 0 and (publication_hour, publication_minute) < (18, 30):
        raise ValueError("Same-day option research requires publication at or after 18:30.")
    dates = pd.Index(pd.Series(signal_dates).to_numpy(dtype=int))
    sessions = pd.Index(pd.Series(calendar).to_numpy(dtype=int))
    for name, values in (("signal", dates), ("calendar", sessions)):
        if values.empty or values.has_duplicates or not values.is_monotonic_increasing:
            raise ValueError(f"Option {name} dates must be unique and chronological.")
    if daily.trade_date.duplicated().any() or len(dates.difference(sessions)):
        raise ValueError("Option data require unique dates and calendar alignment.")
    source = daily.set_index("trade_date").reindex(sessions)
    features = pd.DataFrame(index=sessions)
    for name in MEASURES:
        put, call = source[f"P_{name}"], source[f"C_{name}"]
        for side in (put, call):
            observed = side.dropna()
            if not (np.isfinite(observed) & observed.gt(0)).all():
                raise ValueError("Aggregate option activity must be finite and positive.")
        share = put / (put + call)
        features[f"option_put_{name}_share"] = share
        features[f"option_put_{name}_change_1"] = share.diff()
        features[f"option_put_{name}_mean_20"] = share.rolling(20).mean()
    available = features.shift(lag_sessions).reindex(dates)
    if not np.isfinite(available.to_numpy()).all():
        timing = "prior-session" if lag_sessions else "same-day"
        raise ValueError(f"Missing {timing} option history; no forward filling is allowed.")
    source_dates = pd.Series(sessions, index=sessions).shift(lag_sessions).reindex(dates)
    expected_dates = pd.Series(dates, index=dates)
    aligned = source_dates.lt(expected_dates) if lag_sessions else source_dates.eq(expected_dates)
    if not aligned.all():
        raise ValueError("Option source session violates the declared signal timing.")
    available["option_source_date"] = source_dates.astype(int)
    available.index.name = "trade_date"
    return available.reset_index()
