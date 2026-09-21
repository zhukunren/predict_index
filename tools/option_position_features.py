"""Expiry-aware put/call positions and changes in matched contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.option_features import DAILY_FIELDS, aggregate_options, validate_contracts
from tools.liquidity_features import date_values


NEAR_EXPIRY_DAYS = 30
MEASURES = (
    "near_put_interest_share", "far_put_interest_share", "near_interest_share", "put_interest_term_spread",
    "matched_put_interest_growth", "matched_call_interest_growth", "matched_interest_imbalance",
)
FEATURE_COLUMNS = tuple(f"option_position_{name}" for name in (*MEASURES, "matched_imbalance_mean_5"))


def aggregate_positions(raw, contracts, required_dates):
    totals = aggregate_options(raw, contracts, required_dates)
    metadata = validate_contracts(contracts)
    frame = raw.loc[raw.ts_code.isin(metadata.ts_code), DAILY_FIELDS].copy()
    frame["trade_date"] = date_values(frame.trade_date).to_numpy()
    frame["oi"] = pd.to_numeric(frame.oi, errors="raise")
    frame = frame.merge(metadata[["ts_code", "call_put", "delist_date"]], on="ts_code", validate="many_to_one")
    expiry = pd.to_datetime(frame.delist_date.astype(str), format="%Y%m%d")
    observed = pd.to_datetime(frame.trade_date.astype(str), format="%Y%m%d")
    frame["near_expiry"] = (expiry - observed).dt.days.le(NEAR_EXPIRY_DAYS)
    previous = None
    rows = []
    for date, group in frame.groupby("trade_date", sort=True):
        row = {"trade_date": int(date), **{name: np.nan for name in MEASURES},
               "matched_contract_rows": 0, "matched_prior_interest_coverage": np.nan}
        total_interest = float(group.oi.sum())
        for near, name in ((True, "near"), (False, "far")):
            bucket = group.loc[group.near_expiry.eq(near)]
            interest = float(bucket.oi.sum())
            if interest > 0:
                row[f"{name}_put_interest_share"] = float(bucket.loc[bucket.call_put.eq("P"), "oi"].sum() / interest)
            if near:
                row["near_interest_share"] = interest / total_interest
        row["put_interest_term_spread"] = row["near_put_interest_share"] - row["far_put_interest_share"]
        current = group.set_index("ts_code")[["oi", "call_put"]]
        if previous is not None:
            common = current.index.intersection(previous.index)
            before, after = previous.loc[common], current.loc[common]
            denominator = float(before.oi.sum())
            row["matched_contract_rows"] = len(common)
            row["matched_prior_interest_coverage"] = denominator / float(previous.oi.sum())
            if denominator > 0:
                change = after.oi - before.oi
                put = before.call_put.eq("P")
                put_change, call_change = float(change[put].sum()), float(change[~put].sum())
                for name, mask, difference in (("put", put, put_change), ("call", ~put, call_change)):
                    prior = float(before.loc[mask, "oi"].sum())
                    if prior > 0:
                        row[f"matched_{name}_interest_growth"] = difference / prior
                row["matched_interest_imbalance"] = (put_change - call_change) / denominator
        previous = current
        rows.append(row)
    return totals.merge(pd.DataFrame(rows), on="trade_date", validate="one_to_one")


def position_features(signal_dates, daily, calendar, *, lag_sessions=1, publication_hour=18):
    if lag_sessions not in (0, 1) or not 0 <= publication_hour <= 23:
        raise ValueError("Invalid option position timing policy.")
    if lag_sessions == 0 and publication_hour < 20:
        raise ValueError("Same-day option research requires publication at or after 20:00.")
    signals, calendar = date_values(signal_dates), date_values(calendar)
    source = daily.copy()
    source["trade_date"] = date_values(source.trade_date).to_numpy()
    for dates in (signals, calendar, source.trade_date):
        if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Option position dates must be unique and chronological.")
    positions = pd.Index(calendar).get_indexer(signals)
    if (positions < 5).any():
        raise ValueError("Option position signals require aligned dates and five prior sessions.")
    values = source.set_index("trade_date").reindex(calendar).loc[:, MEASURES].astype(float)
    observed = values.to_numpy()
    if np.isinf(observed).any():
        raise ValueError("Option position values cannot be infinite.")
    for name in ("near_put_interest_share", "far_put_interest_share", "near_interest_share"):
        if not values[name].dropna().between(0, 1).all():
            raise ValueError("Option position shares must be fractions.")
    values["matched_imbalance_mean_5"] = values.matched_interest_imbalance.rolling(5).mean()
    source_dates = calendar.iloc[positions - lag_sessions].to_numpy()
    result = values.reindex(source_dates).reset_index(drop=True).add_prefix("option_position_")
    result.insert(0, "option_position_available", np.isfinite(result.to_numpy(dtype=float)).all(axis=1))
    result.insert(0, "option_position_source_date", source_dates)
    result.insert(0, "trade_date", signals.to_numpy())
    return result
