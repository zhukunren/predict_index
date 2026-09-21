"""Causal cross-sectional tails, turnover concentration and stock transitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.breadth_features import aggregate_day
from tools.liquidity_features import date_values


MEASURES = ("q10", "q90", "iqr", "amount_return_gap", "amount_top_decile",
            "advance_to_decline", "decline_persistence", "decline_to_advance")
FEATURE_COLUMNS = tuple(f"distribution_{name}" for name in (*MEASURES, "q10_change_1", "advance_to_decline_mean_5"))


def active_stocks(frame, date):
    aggregate_day(frame, date)
    return frame.loc[frame.ts_code.str.endswith((".SH", ".SZ")) & frame.vol.gt(0) & frame.amount.gt(0)].set_index("ts_code")


def distribution_day(current, previous, date, previous_date):
    if int(previous_date) >= int(date):
        raise ValueError("Stock transitions require an earlier source date.")
    current, previous = active_stocks(current, date), active_stocks(previous, previous_date)
    matched = current.index.intersection(previous.index)
    if matched.empty:
        raise ValueError("Stock transitions require matched actively traded stocks.")
    returns = current.pct_chg.astype(float) / 100
    previous_returns = previous.loc[matched, "pct_chg"].astype(float) / 100
    matched_returns = returns.loc[matched]
    q01, q10, q25, q75, q90, q99 = returns.quantile([0.01, 0.1, 0.25, 0.75, 0.9, 0.99])
    robust = returns.clip(q01, q99)
    amount = current.amount.astype(float)
    return {
        "trade_date": int(date), "previous_source_date": int(previous_date),
        "active_rows": len(current), "matched_rows": len(matched),
        "q10": q10, "q90": q90, "iqr": q75 - q25,
        "amount_return_gap": float(np.average(robust, weights=amount) - robust.mean()),
        "amount_top_decile": float(amount.nlargest(max(1, int(np.ceil(len(amount) / 10)))).sum() / amount.sum()),
        "advance_to_decline": float((previous_returns.gt(0) & matched_returns.lt(0)).mean()),
        "decline_persistence": float((previous_returns.lt(0) & matched_returns.lt(0)).mean()),
        "decline_to_advance": float((previous_returns.lt(0) & matched_returns.gt(0)).mean()),
    }


def distribution_features(signal_dates, daily, calendar, *, publication_hour=18):
    if not 17 <= publication_hour <= 23:
        raise ValueError("Distribution features require completed same-day data ingestion.")
    signals, calendar = date_values(signal_dates), date_values(calendar)
    source = daily.copy()
    source["trade_date"] = date_values(source.trade_date).to_numpy()
    previous = date_values(source.previous_source_date)
    for dates in (signals, calendar, source.trade_date):
        if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Distribution dates must be unique and chronological.")
    locations = pd.Index(calendar).get_indexer(source.trade_date)
    if (locations < 1).any() or not np.array_equal(calendar.iloc[locations - 1], previous):
        raise ValueError("Distribution transitions require consecutive domestic sessions.")
    if (pd.Index(calendar).get_indexer(signals) < 0).any():
        raise ValueError("Distribution signal dates are outside the market calendar.")
    source = source.set_index("trade_date").reindex(calendar)
    values = source.loc[:, MEASURES].astype(float)
    fractions = values[["amount_top_decile", "advance_to_decline", "decline_persistence", "decline_to_advance"]]
    if ((fractions < 0) | (fractions > 1)).any().any():
        raise ValueError("Distribution fractions must be within [0, 1].")
    values["q10_change_1"] = values.q10.diff()
    values["advance_to_decline_mean_5"] = values.advance_to_decline.rolling(5).mean()
    selected = values.reindex(signals).reset_index(drop=True).add_prefix("distribution_")
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("Missing distribution history; no stale filling.")
    selected.insert(0, "distribution_source_date", signals.to_numpy())
    selected.insert(0, "trade_date", signals.to_numpy())
    return selected
