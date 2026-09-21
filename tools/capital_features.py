"""Lagged Shanghai A-share capitalization breadth and size-conditioned flows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.liquidity_features import date_values
from tools.moneyflow_features import AMOUNTS, MIN_COVERAGE, aggregate_moneyflow


FIELDS = ("ts_code", "trade_date", "total_mv")
LAG_SESSIONS = 1
MEASURES = ("cap_up_fraction", "cap_return", "cap_vs_equal_up", "cap_net_ratio",
            "cap_selling_fraction", "large_cap_net_ratio", "small_cap_net_ratio")
FEATURE_COLUMNS = tuple(f"capital_{name}" for name in (*MEASURES, "cap_net_mean_5", "cap_return_mean_5", "cap_up_change_1"))
COVERAGE_COLUMNS = ("basic_stock_coverage", "flow_stock_coverage", "flow_amount_coverage", "flow_cap_coverage")


def capital_day(basic, daily, flow, date):
    # Reuse historical-universe, duplicate, date and nonnegative-flow validation.
    aggregate_moneyflow(flow, daily, date)
    if basic.empty or not set(FIELDS).issubset(basic.columns):
        raise ValueError("Capital context requires complete daily_basic fields.")
    basic = basic.loc[:, FIELDS].copy()
    if (basic.ts_code.duplicated().any() or not basic.trade_date.astype(str).eq(str(date)).all()
            or not basic.ts_code.str.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)").all()):
        raise ValueError("Invalid capitalization codes, dates or duplicate rows.")
    basic["total_mv"] = pd.to_numeric(basic.total_mv, errors="raise")
    if basic.total_mv.lt(0).any() or np.isinf(basic.total_mv).any():
        raise ValueError("Capitalization must not be negative or infinite.")
    active = daily.loc[daily.ts_code.str.fullmatch(r"6[0-9]{5}\.SH") & daily.vol.gt(0) & daily.amount.gt(0)].set_index("ts_code")
    if len(active) < 2:
        raise ValueError("Capital context requires at least two active Shanghai A-shares.")
    caps = basic.set_index("ts_code").total_mv.reindex(active.index)
    basic_coverage = float(caps.gt(0).mean())
    result = {"trade_date": int(date), "active_rows": len(active), "basic_stock_coverage": basic_coverage,
              "flow_stock_coverage": np.nan, "flow_amount_coverage": np.nan, "flow_cap_coverage": np.nan,
              "matched_rows": 0, **{name: np.nan for name in MEASURES}}
    # Missing capitalization has unknown weight; do not infer it from stock counts.
    if basic_coverage < 1:
        return result
    matched_flow = flow.set_index("ts_code").reindex(active.index)
    gross = matched_flow.loc[:, AMOUNTS].sum(axis=1, min_count=len(AMOUNTS))
    matched = active.index[gross.gt(0)]
    stock_coverage = len(matched) / len(active)
    amount_coverage = float(active.loc[matched, "amount"].sum() / active.amount.sum())
    cap_coverage = float(caps.loc[matched].sum() / caps.sum())
    result.update(matched_rows=len(matched), flow_stock_coverage=stock_coverage,
                  flow_amount_coverage=amount_coverage, flow_cap_coverage=cap_coverage)
    if min(stock_coverage, amount_coverage, cap_coverage) < MIN_COVERAGE or len(matched) < 2:
        return result
    prices, caps = active.loc[matched], caps.loc[matched]
    matched_flow, gross = matched_flow.loc[matched], gross.loc[matched]
    large = matched_flow.buy_lg_amount + matched_flow.buy_elg_amount - matched_flow.sell_lg_amount - matched_flow.sell_elg_amount
    up = prices.pct_chg.gt(0)
    # The historical day's ranking avoids a present-day constituent selection.
    leaders = caps.sort_index().sort_values(ascending=False, kind="stable").iloc[:max(1, int(np.ceil(len(caps) / 10)))].index
    remainder = caps.index.difference(leaders)
    result.update(
        cap_up_fraction=float(np.average(up, weights=caps)),
        cap_return=float(np.average(prices.pct_chg / 100, weights=caps)),
        cap_vs_equal_up=float(np.average(up, weights=caps) - up.mean()),
        cap_net_ratio=float(np.average(large / gross, weights=caps)),
        cap_selling_fraction=float(np.average(large.lt(0), weights=caps)),
        large_cap_net_ratio=float(large.loc[leaders].sum() / gross.loc[leaders].sum()),
        small_cap_net_ratio=float(large.loc[remainder].sum() / gross.loc[remainder].sum()),
    )
    return result


def capital_features(signal_dates, daily, calendar):
    signals, calendar = date_values(signal_dates), date_values(calendar)
    source = daily.copy()
    source["trade_date"] = date_values(source.trade_date).to_numpy()
    for dates in (signals, calendar, source.trade_date):
        if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Capital dates must be unique and chronological.")
    positions = pd.Index(calendar).get_indexer(signals)
    if (positions < 20).any():
        raise ValueError("Capital signals require aligned dates and 20 prior sessions.")
    required = calendar.iloc[positions[0] - 20:positions[-1]]
    if not pd.Index(required).isin(source.trade_date).all():
        raise ValueError("Missing capital sessions; no stale filling.")
    source = source.set_index("trade_date").reindex(calendar)
    for column in COVERAGE_COLUMNS:
        if not source[column].dropna().between(0, 1 + 1e-12).all():
            raise ValueError("Capital coverage must be a fraction.")
    ready = source.basic_stock_coverage.eq(1) & source.loc[:, COVERAGE_COLUMNS[1:]].ge(MIN_COVERAGE).all(axis=1)
    values = source.loc[:, MEASURES].astype(float).where(ready, np.nan)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Capital features must not be infinite.")
    values["cap_net_mean_5"] = values.cap_net_ratio.rolling(5).mean()
    values["cap_return_mean_5"] = values.cap_return.rolling(5).mean()
    values["cap_up_change_1"] = values.cap_up_fraction.diff()
    source_dates = calendar.iloc[positions - LAG_SESSIONS].to_numpy()
    selected = values.reindex(source_dates).reset_index(drop=True).add_prefix("capital_")
    selected.insert(0, "capital_available", np.isfinite(selected.to_numpy(dtype=float)).all(axis=1))
    selected.insert(0, "capital_source_date", source_dates)
    selected.insert(0, "trade_date", signals.to_numpy())
    return selected
