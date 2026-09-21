"""Lagged stock order-size flows with historical-universe coverage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.breadth_features import aggregate_day
from tools.liquidity_features import date_values


AMOUNTS = tuple(f"{side}_{size}_amount" for side in ("buy", "sell") for size in ("sm", "md", "lg", "elg"))
FIELDS = ("ts_code", "trade_date", *AMOUNTS)
MIN_COVERAGE = 0.98
MEASURES = ("large_net_ratio", "small_net_ratio", "large_selling_fraction", "large_selling_amount_fraction",
            "large_net_median", "large_small_divergence", "shanghai_relative")
FEATURE_COLUMNS = tuple(f"moneyflow_{name}" for name in (*MEASURES, "large_net_change_1", "large_net_mean_5", "large_net_mean_20"))
PRICE_MEASURES = ("up_sell_amount_fraction", "down_buy_amount_fraction", "high_turnover_large_ratio", "other_turnover_large_ratio")
PRICE_FEATURE_COLUMNS = tuple(f"moneyflow_{name}" for name in (*PRICE_MEASURES, "up_sell_mean_5", "down_buy_mean_5"))
NORMALIZATION = {"window": 126, "minimum_rows": 126, "clip": 5, "minimum_scale": 1e-12}


def aggregate_moneyflow(flow, daily, date, *, price_context=False):
    aggregate_day(daily, date)
    if flow.empty or not set(FIELDS).issubset(flow.columns):
        raise ValueError("Moneyflow requires complete provider fields.")
    source = flow.loc[:, FIELDS].copy()
    if source.ts_code.duplicated().any() or not source.trade_date.astype(str).eq(str(date)).all():
        raise ValueError("Moneyflow contains duplicate stocks or unexpected dates.")
    if not source.ts_code.str.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)").all():
        raise ValueError("Moneyflow requires supported stock codes.")
    source = source.loc[source.ts_code.str.endswith((".SH", ".SZ"))].set_index("ts_code")
    source.loc[:, AMOUNTS] = source.loc[:, AMOUNTS].apply(pd.to_numeric, errors="raise")
    amounts = source.loc[:, AMOUNTS].to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Moneyflow amounts must be finite and nonnegative.")
    active = daily.loc[daily.ts_code.str.endswith((".SH", ".SZ")) & daily.vol.gt(0) & daily.amount.gt(0)].set_index("ts_code")
    gross = source.loc[:, AMOUNTS].sum(axis=1)
    matched = active.index.intersection(source.index[gross > 0])
    stock_coverage = len(matched) / len(active)
    amount_coverage = float(active.loc[matched, "amount"].sum() / active.amount.sum())
    result = {"trade_date": int(date), "raw_rows": len(flow), "active_rows": len(active), "matched_rows": len(matched),
              "stock_coverage": stock_coverage, "amount_coverage": amount_coverage,
              **{name: np.nan for name in (*MEASURES, *(PRICE_MEASURES if price_context else ()))}}
    shanghai = matched.str.endswith(".SH")
    if min(stock_coverage, amount_coverage) < MIN_COVERAGE or not shanghai.any():
        return result
    source, gross = source.loc[matched], gross.loc[matched]
    large = source.buy_lg_amount + source.buy_elg_amount - source.sell_lg_amount - source.sell_elg_amount
    small = source.buy_sm_amount - source.sell_sm_amount
    total = gross.sum()
    net = float(large.sum() / total)
    result.update({
        "large_net_ratio": net, "small_net_ratio": float(small.sum() / total),
        "large_selling_fraction": float(large.lt(0).mean()),
        "large_selling_amount_fraction": float(gross[large.lt(0)].sum() / total),
        "large_net_median": float((large / gross).median()),
        "large_small_divergence": float((large.lt(0) & small.gt(0)).mean()),
        "shanghai_relative": float(large[shanghai].sum() / gross[shanghai].sum() - net),
    })
    if price_context:
        prices = active.loc[matched]
        up_sell = prices.pct_chg.gt(0) & large.lt(0)
        down_buy = prices.pct_chg.lt(0) & large.gt(0)
        leaders = prices.amount.nlargest(max(1, int(np.ceil(len(prices) / 10)))).index
        remainder = prices.index.difference(leaders)
        result.update({
            "up_sell_amount_fraction": float(gross[up_sell].sum() / total),
            "down_buy_amount_fraction": float(gross[down_buy].sum() / total),
            "high_turnover_large_ratio": float(large.loc[leaders].sum() / gross.loc[leaders].sum()),
            "other_turnover_large_ratio": float(large.loc[remainder].sum() / gross.loc[remainder].sum()) if len(remainder) else np.nan,
        })
    return result


def moneyflow_features(signal_dates, daily, calendar, *, price_context=False, normalize=False):
    signals, calendar = date_values(signal_dates), date_values(calendar)
    source = daily.copy()
    source["trade_date"] = date_values(source.trade_date).to_numpy()
    for dates in (signals, calendar, source.trade_date):
        if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Moneyflow dates must be unique and chronological.")
    positions = pd.Index(calendar).get_indexer(signals)
    if (positions < 20).any():
        raise ValueError("Moneyflow requires aligned signals and 20 prior domestic sessions.")
    required = calendar.iloc[positions[0] - 20:positions[-1]]
    if not pd.Index(required).isin(source.trade_date).all():
        raise ValueError("Missing moneyflow sessions; no stale filling.")
    source = source.set_index("trade_date").reindex(calendar)
    for name in ("stock_coverage", "amount_coverage"):
        values = source[name].dropna()
        if not values.between(0, 1 + 1e-12).all():
            raise ValueError("Moneyflow coverage must be a fraction.")
    measures = (*MEASURES, *(PRICE_MEASURES if price_context else ()))
    values = source.loc[:, measures].astype(float)
    available = source.stock_coverage.ge(MIN_COVERAGE) & source.amount_coverage.ge(MIN_COVERAGE)
    values = values.where(available, np.nan)
    values["large_net_change_1"] = values.large_net_ratio.diff()
    values["large_net_mean_5"] = values.large_net_ratio.rolling(5).mean()
    values["large_net_mean_20"] = values.large_net_ratio.rolling(20).mean()
    if price_context:
        values["up_sell_mean_5"] = values.up_sell_amount_fraction.rolling(5).mean()
        values["down_buy_mean_5"] = values.down_buy_amount_fraction.rolling(5).mean()
    if normalize:
        history = values.shift(1).rolling(NORMALIZATION["window"], min_periods=NORMALIZATION["minimum_rows"])
        scale = history.std(ddof=0)
        scale = scale.where(scale.gt(NORMALIZATION["minimum_scale"]))
        values = ((values - history.mean()) / scale).clip(-NORMALIZATION["clip"], NORMALIZATION["clip"])
    source_dates = calendar.iloc[positions - 1].to_numpy()
    selected = values.reindex(source_dates).reset_index(drop=True).add_prefix("moneyflow_")
    selected.insert(0, "moneyflow_available", np.isfinite(selected.to_numpy(dtype=float)).all(axis=1))
    selected.insert(0, "moneyflow_source_date", source_dates)
    selected.insert(0, "trade_date", signals.to_numpy())
    return selected
