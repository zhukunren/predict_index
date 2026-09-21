"""Cross-sectional multi-session trends from observed stock daily returns.

No present-day constituent list, prices from the future, suspended-day filling,
or direction forecasts enter these features. Returns are compounded provider
pct_chg observations; they are not claimed to be an adjusted-price series.
"""

from collections import deque

import numpy as np
import pandas as pd

from tools.breadth_features import aggregate_day
from tools.liquidity_features import date_values


WINDOWS = (5, 20, 60)
MIN_COVERAGE = 0.90
FEATURE_COLUMNS = (
    "trend_up_5_fraction", "trend_up_20_fraction", "trend_up_60_fraction",
    "trend_negative_20_amount_fraction", "trend_weakening_fraction",
    "trend_shanghai_relative_20", "trend_up_20_change_5", "trend_return_20_median",
)
POLICY = {
    "windows": WINDOWS, "minimum_stock_and_turnover_coverage": MIN_COVERAGE,
    "universe": "historical active Shanghai 6xxxxx.SH and Shenzhen 0xxxxx.SZ/3xxxxx.SZ A-shares",
    "momentum": "product of (1 + pct_chg / 100) over consecutive domestic sessions minus one",
    "missing": "missing or inactive stock session resets its history; no filling",
    "weight": "current-session turnover for negative 20-session momentum amount share",
    "publication_hour_shanghai": 18,
    "historical_delivery_timestamps_available": False,
}


class TrendBreadthAccumulator:
    def __init__(self, calendar):
        self.calendar = date_values(calendar).to_numpy()
        if len(self.calendar) == 0 or (np.diff(self.calendar) <= 0).any():
            raise ValueError("Trend calendar must be unique and chronological.")
        self.position = 0
        self.history = {}
        self.up20_history = deque(maxlen=5)

    def step(self, daily, date):
        if self.position >= len(self.calendar) or int(date) != self.calendar[self.position]:
            raise ValueError("Trend source must cover each consecutive calendar session.")
        aggregate_day(daily, date)
        selected = daily.loc[daily.ts_code.str.fullmatch(r"(?:6[0-9]{5}\.SH|[03][0-9]{5}\.SZ)")].copy()
        for name in ("pct_chg", "vol", "amount"):
            selected[name] = pd.to_numeric(selected[name], errors="raise")
        selected = selected.loc[selected.vol.gt(0) & selected.amount.gt(0)].sort_values("ts_code")
        if selected.empty or selected.pct_chg.le(-100).any():
            raise ValueError("Trend requires active A-shares with returns greater than -100 percent.")
        codes = selected.ts_code.to_numpy()
        amount = selected.amount.to_numpy(dtype=float)
        log_returns = np.log1p(selected.pct_chg.to_numpy(dtype=float) / 100)
        momentum = np.full((len(codes), len(WINDOWS)), np.nan)
        next_history = {}
        for index, (code, value) in enumerate(zip(codes, log_returns, strict=True)):
            prior = self.history.get(code)
            history = deque(prior, maxlen=max(WINDOWS) + 1) if prior is not None else deque([0.0], maxlen=max(WINDOWS) + 1)
            current = history[-1] + value
            history.append(current)
            for column, window in enumerate(WINDOWS):
                if len(history) > window:
                    momentum[index, column] = np.expm1(current - history[-window - 1])
            next_history[code] = history
        # Dropping absent securities resets suspended and newly relisted histories.
        self.history = next_history
        self.position += 1
        result = {"trade_date": int(date), "active_rows": len(codes)}
        coverage_ok = True
        for column, window in enumerate(WINDOWS):
            valid = np.isfinite(momentum[:, column])
            count_coverage = float(valid.mean())
            amount_coverage = float(amount[valid].sum() / amount.sum())
            result.update({f"eligible_{window}_rows": int(valid.sum()),
                           f"stock_coverage_{window}": count_coverage,
                           f"amount_coverage_{window}": amount_coverage,
                           f"trend_up_{window}_fraction": float((momentum[valid, column] > 0).mean()) if valid.any() else np.nan})
            coverage_ok &= min(count_coverage, amount_coverage) >= MIN_COVERAGE
        valid20 = np.isfinite(momentum[:, 1])
        shanghai = np.array([code.endswith(".SH") for code in codes]) & valid20
        result.update(
            trend_negative_20_amount_fraction=float(amount[valid20 & (momentum[:, 1] < 0)].sum() / amount[valid20].sum()) if valid20.any() else np.nan,
            trend_weakening_fraction=float(((momentum[valid20, 0] < 0) & (momentum[valid20, 1] > 0)).mean()) if valid20.any() else np.nan,
            trend_shanghai_relative_20=float((momentum[shanghai, 1] > 0).mean() - (momentum[valid20, 1] > 0).mean()) if shanghai.any() else np.nan,
            trend_up_20_change_5=result["trend_up_20_fraction"] - self.up20_history[0] if len(self.up20_history) == 5 else np.nan,
            trend_return_20_median=float(np.median(momentum[valid20, 1])) if valid20.any() else np.nan,
        )
        self.up20_history.append(result["trend_up_20_fraction"])
        result["trend_available"] = bool(coverage_ok and np.isfinite([result[name] for name in FEATURE_COLUMNS]).all())
        return result


def trend_breadth_features(signal_dates, daily, calendar, *, publication_hour=18):
    if not 18 <= publication_hour <= 23:
        raise ValueError("Trend breadth requires completed same-day data after 18:00 Shanghai.")
    signals, calendar = date_values(signal_dates), date_values(calendar)
    source = daily.copy()
    source["trade_date"] = date_values(source.trade_date).to_numpy()
    for dates in (signals, calendar, source.trade_date):
        if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Trend dates must be unique and chronological.")
    required = calendar.loc[calendar.ge(source.trade_date.iloc[0]) & calendar.le(signals.iloc[-1])]
    if not signals.isin(calendar).all() or not required.isin(source.trade_date).all() or not signals.isin(source.trade_date).all():
        raise ValueError("Missing trend sessions; stale source rows cannot be filled.")
    result = source.set_index("trade_date").loc[signals].reset_index()
    if not result.trend_available.isin((True, False)).all():
        raise ValueError("Trend availability must be boolean.")
    for window in WINDOWS:
        for kind in ("stock", "amount"):
            coverage = pd.to_numeric(result[f"{kind}_coverage_{window}"], errors="raise")
            if not coverage.between(0, 1 + 1e-12).all():
                raise ValueError("Invalid trend coverage.")
            if (result.trend_available & coverage.lt(MIN_COVERAGE)).any():
                raise ValueError("Incomplete trend histories cannot be marked available.")
    values = result.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if np.isinf(values).any() or not np.isfinite(values[result.trend_available.to_numpy(dtype=bool)]).all():
        raise ValueError("Available trend features must be finite.")
    result.insert(1, "trend_source_date", result.trade_date)
    return result.loc[:, ["trade_date", "trend_source_date", "trend_available", *FEATURE_COLUMNS]]
