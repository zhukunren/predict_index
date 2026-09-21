"""Expose current price direction to the incumbent's unchanged rule ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


CANDIDATES = {"current_price_rules": {"features": ["return_1", "overnight_gap", "intraday_return"]}}
PARAMETERS = {"threshold_window": 756, "minimum_threshold_rows": 40,
              "quantiles": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}


def price_rule_cache(core, base, cache):
    prices = pd.DataFrame({
        "return_1": base.close.pct_change(fill_method=None),
        "overnight_gap": base.open / base.pre_close - 1,
        "intraday_return": base.close / base.open - 1,
    })
    rows, names = [], []
    for name, series in prices.items():
        values = series.to_numpy(dtype=float)
        for quantile in PARAMETERS["quantiles"]:
            threshold = series.shift(1).rolling(PARAMETERS["threshold_window"],
                                               min_periods=PARAMETERS["minimum_threshold_rows"]).quantile(quantile).to_numpy()
            available = np.isfinite(values) & np.isfinite(threshold)
            above = np.full(len(base), -1, dtype=np.int8)
            below = above.copy()
            above[available] = (values[available] > threshold[available]).astype(np.int8)
            below[available] = 1 - above[available]
            rows.extend([above, below])
            names.extend([f"{name}>rolling_q{quantile:.1f}", f"{name}<=rolling_q{quantile:.1f}"])
    return core.NestedRuleCache(predictions=np.vstack([cache.predictions, *rows]),
                                rule_names=[*cache.rule_names, *names], labels=cache.labels.copy(),
                                valid_mask=cache.valid_mask.copy())


def selector(core, name):
    if name not in CANDIDATES:
        raise ValueError(f"Unknown price rule candidate: {name}.")
    original = core._nested_volatility_rule_signal
    last_cache = None
    expanded = None

    def signal(*, base, cache, idx, calibration_window, min_calibration_rows, top_k):
        nonlocal last_cache, expanded
        if cache is not last_cache:
            expanded = price_rule_cache(core, base, cache)
            last_cache = cache
        return original(base=base, cache=expanded, idx=idx, calibration_window=calibration_window,
                        min_calibration_rows=min_calibration_rows, top_k=top_k)

    return signal
