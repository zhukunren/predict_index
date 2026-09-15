"""Causal absolute-error calibration for signed return forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_returns(frame: pd.DataFrame, *, window: int = 252, min_rows: int = 60) -> pd.DataFrame:
    """Fit a nonnegative shrink factor to prior, completed forecasts only.

    The weighted median of realized/predicted returns minimizes absolute error
    for a single scale parameter. Restricting the scale to [0, 1] prevents
    extrapolation and leaves the separately recorded direction unchanged.
    """
    if window < min_rows or min_rows < 2:
        raise ValueError("Return calibration requires window >= min_rows >= 2.")
    result = frame.copy()
    predicted = pd.to_numeric(result["predicted_pct_change"], errors="coerce").to_numpy(dtype=float)
    realized = pd.to_numeric(result["real_pct_change"], errors="coerce").to_numpy(dtype=float)
    scales = np.zeros(len(result), dtype=float)
    counts = np.zeros(len(result), dtype=int)
    for index in range(len(result)):
        start = max(0, index - window)
        x = predicted[start:index]
        y = realized[start:index]
        valid = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-12)
        counts[index] = int(valid.sum())
        if counts[index] < min_rows:
            continue
        ratios = y[valid] / x[valid]
        weights = np.abs(x[valid])
        order = np.argsort(ratios, kind="stable")
        median_index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2.0)
        scales[index] = np.clip(ratios[order[median_index]], 0.0, 1.0)
    previous_close = result["predicted_close"].to_numpy(dtype=float) / (1.0 + predicted)
    result["uncalibrated_predicted_return"] = predicted
    result["return_calibration_scale"] = scales
    result["return_calibration_rows"] = counts
    result["predicted_pct_change"] = predicted * scales
    result["predicted_close"] = previous_close * (1.0 + result["predicted_pct_change"])
    return result
