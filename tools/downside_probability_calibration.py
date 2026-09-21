"""Calibrate emitted downside scores using earlier resolved up forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


PARAMETERS = {"window": 252, "minimum_rows": 60, "method": "isotonic_increasing"}


def calibrate_downside_probabilities(champion, probabilities):
    if not np.array_equal(champion.trade_date, probabilities.trade_date):
        raise ValueError("Downside calibration requires identical signal dates.")
    dates = probabilities.trade_date
    if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Downside calibration requires unique chronological signals.")
    frame = probabilities.copy().reset_index(drop=True)
    raw = frame.downside_probability.to_numpy(dtype=float)
    if not (np.isfinite(raw) & (raw >= 0) & (raw <= 1)).all():
        raise ValueError("Raw downside scores must be finite probabilities.")
    fitted = frame.downside_training_rows.to_numpy(dtype=int) > 0
    if (fitted & frame.downside_last_training_date.ge(frame.trade_date)).any():
        raise ValueError("Underlying model training must precede its signal date.")
    labels = champion.predicted_label.to_numpy(dtype=float)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Downside calibration requires binary incumbent labels.")
    actual = champion.real_pct_change.to_numpy(dtype=float)
    target = (actual <= 0).astype(int)
    eligible = fitted & (labels == 1)
    known = eligible & np.isfinite(actual)
    calibrated = raw.copy()
    sample_counts = np.zeros(len(frame), dtype=int)
    last_dates = np.zeros(len(frame), dtype=int)
    applied = np.zeros(len(frame), dtype=bool)
    for index in range(len(frame)):
        if not eligible[index]:
            continue
        history = np.arange(max(0, index - PARAMETERS["window"]), index)
        history = history[known[history]]
        sample_counts[index] = len(history)
        if len(history):
            last_dates[index] = int(frame.trade_date.iloc[history[-1]])
        if len(history) < PARAMETERS["minimum_rows"] or len(np.unique(target[history])) < 2:
            continue
        calibrator = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")
        calibrator.fit(raw[history], target[history])
        calibrated[index] = float(calibrator.predict(raw[index:index + 1])[0])
        applied[index] = True
    frame["uncalibrated_downside_probability"] = raw
    frame["downside_probability"] = calibrated
    frame["downside_calibration_rows"] = sample_counts
    frame["downside_calibration_last_date"] = last_dates
    frame["downside_calibration_applied"] = applied
    return frame
