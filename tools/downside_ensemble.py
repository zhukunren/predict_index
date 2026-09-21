"""Causal convex pooling of two emitted conditional downside forecasts."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from tools.downside_specialist import mean_downside_probabilities


PARAMETERS = {"window": 252, "minimum_rows": 60, "prior_rows": 20, "loss": "brier", "prior_weight": 0.5}


def adaptive_downside_probabilities(champion, left, right):
    result = mean_downside_probabilities(left, right)
    if not np.array_equal(champion.trade_date, result.trade_date):
        raise ValueError("Adaptive ensemble requires identical signal dates.")
    labels = champion.predicted_label.to_numpy(dtype=float)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Adaptive ensemble requires binary incumbent directions.")
    actual = champion.real_pct_change.to_numpy(dtype=float)
    fitted = result.downside_training_rows.to_numpy() > 0
    eligible = fitted & (labels == 1)
    known = eligible & np.isfinite(actual)
    target = (actual <= 0).astype(float)
    left_probability = left.downside_probability.to_numpy(dtype=float)
    right_probability = right.downside_probability.to_numpy(dtype=float)
    weights = np.full(len(result), PARAMETERS["prior_weight"])
    sample_rows = np.zeros(len(result), dtype=int)
    last_dates = np.zeros(len(result), dtype=int)
    applied = np.zeros(len(result), dtype=bool)
    for index in range(len(result)):
        if not eligible[index]:
            continue
        history = np.arange(max(0, index - PARAMETERS["window"]), index)
        history = history[known[history]]
        sample_rows[index] = len(history)
        if len(history):
            last_dates[index] = int(result.trade_date.iloc[history[-1]])
        if len(history) < PARAMETERS["minimum_rows"]:
            continue
        difference = left_probability[history] - right_probability[history]
        scale = float(np.mean(difference ** 2))
        if scale <= 1e-12:
            continue
        residual = right_probability[history] - target[history]
        # A fixed prior shrinks the learned mixture toward equal weights.
        penalty = PARAMETERS["prior_rows"] / len(history) * scale
        def loss(weight):
            return float(np.mean((residual + weight * difference) ** 2)
                         + penalty * (weight - PARAMETERS["prior_weight"]) ** 2)
        fit = minimize_scalar(loss, bounds=(0, 1), method="bounded", options={"xatol": 1e-8})
        if not fit.success or not np.isfinite(fit.x):
            raise ValueError("Adaptive ensemble weight optimization failed.")
        weights[index] = float(fit.x)
        applied[index] = True
    result["downside_probability"] = np.where(fitted, weights * left_probability + (1 - weights) * right_probability, 0)
    result["downside_left_weight"] = weights
    result["downside_weight_training_rows"] = sample_rows
    result["downside_weight_last_training_date"] = last_dates
    result["downside_weight_applied"] = applied
    return result
