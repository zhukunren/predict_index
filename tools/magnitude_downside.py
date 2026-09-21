"""Estimate the downside share of absolute returns on incumbent-up samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from tools.downside_specialist import COMMON, CANDIDATES


POLICY = {
    "weight": "absolute resolved next return / mean absolute return in the prior training window",
    "zero_returns": "exclude from training because their absolute-return loss is zero",
    "score_semantics": "downside share of expected absolute return, not event probability",
    "weight_clipping": None,
}


def magnitude_scores(champion, features, name, *, available=None):
    if name not in ("downside_logistic", "downside_shallow"):
        raise ValueError("Magnitude weighting supports only the two frozen classic models.")
    frame = champion.copy().reset_index(drop=True)
    if (frame.empty or frame.trade_date.duplicated().any() or not frame.trade_date.is_monotonic_increasing
            or not np.array_equal(frame.trade_date, features.trade_date)):
        raise ValueError("Magnitude scores require aligned, unique chronological signals.")
    labels = frame.predicted_label.to_numpy(dtype=float)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Magnitude scores require binary incumbent labels.")
    values = features.drop(columns="trade_date").to_numpy(dtype=float)
    available = np.ones(len(frame), dtype=bool) if available is None else np.asarray(available)
    if available.shape != (len(frame),) or available.dtype != np.dtype(bool):
        raise ValueError("Magnitude availability must be an aligned boolean mask.")
    if not np.isfinite(values[available]).all():
        raise ValueError("Available magnitude features must be finite.")
    actual = frame.real_pct_change.to_numpy(dtype=float)
    if np.isinf(actual).any():
        raise ValueError("Resolved returns must be finite or unknown.")
    known = np.isfinite(actual) & (actual != 0) & (labels == 1) & available
    target = (actual <= 0).astype(int)
    scores = np.zeros(len(frame))
    counts, last_dates = np.zeros(len(frame), dtype=int), np.zeros(len(frame), dtype=int)
    effective_rows, max_shares = np.zeros(len(frame)), np.zeros(len(frame))
    model = None
    last_fit, count, last_date, effective, max_share = -COMMON["refit_interval"], 0, 0, 0., 0.
    settings = CANDIDATES[name]
    for index in range(len(frame)):
        if labels[index] != 1 or not available[index]:
            continue
        history = np.arange(max(0, index - COMMON["train_window"]), index)
        history = history[known[history]]
        if len(history) < COMMON["min_train_rows"] or len(np.unique(target[history])) < 2:
            continue
        if model is None or index - last_fit >= COMMON["refit_interval"]:
            weights = np.abs(actual[history])
            weights = weights / weights.mean()
            if name == "downside_logistic":
                model = make_pipeline(StandardScaler(), LogisticRegression(**settings))
                model.fit(values[history], target[history], standardscaler__sample_weight=weights,
                          logisticregression__sample_weight=weights)
            else:
                parameters = {key: value for key, value in settings.items() if key != "num_boost_round"}
                model = xgb.train(parameters, xgb.DMatrix(values[history], label=target[history], weight=weights, nthread=1),
                                  num_boost_round=settings["num_boost_round"])
            last_fit, count, last_date = index, len(history), int(frame.trade_date.iloc[history[-1]])
            effective = float(weights.sum() ** 2 / np.square(weights).sum())
            max_share = float(weights.max() / weights.sum())
        if name == "downside_logistic":
            scores[index] = model.predict_proba(values[index:index + 1])[0, 1]
        else:
            scores[index] = model.predict(xgb.DMatrix(values[index:index + 1], nthread=1))[0]
        counts[index], last_dates[index] = count, last_date
        effective_rows[index], max_shares[index] = effective, max_share
    # Keep the shared research prediction schema; POLICY records the distinct score semantics.
    return pd.DataFrame({"trade_date": frame.trade_date, "downside_probability": scores,
                         "downside_training_rows": counts, "downside_last_training_date": last_dates,
                         "magnitude_effective_training_rows": effective_rows, "magnitude_largest_weight_share": max_shares})
