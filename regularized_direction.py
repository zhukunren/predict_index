"""Fixed, regularized tree classifier with causal rolling refits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


def rolling_probabilities(
    features: pd.DataFrame,
    close: pd.Series,
    *,
    train_window: int = 756,
    min_train_rows: int = 252,
    refit_interval: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict every row from labels realized by that row's close.

    Refit dates are anchored to the complete input history, not the requested
    output range, so a latest-day prediction matches historical replay.
    """
    if min_train_rows < 2 or train_window < min_train_rows or refit_interval < 1:
        raise ValueError("Invalid rolling model window or refit interval.")
    values = features.to_numpy(dtype=np.float64)
    returns = close.shift(-1) / close - 1.0
    labels = returns.gt(0).to_numpy(dtype=np.int8)
    valid = np.isfinite(values).all(axis=1) & returns.notna().to_numpy()
    probabilities = np.full(len(features), np.nan)
    training_rows = np.zeros(len(features), dtype=int)
    model = None
    last_training_rows = 0
    for index in range(len(features)):
        history = np.arange(max(0, index - train_window), index)
        history = history[valid[history]]
        if not np.isfinite(values[index]).all():
            continue
        if len(history) < min_train_rows or len(np.unique(labels[history])) < 2:
            probabilities[index] = (labels[history].sum() + 1.0) / (len(history) + 2.0)
            training_rows[index] = len(history)
            continue
        if model is None or index % refit_interval == 0:
            model = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=30,
                max_features=0.7,
                class_weight=None,
                n_jobs=1,
                random_state=42,
            )
            model.fit(values[history], labels[history])
            last_training_rows = len(history)
        probabilities[index] = model.predict_proba(values[index:index + 1])[0, 1]
        training_rows[index] = last_training_rows
    return probabilities, training_rows
