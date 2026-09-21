"""Estimate incumbent errors separately for its up and down predictions."""

from __future__ import annotations

import numpy as np

from tools.downside_specialist import downside_probabilities
from tools.evaluate_selective_context import selective_predictions


def directional_probabilities(champion, features, name, *, available=None, training_policy="uniform"):
    down = downside_probabilities(champion, features, name, available=available, training_policy=training_policy)
    mirrored = champion.copy()
    mirrored["predicted_label"] = 1 - champion.predicted_label
    actual = champion.real_pct_change.to_numpy(dtype=float)
    # The shared learner uses return <= 0 as its event. Preserve zero as a
    # correct original-down outcome, and unresolved outcomes as unknown.
    mirrored["real_pct_change"] = np.where(np.isfinite(actual), np.where(actual > 0, -1., 1.), np.nan)
    up = downside_probabilities(mirrored, features, name, available=available, training_policy=training_policy)
    is_up = champion.predicted_label.eq(1).to_numpy()
    result = down.loc[:, ["trade_date"]].copy()
    for suffix in ("probability", "training_rows", "last_training_date"):
        result[f"error_{suffix}"] = np.where(is_up, down[f"downside_{suffix}"], up[f"downside_{suffix}"])
    result["incumbent_label"] = champion.predicted_label.to_numpy()
    return result


def directional_predictions(core, champion, probabilities, threshold):
    if not 0.5 < threshold <= 1:
        raise ValueError("Correction threshold must be in (0.5, 1].")
    if (not np.array_equal(champion.trade_date, probabilities.trade_date)
            or not np.array_equal(champion.predicted_label, probabilities.incumbent_label)):
        raise ValueError("Error probabilities require identical dates and incumbent labels.")
    labels = champion.predicted_label.to_numpy(dtype=float)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Error correction requires binary incumbent labels.")
    probability = probabilities.error_probability.to_numpy(dtype=float)
    if not (np.isfinite(probability) & (probability >= 0) & (probability <= 1)).all():
        raise ValueError("Error probabilities must be finite and in [0, 1].")
    fitted = probabilities.error_training_rows.gt(0).to_numpy()
    if (fitted & probabilities.error_last_training_date.ge(probabilities.trade_date)).any():
        raise ValueError("Error model training must precede its signal date.")
    selected = fitted & (probability >= threshold)
    proposal = champion.copy()
    proposal["predicted_label"] = np.where(selected, 1 - labels, labels).astype(int)
    proposal["residual_probability_up"] = np.where(selected, np.where(labels == 1, 1 - probability, probability), labels)
    result = selective_predictions(core, champion, proposal, threshold)
    for column in probabilities.columns.difference(["trade_date"]):
        result[column] = probabilities[column].to_numpy()
    return result
