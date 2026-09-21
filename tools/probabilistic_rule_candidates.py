"""Probabilistic aggregation of the champion's unchanged ranked rules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


CANDIDATES = {
    "conditional_probability": {"aggregation": "conditional", "half_life": None, "prior_count": 2.0},
    "recent_conditional_probability": {"aggregation": "conditional", "half_life": 60, "prior_count": 2.0},
    "logistic_rule_probability": {"aggregation": "logistic", "C": 0.1, "max_iter": 300},
}


def selector(core: Any, candidate: str) -> Callable:
    settings = CANDIDATES[candidate]
    original = core._nested_volatility_rule_signal

    def signal(*, base, cache, idx, calibration_window, min_calibration_rows, top_k):
        history = np.arange(max(0, idx - calibration_window), idx)
        history = history[cache.valid_mask[history]]
        fallback_args = dict(base=base, cache=cache, idx=idx, calibration_window=calibration_window,
                             min_calibration_rows=min_calibration_rows, top_k=top_k)
        if len(history) < min_calibration_rows:
            return original(**fallback_args)
        labels = cache.labels[history]
        matrix = cache.predictions[:, history]
        valid = matrix >= 0
        counts = valid.sum(axis=1)
        accuracy = ((matrix == labels) & valid).sum(axis=1) / np.maximum(counts, 1)
        reversed_rules = accuracy < 0.5
        accuracy = np.where(reversed_rules, 1 - accuracy, accuracy)
        eligible = np.flatnonzero((counts >= min_calibration_rows) & (cache.predictions[:, idx] >= 0))
        ranked = sorted(eligible, key=lambda rule: (-accuracy[rule], int(rule)))[:max(1, top_k)]
        if not ranked:
            return original(**fallback_args)
        matrix = cache.predictions[ranked][:, np.append(history, idx)].copy()
        matrix = np.where(reversed_rules[ranked, None] & (matrix >= 0), 1 - matrix, matrix)
        weights = np.maximum(accuracy[ranked] - 0.5, 0.001)
        if settings["aggregation"] == "conditional":
            ages = idx - 1 - history
            sample_weights = np.ones(len(history)) if settings["half_life"] is None else 0.5 ** (ages / settings["half_life"])
            probabilities = []
            for prediction in matrix:
                mask = prediction[:-1] == prediction[-1]
                mass = sample_weights[mask].sum()
                positive_mass = np.dot(sample_weights[mask], labels[mask])
                probabilities.append((positive_mass + settings["prior_count"] / 2) / (mass + settings["prior_count"]))
            probability_up = float(np.dot(weights, probabilities) / weights.sum())
        else:
            # Identical columns add no new explanatory variable to regression.
            _, positions = np.unique(matrix, axis=0, return_index=True)
            matrix = matrix[np.sort(positions)]
            complete = (matrix[:, :-1] >= 0).all(axis=0)
            if complete.sum() < min_calibration_rows or len(np.unique(labels[complete])) < 2:
                return original(**fallback_args)
            model = LogisticRegression(C=settings["C"], max_iter=settings["max_iter"], solver="lbfgs")
            model.fit(matrix[:, :-1][:, complete].T, labels[complete])
            probability_up = float(model.predict_proba(matrix[:, -1].reshape(1, -1))[0, 1])
        label = int(probability_up >= 0.5)
        return core._rule_signal_from_label(
            predicted_label=label, base=base, idx=idx,
            rule_names=[cache.rule_names[rule] for rule in ranked],
            calibration_accuracy=probability_up if label else 1 - probability_up,
            calibration_rows=len(history),
            diagnostics={"rule_mode": candidate, "probability_up": probability_up,
                         "selected_ranked_rule_count": len(ranked),
                         "selected_expert_count": len(np.unique(matrix, axis=0)),
                         "selected_rule_count": len(np.unique(matrix, axis=0)),
                         "top_rule_score": float(accuracy[ranked[0]])},
        )

    return signal
