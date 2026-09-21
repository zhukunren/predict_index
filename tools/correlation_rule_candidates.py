"""Rank causal binary rules by directional correlation instead of accuracy."""

from __future__ import annotations

import numpy as np


CANDIDATES = {"mcc_ranked_rules": {"score": "matthews_correlation", "retain_rank_weights": True}}
PARAMETERS = {"score_range": [-1, 1], "constant_prediction_score": 0,
              "orientation": "reverse only negative historical correlation"}


def correlations(predictions, labels):
    valid = predictions >= 0
    up, down = labels == 1, labels == 0
    tp = ((predictions == 1) & up & valid).sum(axis=1).astype(float)
    tn = ((predictions == 0) & down & valid).sum(axis=1).astype(float)
    fp = ((predictions == 1) & down & valid).sum(axis=1).astype(float)
    fn = ((predictions == 0) & up & valid).sum(axis=1).astype(float)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    result = np.divide(tp * tn - fp * fn, denominator, out=np.zeros(len(predictions)), where=denominator > 0)
    return result, valid.sum(axis=1)


def selector(core, name):
    if name not in CANDIDATES:
        raise ValueError(f"Unknown correlation rule candidate: {name}.")
    original = core._nested_volatility_rule_signal

    def signal(*, base, cache, idx, calibration_window, min_calibration_rows, top_k):
        history = np.arange(max(0, idx - calibration_window), idx)
        history = history[cache.valid_mask[history]]
        fallback = dict(base=base, cache=cache, idx=idx, calibration_window=calibration_window,
                        min_calibration_rows=min_calibration_rows, top_k=top_k)
        if len(history) < min_calibration_rows:
            return original(**fallback)
        scores, counts = correlations(cache.predictions[:, history], cache.labels[history])
        eligible = np.flatnonzero((counts >= min_calibration_rows) & (cache.predictions[:, idx] >= 0) & (np.abs(scores) > 0))
        ranked = sorted(eligible, key=lambda rule: (-abs(scores[rule]), int(rule)))[:max(1, top_k)]
        if not ranked:
            return original(**fallback)
        indices = np.append(history, idx)
        matrix = cache.predictions[ranked][:, indices].copy()
        reversed_rules = scores[ranked] < 0
        matrix = np.where(reversed_rules[:, None] & (matrix >= 0), 1 - matrix, matrix)
        # Identical paths keep their accumulated rank weight, as in production.
        weights = np.abs(scores[ranked])
        label = int(weights @ matrix[:, -1] / weights.sum() >= 0.5)
        complete = (matrix[:, :-1] >= 0).all(axis=0)
        if not complete.any():
            return original(**fallback)
        historical = weights @ matrix[:, :-1][:, complete] / weights.sum() >= 0.5
        accuracy = float((historical == cache.labels[history[complete]]).mean())
        return core._rule_signal_from_label(
            predicted_label=label, base=base, idx=idx,
            rule_names=[("NOT(" + cache.rule_names[rule] + ")" if reverse else cache.rule_names[rule])
                        + f":mcc={abs(scores[rule]):.6f}" for rule, reverse in zip(ranked, reversed_rules)],
            calibration_accuracy=accuracy, calibration_rows=int(complete.sum()),
            diagnostics={"rule_mode": name, "selected_ranked_rule_count": len(ranked),
                         "selected_expert_count": len(np.unique(matrix, axis=0)),
                         "selected_rule_count": len(np.unique(matrix, axis=0)),
                         "top_rule_correlation": float(abs(scores[ranked[0]]))},
        )

    return signal
