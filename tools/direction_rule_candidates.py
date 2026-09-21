"""Predeclared rule-selection candidates; production keeps its own selector."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


CANDIDATES = {
    "accuracy_unique": {"score": "accuracy", "one_per_feature": False},
    "balanced_unique": {"score": "balanced_accuracy", "one_per_feature": False},
    "balanced_features": {"score": "balanced_accuracy", "one_per_feature": True},
}


def selector(core: Any, candidate: str) -> Callable:
    settings = CANDIDATES[candidate]
    fallback = core._nested_volatility_rule_signal

    def signal(*, base, cache, idx, calibration_window, min_calibration_rows, top_k):
        history = np.arange(max(0, idx - calibration_window), idx)
        history = history[cache.valid_mask[history]]
        if len(history) < min_calibration_rows:
            return fallback(base=base, cache=cache, idx=idx, calibration_window=calibration_window,
                            min_calibration_rows=min_calibration_rows, top_k=top_k)
        labels = cache.labels[history]
        predictions = cache.predictions[:, history]
        valid = predictions >= 0
        hits = (predictions == labels) & valid
        counts = valid.sum(axis=1)
        accuracy = hits.sum(axis=1) / np.maximum(counts, 1)
        scores = accuracy
        if settings["score"] == "balanced_accuracy":
            class_counts = [((labels == label) & valid).sum(axis=1) for label in (0, 1)]
            class_hits = [((labels == label) & hits).sum(axis=1) for label in (0, 1)]
            balanced = sum(hit / np.maximum(count, 1) for hit, count in zip(class_hits, class_counts)) / 2
            scores = np.where((class_counts[0] > 0) & (class_counts[1] > 0), balanced, 0.5)
        reverse = scores < 0.5
        scores = np.where(reverse, 1 - scores, scores)
        eligible = np.flatnonzero((counts >= min_calibration_rows) & (cache.predictions[:, idx] >= 0))
        ranked = sorted(eligible, key=lambda rule: (-scores[rule], int(rule)))
        selected = []
        signatures = set()
        features = set()
        for rule in ranked:
            values = cache.predictions[rule, np.append(history, idx)].copy()
            if reverse[rule]:
                values = np.where(values >= 0, 1 - values, -1)
            signature = values.astype(np.int8).tobytes()
            name = cache.rule_names[rule]
            feature = name.split(">", 1)[0].split("<=", 1)[0]
            if signature in signatures or (settings["one_per_feature"] and feature in features):
                continue
            signatures.add(signature)
            features.add(feature)
            selected.append((int(rule), values))
            if len(selected) == max(1, top_k):
                break
        if not selected:
            return fallback(base=base, cache=cache, idx=idx, calibration_window=calibration_window,
                            min_calibration_rows=min_calibration_rows, top_k=top_k)
        weights = np.array([max(float(scores[rule]) - 0.5, 0.001) for rule, _ in selected])
        matrix = np.vstack([values for _, values in selected])
        prediction = int(np.dot(weights, matrix[:, -1]) / weights.sum() >= 0.5)
        complete = (matrix[:, :-1] >= 0).all(axis=0)
        historical_vote = np.dot(weights, matrix[:, :-1][:, complete]) / weights.sum() >= 0.5
        calibration_accuracy = float((historical_vote == labels[complete]).mean()) if complete.any() else 0.5
        return core._rule_signal_from_label(
            predicted_label=prediction, base=base, idx=idx,
            rule_names=[("NOT(" + cache.rule_names[rule] + ")" if reverse[rule] else cache.rule_names[rule])
                        + f":score={scores[rule]:.6f}" for rule, _ in selected],
            calibration_accuracy=calibration_accuracy, calibration_rows=int(complete.sum()),
            diagnostics={"rule_mode": candidate, "selected_ranked_rule_count": len(selected),
                         "selected_expert_count": len(selected), "selected_rule_count": len(selected),
                         "top_rule_score": float(scores[selected[0][0]])},
        )

    return signal
