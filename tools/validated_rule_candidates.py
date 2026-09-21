"""Separate rule ranking from its subsequent directional skill check."""

from __future__ import annotations

import numpy as np


CANDIDATES = {
    "validated_accuracy_rules": {"rank_metric": "accuracy"},
    "validated_balanced_rules": {"rank_metric": "balanced_accuracy"},
}
PARAMETERS = {"validation_window": 60, "minimum_class_rows": 10}


def rule_plan(cache, idx, calibration_window, min_calibration_rows, top_k, name):
    end = max(0, idx - PARAMETERS["validation_window"])
    ranking = np.arange(max(0, end - calibration_window), end)
    ranking = ranking[cache.valid_mask[ranking]]
    validation = np.arange(end, idx)
    validation = validation[cache.valid_mask[validation]]
    if len(ranking) < min_calibration_rows or len(validation) < min_calibration_rows:
        return None
    rank_labels = cache.labels[ranking]
    matrix = cache.predictions[:, ranking]
    valid = matrix >= 0
    counts = valid.sum(axis=1)
    hits = (matrix == rank_labels) & valid
    scores = hits.sum(axis=1) / np.maximum(counts, 1)
    if CANDIDATES[name]["rank_metric"] == "balanced_accuracy":
        recalls = []
        for side in (0, 1):
            side_rows = (rank_labels == side) & valid
            side_counts = side_rows.sum(axis=1)
            recalls.append((hits & side_rows).sum(axis=1) / np.maximum(side_counts, 1))
            counts = np.where(side_counts >= PARAMETERS["minimum_class_rows"], counts, 0)
        scores = sum(recalls) / 2
    reverse = scores < 0.5
    scores = np.where(reverse, 1 - scores, scores)
    ranked = sorted(np.flatnonzero(counts >= min_calibration_rows), key=lambda rule: (-scores[rule], int(rule)))
    selected, signatures = [], set()
    for rule in ranked:
        prediction = cache.predictions[rule].copy()
        if reverse[rule]:
            prediction = np.where(prediction >= 0, 1 - prediction, -1)
        # Select unique experts using only the older ranking period.
        signature = prediction[ranking].astype(np.int8).tobytes()
        if signature in signatures:
            continue
        signatures.add(signature)
        selected.append((rule, prediction, max(float(scores[rule]) - 0.5, 0.001)))
        if len(selected) >= max(1, top_k):
            break
    accepted = []
    for rule, prediction, weight in selected:
        available = prediction[validation] >= 0
        outcomes = cache.labels[validation][available]
        votes = prediction[validation][available]
        if prediction[idx] < 0 or any((outcomes == side).sum() < PARAMETERS["minimum_class_rows"] for side in (0, 1)):
            continue
        balanced_accuracy = sum(float((votes[outcomes == side] == side).mean()) for side in (0, 1)) / 2
        if balanced_accuracy > 0.5:
            accepted.append((int(rule), prediction, weight, float(balanced_accuracy)))
    if not accepted:
        return None
    weights = np.array([entry[2] for entry in accepted])
    matrix = np.vstack([entry[1] for entry in accepted])
    complete = (matrix[:, validation] >= 0).all(axis=0)
    historical = (weights @ matrix[:, validation[complete]] / weights.sum() >= 0.5).astype(int)
    if not complete.any():
        return None
    return {
        "predicted_label": int(weights @ matrix[:, idx] / weights.sum() >= 0.5),
        "calibration_accuracy": float((historical == cache.labels[validation[complete]]).mean()),
        "calibration_rows": int(complete.sum()),
        "ranked_rules": [int(entry[0]) for entry in selected],
        "accepted_rules": [entry[0] for entry in accepted],
        "reverse": [bool(reverse[entry[0]]) for entry in accepted],
        "validation_scores": [entry[3] for entry in accepted],
        "ranking_end_index": int(ranking[-1]), "validation_start_index": int(validation[0]),
        "validation_end_index": int(validation[-1]),
    }


def selector(core, name):
    if name not in CANDIDATES:
        raise ValueError(f"Unknown validated rule candidate: {name}.")
    original = core._nested_volatility_rule_signal

    def signal(*, base, cache, idx, calibration_window, min_calibration_rows, top_k):
        plan = rule_plan(cache, idx, calibration_window, min_calibration_rows, top_k, name)
        if plan is None:
            return original(base=base, cache=cache, idx=idx, calibration_window=calibration_window,
                            min_calibration_rows=min_calibration_rows, top_k=top_k)
        return core._rule_signal_from_label(
            predicted_label=plan["predicted_label"], base=base, idx=idx,
            rule_names=[("NOT(" + cache.rule_names[rule] + ")" if reverse else cache.rule_names[rule])
                        for rule, reverse in zip(plan["accepted_rules"], plan["reverse"])],
            calibration_accuracy=plan["calibration_accuracy"], calibration_rows=plan["calibration_rows"],
            diagnostics={"rule_mode": name, "selected_ranked_rule_count": len(plan["ranked_rules"]),
                         "selected_expert_count": len(plan["accepted_rules"]),
                         "selected_rule_count": len(plan["accepted_rules"]),
                         "rule_validation_balanced_accuracy": min(plan["validation_scores"]),
                         "rule_ranking_end_index": plan["ranking_end_index"],
                         "rule_validation_start_index": plan["validation_start_index"],
                         "rule_validation_end_index": plan["validation_end_index"]},
        )

    return signal
