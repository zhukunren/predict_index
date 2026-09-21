"""Rank and calibrate rules from predictions oriented before each outcome."""

from __future__ import annotations

import numpy as np


CANDIDATES = {
    "prequential_accuracy_rules": {"score": "accuracy"},
    "prequential_balanced_rules": {"score": "balanced_accuracy"},
}
PARAMETERS = {
    "orientation": "prior rolling raw-rule accuracy, never current outcome",
    "ranking": "prior rolling accuracy of predictions actually oriented on each historical date",
    "calibration": "prior emitted ensemble predictions, not today's re-ranked historical vote",
    "minimum_class_rows": 10,
    "positive_edge_only": True,
    "rank_weights": "retain all top-k rank weights including equivalent rules",
}


def prior_sum(values, window):
    values = np.asarray(values)
    if values.ndim != 2 or window <= 0:
        raise ValueError("Prequential sums require a matrix and positive window.")
    cumulative = np.pad(np.cumsum(values, axis=1), ((0, 0), (1, 0)))
    ends = np.arange(values.shape[1])
    return cumulative[:, ends] - cumulative[:, np.maximum(0, ends - window)]


def prequential_paths(cache, window, minimum_rows):
    raw = np.asarray(cache.predictions)
    labels = np.asarray(cache.labels)
    resolved = np.asarray(cache.valid_mask)
    if (raw.ndim != 2 or raw.shape[1] != len(labels) or resolved.shape != labels.shape
            or not np.isin(raw, (-1, 0, 1)).all() or not np.isin(labels[resolved], (0, 1)).all()):
        raise ValueError("Prequential rules require aligned binary outcomes and predictions.")
    valid = (raw >= 0) & resolved
    counts = prior_sum(valid, window)
    hits = prior_sum((raw == labels) & valid, window)
    reverse = hits * 2 < counts
    available = (raw >= 0) & (counts >= minimum_rows)
    return np.where(available, np.where(reverse, 1 - raw, raw), -1).astype(np.int8)


def ensemble_history(cache, name, window, minimum_rows, top_k):
    paths = prequential_paths(cache, window, minimum_rows)
    labels = np.asarray(cache.labels)
    valid = (paths >= 0) & cache.valid_mask
    counts = prior_sum(valid, window)
    hits = prior_sum((paths == labels) & valid, window)
    scores = np.divide(hits, counts, out=np.zeros_like(hits, dtype=float), where=counts > 0)
    eligible = (counts >= minimum_rows) & (paths >= 0)
    if CANDIDATES[name]["score"] == "balanced_accuracy":
        recalls = []
        for side in (0, 1):
            side_rows = valid & (labels == side)
            side_counts = prior_sum(side_rows, window)
            side_hits = prior_sum((paths == side) & side_rows, window)
            recalls.append(np.divide(side_hits, side_counts, out=np.zeros_like(scores), where=side_counts > 0))
            eligible &= side_counts >= PARAMETERS["minimum_class_rows"]
        scores = sum(recalls) / 2
    eligible &= scores > 0.5
    ranked = np.argsort(-np.where(eligible, scores, -np.inf), axis=0, kind="stable")[:max(1, top_k)]
    selected_scores = np.take_along_axis(scores, ranked, axis=0)
    weights = np.where(np.take_along_axis(eligible, ranked, axis=0), selected_scores - 0.5, 0)
    votes = np.take_along_axis(paths, ranked, axis=0)
    weight_sum = weights.sum(axis=0)
    prediction = np.where(weight_sum > 0, ((weights * votes).sum(axis=0) * 2 >= weight_sum).astype(int), -1)
    return {"paths": paths, "scores": scores, "counts": counts, "ranked": ranked,
            "weights": weights, "prediction": prediction}


def selector(core, name):
    if name not in CANDIDATES:
        raise ValueError(f"Unknown prequential rule candidate: {name}.")
    original = core._nested_volatility_rule_signal
    cached_source, cached_parameters, computed = None, None, None

    def signal(*, base, cache, idx, calibration_window, min_calibration_rows, top_k):
        nonlocal cached_source, cached_parameters, computed
        parameters = (calibration_window, min_calibration_rows, top_k)
        if cached_source is not cache or cached_parameters != parameters:
            computed = ensemble_history(cache, name, *parameters)
            cached_source, cached_parameters = cache, parameters
        fallback = dict(base=base, cache=cache, idx=idx, calibration_window=calibration_window,
                        min_calibration_rows=min_calibration_rows, top_k=top_k)
        if computed["prediction"][idx] < 0:
            return original(**fallback)
        history = np.arange(max(0, idx - calibration_window), idx)
        known = cache.valid_mask[history] & (computed["prediction"][history] >= 0)
        history = history[known]
        selected = computed["ranked"][:, idx][computed["weights"][:, idx] > 0]
        signature = computed["paths"][selected][:, np.append(history, idx)]
        names = [cache.rule_names[rule] + f":prior_emitted_score={computed['scores'][rule, idx]:.6f}" for rule in selected]
        return core._rule_signal_from_label(
            predicted_label=int(computed["prediction"][idx]), base=base, idx=idx,
            rule_names=names,
            calibration_accuracy=(float((computed["prediction"][history] == cache.labels[history]).mean())
                                  if len(history) >= min_calibration_rows else 0.5),
            calibration_rows=len(history),
            diagnostics={"rule_mode": name, "selected_ranked_rule_count": len(selected),
                         "selected_expert_count": len(np.unique(signature, axis=0)),
                         "selected_rule_count": len(np.unique(signature, axis=0)),
                         "prequential_last_calibration_index": int(history[-1]) if len(history) else -1,
                         "top_rule_score": float(computed["scores"][selected[0], idx])},
        )

    return signal
