"""One predeclared rule-state duration hypothesis; no production integration."""

from __future__ import annotations

import numpy as np


PARAMETERS = {
    "duration_upper_bounds": [1, 5, 20],
    "state_prior_positive": 1.0,
    "state_prior_negative": 1.0,
    "duration_prior_mass": 10.0,
    "decision_threshold": 0.5,
}
CANDIDATE = "duration_conditioned_probability"


def state_durations(predictions):
    values = np.asarray(predictions)
    if values.ndim != 2 or not np.isin(values, [-1, 0, 1]).all():
        raise ValueError("Rule states must be a matrix containing -1, 0, or 1.")
    durations = np.zeros(values.shape, dtype=np.int32)
    for idx in range(values.shape[1]):
        valid = values[:, idx] >= 0
        if idx == 0:
            durations[:, idx] = valid.astype(np.int32)
        else:
            repeated = valid & (values[:, idx] == values[:, idx - 1])
            durations[:, idx] = np.where(valid, np.where(repeated, durations[:, idx - 1] + 1, 1), 0)
    return durations


def conditional_probability(predictions, durations, labels, history, idx):
    state = predictions[idx]
    if state < 0 or durations[idx] < 1:
        raise ValueError("Current rule state and duration must be available.")
    if np.any(np.asarray(history) >= idx):
        raise ValueError("Conditional estimates cannot include the current or future result.")
    mask = predictions[history] == state
    state_rows = history[mask]
    prior_up = PARAMETERS["state_prior_positive"]
    prior_down = PARAMETERS["state_prior_negative"]
    state_probability = float((labels[state_rows].sum() + prior_up) / (len(state_rows) + prior_up + prior_down))
    bins = np.searchsorted(PARAMETERS["duration_upper_bounds"], durations, side="left")
    duration_rows = state_rows[bins[state_rows] == bins[idx]]
    mass = PARAMETERS["duration_prior_mass"]
    probability = float((labels[duration_rows].sum() + mass * state_probability) / (len(duration_rows) + mass))
    return probability, len(duration_rows), state_probability


def selector(core, candidate=CANDIDATE):
    if candidate != CANDIDATE:
        raise ValueError("Unknown duration candidate.")
    original = core._nested_volatility_rule_signal
    active_cache = None
    duration_matrix = None

    def signal(*, base, cache, idx, calibration_window, min_calibration_rows, top_k):
        nonlocal active_cache, duration_matrix
        if cache is not active_cache:
            duration_matrix = state_durations(cache.predictions)
            active_cache = cache
        history = np.arange(max(0, idx - calibration_window), idx)
        history = history[cache.valid_mask[history]]
        fallback = dict(base=base, cache=cache, idx=idx, calibration_window=calibration_window,
                        min_calibration_rows=min_calibration_rows, top_k=top_k)
        if len(history) < min_calibration_rows:
            return original(**fallback)
        matrix = cache.predictions[:, history]
        valid = matrix >= 0
        counts = valid.sum(axis=1)
        scores = ((matrix == cache.labels[history]) & valid).sum(axis=1) / np.maximum(counts, 1)
        scores = np.maximum(scores, 1 - scores)
        eligible = np.flatnonzero((counts >= min_calibration_rows) & (cache.predictions[:, idx] >= 0))
        ranked = sorted(eligible, key=lambda rule: (-scores[rule], int(rule)))[:max(1, top_k)]
        if not ranked:
            return original(**fallback)
        estimates = [conditional_probability(cache.predictions[rule], duration_matrix[rule],
                       cache.labels, history, idx) for rule in ranked]
        weights = np.maximum(scores[ranked] - 0.5, 0.001)
        probability = float(np.average([item[0] for item in estimates], weights=weights))
        state_probability = float(np.average([item[2] for item in estimates], weights=weights))
        label = int(probability >= PARAMETERS["decision_threshold"])
        signatures = set()
        for rule in ranked:
            raw = cache.predictions[rule, np.append(history, idx)]
            # Complementary threshold paths describe one information source.
            signatures.add(min(raw.tobytes(), np.where(raw >= 0, 1 - raw, -1).astype(raw.dtype).tobytes()))
        return core._rule_signal_from_label(
            predicted_label=label, base=base, idx=idx,
            rule_names=[cache.rule_names[rule] for rule in ranked],
            calibration_accuracy=probability if label else 1 - probability,
            calibration_rows=len(history),
            diagnostics={"rule_mode": CANDIDATE, "probability_up": probability,
                         "state_probability_up": state_probability,
                         "selected_duration_mean": float(np.average(duration_matrix[ranked, idx], weights=weights)),
                         "duration_history_rows_mean": float(np.average([item[1] for item in estimates], weights=weights)),
                         "selected_ranked_rule_count": len(ranked),
                         "selected_expert_count": len(signatures), "selected_rule_count": len(signatures),
                         "top_rule_score": float(scores[ranked[0]])},
        )

    return signal
