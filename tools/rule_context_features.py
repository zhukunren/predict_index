"""Causal context of the incumbent's ranked vote, without changing its rules."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


RULE_PATTERN = re.compile(r"(?P<feature>.+?)(?:>|<=)rolling_q[0-9]+\.[0-9]+")
SLOW_COLUMNS = {"volatility_20", "bollinger_width", "macd_hist"}
RECENT_WINDOW = 20


def ranked_context(cache, idx, *, calibration_window, min_calibration_rows, top_k):
    history = np.arange(max(0, idx - calibration_window), idx)
    history = history[cache.valid_mask[history]]
    if len(history) < min_calibration_rows:
        raise ValueError("Rule context requires a complete calibration history.")
    labels = cache.labels[history]
    matrix = cache.predictions[:, history]
    valid = matrix >= 0
    counts = valid.sum(axis=1)
    accuracy = ((matrix == labels) & valid).sum(axis=1) / np.maximum(counts, 1)
    reverse = accuracy < 0.5
    scores = np.where(reverse, 1 - accuracy, accuracy)
    eligible = np.flatnonzero((counts >= min_calibration_rows) & (cache.predictions[:, idx] >= 0))
    ranked = sorted(eligible, key=lambda rule: (-scores[rule], int(rule)))[:max(1, top_k)]
    if not ranked:
        raise ValueError("Rule context requires eligible ranked rules.")
    predictions = cache.predictions[ranked][:, np.append(history, idx)].copy()
    predictions = np.where(reverse[ranked, None] & (predictions >= 0), 1 - predictions, predictions)
    weights = np.maximum(scores[ranked] - 0.5, 0.001)
    grouped = {}
    for rule, prediction, weight in zip(ranked, predictions, weights):
        signature = prediction.astype(np.int8).tobytes()
        if signature not in grouped:
            grouped[signature] = [prediction, 0.0]
        grouped[signature][1] += float(weight)
    unique = np.vstack([entry[0] for entry in grouped.values()])
    unique_weights = np.array([entry[1] for entry in grouped.values()])
    vote = float(np.dot(unique_weights, unique[:, -1]) / unique_weights.sum())
    predicted_label = int(vote >= 0.5)
    complete = (unique[:, :-1] >= 0).all(axis=0)
    if not complete.any():
        raise ValueError("Rule context requires overlapping calibration votes.")
    historical_vote = np.dot(unique_weights, unique[:, :-1][:, complete]) / unique_weights.sum() >= 0.5
    historical_accuracy = float((historical_vote == labels[complete]).mean())
    feature_names = []
    for rule in ranked:
        parsed = RULE_PATTERN.fullmatch(cache.rule_names[rule])
        if parsed is None:
            raise ValueError("Unsupported rule name in the incumbent cache.")
        feature_names.append(parsed["feature"])
    slow = np.array([name in SLOW_COLUMNS or "_vol20" in name or "_ret5" in name for name in feature_names])
    close_position = np.array([name == "close_position" for name in feature_names])
    external = np.array([name.startswith("ext_") for name in feature_names])
    recent = history >= idx - RECENT_WINDOW
    recent_values = predictions[:, :-1][:, recent]
    valid_recent = recent_values >= 0
    persistence = ((recent_values == predictions[:, -1, None]) & valid_recent).sum(axis=1) / np.maximum(valid_recent.sum(axis=1), 1)
    return {
        "base_predicted_label": predicted_label,
        "last_calibration_position": int(history[-1]),
        "rule_margin": abs(2 * vote - 1),
        "rule_vote_up": vote,
        "rule_effective_experts": float(unique_weights.sum() ** 2 / np.square(unique_weights).sum()),
        "rule_slow_weight": float(np.dot(weights, slow) / weights.sum()),
        "rule_close_position_weight": float(np.dot(weights, close_position) / weights.sum()),
        "rule_external_weight": float(np.dot(weights, external) / weights.sum()),
        "rule_persistence_20": float(np.dot(weights, persistence) / weights.sum()),
        "rule_majority_edge": historical_accuracy - max(float(labels.mean()), 1 - float(labels.mean())),
    }


def rule_context_features(core, data, champion, diagnostics, config, options):
    if not np.array_equal(champion.trade_date, diagnostics.trade_date):
        raise ValueError("Rule diagnostics require identical incumbent dates.")
    if not np.array_equal(champion.predicted_label, diagnostics.predicted_label):
        raise ValueError("Rule diagnostics disagree with incumbent directions.")
    base = core._normalize_market_frame(data, config)
    features = core._clean_feature_frame(core._build_features(base, config))
    cache = core._build_nested_rule_cache(
        base=base, features=features,
        threshold_window=options["nested_rule_threshold_window"],
        min_threshold_rows=options["nested_rule_min_threshold_rows"],
    )
    dates = base.date.dt.strftime("%Y%m%d").astype(int)
    positions = pd.Index(dates).get_indexer(champion.trade_date)
    if (positions < 0).any() or (np.diff(positions) <= 0).any():
        raise ValueError("Incumbent dates must align chronologically with market data.")
    rows = []
    streak = 0
    previous_label = None
    for offset, position in enumerate(positions):
        context = ranked_context(
            cache, position, calibration_window=options["nested_rule_calibration_window"],
            min_calibration_rows=options["nested_rule_min_calibration_rows"], top_k=options["rule_top_k"],
        )
        if context.pop("base_predicted_label") != diagnostics.base_predicted_label.iloc[offset]:
            raise ValueError("Rebuilt ranked vote disagrees with frozen baseline diagnostics.")
        context["rule_last_calibration_date"] = int(dates.iloc[context.pop("last_calibration_position")])
        label = int(champion.predicted_label.iloc[offset])
        streak = streak + 1 if label == previous_label else 1
        previous_label = label
        context["rule_aligned_margin"] = (2 * context.pop("rule_vote_up") - 1) * (2 * label - 1)
        context["rule_direction_streak"] = float(np.log1p(streak))
        for name in ("veto_applied", "recent_failure_guard_applied"):
            context[f"rule_{name}"] = int(diagnostics[name].iloc[offset])
        context["trade_date"] = int(champion.trade_date.iloc[offset])
        rows.append(context)
    result = pd.DataFrame(rows)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Rule context must be finite.")
    return result
