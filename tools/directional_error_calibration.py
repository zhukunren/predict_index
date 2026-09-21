"""Small causal error model for selectively correcting the incumbent."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools.evaluate_selective_context import selective_predictions


COMMON = {
    "min_train_rows": 120, "refit_interval": 5, "C": 1.0,
    "correction_probability": 0.60, "side_window": 120,
    "side_prior_rows": 10, "recent_window": 20,
}
CANDIDATES = {
    "directional_error_year": {"train_window": 252},
    "directional_error_long": {"train_window": 756},
}
STACKED_CANDIDATES = {
    "stacked_error": {"train_window": 756, "balance_actual_direction": False},
    "stacked_error_balanced": {"train_window": 756, "balance_actual_direction": True},
}
GLOBAL_CANDIDATES = {
    "global_risk_error": {"train_window": 756, "min_train_rows": 252, "balance_actual_direction": False},
    "global_risk_error_balanced": {"train_window": 756, "min_train_rows": 252, "balance_actual_direction": True},
}
RULE_CANDIDATES = {
    "rule_context_error": {"train_window": 756, "min_train_rows": 252, "C": 0.1},
}


def error_features(champion):
    frame = champion.reset_index(drop=True)
    labels = frame.predicted_label.astype(int)
    returns = frame.real_pct_change.astype(float)
    resolved = returns.notna()
    correct = labels.eq(returns.gt(0)) & resolved
    recent = COMMON["recent_window"]
    features = pd.DataFrame({
        "direction": labels,
        "raw_confidence": frame.confidence,
        "calibrated_confidence": frame.calibrated_confidence,
        "previous_return": returns.shift(1),
        "recent_return": returns.shift(1).rolling(5, min_periods=1).sum(),
        "recent_volatility": returns.shift(1).rolling(recent, min_periods=2).std(),
        "recent_prediction_rate": labels.shift(1).rolling(recent, min_periods=1).mean(),
        "recent_up_rate": returns.gt(0).where(resolved).shift(1).rolling(recent, min_periods=1).mean(),
    })
    for side in (0, 1):
        matching = labels.eq(side) & resolved
        rows = matching.shift(1, fill_value=False).rolling(COMMON["side_window"], min_periods=1).sum()
        hits = (matching & correct).shift(1, fill_value=False).rolling(COMMON["side_window"], min_periods=1).sum()
        features[f"side_{side}_accuracy"] = (hits + COMMON["side_prior_rows"] / 2) / (rows + COMMON["side_prior_rows"])
    return features.replace([np.inf, -np.inf], np.nan).fillna({
        "previous_return": 0, "recent_return": 0, "recent_volatility": 0,
        "recent_prediction_rate": 0.5, "recent_up_rate": 0.5,
    })


def auxiliary_features(champion, auxiliary):
    features = pd.DataFrame(index=range(len(champion)))
    fitted = np.ones(len(champion), dtype=bool)
    for name, candidate in auxiliary.items():
        if not np.array_equal(champion.trade_date, candidate.trade_date):
            raise ValueError("Auxiliary predictions require identical signal dates.")
        if not np.allclose(champion.real_pct_change, candidate.real_pct_change, atol=0, rtol=0, equal_nan=True):
            raise ValueError("Auxiliary predictions require identical targets.")
        if "residual_probability_up" in candidate:
            probability_up = candidate.residual_probability_up.to_numpy(dtype=float)
            probability = np.where(champion.predicted_label == 1, probability_up, 1 - probability_up)
            rows = candidate.residual_training_rows
            training_date = candidate.residual_last_training_signal
        else:
            probability = candidate.error_model_correctness_probability.to_numpy(dtype=float)
            rows = candidate.error_model_training_rows
            training_date = candidate.error_model_last_training_date
        if not (np.isfinite(probability) & (probability >= 0) & (probability <= 1)).all():
            raise ValueError("Auxiliary probabilities must be finite and in [0, 1].")
        if (rows.gt(0) & training_date.ge(candidate.trade_date)).any():
            raise ValueError("Auxiliary training must precede its prediction signal.")
        fitted &= rows.to_numpy() > 0
        probability = np.clip(probability, 0.01, 0.99)
        features[f"aux_{name}"] = np.log(probability / (1 - probability))
    return features, fitted


def error_threshold_predictions(core, champion, model_predictions, threshold):
    if not 0.5 <= threshold <= 1:
        raise ValueError("Error correction threshold must be between 0.5 and 1.")
    if not np.array_equal(champion.trade_date, model_predictions.trade_date):
        raise ValueError("Error probabilities require identical signal dates.")
    if not np.allclose(champion.real_pct_change, model_predictions.real_pct_change, atol=0, rtol=0, equal_nan=True):
        raise ValueError("Error probabilities require identical outcomes.")
    probability = model_predictions.error_model_correctness_probability.to_numpy(dtype=float)
    if not (np.isfinite(probability) & (probability >= 0) & (probability <= 1)).all():
        raise ValueError("Error probabilities must be finite and in [0, 1].")
    proposal = champion.copy()
    labels = champion.predicted_label.to_numpy(dtype=int)
    proposal["predicted_label"] = np.where(probability < 0.5, 1 - labels, labels)
    proposal["residual_probability_up"] = np.where(labels == 1, probability, 1 - probability)
    result = selective_predictions(core, champion, proposal, threshold)
    for column in ("error_model_correctness_probability", "error_model_training_rows", "error_model_last_training_date"):
        result[column] = model_predictions[column].to_numpy()
    return result


def corrected_predictions(core, champion, name, *, auxiliary=None, context=None, rule_context=None):
    settings = COMMON | (CANDIDATES | STACKED_CANDIDATES | GLOBAL_CANDIDATES | RULE_CANDIDATES)[name]
    if (name in STACKED_CANDIDATES) != bool(auxiliary):
        raise ValueError("Stacked models require auxiliary forecasts; standalone models do not.")
    if (name in GLOBAL_CANDIDATES) != (context is not None):
        raise ValueError("Global risk models require aligned overseas context only.")
    if (name in RULE_CANDIDATES) != (rule_context is not None):
        raise ValueError("Rule error models require aligned vote context only.")
    frame = champion.copy().reset_index(drop=True)
    if frame.empty or frame.trade_date.duplicated().any() or not frame.trade_date.is_monotonic_increasing:
        raise ValueError("Error calibration requires unique chronological signal dates.")
    old_labels = frame.predicted_label.to_numpy(dtype=int).copy()
    if not np.isin(old_labels, (0, 1)).all():
        raise ValueError("Error calibration requires binary incumbent labels.")
    known = np.isfinite(frame.real_pct_change.to_numpy(dtype=float))
    target = (old_labels == frame.real_pct_change.gt(0).to_numpy()).astype(int)
    features = error_features(frame)
    if context is not None:
        if not np.array_equal(frame.trade_date, context.trade_date):
            raise ValueError("Overseas context requires identical signal dates.")
        source_columns = [column for column in context if column.endswith("_source_date")]
        if not source_columns or not context[source_columns].lt(context.trade_date, axis=0).all().all():
            raise ValueError("Overseas sessions must precede each signal date.")
        additional = context.drop(columns=["trade_date", *source_columns]).reset_index(drop=True)
        features = pd.concat([features, additional], axis=1)
    if rule_context is not None:
        if not np.array_equal(frame.trade_date, rule_context.trade_date):
            raise ValueError("Rule context requires identical signal dates.")
        if not rule_context.rule_last_calibration_date.lt(rule_context.trade_date).all():
            raise ValueError("Rule calibration must precede each signal date.")
        additional = rule_context.drop(columns=["trade_date", "rule_last_calibration_date"]).reset_index(drop=True)
        # Allow a concentrated up vote and down vote to have different risks.
        directional = additional.mul(2 * frame.predicted_label - 1, axis=0).add_suffix("_by_direction")
        features = pd.concat([features, additional, directional], axis=1)
    fitted = np.ones(len(frame), dtype=bool)
    if auxiliary:
        additional, fitted = auxiliary_features(frame, auxiliary)
        features = pd.concat([features, additional], axis=1)
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Error calibration requires finite causal features.")
    correctness = np.ones(len(frame), dtype=float)
    training_rows = np.zeros(len(frame), dtype=int)
    last_training_date = np.zeros(len(frame), dtype=int)
    model = None
    last_rows, last_date = 0, 0
    for index in range(len(frame)):
        history = np.arange(max(0, index - settings["train_window"]), index)
        history = history[known[history] & fitted[history]]
        if not fitted[index] or len(history) < settings["min_train_rows"] or len(np.unique(target[history])) < 2:
            continue
        # The origin is the full incumbent stream, never an output-window slice.
        if model is None or index % settings["refit_interval"] == 0:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=settings["C"], max_iter=500))
            fit_options = {}
            if settings.get("balance_actual_direction", False):
                actual = frame.real_pct_change.gt(0).to_numpy(dtype=int)[history]
                counts = np.bincount(actual, minlength=2)
                weights = len(history) / (2 * np.maximum(counts[actual], 1))
                fit_options = {"standardscaler__sample_weight": weights, "logisticregression__sample_weight": weights}
            model.fit(values[history], target[history], **fit_options)
            last_rows, last_date = len(history), int(frame.trade_date.iloc[history[-1]])
        correctness[index] = float(model.predict_proba(values[index:index + 1])[0, 1])
        training_rows[index], last_training_date[index] = last_rows, last_date
    proposed = frame.copy()
    probability_up = np.where(old_labels == 1, correctness, 1 - correctness)
    proposed["predicted_label"] = np.where(correctness < 0.5, 1 - old_labels, old_labels)
    proposed["residual_probability_up"] = probability_up
    result = selective_predictions(core, frame, proposed, settings["correction_probability"])
    result["error_model_correctness_probability"] = correctness
    result["error_model_training_rows"] = training_rows
    result["error_model_last_training_date"] = last_training_date
    return result
