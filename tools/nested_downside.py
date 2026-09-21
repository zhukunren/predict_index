"""Choose shrinkage and optional signals using only earlier validation blocks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NestedDownsideConfig:
    train_window: int = 756
    min_train_rows: int = 120
    validation_block_rows: int = 40
    validation_blocks: int = 3
    refit_interval: int = 5
    inverse_penalties: tuple[float, ...] = (0.001, 0.01, 0.1)
    max_iter: int = 1000
    require_paired_nonregression: bool = False
    correction_threshold: float = 0.60
    threshold_candidates: tuple[float, ...] = ()


DEFAULT_CONFIG = NestedDownsideConfig()


def _fit(values, target, inverse_penalty, max_iter):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=inverse_penalty, max_iter=max_iter))
    model.fit(values, target)
    return model


def validation_blocks(history, config):
    history = np.asarray(history)
    if history.ndim != 1 or len(history) < config.min_train_rows + config.validation_blocks * config.validation_block_rows:
        return []
    if not np.issubdtype(history.dtype, np.integer) or (np.diff(history) <= 0).any():
        raise ValueError("Nested history must contain increasing unique sample positions.")
    start = len(history) - config.validation_blocks * config.validation_block_rows
    return [(history[:start + block * config.validation_block_rows],
             history[start + block * config.validation_block_rows:start + (block + 1) * config.validation_block_rows])
            for block in range(config.validation_blocks)]


def paired_validation(actual, incumbent, validation, probability, threshold):
    """Compare only the corrections with the incumbent on the same prior dates.

    Balanced-accuracy denominators include all resolved signals between the
    validation block endpoints, including incumbent down signals that are kept.
    """
    positions = np.arange(validation[0], validation[-1] + 1)
    known = np.isfinite(actual[positions])
    truth = actual[positions][known] > 0
    original = incumbent[positions].copy()
    proposed = original.copy()
    selected = validation[np.asarray(probability) >= threshold]
    proposed[selected - positions[0]] = 1 - incumbent[selected]
    original_hits = original[known] == truth
    proposed_hits = proposed[known] == truth
    delta = proposed_hits.astype(int) - original_hits.astype(int)
    up_rows, down_rows = int(truth.sum()), int((~truth).sum())
    accuracy_delta = float(delta.mean())
    balanced_delta = (float((delta[truth].mean() + delta[~truth].mean()) / 2)
                      if up_rows and down_rows else None)
    return {"rows": int(known.sum()), "actual_up_rows": up_rows, "actual_down_rows": down_rows,
            "corrected": int((delta > 0).sum()), "damaged": int((delta < 0).sum()),
            "up_hit_delta": int(delta[truth].sum()), "down_hit_delta": int(delta[~truth].sum()),
            "accuracy_delta": accuracy_delta, "balanced_accuracy_delta": balanced_delta,
            "nondecreasing": bool(accuracy_delta >= -1e-12 and balanced_delta is not None and balanced_delta >= -1e-12)}


def select_choice(choices, *, learn_threshold=False):
    eligible = [choice for choice in choices if choice["admissible"]]
    if not eligible:
        return None
    if learn_threshold:
        # Optimize historical direction decisions; never optimize output counts.
        return min(eligible, key=lambda choice: (
            -choice["pooled_accuracy_gain"], -choice["pooled_balanced_gain"],
            choice["changed_validation_rows"], choice["log_loss"],
            choice["features"] != "base", choice["inverse_penalty"], -choice["correction_threshold"],
        ))
    minimum_loss = min(choice["log_loss"] for choice in eligible)
    tied = [choice for choice in eligible if choice["log_loss"] <= minimum_loss + 1e-12]
    return min(tied, key=lambda choice: (choice["features"] != "base", choice["inverse_penalty"]))


def nested_downside_probabilities(champion, features, optional_columns=(), *, available=None, config=DEFAULT_CONFIG,
                                 target_mode="conditional_downside"):
    """Return causal probabilities plus every inner selection attempt.

    Current and future outcomes are never included in training or selection.
    All data-dependent transforms are refitted inside each validation block.
    Optional columns are admitted as one declared block; no per-feature search.
    """
    positive_counts = (config.train_window, config.min_train_rows, config.validation_block_rows,
                       config.validation_blocks, config.refit_interval, config.max_iter)
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in positive_counts):
        raise ValueError("Nested sample counts must be positive integers.")
    if config.train_window < config.min_train_rows + config.validation_blocks * config.validation_block_rows:
        raise ValueError("Training window cannot hold the declared validation blocks.")
    penalties = config.inverse_penalties
    if not penalties or any(not np.isfinite(c) or c <= 0 for c in penalties) or len(set(penalties)) != len(penalties):
        raise ValueError("Inverse penalties must be distinct finite positive values.")
    if not isinstance(config.require_paired_nonregression, bool) or not np.isfinite(config.correction_threshold) or not 0.5 <= config.correction_threshold <= 1:
        raise ValueError("Paired selection requires a boolean policy and threshold in [0.5, 1].")
    if target_mode not in ("conditional_downside", "incumbent_error"):
        raise ValueError("Unknown nested probability target.")
    thresholds = config.threshold_candidates or (config.correction_threshold,)
    if (len(set(thresholds)) != len(thresholds) or any(not np.isfinite(t) or not 0.5 <= t <= 1 for t in thresholds)
            or (config.threshold_candidates and not config.require_paired_nonregression)):
        raise ValueError("Learned thresholds require paired selection and distinct probabilities in [0.5, 1].")
    frame, feature_frame = champion.reset_index(drop=True), features.reset_index(drop=True)
    if (frame.empty or frame.trade_date.duplicated().any() or not frame.trade_date.is_monotonic_increasing
            or not np.array_equal(frame.trade_date, feature_frame.trade_date) or feature_frame.columns.duplicated().any()):
        raise ValueError("Nested features require unique chronological aligned dates and columns.")
    labels = frame.predicted_label.to_numpy(dtype=float)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Incumbent directions must be binary.")
    optional = tuple(optional_columns)
    all_columns = tuple(name for name in feature_frame if name != "trade_date")
    if len(set(optional)) != len(optional) or not set(optional) <= set(all_columns):
        raise ValueError("Optional columns must be distinct observed feature names.")
    base_columns = tuple(name for name in all_columns if name not in optional)
    if not base_columns:
        raise ValueError("Nested comparison requires nonempty base features.")
    matrices = {"base": feature_frame.loc[:, base_columns].to_numpy(dtype=float)}
    if optional:
        matrices["extended"] = feature_frame.loc[:, all_columns].to_numpy(dtype=float)
    if available is None:
        available = np.ones(len(frame), dtype=bool)
    else:
        available = np.asarray(available)
    if available.shape != (len(frame),) or available.dtype != np.dtype(bool):
        raise ValueError("Availability must be an aligned boolean mask.")
    if any(not np.isfinite(values[available]).all() for values in matrices.values()):
        raise ValueError("Available nested features must be finite.")
    actual = frame.real_pct_change.to_numpy(dtype=float)
    if np.isinf(actual).any():
        raise ValueError("Infinite outcomes cannot train a model.")
    eligible_direction = labels == 1 if target_mode == "conditional_downside" else np.ones(len(frame), dtype=bool)
    known = np.isfinite(actual) & eligible_direction & available
    target = ((actual <= 0) if target_mode == "conditional_downside" else (labels != (actual > 0))).astype(int)
    size = len(frame)
    result = pd.DataFrame({
        "trade_date": frame.trade_date, "downside_probability": np.zeros(size),
        "downside_training_rows": np.zeros(size, dtype=int),
        "downside_last_training_date": np.zeros(size, dtype=int),
        "nested_validation_rows": np.zeros(size, dtype=int),
        "nested_last_validation_date": np.zeros(size, dtype=int),
        "nested_selected_c": np.zeros(size), "nested_selected_features": ["unfitted"] * size,
        "nested_validation_log_loss": np.zeros(size), "nested_validation_brier": np.zeros(size),
        "nested_selected_threshold": np.zeros(size),
    })
    attempts = []
    fitted = None
    last_fit = -config.refit_interval
    selected = None
    selection_checked = False
    last_rows = last_date = 0
    minimum = config.min_train_rows + config.validation_blocks * config.validation_block_rows
    for index in range(size):
        if not eligible_direction[index] or not available[index]:
            continue
        history = np.arange(max(0, index - config.train_window), index)
        history = history[known[history]]
        if len(history) < minimum:
            continue
        if not selection_checked or index - last_fit >= config.refit_interval:
            folds = validation_blocks(history, config)
            if any(len(np.unique(target[training])) < 2 for training, _ in folds):
                fitted = None
                selection_checked = False
                continue
            choices = []
            for mode, values in matrices.items():
                for c in sorted(penalties):
                    fold_losses, fold_briers, fold_probabilities = [], [], []
                    for training, validation in folds:
                        model = _fit(values[training], target[training], c, config.max_iter)
                        probability = model.predict_proba(values[validation])[:, 1]
                        clipped = np.clip(probability, 1e-12, 1 - 1e-12)
                        y = target[validation]
                        fold_losses.append(float(-np.mean(y * np.log(clipped) + (1-y) * np.log1p(-clipped))))
                        fold_briers.append(float(np.mean((probability-y)**2)))
                        fold_probabilities.append(probability)
                    for threshold in sorted(thresholds):
                        paired = ([paired_validation(actual, labels, validation, probability, threshold)
                                   for (_, validation), probability in zip(folds, fold_probabilities, strict=True)]
                                  if config.require_paired_nonregression else [])
                        corrected = sum(fold["corrected"] for fold in paired)
                        damaged = sum(fold["damaged"] for fold in paired)
                        up_rows = sum(fold["actual_up_rows"] for fold in paired)
                        down_rows = sum(fold["actual_down_rows"] for fold in paired)
                        up_delta = sum(fold["up_hit_delta"] for fold in paired)
                        down_delta = sum(fold["down_hit_delta"] for fold in paired)
                        admissible = (all(fold["nondecreasing"] for fold in paired) and corrected > damaged) if paired else True
                        choices.append({"features": mode, "inverse_penalty": c, "correction_threshold": threshold,
                                        "log_loss": float(np.mean(fold_losses)), "brier": float(np.mean(fold_briers)),
                                        "fold_log_loss": fold_losses, "paired_validation": paired, "admissible": admissible,
                                        "pooled_accuracy_gain": (corrected-damaged)/(up_rows+down_rows) if paired else 0,
                                        "pooled_balanced_gain": .5*(down_delta/down_rows+up_delta/up_rows) if up_rows and down_rows else None,
                                        "changed_validation_rows": corrected + damaged})
            selected = select_choice(choices, learn_threshold=bool(config.threshold_candidates))
            if selected is not None:
                fitted = _fit(matrices[selected["features"]][history], target[history], selected["inverse_penalty"], config.max_iter)
            else:
                selected, fitted = None, None
            selection_checked = True
            last_rows, last_date, last_fit = len(history), int(frame.trade_date.iloc[history[-1]]), index
            attempts.append({"signal_date": int(frame.trade_date.iloc[index]), "training_rows": last_rows,
                             "last_training_signal_date": last_date,
                             "folds": [{"train_last_signal_date": int(frame.trade_date.iloc[training[-1]]),
                                        "validation_first_signal_date": int(frame.trade_date.iloc[validation[0]]),
                                        "validation_last_signal_date": int(frame.trade_date.iloc[validation[-1]])}
                                       for training, validation in folds],
                             "selected": dict(selected) if selected is not None else {"features": "incumbent"}, "choices": choices})
        if fitted is None:
            result.loc[index, ["nested_selected_features", "nested_validation_rows", "nested_last_validation_date"]] = [
                "incumbent", config.validation_blocks * config.validation_block_rows, last_date,
            ]
            continue
        probability = float(fitted.predict_proba(matrices[selected["features"]][index:index + 1])[0, 1])
        result.loc[index, ["downside_probability", "downside_training_rows", "downside_last_training_date",
                           "nested_validation_rows", "nested_last_validation_date", "nested_selected_c",
                           "nested_selected_features", "nested_validation_log_loss", "nested_validation_brier", "nested_selected_threshold"]] = [
            probability, last_rows, last_date, config.validation_blocks * config.validation_block_rows,
            last_date, selected["inverse_penalty"], selected["features"], selected["log_loss"], selected["brier"], selected["correction_threshold"],
        ]
    return result, attempts


def selected_threshold_predictions(core, champion, probabilities, *, bidirectional=False):
    """Use each model's previously selected threshold with common causal calibration."""
    from tools.evaluate_selective_context import selective_predictions

    if not np.array_equal(champion.trade_date, probabilities.trade_date):
        raise ValueError("Selected thresholds require identical signal dates.")
    probability = probabilities.downside_probability.to_numpy(dtype=float)
    threshold = probabilities.nested_selected_threshold.to_numpy(dtype=float)
    fitted = probabilities.downside_training_rows.to_numpy(dtype=int) > 0
    if not (np.isfinite(probability) & (probability >= 0) & (probability <= 1)).all():
        raise ValueError("Downside scores must be finite probabilities.")
    if not (np.isfinite(threshold[fitted]) & (threshold[fitted] >= .5) & (threshold[fitted] <= 1)).all():
        raise ValueError("Fitted decision thresholds must be in [0.5, 1].")
    for name in ("nested_last_validation_date", "downside_last_training_date"):
        if (fitted & probabilities[name].ge(probabilities.trade_date)).any():
            raise ValueError("Threshold selection and training must precede the signal date.")
    proposal = champion.copy()
    labels = champion.predicted_label.to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Selected thresholds require binary incumbent labels.")
    selected = fitted & (probability >= threshold)
    if not bidirectional:
        selected &= labels == 1
    proposal["predicted_label"] = np.where(selected, 1-labels, labels)
    proposal["residual_probability_up"] = np.where(selected, np.where(labels == 1, 1-probability, probability), labels)
    result = selective_predictions(core, champion, proposal, .5)
    for name in probabilities.columns.difference(["trade_date"]):
        result[name] = probabilities[name].to_numpy()
    return result


ERROR_COLUMNS = {"downside_probability": "error_probability", "downside_training_rows": "error_training_rows",
                 "downside_last_training_date": "error_last_training_date"}


def directional_error_features(champion, features, optional_columns=()):
    """Share observations while allowing each market input to depend on original direction."""
    if not np.array_equal(champion.trade_date, features.trade_date):
        raise ValueError("Directional context requires identical dates.")
    labels = champion.predicted_label.to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Directional context requires binary incumbent labels.")
    if "incumbent_direction" in features or any(str(n).endswith("__direction") for n in features):
        raise ValueError("Directional context names are reserved.")
    values = features.reset_index(drop=True).copy()
    signed = 2 * labels - 1
    names = [n for n in features if n != "trade_date"]
    interactions = values.loc[:, names].mul(signed, axis=0).add_suffix("__direction")
    values["incumbent_direction"] = signed
    values = pd.concat([values, interactions], axis=1)
    optional = tuple(optional_columns) + tuple(f"{name}__direction" for name in optional_columns)
    return values, optional


def nested_error_probabilities(champion, features, optional_columns=(), *, available=None, config=DEFAULT_CONFIG):
    values, optional = directional_error_features(champion, features, optional_columns)
    probabilities, attempts = nested_downside_probabilities(
        champion, values, optional, available=available, config=config, target_mode="incumbent_error",
    )
    probabilities = probabilities.rename(columns=ERROR_COLUMNS)
    probabilities["incumbent_label"] = champion.predicted_label.to_numpy()
    return probabilities, attempts


def selected_error_predictions(core, champion, probabilities):
    if not np.array_equal(champion.predicted_label, probabilities.incumbent_label):
        raise ValueError("Error probabilities must match the incumbent direction.")
    converted = probabilities.rename(columns={value:key for key,value in ERROR_COLUMNS.items()})
    result = selected_threshold_predictions(core, champion, converted, bidirectional=True)
    return result.rename(columns=ERROR_COLUMNS)
