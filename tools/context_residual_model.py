"""Causal boosting corrections anchored to the production forecast prior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from 数据拉取脚本_tushare import _asset_feature_frame


COMMON = {
    "train_window": 756, "min_train_rows": 252, "refit_interval": 5,
    "num_boost_round": 60, "eta": 0.05, "min_child_weight": 20,
    "lambda": 20, "alpha": 1, "seed": 42, "nthread": 1,
    "objective": "binary:logistic", "tree_method": "hist", "max_bin": 64,
}
CANDIDATES = {
    "context_residual_stumps": {"max_depth": 1, "balanced": False},
    "context_residual_shallow": {"max_depth": 2, "balanced": False},
    "context_residual_balanced": {"max_depth": 2, "balanced": True},
}
ERROR_CANDIDATES = {
    "context_error_long": {"max_depth": 2, "balanced": False, "target": "correctness"},
    "context_error_year": {"max_depth": 2, "balanced": False, "target": "correctness",
                           "train_window": 252, "min_train_rows": 120, "min_child_weight": 5},
    "context_error_quarter": {"max_depth": 2, "balanced": False, "target": "correctness",
                              "train_window": 120, "min_train_rows": 60, "min_child_weight": 5},
}
FUNDING_CANDIDATES = {
    "funding_direction": {"max_depth": 2, "balanced": False},
    "funding_direction_balanced": {"max_depth": 2, "balanced": True},
    "funding_correctness": {"max_depth": 2, "balanced": False, "target": "correctness"},
}
HISTORY_CANDIDATES = {
    "history_direction": {"max_depth": 2, "balanced": False, "train_window": 2520},
    "history_direction_balanced": {"max_depth": 2, "balanced": True, "train_window": 2520},
    "history_correctness": {"max_depth": 2, "balanced": False, "target": "correctness", "train_window": 2520},
}
BREADTH_CANDIDATES = {
    "breadth_direction": {"max_depth": 2, "balanced": False},
    "breadth_direction_balanced": {"max_depth": 2, "balanced": True},
    "breadth_correctness": {"max_depth": 2, "balanced": False, "target": "correctness"},
}
ASSETS = ("csi300", "csi500", "chinext")


def context_features(core, data: pd.DataFrame, config, assets: dict[str, pd.DataFrame]):
    base = core._normalize_market_frame(data, config)
    features = core._build_features(base, config)
    dates = pd.DatetimeIndex(base.date)
    for name in ASSETS:
        asset = _asset_feature_frame(name, assets[name], lag=0).set_index("trade_date")
        if asset.index.duplicated().any() or len(dates.difference(asset.index)):
            raise ValueError(f"Missing or duplicate context dates for {name}.")
        asset = asset.reindex(dates).reset_index(drop=True)
        asset = asset.drop(columns=[f"{name}_pct_chg"], errors="ignore")
        asset[f"{name}_relative_return_1"] = asset[f"{name}_ret1"] - features["return_1"]
        asset[f"{name}_relative_return_5"] = asset[f"{name}_ret5"] - features["momentum_5"]
        features = pd.concat([features, asset], axis=1)
    return base, core._clean_feature_frame(features)


def residual_predictions(core, champion, base, features, name):
    settings = COMMON | (CANDIDATES | ERROR_CANDIDATES | FUNDING_CANDIDATES | HISTORY_CANDIDATES | BREADTH_CANDIDATES)[name]
    frame = champion.copy().reset_index(drop=True)
    positions = pd.Index(base.date.dt.strftime("%Y%m%d").astype(int)).get_indexer(frame.trade_date)
    if (positions < 0).any() or (np.diff(positions) <= 0).any():
        raise ValueError("Champion dates must align chronologically with the context.")
    matrix = features.iloc[positions].copy().reset_index(drop=True)
    old_labels = frame.predicted_label.to_numpy(dtype=int).copy()
    correctness_probability = np.clip(frame.calibrated_confidence.to_numpy(dtype=float), 0.01, 0.99)
    prior = np.where(old_labels == 1, correctness_probability, 1 - correctness_probability)
    matrix["champion_label"] = old_labels
    matrix["champion_probability_up"] = prior
    predict_correctness = settings.get("target") == "correctness"
    if predict_correctness:
        matrix["champion_correctness_probability"] = correctness_probability
    values = matrix.to_numpy(dtype=float)
    target_prior = correctness_probability if predict_correctness else prior
    margins = np.log(target_prior / (1 - target_prior))
    realized = frame.real_pct_change.to_numpy(dtype=float)
    labels = (realized > 0).astype(int)
    training_labels = (old_labels == labels).astype(int) if predict_correctness else labels
    known = np.isfinite(realized)
    predicted = old_labels.copy()
    probabilities = prior.copy()
    training_rows = np.zeros(len(frame), dtype=int)
    last_fit_signal = np.full(len(frame), -1, dtype=int)
    model = None
    last_rows, last_index = 0, -1
    parameters = {key: value for key, value in settings.items()
                  if key not in {"train_window", "min_train_rows", "refit_interval", "num_boost_round", "balanced", "target"}}
    for index in range(len(frame)):
        history = np.arange(max(0, index - settings["train_window"]), index)
        history = history[known[history]]
        if len(history) < settings["min_train_rows"] or len(np.unique(training_labels[history])) < 2:
            continue
        if model is None or positions[index] % settings["refit_interval"] == 0:
            weights = None
            if settings["balanced"]:
                class_counts = np.bincount(training_labels[history], minlength=2)
                weights = len(history) / (2 * class_counts[training_labels[history]])
            training = xgb.DMatrix(values[history], label=training_labels[history], base_margin=margins[history], weight=weights, nthread=1)
            model = xgb.train(parameters, training, num_boost_round=settings["num_boost_round"])
            last_rows, last_index = len(history), int(frame.trade_date.iloc[history[-1]])
        current = xgb.DMatrix(values[index:index + 1], base_margin=margins[index:index + 1], nthread=1)
        probability = float(model.predict(current)[0])
        probabilities[index] = 1 - probability if predict_correctness and old_labels[index] == 0 else probability
        predicted[index] = int(probabilities[index] >= 0.5)
        training_rows[index], last_fit_signal[index] = last_rows, last_index
    changed = predicted != old_labels
    raw = np.abs(frame.uncalibrated_predicted_return.to_numpy(dtype=float)) * np.where(predicted == 1, 1, -1)
    frame["predicted_label"] = predicted
    frame["predicted_pct_change"] = raw
    frame["predicted_close"] = base.close.to_numpy()[positions] * (1 + raw)
    # Agreement retains the champion's raw confidence; corrections supply their
    # own probability before the common causal calibration of the mixed stream.
    frame["confidence"] = np.where(changed, np.maximum(probabilities, 1 - probabilities), frame.confidence)
    frame["correct"] = pd.Series(predicted == labels).where(known, None)
    frame = core._apply_rolling_confidence_calibration(frame, window=300, min_rows=60, method="platt")
    from return_calibration import calibrate_returns
    frame = calibrate_returns(frame, window=252, min_rows=60)
    frame["residual_probability_up"] = probabilities
    frame["residual_changed"] = changed
    frame["residual_training_rows"] = training_rows
    frame["residual_last_training_signal"] = last_fit_signal
    return frame


def load_assets(directory: Path):
    return {name: pd.read_csv(directory / f"{name}.csv", float_precision="round_trip") for name in ASSETS}
