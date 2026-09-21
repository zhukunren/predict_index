"""Causal downside probabilities conditional on an incumbent up forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from tools.breadth_features import breadth_features
from tools.global_risk_features import global_risk_features
from tools.evaluate_selective_context import selective_predictions


COMMON = {"train_window": 756, "min_train_rows": 120, "refit_interval": 5}
TRAINING_POLICIES = {
    "uniform": {"half_life_sessions": None},
    "recent": {"half_life_sessions": 126},
}
CANDIDATES = {
    "downside_logistic": {"C": 0.1, "max_iter": 1000},
    "downside_shallow": {
        "max_depth": 2, "num_boost_round": 60, "eta": 0.05,
        "min_child_weight": 10, "lambda": 20, "alpha": 1,
        "seed": 42, "nthread": 1, "objective": "binary:logistic",
        "tree_method": "hist", "max_bin": 64,
    },
    "downside_forest": {
        "n_estimators": 200, "max_depth": 3, "min_samples_leaf": 20,
        "max_features": 0.5, "bootstrap": True, "random_state": 42, "n_jobs": 1,
    },
}
MODEL_FAMILIES = {
    "classic": ("downside_logistic", "downside_shallow"),
    "bagged": ("downside_forest",),
}
PRICE_COLUMNS = (
    "return_1", "momentum_5", "momentum_20", "volatility_20",
    "intraday_return", "close_position",
)
BREADTH_COLUMNS = (
    "breadth_all_up_fraction", "breadth_all_down_amount_fraction",
    "breadth_all_large_down_fraction", "breadth_all_up_change_1",
    "breadth_all_up_mean_5", "breadth_shanghai_relative",
)
GLOBAL_COLUMNS = (
    "spx_return_1", "spx_return_5", "nasdaq_return_1", "nasdaq_drawdown_20",
)


def specialist_features(core, market, champion, config, breadth, overseas):
    base = core._normalize_market_frame(market, config)
    dates = base.date.dt.strftime("%Y%m%d").astype(int)
    positions = pd.Index(dates).get_indexer(champion.trade_date)
    if (positions < 0).any() or (np.diff(positions) <= 0).any():
        raise ValueError("Specialist signals must align chronologically with the market.")
    price = core._build_features(base, config).iloc[positions].reset_index(drop=True)
    breadth_frame = breadth_features(champion.trade_date, breadth, dates, lag_sessions=0, publication_hour=18)
    global_frame = global_risk_features(champion.trade_date, overseas)
    features = pd.concat([
        champion[["trade_date", "confidence", "calibrated_confidence"]].reset_index(drop=True),
        price.loc[:, PRICE_COLUMNS], breadth_frame.loc[:, BREADTH_COLUMNS],
        global_frame.loc[:, GLOBAL_COLUMNS],
    ], axis=1)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Specialist features must be finite; missing context cannot be filled.")
    return features


def training_weights(history, current, policy):
    half_life = TRAINING_POLICIES[policy]["half_life_sessions"]
    if half_life is None:
        return None
    weights = np.exp2(-(current - 1 - history) / half_life)
    # Keep regularization comparable when changing the distribution of weights.
    return weights / weights.mean()


def downside_probabilities(champion, features, name, *, available=None, training_policy="uniform"):
    if training_policy not in TRAINING_POLICIES:
        raise ValueError("Unknown specialist training policy.")
    frame = champion.copy().reset_index(drop=True)
    if frame.empty or frame.trade_date.duplicated().any() or not frame.trade_date.is_monotonic_increasing:
        raise ValueError("Specialist signals must be unique and chronological.")
    if not np.array_equal(frame.trade_date, features.trade_date):
        raise ValueError("Specialist features require identical signal dates.")
    labels = frame.predicted_label.to_numpy(dtype=float)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Specialist requires binary incumbent labels.")
    values = features.drop(columns="trade_date").to_numpy(dtype=float)
    if available is None:
        available = np.ones(len(frame), dtype=bool)
    else:
        available = np.asarray(available)
        if available.shape != (len(frame),) or available.dtype != np.dtype(bool):
            raise ValueError("Specialist availability must be an aligned boolean mask.")
    if not np.isfinite(values[available]).all():
        raise ValueError("Specialist features must be finite.")
    settings = CANDIDATES[name]
    realized = frame.real_pct_change.to_numpy(dtype=float)
    known_up = np.isfinite(realized) & (labels == 1) & available
    target = (realized <= 0).astype(int)
    probability = np.zeros(len(frame))
    training_rows = np.zeros(len(frame), dtype=int)
    last_training_date = np.zeros(len(frame), dtype=int)
    model = None
    last_fit, last_rows, last_date = -COMMON["refit_interval"], 0, 0
    for index in range(len(frame)):
        if labels[index] != 1 or not available[index]:
            continue
        history = np.arange(max(0, index - COMMON["train_window"]), index)
        history = history[known_up[history]]
        if len(history) < COMMON["min_train_rows"] or len(np.unique(target[history])) < 2:
            continue
        if model is None or index - last_fit >= COMMON["refit_interval"]:
            weights = training_weights(history, index, training_policy)
            if name == "downside_logistic":
                model = make_pipeline(StandardScaler(), LogisticRegression(**settings))
                if weights is None:
                    model.fit(values[history], target[history])
                else:
                    model.fit(values[history], target[history],
                              standardscaler__sample_weight=weights,
                              logisticregression__sample_weight=weights)
            elif name == "downside_forest":
                model = RandomForestClassifier(**settings)
                model.fit(values[history], target[history], sample_weight=weights)
            else:
                parameters = {key: value for key, value in settings.items() if key != "num_boost_round"}
                training = xgb.DMatrix(values[history], label=target[history], weight=weights, nthread=1)
                model = xgb.train(parameters, training, num_boost_round=settings["num_boost_round"])
            last_fit, last_rows, last_date = index, len(history), int(frame.trade_date.iloc[history[-1]])
        if name == "downside_shallow":
            probability[index] = model.predict(xgb.DMatrix(values[index:index + 1], nthread=1))[0]
        else:
            probability[index] = model.predict_proba(values[index:index + 1])[0, 1]
        training_rows[index], last_training_date[index] = last_rows, last_date
    return pd.DataFrame({
        "trade_date": frame.trade_date, "downside_probability": probability,
        "downside_training_rows": training_rows, "downside_last_training_date": last_training_date,
    })


def specialist_predictions(core, champion, probabilities, threshold):
    if not 0.5 < threshold <= 1:
        raise ValueError("Specialist threshold must be in (0.5, 1].")
    if not np.array_equal(champion.trade_date, probabilities.trade_date):
        raise ValueError("Specialist probabilities require identical signal dates.")
    probability = probabilities.downside_probability.to_numpy(dtype=float)
    fitted = probabilities.downside_training_rows.to_numpy() > 0
    if not (np.isfinite(probability) & (probability >= 0) & (probability <= 1)).all():
        raise ValueError("Downside probabilities must be finite and in [0, 1].")
    if (fitted & probabilities.downside_last_training_date.ge(probabilities.trade_date)).any():
        raise ValueError("Specialist training must precede each signal date.")
    proposal = champion.copy()
    selected = champion.predicted_label.eq(1).to_numpy() & fitted & (probability >= threshold)
    proposal["predicted_label"] = np.where(selected, 0, champion.predicted_label)
    # Use the incumbent label to prevent corrections to any original down signal.
    proposal["residual_probability_up"] = np.where(selected, 1 - probability, champion.predicted_label)
    result = selective_predictions(core, champion, proposal, threshold)
    for column in probabilities.columns.difference(["trade_date"]):
        result[column] = probabilities[column].to_numpy()
    return result


def mean_downside_probabilities(base, extended):
    if not np.array_equal(base.trade_date, extended.trade_date):
        raise ValueError("Ensemble probabilities require identical signal dates.")
    for frame in (base, extended):
        dates = frame.trade_date
        if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Ensemble signal dates must be unique and chronological.")
        probability = frame.downside_probability.to_numpy(dtype=float)
        if not (np.isfinite(probability) & (probability >= 0) & (probability <= 1)).all():
            raise ValueError("Ensemble inputs must be finite probabilities.")
        fitted = frame.downside_training_rows.gt(0)
        if (fitted & frame.downside_last_training_date.ge(dates)).any():
            raise ValueError("Ensemble training must precede its signal date.")
    result = base.copy().reset_index(drop=True)
    rows = np.minimum(base.downside_training_rows.to_numpy(), extended.downside_training_rows.to_numpy())
    ready = rows > 0
    probability = (base.downside_probability.to_numpy() + extended.downside_probability.to_numpy()) / 2
    result["downside_probability"] = np.where(ready, probability, 0)
    result["downside_training_rows"] = rows
    result["downside_last_training_date"] = np.where(
        ready, np.maximum(base.downside_last_training_date.to_numpy(), extended.downside_last_training_date.to_numpy()), 0,
    )
    result["downside_base_probability"] = base.downside_probability.to_numpy()
    result["downside_extended_probability"] = extended.downside_probability.to_numpy()
    return result
