"""Evaluate three predeclared daily models on development dates only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
DEVELOPMENT_START = pd.Timestamp("2023-01-01")
DEVELOPMENT_END = pd.Timestamp("2024-12-31")
TRAIN_WINDOW = 756
REFIT_EVERY = 5
SEED = 42
PARAMETERS = {
    "logistic_l2": {
        "C": 0.1,
        "max_iter": 2000,
        "solver": "liblinear",
        "random_state": SEED,
    },
    "hist_gradient_boosting": {
        "max_iter": 100,
        "max_leaf_nodes": 7,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_samples_leaf": 40,
        "l2_regularization": 10.0,
        "early_stopping": False,
        "random_state": SEED,
    },
    "extra_trees": {
        "n_estimators": 200,
        "max_depth": 4,
        "min_samples_leaf": 30,
        "max_features": 0.7,
        "n_jobs": 1,
        "random_state": SEED,
    },
}


def model_for(name: str):
    if name == "logistic_l2":
        return make_pipeline(StandardScaler(), LogisticRegression(**PARAMETERS[name]))
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**PARAMETERS[name])
    return ExtraTreesClassifier(**PARAMETERS[name])


def main() -> None:
    source_path = ROOT / "\u5faa\u73af\u9a8c\u8bc1\u811a\u672c.py"
    spec = importlib.util.spec_from_file_location("research_direction_core", source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = module.DirectionPredictionConfig(external_feature_mode="core", neutral_band=0.0)
    raw = pd.read_csv(ROOT / "market_data" / "merged_features.csv")
    dates = pd.to_datetime(raw["trade_date"])
    last_development = int(np.flatnonzero(dates.le(DEVELOPMENT_END).to_numpy())[-1])
    # One following close resolves the final development signal; no 2025 signal is scored.
    raw = raw.iloc[: last_development + 2].copy()
    base = module._normalize_market_frame(raw, config)
    features = module._clean_feature_frame(module._build_features(base, config))
    values = features.to_numpy(dtype=float)
    next_returns = base["close"].shift(-1) / base["close"] - 1.0
    labels = next_returns.gt(0).to_numpy(dtype=int)
    eligible = (
        base["date"].between(DEVELOPMENT_START, DEVELOPMENT_END)
        & next_returns.notna()
    ).to_numpy()
    indices = np.flatnonzero(eligible)
    contract = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "development_input_sha256": hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest(),
        "development_signal_start": DEVELOPMENT_START.strftime("%Y-%m-%d"),
        "development_signal_end": DEVELOPMENT_END.strftime("%Y-%m-%d"),
        "unseen_test_signal_start": "2025-01-01",
        "test_performance_read": False,
        "label": "next close / current close - 1 > 0; no neutral filtering",
        "prediction_time": "after signal-day close",
        "features": list(features.columns),
        "cleaning": "existing forward-fill then zero; no backward fill",
        "train_window": TRAIN_WINDOW,
        "minimum_training_rows": 252,
        "refit_every": REFIT_EVERY,
        "refit_anchor": "normalized row index modulo 5, with initial fit",
        "training_label_cutoff": "signal rows strictly before current signal row",
        "direction_threshold": 0.5,
        "candidate_parameters": PARAMETERS,
        "candidate_attempt_budget": 3,
        "selection_policy": "development only; no subsequent parameter search",
        "boundary": "first 2025 close may resolve 2024-12-31 signal only",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract_path = OUTPUT / "fixed_model_development_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    rows = []
    attempts = []
    for attempt, name in enumerate(PARAMETERS, start=1):
        start_time = time.perf_counter()
        model = None
        fit_count = 0
        for idx in indices:
            train_indices = np.arange(max(0, idx - TRAIN_WINDOW), idx)
            train_indices = train_indices[next_returns.iloc[train_indices].notna().to_numpy()]
            assert len(train_indices) >= 252
            assert int(train_indices.max()) < int(idx)
            if model is None or idx % REFIT_EVERY == 0:
                model = model_for(name)
                with threadpool_limits(limits=1):
                    model.fit(values[train_indices], labels[train_indices])
                fit_count += 1
                fit_last_label_signal = int(train_indices[-1])
                fit_prior = float(labels[train_indices].mean())
            with threadpool_limits(limits=1):
                probability_up = float(model.predict_proba(values[idx : idx + 1])[0, 1])
            rows.append({
                "model": name,
                "signal_date": base["date"].iloc[idx].strftime("%Y-%m-%d"),
                "target_date": base["date"].iloc[idx + 1].strftime("%Y-%m-%d"),
                "train_last_label_signal_date": base["date"].iloc[fit_last_label_signal].strftime("%Y-%m-%d"),
                "probability_up": probability_up,
                "predicted_up": int(probability_up >= 0.5),
                "actual_up": int(labels[idx]),
                "actual_return": float(next_returns.iloc[idx]),
                "train_up_prior": fit_prior,
            })
        attempts.append({
            "attempt": attempt,
            "model": name,
            "parameters": PARAMETERS[name],
            "rows": len(indices),
            "fit_count": fit_count,
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            "status": "complete",
            "parameter_revision": False,
        })
        (OUTPUT / "fixed_model_development_attempts.json").write_text(
            json.dumps(attempts, indent=2), encoding="utf-8"
        )
        print(json.dumps(attempts[-1]), flush=True)
    predictions = pd.DataFrame(rows)
    baseline_rows = predictions.loc[predictions["model"].eq("logistic_l2")].copy()
    always_up = baseline_rows.assign(model="always_up", probability_up=1.0, predicted_up=1)
    training_prior = baseline_rows.assign(
        model="training_prior",
        probability_up=baseline_rows["train_up_prior"],
        predicted_up=baseline_rows["train_up_prior"].ge(0.5).astype(int),
    )
    predictions = pd.concat([predictions, always_up, training_prior], ignore_index=True)
    predictions["correct"] = predictions["predicted_up"].eq(predictions["actual_up"])
    predictions.to_csv(OUTPUT / "fixed_model_development_predictions.csv", index=False)
    summaries = []
    for name, model_rows in predictions.groupby("model", sort=False):
        for period in ("2023-2024", "2023", "2024"):
            sample = model_rows if period == "2023-2024" else model_rows.loc[
                model_rows["signal_date"].str.startswith(period)
            ]
            up_mask = sample["predicted_up"].eq(1)
            down_mask = ~up_mask
            summaries.append({
                "model": name,
                "period": period,
                "rows": len(sample),
                "correct": int(sample["correct"].sum()),
                "accuracy": float(accuracy_score(sample["actual_up"], sample["predicted_up"])),
                "balanced_accuracy": float(balanced_accuracy_score(sample["actual_up"], sample["predicted_up"])),
                "brier_score": float(brier_score_loss(sample["actual_up"], sample["probability_up"])),
                "always_up_accuracy": float(sample["actual_up"].mean()),
                "up_predictions": int(up_mask.sum()),
                "down_predictions": int(down_mask.sum()),
                "up_precision": float(sample.loc[up_mask, "correct"].mean()) if up_mask.any() else None,
                "down_precision": float(sample.loc[down_mask, "correct"].mean()) if down_mask.any() else None,
            })
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "fixed_model_development_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
