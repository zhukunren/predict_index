"""Direction-first BiLSTM-Attention predictor for daily market data.

This module is intentionally independent from the existing prediction scripts in
the workspace. It follows the method described in the PDF:

* daily OHLCV features plus technical indicators
* 30-trading-day sequence inputs
* next-trading-day direction label with a +/-0.1% neutral band
* BiLSTM encoder, temporal attention, dropout, small dense layer
* validation-tuned classification threshold

The public one-call API is:

    result = predict_next_day(df)

Optional walk-forward validation is available through:

    validation = walk_forward_validate(df)

The model is direction-first. The returned next-return estimate is only a
secondary historical calibration based on realized returns in the predicted
direction; it is not used to decide up/down.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from technical_features import legacy_preprocess_data as _legacy_technical_features
except Exception:  # pragma: no cover - keep this predictor usable standalone.
    _legacy_technical_features = None

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover - sklearn is expected, but keep import safe.
    average_precision_score = None
    roc_auc_score = None


DeviceName = Literal["auto", "cpu", "cuda"]
ExternalFeatureMode = Literal["none", "core", "all"]
TechnicalFeatureMode = Literal["none", "v1", "v1_core"]
ReturnMagnitudeMode = Literal["directional_median", "range_scaled"]


# =========================
# Script Run Settings
# =========================
# Edit these variables when you want to run this file directly without typing
# command-line arguments. Command-line arguments still override these values.
#
# SCRIPT_MODE:
#   "predict"       -> train once and predict the next trading day
#   "loop_validate" -> generate prediction_results.csv-style daily validation
#   "walk_forward"  -> fold-level walk-forward metrics
SCRIPT_MODE = "loop_validate"
SCRIPT_CSV_PATH: str | None = "market_data/merged_features.csv"
SCRIPT_OUTPUT_PATH: str | None = "drp_feim_prediction_results.csv"
SCRIPT_DIAGNOSTICS_OUTPUT_PATH: str | None = "drp_feim_rule_diagnostics.csv"
SCRIPT_CONFIDENCE_OUTPUT_PATH: str | None = "drp_feim_high_confidence_results.csv"
SCRIPT_CONFIDENCE_SUMMARY_PATH: str | None = "drp_feim_confidence_summary.csv"
SCRIPT_ENCODING = "utf-8-sig"

# Date range for loop validation. Keep start/end as None to validate the latest
# SCRIPT_PERIODS verifiable trading days.
SCRIPT_START_DATE: str | None = None
SCRIPT_END_DATE: str | None = None
SCRIPT_PERIODS = 10 #回测周期

SCRIPT_EPOCHS = 10
SCRIPT_LOOKBACK = 30
SCRIPT_NEUTRAL_BAND = 0.001
SCRIPT_DEVICE: DeviceName = "auto"
SCRIPT_EXTERNAL_FEATURE_MODE: ExternalFeatureMode = "core"
SCRIPT_TECHNICAL_FEATURE_MODE: TechnicalFeatureMode = "none"
SCRIPT_VERBOSE = False
SCRIPT_SHOW_PROGRESS = True

# Direction engine for loop validation:
#   "bilstm"          -> original PDF-style BiLSTM-Attention, GPU accelerated
#   "volatility_rule" -> strict nested daily rule selection; every day selects
#                        feature/threshold/direction from prior history only
#   "state_veto_rule" -> volatility_rule plus strict historical low-volatility
#                        state veto/reversal when that state has underperformed
#   "stability_rule"  -> strict nested rule selection with recent/long-term
#                        accuracy blend and stability penalty
#   "nested_ml"       -> strict nested ML model selection over existing features
#   "hybrid"          -> strict daily chooser between volatility_rule and nested_ml
#   "calibrated_rule" -> historical volatility/range rule ensemble calibrated
#                        before the validation window
#   "historical_selector" -> strict daily chooser between state_veto_rule and
#                        volatility_rule, with disagreement-only switch guard
SCRIPT_SIGNAL_ENGINE = "state_veto_rule"
SCRIPT_RULE_THRESHOLD_END_DATE: str | None = "20241231"
SCRIPT_RULE_CALIBRATION_START_DATE: str | None = "20240101"
SCRIPT_RULE_CALIBRATION_END_DATE: str | None = "20241231"
SCRIPT_RULE_TOP_K = 5
# Kept for CLI compatibility; strict volatility_rule now searches quantiles
# from prior history instead of using a fixed quantile.
SCRIPT_VOLATILITY_RULE_QUANTILE = 0.2
SCRIPT_NESTED_RULE_THRESHOLD_WINDOW = 756
SCRIPT_NESTED_RULE_CALIBRATION_WINDOW = 120
SCRIPT_NESTED_RULE_MIN_THRESHOLD_ROWS = 40
SCRIPT_NESTED_RULE_MIN_CALIBRATION_ROWS = 30
SCRIPT_STATE_VETO_WINDOW = 75
SCRIPT_STATE_VETO_STATE_WINDOW = 360
SCRIPT_STATE_VETO_MIN_ROWS = 10
SCRIPT_STATE_VETO_BAD_ACCURACY = 0.46
SCRIPT_STATE_VETO_QUANTILES = "0.4"
SCRIPT_HIGH_CONFIDENCE_MIN_BASE_CALIBRATION = 0.575
SCRIPT_HIGH_CONFIDENCE_MIN_VETO_STATE_ACCURACY = 0.333
SCRIPT_HIGH_CONFIDENCE_MAX_VETO_STATE_ROWS: int | None = 45
SCRIPT_STABILITY_RULE_LONG_WINDOW = 756
SCRIPT_STABILITY_RULE_RECENT_WEIGHT = 0.65
SCRIPT_STABILITY_RULE_MIN_EDGE = 0.005
SCRIPT_STABILITY_RULE_TOP_K = 3
SCRIPT_NESTED_ML_TRAIN_WINDOW = 1008
SCRIPT_NESTED_ML_STEP = 5
SCRIPT_NESTED_ML_CALIBRATION_WINDOW = 120
SCRIPT_NESTED_ML_MIN_TRAIN_ROWS = 252
SCRIPT_NESTED_ML_TOP_K = 3

# Magnitude calibration is sign-preserving: it may change the predicted return
# size, but it must not change the direction chosen by the signal engine.
SCRIPT_RETURN_MAGNITUDE_MODE: ReturnMagnitudeMode = "range_scaled"
SCRIPT_RETURN_MAGNITUDE_WINDOW = 756
SCRIPT_RETURN_MAGNITUDE_MIN_ROWS = 80
SCRIPT_RETURN_MAGNITUDE_GRID_SIZE = 120
SCRIPT_RETURN_MAGNITUDE_CLIP_LOW_QUANTILE = 0.05
SCRIPT_RETURN_MAGNITUDE_CLIP_HIGH_QUANTILE = 0.95

# Strictly historical failure guard. It does not use the current row's realized
# return. The guard first marks degraded recent performance, and only flips the
# current signal when a shorter confirmation window also remains weak.
SCRIPT_RECENT_FAILURE_GUARD = True
SCRIPT_RECENT_FAILURE_WINDOW = 15
SCRIPT_RECENT_FAILURE_DEGRADE_THRESHOLD = 0.45
SCRIPT_RECENT_FAILURE_INVERT_THRESHOLD = 0.40
SCRIPT_RECENT_FAILURE_SHORT_WINDOW = 5
SCRIPT_RECENT_FAILURE_SHORT_THRESHOLD = 0.40
# Backward-compatible alias for older code/notes.
SCRIPT_RECENT_FAILURE_THRESHOLD = SCRIPT_RECENT_FAILURE_DEGRADE_THRESHOLD

SCRIPT_SELECTOR_WINDOW = 240
SCRIPT_SELECTOR_MIN_HISTORY = 60
SCRIPT_SELECTOR_SWITCH_EDGE = 0.0
SCRIPT_SELECTOR_DISAGREEMENT_WINDOW = 240
SCRIPT_SELECTOR_DISAGREEMENT_MIN_HISTORY = 30
SCRIPT_SELECTOR_DISAGREEMENT_EDGE = 0.0

SCRIPT_REGIME_POSTPROCESS = False
SCRIPT_REGIME_POSTPROCESS_DIAGNOSTICS_OUTPUT_PATH: str | None = (
    "drp_feim_regime_postprocess_diagnostics.csv"
)
SCRIPT_REGIME_POSTPROCESS_STATE_COLUMNS = (
    "vol20_bucket,"
    "domestic_index_alignment,"
    "china_futures_alignment,"
    "close_position_zone,"
    "selected_engine,"
    "selector_hist_edge_bucket,"
    "streak_state,"
    "intraday_direction"
)
SCRIPT_REGIME_POSTPROCESS_HISTORY_WINDOW = 180
SCRIPT_REGIME_POSTPROCESS_MIN_HISTORY = 30
SCRIPT_REGIME_POSTPROCESS_FLIP_BELOW = 0.42
SCRIPT_REGIME_POSTPROCESS_MAX_FLIP_RATE = 1.0

SCRIPT_INITIAL_TRAIN_FRACTION = 0.65
SCRIPT_TEST_SIZE = 120
SCRIPT_MAX_SPLITS = 5


@dataclass(slots=True)
class DirectionPredictionConfig:
    """Configuration for the direction-first predictor."""

    lookback: int = 30
    neutral_band: float = 0.001
    validation_fraction: float = 0.15
    min_train_sequences: int = 200
    hidden_size: int = 64
    dense_size: int = 16
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    threshold_grid_min: float = 0.40
    threshold_grid_max: float = 0.60
    threshold_grid_step: float = 0.002
    fallback_threshold: float = 0.516
    random_seed: int = 42
    device: DeviceName = "auto"
    external_feature_mode: ExternalFeatureMode = "none"
    technical_feature_mode: TechnicalFeatureMode = "none"
    extra_core_external_prefixes: tuple[str, ...] = ()
    dayfirst: bool = True
    drop_zero_volume: bool = True
    verbose: bool = False
    use_timeseries_cv: bool = True  # 新增：使用时间序列交叉验证


@dataclass(slots=True)
class PreparedMarketData:
    frame: pd.DataFrame
    feature_columns: list[str]
    labeled_sequences: np.ndarray
    labels: np.ndarray
    next_returns: np.ndarray
    sample_end_dates: pd.Series
    latest_sequence: np.ndarray
    latest_date: pd.Timestamp
    latest_close: float
    original_rows: int
    cleaned_rows: int


@dataclass(slots=True)
class FittedDirectionModel:
    model: nn.Module
    scaler: StandardScaler
    config: DirectionPredictionConfig
    feature_columns: list[str]
    threshold: float
    validation_metrics: dict[str, float]
    return_stats: dict[str, float]
    latest_sequence_scaled: np.ndarray
    latest_date: pd.Timestamp
    latest_close: float
    probability_up: float
    device: str
    original_rows: int
    cleaned_rows: int
    labeled_sequences: int
    train_sequences: int
    validation_sequences: int


class RuleSignal(NamedTuple):
    predicted_return: float
    predicted_label: int
    rule_names: list[str]
    calibration_accuracy: float
    calibration_rows: int
    diagnostics: dict[str, Any]


class NestedRuleCache(NamedTuple):
    predictions: np.ndarray
    rule_names: list[str]
    labels: np.ndarray
    valid_mask: np.ndarray


class NestedMLCache(NamedTuple):
    predictions: np.ndarray
    model_names: list[str]
    labels: np.ndarray
    valid_mask: np.ndarray
    feature_columns: list[str]


class _BiLSTMAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dense_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.attention_score = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size * 2, dense_size)
        self.output = nn.Linear(dense_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attention_score(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        hidden = torch.relu(self.dense(self.dropout(context)))
        return self.output(self.dropout(hidden)).squeeze(-1)


_ALIASES: dict[str, tuple[str, ...]] = {
    "date": (
        "trade_date",
        "trade_dt",
        "date",
        "datetime",
        "time",
        "opdate",
    ),
    "open": ("open", "s_dq_open", "s_open", "adj_open"),
    "high": ("high", "s_dq_high", "s_high", "adj_high"),
    "low": ("low", "s_dq_low", "s_low", "adj_low"),
    "close": ("close", "s_dq_close", "s_close", "adj_close"),
    "pre_close": (
        "pre_close",
        "preclose",
        "prev_close",
        "previous_close",
        "s_dq_preclose",
    ),
    "volume": ("volume", "vol", "s_dq_volume", "s_volume"),
    "amount": ("amount", "turnover", "s_dq_amount", "s_amount"),
}


_BASE_INPUT_COLUMNS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
}
_FORBIDDEN_EXTERNAL_PREFIXES = (
    "target_next",
    "next_",
    "future_",
    "label",
    "correct",
    "real_",
    "predicted_",
)
_CORE_EXTERNAL_PREFIXES = (
    "hangseng_",
    "hstech_",
)
_INTERNAL_RULE_COLUMNS = (
    "volatility_20",
    "bollinger_width",
    "range_pct",
    "close_position",
    "volume_log_change",
    "amount_log_change",
    "return_lag_1",
    "macd_hist",
)
_EXTERNAL_RULE_SUFFIXES = (
    "_ret1",
    "_ret1_lag1",
    "_ret5",
    "_ret5_lag1",
    "_pct_chg",
    "_pct_chg_lag1",
    "_gap",
    "_gap_lag1",
    "_intraday",
    "_intraday_lag1",
    "_range",
    "_range_lag1",
    "_vol20",
    "_vol20_lag1",
    "_vol_chg",
    "_vol_chg_lag1",
    "_z60",
)
_MONEYFLOW_RULE_PREFIXES = (
    "ext_mkt_dc_",
    "ext_hsgt_flow_",
)
_FORBIDDEN_TECHNICAL_COLUMNS = {
    "ichimoku_chikou",
}
_CORE_TECHNICAL_COLUMNS = {
    "pct_change",
    "chaikin_osc",
    "kst_signal",
    "bollinger_width",
    "volume",
}


def predict_next_day(
    df: pd.DataFrame,
    config: DirectionPredictionConfig | None = None,
) -> dict[str, Any]:
    """Fit on historical data and predict the next trading day's direction.

    Parameters
    ----------
    df:
        Market data sorted either ascending or descending. Supported schemas
        include common Tushare style columns (trade_date/open/high/low/close)
        and Wind style columns (trade_dt/s_dq_open/...).
    config:
        Optional model and preprocessing configuration.

    Returns
    -------
    dict
        Serializable prediction summary. The direction fields are primary; the
        estimated return fields are auxiliary.
    """

    fitted = fit_direction_model(df, config=config)
    return prediction_to_dict(fitted)


def fit_direction_model(
    df: pd.DataFrame,
    config: DirectionPredictionConfig | None = None,
) -> FittedDirectionModel:
    """Fit the BiLSTM-Attention direction model and return the fitted object."""

    cfg = config or DirectionPredictionConfig()
    _validate_config(cfg)
    _set_random_seed(cfg.random_seed)

    prepared = _prepare_market_data(df, cfg)
    n_samples = len(prepared.labels)

    n_val = _validation_size(n_samples, cfg)
    train_end = n_samples - n_val
    if train_end < cfg.min_train_sequences:
        raise ValueError(
            "Not enough training sequences after neutral-band filtering: "
            f"{train_end}. Need at least {cfg.min_train_sequences}."
        )

    cv_result = None
    if cfg.use_timeseries_cv and n_samples >= cfg.min_train_sequences + 100:
        cv_result = _timeseries_cv_threshold(
            prepared=prepared,
            config=cfg,
        )

    scaler = StandardScaler()
    scaler.fit(prepared.labeled_sequences[:train_end].reshape(-1, len(prepared.feature_columns)))
    x_all = _scale_sequences(prepared.labeled_sequences, scaler)
    x_latest = _scale_sequences(prepared.latest_sequence[None, :, :], scaler)
    y_all = prepared.labels.astype(np.float32)
    device = _resolve_device(cfg.device)

    model = _train_classifier(
        x_train=x_all[:train_end],
        y_train=y_all[:train_end],
        x_val=x_all[train_end:],
        y_val=y_all[train_end:],
        input_size=len(prepared.feature_columns),
        config=cfg,
        device=device,
    )

    val_probs = _predict_proba(model, x_all[train_end:], device, cfg.batch_size)
    threshold = cv_result[0] if cv_result is not None else _tune_threshold(
        val_probs,
        y_all[train_end:],
        cfg,
    )
    validation_metrics = _classification_metrics(val_probs, y_all[train_end:], threshold)
    if cv_result is not None:
        _, cv_metrics = cv_result
        validation_metrics.update({f"cv_{key}": value for key, value in cv_metrics.items()})
    return_stats = _return_stats(prepared.next_returns, prepared.labels)
    latest_probability = float(_predict_proba(model, x_latest, device, cfg.batch_size)[0])

    return FittedDirectionModel(
        model=model,
        scaler=scaler,
        config=cfg,
        feature_columns=prepared.feature_columns,
        threshold=threshold,
        validation_metrics=validation_metrics,
        return_stats=return_stats,
        latest_sequence_scaled=x_latest,
        latest_date=prepared.latest_date,
        latest_close=prepared.latest_close,
        probability_up=latest_probability,
        device=device,
        original_rows=prepared.original_rows,
        cleaned_rows=prepared.cleaned_rows,
        labeled_sequences=n_samples,
        train_sequences=train_end,
        validation_sequences=n_samples - train_end,
    )


def walk_forward_validate(
    df: pd.DataFrame,
    config: DirectionPredictionConfig | None = None,
    *,
    initial_train_fraction: float = 0.65,
    test_size: int = 120,
    step_size: int | None = None,
    max_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate the direction model with expanding-window walk-forward splits.

    Each fold uses an expanding historical pool. The tail of that pool is used
    to tune the classification threshold, and the next chronological segment is
    used as the out-of-sample test window.
    """

    cfg = config or DirectionPredictionConfig()
    _validate_config(cfg)
    if not 0.2 <= initial_train_fraction < 0.95:
        raise ValueError("initial_train_fraction must be in [0.2, 0.95).")
    if test_size < 10:
        raise ValueError("test_size must be at least 10.")
    if max_splits < 1:
        raise ValueError("max_splits must be positive.")

    _set_random_seed(cfg.random_seed)
    prepared = _prepare_market_data(df, cfg)
    n_samples = len(prepared.labels)
    fold_step = step_size or test_size
    first_test_start = max(
        cfg.min_train_sequences + 20,
        int(round(n_samples * initial_train_fraction)),
    )
    if first_test_start >= n_samples - 1:
        raise ValueError("Not enough samples for walk-forward validation.")

    device = _resolve_device(cfg.device)
    rows: list[dict[str, Any]] = []
    test_start = first_test_start
    split = 0
    while test_start < n_samples and split < max_splits:
        test_end = min(test_start + test_size, n_samples)
        if test_end - test_start < 10:
            break

        n_val = _validation_size(test_start, cfg)
        train_end = test_start - n_val
        if train_end < cfg.min_train_sequences:
            break

        scaler = StandardScaler()
        scaler.fit(
            prepared.labeled_sequences[:train_end].reshape(
                -1, len(prepared.feature_columns)
            )
        )
        x_all = _scale_sequences(prepared.labeled_sequences[:test_end], scaler)
        y_all = prepared.labels[:test_end].astype(np.float32)

        model = _train_classifier(
            x_train=x_all[:train_end],
            y_train=y_all[:train_end],
            x_val=x_all[train_end:test_start],
            y_val=y_all[train_end:test_start],
            input_size=len(prepared.feature_columns),
            config=cfg,
            device=device,
        )
        val_probs = _predict_proba(
            model,
            x_all[train_end:test_start],
            device,
            cfg.batch_size,
        )
        threshold = _tune_threshold(val_probs, y_all[train_end:test_start], cfg)
        test_probs = _predict_proba(
            model,
            x_all[test_start:test_end],
            device,
            cfg.batch_size,
        )
        metrics = _classification_metrics(
            test_probs,
            y_all[test_start:test_end],
            threshold,
        )

        row = {
            "split": split + 1,
            "train_sequences": int(train_end),
            "validation_sequences": int(test_start - train_end),
            "test_sequences": int(test_end - test_start),
            "test_start_date": prepared.sample_end_dates.iloc[test_start].strftime(
                "%Y-%m-%d"
            ),
            "test_end_date": prepared.sample_end_dates.iloc[test_end - 1].strftime(
                "%Y-%m-%d"
            ),
        }
        row.update(metrics)
        rows.append(row)

        split += 1
        test_start += fold_step

    if not rows:
        raise ValueError("No walk-forward folds could be created.")
    return pd.DataFrame(rows)


def loop_validate_prediction_results(
    df: pd.DataFrame,
    config: DirectionPredictionConfig | None = None,
    *,
    start_date: str | int | pd.Timestamp | None = None,
    end_date: str | int | pd.Timestamp | None = None,
    periods: int = 60,
    output_path: str | None = None,
    diagnostics_output_path: str | None = None,
    confidence_output_path: str | None = None,
    confidence_summary_path: str | None = None,
    progress: bool = True,
    signal_engine: Literal[
        "bilstm",
        "volatility_rule",
        "state_veto_rule",
        "stability_rule",
        "nested_ml",
        "hybrid",
        "calibrated_rule",
        "historical_selector",
    ] = "bilstm",
    rule_threshold_end_date: str | int | pd.Timestamp | None = None,
    rule_calibration_start_date: str | int | pd.Timestamp | None = None,
    rule_calibration_end_date: str | int | pd.Timestamp | None = None,
    rule_top_k: int = 5,
    volatility_rule_quantile: float = 0.2,
    nested_rule_threshold_window: int = 756,
    nested_rule_calibration_window: int = 252,
    nested_rule_min_threshold_rows: int = 120,
    nested_rule_min_calibration_rows: int = 60,
    state_veto_window: int = 120,
    state_veto_state_window: int = 252,
    state_veto_min_rows: int = 30,
    state_veto_bad_accuracy: float = 0.38,
    state_veto_quantiles: tuple[float, ...] = (0.2, 0.4),
    high_confidence_min_base_calibration: float = (
        SCRIPT_HIGH_CONFIDENCE_MIN_BASE_CALIBRATION
    ),
    high_confidence_min_veto_state_accuracy: float = (
        SCRIPT_HIGH_CONFIDENCE_MIN_VETO_STATE_ACCURACY
    ),
    high_confidence_max_veto_state_rows: int | None = 45,
    stability_rule_long_window: int = 756,
    stability_rule_recent_weight: float = 0.65,
    stability_rule_min_edge: float = 0.005,
    stability_rule_top_k: int = 3,
    nested_ml_train_window: int = 1008,
    nested_ml_step: int = 5,
    nested_ml_calibration_window: int = 120,
    nested_ml_min_train_rows: int = 252,
    nested_ml_top_k: int = 3,
    return_magnitude_mode: ReturnMagnitudeMode = SCRIPT_RETURN_MAGNITUDE_MODE,
    return_magnitude_window: int = SCRIPT_RETURN_MAGNITUDE_WINDOW,
    return_magnitude_min_rows: int = SCRIPT_RETURN_MAGNITUDE_MIN_ROWS,
    return_magnitude_grid_size: int = SCRIPT_RETURN_MAGNITUDE_GRID_SIZE,
    return_magnitude_clip_low_quantile: float = (
        SCRIPT_RETURN_MAGNITUDE_CLIP_LOW_QUANTILE
    ),
    return_magnitude_clip_high_quantile: float = (
        SCRIPT_RETURN_MAGNITUDE_CLIP_HIGH_QUANTILE
    ),
    recent_failure_guard: bool = SCRIPT_RECENT_FAILURE_GUARD,
    recent_failure_window: int = SCRIPT_RECENT_FAILURE_WINDOW,
    recent_failure_degrade_threshold: float = (
        SCRIPT_RECENT_FAILURE_DEGRADE_THRESHOLD
    ),
    recent_failure_invert_threshold: float = SCRIPT_RECENT_FAILURE_INVERT_THRESHOLD,
    recent_failure_short_window: int = SCRIPT_RECENT_FAILURE_SHORT_WINDOW,
    recent_failure_short_threshold: float = SCRIPT_RECENT_FAILURE_SHORT_THRESHOLD,
    selector_window: int = SCRIPT_SELECTOR_WINDOW,
    selector_min_history: int = SCRIPT_SELECTOR_MIN_HISTORY,
    selector_switch_edge: float = SCRIPT_SELECTOR_SWITCH_EDGE,
    selector_disagreement_window: int = SCRIPT_SELECTOR_DISAGREEMENT_WINDOW,
    selector_disagreement_min_history: int = SCRIPT_SELECTOR_DISAGREEMENT_MIN_HISTORY,
    selector_disagreement_edge: float = SCRIPT_SELECTOR_DISAGREEMENT_EDGE,
    regime_postprocess: bool = SCRIPT_REGIME_POSTPROCESS,
    regime_postprocess_diagnostics_output_path: str | None = (
        SCRIPT_REGIME_POSTPROCESS_DIAGNOSTICS_OUTPUT_PATH
    ),
    regime_postprocess_state_columns: tuple[str, ...] = tuple(
        item.strip()
        for item in SCRIPT_REGIME_POSTPROCESS_STATE_COLUMNS.split(",")
        if item.strip()
    ),
    regime_postprocess_history_window: int = SCRIPT_REGIME_POSTPROCESS_HISTORY_WINDOW,
    regime_postprocess_min_history: int = SCRIPT_REGIME_POSTPROCESS_MIN_HISTORY,
    regime_postprocess_flip_below: float = SCRIPT_REGIME_POSTPROCESS_FLIP_BELOW,
    regime_postprocess_max_flip_rate: float = SCRIPT_REGIME_POSTPROCESS_MAX_FLIP_RATE,
) -> pd.DataFrame:
    """Loop over trading days and output columns aligned to prediction_results.csv.

    Output columns are exactly:

    trade_date,predicted_pct_change,predicted_close,real_pct_change,correct

    For each output row, the model is trained only on rows up to and including
    ``trade_date``. The row's ``real_pct_change`` is the next trading day's
    realized close-to-close return, and ``correct`` checks only direction sign.
    """

    cfg = config or DirectionPredictionConfig()
    _validate_config(cfg)
    base = _normalize_market_frame(df, cfg)
    if len(base) < cfg.lookback + cfg.min_train_sequences + 2:
        raise ValueError("Not enough rows for loop validation.")

    first_candidate = cfg.lookback + cfg.min_train_sequences
    last_candidate = len(base) - 2
    if start_date is not None:
        start_ts = _parse_single_date(start_date, cfg.dayfirst)
        matching = np.flatnonzero(base["date"].ge(start_ts).to_numpy())
        if len(matching) == 0:
            raise ValueError(f"start_date {start_date!r} is after the data end.")
        first_candidate = max(first_candidate, int(matching[0]))
    if end_date is not None:
        end_ts = _parse_single_date(end_date, cfg.dayfirst)
        matching = np.flatnonzero(base["date"].le(end_ts).to_numpy())
        if len(matching) == 0:
            raise ValueError(f"end_date {end_date!r} is before the data start.")
        last_candidate = min(last_candidate, int(matching[-1]))
    if start_date is None and periods > 0:
        first_candidate = max(first_candidate, last_candidate - periods + 1)

    if first_candidate > last_candidate:
        raise ValueError("No validation dates remain after applying filters.")

    if signal_engine not in {
        "bilstm",
        "volatility_rule",
        "state_veto_rule",
        "stability_rule",
        "nested_ml",
        "hybrid",
        "calibrated_rule",
        "historical_selector",
    }:
        raise ValueError(
            "signal_engine must be 'bilstm', 'volatility_rule', 'state_veto_rule', "
            "'stability_rule', 'nested_ml', 'hybrid', 'calibrated_rule', or "
            "'historical_selector'."
        )
    if return_magnitude_mode not in {"directional_median", "range_scaled"}:
        raise ValueError(
            "return_magnitude_mode must be 'directional_median' or 'range_scaled'."
        )

    feature_frame: pd.DataFrame | None = None
    nested_rule_cache: NestedRuleCache | None = None
    nested_ml_cache: NestedMLCache | None = None
    if signal_engine in {
        "volatility_rule",
        "state_veto_rule",
        "stability_rule",
        "nested_ml",
        "hybrid",
        "calibrated_rule",
        "historical_selector",
    }:
        feature_frame = _clean_feature_frame(_build_features(base, cfg))
    if signal_engine in {
        "volatility_rule",
        "state_veto_rule",
        "stability_rule",
        "hybrid",
        "historical_selector",
    }:
        assert feature_frame is not None
        nested_rule_cache = _build_nested_rule_cache(
            base=base,
            features=feature_frame,
            threshold_window=nested_rule_threshold_window,
            min_threshold_rows=nested_rule_min_threshold_rows,
        )
    if signal_engine in {"nested_ml", "hybrid"}:
        assert feature_frame is not None
        nested_ml_cache = _build_nested_ml_cache(
            base=base,
            features=feature_frame,
            train_window=nested_ml_train_window,
            step=nested_ml_step,
            min_train_rows=nested_ml_min_train_rows,
        )

    rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    raw_direction_history: list[bool] = []
    selector_history: list[dict[str, Any]] = []
    state_veto_base_signals: dict[int, RuleSignal] | None = None
    state_veto_masks: list[tuple[str, np.ndarray]] | None = None
    if signal_engine in {"state_veto_rule", "historical_selector"}:
        assert nested_rule_cache is not None
        assert feature_frame is not None
        state_veto_base_signals = {}
        pre_start = max(
            cfg.lookback + cfg.min_train_sequences,
            first_candidate - max(state_veto_window, state_veto_state_window, 600),
        )
        for signal_idx in range(pre_start, last_candidate + 1):
            state_veto_base_signals[signal_idx] = _nested_volatility_rule_signal(
                base=base,
                cache=nested_rule_cache,
                idx=signal_idx,
                calibration_window=nested_rule_calibration_window,
                min_calibration_rows=nested_rule_min_calibration_rows,
                top_k=rule_top_k,
            )
        state_veto_masks = _build_state_veto_masks(
            features=feature_frame,
            state_window=state_veto_state_window,
            quantiles=state_veto_quantiles,
        )

    def _compute_loop_signal(
        idx: int,
    ) -> tuple[RuleSignal | None, float, dict[str, Any], dict[str, RuleSignal] | None]:
        signal: RuleSignal | None = None
        selector_diagnostics: dict[str, Any] = {}
        selector_candidate_signals: dict[str, RuleSignal] | None = None
        if signal_engine == "volatility_rule":
            assert feature_frame is not None
            assert nested_rule_cache is not None
            signal = _nested_volatility_rule_signal(
                base=base,
                cache=nested_rule_cache,
                idx=idx,
                calibration_window=nested_rule_calibration_window,
                min_calibration_rows=nested_rule_min_calibration_rows,
                top_k=rule_top_k,
            )
            predicted_pct_change = signal.predicted_return
        elif signal_engine == "state_veto_rule":
            assert nested_rule_cache is not None
            assert state_veto_base_signals is not None
            assert state_veto_masks is not None
            signal = _state_veto_rule_signal(
                base=base,
                cache=nested_rule_cache,
                idx=idx,
                base_signals=state_veto_base_signals,
                state_masks=state_veto_masks,
                veto_window=state_veto_window,
                min_rows=state_veto_min_rows,
                bad_accuracy=state_veto_bad_accuracy,
            )
            predicted_pct_change = signal.predicted_return
        elif signal_engine == "historical_selector":
            assert nested_rule_cache is not None
            assert state_veto_base_signals is not None
            assert state_veto_masks is not None
            volatility_signal = state_veto_base_signals[idx]
            state_veto_signal = _state_veto_rule_signal(
                base=base,
                cache=nested_rule_cache,
                idx=idx,
                base_signals=state_veto_base_signals,
                state_masks=state_veto_masks,
                veto_window=state_veto_window,
                min_rows=state_veto_min_rows,
                bad_accuracy=state_veto_bad_accuracy,
            )
            selected_engine, selector_diagnostics = _historical_selector_decision(
                history=selector_history,
                selector_window=selector_window,
                min_history=selector_min_history,
                switch_edge=selector_switch_edge,
                disagreement_window=selector_disagreement_window,
                disagreement_min_history=selector_disagreement_min_history,
                disagreement_edge=selector_disagreement_edge,
            )
            selector_candidate_signals = {
                "state_veto_rule": state_veto_signal,
                "volatility_rule": volatility_signal,
            }
            signal = selector_candidate_signals[selected_engine]
            predicted_pct_change = signal.predicted_return
        elif signal_engine == "stability_rule":
            assert feature_frame is not None
            assert nested_rule_cache is not None
            signal = _stability_weighted_rule_signal(
                base=base,
                cache=nested_rule_cache,
                idx=idx,
                recent_window=nested_rule_calibration_window,
                long_window=stability_rule_long_window,
                min_calibration_rows=nested_rule_min_calibration_rows,
                top_k=stability_rule_top_k,
                recent_weight=stability_rule_recent_weight,
                min_edge=stability_rule_min_edge,
            )
            predicted_pct_change = signal.predicted_return
        elif signal_engine == "calibrated_rule":
            assert feature_frame is not None
            signal = _calibrated_rule_signal(
                base=base,
                features=feature_frame,
                idx=idx,
                config=cfg,
                threshold_end_date=rule_threshold_end_date,
                calibration_start_date=rule_calibration_start_date,
                calibration_end_date=rule_calibration_end_date,
                top_k=rule_top_k,
            )
            predicted_pct_change = signal.predicted_return
        elif signal_engine == "nested_ml":
            assert nested_ml_cache is not None
            signal = _nested_ml_signal(
                base=base,
                cache=nested_ml_cache,
                idx=idx,
                calibration_window=nested_ml_calibration_window,
                top_k=nested_ml_top_k,
            )
            predicted_pct_change = signal.predicted_return
        elif signal_engine == "hybrid":
            assert nested_rule_cache is not None
            assert nested_ml_cache is not None
            signal = _hybrid_signal(
                base=base,
                rule_cache=nested_rule_cache,
                ml_cache=nested_ml_cache,
                idx=idx,
                rule_calibration_window=nested_rule_calibration_window,
                rule_min_calibration_rows=nested_rule_min_calibration_rows,
                rule_top_k=rule_top_k,
                ml_calibration_window=nested_ml_calibration_window,
                ml_top_k=nested_ml_top_k,
            )
            predicted_pct_change = signal.predicted_return
        else:
            train_frame = base.iloc[: idx + 1].copy()
            fitted = fit_direction_model(train_frame, config=cfg)
            result = prediction_to_dict(fitted)
            predicted_pct_change = float(result["estimated_next_return"])
        return (
            signal,
            float(predicted_pct_change),
            selector_diagnostics,
            selector_candidate_signals,
        )

    def _record_selector_history(
        *,
        idx: int,
        selector_candidate_signals: dict[str, RuleSignal] | None,
        real_pct_change: float,
    ) -> None:
        if selector_candidate_signals is None:
            return
        selector_history.append(
            {
                "trade_date": int(base["date"].iloc[idx].strftime("%Y%m%d")),
                "state_veto_rule_predicted_pct_change": float(
                    selector_candidate_signals["state_veto_rule"].predicted_return
                ),
                "state_veto_rule_correct": bool(
                    _direction_sign(
                        selector_candidate_signals[
                            "state_veto_rule"
                        ].predicted_return
                    )
                    == _direction_sign(real_pct_change)
                ),
                "volatility_rule_predicted_pct_change": float(
                    selector_candidate_signals["volatility_rule"].predicted_return
                ),
                "volatility_rule_correct": bool(
                    _direction_sign(
                        selector_candidate_signals[
                            "volatility_rule"
                        ].predicted_return
                    )
                    == _direction_sign(real_pct_change)
                ),
            }
        )

    if recent_failure_guard:
        warmup_span = int(max(1, recent_failure_window, recent_failure_short_window))
        if signal_engine == "historical_selector":
            warmup_span = max(
                warmup_span,
                int(selector_window),
                int(selector_disagreement_window),
            )
        warmup_start = max(cfg.lookback + cfg.min_train_sequences, first_candidate - warmup_span)
        for warmup_idx in range(warmup_start, first_candidate):
            _, warmup_predicted_pct_change, _, warmup_selector_signals = (
                _compute_loop_signal(warmup_idx)
            )
            warmup_real_pct_change = float(
                base["close"].iloc[warmup_idx + 1]
                / base["close"].iloc[warmup_idx]
                - 1.0
            )
            raw_direction_history.append(
                bool(
                    _direction_sign(warmup_predicted_pct_change)
                    == _direction_sign(warmup_real_pct_change)
                )
            )
            _record_selector_history(
                idx=warmup_idx,
                selector_candidate_signals=warmup_selector_signals,
                real_pct_change=warmup_real_pct_change,
            )

    total = last_candidate - first_candidate + 1
    for offset, idx in enumerate(range(first_candidate, last_candidate + 1), start=1):
        signal, predicted_pct_change, selector_diagnostics, selector_candidate_signals = (
            _compute_loop_signal(idx)
        )
        raw_predicted_pct_change = float(predicted_pct_change)
        predicted_pct_change, magnitude_diagnostics = _calibrate_return_magnitude(
            base=base,
            idx=idx,
            raw_predicted_return=raw_predicted_pct_change,
            mode=return_magnitude_mode,
            window=return_magnitude_window,
            min_rows=return_magnitude_min_rows,
            grid_size=return_magnitude_grid_size,
            clip_low_quantile=return_magnitude_clip_low_quantile,
            clip_high_quantile=return_magnitude_clip_high_quantile,
        )
        pre_guard_predicted_pct_change = float(predicted_pct_change)
        predicted_pct_change, guard_diagnostics = _apply_recent_failure_guard(
            predicted_return=pre_guard_predicted_pct_change,
            raw_direction_history=raw_direction_history,
            enabled=recent_failure_guard,
            window=recent_failure_window,
            degrade_threshold=recent_failure_degrade_threshold,
            invert_threshold=recent_failure_invert_threshold,
            short_window=recent_failure_short_window,
            short_threshold=recent_failure_short_threshold,
        )
        real_pct_change = float(base["close"].iloc[idx + 1] / base["close"].iloc[idx] - 1.0)
        predicted_close = float(base["close"].iloc[idx] * (1.0 + predicted_pct_change))
        raw_correct = bool(
            _direction_sign(pre_guard_predicted_pct_change)
            == _direction_sign(real_pct_change)
        )
        correct = bool(
            _direction_sign(predicted_pct_change) == _direction_sign(real_pct_change)
        )
        raw_direction_history.append(raw_correct)
        _record_selector_history(
            idx=idx,
            selector_candidate_signals=selector_candidate_signals,
            real_pct_change=real_pct_change,
        )
        rows.append(
            {
                "trade_date": int(base["date"].iloc[idx].strftime("%Y%m%d")),
                "predicted_pct_change": predicted_pct_change,
                "predicted_close": predicted_close,
                "real_pct_change": real_pct_change,
                "correct": correct,
            }
        )
        if diagnostics_output_path or regime_postprocess:
            real_label = int(real_pct_change > 0)
            diagnostics = dict(signal.diagnostics) if signal is not None else {}
            diagnostics.update(selector_diagnostics)
            diagnostics.update(magnitude_diagnostics)
            diagnostics.update(guard_diagnostics)
            selected_rules = signal.rule_names if signal is not None else []
            diagnostic_rows.append(
                {
                    "trade_date": int(base["date"].iloc[idx].strftime("%Y%m%d")),
                    "signal_engine": signal_engine,
                    "predicted_label": (
                        int(signal.predicted_label)
                        if signal is not None
                        else int(predicted_pct_change > 0)
                    ),
                    "real_label": real_label,
                    "raw_predicted_pct_change": raw_predicted_pct_change,
                    "pre_guard_predicted_pct_change": pre_guard_predicted_pct_change,
                    "predicted_pct_change": predicted_pct_change,
                    "real_pct_change": real_pct_change,
                    "raw_correct": raw_correct,
                    "correct": correct,
                    "selected_rules": " | ".join(selected_rules),
                    "calibration_accuracy": (
                        float(signal.calibration_accuracy)
                        if signal is not None
                        else np.nan
                    ),
                    "calibration_rows": (
                        int(signal.calibration_rows) if signal is not None else 0
                    ),
                    **diagnostics,
                }
            )
        if progress:
            accuracy = float(np.mean([row["correct"] for row in rows]))
            print(
                f"[{signal_engine}] [{offset}/{total}] {rows[-1]['trade_date']} "
                f"correct={correct} running_accuracy={accuracy:.4f}",
                file=sys.stderr,
                flush=True,
            )

    result_frame = pd.DataFrame(
        rows,
        columns=[
            "trade_date",
            "predicted_pct_change",
            "predicted_close",
            "real_pct_change",
            "correct",
        ],
    )
    if diagnostic_rows:
        diagnostics_frame = pd.DataFrame(diagnostic_rows)
    else:
        diagnostics_frame = pd.DataFrame()

    if regime_postprocess:
        regime_cfg = replace(cfg, external_feature_mode="all")
        regime_base = _normalize_market_frame(df, regime_cfg)
        regime_source = _build_regime_postprocess_frame(
            result_frame=result_frame,
            base=regime_base,
            diagnostics_frame=diagnostics_frame,
        )
        result_frame, postprocess_diagnostics = _apply_regime_postprocess(
            regime_source,
            state_columns=regime_postprocess_state_columns,
            history_window=regime_postprocess_history_window,
            min_history=regime_postprocess_min_history,
            flip_below=regime_postprocess_flip_below,
            max_flip_rate=regime_postprocess_max_flip_rate,
        )
        if regime_postprocess_diagnostics_output_path:
            postprocess_diagnostics.to_csv(
                regime_postprocess_diagnostics_output_path,
                index=False,
                encoding="utf-8-sig",
            )
        postprocess_extra = postprocess_diagnostics.drop(
            columns=[
                "original_correct",
                "postprocess_correct",
            ],
            errors="ignore",
        )
        if diagnostics_frame.empty:
            diagnostics_frame = postprocess_extra.copy()
        else:
            diagnostics_frame = diagnostics_frame.merge(
                postprocess_extra,
                on="trade_date",
                how="left",
            )

    if output_path:
        result_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    if diagnostics_output_path:
        diagnostics_frame.to_csv(
            diagnostics_output_path,
            index=False,
            encoding="utf-8-sig",
        )
    if confidence_output_path or confidence_summary_path:
        if diagnostics_frame.empty:
            diagnostics_frame = _build_minimal_diagnostics(result_frame)
        high_confidence_frame, confidence_summary = _build_high_confidence_outputs(
            result_frame=result_frame,
            diagnostics_frame=diagnostics_frame,
            min_base_calibration=high_confidence_min_base_calibration,
            min_veto_state_accuracy=high_confidence_min_veto_state_accuracy,
            max_veto_state_rows=high_confidence_max_veto_state_rows,
        )
        if confidence_output_path:
            high_confidence_frame.to_csv(
                confidence_output_path,
                index=False,
                encoding="utf-8-sig",
            )
        if confidence_summary_path:
            confidence_summary.to_csv(
                confidence_summary_path,
                index=False,
                encoding="utf-8-sig",
            )
    return result_frame


def _build_minimal_diagnostics(result_frame: pd.DataFrame) -> pd.DataFrame:
    diagnostics = result_frame.copy()
    diagnostics["predicted_label"] = (
        diagnostics["predicted_pct_change"] > 0
    ).astype(int)
    diagnostics["real_label"] = (diagnostics["real_pct_change"] > 0).astype(int)
    diagnostics["signal_engine"] = ""
    diagnostics["rule_mode"] = ""
    return diagnostics


def _build_high_confidence_outputs(
    *,
    result_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    min_base_calibration: float,
    min_veto_state_accuracy: float,
    max_veto_state_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_columns = [
        "trade_date",
        "predicted_pct_change",
        "predicted_close",
        "real_pct_change",
        "correct",
    ]
    base = result_frame[result_columns].copy()
    diagnostics = diagnostics_frame.copy()
    if "trade_date" not in diagnostics.columns:
        empty = base.iloc[0:0].copy()
        return empty, _confidence_summary_frame(base, empty, "missing_diagnostics")

    diagnostic_extra = diagnostics.drop(
        columns=[col for col in result_columns[1:] if col in diagnostics.columns],
        errors="ignore",
    )
    merged = base.merge(diagnostic_extra, on="trade_date", how="left")
    veto_applied = pd.to_numeric(
        merged.get("veto_applied", pd.Series(0, index=merged.index)),
        errors="coerce",
    ).fillna(0)
    base_calibration = pd.to_numeric(
        merged.get("base_calibration_accuracy", pd.Series(np.nan, index=merged.index)),
        errors="coerce",
    )
    veto_state_accuracy = pd.to_numeric(
        merged.get("veto_state_accuracy", pd.Series(np.nan, index=merged.index)),
        errors="coerce",
    )
    veto_state_rows = pd.to_numeric(
        merged.get("veto_state_rows", pd.Series(np.nan, index=merged.index)),
        errors="coerce",
    )

    high_confidence_mask = (
        veto_applied.eq(1)
        & base_calibration.ge(min_base_calibration)
        & veto_state_accuracy.ge(min_veto_state_accuracy)
    )
    if max_veto_state_rows is not None:
        high_confidence_mask &= veto_state_rows.le(max_veto_state_rows)
    high_confidence = merged.loc[high_confidence_mask].copy()
    confidence_rule = (
        f"veto_applied=1 & base_calibration_accuracy>={min_base_calibration:.3f} "
        f"& veto_state_accuracy>={min_veto_state_accuracy:.3f}"
    )
    if max_veto_state_rows is not None:
        confidence_rule += f" & veto_state_rows<={max_veto_state_rows}"
    if not high_confidence.empty:
        high_confidence["confidence_tag"] = "veto_confirmed"
        high_confidence["confidence_rule"] = confidence_rule
    summary = _confidence_summary_frame(base, high_confidence, confidence_rule)
    return high_confidence, summary


def _confidence_summary_frame(
    result_frame: pd.DataFrame,
    high_confidence_frame: pd.DataFrame,
    confidence_rule: str,
) -> pd.DataFrame:
    total_rows = len(result_frame)
    high_rows = len(high_confidence_frame)
    all_accuracy = float(result_frame["correct"].mean()) if total_rows else np.nan
    high_accuracy = (
        float(high_confidence_frame["correct"].mean()) if high_rows else np.nan
    )
    return pd.DataFrame(
        [
            {
                "scope": "all_days",
                "rows": total_rows,
                "coverage": 1.0 if total_rows else np.nan,
                "correct": int(result_frame["correct"].sum()) if total_rows else 0,
                "direction_accuracy": all_accuracy,
                **_direction_side_stats(result_frame),
                "confidence_rule": "",
            },
            {
                "scope": "high_confidence",
                "rows": high_rows,
                "coverage": float(high_rows / total_rows) if total_rows else np.nan,
                "correct": int(high_confidence_frame["correct"].sum())
                if high_rows
                else 0,
                "direction_accuracy": high_accuracy,
                **_direction_side_stats(high_confidence_frame),
                "confidence_rule": confidence_rule,
            },
        ]
    )


def _direction_side_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    """Return long/short hit-rate statistics from a prediction result frame."""

    if frame.empty or "predicted_pct_change" not in frame or "correct" not in frame:
        return {
            "long_rows": 0,
            "long_correct": 0,
            "long_accuracy": np.nan,
            "short_rows": 0,
            "short_correct": 0,
            "short_accuracy": np.nan,
        }
    predicted = pd.to_numeric(frame["predicted_pct_change"], errors="coerce")
    correct = frame["correct"].astype(bool)
    long_mask = predicted > 0
    short_mask = predicted < 0
    long_rows = int(long_mask.sum())
    short_rows = int(short_mask.sum())
    long_correct = int(correct[long_mask].sum()) if long_rows else 0
    short_correct = int(correct[short_mask].sum()) if short_rows else 0
    return {
        "long_rows": long_rows,
        "long_correct": long_correct,
        "long_accuracy": float(long_correct / long_rows) if long_rows else np.nan,
        "short_rows": short_rows,
        "short_correct": short_correct,
        "short_accuracy": float(short_correct / short_rows) if short_rows else np.nan,
    }


def _historical_selector_decision(
    *,
    history: list[dict[str, Any]],
    selector_window: int,
    min_history: int,
    switch_edge: float,
    disagreement_window: int,
    disagreement_min_history: int,
    disagreement_edge: float,
) -> tuple[str, dict[str, Any]]:
    baseline_engine = "state_veto_rule"
    candidate_engine = "volatility_rule"
    diagnostic = {
        "selected_engine": baseline_engine,
        "candidate_before_guard": baseline_engine,
        "selector_window": selector_window,
        "selector_min_history": min_history,
        "selector_switch_edge": switch_edge,
        "disagreement_window": disagreement_window,
        "disagreement_min_history": disagreement_min_history,
        "disagreement_edge": disagreement_edge,
        "selector_veto_reason": "",
        "state_veto_rule_hist_accuracy": np.nan,
        "state_veto_rule_hist_rows": 0,
        "volatility_rule_hist_accuracy": np.nan,
        "volatility_rule_hist_rows": 0,
        "disagreement_rows": 0,
        "candidate_disagreement_accuracy": np.nan,
        "baseline_disagreement_accuracy": np.nan,
        "candidate_disagreement_edge": np.nan,
    }
    if not history:
        return baseline_engine, diagnostic

    hist = pd.DataFrame(history).tail(selector_window)
    baseline_correct = hist[f"{baseline_engine}_correct"].dropna().astype(bool)
    candidate_correct = hist[f"{candidate_engine}_correct"].dropna().astype(bool)
    diagnostic["state_veto_rule_hist_rows"] = int(len(baseline_correct))
    diagnostic["volatility_rule_hist_rows"] = int(len(candidate_correct))
    if len(baseline_correct):
        diagnostic["state_veto_rule_hist_accuracy"] = float(baseline_correct.mean())
    if len(candidate_correct):
        diagnostic["volatility_rule_hist_accuracy"] = float(candidate_correct.mean())
    if len(baseline_correct) < min_history or len(candidate_correct) < min_history:
        return baseline_engine, diagnostic

    baseline_accuracy = float(baseline_correct.mean())
    candidate_accuracy = float(candidate_correct.mean())
    selected_engine = baseline_engine
    if candidate_accuracy > baseline_accuracy + switch_edge:
        selected_engine = candidate_engine
    diagnostic["candidate_before_guard"] = selected_engine

    if selected_engine == candidate_engine and disagreement_window > 0:
        disagree_hist = pd.DataFrame(history).tail(disagreement_window)
        baseline_direction = disagree_hist[
            f"{baseline_engine}_predicted_pct_change"
        ].astype(float).gt(0)
        candidate_direction = disagree_hist[
            f"{candidate_engine}_predicted_pct_change"
        ].astype(float).gt(0)
        disagreement = disagree_hist.loc[baseline_direction.ne(candidate_direction)]
        diagnostic["disagreement_rows"] = int(len(disagreement))
        if len(disagreement):
            candidate_disagreement_accuracy = float(
                disagreement[f"{candidate_engine}_correct"].astype(bool).mean()
            )
            baseline_disagreement_accuracy = float(
                disagreement[f"{baseline_engine}_correct"].astype(bool).mean()
            )
            diagnostic["candidate_disagreement_accuracy"] = (
                candidate_disagreement_accuracy
            )
            diagnostic["baseline_disagreement_accuracy"] = baseline_disagreement_accuracy
            diagnostic["candidate_disagreement_edge"] = (
                candidate_disagreement_accuracy - baseline_disagreement_accuracy
            )
        if len(disagreement) < disagreement_min_history:
            selected_engine = baseline_engine
            diagnostic["selector_veto_reason"] = "disagreement_not_enough_history"
        elif (
            diagnostic["candidate_disagreement_accuracy"]
            < diagnostic["baseline_disagreement_accuracy"] + disagreement_edge
        ):
            selected_engine = baseline_engine
            diagnostic["selector_veto_reason"] = "disagreement_edge_too_small"

    diagnostic["selected_engine"] = selected_engine
    return selected_engine, diagnostic


def _build_regime_postprocess_frame(
    *,
    result_frame: pd.DataFrame,
    base: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
) -> pd.DataFrame:
    context = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(base["date"]).dt.strftime("%Y%m%d").astype(int),
            "open": base["open"].astype(float),
            "high": base["high"].astype(float),
            "low": base["low"].astype(float),
            "close": base["close"].astype(float),
            "pre_close": base["pre_close"].astype(float),
            "volume": base["volume"].astype(float),
            "amount": base["amount"].astype(float),
        }
    )
    external_cols = [column for column in base.columns if column.startswith("ext_")]
    if external_cols:
        context = pd.concat([context, base[external_cols].reset_index(drop=True)], axis=1)

    result = result_frame.copy()
    result["trade_date"] = pd.to_numeric(result["trade_date"], errors="raise").astype(int)
    merged = result.merge(context, on="trade_date", how="left")
    if not diagnostics_frame.empty and "trade_date" in diagnostics_frame.columns:
        diagnostic_extra = diagnostics_frame.drop(
            columns=[
                col
                for col in [
                    "predicted_pct_change",
                    "predicted_close",
                    "real_pct_change",
                    "correct",
                ]
                if col in diagnostics_frame.columns
            ],
            errors="ignore",
        ).copy()
        diagnostic_extra["trade_date"] = pd.to_numeric(
            diagnostic_extra["trade_date"],
            errors="coerce",
        ).astype("Int64")
        merged = merged.merge(diagnostic_extra, on="trade_date", how="left")

    merged["prediction_side"] = np.where(
        pd.to_numeric(merged["predicted_pct_change"], errors="coerce").gt(0),
        "long",
        "short",
    )
    return _add_regime_postprocess_states(merged)


def _add_regime_postprocess_states(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    pre_close = pd.to_numeric(frame["pre_close"], errors="coerce")
    pct_chg = close / pre_close - 1.0

    frame["sh_intraday_pct"] = close / open_ - 1.0
    frame["sh_close_position"] = ((close - low) / (high - low).replace(0, np.nan)).clip(
        0.0,
        1.0,
    )
    frame["sh_vol20_pct"] = pct_chg.rolling(20, min_periods=10).std()
    frame["vol20_bucket"] = _regime_rolling_bucket(
        frame["sh_vol20_pct"],
        window=252,
        min_rows=60,
        prefix="vol20",
    )
    frame["close_position_zone"] = pd.cut(
        frame["sh_close_position"],
        bins=[-np.inf, 0.33, 0.67, np.inf],
        labels=["close_near_low", "close_middle", "close_near_high"],
    ).astype("object").fillna("missing")
    frame["intraday_direction"] = _regime_sign_state(
        frame["sh_intraday_pct"],
        "intraday_up",
        "intraday_down",
        "intraday_flat",
        flat_band=0.0005,
    )
    frame["streak_state"] = _regime_streak_state(pct_chg)
    _add_regime_alignment_state(
        frame,
        "domestic_index_alignment",
        [
            "ext_csi300_pct_chg",
            "ext_zz500_pct_chg",
            "ext_sz399001_pct_chg",
            "ext_cyb399006_pct_chg",
            "ext_kc50_pct_chg",
        ],
    )
    _add_regime_alignment_state(
        frame,
        "china_futures_alignment",
        [
            "ext_if_main_ret1",
            "ext_ih_main_ret1",
            "ext_ic_main_ret1",
            "ext_im_main_ret1",
        ],
    )
    if {
        "state_veto_rule_hist_accuracy",
        "volatility_rule_hist_accuracy",
    }.issubset(frame.columns):
        frame["selector_hist_edge"] = (
            pd.to_numeric(frame["volatility_rule_hist_accuracy"], errors="coerce")
            - pd.to_numeric(frame["state_veto_rule_hist_accuracy"], errors="coerce")
        )
        frame["selector_hist_edge_bucket"] = pd.cut(
            frame["selector_hist_edge"],
            bins=[-np.inf, -0.02, 0.0, 0.02, np.inf],
            labels=[
                "vol_rule_weaker_gt2pct",
                "vol_rule_slightly_weaker",
                "vol_rule_slightly_stronger",
                "vol_rule_stronger_gt2pct",
            ],
        ).astype("object").fillna("missing")
    else:
        frame["selector_hist_edge_bucket"] = "missing"
    if "selected_engine" not in frame.columns:
        frame["selected_engine"] = frame.get("signal_engine", "missing")
    frame["selected_engine"] = frame["selected_engine"].fillna("missing")
    return frame


def _apply_regime_postprocess(
    frame: pd.DataFrame,
    *,
    state_columns: tuple[str, ...],
    history_window: int,
    min_history: int,
    flip_below: float,
    max_flip_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    flipped_so_far = 0
    for idx, row in frame.reset_index(drop=True).iterrows():
        selected_key = _select_regime_flip_key(
            frame=frame,
            idx=idx,
            state_columns=state_columns,
            history_window=history_window,
            min_history=min_history,
            flip_below=flip_below,
        )
        projected_flip_rate = (flipped_so_far + 1) / (idx + 1)
        flip = bool(selected_key is not None and projected_flip_rate <= max_flip_rate)
        if flip:
            flipped_so_far += 1
        original_pred = float(row["predicted_pct_change"])
        real = float(row["real_pct_change"])
        predicted_pct_change = -original_pred if flip else original_pred
        close = float(row["close"]) if pd.notna(row.get("close", np.nan)) else np.nan
        predicted_close = (
            close * (1.0 + predicted_pct_change)
            if np.isfinite(close)
            else float(row["predicted_close"])
        )
        correct = bool(
            _direction_sign(predicted_pct_change) == _direction_sign(real)
        )
        rows.append(
            {
                "trade_date": int(row["trade_date"]),
                "predicted_pct_change": predicted_pct_change,
                "predicted_close": float(predicted_close),
                "real_pct_change": real,
                "correct": correct,
            }
        )
        original_correct = bool(row["correct"])
        diagnostics.append(
            {
                "trade_date": int(row["trade_date"]),
                "regime_postprocess_enabled": 1,
                "regime_postprocess_flipped": int(flip),
                "original_correct": original_correct,
                "postprocess_correct": correct,
                "original_prediction_side": row["prediction_side"],
                "postprocess_prediction_side": (
                    "long" if predicted_pct_change > 0 else "short"
                ),
                "flip_state_column": selected_key["state_column"] if selected_key else "",
                "flip_state_value": selected_key["state_value"] if selected_key else "",
                "flip_side": selected_key["side"] if selected_key else "",
                "flip_hist_rows": selected_key["rows"] if selected_key else 0,
                "flip_hist_accuracy": selected_key["accuracy"] if selected_key else np.nan,
                "regime_postprocess_history_window": history_window,
                "regime_postprocess_min_history": min_history,
                "regime_postprocess_flip_below": flip_below,
                "regime_postprocess_max_flip_rate": max_flip_rate,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def _select_regime_flip_key(
    *,
    frame: pd.DataFrame,
    idx: int,
    state_columns: tuple[str, ...],
    history_window: int,
    min_history: int,
    flip_below: float,
) -> dict[str, Any] | None:
    if idx <= 0:
        return None
    row = frame.iloc[idx]
    history = frame.iloc[max(0, idx - history_window) : idx]
    side = str(row["prediction_side"])
    candidates: list[dict[str, Any]] = []
    for column in state_columns:
        if column not in frame.columns:
            continue
        value = row[column]
        if pd.isna(value) or str(value) in {"missing", "insufficient_history"}:
            continue
        mask = history[column].astype(str).eq(str(value))
        mask &= history["prediction_side"].astype(str).eq(side)
        sample = history.loc[mask]
        rows = int(len(sample))
        if rows < min_history:
            continue
        accuracy = float(sample["correct"].astype(bool).mean())
        if accuracy <= flip_below:
            candidates.append(
                {
                    "state_column": column,
                    "state_value": str(value),
                    "side": side,
                    "rows": rows,
                    "accuracy": accuracy,
                }
            )
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item["accuracy"], -item["rows"]))[0]


def _regime_rolling_bucket(
    series: pd.Series,
    *,
    window: int,
    min_rows: int,
    prefix: str,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    low = numeric.shift(1).rolling(window, min_periods=min_rows).quantile(0.33)
    high = numeric.shift(1).rolling(window, min_periods=min_rows).quantile(0.67)
    labels = pd.Series(f"{prefix}_mid", index=series.index, dtype="object")
    labels[numeric.le(low)] = f"{prefix}_low"
    labels[numeric.ge(high)] = f"{prefix}_high"
    labels[low.isna() | high.isna()] = "insufficient_history"
    labels[numeric.isna()] = "missing"
    return labels


def _regime_sign_state(
    series: pd.Series,
    positive_label: str,
    negative_label: str,
    flat_label: str,
    *,
    flat_band: float,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(flat_label, index=series.index, dtype="object")
    labels[numeric.gt(flat_band)] = positive_label
    labels[numeric.lt(-flat_band)] = negative_label
    labels[numeric.isna()] = "missing"
    return labels


def _regime_streak_state(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    states: list[str] = []
    direction = 0
    length = 0
    for value in numeric:
        if pd.isna(value) or value == 0:
            direction = 0
            length = 0
            states.append("streak_flat_or_missing")
            continue
        current = 1 if value > 0 else -1
        if current == direction:
            length += 1
        else:
            direction = current
            length = 1
        side = "up" if direction > 0 else "down"
        bucket = "4plus" if length >= 4 else str(length)
        states.append(f"streak_{side}_{bucket}")
    return pd.Series(states, index=series.index, dtype="object")


def _add_regime_alignment_state(
    frame: pd.DataFrame,
    output_col: str,
    source_cols: list[str],
) -> None:
    available = [column for column in source_cols if column in frame.columns]
    if not available:
        frame[output_col] = "missing"
        return
    values = frame[available].apply(pd.to_numeric, errors="coerce")
    valid_count = values.notna().sum(axis=1)
    positive_count = values.gt(0).sum(axis=1)
    negative_count = values.lt(0).sum(axis=1)
    labels = pd.Series("mixed", index=frame.index, dtype="object")
    labels[valid_count.eq(0)] = "missing"
    labels[positive_count.ge(valid_count * 0.67) & valid_count.gt(0)] = (
        "mostly_positive"
    )
    labels[negative_count.ge(valid_count * 0.67) & valid_count.gt(0)] = (
        "mostly_negative"
    )
    labels[positive_count.eq(valid_count) & valid_count.gt(0)] = "all_positive"
    labels[negative_count.eq(valid_count) & valid_count.gt(0)] = "all_negative"
    frame[output_col] = labels


def prediction_to_dict(fitted: FittedDirectionModel) -> dict[str, Any]:
    """Convert a fitted model result into a serializable prediction summary."""

    prob_up = fitted.probability_up
    threshold = fitted.threshold
    direction = "up" if prob_up >= threshold else "down"
    direction_label = 1 if direction == "up" else 0
    margin = prob_up - threshold if direction == "up" else threshold - prob_up
    confidence = float(np.clip(margin / 0.20, 0.0, 1.0))
    estimated_return = _estimate_directional_return(
        direction=direction,
        probability_up=prob_up,
        threshold=threshold,
        return_stats=fitted.return_stats,
    )
    estimated_close = fitted.latest_close * (1.0 + estimated_return)

    return {
        "model": "BiLSTM-Attention direction classifier",
        "last_date": fitted.latest_date.strftime("%Y-%m-%d"),
        "last_close": float(fitted.latest_close),
        "predicted_direction": direction,
        "predicted_label": direction_label,
        "probability_up": float(prob_up),
        "probability_down": float(1.0 - prob_up),
        "decision_threshold": float(threshold),
        "direction_margin": float(margin),
        "confidence": confidence,
        "estimated_next_return": float(estimated_return),
        "estimated_next_pct_change": float(estimated_return * 100.0),
        "estimated_next_close": float(estimated_close),
        "return_estimate_note": (
            "Auxiliary directional-median calibration; not used for the "
            "up/down decision."
        ),
        "validation": fitted.validation_metrics,
        "data": {
            "original_rows": fitted.original_rows,
            "cleaned_rows": fitted.cleaned_rows,
            "dropped_rows": fitted.original_rows - fitted.cleaned_rows,
            "labeled_sequences": fitted.labeled_sequences,
            "train_sequences": fitted.train_sequences,
            "validation_sequences": fitted.validation_sequences,
            "feature_count": len(fitted.feature_columns),
            "lookback": fitted.config.lookback,
            "neutral_band": fitted.config.neutral_band,
        },
        "training": {
            "epochs_configured": fitted.config.epochs,
            "hidden_size": fitted.config.hidden_size,
            "dropout": fitted.config.dropout,
            "device": fitted.device,
        },
    }


def _calibrated_rule_signal(
    *,
    base: pd.DataFrame,
    features: pd.DataFrame,
    idx: int,
    config: DirectionPredictionConfig,
    threshold_end_date: str | int | pd.Timestamp | None,
    calibration_start_date: str | int | pd.Timestamp | None,
    calibration_end_date: str | int | pd.Timestamp | None,
    top_k: int,
) -> RuleSignal:
    dates = base["date"]
    next_returns = base["close"].shift(-1) / base["close"] - 1.0
    labels = (next_returns > 0).astype(int)

    if threshold_end_date is None:
        threshold_end = dates.iloc[max(0, idx - 252)]
    else:
        threshold_end = _parse_single_date(threshold_end_date, config.dayfirst)
    if calibration_end_date is None:
        calibration_end = dates.iloc[max(0, idx - 1)]
    else:
        calibration_end = min(
            _parse_single_date(calibration_end_date, config.dayfirst),
            dates.iloc[max(0, idx - 1)],
        )
    if calibration_start_date is None:
        calibration_start = calibration_end - pd.Timedelta(days=365)
    else:
        calibration_start = _parse_single_date(calibration_start_date, config.dayfirst)

    valid_feature_mask = features.notna().all(axis=1)
    threshold_mask = valid_feature_mask & dates.le(threshold_end)
    calibration_mask = (
        valid_feature_mask
        & next_returns.notna()
        & dates.ge(calibration_start)
        & dates.le(calibration_end)
        & (np.arange(len(base)) < idx)
    )
    calibration_idx = np.flatnonzero(calibration_mask.to_numpy())
    if len(calibration_idx) < 40:
        calibration_idx = np.arange(max(0, idx - 252), idx)
        calibration_idx = calibration_idx[
            valid_feature_mask.iloc[calibration_idx].to_numpy()
            & next_returns.notna().iloc[calibration_idx].to_numpy()
        ]
    if len(calibration_idx) < 20:
        recent = labels.iloc[max(0, idx - 20) : idx]
        predicted_label = int(recent.mean() >= 0.5) if len(recent) else 1
        return _rule_signal_from_label(
            predicted_label=predicted_label,
            base=base,
            idx=idx,
            rule_names=["recent_majority_fallback"],
            calibration_accuracy=0.5,
            calibration_rows=len(recent),
        )

    candidates = _candidate_rule_predictions(features, threshold_mask)
    if not candidates:
        return _rule_signal_from_label(
            predicted_label=1,
            base=base,
            idx=idx,
            rule_names=["always_up_fallback"],
            calibration_accuracy=0.5,
            calibration_rows=len(calibration_idx),
        )

    y_cal = labels.iloc[calibration_idx].to_numpy(dtype=int)
    scored: list[tuple[float, str, np.ndarray]] = []
    for name, prediction in candidates:
        pred_cal = prediction[calibration_idx]
        score = float((pred_cal == y_cal).mean())
        if score < 0.5:
            prediction = 1 - prediction
            score = 1.0 - score
            name = f"NOT({name})"
        scored.append((score, name, prediction))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: max(1, top_k)]
    weights = np.asarray([max(score - 0.5, 0.001) for score, _, _ in selected])
    votes = np.asarray([prediction[idx] for _, _, prediction in selected], dtype=float)
    vote_score = float(np.dot(votes, weights) / weights.sum())
    predicted_label = int(vote_score >= 0.5)
    selected_names = [
        f"{name}:cal_acc={score:.3f}" for score, name, _ in selected
    ]
    selected_matrix = np.vstack([prediction for _, _, prediction in selected])
    selected_votes = (selected_matrix[:, calibration_idx].mean(axis=0) >= 0.5).astype(int)
    calibration_accuracy = float((selected_votes == y_cal).mean())
    return _rule_signal_from_label(
        predicted_label=predicted_label,
        base=base,
        idx=idx,
        rule_names=selected_names,
        calibration_accuracy=calibration_accuracy,
        calibration_rows=len(calibration_idx),
    )


def _build_nested_rule_cache(
    *,
    base: pd.DataFrame,
    features: pd.DataFrame,
    threshold_window: int,
    min_threshold_rows: int,
) -> NestedRuleCache:
    next_returns = base["close"].shift(-1) / base["close"] - 1.0
    labels = (next_returns > 0).astype(int).to_numpy()
    valid_mask = features.notna().all(axis=1).to_numpy() & next_returns.notna().to_numpy()

    rule_columns = _rule_candidate_columns(features)
    quantiles = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    prediction_rows: list[np.ndarray] = []
    rule_names: list[str] = []
    n_rows = len(features)

    for column in rule_columns:
        if column not in features.columns:
            continue
        series = features[column].astype(float)
        values = series.to_numpy(dtype=float)
        not_nan = np.isfinite(values)
        for quantile in quantiles:
            threshold = (
                series.shift(1)
                .rolling(threshold_window, min_periods=min_threshold_rows)
                .quantile(quantile)
                .to_numpy(dtype=float)
            )
            valid = np.isfinite(threshold) & not_nan
            gt_prediction = np.full(n_rows, -1, dtype=np.int8)
            le_prediction = np.full(n_rows, -1, dtype=np.int8)
            gt_prediction[valid] = (values[valid] > threshold[valid]).astype(np.int8)
            le_prediction[valid] = (values[valid] <= threshold[valid]).astype(np.int8)
            prediction_rows.append(gt_prediction)
            rule_names.append(f"{column}>rolling_q{quantile:.1f}")
            prediction_rows.append(le_prediction)
            rule_names.append(f"{column}<=rolling_q{quantile:.1f}")

    if not prediction_rows:
        raise ValueError("No nested volatility rules could be built.")
    return NestedRuleCache(
        predictions=np.vstack(prediction_rows),
        rule_names=rule_names,
        labels=labels,
        valid_mask=valid_mask,
    )


def _nested_volatility_rule_signal(
    *,
    base: pd.DataFrame,
    cache: NestedRuleCache,
    idx: int,
    calibration_window: int,
    min_calibration_rows: int,
    top_k: int,
) -> RuleSignal:
    start = max(0, idx - calibration_window)
    calibration_idx = np.arange(start, idx)
    valid_calibration = cache.valid_mask[calibration_idx]
    calibration_idx = calibration_idx[valid_calibration]
    if len(calibration_idx) < min_calibration_rows:
        recent = cache.labels[max(0, idx - 20) : idx]
        predicted_label = int(np.nanmean(recent) >= 0.5) if len(recent) else 1
        return _rule_signal_from_label(
            predicted_label=predicted_label,
            base=base,
            idx=idx,
            rule_names=["recent_majority_fallback"],
            calibration_accuracy=0.5,
            calibration_rows=len(calibration_idx),
        )

    y_cal = cache.labels[calibration_idx]
    scored: list[tuple[float, str, np.ndarray]] = []
    for rule_idx, raw_prediction in enumerate(cache.predictions):
        pred_cal = raw_prediction[calibration_idx]
        ok = pred_cal >= 0
        if int(ok.sum()) < min_calibration_rows:
            continue
        score = float((pred_cal[ok] == y_cal[ok]).mean())
        prediction = raw_prediction
        name = cache.rule_names[rule_idx]
        if score < 0.5:
            prediction = np.where(raw_prediction >= 0, 1 - raw_prediction, -1)
            score = 1.0 - score
            name = f"NOT({name})"
        if prediction[idx] >= 0:
            scored.append((score, name, prediction))

    if not scored:
        recent = cache.labels[max(0, idx - 20) : idx]
        predicted_label = int(np.nanmean(recent) >= 0.5) if len(recent) else 1
        return _rule_signal_from_label(
            predicted_label=predicted_label,
            base=base,
            idx=idx,
            rule_names=["recent_majority_fallback"],
            calibration_accuracy=0.5,
            calibration_rows=len(calibration_idx),
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: max(1, top_k)]
    weights = np.asarray([max(score - 0.5, 0.001) for score, _, _ in selected])
    votes = np.asarray([prediction[idx] for _, _, prediction in selected], dtype=float)
    vote_score = float(np.dot(votes, weights) / weights.sum())
    predicted_label = int(vote_score >= 0.5)

    selected_matrix = np.vstack([prediction for _, _, prediction in selected])
    calibration_votes = selected_matrix[:, calibration_idx]
    calibration_valid = (calibration_votes >= 0).all(axis=0)
    if calibration_valid.any():
        ensemble_calibration_pred = (
            calibration_votes[:, calibration_valid].mean(axis=0) >= 0.5
        ).astype(int)
        calibration_accuracy = float(
            (ensemble_calibration_pred == y_cal[calibration_valid]).mean()
        )
    else:
        calibration_accuracy = float(selected[0][0])

    return _rule_signal_from_label(
        predicted_label=predicted_label,
        base=base,
        idx=idx,
        rule_names=[f"{name}:cal_acc={score:.3f}" for score, name, _ in selected],
        calibration_accuracy=calibration_accuracy,
        calibration_rows=len(calibration_idx),
        diagnostics={
            "rule_mode": "volatility_rule",
            "selected_rule_count": len(selected),
            "top_rule_score": float(selected[0][0]),
        },
    )


def _stability_weighted_rule_signal(
    *,
    base: pd.DataFrame,
    cache: NestedRuleCache,
    idx: int,
    recent_window: int,
    long_window: int,
    min_calibration_rows: int,
    top_k: int,
    recent_weight: float,
    min_edge: float,
) -> RuleSignal:
    recent_start = max(0, idx - recent_window)
    long_start = max(0, idx - long_window)
    recent_idx = np.arange(recent_start, idx)
    long_idx = np.arange(long_start, idx)
    recent_idx = recent_idx[cache.valid_mask[recent_idx]]
    long_idx = long_idx[cache.valid_mask[long_idx]]

    if len(recent_idx) < min_calibration_rows or len(long_idx) < min_calibration_rows:
        recent = cache.labels[max(0, idx - 20) : idx]
        predicted_label = int(np.nanmean(recent) >= 0.5) if len(recent) else 1
        return _rule_signal_from_label(
            predicted_label=predicted_label,
            base=base,
            idx=idx,
            rule_names=["recent_majority_fallback"],
            calibration_accuracy=0.5,
            calibration_rows=len(recent_idx),
            diagnostics={
                "rule_mode": "stability_rule",
                "fallback_reason": "not_enough_calibration_rows",
                "recent_rows": len(recent_idx),
                "long_rows": len(long_idx),
            },
        )

    y_recent = cache.labels[recent_idx]
    y_long = cache.labels[long_idx]
    recent_weight = float(np.clip(recent_weight, 0.0, 1.0))
    scored: list[tuple[float, str, np.ndarray, dict[str, float]]] = []

    for rule_idx, raw_prediction in enumerate(cache.predictions):
        raw_current = raw_prediction[idx]
        if raw_current < 0:
            continue

        recent_pred = raw_prediction[recent_idx]
        recent_ok = recent_pred >= 0
        long_pred = raw_prediction[long_idx]
        long_ok = long_pred >= 0
        recent_rows = int(recent_ok.sum())
        long_rows = int(long_ok.sum())
        if recent_rows < min_calibration_rows or long_rows < min_calibration_rows:
            continue

        recent_acc = float((recent_pred[recent_ok] == y_recent[recent_ok]).mean())
        long_acc = float((long_pred[long_ok] == y_long[long_ok]).mean())
        prediction = raw_prediction
        name = cache.rule_names[rule_idx]
        current_prediction = int(raw_current)
        if recent_acc < 0.5:
            prediction = np.where(raw_prediction >= 0, 1 - raw_prediction, -1)
            recent_acc = 1.0 - recent_acc
            long_acc = 1.0 - long_acc
            current_prediction = 1 - current_prediction
            name = f"NOT({name})"

        stability_penalty = abs(recent_acc - long_acc)
        sample_penalty = 0.5 / math.sqrt(max(recent_rows, 1))
        blended_acc = recent_weight * recent_acc + (1.0 - recent_weight) * long_acc
        edge = blended_acc - 0.5
        score = edge - 0.5 * stability_penalty - sample_penalty
        if edge < min_edge:
            continue
        if current_prediction >= 0:
            scored.append(
                (
                    float(score),
                    name,
                    prediction,
                    {
                        "recent_accuracy": recent_acc,
                        "long_accuracy": long_acc,
                        "blended_accuracy": blended_acc,
                        "stability_penalty": stability_penalty,
                        "sample_penalty": sample_penalty,
                        "recent_rows": float(recent_rows),
                        "long_rows": float(long_rows),
                    },
                )
            )

    if not scored:
        recent = cache.labels[max(0, idx - 20) : idx]
        predicted_label = int(np.nanmean(recent) >= 0.5) if len(recent) else 1
        return _rule_signal_from_label(
            predicted_label=predicted_label,
            base=base,
            idx=idx,
            rule_names=["recent_majority_fallback"],
            calibration_accuracy=0.5,
            calibration_rows=len(recent_idx),
            diagnostics={
                "rule_mode": "stability_rule",
                "fallback_reason": "no_positive_stability_rules",
                "recent_rows": len(recent_idx),
                "long_rows": len(long_idx),
            },
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: max(1, top_k)]
    weights = np.asarray([max(score, 0.001) for score, _, _, _ in selected])
    votes = np.asarray([prediction[idx] for _, _, prediction, _ in selected], dtype=float)
    vote_score = float(np.dot(votes, weights) / weights.sum())
    predicted_label = int(vote_score >= 0.5)

    selected_matrix = np.vstack([prediction for _, _, prediction, _ in selected])
    calibration_votes = selected_matrix[:, recent_idx]
    calibration_valid = (calibration_votes >= 0).all(axis=0)
    if calibration_valid.any():
        calibration_pred = (
            np.average(
                calibration_votes[:, calibration_valid],
                axis=0,
                weights=weights,
            )
            >= 0.5
        ).astype(int)
        calibration_accuracy = float(
            (calibration_pred == y_recent[calibration_valid]).mean()
        )
    else:
        calibration_accuracy = float(selected[0][3]["recent_accuracy"])

    top_diag = selected[0][3]
    selected_names = [
        (
            f"{name}:score={score:.3f}:recent_acc={diag['recent_accuracy']:.3f}:"
            f"long_acc={diag['long_accuracy']:.3f}"
        )
        for score, name, _, diag in selected
    ]
    return _rule_signal_from_label(
        predicted_label=predicted_label,
        base=base,
        idx=idx,
        rule_names=selected_names,
        calibration_accuracy=calibration_accuracy,
        calibration_rows=len(recent_idx),
        diagnostics={
            "rule_mode": "stability_rule",
            "selected_rule_count": len(selected),
            "top_rule_score": float(selected[0][0]),
            "top_recent_accuracy": float(top_diag["recent_accuracy"]),
            "top_long_accuracy": float(top_diag["long_accuracy"]),
            "top_blended_accuracy": float(top_diag["blended_accuracy"]),
            "top_stability_penalty": float(top_diag["stability_penalty"]),
            "top_sample_penalty": float(top_diag["sample_penalty"]),
            "recent_weight": recent_weight,
            "vote_score": vote_score,
            "recent_rows": len(recent_idx),
            "long_rows": len(long_idx),
        },
    )


def _build_state_veto_masks(
    *,
    features: pd.DataFrame,
    state_window: int,
    quantiles: tuple[float, ...],
) -> list[tuple[str, np.ndarray]]:
    masks: list[tuple[str, np.ndarray]] = []
    state_columns = [
        "volatility_20",
    ]
    for column in state_columns:
        if column not in features.columns:
            continue
        series = features[column].astype(float)
        for quantile in quantiles:
            threshold = (
                series.shift(1)
                .rolling(state_window, min_periods=40)
                .quantile(float(quantile))
            )
            masks.append(
                (
                    f"{column}<=rolling_q{float(quantile):.1f}",
                    (series <= threshold).fillna(False).to_numpy(dtype=bool),
                )
            )
    return masks


def _state_veto_rule_signal(
    *,
    base: pd.DataFrame,
    cache: NestedRuleCache,
    idx: int,
    base_signals: dict[int, RuleSignal],
    state_masks: list[tuple[str, np.ndarray]],
    veto_window: int,
    min_rows: int,
    bad_accuracy: float,
) -> RuleSignal:
    base_signal = base_signals[idx]
    base_label_by_idx = {
        signal_idx: signal.predicted_label
        for signal_idx, signal in base_signals.items()
        if signal_idx < idx
    }
    start = max(min(base_signals), idx - veto_window)
    history_idx = np.asarray(
        [signal_idx for signal_idx in range(start, idx) if signal_idx in base_label_by_idx],
        dtype=int,
    )

    candidates: list[tuple[float, int, str]] = []
    if len(history_idx) > 0:
        history_labels = np.asarray(
            [base_label_by_idx[int(signal_idx)] for signal_idx in history_idx],
            dtype=np.int8,
        )
        history_correct = history_labels == cache.labels[history_idx]
        for state_name, mask in state_masks:
            if idx >= len(mask) or not bool(mask[idx]):
                continue
            state_history = history_idx[mask[history_idx]]
            if len(state_history) < min_rows:
                continue
            state_labels = np.asarray(
                [base_label_by_idx[int(signal_idx)] for signal_idx in state_history],
                dtype=np.int8,
            )
            state_accuracy = float((state_labels == cache.labels[state_history]).mean())
            if state_accuracy <= bad_accuracy:
                candidates.append((state_accuracy, len(state_history), state_name))

    if not candidates:
        return RuleSignal(
            predicted_return=base_signal.predicted_return,
            predicted_label=base_signal.predicted_label,
            rule_names=base_signal.rule_names,
            calibration_accuracy=base_signal.calibration_accuracy,
            calibration_rows=base_signal.calibration_rows,
            diagnostics={
                **base_signal.diagnostics,
                "rule_mode": "state_veto_rule",
                "veto_applied": 0,
                "veto_reason": "",
            },
        )

    candidates.sort(key=lambda item: item[0])
    state_accuracy, state_rows, state_name = candidates[0]
    veto_label = 1 - base_signal.predicted_label
    return _rule_signal_from_label(
        predicted_label=veto_label,
        base=base,
        idx=idx,
        rule_names=[
            f"state_veto({state_name}:hist_acc={state_accuracy:.3f}:rows={state_rows})",
            *base_signal.rule_names,
        ],
        calibration_accuracy=state_accuracy,
        calibration_rows=state_rows,
        diagnostics={
            **base_signal.diagnostics,
            "rule_mode": "state_veto_rule",
            "veto_applied": 1,
            "veto_state": state_name,
            "veto_state_accuracy": float(state_accuracy),
            "veto_state_rows": int(state_rows),
            "base_calibration_accuracy": base_signal.calibration_accuracy,
            "base_calibration_rows": base_signal.calibration_rows,
        },
    )


def _build_nested_ml_cache(
    *,
    base: pd.DataFrame,
    features: pd.DataFrame,
    train_window: int,
    step: int,
    min_train_rows: int,
) -> NestedMLCache:
    next_returns = base["close"].shift(-1) / base["close"] - 1.0
    labels = (next_returns > 0).astype(int).to_numpy()
    valid_mask = features.notna().all(axis=1).to_numpy() & next_returns.notna().to_numpy()
    x_all = features.to_numpy(dtype=np.float32)
    n_rows = len(features)
    model_builders = _nested_ml_model_builders()
    model_names = list(model_builders)
    predictions = np.full((len(model_names), n_rows), -1, dtype=np.int8)

    last_models: dict[str, Any] = {}
    for idx in range(n_rows):
        if not valid_mask[idx]:
            continue
        start = max(0, idx - train_window)
        train_idx = np.arange(start, idx)
        train_idx = train_idx[valid_mask[train_idx]]
        if len(train_idx) < min_train_rows or len(np.unique(labels[train_idx])) < 2:
            continue

        should_refit = (idx % max(1, step) == 0) or not last_models
        if should_refit:
            x_train = x_all[train_idx]
            y_train = labels[train_idx]
            fitted: dict[str, Any] = {}
            for name, builder in model_builders.items():
                try:
                    model = builder()
                    model.fit(x_train, y_train)
                    fitted[name] = model
                except Exception:
                    continue
            if fitted:
                last_models = fitted

        for model_idx, name in enumerate(model_names):
            model = last_models.get(name)
            if model is None:
                continue
            try:
                predictions[model_idx, idx] = int(model.predict(x_all[idx : idx + 1])[0])
            except Exception:
                continue

    return NestedMLCache(
        predictions=predictions,
        model_names=model_names,
        labels=labels,
        valid_mask=valid_mask,
        feature_columns=list(features.columns),
    )


def _nested_ml_model_builders() -> dict[str, Any]:
    return {
        "log_l2_bal": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                C=0.1,
                class_weight="balanced",
                solver="liblinear",
            ),
        ),
        "ridge_bal": lambda: make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=5.0, class_weight="balanced"),
        ),
        "gaussian_nb": lambda: GaussianNB(),
        "gb_stump": lambda: GradientBoostingClassifier(
            n_estimators=30,
            learning_rate=0.03,
            max_depth=1,
            random_state=42,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=50,
            max_depth=2,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    }


def _nested_ml_signal(
    *,
    base: pd.DataFrame,
    cache: NestedMLCache,
    idx: int,
    calibration_window: int,
    top_k: int,
) -> RuleSignal:
    start = max(0, idx - calibration_window)
    calibration_idx = np.arange(start, idx)
    calibration_idx = calibration_idx[cache.valid_mask[calibration_idx]]
    if len(calibration_idx) < 40:
        recent = cache.labels[max(0, idx - 20) : idx]
        predicted_label = int(np.nanmean(recent) >= 0.5) if len(recent) else 1
        return _rule_signal_from_label(
            predicted_label=predicted_label,
            base=base,
            idx=idx,
            rule_names=["recent_majority_fallback"],
            calibration_accuracy=0.5,
            calibration_rows=len(calibration_idx),
        )

    y_cal = cache.labels[calibration_idx]
    scored: list[tuple[float, str, np.ndarray]] = []
    for model_idx, raw_prediction in enumerate(cache.predictions):
        pred_cal = raw_prediction[calibration_idx]
        ok = pred_cal >= 0
        if int(ok.sum()) < 40 or raw_prediction[idx] < 0:
            continue
        score = float((pred_cal[ok] == y_cal[ok]).mean())
        prediction = raw_prediction
        name = cache.model_names[model_idx]
        if score < 0.5:
            prediction = np.where(raw_prediction >= 0, 1 - raw_prediction, -1)
            score = 1.0 - score
            name = f"NOT({name})"
        scored.append((score, name, prediction))

    if not scored:
        recent = cache.labels[max(0, idx - 20) : idx]
        predicted_label = int(np.nanmean(recent) >= 0.5) if len(recent) else 1
        return _rule_signal_from_label(
            predicted_label=predicted_label,
            base=base,
            idx=idx,
            rule_names=["recent_majority_fallback"],
            calibration_accuracy=0.5,
            calibration_rows=len(calibration_idx),
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: max(1, top_k)]
    weights = np.asarray([max(score - 0.5, 0.001) for score, _, _ in selected])
    votes = np.asarray([prediction[idx] for _, _, prediction in selected], dtype=float)
    predicted_label = int(float(np.dot(votes, weights) / weights.sum()) >= 0.5)

    selected_matrix = np.vstack([prediction for _, _, prediction in selected])
    calibration_votes = selected_matrix[:, calibration_idx]
    calibration_valid = (calibration_votes >= 0).all(axis=0)
    if calibration_valid.any():
        calibration_pred = (
            calibration_votes[:, calibration_valid].mean(axis=0) >= 0.5
        ).astype(int)
        calibration_accuracy = float(
            (calibration_pred == y_cal[calibration_valid]).mean()
        )
    else:
        calibration_accuracy = float(selected[0][0])

    return _rule_signal_from_label(
        predicted_label=predicted_label,
        base=base,
        idx=idx,
        rule_names=[f"{name}:cal_acc={score:.3f}" for score, name, _ in selected],
        calibration_accuracy=calibration_accuracy,
        calibration_rows=len(calibration_idx),
    )


def _hybrid_signal(
    *,
    base: pd.DataFrame,
    rule_cache: NestedRuleCache,
    ml_cache: NestedMLCache,
    idx: int,
    rule_calibration_window: int,
    rule_min_calibration_rows: int,
    rule_top_k: int,
    ml_calibration_window: int,
    ml_top_k: int,
) -> RuleSignal:
    rule_signal = _nested_volatility_rule_signal(
        base=base,
        cache=rule_cache,
        idx=idx,
        calibration_window=rule_calibration_window,
        min_calibration_rows=rule_min_calibration_rows,
        top_k=rule_top_k,
    )
    ml_signal = _nested_ml_signal(
        base=base,
        cache=ml_cache,
        idx=idx,
        calibration_window=ml_calibration_window,
        top_k=ml_top_k,
    )
    if ml_signal.calibration_rows >= 40 and (
        ml_signal.calibration_accuracy > rule_signal.calibration_accuracy + 0.02
    ):
        chosen = ml_signal
        source = "ml"
    else:
        chosen = rule_signal
        source = "rule"
    return RuleSignal(
        predicted_return=chosen.predicted_return,
        predicted_label=chosen.predicted_label,
        rule_names=[f"hybrid_source={source}"] + chosen.rule_names,
        calibration_accuracy=chosen.calibration_accuracy,
        calibration_rows=chosen.calibration_rows,
        diagnostics={
            **chosen.diagnostics,
            "rule_mode": "hybrid",
            "hybrid_source": source,
            "rule_calibration_accuracy": rule_signal.calibration_accuracy,
            "ml_calibration_accuracy": ml_signal.calibration_accuracy,
        },
    )


def _rule_candidate_columns(features: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in _INTERNAL_RULE_COLUMNS:
        if column in features.columns:
            columns.append(column)
    for column in features.columns:
        if not column.startswith("ext_"):
            continue
        is_moneyflow = column.startswith(_MONEYFLOW_RULE_PREFIXES)
        if not is_moneyflow and not column.endswith(_EXTERNAL_RULE_SUFFIXES):
            continue
        columns.append(column)
    for column in features.columns:
        if not column.startswith("tf_"):
            continue
        columns.append(column)

    return columns


def _clean_feature_frame(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Fill feature gaps without using labels or future values."""

    cleaned = features.replace([np.inf, -np.inf], np.nan).copy()
    if cleaned.empty:
        raise ValueError("No usable feature columns remain after cleaning.")

    cleaned = cleaned.ffill().fillna(0.0)
    return cleaned


def _candidate_rule_predictions(
    features: pd.DataFrame,
    threshold_mask: pd.Series,
) -> list[tuple[str, np.ndarray]]:
    rule_columns = _rule_candidate_columns(features)
    quantiles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    rules: list[tuple[str, np.ndarray]] = []
    for column in rule_columns:
        if column not in features.columns:
            continue
        series = features[column].astype(float)
        threshold_values = series.loc[threshold_mask].dropna()
        if len(threshold_values) < 50:
            continue
        for quantile in quantiles:
            threshold = float(threshold_values.quantile(quantile))
            values = series.to_numpy(dtype=float)
            rules.append(
                (
                    f"{column}>{quantile:.1f}",
                    (values > threshold).astype(np.int8),
                )
            )
            rules.append(
                (
                    f"{column}<={quantile:.1f}",
                    (values <= threshold).astype(np.int8),
                )
            )
    return rules


def _rule_signal_from_label(
    *,
    predicted_label: int,
    base: pd.DataFrame,
    idx: int,
    rule_names: list[str],
    calibration_accuracy: float,
    calibration_rows: int,
    diagnostics: dict[str, Any] | None = None,
) -> RuleSignal:
    recent_returns = base["close"].pct_change().iloc[max(1, idx - 252) : idx + 1]
    median_abs_return = float(np.nanmedian(np.abs(recent_returns)))
    if not np.isfinite(median_abs_return) or median_abs_return <= 0:
        median_abs_return = 0.003
    predicted_return = median_abs_return if predicted_label == 1 else -median_abs_return
    return RuleSignal(
        predicted_return=float(predicted_return),
        predicted_label=int(predicted_label),
        rule_names=rule_names,
        calibration_accuracy=float(calibration_accuracy),
        calibration_rows=int(calibration_rows),
        diagnostics=diagnostics or {},
    )


def _calibrate_return_magnitude(
    *,
    base: pd.DataFrame,
    idx: int,
    raw_predicted_return: float,
    mode: ReturnMagnitudeMode,
    window: int,
    min_rows: int,
    grid_size: int,
    clip_low_quantile: float,
    clip_high_quantile: float,
) -> tuple[float, dict[str, Any]]:
    """Adjust predicted return size without changing the predicted direction."""

    raw_return = float(raw_predicted_return)
    raw_sign = 1.0 if raw_return > 0 else -1.0
    raw_abs = abs(raw_return)
    diagnostics: dict[str, Any] = {
        "return_magnitude_mode": mode,
        "return_magnitude_method": "directional_median",
        "return_magnitude_raw_abs": raw_abs,
        "return_magnitude_calibrated_abs": raw_abs,
        "return_magnitude_current_range": np.nan,
        "return_magnitude_hist_rows": 0,
        "return_magnitude_scale": np.nan,
        "return_magnitude_clip_low": np.nan,
        "return_magnitude_clip_high": np.nan,
    }
    if mode == "directional_median":
        return raw_return, diagnostics

    close = pd.to_numeric(base["close"], errors="coerce")
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    next_abs_return = (close.shift(-1) / close - 1.0).abs()
    range_pct = (high - low) / close.replace(0.0, np.nan)

    start = max(0, idx - max(1, int(window)))
    hist = pd.DataFrame(
        {
            "range_pct": range_pct.iloc[start:idx],
            "next_abs_return": next_abs_return.iloc[start:idx],
        }
    ).replace([np.inf, -np.inf], np.nan)
    hist = hist.dropna()
    hist = hist[(hist["range_pct"] > 0.0) & (hist["next_abs_return"] > 0.0)]
    hist_rows = int(len(hist))
    current_range = float(range_pct.iloc[idx]) if idx < len(range_pct) else np.nan
    diagnostics["return_magnitude_current_range"] = current_range
    diagnostics["return_magnitude_hist_rows"] = hist_rows

    magnitude = np.nan
    scale = np.nan
    method = "range_scaled_fallback_median"
    if (
        hist_rows >= max(1, int(min_rows))
        and np.isfinite(current_range)
        and current_range > 0.0
    ):
        x = hist["range_pct"].to_numpy(dtype=float)
        y = hist["next_abs_return"].to_numpy(dtype=float)
        median_x = float(np.nanmedian(x))
        median_y = float(np.nanmedian(y))
        if np.isfinite(median_x) and median_x > 0.0 and np.isfinite(median_y):
            ratio = median_y / median_x
            grid_low = max(0.05, ratio * 0.3)
            grid_high = min(3.0, ratio * 2.5)
            if np.isfinite(grid_low) and np.isfinite(grid_high) and grid_high > grid_low:
                grid = np.linspace(grid_low, grid_high, max(2, int(grid_size)))
                errors = np.mean(np.abs(grid[:, None] * x[None, :] - y[None, :]), axis=1)
                scale = float(grid[int(np.argmin(errors))])
                magnitude = float(scale * current_range)
                method = "range_scaled"

    clip_source = hist["next_abs_return"].to_numpy(dtype=float) if hist_rows else np.array([])
    if len(clip_source) < max(10, min_rows // 4):
        clip_source = (
            next_abs_return.iloc[start:idx]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
    if len(clip_source):
        low_q = float(np.clip(clip_low_quantile, 0.0, 1.0))
        high_q = float(np.clip(clip_high_quantile, 0.0, 1.0))
        if low_q >= high_q:
            low_q, high_q = 0.05, 0.95
        clip_low = float(np.nanquantile(clip_source, low_q))
        clip_high = float(np.nanquantile(clip_source, high_q))
    else:
        clip_low = 0.0002
        clip_high = 0.08
    if not np.isfinite(clip_low) or clip_low <= 0.0:
        clip_low = 0.0002
    if not np.isfinite(clip_high) or clip_high < clip_low:
        clip_high = max(clip_low, 0.08)

    if not np.isfinite(magnitude) or magnitude <= 0.0:
        fallback_source = (
            next_abs_return.iloc[start:idx]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(fallback_source):
            magnitude = float(np.nanmedian(fallback_source))
        elif raw_abs > 0.0 and np.isfinite(raw_abs):
            magnitude = raw_abs
        else:
            magnitude = 0.003

    magnitude = float(np.clip(magnitude, clip_low, clip_high))
    calibrated = float(raw_sign * magnitude)
    diagnostics.update(
        {
            "return_magnitude_method": method,
            "return_magnitude_calibrated_abs": abs(calibrated),
            "return_magnitude_scale": scale,
            "return_magnitude_clip_low": clip_low,
            "return_magnitude_clip_high": clip_high,
        }
    )
    return calibrated, diagnostics


def _apply_recent_failure_guard(
    *,
    predicted_return: float,
    raw_direction_history: list[bool],
    enabled: bool,
    window: int,
    degrade_threshold: float,
    invert_threshold: float,
    short_window: int,
    short_threshold: float,
) -> tuple[float, dict[str, Any]]:
    window = max(1, int(window))
    short_window = max(1, int(short_window))
    degrade_threshold = float(np.clip(degrade_threshold, 0.0, 1.0))
    invert_threshold = float(np.clip(invert_threshold, 0.0, 1.0))
    short_threshold = float(np.clip(short_threshold, 0.0, 1.0))
    recent = raw_direction_history[-window:]
    recent_rows = len(recent)
    recent_accuracy = float(np.mean(recent)) if recent_rows else np.nan
    short_recent = raw_direction_history[-short_window:]
    short_rows = len(short_recent)
    short_accuracy = float(np.mean(short_recent)) if short_rows else np.nan
    degraded = bool(
        enabled
        and recent_rows >= window
        and np.isfinite(recent_accuracy)
        and recent_accuracy <= degrade_threshold
    )
    inverted = bool(
        degraded
        and recent_accuracy <= invert_threshold
        and short_rows >= short_window
        and np.isfinite(short_accuracy)
        and short_accuracy <= short_threshold
    )
    status = "inverted" if inverted else ("degraded" if degraded else "normal")
    diagnostics = {
        "recent_failure_guard_enabled": int(bool(enabled)),
        "recent_failure_guard_status": status,
        "recent_failure_guard_degraded": int(degraded),
        "recent_failure_guard_applied": int(inverted),
        "recent_failure_guard_window": window,
        "recent_failure_guard_threshold": degrade_threshold,
        "recent_failure_guard_degrade_threshold": degrade_threshold,
        "recent_failure_guard_invert_threshold": invert_threshold,
        "recent_failure_guard_rows": recent_rows,
        "recent_failure_guard_accuracy": recent_accuracy,
        "recent_failure_guard_short_window": short_window,
        "recent_failure_guard_short_threshold": short_threshold,
        "recent_failure_guard_short_rows": short_rows,
        "recent_failure_guard_short_accuracy": short_accuracy,
    }
    if inverted:
        return float(-predicted_return), diagnostics
    return float(predicted_return), diagnostics


def _timeseries_cv_threshold(
    *,
    prepared: PreparedMarketData,
    config: DirectionPredictionConfig,
) -> tuple[float, dict[str, float]] | None:
    """Use historical CV folds only to estimate a robust decision threshold."""
    n_samples = len(prepared.labels)
    device = _resolve_device(config.device)

    tscv = TimeSeriesSplit(n_splits=3, test_size=max(50, int(n_samples * 0.1)))
    cv_scores: list[tuple[float, float, dict[str, float]]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(prepared.labeled_sequences)):
        if len(train_idx) < config.min_train_sequences:
            continue

        scaler = StandardScaler()
        scaler.fit(prepared.labeled_sequences[train_idx].reshape(-1, len(prepared.feature_columns)))

        x_train = _scale_sequences(prepared.labeled_sequences[train_idx], scaler)
        x_val = _scale_sequences(prepared.labeled_sequences[val_idx], scaler)
        y_train = prepared.labels[train_idx].astype(np.float32)
        y_val = prepared.labels[val_idx].astype(np.float32)

        model = _train_classifier(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            input_size=len(prepared.feature_columns),
            config=config,
            device=device,
        )

        val_probs = _predict_proba(model, x_val, device, config.batch_size)
        threshold = _tune_threshold(val_probs, y_val, config)
        metrics = _classification_metrics(val_probs, y_val, threshold)
        score = metrics["balanced_accuracy"]

        cv_scores.append((score, threshold, metrics))

        if config.verbose:
            print(f"  CV fold {fold_idx+1}: balanced_acc={score:.4f} threshold={threshold:.3f}")

    if not cv_scores:
        return None

    cv_scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_threshold, best_metrics = cv_scores[0]
    thresholds = np.asarray([item[1] for item in cv_scores], dtype=float)
    median_threshold = float(np.median(thresholds))
    summary = {
        "best_balanced_accuracy": float(best_score),
        "best_hit_ratio": float(best_metrics["hit_ratio"]),
        "best_threshold": float(best_threshold),
        "median_threshold": median_threshold,
        "folds": float(len(cv_scores)),
    }

    if config.verbose:
        print(
            f"  Best CV fold: balanced_acc={best_score:.4f} "
            f"threshold={best_threshold:.3f}; median_threshold={median_threshold:.3f}"
        )

    return median_threshold, summary


def _prepare_market_data(
    df: pd.DataFrame,
    config: DirectionPredictionConfig,
) -> PreparedMarketData:
    normalized = _normalize_market_frame(df, config)
    original_rows = len(df)
    cleaned_rows = len(normalized)

    features = _clean_feature_frame(_build_features(normalized, config))

    next_returns = normalized["close"].shift(-1) / normalized["close"] - 1.0
    labels = pd.Series(np.nan, index=normalized.index, dtype="float64")
    labels.loc[next_returns > config.neutral_band] = 1.0
    labels.loc[next_returns < -config.neutral_band] = 0.0

    valid_feature_mask = features.notna().all(axis=1)
    model_frame = normalized.loc[valid_feature_mask].copy()
    feature_frame = features.loc[valid_feature_mask].copy()
    label_series = labels.loc[valid_feature_mask]
    return_series = next_returns.loc[valid_feature_mask]

    if len(feature_frame) < config.lookback + config.min_train_sequences:
        raise ValueError(
            "Not enough usable rows after feature engineering. "
            f"Usable rows={len(feature_frame)}, lookback={config.lookback}."
        )

    feature_columns = list(feature_frame.columns)
    feature_values = feature_frame.to_numpy(dtype=np.float32)
    latest_sequence = feature_values[-config.lookback :]
    latest_date = pd.Timestamp(model_frame["date"].iloc[-1])
    latest_close = float(model_frame["close"].iloc[-1])

    sequences: list[np.ndarray] = []
    y_values: list[float] = []
    realized_returns: list[float] = []
    end_dates: list[pd.Timestamp] = []
    for end_idx in range(config.lookback - 1, len(feature_frame)):
        label_value = label_series.iloc[end_idx]
        next_return = return_series.iloc[end_idx]
        if pd.isna(label_value) or pd.isna(next_return):
            continue
        sequences.append(feature_values[end_idx - config.lookback + 1 : end_idx + 1])
        y_values.append(float(label_value))
        realized_returns.append(float(next_return))
        end_dates.append(pd.Timestamp(model_frame["date"].iloc[end_idx]))

    if not sequences:
        raise ValueError("No labeled sequences were created from the input data.")

    labeled_sequences = np.stack(sequences).astype(np.float32)
    label_array = np.asarray(y_values, dtype=np.float32)
    return_array = np.asarray(realized_returns, dtype=np.float32)
    if len(label_array) < config.min_train_sequences + 20:
        raise ValueError(
            "Too few non-neutral labeled sequences: "
            f"{len(label_array)}. Provide more history or lower neutral_band."
        )

    if len(np.unique(label_array)) < 2:
        raise ValueError(
            "Only one direction class remains after neutral-band filtering. "
            "Provide more history or lower neutral_band."
        )

    return PreparedMarketData(
        frame=model_frame,
        feature_columns=feature_columns,
        labeled_sequences=labeled_sequences,
        labels=label_array,
        next_returns=return_array,
        sample_end_dates=pd.Series(end_dates),
        latest_sequence=latest_sequence.astype(np.float32),
        latest_date=latest_date,
        latest_close=latest_close,
        original_rows=original_rows,
        cleaned_rows=cleaned_rows,
    )


def _normalize_market_frame(
    df: pd.DataFrame,
    config: DirectionPredictionConfig,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input market DataFrame is empty.")

    columns_by_lower = {str(col).strip().lower(): col for col in df.columns}

    def find_column(name: str) -> Any | None:
        for alias in _ALIASES[name]:
            if alias.lower() in columns_by_lower:
                return columns_by_lower[alias.lower()]
        return None

    date_col = find_column("date")
    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            date_values = pd.Series(df.index, index=df.index)
        else:
            raise ValueError(
                "Could not find a date column. Expected one of: "
                + ", ".join(_ALIASES["date"])
            )
    else:
        date_values = df[date_col]

    required = ["open", "high", "low", "close"]
    missing = [name for name in required if find_column(name) is None]
    if missing:
        raise ValueError(
            "Missing required OHLC columns: "
            + ", ".join(missing)
            + ". Supported aliases are defined in _ALIASES."
        )

    normalized = pd.DataFrame(index=df.index)
    normalized["date"] = _parse_dates(date_values, dayfirst=config.dayfirst)
    for name in required:
        normalized[name] = pd.to_numeric(df[find_column(name)], errors="coerce")

    pre_close_col = find_column("pre_close")
    if pre_close_col is not None:
        normalized["pre_close"] = pd.to_numeric(df[pre_close_col], errors="coerce")
    else:
        normalized["pre_close"] = np.nan

    volume_col = find_column("volume")
    amount_col = find_column("amount")
    normalized["volume"] = (
        pd.to_numeric(df[volume_col], errors="coerce")
        if volume_col is not None
        else np.nan
    )
    normalized["amount"] = (
        pd.to_numeric(df[amount_col], errors="coerce")
        if amount_col is not None
        else np.nan
    )

    external_columns = _external_feature_columns(df, columns_by_lower, config)
    if external_columns:
        external_data = {
            _safe_external_feature_name(str(column)): pd.to_numeric(
                df[column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
            for column in external_columns
        }
        normalized = pd.concat(
            [normalized, pd.DataFrame(external_data, index=df.index)],
            axis=1,
        )

    normalized["_source_order"] = np.arange(len(normalized))
    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"])
    normalized = normalized.sort_values(["date", "_source_order"])
    normalized = normalized.drop_duplicates(subset=["date"], keep="last")
    normalized = normalized.sort_values("date").reset_index(drop=True)

    normalized["pre_close"] = normalized["pre_close"].where(
        normalized["pre_close"].gt(0),
        normalized["close"].shift(1),
    )

    valid = (
        normalized["open"].gt(0)
        & normalized["high"].gt(0)
        & normalized["low"].gt(0)
        & normalized["close"].gt(0)
        & normalized["high"].ge(normalized["low"])
    )
    if volume_col is not None and config.drop_zero_volume:
        valid &= normalized["volume"].gt(0)

    normalized = normalized.loc[valid].copy().reset_index(drop=True)
    if len(normalized) < config.lookback + config.min_train_sequences:
        raise ValueError(
            "Not enough valid market rows after cleaning. "
            f"Rows={len(normalized)}."
        )

    if normalized["volume"].isna().all():
        normalized["volume"] = 1.0
    if normalized["amount"].isna().all():
        normalized["amount"] = normalized["close"] * normalized["volume"]

    normalized["pre_close"] = normalized["pre_close"].where(
        normalized["pre_close"].gt(0),
        normalized["close"].shift(1),
    )
    normalized = normalized.dropna(subset=["pre_close"]).reset_index(drop=True)
    normalized["amount"] = normalized["amount"].fillna(
        normalized["close"] * normalized["volume"]
    )
    normalized["volume"] = normalized["volume"].ffill().fillna(1.0)
    external_feature_cols = [col for col in normalized.columns if col.startswith("ext_")]
    if external_feature_cols:
        normalized[external_feature_cols] = normalized[external_feature_cols].ffill()
    normalized = normalized.drop(columns=["_source_order"], errors="ignore")
    return normalized


def _parse_dates(values: pd.Series, dayfirst: bool) -> pd.Series:
    series = pd.Series(values).copy()
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    as_text = series.astype(str).str.strip()
    yyyymmdd = as_text.str.fullmatch(r"\d{8}")
    if bool(yyyymmdd.mean() > 0.8):
        return pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")

    iso_ymd = as_text.str.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    if bool(iso_ymd.mean() > 0.8):
        normalized = as_text.str.replace("/", "-", regex=False)
        return pd.to_datetime(normalized, format="%Y-%m-%d", errors="coerce")

    return pd.to_datetime(as_text, errors="coerce", dayfirst=dayfirst)


def _parse_single_date(value: str | int | pd.Timestamp, dayfirst: bool) -> pd.Timestamp:
    parsed = _parse_dates(pd.Series([value]), dayfirst=dayfirst).iloc[0]
    if pd.isna(parsed):
        raise ValueError(f"Could not parse date value: {value!r}")
    return pd.Timestamp(parsed)


def _external_feature_columns(
    df: pd.DataFrame,
    columns_by_lower: dict[str, Any],
    config: DirectionPredictionConfig,
) -> list[Any]:
    if config.external_feature_mode == "none":
        return []

    base_columns = set()
    for aliases in _ALIASES.values():
        for alias in aliases:
            column = columns_by_lower.get(alias.lower())
            if column is not None:
                base_columns.add(column)

    selected: list[Any] = []
    used_names: set[str] = set()
    for column in df.columns:
        if column in base_columns:
            continue
        raw_name = str(column).strip()
        lower_name = raw_name.lower()
        if lower_name in _BASE_INPUT_COLUMNS:
            continue
        if lower_name.startswith(_FORBIDDEN_EXTERNAL_PREFIXES):
            continue
        core_prefixes = _CORE_EXTERNAL_PREFIXES + tuple(
            prefix.lower() for prefix in config.extra_core_external_prefixes
        )
        if config.external_feature_mode == "core" and not lower_name.startswith(
            core_prefixes
        ):
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            converted = pd.to_numeric(df[column], errors="coerce")
            non_na_ratio = float(converted.notna().mean()) if len(converted) else 0.0
            if non_na_ratio < 0.5:
                continue
        safe_name = _safe_external_feature_name(raw_name)
        if safe_name in used_names:
            continue
        used_names.add(safe_name)
        selected.append(column)
    return selected


def _safe_external_feature_name(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return f"ext_{cleaned}" if cleaned else "ext_feature"


def _build_features(
    data: pd.DataFrame,
    config: DirectionPredictionConfig | None = None,
) -> pd.DataFrame:
    close = data["close"].astype(float)
    open_ = data["open"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    pre_close = data["pre_close"].astype(float)
    volume = data["volume"].astype(float).replace(0, np.nan)
    amount = data["amount"].astype(float).replace(0, np.nan)

    ret = close.pct_change()
    log_ret = np.log(close).diff()

    features = pd.DataFrame(index=data.index)
    features["return_1"] = ret
    features["log_return_1"] = log_ret
    features["overnight_gap"] = open_ / pre_close - 1.0
    features["intraday_return"] = close / open_ - 1.0
    features["range_pct"] = (high - low) / pre_close
    features["close_position"] = (close - low) / (high - low).replace(0, np.nan)

    # Features are evaluated after the current close, so same-day OHLCV is valid
    # for predicting the next trading day. Rule thresholds still use shift(1).
    for window in (5, 10):
        sma = close.rolling(window).mean()
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        features[f"sma_{window}_gap"] = close / sma - 1.0
        features[f"ema_{window}_gap"] = close / ema - 1.0

    features["rsi_14"] = _rsi(close, 14) / 100.0
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = (ema_12 - ema_26) / close
    features["macd"] = macd
    features["macd_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    for window in (5, 20):
        features[f"momentum_{window}"] = close / close.shift(window) - 1.0
        features[f"volatility_{window}"] = ret.rolling(window).std()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_denom = (4.0 * bb_std).replace(0, np.nan)
    features["bollinger_pos"] = (close - (bb_mid - 2.0 * bb_std)) / bb_denom
    features["bollinger_width"] = bb_denom / close

    for lag in range(1, 6):
        features[f"return_lag_{lag}"] = ret.shift(lag)
    for lag in range(1, 4):
        features[f"momentum_5_lag_{lag}"] = features["momentum_5"].shift(lag)

    log_volume = np.log(volume)
    log_amount = np.log(amount)
    features["volume_log_change"] = log_volume.diff()
    features["amount_log_change"] = log_amount.diff()
    volume_std = log_volume.rolling(20).std().replace(0, np.nan)
    amount_std = log_amount.rolling(20).std().replace(0, np.nan)
    features["volume_z_20"] = (log_volume - log_volume.rolling(20).mean()) / volume_std
    features["amount_z_20"] = (log_amount - log_amount.rolling(20).mean()) / amount_std

    external_cols = [col for col in data.columns if col.startswith("ext_")]
    external_features: dict[str, pd.Series] = {}
    for column in external_cols:
        series = data[column].astype(float)
        external_features[column] = series
        if (
            column.endswith("_ret1")
            or column.endswith("_ret1_lag1")
            or column.endswith("_pct_chg")
            or column.endswith("_pct_chg_lag1")
        ):
            rolling_std = series.rolling(60).std().replace(0, np.nan)
            external_features[f"{column}_z60"] = (
                series - series.rolling(60).mean()
            ) / rolling_std
    if external_features:
        features = pd.concat([features, pd.DataFrame(external_features)], axis=1)

    if config is not None and config.technical_feature_mode in {"v1", "v1_core"}:
        technical_features = _build_v1_technical_feature_frame(
            data,
            core_only=config.technical_feature_mode == "v1_core",
        )
        if not technical_features.empty:
            features = pd.concat([features, technical_features], axis=1)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def _build_v1_technical_feature_frame(
    data: pd.DataFrame,
    *,
    core_only: bool = False,
) -> pd.DataFrame:
    if _legacy_technical_features is None:
        raise RuntimeError(
            "technical_feature_mode='v1' requires technical_features.py to be importable."
        )

    input_frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(data["date"]).dt.strftime("%Y%m%d"),
            "open": data["open"].to_numpy(dtype=float),
            "high": data["high"].to_numpy(dtype=float),
            "low": data["low"].to_numpy(dtype=float),
            "close": data["close"].to_numpy(dtype=float),
            "volume": data["volume"].to_numpy(dtype=float),
        },
        index=data.index,
    )
    processed, feature_columns = _legacy_technical_features(
        input_frame,
        N=30,
        mixture_depth=1,
        mark_labels=False,
        selected_features=None,
    )
    selected = [
        column
        for column in feature_columns
        if column in processed.columns and column not in _FORBIDDEN_TECHNICAL_COLUMNS
    ]
    if core_only:
        selected = [column for column in selected if column in _CORE_TECHNICAL_COLUMNS]
    if not selected:
        return pd.DataFrame(index=data.index)

    out = processed[selected].apply(pd.to_numeric, errors="coerce")
    out = out.reset_index(drop=True)
    out.index = data.index
    out = out.add_prefix("tf_")
    return out


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _validation_size(n_samples: int, config: DirectionPredictionConfig) -> int:
    requested = int(round(n_samples * config.validation_fraction))
    requested = max(20, requested)
    max_allowed = n_samples - config.min_train_sequences
    if max_allowed < 20:
        return max(1, max_allowed)
    return int(min(requested, max_allowed))


def _scale_sequences(sequences: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    original_shape = sequences.shape
    flat = sequences.reshape(-1, original_shape[-1])
    scaled = scaler.transform(flat).reshape(original_shape)
    return scaled.astype(np.float32)


def _train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    config: DirectionPredictionConfig,
    device: str,
) -> nn.Module:
    _configure_torch_runtime(device)
    model = _BiLSTMAttention(
        input_size=input_size,
        hidden_size=config.hidden_size,
        dense_size=config.dense_size,
        dropout=config.dropout,
    ).to(device)

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight_value = negatives / max(positives, 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(config.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = math.inf
    stale_epochs = 0
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * len(y_batch)
            seen += len(y_batch)

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                val_loss = float(criterion(model(x_val_t), y_val_t).item())

        if config.verbose:
            train_loss = running_loss / max(seen, 1)
            print(
                f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _predict_proba(
    model: nn.Module,
    x: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            logits = model(x_batch.to(device))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            outputs.append(probs)
    return np.concatenate(outputs).astype(np.float64)


def _tune_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    config: DirectionPredictionConfig,
) -> float:
    if len(np.unique(labels)) < 2:
        return float(config.fallback_threshold)

    thresholds = np.arange(
        config.threshold_grid_min,
        config.threshold_grid_max + config.threshold_grid_step / 2.0,
        config.threshold_grid_step,
    )
    best_threshold = float(config.fallback_threshold)
    best_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    for threshold in thresholds:
        metrics = _classification_metrics(probabilities, labels, float(threshold))
        key = (
            metrics["balanced_accuracy"],
            metrics["hit_ratio"],
            metrics["f1"],
            -abs(float(threshold) - config.fallback_threshold),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = labels.astype(int)
    pred = (probabilities >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    hit_ratio = float((pred == y).mean()) if len(y) else math.nan
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    balanced_accuracy = 0.5 * (recall + specificity)

    auc = math.nan
    pr_auc = math.nan
    if len(np.unique(y)) == 2:
        if roc_auc_score is not None:
            auc = float(roc_auc_score(y, probabilities))
        if average_precision_score is not None:
            pr_auc = float(average_precision_score(y, probabilities))

    return {
        "hit_ratio": hit_ratio,
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "threshold": float(threshold),
        "positive_rate": float(pred.mean()) if len(pred) else math.nan,
    }


def _return_stats(next_returns: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    returns = np.asarray(next_returns, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    up_returns = returns[y == 1.0]
    down_returns = returns[y == 0.0]
    abs_median = float(np.nanmedian(np.abs(returns))) if len(returns) else 0.0
    up_median = float(np.nanmedian(up_returns)) if len(up_returns) else abs_median
    down_median = (
        float(np.nanmedian(down_returns)) if len(down_returns) else -abs_median
    )
    if not np.isfinite(up_median) or up_median <= 0:
        up_median = abs_median
    if not np.isfinite(down_median) or down_median >= 0:
        down_median = -abs_median
    return {
        "up_median": float(up_median),
        "down_median": float(down_median),
        "absolute_median": float(abs_median),
        "overall_median": float(np.nanmedian(returns)) if len(returns) else 0.0,
    }


def _estimate_directional_return(
    direction: Literal["up", "down"],
    probability_up: float,
    threshold: float,
    return_stats: dict[str, float],
) -> float:
    base = (
        return_stats["up_median"]
        if direction == "up"
        else return_stats["down_median"]
    )
    margin = probability_up - threshold if direction == "up" else threshold - probability_up
    # 改进：根据概率边距动态调整，置信度越高预测越接近历史中位数
    strength = float(np.clip(margin / 0.20, 0.0, 1.0))
    # 原始：0.35 + 0.65 * strength，改进为更陡峭的曲线
    shrink = 0.20 + 0.80 * (strength ** 1.5)
    return float(base * shrink)


def _direction_sign(value: float) -> int:
    return 1 if value > 0 else 0


def _validate_config(config: DirectionPredictionConfig) -> None:
    if config.lookback < 5:
        raise ValueError("lookback must be at least 5.")
    if not 0.0 <= config.neutral_band < 0.05:
        raise ValueError("neutral_band must be in [0, 0.05).")
    if not 0.01 <= config.validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be in [0.01, 0.5].")
    if config.epochs < 1:
        raise ValueError("epochs must be positive.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if config.threshold_grid_step <= 0:
        raise ValueError("threshold_grid_step must be positive.")


def _resolve_device(requested: DeviceName) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _configure_torch_runtime(device: str) -> None:
    if device != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, (np.floating, float)):
        float_value = float(value)
        return float_value if math.isfinite(float_value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _parse_float_tuple(value: str | tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if isinstance(value, tuple):
        return tuple(float(item) for item in value)
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return ()
    return tuple(float(part) for part in parts)


def _parse_string_tuple(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _default_csv_path() -> str:
    candidates = (
        "market_data/merged_features.csv",
        "market_data/wind.csv",
        "wind.csv",
        "akshare.csv",
        "000001_sh.csv",
    )
    for candidate in candidates:
        try:
            with open(candidate, "rb"):
                return candidate
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "No CSV path was provided and none of market_data/merged_features.csv, "
        "wind.csv, akshare.csv, 000001_sh.csv exists in the current directory."
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the standalone BiLSTM-Attention direction predictor."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=SCRIPT_CSV_PATH,
        help=(
            "Path to a market CSV file. If omitted, the script uses "
            "SCRIPT_CSV_PATH, then tries market_data/merged_features.csv, "
            "wind.csv, akshare.csv, and 000001_sh.csv."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "loop_validate", "walk_forward"],
        default=SCRIPT_MODE,
    )
    parser.add_argument("--encoding", default=SCRIPT_ENCODING)
    parser.add_argument("--epochs", type=int, default=SCRIPT_EPOCHS)
    parser.add_argument("--lookback", type=int, default=SCRIPT_LOOKBACK)
    parser.add_argument("--neutral-band", type=float, default=SCRIPT_NEUTRAL_BAND)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=SCRIPT_DEVICE,
    )
    parser.add_argument(
        "--external-feature-mode",
        choices=["none", "core", "all"],
        default=SCRIPT_EXTERNAL_FEATURE_MODE,
        help=(
            "Use external columns from merged feature CSV: none=OHLCV only, "
            "core=A50/US/HK/VIX/FX/index futures, all=all safe numeric columns."
        ),
    )
    parser.add_argument(
        "--technical-feature-mode",
        choices=["none", "v1", "v1_core"],
        default=SCRIPT_TECHNICAL_FEATURE_MODE,
        help="Use technical features from technical_features.py.",
    )
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--loop-validate", action="store_true")
    parser.add_argument(
        "--initial-train-fraction",
        type=float,
        default=SCRIPT_INITIAL_TRAIN_FRACTION,
    )
    parser.add_argument("--test-size", type=int, default=SCRIPT_TEST_SIZE)
    parser.add_argument("--max-splits", type=int, default=SCRIPT_MAX_SPLITS)
    parser.add_argument("--periods", type=int, default=SCRIPT_PERIODS)
    parser.add_argument(
        "--latest-periods",
        type=int,
        default=None,
        help="Validate the latest N days and ignore configured start/end dates.",
    )
    parser.add_argument("--start-date", default=SCRIPT_START_DATE)
    parser.add_argument("--end-date", default=SCRIPT_END_DATE)
    parser.add_argument("--output", default=SCRIPT_OUTPUT_PATH)
    parser.add_argument("--diagnostics-output", default=SCRIPT_DIAGNOSTICS_OUTPUT_PATH)
    parser.add_argument("--confidence-output", default=SCRIPT_CONFIDENCE_OUTPUT_PATH)
    parser.add_argument(
        "--confidence-summary-output",
        default=SCRIPT_CONFIDENCE_SUMMARY_PATH,
    )
    parser.add_argument(
        "--high-confidence-min-base-calibration",
        type=float,
        default=SCRIPT_HIGH_CONFIDENCE_MIN_BASE_CALIBRATION,
    )
    parser.add_argument(
        "--high-confidence-min-veto-state-accuracy",
        type=float,
        default=SCRIPT_HIGH_CONFIDENCE_MIN_VETO_STATE_ACCURACY,
    )
    parser.add_argument(
        "--high-confidence-max-veto-state-rows",
        type=int,
        default=SCRIPT_HIGH_CONFIDENCE_MAX_VETO_STATE_ROWS,
        help=(
            "Optional upper bound for same-state historical samples in the "
            "high-confidence subset. Use a negative value to disable it."
        ),
    )
    parser.add_argument(
        "--signal-engine",
        choices=[
            "bilstm",
            "volatility_rule",
            "state_veto_rule",
            "stability_rule",
            "nested_ml",
            "hybrid",
            "calibrated_rule",
            "historical_selector",
        ],
        default=SCRIPT_SIGNAL_ENGINE,
    )
    parser.add_argument(
        "--rule-threshold-end-date",
        default=SCRIPT_RULE_THRESHOLD_END_DATE,
    )
    parser.add_argument(
        "--rule-calibration-start-date",
        default=SCRIPT_RULE_CALIBRATION_START_DATE,
    )
    parser.add_argument(
        "--rule-calibration-end-date",
        default=SCRIPT_RULE_CALIBRATION_END_DATE,
    )
    parser.add_argument("--rule-top-k", type=int, default=SCRIPT_RULE_TOP_K)
    parser.add_argument(
        "--volatility-rule-quantile",
        type=float,
        default=SCRIPT_VOLATILITY_RULE_QUANTILE,
    )
    parser.add_argument(
        "--nested-rule-threshold-window",
        type=int,
        default=SCRIPT_NESTED_RULE_THRESHOLD_WINDOW,
    )
    parser.add_argument(
        "--nested-rule-calibration-window",
        type=int,
        default=SCRIPT_NESTED_RULE_CALIBRATION_WINDOW,
    )
    parser.add_argument(
        "--nested-rule-min-threshold-rows",
        type=int,
        default=SCRIPT_NESTED_RULE_MIN_THRESHOLD_ROWS,
    )
    parser.add_argument(
        "--nested-rule-min-calibration-rows",
        type=int,
        default=SCRIPT_NESTED_RULE_MIN_CALIBRATION_ROWS,
    )
    parser.add_argument(
        "--state-veto-window",
        type=int,
        default=SCRIPT_STATE_VETO_WINDOW,
    )
    parser.add_argument(
        "--state-veto-state-window",
        type=int,
        default=SCRIPT_STATE_VETO_STATE_WINDOW,
    )
    parser.add_argument(
        "--state-veto-min-rows",
        type=int,
        default=SCRIPT_STATE_VETO_MIN_ROWS,
    )
    parser.add_argument(
        "--state-veto-bad-accuracy",
        type=float,
        default=SCRIPT_STATE_VETO_BAD_ACCURACY,
    )
    parser.add_argument(
        "--state-veto-quantiles",
        default=SCRIPT_STATE_VETO_QUANTILES,
        help="Comma-separated rolling quantiles used by state_veto_rule.",
    )
    parser.add_argument(
        "--stability-rule-long-window",
        type=int,
        default=SCRIPT_STABILITY_RULE_LONG_WINDOW,
    )
    parser.add_argument(
        "--stability-rule-recent-weight",
        type=float,
        default=SCRIPT_STABILITY_RULE_RECENT_WEIGHT,
    )
    parser.add_argument(
        "--stability-rule-min-edge",
        type=float,
        default=SCRIPT_STABILITY_RULE_MIN_EDGE,
    )
    parser.add_argument(
        "--stability-rule-top-k",
        type=int,
        default=SCRIPT_STABILITY_RULE_TOP_K,
    )
    parser.add_argument(
        "--nested-ml-train-window",
        type=int,
        default=SCRIPT_NESTED_ML_TRAIN_WINDOW,
    )
    parser.add_argument("--nested-ml-step", type=int, default=SCRIPT_NESTED_ML_STEP)
    parser.add_argument(
        "--nested-ml-calibration-window",
        type=int,
        default=SCRIPT_NESTED_ML_CALIBRATION_WINDOW,
    )
    parser.add_argument(
        "--nested-ml-min-train-rows",
        type=int,
        default=SCRIPT_NESTED_ML_MIN_TRAIN_ROWS,
    )
    parser.add_argument("--nested-ml-top-k", type=int, default=SCRIPT_NESTED_ML_TOP_K)
    parser.add_argument(
        "--return-magnitude-mode",
        choices=["directional_median", "range_scaled"],
        default=SCRIPT_RETURN_MAGNITUDE_MODE,
        help=(
            "How to estimate return size after direction is chosen. "
            "range_scaled keeps the predicted sign and calibrates magnitude "
            "from prior range/next-absolute-return history."
        ),
    )
    parser.add_argument(
        "--return-magnitude-window",
        type=int,
        default=SCRIPT_RETURN_MAGNITUDE_WINDOW,
    )
    parser.add_argument(
        "--return-magnitude-min-rows",
        type=int,
        default=SCRIPT_RETURN_MAGNITUDE_MIN_ROWS,
    )
    parser.add_argument(
        "--return-magnitude-grid-size",
        type=int,
        default=SCRIPT_RETURN_MAGNITUDE_GRID_SIZE,
    )
    parser.add_argument(
        "--return-magnitude-clip-low-quantile",
        type=float,
        default=SCRIPT_RETURN_MAGNITUDE_CLIP_LOW_QUANTILE,
    )
    parser.add_argument(
        "--return-magnitude-clip-high-quantile",
        type=float,
        default=SCRIPT_RETURN_MAGNITUDE_CLIP_HIGH_QUANTILE,
    )
    parser.add_argument(
        "--recent-failure-guard",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_RECENT_FAILURE_GUARD,
        help=(
            "Enable the strictly historical degraded/inverted recent-failure "
            "guard for raw direction signals."
        ),
    )
    parser.add_argument(
        "--recent-failure-window",
        type=int,
        default=SCRIPT_RECENT_FAILURE_WINDOW,
    )
    parser.add_argument(
        "--recent-failure-threshold",
        type=float,
        default=SCRIPT_RECENT_FAILURE_DEGRADE_THRESHOLD,
        help="Backward-compatible alias for --recent-failure-degrade-threshold.",
    )
    parser.add_argument(
        "--recent-failure-degrade-threshold",
        type=float,
        default=None,
        help="Long-window raw accuracy at or below this marks the signal degraded.",
    )
    parser.add_argument(
        "--recent-failure-invert-threshold",
        type=float,
        default=SCRIPT_RECENT_FAILURE_INVERT_THRESHOLD,
        help="Long-window raw accuracy at or below this allows signal inversion.",
    )
    parser.add_argument(
        "--recent-failure-short-window",
        type=int,
        default=SCRIPT_RECENT_FAILURE_SHORT_WINDOW,
    )
    parser.add_argument(
        "--recent-failure-short-threshold",
        type=float,
        default=SCRIPT_RECENT_FAILURE_SHORT_THRESHOLD,
        help="Short-window raw accuracy confirmation required for inversion.",
    )
    parser.add_argument("--selector-window", type=int, default=SCRIPT_SELECTOR_WINDOW)
    parser.add_argument(
        "--selector-min-history",
        type=int,
        default=SCRIPT_SELECTOR_MIN_HISTORY,
    )
    parser.add_argument(
        "--selector-switch-edge",
        type=float,
        default=SCRIPT_SELECTOR_SWITCH_EDGE,
    )
    parser.add_argument(
        "--selector-disagreement-window",
        type=int,
        default=SCRIPT_SELECTOR_DISAGREEMENT_WINDOW,
    )
    parser.add_argument(
        "--selector-disagreement-min-history",
        type=int,
        default=SCRIPT_SELECTOR_DISAGREEMENT_MIN_HISTORY,
    )
    parser.add_argument(
        "--selector-disagreement-edge",
        type=float,
        default=SCRIPT_SELECTOR_DISAGREEMENT_EDGE,
    )
    parser.add_argument(
        "--regime-postprocess",
        action="store_true",
        default=SCRIPT_REGIME_POSTPROCESS,
        help="Enable strict historical regime flip post-processing.",
    )
    parser.add_argument(
        "--regime-postprocess-diagnostics-output",
        default=SCRIPT_REGIME_POSTPROCESS_DIAGNOSTICS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--regime-postprocess-state-columns",
        default=SCRIPT_REGIME_POSTPROCESS_STATE_COLUMNS,
    )
    parser.add_argument(
        "--regime-postprocess-history-window",
        type=int,
        default=SCRIPT_REGIME_POSTPROCESS_HISTORY_WINDOW,
    )
    parser.add_argument(
        "--regime-postprocess-min-history",
        type=int,
        default=SCRIPT_REGIME_POSTPROCESS_MIN_HISTORY,
    )
    parser.add_argument(
        "--regime-postprocess-flip-below",
        type=float,
        default=SCRIPT_REGIME_POSTPROCESS_FLIP_BELOW,
    )
    parser.add_argument(
        "--regime-postprocess-max-flip-rate",
        type=float,
        default=SCRIPT_REGIME_POSTPROCESS_MAX_FLIP_RATE,
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=SCRIPT_VERBOSE)
    args = parser.parse_args()

    csv_path = args.csv or _default_csv_path()
    data = pd.read_csv(csv_path, encoding=args.encoding)
    config = DirectionPredictionConfig(
        epochs=args.epochs,
        lookback=args.lookback,
        neutral_band=args.neutral_band,
        device=args.device,
        external_feature_mode=args.external_feature_mode,
        technical_feature_mode=args.technical_feature_mode,
        verbose=args.verbose,
    )
    mode = args.mode
    if args.loop_validate:
        mode = "loop_validate"
    elif args.walk_forward:
        mode = "walk_forward"

    if mode == "loop_validate":
        output_path = args.output or "drp_feim_prediction_results.csv"
        start_date = args.start_date
        end_date = args.end_date
        periods = args.periods
        if args.latest_periods is not None:
            start_date = None
            end_date = None
            periods = args.latest_periods
        result = loop_validate_prediction_results(
            data,
            config=config,
            start_date=start_date,
            end_date=end_date,
            periods=periods,
            output_path=output_path,
            diagnostics_output_path=args.diagnostics_output,
            confidence_output_path=args.confidence_output,
            confidence_summary_path=args.confidence_summary_output,
            progress=not args.no_progress,
            signal_engine=args.signal_engine,
            rule_threshold_end_date=args.rule_threshold_end_date,
            rule_calibration_start_date=args.rule_calibration_start_date,
            rule_calibration_end_date=args.rule_calibration_end_date,
            rule_top_k=args.rule_top_k,
            volatility_rule_quantile=args.volatility_rule_quantile,
            nested_rule_threshold_window=args.nested_rule_threshold_window,
            nested_rule_calibration_window=args.nested_rule_calibration_window,
            nested_rule_min_threshold_rows=args.nested_rule_min_threshold_rows,
            nested_rule_min_calibration_rows=args.nested_rule_min_calibration_rows,
            state_veto_window=args.state_veto_window,
            state_veto_state_window=args.state_veto_state_window,
            state_veto_min_rows=args.state_veto_min_rows,
            state_veto_bad_accuracy=args.state_veto_bad_accuracy,
            state_veto_quantiles=_parse_float_tuple(args.state_veto_quantiles),
            high_confidence_min_base_calibration=(
                args.high_confidence_min_base_calibration
            ),
            high_confidence_min_veto_state_accuracy=(
                args.high_confidence_min_veto_state_accuracy
            ),
            high_confidence_max_veto_state_rows=(
                args.high_confidence_max_veto_state_rows
                if args.high_confidence_max_veto_state_rows is not None
                and args.high_confidence_max_veto_state_rows >= 0
                else None
            ),
            stability_rule_long_window=args.stability_rule_long_window,
            stability_rule_recent_weight=args.stability_rule_recent_weight,
            stability_rule_min_edge=args.stability_rule_min_edge,
            stability_rule_top_k=args.stability_rule_top_k,
            nested_ml_train_window=args.nested_ml_train_window,
            nested_ml_step=args.nested_ml_step,
            nested_ml_calibration_window=args.nested_ml_calibration_window,
            nested_ml_min_train_rows=args.nested_ml_min_train_rows,
            nested_ml_top_k=args.nested_ml_top_k,
            return_magnitude_mode=args.return_magnitude_mode,
            return_magnitude_window=args.return_magnitude_window,
            return_magnitude_min_rows=args.return_magnitude_min_rows,
            return_magnitude_grid_size=args.return_magnitude_grid_size,
            return_magnitude_clip_low_quantile=(
                args.return_magnitude_clip_low_quantile
            ),
            return_magnitude_clip_high_quantile=(
                args.return_magnitude_clip_high_quantile
            ),
            recent_failure_guard=args.recent_failure_guard,
            recent_failure_window=args.recent_failure_window,
            recent_failure_degrade_threshold=(
                args.recent_failure_degrade_threshold
                if args.recent_failure_degrade_threshold is not None
                else args.recent_failure_threshold
            ),
            recent_failure_invert_threshold=args.recent_failure_invert_threshold,
            recent_failure_short_window=args.recent_failure_short_window,
            recent_failure_short_threshold=args.recent_failure_short_threshold,
            selector_window=args.selector_window,
            selector_min_history=args.selector_min_history,
            selector_switch_edge=args.selector_switch_edge,
            selector_disagreement_window=args.selector_disagreement_window,
            selector_disagreement_min_history=(
                args.selector_disagreement_min_history
            ),
            selector_disagreement_edge=args.selector_disagreement_edge,
            regime_postprocess=args.regime_postprocess,
            regime_postprocess_diagnostics_output_path=(
                args.regime_postprocess_diagnostics_output
            ),
            regime_postprocess_state_columns=_parse_string_tuple(
                args.regime_postprocess_state_columns
            ),
            regime_postprocess_history_window=(
                args.regime_postprocess_history_window
            ),
            regime_postprocess_min_history=args.regime_postprocess_min_history,
            regime_postprocess_flip_below=args.regime_postprocess_flip_below,
            regime_postprocess_max_flip_rate=(
                args.regime_postprocess_max_flip_rate
            ),
        )
        print(result.to_string(index=False))
        side_stats = _direction_side_stats(result)
        print(f"saved_to={output_path}", file=sys.stderr)
        print(f"direction_accuracy={result['correct'].mean():.6f}", file=sys.stderr)
        print(
            "long_accuracy="
            f"{side_stats['long_accuracy']:.6f} "
            f"({side_stats['long_correct']}/{side_stats['long_rows']})",
            file=sys.stderr,
        )
        print(
            "short_accuracy="
            f"{side_stats['short_accuracy']:.6f} "
            f"({side_stats['short_correct']}/{side_stats['short_rows']})",
            file=sys.stderr,
        )
    elif mode == "walk_forward":
        result = walk_forward_validate(
            data,
            config=config,
            initial_train_fraction=args.initial_train_fraction,
            test_size=args.test_size,
            max_splits=args.max_splits,
        )
        print(result.to_string(index=False))
    else:
        result = predict_next_day(data, config=config)
        print(
            json.dumps(
                _json_sanitize(result),
                ensure_ascii=False,
                indent=2,
                default=_json_default,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    _main()
