"""Causal daily market direction predictor.

This module is intentionally independent from the existing prediction scripts in
the workspace. It follows the method described in the PDF:

* daily OHLCV features plus technical indicators
* 30-trading-day sequence inputs
* causal, rank-weighted rule ensemble as the accepted default signal engine
* optional BiLSTM-Attention and regularized-tree research engines
* fixed-window causal confidence and return calibration

The public one-call API is:

    result = predict_next_day(df)

Optional walk-forward validation is available through:

    validation = walk_forward_validate(df)

The default is direction-first. Returned next-return estimates are calibrated
separately and never change the recorded direction label.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from technical_features import legacy_preprocess_data as _legacy_technical_features
except Exception:  # pragma: no cover - 保证预测器可以独立运行。
    _legacy_technical_features = None

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover - sklearn 通常存在，但保持导入安全。
    average_precision_score = None
    roc_auc_score = None


DeviceName = Literal["auto", "cpu", "cuda"]
ExternalFeatureMode = Literal["none", "core", "all"]
TechnicalFeatureMode = Literal["none", "v1", "v1_core"]
ReturnMagnitudeMode = Literal["directional_median", "range_scaled"]
CliMode = Literal["predict", "loop_validate", "walk_forward"]
ConfidenceCalibrationMethod = Literal["platt", "isotonic"]
SignalEngine = Literal[
    "regularized_trees",
    "bilstm",
    "volatility_rule",
    "state_veto_rule",
    "stability_rule",
    "nested_ml",
    "hybrid",
    "calibrated_rule",
    "historical_selector",
]

_CLI_MODES: tuple[CliMode, ...] = ("predict", "loop_validate", "walk_forward")
_SIGNAL_ENGINES: tuple[SignalEngine, ...] = (
    "regularized_trees",
    "bilstm",
    "volatility_rule",
    "state_veto_rule",
    "stability_rule",
    "nested_ml",
    "hybrid",
    "calibrated_rule",
    "historical_selector",
)


# ---------------------------------------------------------------------------
# 脚本默认设置
# ---------------------------------------------------------------------------
# 直接运行本文件时可以编辑这些变量，命令行参数仍然拥有更高优先级。
#
# SCRIPT_MODE：
#   "predict"       -> 训练一次并预测下一个交易日
#   "loop_validate" -> 生成 prediction_results.csv 风格的逐日验证结果
#   "walk_forward"  -> 输出分折的滚动验证指标
SCRIPT_MODE = "loop_validate"
SCRIPT_CSV_PATH: str | None = "market_data/merged_features.csv"
SCRIPT_OUTPUT_PATH: str | None = "drp_feim_prediction_results.csv"
# 默认只保存主结果 CSV。诊断和分析明细仍可通过 API/CLI 显式指定输出路径。
SCRIPT_DIAGNOSTICS_OUTPUT_PATH: str | None = None
SCRIPT_CONFIDENCE_OUTPUT_PATH: str | None = None
SCRIPT_CONFIDENCE_SUMMARY_PATH: str | None = None
SCRIPT_CONFIDENCE_CALIBRATION_OUTPUT_PATH: str | None = None
SCRIPT_CONFIDENCE_CALIBRATION_SUMMARY_PATH: str | None = None
SCRIPT_CONFIDENCE_CALIBRATION_BIN_EDGES = (
    0.0,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.90,
    1.0,
)
# 滚动置信度校准与信号引擎自身的历史校准相互独立。
# 设置为 0 或 None 可以禁用滚动映射。默认窗口固定为 300，
# 不从评估期结果选择窗口，避免事后校准选择。
SCRIPT_CONFIDENCE_CALIBRATION_WINDOW: int | None = 300
SCRIPT_CONFIDENCE_CALIBRATION_MIN_ROWS = 60
SCRIPT_CONFIDENCE_CALIBRATION_METHOD = "platt"
SCRIPT_CONFIDENCE_CALIBRATION_COMPARE_WINDOWS: tuple[int, ...] = ()
SCRIPT_ROLLING_CONFIDENCE_OUTPUT_PATH: str | None = None
SCRIPT_ROLLING_CONFIDENCE_COMPARISON_OUTPUT_PATH: str | None = None
SCRIPT_ENCODING = "utf-8-sig"

# 循环验证的日期范围。start/end 都为 None 时，验证最近 SCRIPT_PERIODS 个可验证交易日。
SCRIPT_START_DATE: str | None = None
SCRIPT_END_DATE: str | None = None
SCRIPT_PERIODS = 20  # 回测周期

SCRIPT_EPOCHS = 10
SCRIPT_LOOKBACK = 30
SCRIPT_NEUTRAL_BAND = 0.001
SCRIPT_DEVICE: DeviceName = "auto"
SCRIPT_EXTERNAL_FEATURE_MODE: ExternalFeatureMode = "core"
SCRIPT_TECHNICAL_FEATURE_MODE: TechnicalFeatureMode = "none"
SCRIPT_VERBOSE = False
SCRIPT_SHOW_PROGRESS = True

# 循环验证使用的方向引擎：
#   "bilstm"          -> 原始 PDF 风格的 BiLSTM-Attention，支持 GPU 加速
#   "volatility_rule" -> 严格嵌套的逐日规则选择，只使用当日之前的历史特征、阈值和方向
#   "state_veto_rule" -> volatility_rule 加上严格历史低波动状态否决/反转
#                         （该状态历史表现不佳时生效）
#   "stability_rule"  -> 严格嵌套规则选择，融合近期/长期准确率并惩罚不稳定性
#   "nested_ml"       -> 基于已有特征的严格嵌套 ML 模型选择
#   "hybrid"          -> 每日严格选择 volatility_rule 或 nested_ml
#   "calibrated_rule" -> 在验证窗口之前完成历史波动率/区间规则集校准
#   "historical_selector" -> 每日严格选择 state_veto_rule 或 volatility_rule，
#                             仅在分歧时启用切换保护
# Accepted champion: see artifacts/evaluation/default_champion_v2/contract.json.
# Other engines remain explicit research/challenger choices and do not replace it.
SCRIPT_ACCEPTED_ALGORITHM_ID = "rank_weighted_state_veto_v1"
SCRIPT_SIGNAL_ENGINE = "state_veto_rule"
SCRIPT_RULE_THRESHOLD_END_DATE: str | None = "20241231"
SCRIPT_RULE_CALIBRATION_START_DATE: str | None = "20240101"
SCRIPT_RULE_CALIBRATION_END_DATE: str | None = "20241231"
SCRIPT_RULE_TOP_K = 5
# 为兼容 CLI 保留；严格 volatility_rule 现在从历史数据搜索分位数，而不是使用固定分位数。
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

# 幅度校准保持方向不变：可以调整预测收益的大小，但不能改变信号引擎选择的方向。
SCRIPT_RETURN_MAGNITUDE_MODE: ReturnMagnitudeMode = "range_scaled"
SCRIPT_RETURN_MAGNITUDE_WINDOW = 756
SCRIPT_RETURN_MAGNITUDE_MIN_ROWS = 80
SCRIPT_RETURN_MAGNITUDE_GRID_SIZE = 120
SCRIPT_RETURN_MAGNITUDE_CLIP_LOW_QUANTILE = 0.05
SCRIPT_RETURN_MAGNITUDE_CLIP_HIGH_QUANTILE = 0.95
SCRIPT_RETURN_CALIBRATION_WINDOW = 252
SCRIPT_RETURN_CALIBRATION_MIN_ROWS = 60

# 严格使用历史数据的失败保护，不使用当前行的实际收益。
# 保护机制先识别近期表现恶化，只有较短确认窗口也持续较弱时才反转当前信号。
SCRIPT_RECENT_FAILURE_GUARD = True
SCRIPT_RECENT_FAILURE_WINDOW = 15
SCRIPT_RECENT_FAILURE_DEGRADE_THRESHOLD = 0.45
SCRIPT_RECENT_FAILURE_INVERT_THRESHOLD = 0.40
SCRIPT_RECENT_FAILURE_SHORT_WINDOW = 5
SCRIPT_RECENT_FAILURE_SHORT_THRESHOLD = 0.40
# 为旧代码和旧记录保留的兼容别名。
SCRIPT_RECENT_FAILURE_THRESHOLD = SCRIPT_RECENT_FAILURE_DEGRADE_THRESHOLD

SCRIPT_SELECTOR_WINDOW = 240
SCRIPT_SELECTOR_MIN_HISTORY = 60
SCRIPT_SELECTOR_SWITCH_EDGE = 0.0
SCRIPT_SELECTOR_DISAGREEMENT_WINDOW = 240
SCRIPT_SELECTOR_DISAGREEMENT_MIN_HISTORY = 30
SCRIPT_SELECTOR_DISAGREEMENT_EDGE = 0.0

SCRIPT_REGIME_POSTPROCESS = False
SCRIPT_REGIME_POSTPROCESS_DIAGNOSTICS_OUTPUT_PATH: str | None = None
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


# ---------------------------------------------------------------------------
# 配置与内部数据模型
# ---------------------------------------------------------------------------


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
    confidence_calibrator: Any | None
    confidence_calibration_rows: int
    confidence_calibration_method: ConfidenceCalibrationMethod
    confidence_calibration_fallback: int
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


@dataclass(frozen=True, slots=True)
class _LoopValidationRange:
    """记录内部处理范围，以及返回给调用方的较窄范围。"""

    processing_start_index: int
    output_start_index: int
    end_index: int
    output_start_trade_date: int
    output_end_trade_date: int

    @property
    def output_rows(self) -> int:
        return self.end_index - self.output_start_index + 1

    @property
    def processing_warmup_rows(self) -> int:
        return self.output_start_index - self.processing_start_index


@dataclass(frozen=True, slots=True)
class _RegimePostprocessOptions:
    enabled: bool
    state_columns: tuple[str, ...]
    history_window: int
    min_history: int
    flip_below: float
    max_flip_rate: float
    diagnostics_output_path: str | None


@dataclass(frozen=True, slots=True)
class _ConfidenceCalibrationOptions:
    bin_edges: tuple[float, ...]
    window: int | None
    min_rows: int
    method: ConfidenceCalibrationMethod
    compare_windows: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    rolling_output_path: str | None
    comparison_output_path: str | None


@dataclass(frozen=True, slots=True)
class _HighConfidenceOptions:
    min_base_calibration: float
    min_veto_state_accuracy: float
    max_veto_state_rows: int | None


@dataclass(frozen=True, slots=True)
class _LoopOutputPaths:
    result: str | None
    diagnostics: str | None
    high_confidence: str | None
    high_confidence_summary: str | None
    confidence_calibration: str | None
    confidence_calibration_summary: str | None


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


# ---------------------------------------------------------------------------
# 输入结构与特征常量
# ---------------------------------------------------------------------------


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
    "_ret1_lag2",
    "_ret5",
    "_ret5_lag1",
    "_ret5_lag2",
    "_pct_chg",
    "_pct_chg_lag1",
    "_pct_chg_lag2",
    "_gap",
    "_gap_lag1",
    "_gap_lag2",
    "_intraday",
    "_intraday_lag1",
    "_intraday_lag2",
    "_range",
    "_range_lag1",
    "_range_lag2",
    "_vol20",
    "_vol20_lag1",
    "_vol20_lag2",
    "_vol_chg",
    "_vol_chg_lag1",
    "_vol_chg_lag2",
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

_INTERNAL_RESULT_CSV_COLUMNS = (
    "trade_date",
    "predicted_pct_change",
    "predicted_direction",
    "predicted_close",
    "calibrated_confidence",
    "confidence",
    "real_pct_change",
    "correct",
    "confidence_calibration_status",
    "confidence_calibration_window",
    "confidence_calibration_method",
    "confidence_calibration_rows",
    "confidence_calibration_fallback",
)
_PUBLIC_OUTPUT_COLUMN_RENAMES = {
    "trade_date": "信号日期",
    "target_trade_date": "目标交易日",
    "predicted_pct_change": "预测次日涨跌幅",
    "predicted_direction": "预测方向",
    "predicted_close": "预测次日收盘价",
    "confidence": "原始边界分数",
    "calibrated_confidence": "置信度",
    "real_pct_change": "次日实际涨跌幅",
    "correct": "方向预测正确",
    "confidence_calibration_status": "置信度校准状态",
    "confidence_calibration_window": "置信度校准窗口",
    "confidence_calibration_method": "置信度校准方法",
    "confidence_calibration_rows": "置信度校准样本数",
    "confidence_calibration_fallback": "置信度校准回退标记",
    "raw_predicted_pct_change": "原始预测次日涨跌幅",
    "pre_guard_predicted_pct_change": "失败保护前预测次日涨跌幅",
    "raw_correct": "原始方向预测正确",
    "original_predicted_pct_change": "后处理前预测次日涨跌幅",
    "postprocess_predicted_pct_change": "后处理后预测次日涨跌幅",
    "original_correct": "后处理前方向预测正确",
    "postprocess_correct": "后处理后方向预测正确",
}
_PUBLIC_RESULT_COLUMNS = tuple(
    _PUBLIC_OUTPUT_COLUMN_RENAMES[column]
    for column in _INTERNAL_RESULT_CSV_COLUMNS
)


# ---------------------------------------------------------------------------
# 公开预测 API
# ---------------------------------------------------------------------------


def predict_next_day(
    df: pd.DataFrame,
    config: DirectionPredictionConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Run the same causal pipeline as historical validation, on the latest row."""

    defaults = _build_argument_parser().parse_args([])
    cfg = config or _config_from_cli_args(defaults)
    options = _loop_validation_kwargs_from_cli_args(defaults, output_path=None)
    options.update(overrides)
    for name in tuple(options):
        if name.endswith("_path"):
            options[name] = None
    options.update(start_date=None, end_date=None, periods=1, include_latest=True, progress=False)
    frame = loop_validate_prediction_results(df, config=cfg, **options)
    latest = frame.iloc[-1]
    public = _format_result_frame_for_csv(frame).iloc[-1]
    base = _normalize_market_frame(df, cfg)
    label = int(latest["predicted_label"])
    confidence = float(latest.get("calibrated_confidence", latest["confidence"]))
    return {
        "model": options["signal_engine"],
        "algorithm_id": SCRIPT_ACCEPTED_ALGORITHM_ID if options["signal_engine"] == SCRIPT_SIGNAL_ENGINE else options["signal_engine"],
        "signal_engine": options["signal_engine"],
        "last_date": base["date"].iloc[-1].strftime("%Y-%m-%d"),
        "last_close": float(base["close"].iloc[-1]),
        "predicted_direction": "up" if label else "down",
        "predicted_label": label,
        "probability_up": confidence if label else 1.0 - confidence,
        "probability_down": 1.0 - confidence if label else confidence,
        "direction_correctness_probability": confidence,
        "probability_source": "up_down_probability_derived_from_calibrated_direction_correctness",
        "decision_threshold": None,
        "confidence": confidence,
        "calibrated_confidence": confidence,
        "raw_confidence": float(latest["confidence"]),
        "confidence_calibration_status": public["置信度校准状态"],
        "confidence_calibration_method": latest.get("confidence_calibration_method", None),
        "confidence_calibration_rows": int(latest.get("confidence_calibration_rows", 0)),
        "confidence_calibration_window": latest.get("confidence_calibration_window", None),
        "confidence_calibration_fallback": int(public["置信度校准回退标记"]),
        "estimated_next_return": float(latest["predicted_pct_change"]),
        "estimated_next_pct_change": float(latest["predicted_pct_change"] * 100.0),
        "estimated_next_close": float(latest["predicted_close"]),
        "return_calibration_scale": float(latest.get("return_calibration_scale", 1.0)),
        "return_estimate_note": "Direction and the historical calibrated return estimate are separate outputs.",
    }


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
    validation_raw_confidence = _raw_confidence_from_probability(
        val_probs,
        threshold,
    )
    validation_correct = (
        (val_probs >= threshold).astype(int) == y_all[train_end:].astype(int)
    ).astype(int)
    (
        confidence_calibrator,
        confidence_calibration_rows,
        confidence_calibration_fallback,
    ) = _fit_confidence_calibrator(
        raw_confidence=validation_raw_confidence,
        correct=validation_correct,
        min_rows=SCRIPT_CONFIDENCE_CALIBRATION_MIN_ROWS,
        method=SCRIPT_CONFIDENCE_CALIBRATION_METHOD,
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
        confidence_calibrator=confidence_calibrator,
        confidence_calibration_rows=confidence_calibration_rows,
        confidence_calibration_method=SCRIPT_CONFIDENCE_CALIBRATION_METHOD,
        confidence_calibration_fallback=confidence_calibration_fallback,
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


# ---------------------------------------------------------------------------
# 循环验证
# ---------------------------------------------------------------------------


def _validate_loop_options(
    *,
    signal_engine: str,
    return_magnitude_mode: str,
    confidence_calibration_window: int | None,
    confidence_calibration_compare_windows: tuple[int, ...],
    confidence_calibration_min_rows: int,
    confidence_calibration_method: str,
) -> tuple[int, ...]:
    if signal_engine not in _SIGNAL_ENGINES:
        choices = ", ".join(repr(engine) for engine in _SIGNAL_ENGINES)
        raise ValueError(f"signal_engine must be one of: {choices}.")
    if return_magnitude_mode not in {"directional_median", "range_scaled"}:
        raise ValueError(
            "return_magnitude_mode must be 'directional_median' or 'range_scaled'."
        )
    if confidence_calibration_min_rows < 2:
        raise ValueError("confidence_calibration_min_rows must be at least 2.")
    if confidence_calibration_method not in {"platt", "isotonic"}:
        raise ValueError("confidence_calibration_method must be 'platt' or 'isotonic'.")
    return _normalize_confidence_calibration_windows(
        confidence_calibration_window,
        confidence_calibration_compare_windows,
    )


def _resolve_loop_validation_range(
    *,
    base: pd.DataFrame,
    config: DirectionPredictionConfig,
    start_date: str | int | pd.Timestamp | None,
    end_date: str | int | pd.Timestamp | None,
    periods: int,
    regime_postprocess: bool,
    regime_history_window: int,
    regime_min_history: int,
    rolling_calibration_windows: tuple[int, ...],
    include_latest: bool = False,
) -> _LoopValidationRange:
    minimum_start_index = config.lookback + config.min_train_sequences
    output_start_index = minimum_start_index
    end_index = len(base) - (1 if include_latest else 2)

    if start_date is not None:
        start_ts = _parse_single_date(start_date, config.dayfirst)
        matching = np.flatnonzero(base["date"].ge(start_ts).to_numpy())
        if len(matching) == 0:
            raise ValueError(f"start_date {start_date!r} is after the data end.")
        output_start_index = max(output_start_index, int(matching[0]))
    if end_date is not None:
        end_ts = _parse_single_date(end_date, config.dayfirst)
        matching = np.flatnonzero(base["date"].le(end_ts).to_numpy())
        if len(matching) == 0:
            raise ValueError(f"end_date {end_date!r} is before the data start.")
        end_index = min(end_index, int(matching[-1]))
    if start_date is None and periods > 0:
        output_start_index = max(output_start_index, end_index - periods + 1)

    if output_start_index > end_index:
        raise ValueError("No validation dates remain after applying filters.")

    processing_start_index = output_start_index
    if regime_postprocess:
        normalized_history_window = max(1, int(regime_history_window))
        regime_warmup_span = max(
            1,
            normalized_history_window * 2,
            int(regime_min_history),
        )
        processing_start_index = max(
            minimum_start_index,
            output_start_index - regime_warmup_span,
        )
    if rolling_calibration_windows:
        processing_start_index = max(
            minimum_start_index,
            output_start_index - max(rolling_calibration_windows),
        )

    return _LoopValidationRange(
        processing_start_index=processing_start_index,
        output_start_index=output_start_index,
        end_index=end_index,
        output_start_trade_date=int(
            base["date"].iloc[output_start_index].strftime("%Y%m%d")
        ),
        output_end_trade_date=int(base["date"].iloc[end_index].strftime("%Y%m%d")),
    )


def _append_selector_history(
    *,
    history: list[dict[str, Any]],
    base: pd.DataFrame,
    idx: int,
    candidate_signals: dict[str, RuleSignal] | None,
    real_pct_change: float,
) -> None:
    if candidate_signals is None:
        return
    history.append(
        {
            "trade_date": int(base["date"].iloc[idx].strftime("%Y%m%d")),
            "state_veto_rule_predicted_pct_change": float(
                candidate_signals["state_veto_rule"].predicted_return
            ),
            "state_veto_rule_correct": bool(
                _direction_sign(
                    candidate_signals["state_veto_rule"].predicted_return
                )
                == _direction_sign(real_pct_change)
            ),
            "volatility_rule_predicted_pct_change": float(
                candidate_signals["volatility_rule"].predicted_return
            ),
            "volatility_rule_correct": bool(
                _direction_sign(
                    candidate_signals["volatility_rule"].predicted_return
                )
                == _direction_sign(real_pct_change)
            ),
        }
    )


def loop_validate_prediction_results(
    df: pd.DataFrame,
    config: DirectionPredictionConfig | None = None,
    *,
    start_date: str | int | pd.Timestamp | None = None,
    end_date: str | int | pd.Timestamp | None = None,
    periods: int = 60,
    include_latest: bool = False,
    output_path: str | None = None,
    diagnostics_output_path: str | None = None,
    confidence_output_path: str | None = None,
    confidence_summary_path: str | None = None,
    confidence_calibration_output_path: str | None = None,
    confidence_calibration_summary_path: str | None = None,
    confidence_calibration_bin_edges: tuple[float, ...] = (
        SCRIPT_CONFIDENCE_CALIBRATION_BIN_EDGES
    ),
    confidence_calibration_window: int | None = (
        SCRIPT_CONFIDENCE_CALIBRATION_WINDOW
    ),
    confidence_calibration_min_rows: int = SCRIPT_CONFIDENCE_CALIBRATION_MIN_ROWS,
    confidence_calibration_method: ConfidenceCalibrationMethod = (
        SCRIPT_CONFIDENCE_CALIBRATION_METHOD
    ),
    confidence_calibration_compare_windows: tuple[int, ...] = (
        SCRIPT_CONFIDENCE_CALIBRATION_COMPARE_WINDOWS
    ),
    rolling_confidence_output_path: str | None = SCRIPT_ROLLING_CONFIDENCE_OUTPUT_PATH,
    rolling_confidence_comparison_output_path: str | None = (
        SCRIPT_ROLLING_CONFIDENCE_COMPARISON_OUTPUT_PATH
    ),
    progress: bool = True,
    signal_engine: SignalEngine = SCRIPT_SIGNAL_ENGINE,
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
    return_calibration_window: int = SCRIPT_RETURN_CALIBRATION_WINDOW,
    return_calibration_min_rows: int = SCRIPT_RETURN_CALIBRATION_MIN_ROWS,
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
    """Loop over trading days and optionally save the result CSV.

    The returned DataFrame keeps its internal English column names for API
    compatibility.  CSV output uses Chinese column names, including ``信号日期``,
    ``预测次日涨跌幅``, ``预测方向``, ``预测次日收盘价``, ``置信度``,
    ``原始边界分数``, ``次日实际涨跌幅`` and ``方向预测正确``.  ``置信度`` is the
    historical-calibrated probability that the direction is correct; when the
    calibration sample is insufficient, it falls back to the raw boundary
    score and marks ``置信度校准状态`` accordingly.  ``预测方向`` is ``上涨``
    when the final predicted return is positive and ``下跌`` otherwise.  Return
    columns contain decimal returns (0.01 means 1%), not percentage points.
    ``confidence`` is a value in [0, 1] and is displayed as a percentage by the
    command-line interface.
    """

    cfg = config or DirectionPredictionConfig()
    _validate_config(cfg)
    base = _normalize_market_frame(df, cfg)
    if len(base) < cfg.lookback + cfg.min_train_sequences + 2:
        raise ValueError("Not enough rows for loop validation.")

    rolling_windows = _validate_loop_options(
        signal_engine=signal_engine,
        return_magnitude_mode=return_magnitude_mode,
        confidence_calibration_window=confidence_calibration_window,
        confidence_calibration_compare_windows=(
            confidence_calibration_compare_windows
        ),
        confidence_calibration_min_rows=confidence_calibration_min_rows,
        confidence_calibration_method=confidence_calibration_method,
    )
    if (
        return_calibration_window < 0
        or return_calibration_min_rows < 2
        or (return_calibration_window and return_calibration_window < return_calibration_min_rows)
    ):
        raise ValueError("Return calibration requires a nonnegative window and at least two rows.")
    warmup_windows = rolling_windows + ((return_calibration_window,) if return_calibration_window else ())
    validation_range = _resolve_loop_validation_range(
        base=base,
        config=cfg,
        start_date=start_date,
        end_date=end_date,
        periods=periods,
        regime_postprocess=regime_postprocess,
        regime_history_window=regime_postprocess_history_window,
        regime_min_history=regime_postprocess_min_history,
        rolling_calibration_windows=warmup_windows,
        include_latest=include_latest,
    )

    # 为整个验证过程一次性准备引擎所需的特征和缓存。
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
    model_probabilities: np.ndarray | None = None
    model_training_rows: np.ndarray | None = None
    if signal_engine == "regularized_trees":
        from regularized_direction import rolling_probabilities

        feature_frame = _clean_feature_frame(_build_features(base, cfg))
        model_probabilities, model_training_rows = rolling_probabilities(feature_frame, base["close"])
    if signal_engine in {"state_veto_rule", "historical_selector"}:
        assert nested_rule_cache is not None
        assert feature_frame is not None
        state_veto_base_signals = {}
        pre_start = max(
            cfg.lookback + cfg.min_train_sequences,
            validation_range.processing_start_index
            - max(state_veto_window, state_veto_state_window, 600),
        )
        for signal_idx in range(pre_start, validation_range.end_index + 1):
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

    # 下面的引擎分支负责生成原始方向、收益和置信度。
    def _compute_loop_signal(
        idx: int,
    ) -> tuple[
        RuleSignal | None,
        float,
        float,
        dict[str, Any],
        dict[str, RuleSignal] | None,
    ]:
        signal: RuleSignal | None = None
        selector_diagnostics: dict[str, Any] = {}
        selector_candidate_signals: dict[str, RuleSignal] | None = None
        if signal_engine == "regularized_trees":
            assert model_probabilities is not None and model_training_rows is not None
            probability = float(model_probabilities[idx])
            label = int(probability >= 0.5)
            signal = _rule_signal_from_label(
                predicted_label=label,
                base=base,
                idx=idx,
                rule_names=["regularized_trees"],
                calibration_accuracy=probability if label else 1.0 - probability,
                calibration_rows=int(model_training_rows[idx]),
                diagnostics={"model_probability_up": probability, "rule_mode": "regularized_trees"},
            )
            predicted_pct_change = signal.predicted_return
        elif signal_engine == "volatility_rule":
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
            fitted_result = prediction_to_dict(fitted)
            predicted_pct_change = float(fitted_result["estimated_next_return"])
        confidence = (
            _rule_signal_confidence(signal)
            if signal is not None
            else float(fitted_result["raw_confidence"])
        )
        return (
            signal,
            float(predicted_pct_change),
            confidence,
            selector_diagnostics,
            selector_candidate_signals,
        )

    # 在第一条处理记录前，先用严格历史数据初始化所需状态。
    needs_selector_warmup = signal_engine == "historical_selector"
    if recent_failure_guard or needs_selector_warmup:
        warmup_span = 0
        if recent_failure_guard:
            warmup_span = max(
                warmup_span,
                1,
                int(recent_failure_window),
                int(recent_failure_short_window),
            )
        if needs_selector_warmup:
            warmup_span = max(
                warmup_span,
                1,
                int(selector_window),
                int(selector_disagreement_window),
            )
        warmup_start = max(
            cfg.lookback + cfg.min_train_sequences,
            validation_range.processing_start_index - warmup_span,
        )
        for warmup_idx in range(
            warmup_start,
            validation_range.processing_start_index,
        ):
            _, warmup_predicted_pct_change, _, _, warmup_selector_signals = (
                _compute_loop_signal(warmup_idx)
            )
            warmup_real_pct_change = float(
                base["close"].iloc[warmup_idx + 1]
                / base["close"].iloc[warmup_idx]
                - 1.0
            )
            if recent_failure_guard:
                raw_direction_history.append(
                    bool(
                        _direction_sign(warmup_predicted_pct_change)
                        == _direction_sign(warmup_real_pct_change)
                    )
                )
            _append_selector_history(
                history=selector_history,
                base=base,
                idx=warmup_idx,
                candidate_signals=warmup_selector_signals,
                real_pct_change=warmup_real_pct_change,
            )

    # 按时间顺序生成预热记录和可见记录，并在每条记录完成后更新状态。
    for idx in range(
        validation_range.processing_start_index,
        validation_range.end_index + 1,
    ):
        (
            signal,
            predicted_pct_change,
            confidence,
            selector_diagnostics,
            selector_candidate_signals,
        ) = _compute_loop_signal(idx)
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
        confidence = _confidence_after_failure_guard(confidence, guard_diagnostics)
        has_outcome = idx + 1 < len(base)
        real_pct_change = (
            float(base["close"].iloc[idx + 1] / base["close"].iloc[idx] - 1.0)
            if has_outcome else np.nan
        )
        predicted_close = float(base["close"].iloc[idx] * (1.0 + predicted_pct_change))
        raw_correct = bool(
            _direction_sign(pre_guard_predicted_pct_change)
            == _direction_sign(real_pct_change)
        ) if has_outcome else None
        correct = bool(
            _direction_sign(predicted_pct_change) == _direction_sign(real_pct_change)
        ) if has_outcome else None
        if has_outcome:
            raw_direction_history.append(raw_correct)
            _append_selector_history(
                history=selector_history,
                base=base,
                idx=idx,
                candidate_signals=selector_candidate_signals,
                real_pct_change=real_pct_change,
            )
        rows.append(
            {
                "trade_date": int(base["date"].iloc[idx].strftime("%Y%m%d")),
                "predicted_pct_change": predicted_pct_change,
                "predicted_label": int(predicted_pct_change > 0),
                "predicted_close": predicted_close,
                "confidence": confidence,
                "real_pct_change": real_pct_change,
                "correct": correct,
            }
        )
        if diagnostics_output_path or regime_postprocess:
            real_label = int(real_pct_change > 0) if has_outcome else None
            diagnostics = dict(signal.diagnostics) if signal is not None else {}
            diagnostics.update(selector_diagnostics)
            diagnostics.update(magnitude_diagnostics)
            diagnostics.update(guard_diagnostics)
            selected_rules = signal.rule_names if signal is not None else []
            diagnostic_rows.append(
                {
                    "trade_date": int(base["date"].iloc[idx].strftime("%Y%m%d")),
                    "signal_engine": signal_engine,
                    "raw_predicted_label": (
                        int(signal.predicted_label)
                        if signal is not None
                        else int(raw_predicted_pct_change > 0)
                    ),
                    "predicted_label": int(predicted_pct_change > 0),
                    "real_label": real_label,
                    "raw_predicted_pct_change": raw_predicted_pct_change,
                    "pre_guard_predicted_pct_change": pre_guard_predicted_pct_change,
                    "predicted_pct_change": predicted_pct_change,
                    "confidence": confidence,
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
        if progress and idx >= validation_range.output_start_index:
            output_offset = idx - validation_range.output_start_index + 1
            visible_rows = rows[validation_range.processing_warmup_rows :]
            completed = [row["correct"] for row in visible_rows if row["correct"] is not None]
            accuracy = float(np.mean(completed)) if completed else np.nan
            print(
                f"[{signal_engine}] [{output_offset}/{validation_range.output_rows}] "
                f"{rows[-1]['trade_date']} "
                f"confidence={confidence:.2%} "
                f"correct={correct} running_accuracy={accuracy:.4f}",
                file=sys.stderr,
                flush=True,
            )

    result_frame = pd.DataFrame(
        rows,
        columns=[
            "trade_date",
            "predicted_pct_change",
            "predicted_label",
            "predicted_close",
            "confidence",
            "real_pct_change",
            "correct",
        ],
    )
    if diagnostic_rows:
        diagnostics_frame = pd.DataFrame(diagnostic_rows)
    else:
        diagnostics_frame = pd.DataFrame()

    # 后处理、校准、裁剪和写文件统一在一个有序流程中完成。
    return _finalize_loop_validation_results(
        source_frame=df,
        config=cfg,
        validation_range=validation_range,
        result_frame=result_frame,
        diagnostics_frame=diagnostics_frame,
        regime_options=_RegimePostprocessOptions(
            enabled=regime_postprocess,
            state_columns=regime_postprocess_state_columns,
            history_window=regime_postprocess_history_window,
            min_history=regime_postprocess_min_history,
            flip_below=regime_postprocess_flip_below,
            max_flip_rate=regime_postprocess_max_flip_rate,
            diagnostics_output_path=(
                regime_postprocess_diagnostics_output_path
            ),
        ),
        calibration_options=_ConfidenceCalibrationOptions(
            bin_edges=confidence_calibration_bin_edges,
            window=confidence_calibration_window,
            min_rows=confidence_calibration_min_rows,
            method=confidence_calibration_method,
            compare_windows=confidence_calibration_compare_windows,
            rolling_windows=rolling_windows,
            rolling_output_path=rolling_confidence_output_path,
            comparison_output_path=(
                rolling_confidence_comparison_output_path
            ),
        ),
        high_confidence_options=_HighConfidenceOptions(
            min_base_calibration=high_confidence_min_base_calibration,
            min_veto_state_accuracy=high_confidence_min_veto_state_accuracy,
            max_veto_state_rows=high_confidence_max_veto_state_rows,
        ),
        output_paths=_LoopOutputPaths(
            result=output_path,
            diagnostics=diagnostics_output_path,
            high_confidence=confidence_output_path,
            high_confidence_summary=confidence_summary_path,
            confidence_calibration=confidence_calibration_output_path,
            confidence_calibration_summary=(
                confidence_calibration_summary_path
            ),
        ),
        return_calibration_window=return_calibration_window,
        return_calibration_min_rows=return_calibration_min_rows,
    )


# ---------------------------------------------------------------------------
# 循环验证后处理与输出
# ---------------------------------------------------------------------------


def _finalize_loop_validation_results(
    *,
    source_frame: pd.DataFrame,
    config: DirectionPredictionConfig,
    validation_range: _LoopValidationRange,
    result_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    regime_options: _RegimePostprocessOptions,
    calibration_options: _ConfidenceCalibrationOptions,
    high_confidence_options: _HighConfidenceOptions,
    output_paths: _LoopOutputPaths,
    return_calibration_window: int = 0,
    return_calibration_min_rows: int = SCRIPT_RETURN_CALIBRATION_MIN_ROWS,
) -> pd.DataFrame:
    result_frame, diagnostics_frame, postprocess_diagnostics = (
        _apply_loop_regime_postprocess(
            source_frame=source_frame,
            config=config,
            result_frame=result_frame,
            diagnostics_frame=diagnostics_frame,
            options=regime_options,
        )
    )
    result_frame, rolling_comparison = _apply_loop_confidence_calibration(
        result_frame=result_frame,
        validation_range=validation_range,
        options=calibration_options,
    )
    # Freeze the final direction before shrinking the numerical return estimate.
    result_frame["predicted_label"] = (result_frame["predicted_pct_change"] > 0).astype(int)
    if return_calibration_window:
        from return_calibration import calibrate_returns

        result_frame = calibrate_returns(
            result_frame, window=return_calibration_window, min_rows=return_calibration_min_rows
        )
    result_frame, diagnostics_frame = _trim_loop_validation_frames(
        result_frame=result_frame,
        diagnostics_frame=diagnostics_frame,
        validation_range=validation_range,
    )
    _write_loop_validation_outputs(
        result_frame=result_frame,
        diagnostics_frame=diagnostics_frame,
        postprocess_diagnostics=postprocess_diagnostics,
        rolling_comparison=rolling_comparison,
        validation_range=validation_range,
        regime_options=regime_options,
        calibration_options=calibration_options,
        high_confidence_options=high_confidence_options,
        output_paths=output_paths,
    )
    return result_frame


def _apply_loop_regime_postprocess(
    *,
    source_frame: pd.DataFrame,
    config: DirectionPredictionConfig,
    result_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    options: _RegimePostprocessOptions,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not options.enabled:
        return result_frame, diagnostics_frame, pd.DataFrame()

    regime_config = replace(config, external_feature_mode="all")
    regime_base = _normalize_market_frame(source_frame, regime_config)
    regime_source = _build_regime_postprocess_frame(
        result_frame=result_frame,
        base=regime_base,
        diagnostics_frame=diagnostics_frame,
    )
    result_frame, postprocess_diagnostics = _apply_regime_postprocess(
        regime_source,
        state_columns=options.state_columns,
        history_window=options.history_window,
        min_history=options.min_history,
        flip_below=options.flip_below,
        max_flip_rate=options.max_flip_rate,
    )
    postprocess_extra = postprocess_diagnostics.drop(
        columns=["original_correct", "postprocess_correct"],
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
    return result_frame, diagnostics_frame, postprocess_diagnostics


def _apply_loop_confidence_calibration(
    *,
    result_frame: pd.DataFrame,
    validation_range: _LoopValidationRange,
    options: _ConfidenceCalibrationOptions,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not options.rolling_windows:
        return result_frame, pd.DataFrame()

    comparison = pd.DataFrame()
    if options.compare_windows:
        comparison, rolling_frames = compare_rolling_confidence_calibration(
            result_frame,
            windows=options.rolling_windows,
            min_rows=options.min_rows,
            method=options.method,
            evaluation_start_trade_date=validation_range.output_start_trade_date,
            evaluation_end_trade_date=validation_range.output_end_trade_date,
            bin_edges=options.bin_edges,
        )
    else:
        rolling_frames = {
            window: _apply_rolling_confidence_calibration(
                result_frame,
                window=window,
                min_rows=options.min_rows,
                method=options.method,
            )
            for window in options.rolling_windows
        }

    # Evaluation-period rankings are diagnostics, never a production selector.
    primary_window = (
        int(options.window)
        if options.window is not None and int(options.window) > 0
        else None
    )
    if not comparison.empty:
        comparison["used_for_output"] = comparison["window"].eq(primary_window)
    if primary_window is None:
        return result_frame, comparison
    return rolling_frames[primary_window], comparison


def _trim_loop_validation_frames(
    *,
    result_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    validation_range: _LoopValidationRange,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frame = result_frame.loc[
        result_frame["trade_date"].between(
            validation_range.output_start_trade_date,
            validation_range.output_end_trade_date,
        )
    ].reset_index(drop=True)
    if diagnostics_frame.empty:
        return result_frame, diagnostics_frame

    diagnostics_frame = diagnostics_frame.loc[
        diagnostics_frame["trade_date"].between(
            validation_range.output_start_trade_date,
            validation_range.output_end_trade_date,
        )
    ].reset_index(drop=True)
    final_by_date = result_frame.set_index("trade_date")
    synchronized_columns = (
        "predicted_pct_change",
        "predicted_label",
        "confidence",
        "calibrated_confidence",
        "real_pct_change",
        "correct",
    )
    for column in synchronized_columns:
        if column in final_by_date.columns:
            diagnostics_frame[column] = diagnostics_frame["trade_date"].map(
                final_by_date[column]
            )
    if "predicted_label" not in final_by_date:
        diagnostics_frame["predicted_label"] = (
            diagnostics_frame["predicted_pct_change"] > 0
        ).astype(int)
    diagnostics_frame["real_label"] = (
        diagnostics_frame["real_pct_change"] > 0
    ).astype("Int64").where(diagnostics_frame["real_pct_change"].notna())
    return result_frame, diagnostics_frame


def _write_loop_validation_outputs(
    *,
    result_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    postprocess_diagnostics: pd.DataFrame,
    rolling_comparison: pd.DataFrame,
    validation_range: _LoopValidationRange,
    regime_options: _RegimePostprocessOptions,
    calibration_options: _ConfidenceCalibrationOptions,
    high_confidence_options: _HighConfidenceOptions,
    output_paths: _LoopOutputPaths,
) -> None:
    public_result_frame = _format_result_frame_for_csv(result_frame)
    if regime_options.enabled and regime_options.diagnostics_output_path:
        visible_postprocess_diagnostics = postprocess_diagnostics.loc[
            postprocess_diagnostics["trade_date"].between(
                validation_range.output_start_trade_date,
                validation_range.output_end_trade_date,
            )
        ].reset_index(drop=True)
        visible_postprocess_diagnostics.to_csv(
            regime_options.diagnostics_output_path,
            index=False,
            encoding="utf-8-sig",
        )
    if calibration_options.rolling_output_path and calibration_options.rolling_windows:
        public_result_frame.to_csv(
            calibration_options.rolling_output_path,
            index=False,
            encoding="utf-8-sig",
        )
    if calibration_options.comparison_output_path and not rolling_comparison.empty:
        rolling_comparison.to_csv(
            calibration_options.comparison_output_path,
            index=False,
            encoding="utf-8-sig",
        )
    if output_paths.result:
        public_result_frame.to_csv(
            output_paths.result,
            index=False,
            encoding="utf-8-sig",
        )
    if output_paths.diagnostics:
        diagnostics_frame.to_csv(
            output_paths.diagnostics,
            index=False,
            encoding="utf-8-sig",
        )

    if output_paths.high_confidence or output_paths.high_confidence_summary:
        report_diagnostics = (
            diagnostics_frame
            if not diagnostics_frame.empty
            else _build_minimal_diagnostics(result_frame)
        )
        high_confidence_frame, confidence_summary = _build_high_confidence_outputs(
            result_frame=result_frame,
            diagnostics_frame=report_diagnostics,
            min_base_calibration=high_confidence_options.min_base_calibration,
            min_veto_state_accuracy=(
                high_confidence_options.min_veto_state_accuracy
            ),
            max_veto_state_rows=high_confidence_options.max_veto_state_rows,
        )
        if output_paths.high_confidence:
            high_confidence_frame.to_csv(
                output_paths.high_confidence,
                index=False,
                encoding="utf-8-sig",
            )
        if output_paths.high_confidence_summary:
            confidence_summary.to_csv(
                output_paths.high_confidence_summary,
                index=False,
                encoding="utf-8-sig",
            )

    if (
        output_paths.confidence_calibration
        or output_paths.confidence_calibration_summary
    ):
        calibration_by_bin, calibration_summary = confidence_calibration_report(
            result_frame,
            bin_edges=calibration_options.bin_edges,
        )
        if output_paths.confidence_calibration:
            calibration_by_bin.to_csv(
                output_paths.confidence_calibration,
                index=False,
                encoding="utf-8-sig",
            )
        if output_paths.confidence_calibration_summary:
            calibration_summary.to_csv(
                output_paths.confidence_calibration_summary,
                index=False,
                encoding="utf-8-sig",
            )


def _format_result_frame_for_csv(result_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert an internal result frame to the shared Chinese CSV schema."""

    public_result_frame = result_frame.copy()
    raw_confidence = pd.to_numeric(
        public_result_frame["confidence"], errors="coerce"
    )
    direction_values = public_result_frame.get("predicted_label", public_result_frame["predicted_pct_change"])
    predicted_direction = np.where(
        pd.to_numeric(direction_values, errors="coerce") > 0,
        "上涨",
        "下跌",
    )
    public_result_frame["predicted_direction"] = predicted_direction
    if "calibrated_confidence" not in public_result_frame.columns:
        public_result_frame["calibrated_confidence"] = raw_confidence
        public_result_frame["confidence_calibration_fallback"] = 1
    else:
        calibrated_confidence = pd.to_numeric(
            public_result_frame["calibrated_confidence"], errors="coerce"
        )
        public_result_frame["calibrated_confidence"] = calibrated_confidence.fillna(
            raw_confidence
        )
        if "confidence_calibration_fallback" not in public_result_frame.columns:
            public_result_frame["confidence_calibration_fallback"] = 1
    fallback = pd.to_numeric(
        public_result_frame["confidence_calibration_fallback"], errors="coerce"
    ).fillna(1).astype(int)
    public_result_frame["confidence_calibration_fallback"] = fallback
    if "confidence_calibration_rows" not in public_result_frame.columns:
        public_result_frame["confidence_calibration_rows"] = 0
    public_result_frame["confidence_calibration_status"] = np.where(
        fallback.to_numpy() == 0,
        "已校准",
        "未校准（原始边界分数）",
    )
    for column in _INTERNAL_RESULT_CSV_COLUMNS:
        if column not in public_result_frame.columns:
            public_result_frame[column] = pd.NA
    public_result_frame = public_result_frame.loc[
        :, list(_INTERNAL_RESULT_CSV_COLUMNS)
    ]
    public_result_frame = public_result_frame.rename(
        columns=_PUBLIC_OUTPUT_COLUMN_RENAMES
    )
    return public_result_frame


# ---------------------------------------------------------------------------
# 置信度报告与滚动校准
# ---------------------------------------------------------------------------


def _build_minimal_diagnostics(result_frame: pd.DataFrame) -> pd.DataFrame:
    diagnostics = result_frame.copy()
    if "predicted_label" not in diagnostics:
        diagnostics["predicted_label"] = (diagnostics["predicted_pct_change"] > 0).astype(int)
    diagnostics["real_label"] = (diagnostics["real_pct_change"] > 0).astype("Int64").where(
        diagnostics["real_pct_change"].notna()
    )
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
    if "confidence" in result_frame.columns:
        result_columns.insert(3, "confidence")
    if "calibrated_confidence" in result_frame.columns:
        result_columns.insert(4 if "confidence" in result_frame.columns else 3, "calibrated_confidence")
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


def confidence_calibration_report(
    result_frame: pd.DataFrame,
    *,
    confidence_column: str = "confidence",
    bin_edges: tuple[float, ...] = SCRIPT_CONFIDENCE_CALIBRATION_BIN_EDGES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare confidence values with realized direction accuracy.

    The calibration table groups predictions by their *predicted* confidence.
    A well-calibrated confidence score has a bin accuracy close to the bin's
    mean confidence.  All metrics are computed from realized ``correct``
    outcomes, so this check is only meaningful for completed validation rows.
    """

    required = {confidence_column, "correct"}
    missing = sorted(required.difference(result_frame.columns))
    if missing:
        raise ValueError(
            "Confidence calibration requires columns: " + ", ".join(missing)
        )

    edges = np.asarray(tuple(float(edge) for edge in bin_edges), dtype=float)
    if (
        len(edges) < 2
        or not np.isfinite(edges).all()
        or not np.isclose(edges[0], 0.0)
        or not np.isclose(edges[-1], 1.0)
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError(
            "bin_edges must be strictly increasing, finite, and start at 0 and end at 1."
        )

    confidence = pd.to_numeric(result_frame[confidence_column], errors="coerce")
    correct = result_frame["correct"].astype("boolean")
    valid = confidence.notna() & correct.notna()
    confidence_values = confidence.loc[valid].to_numpy(dtype=float)
    correct_values = correct.loc[valid].astype(float).to_numpy()
    if len(confidence_values):
        if ((confidence_values < 0.0) | (confidence_values > 1.0)).any():
            raise ValueError("confidence values must be within [0, 1].")

    labels = [
        f"[{edges[index]:.2f}, {edges[index + 1]:.2f})"
        for index in range(len(edges) - 1)
    ]
    bin_indexes = np.searchsorted(edges, confidence_values, side="right") - 1
    bin_indexes = np.clip(bin_indexes, 0, len(edges) - 2)
    bin_rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        mask = bin_indexes == index
        rows = int(mask.sum())
        mean_confidence = (
            float(confidence_values[mask].mean()) if rows else np.nan
        )
        accuracy = float(correct_values[mask].mean()) if rows else np.nan
        gap = accuracy - mean_confidence if rows else np.nan
        bin_rows.append(
            {
                "confidence_bin": label,
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "rows": rows,
                "mean_confidence": mean_confidence,
                "direction_accuracy": accuracy,
                "calibration_gap": gap,
                "absolute_calibration_gap": abs(gap) if rows else np.nan,
            }
        )

    rows_used = len(confidence_values)
    mean_confidence = (
        float(confidence_values.mean()) if rows_used else np.nan
    )
    direction_accuracy = float(correct_values.mean()) if rows_used else np.nan
    calibration_gap = (
        direction_accuracy - mean_confidence if rows_used else np.nan
    )
    absolute_gaps = np.asarray(
        [row["absolute_calibration_gap"] for row in bin_rows if row["rows"]],
        dtype=float,
    )
    weights = np.asarray(
        [row["rows"] / rows_used for row in bin_rows if row["rows"]],
        dtype=float,
    )
    expected_calibration_error = (
        float(np.dot(weights, absolute_gaps)) if rows_used else np.nan
    )
    max_calibration_error = float(absolute_gaps.max()) if len(absolute_gaps) else np.nan
    brier_score = (
        float(np.mean((confidence_values - correct_values) ** 2))
        if rows_used
        else np.nan
    )
    summary = pd.DataFrame(
        [
            {
                "rows_total": int(len(result_frame)),
                "rows_used": rows_used,
                "rows_missing_confidence_or_result": int(len(result_frame) - rows_used),
                "mean_confidence": mean_confidence,
                "direction_accuracy": direction_accuracy,
                "calibration_gap": calibration_gap,
                "absolute_calibration_gap": abs(calibration_gap)
                if rows_used
                else np.nan,
                "expected_calibration_error": expected_calibration_error,
                "max_calibration_error": max_calibration_error,
                "brier_score": brier_score,
            }
        ]
    )
    return pd.DataFrame(bin_rows), summary


def _normalize_confidence_calibration_windows(
    primary_window: int | None,
    comparison_windows: tuple[int, ...],
) -> tuple[int, ...]:
    windows: list[int] = []
    if primary_window is not None and int(primary_window) > 0:
        windows.append(int(primary_window))
    for window in comparison_windows:
        normalized = int(window)
        if normalized <= 0:
            raise ValueError("confidence calibration windows must be positive.")
        if normalized not in windows:
            windows.append(normalized)
    return tuple(windows)


def _raw_confidence_from_probability(
    probability: float | np.ndarray,
    threshold: float,
) -> float | np.ndarray:
    """Return the uncalibrated distance-from-threshold confidence score."""

    values = np.asarray(probability, dtype=float)
    margin = np.where(values >= float(threshold), values - threshold, threshold - values)
    raw_confidence = np.clip(margin / 0.20, 0.0, 1.0)
    return float(raw_confidence) if raw_confidence.ndim == 0 else raw_confidence


def _fit_confidence_calibrator(
    *,
    raw_confidence: np.ndarray,
    correct: np.ndarray,
    min_rows: int,
    method: ConfidenceCalibrationMethod,
) -> tuple[Any | None, int, int]:
    """Fit a confidence-to-correctness mapper on completed predictions only."""

    if min_rows < 2:
        raise ValueError("confidence calibration min_rows must be at least 2.")
    if method not in {"platt", "isotonic"}:
        raise ValueError("confidence_calibration_method must be 'platt' or 'isotonic'.")

    confidence_values = np.asarray(raw_confidence, dtype=float).reshape(-1)
    correct_values = np.asarray(correct, dtype=float).reshape(-1)
    if len(confidence_values) != len(correct_values):
        raise ValueError("raw_confidence and correct must have the same length.")
    valid = np.isfinite(confidence_values) & np.isfinite(correct_values)
    confidence_values = confidence_values[valid]
    correct_values = correct_values[valid].astype(int)
    rows = int(len(confidence_values))
    if rows < min_rows or len(np.unique(correct_values)) < 2:
        return None, rows, 1

    try:
        if method == "platt":
            calibrator: Any = LogisticRegression(
                solver="lbfgs",
                C=1.0,
                max_iter=200,
            )
            calibrator.fit(confidence_values.reshape(-1, 1), correct_values)
        else:
            calibrator = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            calibrator.fit(confidence_values, correct_values)
    except (ValueError, TypeError):
        return None, rows, 1
    return calibrator, rows, 0


def _apply_confidence_calibrator(
    raw_confidence: float,
    calibrator: Any | None,
) -> float:
    """Apply a fitted calibrator, falling back to the raw score if needed."""

    raw_value = float(np.clip(raw_confidence, 0.0, 1.0))
    if calibrator is None:
        return raw_value
    try:
        if hasattr(calibrator, "predict_proba"):
            calibrated = float(
                calibrator.predict_proba([[raw_value]])[0, 1]
            )
        else:
            calibrated = float(calibrator.predict([raw_value])[0])
    except (ValueError, TypeError, IndexError):
        return raw_value
    return float(np.clip(calibrated, 0.0, 1.0))


def _apply_rolling_confidence_calibration(
    result_frame: pd.DataFrame,
    *,
    window: int,
    min_rows: int,
    method: ConfidenceCalibrationMethod,
) -> pd.DataFrame:
    """Fit a confidence-to-correctness mapper using prior rows only."""

    if window < 1:
        raise ValueError("confidence calibration window must be positive.")
    if min_rows < 2:
        raise ValueError("confidence calibration min_rows must be at least 2.")
    if method not in {"platt", "isotonic"}:
        raise ValueError("confidence calibration method must be 'platt' or 'isotonic'.")
    if not {"confidence", "correct"}.issubset(result_frame.columns):
        raise ValueError(
            "Rolling confidence calibration requires confidence and correct columns."
        )

    frame = result_frame.copy().reset_index(drop=True)
    raw_confidence = pd.to_numeric(frame["confidence"], errors="coerce")
    calibrated = raw_confidence.to_numpy(dtype=float).copy()
    fit_rows = np.zeros(len(frame), dtype=int)
    fallback = np.ones(len(frame), dtype=int)
    min_fit_rows = min(int(min_rows), int(window))

    for index in range(len(frame)):
        history = frame.iloc[max(0, index - window) : index]
        history_confidence = pd.to_numeric(history["confidence"], errors="coerce")
        history_correct = history["correct"].astype("boolean")
        valid = history_confidence.notna() & history_correct.notna()
        fit_rows[index] = int(valid.sum())
        if fit_rows[index] < min_fit_rows:
            continue

        x_history = history_confidence.loc[valid].to_numpy(dtype=float)
        y_history = history_correct.loc[valid].astype(int).to_numpy()
        if len(np.unique(y_history)) < 2 or not np.isfinite(raw_confidence.iloc[index]):
            continue
        try:
            if method == "platt":
                calibrator = LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    max_iter=200,
                )
                calibrator.fit(x_history.reshape(-1, 1), y_history)
                calibrated[index] = float(
                    calibrator.predict_proba([[float(raw_confidence.iloc[index])]])[0, 1]
                )
            else:
                calibrator = IsotonicRegression(
                    y_min=0.0,
                    y_max=1.0,
                    out_of_bounds="clip",
                )
                calibrator.fit(x_history, y_history)
                calibrated[index] = float(
                    calibrator.predict([float(raw_confidence.iloc[index])])[0]
                )
        except (ValueError, TypeError):
            continue
        fallback[index] = 0

    frame["calibrated_confidence"] = np.clip(calibrated, 0.0, 1.0)
    frame["confidence_calibration_window"] = int(window)
    frame["confidence_calibration_method"] = method
    frame["confidence_calibration_rows"] = fit_rows
    frame["confidence_calibration_fallback"] = fallback
    return frame


def compare_rolling_confidence_calibration(
    result_frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (120, 300, 600),
    min_rows: int = SCRIPT_CONFIDENCE_CALIBRATION_MIN_ROWS,
    method: ConfidenceCalibrationMethod = SCRIPT_CONFIDENCE_CALIBRATION_METHOD,
    evaluation_start_trade_date: int | None = None,
    evaluation_end_trade_date: int | None = None,
    bin_edges: tuple[float, ...] = SCRIPT_CONFIDENCE_CALIBRATION_BIN_EDGES,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Compare rolling calibrators on a later, strictly out-of-sample period."""

    normalized_windows = _normalize_confidence_calibration_windows(None, windows)
    if not normalized_windows:
        raise ValueError("At least one confidence calibration window is required.")

    calibrated_frames: dict[int, pd.DataFrame] = {}
    comparison_rows: list[dict[str, Any]] = []
    for window in normalized_windows:
        calibrated_frame = _apply_rolling_confidence_calibration(
            result_frame,
            window=window,
            min_rows=min_rows,
            method=method,
        )
        calibrated_frames[window] = calibrated_frame
        evaluation_mask = calibrated_frame["confidence_calibration_fallback"].eq(0)
        if evaluation_start_trade_date is not None:
            evaluation_mask &= calibrated_frame["trade_date"].ge(
                int(evaluation_start_trade_date)
            )
        if evaluation_end_trade_date is not None:
            evaluation_mask &= calibrated_frame["trade_date"].le(
                int(evaluation_end_trade_date)
            )
        evaluation_frame = calibrated_frame.loc[evaluation_mask]
        if evaluation_frame.empty:
            summary = {
                "window": int(window),
                "method": method,
                "evaluation_rows": 0,
                "mean_calibrated_confidence": np.nan,
                "direction_accuracy": np.nan,
                "calibration_gap": np.nan,
                "expected_calibration_error": np.nan,
                "max_calibration_error": np.nan,
                "brier_score": np.nan,
            }
        else:
            _, metrics = confidence_calibration_report(
                evaluation_frame,
                confidence_column="calibrated_confidence",
                bin_edges=bin_edges,
            )
            metric_row = metrics.iloc[0]
            summary = {
                "window": int(window),
                "method": method,
                "evaluation_rows": int(metric_row["rows_used"]),
                "mean_calibrated_confidence": float(metric_row["mean_confidence"]),
                "direction_accuracy": float(metric_row["direction_accuracy"]),
                "calibration_gap": float(metric_row["calibration_gap"]),
                "expected_calibration_error": float(
                    metric_row["expected_calibration_error"]
                ),
                "max_calibration_error": float(metric_row["max_calibration_error"]),
                "brier_score": float(metric_row["brier_score"]),
            }
        comparison_rows.append(summary)

    comparison = pd.DataFrame(comparison_rows)
    comparison["selected"] = False
    valid = comparison["evaluation_rows"].gt(0)
    if valid.any():
        selected_index = (
            comparison.loc[valid]
            .sort_values(
                ["expected_calibration_error", "brier_score", "window"],
                na_position="last",
            )
            .index[0]
        )
        comparison.loc[selected_index, "selected"] = True
    comparison["selection_rule"] = "min_ece_then_brier"
    return comparison, calibrated_frames


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
    predicted = pd.to_numeric(frame.get("predicted_label", frame["predicted_pct_change"]), errors="coerce")
    correct = frame["correct"].astype("boolean")
    completed = correct.notna()
    long_mask = (predicted > 0) & completed
    short_mask = (predicted <= 0) & completed
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


# ---------------------------------------------------------------------------
# 历史选择器与状态后处理
# ---------------------------------------------------------------------------


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
    context = _add_regime_market_states(context)

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
                    "confidence",
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
    return _add_regime_diagnostic_states(merged)


def _add_regime_postprocess_states(frame: pd.DataFrame) -> pd.DataFrame:
    """Add all regime states when the caller already has a full history frame."""

    return _add_regime_diagnostic_states(_add_regime_market_states(frame))


def _add_regime_market_states(frame: pd.DataFrame) -> pd.DataFrame:
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
    return frame


def _add_regime_diagnostic_states(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
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
    history_window = max(1, int(history_window))
    flip_history: list[bool] = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        selected_key = _select_regime_flip_key(
            frame=frame,
            idx=idx,
            state_columns=state_columns,
            history_window=history_window,
            min_history=min_history,
            flip_below=flip_below,
        )
        flip_history_limit = max(0, history_window - 1)
        recent_flips = (
            flip_history[-flip_history_limit:] if flip_history_limit else []
        )
        projected_flip_rate = (sum(recent_flips) + 1) / (len(recent_flips) + 1)
        flip = bool(selected_key is not None and projected_flip_rate <= max_flip_rate)
        flip_history.append(flip)
        original_pred = float(row["predicted_pct_change"])
        original_confidence = float(row.get("confidence", np.nan))
        real = float(row["real_pct_change"])
        predicted_pct_change = -original_pred if flip else original_pred
        confidence = original_confidence
        if flip and selected_key is not None:
            historical_accuracy = float(selected_key["accuracy"])
            if np.isfinite(historical_accuracy):
                confidence = float(np.clip(1.0 - historical_accuracy, 0.0, 1.0))
        close = float(row["close"]) if pd.notna(row.get("close", np.nan)) else np.nan
        predicted_close = (
            close * (1.0 + predicted_pct_change)
            if np.isfinite(close)
            else float(row["predicted_close"])
        )
        correct = bool(
            _direction_sign(predicted_pct_change) == _direction_sign(real)
        ) if np.isfinite(real) else None
        rows.append(
            {
                "trade_date": int(row["trade_date"]),
                "predicted_pct_change": predicted_pct_change,
                "predicted_close": float(predicted_close),
                "confidence": confidence,
                "real_pct_change": real,
                "correct": correct,
            }
        )
        original_correct = bool(row["correct"]) if pd.notna(row["correct"]) else None
        diagnostics.append(
            {
                "trade_date": int(row["trade_date"]),
                "regime_postprocess_enabled": 1,
                "regime_postprocess_flipped": int(flip),
                "original_correct": original_correct,
                "postprocess_correct": correct,
                "original_predicted_pct_change": original_pred,
                "postprocess_predicted_pct_change": predicted_pct_change,
                "original_confidence": original_confidence,
                "postprocess_confidence": confidence,
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


# ---------------------------------------------------------------------------
# 信号引擎与收益控制
# ---------------------------------------------------------------------------


def prediction_to_dict(fitted: FittedDirectionModel) -> dict[str, Any]:
    """Convert a fitted model result into a serializable prediction summary."""

    prob_up = fitted.probability_up
    threshold = fitted.threshold
    direction = "up" if prob_up >= threshold else "down"
    direction_label = 1 if direction == "up" else 0
    margin = prob_up - threshold if direction == "up" else threshold - prob_up
    raw_confidence = float(_raw_confidence_from_probability(prob_up, threshold))
    calibrated_confidence = _apply_confidence_calibrator(
        raw_confidence,
        fitted.confidence_calibrator,
    )
    confidence_status = (
        "已校准"
        if fitted.confidence_calibration_fallback == 0
        else "未校准（原始边界分数）"
    )
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
        # confidence 是面向用户的历史校准后方向正确概率。
        "confidence": calibrated_confidence,
        "raw_confidence": raw_confidence,
        "calibrated_confidence": calibrated_confidence,
        "confidence_calibration_status": confidence_status,
        "confidence_calibration_rows": int(fitted.confidence_calibration_rows),
        "confidence_calibration_method": fitted.confidence_calibration_method,
        "confidence_calibration_fallback": int(
            fitted.confidence_calibration_fallback
        ),
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

    latest_historical_date = dates.iloc[max(0, idx - 1)]
    if threshold_end_date is None:
        threshold_end = dates.iloc[max(0, idx - 252)]
    else:
        threshold_end = min(
            _parse_single_date(threshold_end_date, config.dayfirst),
            latest_historical_date,
        )
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
    historical_mask = pd.Series(np.arange(len(base)) < idx, index=base.index)
    threshold_mask = valid_feature_mask & dates.le(threshold_end) & historical_mask
    calibration_mask = (
        valid_feature_mask
        & next_returns.notna()
        & dates.ge(calibration_start)
        & dates.le(calibration_end)
        & historical_mask
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
    ranked = scored[: max(1, top_k)]

    # A complementary threshold can become the same signal after the losing
    # side is reversed. Preserve the established rank-derived weight, while
    # exposing identical causal prediction paths as one expert. Summing the
    # rank weights makes this algebraically equivalent to the previous vote.
    grouped: dict[bytes, dict[str, Any]] = {}
    signature_idx = np.append(calibration_idx, idx)
    for score, name, prediction in ranked:
        signature = prediction[signature_idx].astype(np.int8, copy=False).tobytes()
        weight = max(score - 0.5, 0.001)
        expert = grouped.get(signature)
        if expert is None:
            grouped[signature] = {
                "prediction": prediction,
                "weight": weight,
                "names": [name],
                "scores": [score],
            }
        else:
            expert["weight"] += weight
            expert["names"].append(name)
            expert["scores"].append(score)
    experts = list(grouped.values())
    weights = np.asarray([float(expert["weight"]) for expert in experts])
    votes = np.asarray([expert["prediction"][idx] for expert in experts], dtype=float)
    vote_score = float(np.dot(votes, weights) / weights.sum())
    predicted_label = int(vote_score >= 0.5)

    selected_matrix = np.vstack([expert["prediction"] for expert in experts])
    calibration_votes = selected_matrix[:, calibration_idx]
    calibration_valid = (calibration_votes >= 0).all(axis=0)
    if calibration_valid.any():
        ensemble_calibration_pred = (
            np.dot(weights, calibration_votes[:, calibration_valid]) / weights.sum()
            >= 0.5
        ).astype(int)
        calibration_accuracy = float(
            (ensemble_calibration_pred == y_cal[calibration_valid]).mean()
        )
    else:
        calibration_accuracy = float(ranked[0][0])

    return _rule_signal_from_label(
        predicted_label=predicted_label,
        base=base,
        idx=idx,
        rule_names=[
            "+".join(expert["names"])
            + f":cal_acc={max(expert['scores']):.3f}:rank_weight={expert['weight']:.3f}"
            for expert in experts
        ],
        calibration_accuracy=calibration_accuracy,
        calibration_rows=len(calibration_idx),
        diagnostics={
            "rule_mode": "rank_weighted_rule",
            "selected_ranked_rule_count": len(ranked),
            "selected_expert_count": len(experts),
            "selected_rule_count": len(experts),
            "top_rule_score": float(ranked[0][0]),
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
        if not np.isfinite(x_all[idx]).all():
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


def _rule_signal_confidence(signal: RuleSignal) -> float:
    """Estimate the historical reliability of the direction actually returned."""

    confidence = float(np.clip(signal.calibration_accuracy, 0.0, 1.0))
    if bool(signal.diagnostics.get("veto_applied", 0)):
        confidence = 1.0 - confidence
    return float(confidence)


def _confidence_after_failure_guard(
    confidence: float,
    diagnostics: dict[str, Any],
) -> float:
    """Keep confidence aligned when the recent-failure guard reverses a signal."""

    if not bool(diagnostics.get("recent_failure_guard_applied", 0)):
        return float(np.clip(confidence, 0.0, 1.0))
    recent_accuracy = float(
        diagnostics.get("recent_failure_guard_accuracy", np.nan)
    )
    if np.isfinite(recent_accuracy):
        return float(np.clip(1.0 - recent_accuracy, 0.0, 1.0))
    return float(np.clip(1.0 - confidence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 行情数据准备与神经网络训练
# ---------------------------------------------------------------------------


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

    # 特征在当前收盘后计算，因此当天 OHLCV 可以用于预测下一个交易日；
    # 规则阈值仍然使用 shift(1) 避免未来信息。
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
            or column.endswith("_ret1_lag2")
            or column.endswith("_pct_chg")
            or column.endswith("_pct_chg_lag1")
            or column.endswith("_pct_chg_lag2")
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


# ---------------------------------------------------------------------------
# 运行时与序列化通用工具
# ---------------------------------------------------------------------------


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


def _parse_int_tuple(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def _parse_string_tuple(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


# ---------------------------------------------------------------------------
# 命令行接口
# ---------------------------------------------------------------------------


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


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the standalone BiLSTM-Attention direction predictor."
    )
    _add_runtime_cli_arguments(parser)
    _add_loop_and_confidence_cli_arguments(parser)
    _add_signal_engine_cli_arguments(parser)
    _add_postprocess_cli_arguments(parser)
    return parser


def _add_runtime_cli_arguments(parser: argparse.ArgumentParser) -> None:
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
        choices=_CLI_MODES,
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


def _add_loop_and_confidence_cli_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--periods", type=int, default=SCRIPT_PERIODS)
    parser.add_argument("--return-calibration-window", type=int, default=SCRIPT_RETURN_CALIBRATION_WINDOW)
    parser.add_argument("--return-calibration-min-rows", type=int, default=SCRIPT_RETURN_CALIBRATION_MIN_ROWS)
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
        "--confidence-calibration-output",
        default=SCRIPT_CONFIDENCE_CALIBRATION_OUTPUT_PATH,
        help="Write confidence-bin calibration statistics to this CSV path.",
    )
    parser.add_argument(
        "--confidence-calibration-summary-output",
        default=SCRIPT_CONFIDENCE_CALIBRATION_SUMMARY_PATH,
        help="Write aggregate confidence calibration metrics to this CSV path.",
    )
    parser.add_argument(
        "--confidence-calibration-window",
        type=int,
        default=SCRIPT_CONFIDENCE_CALIBRATION_WINDOW,
        help="Prior completed predictions used by the rolling confidence calibrator; 0 disables it.",
    )
    parser.add_argument(
        "--confidence-calibration-min-rows",
        type=int,
        default=SCRIPT_CONFIDENCE_CALIBRATION_MIN_ROWS,
        help="Minimum valid prior rows required before fitting a rolling calibrator.",
    )
    parser.add_argument(
        "--confidence-calibration-method",
        choices=["platt", "isotonic"],
        default=SCRIPT_CONFIDENCE_CALIBRATION_METHOD,
    )
    parser.add_argument(
        "--confidence-calibration-compare-windows",
        default=", ".join(
            str(window) for window in SCRIPT_CONFIDENCE_CALIBRATION_COMPARE_WINDOWS
        ),
        help="Comma-separated rolling windows to compare out of sample, e.g. 120,300,600.",
    )
    parser.add_argument(
        "--rolling-confidence-output",
        default=SCRIPT_ROLLING_CONFIDENCE_OUTPUT_PATH,
        help="Write the selected rolling-calibrated confidence result to this CSV path.",
    )
    parser.add_argument(
        "--rolling-confidence-comparison-output",
        default=SCRIPT_ROLLING_CONFIDENCE_COMPARISON_OUTPUT_PATH,
        help="Write rolling-window ECE/Brier comparison to this CSV path.",
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


def _add_signal_engine_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--signal-engine",
        choices=_SIGNAL_ENGINES,
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


def _add_postprocess_cli_arguments(parser: argparse.ArgumentParser) -> None:
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


_LOOP_CLI_PASSTHROUGH_ARGUMENTS = (
    "return_calibration_window",
    "return_calibration_min_rows",
    "start_date",
    "end_date",
    "periods",
    "confidence_calibration_min_rows",
    "confidence_calibration_method",
    "signal_engine",
    "rule_threshold_end_date",
    "rule_calibration_start_date",
    "rule_calibration_end_date",
    "rule_top_k",
    "volatility_rule_quantile",
    "nested_rule_threshold_window",
    "nested_rule_calibration_window",
    "nested_rule_min_threshold_rows",
    "nested_rule_min_calibration_rows",
    "state_veto_window",
    "state_veto_state_window",
    "state_veto_min_rows",
    "state_veto_bad_accuracy",
    "high_confidence_min_base_calibration",
    "high_confidence_min_veto_state_accuracy",
    "stability_rule_long_window",
    "stability_rule_recent_weight",
    "stability_rule_min_edge",
    "stability_rule_top_k",
    "nested_ml_train_window",
    "nested_ml_step",
    "nested_ml_calibration_window",
    "nested_ml_min_train_rows",
    "nested_ml_top_k",
    "return_magnitude_mode",
    "return_magnitude_window",
    "return_magnitude_min_rows",
    "return_magnitude_grid_size",
    "return_magnitude_clip_low_quantile",
    "return_magnitude_clip_high_quantile",
    "recent_failure_guard",
    "recent_failure_window",
    "recent_failure_invert_threshold",
    "recent_failure_short_window",
    "recent_failure_short_threshold",
    "selector_window",
    "selector_min_history",
    "selector_switch_edge",
    "selector_disagreement_window",
    "selector_disagreement_min_history",
    "selector_disagreement_edge",
    "regime_postprocess",
    "regime_postprocess_history_window",
    "regime_postprocess_min_history",
    "regime_postprocess_flip_below",
    "regime_postprocess_max_flip_rate",
)


def _config_from_cli_args(args: argparse.Namespace) -> DirectionPredictionConfig:
    return DirectionPredictionConfig(
        epochs=args.epochs,
        lookback=args.lookback,
        neutral_band=args.neutral_band,
        device=args.device,
        external_feature_mode=args.external_feature_mode,
        technical_feature_mode=args.technical_feature_mode,
        verbose=args.verbose,
    )


def _resolve_cli_mode(args: argparse.Namespace) -> CliMode:
    if args.loop_validate:
        return "loop_validate"
    if args.walk_forward:
        return "walk_forward"
    return args.mode


def _loop_validation_kwargs_from_cli_args(
    args: argparse.Namespace,
    *,
    output_path: str | None,
) -> dict[str, Any]:
    kwargs = {
        name: getattr(args, name) for name in _LOOP_CLI_PASSTHROUGH_ARGUMENTS
    }
    if args.latest_periods is not None:
        kwargs.update(
            start_date=None,
            end_date=None,
            periods=args.latest_periods,
        )
    kwargs.update(
        output_path=output_path,
        diagnostics_output_path=args.diagnostics_output,
        confidence_output_path=args.confidence_output,
        confidence_summary_path=args.confidence_summary_output,
        confidence_calibration_output_path=args.confidence_calibration_output,
        confidence_calibration_summary_path=(
            args.confidence_calibration_summary_output
        ),
        confidence_calibration_window=(
            args.confidence_calibration_window
            if args.confidence_calibration_window is not None
            and args.confidence_calibration_window > 0
            else None
        ),
        confidence_calibration_compare_windows=_parse_int_tuple(
            args.confidence_calibration_compare_windows
        ),
        rolling_confidence_output_path=args.rolling_confidence_output,
        rolling_confidence_comparison_output_path=(
            args.rolling_confidence_comparison_output
        ),
        progress=not args.no_progress,
        state_veto_quantiles=_parse_float_tuple(args.state_veto_quantiles),
        high_confidence_max_veto_state_rows=(
            args.high_confidence_max_veto_state_rows
            if args.high_confidence_max_veto_state_rows is not None
            and args.high_confidence_max_veto_state_rows >= 0
            else None
        ),
        recent_failure_degrade_threshold=(
            args.recent_failure_degrade_threshold
            if args.recent_failure_degrade_threshold is not None
            else args.recent_failure_threshold
        ),
        regime_postprocess_diagnostics_output_path=(
            args.regime_postprocess_diagnostics_output
        ),
        regime_postprocess_state_columns=_parse_string_tuple(
            args.regime_postprocess_state_columns
        ),
    )
    return kwargs


def _print_loop_validation_summary(
    result: pd.DataFrame,
    *,
    output_path: str,
) -> None:
    display_result = _format_result_frame_for_csv(result)
    print(
        display_result.to_string(
            index=False,
            formatters={
                "原始边界分数": lambda value: f"{value:.2%}",
                "置信度": lambda value: f"{value:.2%}",
            },
        )
    )
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
    _, calibration_summary = confidence_calibration_report(result)
    calibration_metrics = calibration_summary.iloc[0]
    print(
        "confidence_calibration="
        f"mean={calibration_metrics['mean_confidence']:.6f} "
        f"direction_accuracy={calibration_metrics['direction_accuracy']:.6f} "
        f"ece={calibration_metrics['expected_calibration_error']:.6f} "
        f"brier={calibration_metrics['brier_score']:.6f}",
        file=sys.stderr,
    )
    if "calibrated_confidence" not in result.columns:
        return

    _, rolling_summary = confidence_calibration_report(
        result,
        confidence_column="calibrated_confidence",
    )
    rolling_metrics = rolling_summary.iloc[0]
    selected_window = int(result["confidence_calibration_window"].iloc[0])
    print(
        "rolling_confidence_calibration="
        f"window={selected_window} "
        f"ece={rolling_metrics['expected_calibration_error']:.6f} "
        f"brier={rolling_metrics['brier_score']:.6f}",
        file=sys.stderr,
    )


def _run_loop_validation_cli(
    data: pd.DataFrame,
    config: DirectionPredictionConfig,
    args: argparse.Namespace,
) -> None:
    output_path = args.output or "drp_feim_prediction_results.csv"
    result = loop_validate_prediction_results(
        data,
        config=config,
        **_loop_validation_kwargs_from_cli_args(args, output_path=output_path),
    )
    _print_loop_validation_summary(result, output_path=output_path)


def _run_walk_forward_cli(
    data: pd.DataFrame,
    config: DirectionPredictionConfig,
    args: argparse.Namespace,
) -> None:
    result = walk_forward_validate(
        data,
        config=config,
        initial_train_fraction=args.initial_train_fraction,
        test_size=args.test_size,
        max_splits=args.max_splits,
    )
    print(result.to_string(index=False))


def _run_predict_cli(
    data: pd.DataFrame,
    config: DirectionPredictionConfig,
    args: argparse.Namespace,
) -> None:
    options = _loop_validation_kwargs_from_cli_args(args, output_path=None)
    result = predict_next_day(data, config=config, **options)
    print(
        json.dumps(
            _json_sanitize(result),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
    )


def _main(argv: list[str] | None = None) -> None:
    args = _build_argument_parser().parse_args(argv)
    csv_path = args.csv or _default_csv_path()
    data = pd.read_csv(csv_path, encoding=args.encoding)
    config = _config_from_cli_args(args)
    mode = _resolve_cli_mode(args)
    if mode == "loop_validate":
        _run_loop_validation_cli(data, config, args)
    elif mode == "walk_forward":
        _run_walk_forward_cli(data, config, args)
    else:
        _run_predict_cli(data, config, args)


if __name__ == "__main__":
    _main()
