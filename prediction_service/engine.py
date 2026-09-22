"""Deterministic adapter around the existing prediction engine."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import tushare_prediction_pipeline as pipeline

from .archive import frame_to_csv_bytes, sha256_bytes
from .config import PROJECT_ROOT, Settings


SOURCE_FILES = (
    "循环验证脚本.py",
    "return_calibration.py",
    "regularized_direction.py",
    "tushare_prediction_pipeline.py",
)
TARGET_COLUMNS = {"target_next_return", "target_next_direction"}
# Late provider responses can fill optional Hong Kong features that were unknown
# when a snapshot was frozen. Keep those frozen gaps for every later replay.
OPTIONAL_LAGGED_FEATURES = frozenset(
    f"hangseng_{name}_lag1"
    for name in ("ret1", "ret5", "vol20", "gap", "intraday", "range", "vol_chg", "pct_chg")
)


class HistoricalMarketDataDriftError(RuntimeError):
    """Raised when a provider tries to revise a previously frozen market row."""


def source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for filename in SOURCE_FILES:
        path = PROJECT_ROOT / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def model_configuration(settings: Settings) -> tuple[dict[str, Any], str]:
    config, options = pipeline._default_calculation_options()
    options = {
        key: value
        for key, value in options.items()
        if not key.endswith("_path") and key != "output_path"
    }
    options["signal_engine"] = settings.signal_engine
    if settings.signal_engine == "bilstm_causal":
        options["bilstm_refit_interval"] = settings.bilstm_shadow_refit_interval
        options["recent_failure_guard"] = False
    payload = _jsonable(
        {
            "algorithm_id": (
                pipeline.prediction_core.SCRIPT_ACCEPTED_ALGORITHM_ID
                if settings.signal_engine == pipeline.prediction_core.SCRIPT_SIGNAL_ENGINE
                else settings.signal_engine
            ),
            "signal_engine": settings.signal_engine,
            "validation_days": settings.validation_days,
            "direction_prediction_config": asdict(config),
            "loop_options": options,
        }
    )
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return payload, sha256_bytes(canonical.encode("utf-8"))


def release_id(settings: Settings) -> tuple[str, dict[str, Any], str, str]:
    config, config_sha256 = model_configuration(settings)
    source_sha256 = source_bundle_sha256()
    identifier = sha256_bytes(
        f"{source_sha256}:{config_sha256}".encode("ascii")
    )
    return identifier, config, config_sha256, source_sha256


def canonicalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    if "trade_date" not in frame.columns:
        raise ValueError("特征数据缺少 trade_date 列。")
    result = frame.copy()
    if result.empty:
        raise ValueError("特征数据为空。")
    values = result["trade_date"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = values.str.fullmatch(r"\d{8}")
    dates = pd.to_datetime(values.where(~compact), errors="coerce", format="mixed")
    dates.loc[compact] = pd.to_datetime(values.loc[compact], format="%Y%m%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("特征数据包含无法解析的 trade_date。")
    result["trade_date"] = dates.dt.strftime("%Y-%m-%d")
    if result["trade_date"].duplicated().any():
        raise ValueError("特征数据包含重复交易日。")
    for column in ("open", "high", "low", "close"):
        if column not in result:
            raise ValueError(f"特征数据缺少 {column} 列。")
        numeric = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(numeric).all() or numeric.le(0).any():
            raise ValueError(f"特征数据的 {column} 必须为正的有限数值。")
        result[column] = numeric
    if result["high"].lt(result["low"]).any():
        raise ValueError("最高价不能低于最低价。")
    result = result.sort_values("trade_date")
    return result.reset_index(drop=True)


def data_as_of(features: pd.DataFrame) -> str:
    normalized = canonicalize_features(features)
    return str(normalized["trade_date"].iloc[-1]).replace("-", "")


def _series_match_mask(left: pd.Series, right: pd.Series) -> pd.Series:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    numeric_like = (
        (left.isna() | left_numeric.notna()).all()
        and (right.isna() | right_numeric.notna()).all()
    )
    if numeric_like:
        return pd.Series(
            np.isclose(
                left_numeric.to_numpy(dtype=float),
                right_numeric.to_numpy(dtype=float),
                # Allow CSV round-trip noise, while keeping the frozen values.
                rtol=1e-14,
                atol=1e-12,
                equal_nan=True,
            ),
            index=left.index,
        )
    return left.fillna("<NA>").astype(str).eq(right.fillna("<NA>").astype(str))


def _series_matches(left: pd.Series, right: pd.Series) -> bool:
    return bool(_series_match_mask(left, right).all())


def merge_append_only_features(
    canonical: pd.DataFrame,
    provider_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Append newer rows and reject any mutation to historic model inputs."""

    existing = canonicalize_features(canonical)
    candidate = canonicalize_features(provider_frame)
    existing_columns = set(existing.columns) - TARGET_COLUMNS
    candidate_columns = set(candidate.columns) - TARGET_COLUMNS
    if existing_columns != candidate_columns:
        missing = sorted(existing_columns - candidate_columns)
        unexpected = sorted(candidate_columns - existing_columns)
        raise HistoricalMarketDataDriftError(
            "特征列结构发生变化；"
            f"缺失={missing or '无'}，新增={unexpected or '无'}。"
        )

    existing_by_date = existing.set_index("trade_date", drop=False)
    candidate_by_date = candidate.set_index("trade_date", drop=False)
    missing_dates = sorted(set(existing_by_date.index) - set(candidate_by_date.index))
    if missing_dates:
        raise HistoricalMarketDataDriftError(
            f"数据源缺少已冻结交易日：{', '.join(missing_dates[:3])}"
        )

    common_dates = existing_by_date.index.intersection(candidate_by_date.index)
    new_dates = candidate_by_date.index.difference(existing_by_date.index)
    for column in sorted(existing_columns - {"trade_date"}):
        frozen = existing_by_date.loc[common_dates, column]
        candidate = candidate_by_date.loc[common_dates, column]
        matches = _series_match_mask(frozen, candidate)
        late_optional_value = pd.Series(False, index=common_dates)
        if column in OPTIONAL_LAGGED_FEATURES:
            late_optional_value = frozen.isna() & np.isfinite(pd.to_numeric(candidate, errors="coerce"))
        invalid = (~matches) & (~late_optional_value)
        if invalid.any():
            raise HistoricalMarketDataDriftError(
                f"数据源修订了已冻结特征：{column}。"
            )

    if any(date <= existing_by_date.index.max() for date in new_dates):
        raise HistoricalMarketDataDriftError("数据源插入了早于已冻结截止日的交易日。")
    appended = candidate_by_date.loc[new_dates].reset_index(drop=True)
    combined = pd.concat([existing, appended], ignore_index=True, sort=False)
    return canonicalize_features(combined)


def calculate_results(features: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    return pipeline.run_validation_and_prediction(
        canonicalize_features(features),
        validation_days=settings.validation_days,
        signal_engine=settings.signal_engine,
        progress=False,
    )


def shadow_settings(settings: Settings) -> Settings:
    """Return the fixed, non-public BiLSTM research configuration."""

    return replace(
        settings,
        signal_engine="bilstm_causal",
        validation_days=settings.bilstm_shadow_validation_days,
    )


def calculate_bilstm_shadow_results(
    features: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Run the causal BiLSTM shadow without changing the public engine."""

    shadow = shadow_settings(settings)
    return pipeline.run_validation_and_prediction(
        canonicalize_features(features),
        validation_days=shadow.validation_days,
        signal_engine=shadow.signal_engine,
        progress=False,
        loop_overrides={
            "bilstm_refit_interval": shadow.bilstm_shadow_refit_interval,
            "recent_failure_guard": False,
        },
    )


def calculate_results_with_diagnostics(
    features: pd.DataFrame,
    settings: Settings,
    diagnostics_output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate production output and persist the engine diagnostics for archive."""

    result = pipeline.run_validation_and_prediction(
        canonicalize_features(features),
        validation_days=settings.validation_days,
        signal_engine=settings.signal_engine,
        progress=False,
        diagnostics_output_path=diagnostics_output_path,
    )
    diagnostics = (
        pd.read_csv(diagnostics_output_path, encoding="utf-8-sig")
        if diagnostics_output_path.exists()
        else pd.DataFrame()
    )
    return result, diagnostics


def _stable_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None
        return format(float(value), ".15g")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def payload_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(canonical.encode("utf-8"))


def feature_close_by_date(features: pd.DataFrame) -> dict[str, float]:
    normalized = canonicalize_features(features)
    return {
        date.replace("-", ""): float(close)
        for date, close in zip(normalized["trade_date"], normalized["close"], strict=True)
    }


def target_dates_by_signal(features: pd.DataFrame) -> dict[str, str | None]:
    normalized = canonicalize_features(features)
    dates = normalized["trade_date"].str.replace("-", "", regex=False).tolist()
    mapping: dict[str, str | None] = {}
    for index, signal_date in enumerate(dates):
        mapping[signal_date] = dates[index + 1] if index + 1 < len(dates) else None
    return mapping


def prediction_payload(
    row: pd.Series,
    *,
    base_close: float,
) -> dict[str, Any]:
    return {
        "signal_date": _stable_value(row["trade_date"]),
        "base_close": _stable_value(base_close),
        "predicted_return": _stable_value(row["predicted_pct_change"]),
        "predicted_label": _stable_value(row["predicted_label"]),
        "predicted_close": _stable_value(row["predicted_close"]),
        "raw_confidence": _stable_value(row["confidence"]),
        "calibrated_confidence": _stable_value(row.get("calibrated_confidence")),
        "return_calibration_scale": _stable_value(row.get("return_calibration_scale")),
        "confidence_calibration_window": _stable_value(
            row.get("confidence_calibration_window")
        ),
        "confidence_calibration_method": _stable_value(
            row.get("confidence_calibration_method")
        ),
        "confidence_calibration_rows": _stable_value(
            row.get("confidence_calibration_rows")
        ),
        "confidence_calibration_fallback": _stable_value(
            row.get("confidence_calibration_fallback")
        ),
    }


def outcome_payload(
    row: pd.Series,
    *,
    target_date: str,
    base_close: float,
) -> dict[str, Any] | None:
    realized = row.get("real_pct_change")
    if realized is None or pd.isna(realized):
        return None
    return {
        "target_trade_date": target_date,
        "actual_return": _stable_value(realized),
        "actual_close": _stable_value(base_close * (1.0 + float(realized))),
        "correct": _stable_value(row.get("correct")),
    }


def public_frame(
    result: pd.DataFrame,
    *,
    snapshot_id: str,
    release_id: str,
    data_as_of_date: str,
    prediction_ids: dict[str, str],
) -> pd.DataFrame:
    frame = pipeline.build_combined_csv_frame(result)
    frame["预测ID"] = frame["信号日期"].astype(str).map(prediction_ids)
    frame["快照ID"] = snapshot_id
    frame["模型版本"] = release_id
    frame["输入数据截止日"] = data_as_of_date
    return frame


def public_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame_to_csv_bytes(frame)
