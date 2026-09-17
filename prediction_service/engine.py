"""Deterministic adapter around the existing prediction engine."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
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
    payload = _jsonable(
        {
            "algorithm_id": pipeline.prediction_core.SCRIPT_ACCEPTED_ALGORITHM_ID,
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
    dates = pd.to_datetime(result["trade_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("特征数据包含无法解析的 trade_date。")
    result["trade_date"] = dates.dt.strftime("%Y-%m-%d")
    result = result.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    return result.reset_index(drop=True)


def data_as_of(features: pd.DataFrame) -> str:
    normalized = canonicalize_features(features)
    return str(normalized["trade_date"].iloc[-1]).replace("-", "")


def _series_matches(left: pd.Series, right: pd.Series) -> bool:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    comparable = left.notna() | right.notna()
    numeric_like = (
        comparable.sum() == 0
        or (
            left_numeric[comparable].notna().all()
            and right_numeric[comparable].notna().all()
        )
    )
    if numeric_like:
        return bool(
            np.isclose(
                left_numeric.to_numpy(dtype=float),
                right_numeric.to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ).all()
        )
    return bool(left.fillna("<NA>").astype(str).eq(right.fillna("<NA>").astype(str)).all())


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
    for column in sorted(existing_columns - {"trade_date"}):
        if not _series_matches(
            existing_by_date.loc[common_dates, column],
            candidate_by_date.loc[common_dates, column],
        ):
            raise HistoricalMarketDataDriftError(
                f"数据源修订了已冻结特征：{column}。"
            )

    new_dates = candidate_by_date.index.difference(existing_by_date.index)
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
