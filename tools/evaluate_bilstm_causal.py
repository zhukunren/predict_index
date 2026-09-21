"""Freeze a causal BiLSTM versus default-engine comparison on an independent test.

This research utility intentionally does not select or promote a champion. It
uses the historical market-data blob pinned at ``FROZEN_INPUT_REVISION`` and
compares the current default ``state_veto_rule`` against a pure BiLSTM engine.
The output directory is write-once so that test metrics cannot be overwritten
after inspection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = "循环验证脚本.py"
FROZEN_INPUT_REVISION = "b1a2929841270f2fe641ebe537e0bb9bcbfb450a"
FROZEN_INPUT_PATH = "market_data/merged_features.csv"
FROZEN_INPUT_SHA256 = "11679c0c7a9f1720cd25b37343c84d05ddf2eb2043b6b9cbfbbb2461725cafd2"
DEVELOPMENT_START = 20230101
DEVELOPMENT_END = 20241231
TEST_START = 20250102
TEST_END = 20260911
V1_CANDIDATE_DIR = ROOT / "artifacts/evaluation/bilstm_causal_v1"
PARITY_COLUMNS = (
    "predicted_label",
    "predicted_pct_change",
    "predicted_close",
    "confidence",
    "calibrated_confidence",
    "return_calibration_scale",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_module(name: str, path: str, source: bytes) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def frozen_input_bytes() -> bytes:
    value = subprocess.check_output(
        ["git", "show", f"{FROZEN_INPUT_REVISION}:{FROZEN_INPUT_PATH}"],
        cwd=ROOT,
    )
    if sha256(value) != FROZEN_INPUT_SHA256:
        raise ValueError("Frozen input blob SHA-256 does not match the declared contract.")
    return value


def load_frozen_input() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(frozen_input_bytes()), encoding="utf-8-sig")


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    required = {
        "trade_date",
        "predicted_label",
        "predicted_pct_change",
        "predicted_close",
        "confidence",
        "calibrated_confidence",
        "real_pct_change",
        "correct",
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"Evaluation frame is missing columns: {missing}")
    if result.empty or result["trade_date"].duplicated().any():
        raise ValueError("Evaluation predictions must be nonempty with unique dates.")
    if not result["trade_date"].is_monotonic_increasing:
        raise ValueError("Evaluation predictions must be chronological.")
    if not result["trade_date"].between(TEST_START, TEST_END).all():
        raise ValueError("Evaluation output leaked outside the frozen test period.")
    labels = result["predicted_label"].to_numpy(dtype=int)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("predicted_label must be binary.")
    predicted = result["predicted_pct_change"].to_numpy(dtype=float)
    realized = result["real_pct_change"].to_numpy(dtype=float)
    if not np.isfinite(predicted).all() or not np.isfinite(realized).all():
        raise ValueError("Evaluation returns must be finite and completed.")
    expected_correct = labels == (realized > 0)
    if not np.array_equal(expected_correct, result["correct"].to_numpy(dtype=bool)):
        raise ValueError("Recorded correctness is inconsistent with labels and realized returns.")
    for column in ("confidence", "calibrated_confidence"):
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((0.0 <= values) & (values <= 1.0)).all():
            raise ValueError(f"{column} must be a finite probability.")
    return result


def metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["predicted_label"].to_numpy(dtype=int)
    predicted = frame["predicted_pct_change"].to_numpy(dtype=float)
    actual = frame["real_pct_change"].to_numpy(dtype=float)
    actual_up = actual > 0
    correct = labels == actual_up
    recall_down = float(correct[~actual_up].mean()) if np.any(~actual_up) else np.nan
    recall_up = float(correct[actual_up].mean()) if np.any(actual_up) else np.nan
    error = predicted - actual
    return {
        "rows": int(len(frame)),
        "start_date": int(frame["trade_date"].iloc[0]),
        "end_date": int(frame["trade_date"].iloc[-1]),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.nanmean([recall_down, recall_up])),
        "recall_down": _finite_or_none(recall_down),
        "recall_up": _finite_or_none(recall_up),
        "brier": float(
            np.square(frame["calibrated_confidence"].to_numpy(dtype=float) - correct).mean()
        ),
        "raw_brier": float(
            np.square(frame["confidence"].to_numpy(dtype=float) - correct).mean()
        ),
        "return_mae": float(np.abs(error).mean()),
        "return_rmse": float(np.sqrt(np.square(error).mean())),
        "actual_up_rows": int(actual_up.sum()),
        "predicted_up_rows": int(labels.sum()),
    }


def stability_summary(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    working = frame.copy()
    actual_up = working["real_pct_change"].to_numpy(dtype=float) > 0
    correct = working["predicted_label"].to_numpy(dtype=int) == actual_up
    working["month"] = working["trade_date"].astype(str).str[:6]
    monthly_rows: list[dict[str, Any]] = []
    for month, group in working.groupby("month", sort=True):
        monthly_rows.append({"month": month, **metric_summary(group)})
    monthly = pd.DataFrame(monthly_rows)
    rolling_20 = pd.Series(correct, dtype=float).rolling(20, min_periods=20).mean().dropna()
    flips = np.diff(working["predicted_label"].to_numpy(dtype=int)) != 0
    return {
        "monthly_periods": int(len(monthly)),
        "monthly_accuracy_std": _finite_or_none(float(monthly["accuracy"].std(ddof=0))),
        "monthly_accuracy_min": _finite_or_none(float(monthly["accuracy"].min())),
        "monthly_balanced_accuracy_std": _finite_or_none(
            float(monthly["balanced_accuracy"].std(ddof=0))
        ),
        "rolling_20_accuracy_std": _finite_or_none(float(rolling_20.std(ddof=0))),
        "rolling_20_accuracy_min": _finite_or_none(float(rolling_20.min())),
        "direction_flip_rate": float(flips.mean()) if len(flips) else 0.0,
    }, monthly


def production_config_and_options(module: ModuleType) -> tuple[Any, dict[str, Any]]:
    """Use the same parser-derived defaults as the production predictor."""

    defaults = module._build_argument_parser().parse_args([])
    config = replace(module._config_from_cli_args(defaults), device="cpu")
    options = module._loop_validation_kwargs_from_cli_args(defaults, output_path=None)
    for key in tuple(options):
        if key.endswith("_path"):
            options[key] = None
    options.update(
        start_date=str(TEST_START),
        end_date=str(TEST_END),
        periods=0,
        include_latest=False,
        progress=False,
        confidence_calibration_window=300,
        confidence_calibration_min_rows=60,
        confidence_calibration_method="platt",
        confidence_calibration_compare_windows=(),
        return_calibration_window=252,
        return_calibration_min_rows=60,
        regime_postprocess=False,
    )
    return config, options


DEFAULT_OVERRIDES = {
    "signal_engine": "state_veto_rule",
    "recent_failure_guard": True,
}
BILSTM_OVERRIDES = {
    "signal_engine": "bilstm_causal",
    "bilstm_refit_interval": 5,
    # Keep the candidate as BiLSTM rather than blending it with a rule-based
    # direction inversion overlay. Confidence and return calibration remain
    # identical causal post-processing for both engines.
    "recent_failure_guard": False,
}


def _candidate_options(
    module: ModuleType,
    data: pd.DataFrame,
    config: Any,
    production_options: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    options = production_options | overrides
    bound = inspect.signature(module.loop_validate_prediction_results).bind(
        data, config=config, **options
    )
    bound.apply_defaults()
    return {
        name: value
        for name, value in bound.arguments.items()
        if name not in {"df", "config"}
    }


BILSTM_RELEVANT_OPTION_KEYS = (
    "start_date",
    "end_date",
    "periods",
    "include_latest",
    "confidence_calibration_window",
    "confidence_calibration_min_rows",
    "confidence_calibration_method",
    "confidence_calibration_compare_windows",
    "return_magnitude_mode",
    "return_magnitude_window",
    "return_magnitude_min_rows",
    "return_magnitude_grid_size",
    "return_magnitude_clip_low_quantile",
    "return_magnitude_clip_high_quantile",
    "return_calibration_window",
    "return_calibration_min_rows",
    "recent_failure_guard",
    "regime_postprocess",
    "bilstm_refit_interval",
    "signal_engine",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def load_reusable_bilstm_candidate(
    candidate_dir: Path,
    *,
    config: Any,
    options: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Validate the completed v1 candidate before reusing its frozen bytes."""

    contract_path = candidate_dir / "contract.json"
    summary_path = candidate_dir / "summary.json"
    prediction_path = candidate_dir / "bilstm_causal_predictions.csv"
    parity_path = candidate_dir / "bilstm_live_replay_parity.json"
    if not all(path.exists() for path in (contract_path, summary_path, prediction_path, parity_path)):
        raise FileNotFoundError("Reusable BiLSTM candidate artifact is incomplete.")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    frozen = contract.get("frozen_input", {})
    if frozen.get("sha256") != FROZEN_INPUT_SHA256:
        raise ValueError("Reusable BiLSTM candidate uses a different frozen input.")
    if contract.get("independent_test_period") != {
        "start_date": TEST_START,
        "end_date": TEST_END,
    }:
        raise ValueError("Reusable BiLSTM candidate uses a different independent test period.")
    if _canonical(contract.get("model_config")) != _canonical(asdict(config)):
        raise ValueError("Reusable BiLSTM candidate uses a different model configuration.")
    previous_options = contract.get("bilstm_options", {})
    mismatches = {
        key: {"expected": options.get(key), "actual": previous_options.get(key)}
        for key in BILSTM_RELEVANT_OPTION_KEYS
        if _canonical(options.get(key)) != _canonical(previous_options.get(key))
    }
    if mismatches:
        raise ValueError(f"Reusable BiLSTM candidate options differ: {mismatches}")
    if not bool(summary.get("parity_passed")) or not bool(parity.get("passed")):
        raise ValueError("Reusable BiLSTM candidate did not pass live/replay parity.")
    frame = validate_frame(pd.read_csv(prediction_path, float_precision="round_trip"))
    expected_hash = summary["bilstm_causal"]["predictions_sha256"]
    actual_hash = sha256(prediction_path.read_bytes())
    if actual_hash != expected_hash:
        raise ValueError("Reusable BiLSTM predictions failed their SHA-256 check.")
    provenance = {
        "artifact_dir": str(candidate_dir.relative_to(ROOT)).replace("\\", "/"),
        "contract_sha256": sha256(contract_path.read_bytes()),
        "predictions_sha256": actual_hash,
        "parity_sha256": sha256(parity_path.read_bytes()),
        "source_sha256": contract.get("source_sha256"),
        "reused_because": (
            "v1 passed five independent predict_next_day versus replay probes, "
            "and its BiLSTM-relevant fixed options match this comparison contract."
        ),
    }
    return frame, parity, provenance


def choose_parity_dates(frame: pd.DataFrame, count: int) -> list[int]:
    if count < 1:
        raise ValueError("parity_probe_count must be positive.")
    positions = np.linspace(0, len(frame) - 1, min(count, len(frame)), dtype=int)
    return [int(frame["trade_date"].iloc[position]) for position in np.unique(positions)]


def verify_live_replay_parity(
    module: ModuleType,
    data: pd.DataFrame,
    config: Any,
    replay: pd.DataFrame,
    options: dict[str, Any],
    *,
    probe_count: int,
) -> dict[str, Any]:
    source_dates = pd.to_datetime(data["trade_date"])
    records: list[dict[str, Any]] = []
    for trade_date in choose_parity_dates(replay, probe_count):
        signal_ts = pd.Timestamp(str(trade_date))
        prefix = data.loc[source_dates.le(signal_ts)].copy()
        live = module.predict_next_day(prefix, config=config, **options)
        replay_row = replay.loc[replay["trade_date"].eq(trade_date)].iloc[0]
        comparisons = {
            "predicted_label": int(live["predicted_label"]) == int(replay_row["predicted_label"]),
            "predicted_pct_change": bool(
                np.isclose(live["estimated_next_return"], replay_row["predicted_pct_change"], atol=1e-12, rtol=0.0)
            ),
            "predicted_close": bool(
                np.isclose(live["estimated_next_close"], replay_row["predicted_close"], atol=1e-10, rtol=0.0)
            ),
            "raw_confidence": bool(
                np.isclose(live["raw_confidence"], replay_row["confidence"], atol=1e-12, rtol=0.0)
            ),
            "calibrated_confidence": bool(
                np.isclose(live["calibrated_confidence"], replay_row["calibrated_confidence"], atol=1e-12, rtol=0.0)
            ),
        }
        records.append(
            {
                "trade_date": trade_date,
                "prefix_last_date": pd.Timestamp(prefix["trade_date"].iloc[-1]).strftime("%Y%m%d"),
                "passed": all(comparisons.values()),
                "checks": comparisons,
            }
        )
    return {
        "method": "shared_causal_loop_with_independent_live_prefix_probes",
        "probe_count": len(records),
        "passed": all(record["passed"] for record in records),
        "records": records,
    }


def comparison_summary(default: dict[str, Any], bilstm: dict[str, Any]) -> dict[str, Any]:
    metrics = ("accuracy", "balanced_accuracy", "brier", "return_mae", "return_rmse")
    stability_metrics = (
        "monthly_accuracy_std",
        "monthly_accuracy_min",
        "monthly_balanced_accuracy_std",
        "rolling_20_accuracy_std",
        "rolling_20_accuracy_min",
        "direction_flip_rate",
    )
    return {
        "test_metric_delta_bilstm_minus_default": {
            metric: float(bilstm["metrics"][metric] - default["metrics"][metric])
            for metric in metrics
        },
        "stability_delta_bilstm_minus_default": {
            metric: (
                None
                if bilstm["stability"][metric] is None or default["stability"][metric] is None
                else float(bilstm["stability"][metric] - default["stability"][metric])
            )
            for metric in stability_metrics
        },
        "automatic_promotion": False,
        "promotion_policy": "This report is observational. BiLSTM cannot replace the default without a separately approved release gate.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/evaluation/bilstm_causal_v3",
        help="New immutable output directory.",
    )
    parser.add_argument(
        "--bilstm-artifact",
        type=Path,
        default=V1_CANDIDATE_DIR,
        help="Completed immutable v1 BiLSTM artifact to verify and reuse.",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output}")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    input_bytes = frozen_input_bytes()
    data = load_frozen_input()
    sources = {
        path: (ROOT / path).read_bytes()
        for path in (SOURCE_PATH, "return_calibration.py", "regularized_direction.py")
    }
    sys.path.insert(0, str(ROOT))
    for dependency in ("return_calibration.py", "regularized_direction.py"):
        load_module(Path(dependency).stem, dependency, sources[dependency])
    module = load_module("_bilstm_causal_evaluation", SOURCE_PATH, sources[SOURCE_PATH])
    module.torch.set_num_threads(1)
    config, production_options = production_config_and_options(module)
    default_options = _candidate_options(
        module, data, config, production_options, DEFAULT_OVERRIDES
    )
    bilstm_options = _candidate_options(
        module, data, config, production_options, BILSTM_OVERRIDES
    )
    reusable_bilstm, reused_parity, candidate_provenance = load_reusable_bilstm_candidate(
        args.bilstm_artifact,
        config=config,
        options=bilstm_options,
    )

    contract = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_input": {
            "git_revision": FROZEN_INPUT_REVISION,
            "path": FROZEN_INPUT_PATH,
            "sha256": sha256(input_bytes),
            "rows": int(len(data)),
            "relationship_to_legacy_default_baseline": (
                "Independent Git blob. Its SHA-256 intentionally differs from the "
                "legacy baseline artifact's uncommitted input snapshot."
            ),
        },
        "independent_test_period": {"start_date": TEST_START, "end_date": TEST_END},
        "development_period": {"start_date": DEVELOPMENT_START, "end_date": DEVELOPMENT_END},
        "source_sha256": {path: sha256(value) for path, value in sources.items()},
        "evaluation_script_sha256": sha256(Path(__file__).read_bytes()),
        "model_config": asdict(config),
        "production_option_source": "_build_argument_parser().parse_args([])",
        "default_options": default_options,
        "bilstm_options": bilstm_options,
        "bilstm_candidate_provenance": candidate_provenance,
        "bilstm_direction_policy": "pure_bilstm_without_recent_failure_guard",
        "confidence_policy": "fixed causal 300-row Platt calibration for both engines",
        "parity_policy": "The reusable candidate passed five fixed independent predict_next_day versus replay probes. This v2 report verifies that its relevant options and SHA-256 match before reuse.",
        "metrics": [
            "accuracy",
            "balanced_accuracy",
            "causal_calibrated_brier",
            "return_mae",
            "return_rmse",
            "monthly_accuracy_std",
            "monthly_balanced_accuracy_std",
            "rolling_20_accuracy_std",
            "direction_flip_rate",
        ],
        "automatic_promotion": False,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "contract.json", contract)

    reports: dict[str, dict[str, Any]] = {}
    stability_frames: list[pd.DataFrame] = []
    print("Evaluating default_state_veto with production CLI defaults", flush=True)
    started = time.perf_counter()
    default_frame = validate_frame(
        module.loop_validate_prediction_results(data, config=config, **default_options)
    )
    default_metrics = metric_summary(default_frame)
    default_stability, default_monthly = stability_summary(default_frame)
    default_path = args.output / "default_state_veto_predictions.csv"
    default_frame.to_csv(default_path, index=False, encoding="utf-8", float_format="%.17g")
    reports["default_state_veto"] = {
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": default_metrics,
        "stability": default_stability,
        "predictions_sha256": sha256(default_path.read_bytes()),
    }
    write_json(args.output / "default_state_veto_metrics.json", reports["default_state_veto"])
    default_monthly.insert(0, "engine", "default_state_veto")
    stability_frames.append(default_monthly)

    reusable_prediction_path = args.bilstm_artifact / "bilstm_causal_predictions.csv"
    output_prediction_path = args.output / "bilstm_causal_predictions.csv"
    shutil.copyfile(reusable_prediction_path, output_prediction_path)
    copied_bilstm = validate_frame(
        pd.read_csv(output_prediction_path, float_precision="round_trip")
    )
    if sha256(output_prediction_path.read_bytes()) != candidate_provenance["predictions_sha256"]:
        raise ValueError("Copied BiLSTM candidate does not match reusable artifact SHA-256.")
    bilstm_metrics = metric_summary(copied_bilstm)
    bilstm_stability, bilstm_monthly = stability_summary(copied_bilstm)
    reports["bilstm_causal"] = {
        "status": "reused_completed_candidate",
        "elapsed_seconds": 0.0,
        "metrics": bilstm_metrics,
        "stability": bilstm_stability,
        "predictions_sha256": candidate_provenance["predictions_sha256"],
        "provenance": candidate_provenance,
    }
    write_json(args.output / "bilstm_causal_metrics.json", reports["bilstm_causal"])
    bilstm_monthly.insert(0, "engine", "bilstm_causal")
    stability_frames.append(bilstm_monthly)
    parity = {
        "status": "reused_verified_parity",
        "source": candidate_provenance,
        **reused_parity,
    }
    write_json(args.output / "bilstm_live_replay_parity.json", parity)
    pd.concat(stability_frames, ignore_index=True).to_csv(
        args.output / "monthly_stability.csv", index=False, encoding="utf-8", float_format="%.17g"
    )
    comparison = comparison_summary(reports["default_state_veto"], reports["bilstm_causal"])
    write_json(args.output / "comparison.json", comparison)
    summary = {
        "completed": True,
        "parity_passed": parity["passed"],
        "default_state_veto": reports["default_state_veto"],
        "bilstm_causal": reports["bilstm_causal"],
        "comparison": comparison,
        "contract_sha256": sha256((args.output / "contract.json").read_bytes()),
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False), flush=True)
    return 0 if parity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
