"""Evaluate the two declared candidates against the immutable original baseline.

No parameters are searched here. A failed gate exits nonzero and must not be
used to tune the candidates on the already observed test dates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = "\u5faa\u73af\u9a8c\u8bc1\u811a\u672c.py"
TEST_START = 20250101
DEVELOPMENT_START = 20230101
DEVELOPMENT_END = 20241231
CANDIDATES = {
    "fixed_state_veto": {"signal_engine": "state_veto_rule", "recent_failure_guard": True},
    "regularized_trees": {"signal_engine": "regularized_trees", "recent_failure_guard": False},
}
DIRECTION_PARITY_CANDIDATES = frozenset(("fixed_state_veto",))
SHARED_OVERRIDES = {
    "start_date": None,
    "end_date": None,
    "periods": 0,
    "progress": False,
    "confidence_calibration_window": 300,
    "confidence_calibration_compare_windows": (),
    "return_calibration_window": 252,
    "return_calibration_min_rows": 60,
    "include_latest": False,
}
TREE_PARAMETERS = {
    "train_window": 756,
    "min_train_rows": 252,
    "refit_interval": 5,
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": 30,
    "max_features": 0.7,
    "class_weight": None,
    "n_jobs": 1,
    "random_state": 42,
    "direction_threshold": 0.5,
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def load_module(name: str, path: str, source: bytes) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if frame.empty or frame["trade_date"].duplicated().any():
        raise ValueError("Prediction dates must be unique and nonempty.")
    if not frame["trade_date"].is_monotonic_increasing:
        raise ValueError("Prediction dates must be chronological.")
    if "predicted_label" not in frame:
        frame["predicted_label"] = (frame["predicted_pct_change"] > 0).astype(int)
    labels = frame["predicted_label"].to_numpy(dtype=float)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("predicted_label must be a binary direction independent of magnitude.")
    predicted = frame["predicted_pct_change"].to_numpy(dtype=float)
    actual = frame["real_pct_change"].to_numpy(dtype=float)
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Only completed finite predictions may be evaluated.")
    nonzero = predicted != 0
    if not np.array_equal(predicted[nonzero] > 0, labels[nonzero] > 0):
        raise ValueError("Nonzero return forecasts contradict their direction labels.")
    correct = labels == (actual > 0)
    if not np.array_equal(correct, frame["correct"].to_numpy(dtype=bool)):
        raise ValueError("Recorded correctness does not match predicted_label and actual direction.")
    for column in ("confidence", "calibrated_confidence"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
            raise ValueError(f"{column} must contain finite probabilities.")
    return frame


def metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    predicted = frame["predicted_pct_change"].to_numpy(dtype=float)
    actual = frame["real_pct_change"].to_numpy(dtype=float)
    labels = frame["predicted_label"].to_numpy(dtype=int)
    actual_up = actual > 0
    correct = labels == actual_up
    recalls = [float(correct[actual_up == label].mean()) for label in (False, True)
               if np.any(actual_up == label)]
    error = predicted - actual
    return {
        "rows": len(frame),
        "start_date": int(frame["trade_date"].iloc[0]),
        "end_date": int(frame["trade_date"].iloc[-1]),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "return_mae": float(np.abs(error).mean()),
        "return_rmse": float(np.sqrt(np.square(error).mean())),
        "calibrated_confidence_brier": float(np.square(frame["calibrated_confidence"].to_numpy() - correct).mean()),
        "raw_confidence_brier": float(np.square(frame["confidence"].to_numpy() - correct).mean()),
        "actual_up_rows": int(actual_up.sum()),
        "predicted_up_rows": int(labels.sum()),
        "zero_return_forecasts": int((predicted == 0).sum()),
        "always_up_accuracy": float(actual_up.mean()),
        "zero_return_mae": float(np.abs(actual).mean()),
        "zero_return_rmse": float(np.sqrt(np.square(actual).mean())),
    }


def grouped_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    groups = {"all": metric_summary(frame)}
    groups["test"] = metric_summary(frame[frame["trade_date"] >= TEST_START])
    development = frame[frame["trade_date"].between(DEVELOPMENT_START, DEVELOPMENT_END)]
    if not development.empty:
        groups["development"] = metric_summary(development)
    for year, group in frame.groupby(frame["trade_date"].astype(str).str[:4], sort=True):
        groups[f"year_{year}"] = metric_summary(group)
    for days in (252, 20):
        groups[f"latest_{days}"] = metric_summary(frame.tail(days))
    return groups


def compare_candidate(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    confidence_calibrator: Any,
    require_direction_parity: bool = False,
) -> dict[str, Any]:
    common_dates = np.intersect1d(baseline["trade_date"], candidate["trade_date"])
    baseline_only_dates = sorted(set(baseline["trade_date"]) - set(common_dates))
    candidate_only_dates = sorted(set(candidate["trade_date"]) - set(common_dates))
    reference = baseline[baseline["trade_date"].isin(common_dates)].reset_index(drop=True)
    aligned = candidate[candidate["trade_date"].isin(common_dates)].reset_index(drop=True)
    if not np.array_equal(reference["trade_date"], aligned["trade_date"]):
        raise ValueError("Candidate and baseline date alignment failed.")
    if not np.allclose(reference["real_pct_change"], aligned["real_pct_change"], atol=1e-12, rtol=0):
        raise ValueError("Candidate and baseline targets disagree on identical dates.")
    baseline_metrics = grouped_metrics(reference)
    candidate_metrics = grouped_metrics(aligned)
    missing_test_dates = sorted(set(baseline.loc[baseline["trade_date"] >= TEST_START, "trade_date"]) - set(common_dates))
    checks = [
        {
            "group": "all",
            "metric": "exact_date_coverage",
            "passed": not baseline_only_dates and not candidate_only_dates,
            "baseline_only_dates": baseline_only_dates,
            "candidate_only_dates": candidate_only_dates,
        },
        {
            "group": "test",
            "metric": "complete_baseline_date_coverage",
            "passed": not missing_test_dates,
            "missing_dates": missing_test_dates,
        },
    ]
    direction_differences = int(
        (reference["predicted_label"].to_numpy(dtype=int)
         != aligned["predicted_label"].to_numpy(dtype=int)).sum()
    )
    if require_direction_parity:
        checks.append(
            {
                "group": "all",
                "metric": "exact_direction_parity",
                "operator": "==",
                "candidate": direction_differences,
                "baseline": 0,
                "passed": direction_differences == 0,
            }
        )
    for group in ("test", "latest_252", "latest_20"):
        for metric, operator in (
            ("accuracy", ">="),
            ("balanced_accuracy", ">="),
            ("return_mae", "<="),
            ("return_rmse", "<="),
        ):
            left = candidate_metrics[group][metric]
            right = baseline_metrics[group][metric]
            passed = left >= right if operator == ">=" else left <= right + 1e-12
            checks.append({"group": group, "metric": metric, "operator": operator,
                           "candidate": left, "baseline": right, "delta": left - right,
                           "passed": bool(passed)})
    # The frozen legacy CSV selected its calibration window using evaluation
    # outcomes. Refit both sides with the same fixed causal protocol instead.
    fair_baseline = confidence_calibrator(reference, window=300, min_rows=60, method="platt")
    fair_candidate = confidence_calibrator(aligned, window=300, min_rows=60, method="platt")
    metric = "causal_fixed_300_confidence_brier"
    left = metric_summary(fair_candidate.loc[fair_candidate["trade_date"] >= TEST_START])["calibrated_confidence_brier"]
    right = metric_summary(fair_baseline.loc[fair_baseline["trade_date"] >= TEST_START])["calibrated_confidence_brier"]
    checks.append({"group": "test", "metric": metric, "operator": "<=", "candidate": left,
                   "baseline": right, "delta": left - right, "passed": bool(left <= right + 1e-12)})
    return {
        "passed": all(check["passed"] for check in checks),
        "common_rows": len(common_dates),
        "baseline_only_dates": baseline_only_dates,
        "candidate_only_dates": candidate_only_dates,
        "direction_differences": direction_differences,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "market_data/merged_features.csv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/baseline/default")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluation/fixed_candidates")
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), action="append")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Evaluation output already exists and is immutable: {args.output}")
    baseline_manifest_bytes = (args.baseline / "manifest.json").read_bytes()
    baseline_manifest = json.loads(baseline_manifest_bytes)
    input_bytes = args.csv.read_bytes()
    baseline_bytes = (args.baseline / "predictions.csv").read_bytes()
    if baseline_manifest.get("variant") != "default" or not baseline_manifest.get("is_primary_baseline"):
        raise ValueError("Evaluation requires the predeclared default baseline.")
    if sha256(input_bytes) != baseline_manifest["input_sha256"]:
        raise ValueError("Candidate input differs from the frozen baseline input.")
    if sha256(baseline_bytes) != baseline_manifest["predictions_sha256"]:
        raise ValueError("Frozen baseline predictions failed their integrity check.")
    baseline = validate_frame(pd.read_csv(args.baseline / "predictions.csv", float_precision="round_trip"))
    data = pd.read_csv(args.csv, encoding="utf-8-sig")

    paths = (SOURCE_PATH, "return_calibration.py", "regularized_direction.py")
    sources = {path: (ROOT / path).read_bytes() for path in paths}
    sys.path.insert(0, str(ROOT))
    # Execute a snapshot so concurrent file edits cannot change this run halfway.
    for dependency in paths[1:]:
        load_module(Path(dependency).stem, dependency, sources[dependency])
    module = load_module("_candidate_prediction_evaluation", SOURCE_PATH, sources[SOURCE_PATH])
    config = module.DirectionPredictionConfig(**baseline_manifest["model_config"])
    common_kwargs = dict(baseline_manifest["evaluation_loop_kwargs"])
    common_kwargs.update(SHARED_OVERRIDES)
    for key in common_kwargs:
        if key.endswith("_path"):
            common_kwargs[key] = None
    signature = inspect.signature(module.loop_validate_prediction_results)
    candidate_options = {}
    requested_candidates = tuple(args.candidate or CANDIDATES)
    for candidate_name in requested_candidates:
        overrides = CANDIDATES[candidate_name]
        bound = signature.bind(data, config=config, **(common_kwargs | overrides))
        bound.apply_defaults()
        candidate_options[candidate_name] = {
            key: value for key, value in bound.arguments.items() if key not in ("df", "config")
        }
    contract = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256(input_bytes),
        "baseline_manifest_sha256": sha256(baseline_manifest_bytes),
        "baseline_predictions_sha256": sha256(baseline_bytes),
        "baseline_source_commit": baseline_manifest["source_commit"],
        "source_sha256": {path: sha256(source) for path, source in sources.items()},
        "evaluation_script_sha256": sha256(Path(__file__).read_bytes()),
        "model_config": asdict(config),
        "candidate_options": candidate_options,
        "evaluated_candidates": list(requested_candidates),
        "regularized_tree_parameters": TREE_PARAMETERS,
        "development_dates": [DEVELOPMENT_START, DEVELOPMENT_END],
        "test_start_date": TEST_START,
        "gate": {
            "accuracy_and_balanced_accuracy_greater_or_equal": ["test", "latest_252", "latest_20"],
            "mae_and_rmse_less_or_equal": ["test", "latest_252", "latest_20"],
            "causal_fixed_300_brier_less_or_equal": ["test"],
            "exact_direction_parity_required": sorted(DIRECTION_PARITY_CANDIDATES.intersection(requested_candidates)),
            "complete_test_date_coverage_required": True,
            "error_metric_float_tolerance": 1e-12,
        },
        "direction_policy": {
            "postprocess_candidate": "exact direction parity on every frozen date is required",
            "direction_changing_candidate": "accuracy and balanced accuracy cannot decline in any declared window",
        },
        "policy": "Fixed candidates; no parameter adjustment after viewing this test. Baseline never changes.",
        "baseline_confidence_caveat": baseline_manifest["confidence_note"],
        "label_definition": "predicted_label independent of magnitude; actual up iff realized return > 0",
        "return_units": "decimal returns; 0.01 means 1 percent",
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "contract.json", contract)
    results: dict[str, Any] = {}
    for name, kwargs in candidate_options.items():
        print(f"Evaluating fixed candidate {name}", flush=True)
        started = time.perf_counter()
        try:
            frame = validate_frame(module.loop_validate_prediction_results(data, config=config, **kwargs))
            result = compare_candidate(
                baseline,
                frame,
                confidence_calibrator=module._apply_rolling_confidence_calibration,
                require_direction_parity=name in DIRECTION_PARITY_CANDIDATES,
            )
            output_path = args.output / f"{name}_predictions.csv"
            frame.to_csv(output_path, index=False, encoding="utf-8", float_format="%.17g")
            result["predictions_sha256"] = sha256(output_path.read_bytes())
            result["status"] = "passed" if result["passed"] else "failed"
        except Exception as error:
            result = {"passed": False, "status": "error", "error": f"{type(error).__name__}: {error}"}
        result["elapsed_seconds"] = time.perf_counter() - started
        results[name] = result
        write_json(args.output / f"{name}_metrics.json", result)
        print(json.dumps({"candidate": name, "status": result["status"],
                          "checks": result.get("checks", []), "error": result.get("error")},
                         indent=2, ensure_ascii=True, allow_nan=False), flush=True)
    batch_passed = all(result["passed"] for result in results.values())
    summary = {
        "passed": batch_passed,
        "batch_passed": batch_passed,
        "passing_candidates": [name for name, result in results.items() if result["passed"]],
        "candidate_status": {name: result["status"] for name, result in results.items()},
        "promotion_policy": (
            "A deployment must name one candidate and require that candidate's "
            "own metrics status to be passed; mixed-batch status never selects a champion."
        ),
        "contract_sha256": sha256((args.output / "contract.json").read_bytes()),
    }
    write_json(args.output / "summary.json", summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
