"""Freeze the committed predictor and its full available-history evaluation.

The default reference is intentionally pinned before algorithm changes. Its
historical confidence-window selection is preserved as part of the reference,
including the known evaluation-window selection bias.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_REVISION = "b1a2929841270f2fe641ebe537e0bb9bcbfb450a"
SOURCE_PATH = "\u5faa\u73af\u9a8c\u8bc1\u811a\u672c.py"
VARIANTS = {
    "default": {},
    "volatility_rule": {"signal_engine": "volatility_rule"},
    "state_veto_no_guard": {"recent_failure_guard": False},
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_frozen_predictor(revision: str = BASELINE_REVISION) -> tuple[ModuleType, bytes, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{revision}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    source = subprocess.check_output(
        ["git", "show", f"{commit}:{SOURCE_PATH}"], cwd=ROOT
    )
    name = f"_frozen_prediction_baseline_{commit}"
    module = ModuleType(name)
    module.__file__ = f"git:{commit}:{SOURCE_PATH}"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, source, commit


def metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    predicted = frame["predicted_pct_change"].to_numpy(dtype=float)
    actual = frame["real_pct_change"].to_numpy(dtype=float)
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Evaluation predictions and targets must be finite.")
    predicted_up = predicted > 0
    actual_up = actual > 0
    correct = predicted_up == actual_up
    if not np.array_equal(correct, frame["correct"].to_numpy(dtype=bool)):
        raise ValueError("Prediction direction and recorded correctness disagree.")
    recalls = [float(correct[actual_up == label].mean()) for label in (False, True)
               if np.any(actual_up == label)]
    error = predicted - actual
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "start_date": int(frame["trade_date"].iloc[0]),
        "end_date": int(frame["trade_date"].iloc[-1]),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "return_mae": float(np.abs(error).mean()),
        "return_rmse": float(np.sqrt(np.square(error).mean())),
        "actual_up_rows": int(actual_up.sum()),
        "predicted_up_rows": int(predicted_up.sum()),
        "always_up_accuracy": float(actual_up.mean()),
        "zero_return_mae": float(np.abs(actual).mean()),
        "zero_return_rmse": float(np.sqrt(np.square(actual).mean())),
    }
    for column in ("confidence", "calibrated_confidence"):
        if column in frame:
            confidence = frame[column].to_numpy(dtype=float)
            valid = np.isfinite(confidence)
            result[f"{column}_brier"] = float(
                np.square(confidence[valid] - correct[valid].astype(float)).mean()
            ) if valid.any() else None
            result[f"{column}_rows"] = int(valid.sum())
    return result


def grouped_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    groups = {"all": metric_summary(frame)}
    for year, group in frame.groupby(frame["trade_date"].astype(str).str[:4], sort=True):
        groups[f"year_{year}"] = metric_summary(group)
    for days in (252, 20):
        groups[f"latest_{days}"] = metric_summary(frame.tail(days))
    return groups


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "market_data/merged_features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/baseline")
    parser.add_argument("--revision", default=BASELINE_REVISION)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="default")
    args = parser.parse_args(argv)
    output = args.output.resolve() / args.variant
    if output.exists():
        raise FileExistsError(f"Immutable baseline already exists: {output}")

    module, source, commit = load_frozen_predictor(args.revision)
    cli_args = module._build_argument_parser().parse_args([])
    config = module._config_from_cli_args(cli_args)
    cli_kwargs = module._loop_validation_kwargs_from_cli_args(cli_args, output_path=None)
    # Keep the algorithm unchanged; evaluate every date and avoid external outputs.
    kwargs = dict(cli_kwargs)
    for name in kwargs:
        if name.endswith("_path"):
            kwargs[name] = None
    kwargs.update(start_date=None, end_date=None, periods=0, progress=False)
    kwargs.update(VARIANTS[args.variant])
    input_bytes = args.csv.read_bytes()
    data = pd.read_csv(args.csv, encoding=cli_args.encoding)
    started = time.perf_counter()
    print(f"Evaluating {args.variant} from {commit}", flush=True)
    frame = module.loop_validate_prediction_results(data, config=config, **kwargs)
    elapsed = time.perf_counter() - started
    if frame.empty or frame["trade_date"].duplicated().any():
        raise ValueError("Baseline must contain unique, nonempty prediction dates.")
    if not frame["trade_date"].is_monotonic_increasing:
        raise ValueError("Baseline prediction dates must be chronological.")
    metrics = grouped_metrics(frame)
    signature = inspect.signature(module.loop_validate_prediction_results)
    complete_kwargs = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty and name != "config"
    }
    complete_kwargs.update(kwargs)
    manifest = {
        "format_version": 1,
        "variant": args.variant,
        "is_primary_baseline": args.variant == "default",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "source_path": SOURCE_PATH,
        "source_sha256": sha256(source),
        "input_path": args.csv.resolve().relative_to(ROOT).as_posix(),
        "input_sha256": sha256(input_bytes),
        "input_rows": len(data),
        "model_config": asdict(config),
        "original_cli_arguments": vars(cli_args),
        "original_cli_loop_kwargs": cli_kwargs,
        "evaluation_loop_kwargs": complete_kwargs,
        "elapsed_seconds": elapsed,
        "confidence_note": (
            "confidence is the original boundary score; calibrated_confidence is "
            "the original public confidence, whose window is selected using this "
            "evaluation period. This known selection bias is preserved, not endorsed."
        ),
        "label_definition": "actual and predicted up iff decimal return > 0",
        "return_units": "decimal returns; 0.01 means 1 percent",
        "columns": frame.columns.tolist(),
        "metrics": metrics,
    }
    output.mkdir(parents=True)
    csv_path = output / "predictions.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8", float_format="%.17g")
    manifest["predictions_sha256"] = sha256(csv_path.read_bytes())
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "elapsed_seconds": elapsed,
                      "metrics": metrics}, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
