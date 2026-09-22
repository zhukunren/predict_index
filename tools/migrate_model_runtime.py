"""Freeze a new runtime release after bounded numerical compatibility checks.

The original bundle remains immutable. Only probability differences <= 1e-12
are accepted; historical dates, directions, returns, prices and scales must be
exact. The resulting bundle still enforces exact replay on this runtime.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits, threadpool_info

from prediction_service.archive import frame_to_csv_bytes, sha256_bytes
from prediction_service.portfolio import activation_features
from prediction_service.forecast_models import MODELS, CONTEXT_NAMES, PARITY_COLUMNS, calculate_models, assert_parity
from prediction_service.model_registry import ModelBundle, canonical_json, read_frame
import tushare_prediction_pipeline as pipeline

PROBABILITY_TOLERANCE = 1e-12


def compatibility_report(expected, actual):
    if actual.trade_date.duplicated().any() or not actual.trade_date.is_monotonic_increasing:
        raise ValueError("Migration predictions must have unique chronological dates.")
    aligned = actual.set_index("trade_date").reindex(expected.trade_date).reset_index()
    report = {"rows": len(expected), "fields": {}}
    for column in PARITY_COLUMNS:
        left = expected[column].to_numpy(dtype=float)
        right = aligned[column].to_numpy(dtype=float)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Missing or nonfinite migration field: {column}")
        difference = float(np.max(np.abs(left - right)))
        if column in {"confidence", "calibrated_confidence"}:
            if not ((right >= 0) & (right <= 1)).all() or difference > PROBABILITY_TOLERANCE:
                raise ValueError(f"Migration probability drift exceeds 1e-12: {column}")
        elif not np.array_equal(left, right):
            raise ValueError(f"Migration changes a frozen forecast: {column}")
        report["fields"][column] = {"exact": bool(np.array_equal(left, right)), "max_abs": difference}
    return report


def validate_inputs(bundle, market, context):
    current = activation_features(bundle, market)
    for name, frozen in bundle.context().items():
        pd.testing.assert_frame_equal(frozen, context[name].iloc[:len(frozen)].reset_index(drop=True), check_exact=True)
    return current


def migrate(source: Path, context_dir: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Migration bundle already exists: {output}")
    bundle = ModelBundle(source)
    context = {name: read_frame(context_dir / f"{name}.csv") for name in CONTEXT_NAMES}
    market = validate_inputs(bundle, read_frame(context_dir / "market.csv"), context)
    expected = read_frame(bundle.directory / "seed/baseline.csv")
    dates = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    count = int(dates.ge(int(expected.trade_date.iloc[0])).sum())
    print(json.dumps({"stage": "calculate", "signal_rows": count}), flush=True)
    with threadpool_limits(limits=1):
        baseline = pipeline.run_validation_and_prediction(market, validation_days=count - 1, progress=False)
        results = calculate_models(market, baseline, context)
        numerical_libraries = threadpool_info()
    reports = {}
    for model in MODELS:
        filename = "baseline.csv" if model.key == "baseline" else f"{model.key}_candidate_predictions.csv"
        reports[model.key] = compatibility_report(read_frame(bundle.directory / "seed" / filename), results[model.key])
    bundle.verify()
    migration = {
        "source_bundle_id": bundle.manifest["bundle_id"],
        "reason": "platform_numerical_runtime_migration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probability_absolute_tolerance": PROBABILITY_TOLERANCE,
        "other_forecast_fields": "exact",
        "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
        "numerical_libraries": numerical_libraries,
        "original_forecast_compatibility": reports,
        "historical_gate_passed": False,
    }
    seed_files = {name: (bundle.directory / "seed" / name).read_bytes() for name in bundle.manifest["seed_files"]}
    seed_files["features.csv"] = frame_to_csv_bytes(market)
    seed_files["baseline.csv"] = frame_to_csv_bytes(baseline)
    for model in MODELS:
        if model.experiment:
            seed_files[f"{model.key}_candidate_predictions.csv"] = frame_to_csv_bytes(results[model.key])
    for name, frame in context.items():
        seed_files[f"{name}.csv"] = frame_to_csv_bytes(frame)
    receipts = context_dir / "receipts.json"
    if receipts.is_file():
        seed_files["migration_receipts.json"] = receipts.read_bytes()
    seeds = {name: sha256_bytes(content) for name, content in seed_files.items()}
    releases = {}
    for model in MODELS:
        old = bundle.manifest["releases"][model.key]
        config = deepcopy(old["configuration"])
        config["seed_files"] = seeds
        config["runtime_migration"] = {
            "source_bundle_id": bundle.manifest["bundle_id"], "source_release_id": old["release_id"],
            "platform": migration["platform"], "probability_absolute_tolerance": PROBABILITY_TOLERANCE,
        }
        releases[model.key] = {"release_id": sha256_bytes(canonical_json(config)), "configuration": config}
    manifest = deepcopy(bundle.manifest)
    manifest.update(seed_files=seeds, releases=releases, migration=migration,
                    parity={key: {"rows": len(frame), "exact_match": True} for key, frame in results.items()})
    manifest.pop("bundle_id")
    manifest["bundle_id"] = sha256_bytes(canonical_json(manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".runtime-migration-", dir=output.parent))
    try:
        (staging / "seed").mkdir()
        for name, content in seed_files.items():
            (staging / "seed" / name).write_bytes(content)
        (staging / "sources.zip").write_bytes((bundle.directory / "sources.zip").read_bytes())
        (staging / "manifest.json").write_bytes(canonical_json(manifest))
        migrated = ModelBundle(staging)
        print(json.dumps({"stage": "exact_independent_replay", "bundle_id": manifest["bundle_id"]}), flush=True)
        with threadpool_limits(limits=1):
            replay = migrated.calculate(market, context)
        for key in results:
            assert_parity(results[key], replay[key])
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = migrate(args.source, args.context_dir, args.output)
    print(json.dumps({"bundle_id": result["bundle_id"], "releases": {k: v["release_id"] for k, v in result["releases"].items()},
                      "compatibility": result["migration"]["original_forecast_compatibility"]}), flush=True)


if __name__ == "__main__":
    main()
