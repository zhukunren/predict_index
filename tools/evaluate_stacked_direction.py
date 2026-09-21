"""Evaluate a frozen stack of all previously declared context/error models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tushare_prediction_pipeline as pipeline
from tools.context_residual_model import CANDIDATES, ERROR_CANDIDATES, context_features, load_assets, residual_predictions
from tools.directional_error_calibration import CANDIDATES as LINEAR_CANDIDATES, COMMON, STACKED_CANDIDATES, corrected_predictions
from tools.evaluate_direction_bias import compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-run", type=Path, required=True)
    parser.add_argument("--correctness-run", type=Path, required=True)
    parser.add_argument("--linear-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = {name: args.direction_run / name for name in ("features.csv", "baseline.csv", "csi300.csv", "csi500.csv", "chinext.csv")}
    sources_by_name = ({name: args.direction_run for name in CANDIDATES}
                       | {name: args.correctness_run for name in ERROR_CANDIDATES}
                       | {name: args.linear_run for name in LINEAR_CANDIDATES})
    hashes = {}
    for name, parent in sources_by_name.items():
        files[f"auxiliary_development/{name}.csv"] = parent / "development" / f"{name}.csv"
    for prefix, parent in (("direction", args.direction_run), ("correctness", args.correctness_run), ("linear", args.linear_run)):
        files[f"parent_contracts/{prefix}.json"] = parent / "contract.json"
        if (parent / "baseline.csv").read_bytes() != files["baseline.csv"].read_bytes():
            raise ValueError("Underlying models must use the same frozen incumbent stream.")
    for name, source in files.items():
        raw = source.read_bytes()
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    source_hashes = {}
    for name in ("循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
                 "数据拉取脚本_tushare.py", "tools/directional_error_calibration.py", "tools/context_residual_model.py",
                 "tools/evaluate_stacked_direction.py", "tools/evaluate_selective_context.py", "tools/evaluate_direction_bias.py",
                 "tools/evaluate_prediction_candidate.py", "tools/direction_rule_candidates.py"):
        raw = (ROOT / name).read_bytes()
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "inputs": hashes, "sources": source_hashes,
        "candidates": STACKED_CANDIDATES, "common": COMMON, "auxiliary_models": list(sources_by_name),
        "sklearn_version": version("scikit-learn"), "xgboost_version": version("xgboost"),
        "selection_years": [2023, 2024], "regression_data_previously_observed": True,
        "eligibility": "accuracy and balanced accuracy nondecreasing in both development years; at least 5 changes",
        "selection": "minimum yearly gain, combined balanced accuracy, fewer changes",
        "gate": "unchanged compare_frames", "automatic_promotion": False,
        "training": "past outcomes and independently causal auxiliary predictions only; all auxiliary models must have fitted",
    })
    champion = validate_frame(pd.read_csv(args.output / "baseline.csv", float_precision="round_trip"))
    development = champion.loc[champion.trade_date <= 20241231].reset_index(drop=True)
    auxiliary = {name: pd.read_csv(args.output / "auxiliary_development" / f"{name}.csv", float_precision="round_trip")
                 for name in sources_by_name}
    core = pipeline.prediction_core
    baseline_years = {year: metrics(development.loc[development.trade_date.between(year * 10000, year * 10000 + 1231)])
                      for year in (2023, 2024)}
    (args.output / "development").mkdir()
    outcomes = []
    for name in STACKED_CANDIDATES:
        result = corrected_predictions(core, development, name, auxiliary=auxiliary)
        result.to_csv(args.output / "development" / f"{name}.csv", index=False, float_format="%.17g")
        yearly = {year: metrics(result.loc[result.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
        gains = [yearly[year][metric] - baseline_years[year][metric] for year in (2023, 2024)
                 for metric in ("accuracy", "balanced_accuracy")]
        selected_rows = result.loc[result.trade_date >= 20230101]
        changes = int(selected_rows.correction_selected.sum())
        outcome = {"candidate": name, "yearly": yearly, "minimum_gain": min(gains),
                   "combined": metrics(selected_rows), "changed_rows": changes,
                   "eligible": min(gains) >= -1e-12 and changes >= 5}
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)
    eligible = [item for item in outcomes if item["eligible"]]
    selected = max(eligible, key=lambda item: (item["minimum_gain"], item["combined"]["balanced_accuracy"], -item["changed_rows"])) if eligible else None
    write_json(args.output / "selection.json", {"selected": selected, "outcomes": outcomes, "baseline_years": baseline_years})
    if selected is None:
        report = {"passed": False, "stage": "development", "reason": "No candidate passed both development years.", "production_changed": False}
    else:
        config, _ = pipeline._default_calculation_options()
        data = pd.read_csv(args.output / "features.csv", float_precision="round_trip")
        base, features = context_features(core, data, config, load_assets(args.output))
        full_auxiliary = {}
        (args.output / "auxiliary_full").mkdir()
        for name in sources_by_name:
            if name in LINEAR_CANDIDATES:
                full = corrected_predictions(core, champion, name)
                probability_column = "error_model_correctness_probability"
            else:
                full = residual_predictions(core, champion, base, features, name)
                probability_column = "residual_probability_up"
            old = auxiliary[name]
            if not np.array_equal(full.trade_date.iloc[:len(old)], old.trade_date) or not np.array_equal(full[probability_column].iloc[:len(old)], old[probability_column]):
                raise ValueError(f"Frozen auxiliary prefix parity failed: {name}.")
            full_auxiliary[name] = full
            full.to_csv(args.output / "auxiliary_full" / f"{name}.csv", index=False, float_format="%.17g")
            print(json.dumps({"auxiliary_completed": name}), flush=True)
        result = corrected_predictions(core, champion, selected["candidate"], auxiliary=full_auxiliary)
        frozen = pd.read_csv(args.output / "development" / f"{selected['candidate']}.csv", float_precision="round_trip")
        columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence", "error_model_correctness_probability"]
        pd.testing.assert_frame_equal(frozen[columns], result.iloc[:len(frozen)][columns])
        result.to_csv(args.output / "candidate_predictions.csv", index=False, float_format="%.17g")
        report = {"candidate": selected["candidate"], **compare_frames(champion, result)}
    write_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
