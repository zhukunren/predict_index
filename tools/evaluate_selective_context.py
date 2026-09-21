"""Select sparse context corrections on two development years only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tushare_prediction_pipeline as pipeline
from return_calibration import calibrate_returns
from tools.context_residual_model import CANDIDATES, ERROR_CANDIDATES, FUNDING_CANDIDATES, COMMON, context_features, load_assets, residual_predictions
from tools.evaluate_direction_bias import compare_frames, metrics, write_json


THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75)


def selective_predictions(core, champion, candidate, threshold):
    if not np.array_equal(champion.trade_date, candidate.trade_date):
        raise ValueError("Selective correction requires identical signal dates.")
    if not np.allclose(champion.real_pct_change, candidate.real_pct_change, atol=0, rtol=0, equal_nan=True):
        raise ValueError("Selective correction requires identical targets.")
    frame = champion.copy().reset_index(drop=True)
    old_label = frame.predicted_label.to_numpy(dtype=int).copy()
    candidate_label = candidate.predicted_label.to_numpy(dtype=int)
    probability = candidate.residual_probability_up.to_numpy(dtype=float)
    certainty = np.where(candidate_label == 1, probability, 1 - probability)
    selected = (old_label != candidate_label) & (certainty >= threshold)
    label = np.where(selected, candidate_label, old_label)
    base_close = frame.predicted_close / (1 + frame.predicted_pct_change)
    frame["predicted_label"] = label
    frame["predicted_pct_change"] = np.abs(frame.uncalibrated_predicted_return) * np.where(label == 1, 1, -1)
    frame["predicted_close"] = base_close * (1 + frame.predicted_pct_change)
    frame["confidence"] = np.where(selected, certainty, frame.confidence)
    frame["correct"] = pd.Series(label == frame.real_pct_change.gt(0)).where(frame.real_pct_change.notna(), None)
    frame = core._apply_rolling_confidence_calibration(frame, window=300, min_rows=60, method="platt")
    frame = calibrate_returns(frame, window=252, min_rows=60)
    frame["correction_selected"] = selected
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-run", type=Path, required=True)
    parser.add_argument("--correctness-run", type=Path)
    parser.add_argument("--family", choices=("context", "funding"), default="context")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.family == "context" and args.correctness_run is None:
        parser.error("--family context requires --correctness-run")
    args.output.mkdir(parents=True, exist_ok=False)
    config, _ = pipeline._default_calculation_options()
    core = pipeline.prediction_core
    files = {"features.csv": args.direction_run / "features.csv", "baseline.csv": args.direction_run / "baseline.csv"}
    files.update({f"{name}.csv": args.direction_run / f"{name}.csv" for name in ("csi300", "csi500", "chinext")})
    candidates = FUNDING_CANDIDATES if args.family == "funding" else CANDIDATES | ERROR_CANDIDATES
    if args.family == "funding":
        files.update({name: args.direction_run / name for name in ("index_basic.csv", "margin.csv", "funding_manifest.json")})
    for name in candidates:
        parent = args.direction_run if args.family == "funding" or name in CANDIDATES else args.correctness_run
        files[f"development/{name}.csv"] = parent / "development" / f"{name}.csv"
    input_hashes = {}
    for name, original in files.items():
        raw = original.read_bytes()
        input_hashes[name] = hashlib.sha256(raw).hexdigest()
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    source_hashes = {}
    sources = ["循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
               "数据拉取脚本_tushare.py", "tools/context_residual_model.py", "tools/evaluate_selective_context.py",
               "tools/evaluate_direction_bias.py", "tools/evaluate_prediction_candidate.py", "tools/direction_rule_candidates.py"]
    if args.family == "funding":
        sources.append("tools/funding_features.py")
    for name in sources:
        raw = (ROOT / name).read_bytes()
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "inputs": input_hashes, "sources": source_hashes,
        "config": asdict(config), "common": COMMON, "candidates": candidates, "thresholds": THRESHOLDS,
        "attempt_count": len(candidates) * len(THRESHOLDS), "selection_years": [2023, 2024],
        "eligibility": "accuracy and balanced accuracy cannot decline in either development year; at least 5 changed development directions",
        "selection": "greatest minimum yearly gain, then combined balanced accuracy, then fewer changes",
        "regression_data_previously_observed": True, "automatic_promotion": False,
    })
    champion = pd.read_csv(args.output / "baseline.csv", float_precision="round_trip")
    development_champion = champion.loc[champion.trade_date <= 20241231].reset_index(drop=True)
    yearly_baseline = {year: metrics(development_champion.loc[development_champion.trade_date.between(year * 10000, year * 10000 + 1231)])
                       for year in (2023, 2024)}
    outcomes = []
    for name in candidates:
        candidate = pd.read_csv(args.output / "development" / f"{name}.csv", float_precision="round_trip")
        for threshold in THRESHOLDS:
            result = selective_predictions(core, development_champion, candidate, threshold)
            yearly = {year: metrics(result.loc[result.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
            gains = [yearly[year][metric] - yearly_baseline[year][metric] for year in (2023, 2024)
                     for metric in ("accuracy", "balanced_accuracy")]
            development = result.loc[result.trade_date >= 20230101]
            changed = int(development.correction_selected.sum())
            outcomes.append({"candidate": name, "threshold": threshold, "yearly": yearly,
                             "combined": metrics(development), "minimum_gain": min(gains), "changed_rows": changed,
                             "eligible": min(gains) >= -1e-12 and changed >= 5})
        print(json.dumps({"candidate": name, "development_thresholds_finished": len(THRESHOLDS)}), flush=True)
    eligible = [outcome for outcome in outcomes if outcome["eligible"]]
    selected = max(eligible, key=lambda item: (item["minimum_gain"], item["combined"]["balanced_accuracy"], -item["changed_rows"])) if eligible else None
    write_json(args.output / "selection.json", {"selected": selected, "outcomes": outcomes, "yearly_baseline": yearly_baseline})
    if selected is None:
        write_json(args.output / "report.json", {"passed": False, "stage": "development", "reason": "No candidate passed both development years.", "production_changed": False})
        print("No candidate passed both development years.", flush=True)
        return 2
    data = pd.read_csv(args.output / "features.csv", float_precision="round_trip")
    base, features = context_features(core, data, config, load_assets(args.output))
    if args.family == "funding":
        from tools.funding_features import funding_features, load_assets as load_funding
        funding = funding_features(base.date, load_funding(args.output))
        features = pd.concat([features, funding.drop(columns="funding_source_date")], axis=1)
    underlying = residual_predictions(core, champion, base, features, selected["candidate"])
    result = selective_predictions(core, champion, underlying, selected["threshold"])
    result.to_csv(args.output / "candidate_predictions.csv", index=False, float_format="%.17g")
    report = {"candidate": selected["candidate"], "threshold": selected["threshold"], **compare_frames(champion, result)}
    write_json(args.output / "report.json", report)
    print(json.dumps({"selected": {"candidate": selected["candidate"], "threshold": selected["threshold"]},
                      "passed": report["passed"], "changes": report["paired_direction_changes"],
                      "failed_checks": [item for item in report["checks"] if not item["passed"]]}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
