"""Freeze a small directional error model, select on development years, gate once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tushare_prediction_pipeline as pipeline
from tools.directional_error_calibration import CANDIDATES, COMMON, GLOBAL_CANDIDATES, RULE_CANDIDATES, corrected_predictions, error_threshold_predictions
from tools.evaluate_direction_bias import compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame
from tools.evaluate_selective_context import THRESHOLDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    family = parser.add_mutually_exclusive_group()
    family.add_argument("--global-context", type=Path)
    family.add_argument("--rule-run", type=Path)
    parser.add_argument("--threshold-grid", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline_bytes = args.baseline.read_bytes()
    (args.output / "baseline.csv").write_bytes(baseline_bytes)
    source_hashes = {}
    sources = ["循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
               "tools/directional_error_calibration.py", "tools/evaluate_directional_error.py",
               "tools/evaluate_direction_bias.py", "tools/evaluate_prediction_candidate.py",
               "tools/evaluate_selective_context.py", "tools/context_residual_model.py", "tools/direction_rule_candidates.py"]
    context_hashes = {}
    if args.global_context is not None:
        sources += ["tools/global_risk_features.py", "tools/fetch_global_risk.py"]
        manifest = json.loads((args.global_context / "manifest.json").read_text(encoding="utf-8"))
        if manifest["baseline_sha256"] != hashlib.sha256(baseline_bytes).hexdigest():
            raise ValueError("Overseas context was frozen against a different baseline.")
        for name in ("spx", "nasdaq"):
            raw = (args.global_context / f"{name}.csv").read_bytes()
            context_hashes[name] = hashlib.sha256(raw).hexdigest()
            if context_hashes[name] != manifest["assets"][name]["sha256"]:
                raise ValueError(f"Overseas context hash mismatch: {name}.")
            (args.output / f"{name}.csv").write_bytes(raw)
        (args.output / "context_manifest.json").write_bytes((args.global_context / "manifest.json").read_bytes())
    candidates = GLOBAL_CANDIDATES if args.global_context is not None else CANDIDATES
    if args.rule_run is not None:
        candidates = RULE_CANDIDATES
        sources += ["tools/rule_context_features.py"]
        parent = json.loads((args.rule_run / "contract.json").read_text(encoding="utf-8"))
        if (args.rule_run / "comparison" / "baseline_predictions.csv").read_bytes() != baseline_bytes:
            raise ValueError("Rule diagnostics were frozen with a different baseline.")
        files = {"features.csv": args.rule_run / "features.csv",
                 "rule_diagnostics.csv": args.rule_run / "comparison" / "baseline_diagnostics.csv",
                 "rule_contract.json": args.rule_run / "contract.json"}
        for name, path in files.items():
            raw = path.read_bytes()
            context_hashes[name] = hashlib.sha256(raw).hexdigest()
            (args.output / name).write_bytes(raw)
        if context_hashes["features.csv"] != parent["input_sha256"]:
            raise ValueError("Rule market input does not match the frozen contract.")
        for name in ("循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py"):
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != parent["sources"][name]:
                raise ValueError(f"Rule source no longer matches frozen baseline: {name}.")
    for name in sources:
        raw = (ROOT / name).read_bytes()
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(), "sources": source_hashes,
        "candidates": candidates, "common": COMMON, "sklearn_version": version("scikit-learn"),
        "context_sha256": context_hashes,
        "thresholds": THRESHOLDS if args.threshold_grid else [COMMON["correction_probability"]],
        "candidate_attempts": len(candidates) * (len(THRESHOLDS) if args.threshold_grid else 1),
        "selection_years": [2023, 2024],
        "eligibility": "accuracy and balanced accuracy nondecreasing in both years; at least 5 changes",
        "selection": "minimum yearly gain, combined balanced accuracy, fewer changes",
        "regression_data_previously_observed": True, "gate": "unchanged compare_frames",
        "automatic_promotion": False,
    })
    champion = validate_frame(pd.read_csv(args.output / "baseline.csv", float_precision="round_trip"))
    core = pipeline.prediction_core
    context = None
    rule_context = None
    if args.global_context is not None:
        from tools.global_risk_features import ASSETS, global_risk_features
        assets = {name: pd.read_csv(args.output / f"{name}.csv", float_precision="round_trip") for name in ASSETS}
        context = global_risk_features(champion.trade_date, assets)
        context.to_csv(args.output / "context_features.csv", index=False, float_format="%.17g")
    if args.rule_run is not None:
        from tools.rule_context_features import rule_context_features
        config = core.DirectionPredictionConfig(**parent["config"])
        data = pd.read_csv(args.output / "features.csv", float_precision="round_trip")
        diagnostics = pd.read_csv(args.output / "rule_diagnostics.csv", float_precision="round_trip")
        rule_context = rule_context_features(core, data, champion, diagnostics, config, parent["loop_options"])
        rule_context.to_csv(args.output / "rule_features.csv", index=False, float_format="%.17g")
    development = champion.loc[champion.trade_date <= 20241231].reset_index(drop=True)
    baseline_years = {year: metrics(development.loc[development.trade_date.between(year * 10000, year * 10000 + 1231)])
                      for year in (2023, 2024)}
    (args.output / "development").mkdir()
    outcomes = []
    for name in candidates:
        development_context = context.iloc[:len(development)].copy() if context is not None else None
        development_rule = rule_context.iloc[:len(development)].copy() if rule_context is not None else None
        underlying = corrected_predictions(core, development, name, context=development_context, rule_context=development_rule)
        for threshold in (THRESHOLDS if args.threshold_grid else (COMMON["correction_probability"],)):
            result = error_threshold_predictions(core, development, underlying, threshold) if args.threshold_grid else underlying
            filename = f"{name}_p{round(threshold * 100)}.csv" if args.threshold_grid else f"{name}.csv"
            result.to_csv(args.output / "development" / filename, index=False, float_format="%.17g")
            yearly = {year: metrics(result.loc[result.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
            gains = [yearly[year][metric] - baseline_years[year][metric] for year in (2023, 2024)
                     for metric in ("accuracy", "balanced_accuracy")]
            selected_rows = result.loc[result.trade_date >= 20230101]
            changes = int(selected_rows.correction_selected.sum())
            outcome = {"candidate": name, "threshold": threshold, "prediction_file": filename,
                       "yearly": yearly, "minimum_gain": min(gains),
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
        result = corrected_predictions(core, champion, selected["candidate"], context=context, rule_context=rule_context)
        if args.threshold_grid:
            result = error_threshold_predictions(core, champion, result, selected["threshold"])
        prefix = pd.read_csv(args.output / "development" / selected["prediction_file"], float_precision="round_trip")
        columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence", "error_model_correctness_probability"]
        pd.testing.assert_frame_equal(prefix[columns], result.iloc[:len(prefix)][columns], check_exact=True)
        result.to_csv(args.output / "candidate_predictions.csv", index=False, float_format="%.17g")
        report = {"candidate": selected["candidate"], "threshold": selected["threshold"], **compare_frames(champion, result)}
    write_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
