"""Test completed offshore RMB quotes with a fixed causal learning protocol."""

import argparse
from dataclasses import asdict, replace
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
from tools.downside_specialist import specialist_features, specialist_predictions
from tools.evaluate_direction_bias import compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame
from tools.evaluate_trend_breadth import probability_metrics
from tools.fx_features import FEATURE_COLUMNS, POLICY, fx_features
from tools.nested_downside import (DEFAULT_CONFIG, nested_downside_probabilities, selected_threshold_predictions,
                                  nested_error_probabilities, selected_error_predictions)

THRESHOLD = 0.60
THRESHOLD_CANDIDATES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def error_probability_metrics(champion, probabilities):
    selected = probabilities.error_training_rows.gt(0) & champion.real_pct_change.notna()
    p = probabilities.loc[selected, "error_probability"].to_numpy()
    y = champion.loc[selected, "predicted_label"].ne(champion.loc[selected, "real_pct_change"].gt(0)).to_numpy(dtype=int)
    if not len(y):
        return {"rows": 0, "brier": None, "log_loss": None}
    bounded = np.clip(p, 1e-12, 1-1e-12)
    return {"rows": len(y), "brier": float(np.mean((p-y)**2)),
            "log_loss": float(-np.mean(y*np.log(bounded)+(1-y)*np.log1p(-bounded)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--fx-context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-selection", action="store_true",
                        help="Keep the incumbent unless prior validation corrections improve it without per-fold direction-metric regression.")
    parser.add_argument("--learn-threshold", action="store_true",
                        help="Select a decision threshold inside earlier validation together with C and features; requires paired selection.")
    parser.add_argument("--pooled-error", action="store_true",
                        help="Learn incumbent error from both directions with signed feature interactions and fixed 0.5 decision threshold.")
    args = parser.parse_args()
    if args.learn_threshold and not args.paired_selection:
        parser.error("--learn-threshold requires --paired-selection")
    if args.pooled_error and (not args.paired_selection or args.learn_threshold):
        parser.error("--pooled-error requires --paired-selection and a fixed threshold")
    learning_config = replace(DEFAULT_CONFIG, require_paired_nonregression=args.paired_selection,
                              correction_threshold=.50 if args.pooled_error else THRESHOLD,
                              threshold_candidates=THRESHOLD_CANDIDATES if args.learn_threshold else ())
    seed_raw = (args.seed / "contract.json").read_bytes()
    seed = json.loads(seed_raw)
    fx_raw = (args.fx_context / "manifest.json").read_bytes()
    fx_contract = json.loads(fx_raw)
    files = {name: args.seed / name for name in ("features.csv", "baseline.csv", "breadth.csv", "spx.csv", "nasdaq.csv")}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    if any(hashes[name] != seed["inputs"][name] for name in files):
        raise ValueError("Original market or model seed changed.")
    if (fx_contract["status"] != "complete"
            or fx_contract["contract"]["baseline_sha256"] != hashes["baseline.csv"]
            or hashlib.sha256((args.fx_context / "cnh.csv").read_bytes()).hexdigest() != fx_contract["cnh_sha256"]):
        raise ValueError("FX data do not match the frozen original-model seed.")
    sources = ["tools/evaluate_fx_downside.py", "tools/fx_features.py", "tools/nested_downside.py", "tools/evaluate_trend_breadth.py",
               "tools/trend_breadth_features.py", "tools/downside_specialist.py", "tools/evaluate_selective_context.py",
               "tools/context_residual_model.py", "tools/evaluate_direction_bias.py", "tools/evaluate_prediction_candidate.py",
               "tools/direction_rule_candidates.py", "tools/breadth_features.py", "tools/global_risk_features.py",
               "tools/liquidity_features.py", "return_calibration.py", "regularized_direction.py",
               "循环验证脚本.py", "tushare_prediction_pipeline.py", "数据拉取脚本_tushare.py"]
    args.output.mkdir(parents=True, exist_ok=False)
    for name, source in files.items():
        (args.output / name).write_bytes(source.read_bytes())
    (args.output / "cnh.csv").write_bytes((args.fx_context / "cnh.csv").read_bytes())
    (args.output / "fx_manifest.json").write_bytes(fx_raw)
    (args.output / "seed_contract.json").write_bytes(seed_raw)
    source_hashes = {}
    for name in sources:
        raw = (ROOT / name).read_bytes()
        dest = args.output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
    write_json(args.output / "contract.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": ("Pooling all prior incumbent outcomes with original-direction feature interactions can learn errors in both directions without discarding half the training observations. Earlier paired validation admits auxiliary corrections."
                       if args.pooled_error else "Joint chronological selection of probability model and decision threshold can align actual direction decisions with prior paired improvement over the incumbent."
                       if args.learn_threshold else "Adding the incumbent as an inner selection option can avoid auxiliary models whose probability fit does not improve actual direction decisions; offshore RMB inputs are admitted only through earlier paired validation."
                       if args.paired_selection else "Completed offshore RMB quote returns and volatility can expose currency pressure absent from domestic price and daily equity breadth, helping identify downside conditional on an incumbent up forecast."),
        "incumbent_release": "956dd395b7f9df52a46bf6f739f81eb5a1b738c1180d72abfc7c741bdf6522e9",
        "inputs": hashes, "fx_manifest_sha256": hashlib.sha256(fx_raw).hexdigest(), "sources": source_hashes,
        "parameters": asdict(learning_config), "threshold": "selected_from_prior_validation" if args.learn_threshold else learning_config.correction_threshold,
        "target": "incumbent error, trained on both original directions" if args.pooled_error else "downside conditional on original up",
        "direction_interactions": "each market feature multiplied by signed original direction, plus direction indicator" if args.pooled_error else None,
        "inner_search": {"candidate_configurations_per_refit": 36 if args.learn_threshold else 6,
                         "control_configurations_per_refit": 18 if args.learn_threshold else 3,
                         "objective": ("pooled paired accuracy gain, then pooled balanced accuracy gain; each fold must be nondecreasing"
                                       if args.learn_threshold else "mean past chronological validation log loss"),
                         "features": ("base 18 plus signed interactions and direction (37 columns) versus base plus FX and their interactions (49 columns)"
                                      if args.pooled_error else "base 18 versus base plus six FX features, admitted as one block"),
                         "tie_break": ("fewer changed validation decisions, lower log loss, base features, stronger shrinkage, higher threshold"
                                       if args.learn_threshold else "base features, then stronger shrinkage"),
                         "incumbent_option": args.paired_selection,
                         "admission": ("Each prior validation block's accuracy and balanced accuracy must not decrease against incumbent; positive total corrected-minus-damaged count required; otherwise keep incumbent."
                                       if args.paired_selection else None)},
        "candidate_attempts": 1, "outer_parameter_search": False,
        "data_policy": POLICY, "fx_features": FEATURE_COLUMNS,
        "development": [2023, 2024],
        "development_eligibility": "Both years accuracy and balanced accuracy must not decline; at least five candidate changes; same algorithm in all years.",
        "regression_gate": "Only after development passes: unchanged 26 checks in evaluate_direction_bias.compare_frames; also report both recalls and paired changes.",
        "regression_data_previously_observed": True, "automatic_promotion": False,
        "packages": {name:version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost")},
    })
    core = pipeline.prediction_core
    config, _ = pipeline._default_calculation_options()
    market = pd.read_csv(args.output / "features.csv", float_precision="round_trip")
    champion = validate_frame(pd.read_csv(args.output / "baseline.csv", float_precision="round_trip"))
    breadth = pd.read_csv(args.output / "breadth.csv", float_precision="round_trip")
    overseas = {name:pd.read_csv(args.output / f"{name}.csv", float_precision="round_trip") for name in ("spx", "nasdaq")}
    quotes = pd.read_csv(args.output / "cnh.csv", float_precision="round_trip")

    def inputs(frame):
        cutoff = int(frame.trade_date.iloc[-1])
        prefix = market.loc[pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int).le(cutoff)]
        common = specialist_features(core, prefix, frame, config, breadth, overseas)
        extra = fx_features(frame.trade_date, quotes.loc[quotes.trade_date.lt(cutoff)])
        return common, pd.concat([common, extra.loc[:, FEATURE_COLUMNS]], axis=1)

    def predict(frame, features, optional, name):
        learner = nested_error_probabilities if args.pooled_error else nested_downside_probabilities
        probabilities, attempts = learner(frame, features, optional, config=learning_config)
        write_json(args.output / name, attempts)
        return probabilities

    def make_predictions(frame, probabilities):
        return (selected_error_predictions(core, frame, probabilities) if args.pooled_error else
                selected_threshold_predictions(core, frame, probabilities) if args.learn_threshold
                else specialist_predictions(core, frame, probabilities, THRESHOLD))

    dev = champion.loc[champion.trade_date.le(20241231)].reset_index(drop=True)
    common, extended = inputs(dev)
    directory = args.output / "development"
    directory.mkdir()
    extended.to_csv(directory / "features.csv", index=False, float_format="%.17g")
    results, probabilities = {}, {}
    for name, values, columns in (("control", common, ()), ("candidate", extended, FEATURE_COLUMNS)):
        p = predict(dev, values, columns, f"development/{name}_inner_selections.json")
        probabilities[name] = p
        results[name] = validate_frame(make_predictions(dev, p))
        p.to_csv(directory / f"{name}_probabilities.csv", index=False, float_format="%.17g")
        results[name].to_csv(directory / f"{name}_predictions.csv", index=False, float_format="%.17g")
        print(json.dumps({"stage": "development_complete", "variant": name}), flush=True)
    yearly, gains = {}, []
    for year in (2023, 2024):
        mask = dev.trade_date.between(year * 10000 + 101, year * 10000 + 1231)
        yearly[year] = {"incumbent": metrics(dev.loc[mask])}
        for name in results:
            yearly[year][name] = metrics(results[name].loc[mask])
            if args.pooled_error:
                yearly[year][name]["conditional_error_probability"] = error_probability_metrics(dev.loc[mask], probabilities[name].loc[mask])
            else:
                yearly[year][name]["conditional_downside_probability"] = probability_metrics(dev.loc[mask], probabilities[name].loc[mask])
        gains.extend(yearly[year]["candidate"][metric] - yearly[year]["incumbent"][metric] for metric in ("accuracy", "balanced_accuracy"))
    changes = int(results["candidate"].loc[dev.trade_date.ge(20230101), "correction_selected"].sum())
    eligible = min(gains) >= -1e-12 and changes >= 5
    write_json(args.output / "development.json", {"years": yearly, "eligible": eligible, "changes": changes, "minimum_gain": min(gains)})
    prefix = dev.loc[dev.trade_date.le(20240628)].reset_index(drop=True)
    _, prefix_features = inputs(prefix)
    unknown = prefix.copy()
    unknown.loc[unknown.index[-1], "real_pct_change"] = np.nan
    p = predict(unknown, prefix_features, FEATURE_COLUMNS, "prefix_inner_selections.json")
    pd.testing.assert_frame_equal(p, probabilities["candidate"].iloc[:len(prefix)], check_exact=True)
    pd.testing.assert_frame_equal(prefix_features, extended.iloc[:len(prefix)], check_exact=True)
    write_json(args.output / "prefix_verification.json", {"signal_date": int(prefix.trade_date.iloc[-1]), "rows": len(prefix),
                                                         "last_outcome_masked": True, "features_and_probabilities_identical": True})
    if not eligible:
        report = {"passed": False, "stage": "development", "reason": "FX candidate failed development requirements; no regression run.", "production_changed": False}
    else:
        _, features = inputs(champion)
        p = predict(champion, features, FEATURE_COLUMNS, "full_inner_selections.json")
        result = validate_frame(make_predictions(champion, p))
        pd.testing.assert_frame_equal(p.iloc[:len(dev)], probabilities["candidate"], check_exact=True)
        pd.testing.assert_frame_equal(result.iloc[:len(dev)], results["candidate"], check_exact=True)
        result.to_csv(args.output / "candidate_predictions.csv", index=False, float_format="%.17g")
        report = {"candidate": ("pooled_cnh_error" if args.pooled_error else "learned_threshold_cnh_downside" if args.learn_threshold else "paired_nested_cnh_downside"
                                if args.paired_selection else "nested_cnh_downside"),
                  **compare_frames(champion, result), "independent_future_validation": False}
        monthly = {month: {"incumbent": metrics(champion.loc[group.index]), "candidate": metrics(group)}
                   for month, group in result.groupby(result.trade_date.astype(str).str[:6])}
        write_json(args.output / "monthly.json", monthly)
    write_json(args.output / "report.json", report)
    print(json.dumps({"development_eligible": eligible, "changes": changes, "minimum_gain": min(gains),
                      "passed": report["passed"], "stage": report.get("stage", "regression"),
                      "failed_checks": [c for c in report.get("checks", []) if not c["passed"]]}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
