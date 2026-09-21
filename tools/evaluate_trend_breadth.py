"""One declared trend-breadth hypothesis, with a matched feature ablation."""

import argparse
from dataclasses import asdict
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
from tools.downside_specialist import COMMON, CANDIDATES, downside_probabilities, specialist_features, specialist_predictions
from tools.evaluate_direction_bias import compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame
from tools.trend_breadth_features import FEATURE_COLUMNS, POLICY, trend_breadth_features

MODEL = "downside_logistic"
THRESHOLD = 0.60


def probability_metrics(champion, probabilities):
    selected = champion.predicted_label.eq(1) & probabilities.downside_training_rows.gt(0) & champion.real_pct_change.notna()
    probability = probabilities.loc[selected, "downside_probability"].to_numpy()
    labels = champion.loc[selected, "real_pct_change"].le(0).to_numpy(dtype=int)
    if len(labels) == 0:
        return {"rows": 0, "brier": None, "log_loss": None}
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return {"rows": len(labels), "brier": float(np.mean((probability - labels)**2)),
            "log_loss": float(-np.mean(labels * np.log(clipped) + (1-labels) * np.log(1-clipped)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--trend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nested-selection", action="store_true",
                        help="Choose shrinkage and the optional trend block on prior chronological validation only.")
    args = parser.parse_args()
    nested_config = None
    if args.nested_selection:
        from tools.nested_downside import DEFAULT_CONFIG, nested_downside_probabilities
        nested_config = asdict(DEFAULT_CONFIG)
    contract_raw = (args.seed / "contract.json").read_bytes()
    seed_contract = json.loads(contract_raw)
    trend_raw = (args.trend / "manifest.json").read_bytes()
    trend_contract = json.loads(trend_raw)
    files = {name: args.seed / name for name in ("features.csv", "baseline.csv", "breadth.csv", "spx.csv", "nasdaq.csv")}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    if any(hashes[name] != seed_contract["inputs"][name] for name in files):
        raise ValueError("Frozen incumbent input changed.")
    if (trend_contract["status"] != "complete"
            or trend_contract["contract"]["input_sha256"] != hashes["features.csv"]
            or trend_contract["contract"]["baseline_sha256"] != hashes["baseline.csv"]
            or hashlib.sha256((args.trend / "trend_breadth.csv").read_bytes()).hexdigest() != trend_contract["trend_sha256"]):
        raise ValueError("Trend context and original model inputs must match.")
    sources = ["tools/evaluate_trend_breadth.py", "tools/trend_breadth_features.py", "tools/downside_specialist.py",
               "tools/evaluate_selective_context.py", "tools/context_residual_model.py", "tools/evaluate_direction_bias.py",
               "tools/evaluate_prediction_candidate.py", "tools/direction_rule_candidates.py", "tools/breadth_features.py",
               "tools/global_risk_features.py", "tools/liquidity_features.py", "return_calibration.py",
               "regularized_direction.py", "循环验证脚本.py", "tushare_prediction_pipeline.py", "数据拉取脚本_tushare.py"]
    if args.nested_selection:
        sources.append("tools/nested_downside.py")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, path in files.items():
        (args.output / name).write_bytes(path.read_bytes())
    (args.output / "trend_breadth.csv").write_bytes((args.trend / "trend_breadth.csv").read_bytes())
    (args.output / "trend_manifest.json").write_bytes(trend_raw)
    (args.output / "seed_contract.json").write_bytes(contract_raw)
    source_hashes = {}
    for name in sources:
        raw = (ROOT / name).read_bytes()
        dest = args.output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
    write_json(args.output / "contract.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": ("Earlier chronological validation can control overfitting from redundant trend features by selecting shrinkage and whether to include their declared block."
                       if args.nested_selection else "Cross-sectional multi-session trend participation adds information about next-session downside missed by daily breadth."),
        "incumbent_release": "956dd395b7f9df52a46bf6f739f81eb5a1b738c1180d72abfc7c741bdf6522e9",
        "inputs": hashes, "sources": source_hashes, "trend_manifest_sha256": hashlib.sha256(trend_raw).hexdigest(),
        "model": "nested_downside_logistic" if args.nested_selection else MODEL,
        "model_parameters": nested_config if args.nested_selection else CANDIDATES[MODEL],
        "training": nested_config if args.nested_selection else COMMON,
        "threshold": THRESHOLD, "threshold_search": False, "candidate_attempts": 1,
        "trend_features": FEATURE_COLUMNS, "data_policy": POLICY,
        "ablation": ("Both use the same nested validation, training dates and availability. Control selects C using base features only; candidate selects among base and base+trend blocks. Control is never picked as an alternative after evaluation."
                     if args.nested_selection else "Same model, training dates and availability mask; control uses only existing 18 features; candidate adds eight trend features. Control is never selected as an alternative."),
        "inner_search": ({"configurations_per_refit": 6, "control_configurations_per_refit": 3,
                          "metric": "mean chronological validation log loss",
                          "tie_break": "base features, then stronger shrinkage",
                          "standardization": "fitted inside each past training fold", "outer_parameter_search": False}
                         if args.nested_selection else None),
        "development": [2023, 2024],
        "development_eligibility": "Each year's accuracy and balanced accuracy must not decline versus incumbent; at least five direction changes; train/predict identically in all years.",
        "regression": "Only if development eligible: original unchanged 26 checks on 2025+ and last 252/60/20; report both recalls and monthly metrics.",
        "regression_data_previously_observed": True, "automatic_promotion": False,
        "excluded_predictors": ["forecast streak length", "forecast direction quotas", "development/regression year flags"],
        "packages": {n:version(n) for n in ("numpy", "pandas", "scikit-learn", "xgboost")},
    })
    core = pipeline.prediction_core
    config, _ = pipeline._default_calculation_options()
    market = pd.read_csv(args.output / "features.csv", float_precision="round_trip")
    champion = validate_frame(pd.read_csv(args.output / "baseline.csv", float_precision="round_trip"))
    breadth = pd.read_csv(args.output / "breadth.csv", float_precision="round_trip")
    overseas = {n:pd.read_csv(args.output / f"{n}.csv", float_precision="round_trip") for n in ("spx", "nasdaq")}
    trend = pd.read_csv(args.output / "trend_breadth.csv", float_precision="round_trip")

    def inputs(frame):
        market_prefix = market.loc[pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int).le(frame.trade_date.iloc[-1])]
        calendar = pd.to_datetime(market_prefix.trade_date).dt.strftime("%Y%m%d").astype(int)
        common = specialist_features(core, market_prefix, frame, config, breadth, overseas)
        extra = trend_breadth_features(frame.trade_date, trend, calendar)
        return common, pd.concat([common, extra.loc[:, FEATURE_COLUMNS]], axis=1), extra.trend_available.to_numpy(dtype=bool)

    dev = champion.loc[champion.trade_date.le(20241231)].reset_index(drop=True)
    common, extended, available = inputs(dev)
    devdir = args.output / "development"
    devdir.mkdir()
    extended.to_csv(devdir / "features.csv", index=False, float_format="%.17g")
    results, probabilities = {}, {}
    def predict_probabilities(frame, values, available, *, allow_trend, attempts_name):
        if args.nested_selection:
            probabilities, attempts = nested_downside_probabilities(
                frame, values, FEATURE_COLUMNS if allow_trend else (), available=available,
            )
            write_json(args.output / attempts_name, attempts)
            return probabilities
        return downside_probabilities(frame, values, MODEL, available=available)

    for name, values in (("control", common), ("candidate", extended)):
        probabilities[name] = predict_probabilities(dev, values, available, allow_trend=name == "candidate",
                                                    attempts_name=f"development/{name}_inner_selections.json")
        results[name] = validate_frame(specialist_predictions(core, dev, probabilities[name], THRESHOLD))
        probabilities[name].to_csv(devdir / f"{name}_probabilities.csv", index=False, float_format="%.17g")
        results[name].to_csv(devdir / f"{name}_predictions.csv", index=False, float_format="%.17g")
        print(json.dumps({"stage": "development_complete", "variant": name}), flush=True)
    yearly = {}
    gains = []
    for year in (2023, 2024):
        mask = dev.trade_date.between(year * 10000 + 101, year * 10000 + 1231)
        yearly[year] = {"incumbent": metrics(dev.loc[mask])}
        for name in results:
            yearly[year][name] = metrics(results[name].loc[mask])
            yearly[year][name]["conditional_downside_probability"] = probability_metrics(dev.loc[mask], probabilities[name].loc[mask])
        gains.extend(yearly[year]["candidate"][metric] - yearly[year]["incumbent"][metric]
                     for metric in ("accuracy", "balanced_accuracy"))
    changes = int(results["candidate"].loc[dev.trade_date.ge(20230101), "correction_selected"].sum())
    eligible = min(gains) >= -1e-12 and changes >= 5
    write_json(args.output / "development.json", {"years": yearly, "eligible": eligible, "changes": changes, "minimum_gain": min(gains)})
    # An independently truncated run must reproduce the features and predictions.
    prefix = dev.loc[dev.trade_date.le(20240628)].reset_index(drop=True)
    _, prefix_features, prefix_available = inputs(prefix)
    prefix_unknown = prefix.copy()
    prefix_unknown.loc[prefix_unknown.index[-1], "real_pct_change"] = np.nan
    p = predict_probabilities(prefix_unknown, prefix_features, prefix_available, allow_trend=True,
                              attempts_name="prefix_inner_selections.json")
    pd.testing.assert_frame_equal(p, probabilities["candidate"].iloc[:len(prefix)], check_exact=True)
    pd.testing.assert_frame_equal(prefix_features, extended.iloc[:len(prefix)], check_exact=True)
    write_json(args.output / "prefix_verification.json", {"end_signal_date": int(prefix.trade_date.iloc[-1]), "rows": len(prefix), "last_outcome_masked": True, "feature_and_probability_parity": True})
    if not eligible:
        report = {"passed": False, "stage": "development", "reason": "The declared trend candidate failed development eligibility; regression not evaluated.", "production_changed": False}
    else:
        _, values, mask = inputs(champion)
        p = predict_probabilities(champion, values, mask, allow_trend=True,
                                  attempts_name="full_inner_selections.json")
        result = validate_frame(specialist_predictions(core, champion, p, THRESHOLD))
        pd.testing.assert_frame_equal(p.iloc[:len(dev)], probabilities["candidate"], check_exact=True)
        pd.testing.assert_frame_equal(result.iloc[:len(dev)], results["candidate"], check_exact=True)
        result.to_csv(args.output / "candidate_predictions.csv", index=False, float_format="%.17g")
        monthly = {month: {"incumbent": metrics(champion.loc[group.index]), "candidate": metrics(group)}
                   for month, group in result.groupby(result.trade_date.astype(str).str[:6])}
        write_json(args.output / "monthly.json", monthly)
        report = {"candidate": "nested_trend_breadth_downside_logistic" if args.nested_selection else "trend_breadth_downside_logistic", "threshold": THRESHOLD,
                  **compare_frames(champion, result), "independent_future_validation": False}
    write_json(args.output / "report.json", report)
    print(json.dumps({"development_eligible": eligible, "changes": changes, "minimum_gain": min(gains),
                      "passed": report["passed"], "stage": report.get("stage", "regression"),
                      "failed_checks": [c for c in report.get("checks", []) if not c["passed"]]}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
