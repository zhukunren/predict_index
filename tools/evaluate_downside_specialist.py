"""Freeze conditional downside models; select only on 2023 and 2024."""

from __future__ import annotations

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
from tools.downside_specialist import (
    BREADTH_COLUMNS, CANDIDATES, COMMON, GLOBAL_COLUMNS, PRICE_COLUMNS, TRAINING_POLICIES, MODEL_FAMILIES,
    downside_probabilities, mean_downside_probabilities, specialist_features, specialist_predictions,
)
from tools.evaluate_direction_bias import compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame
from tools.evaluate_selective_context import THRESHOLDS
from tools.global_risk_features import ASSETS
from tools.futures_features import FEATURE_COLUMNS as FUTURES_COLUMNS, futures_features
from tools.liquidity_features import FEATURE_COLUMNS as LIQUIDITY_COLUMNS, REQUESTS as LIQUIDITY_ASSETS, liquidity_features
from tools.distribution_features import FEATURE_COLUMNS as DISTRIBUTION_COLUMNS, distribution_features
from tools.moneyflow_features import FEATURE_COLUMNS as MONEYFLOW_COLUMNS, PRICE_FEATURE_COLUMNS as MONEYFLOW_PRICE_COLUMNS, NORMALIZATION as MONEYFLOW_NORMALIZATION, moneyflow_features
from tools.etf_features import ASSETS as ETF_ASSETS, FEATURE_COLUMNS as ETF_COLUMNS, LAG_SESSIONS as ETF_LAG, etf_features
from tools.capital_features import FEATURE_COLUMNS as CAPITAL_COLUMNS, LAG_SESSIONS as CAPITAL_LAG, capital_features
from tools.capital_close_features import capital_close_features
from tools.option_position_features import FEATURE_COLUMNS as OPTION_POSITION_COLUMNS, NEAR_EXPIRY_DAYS, position_features


def freeze_inputs(args):
    files = {"features.csv": args.input, "baseline.csv": args.baseline,
             "breadth.csv": args.breadth / "breadth.csv",
             "breadth_manifest.json": args.breadth / "manifest.json",
             "global_manifest.json": args.global_context / "manifest.json"}
    files.update({f"{name}.csv": args.global_context / f"{name}.csv" for name in ASSETS})
    if args.futures_context is not None:
        files.update({"futures.csv": args.futures_context / "futures.csv",
                      "futures_manifest.json": args.futures_context / "manifest.json"})
    if args.option_context is not None:
        files.update({"options.csv": args.option_context / "options.csv",
                      "option_manifest.json": args.option_context / "manifest.json"})
    if args.liquidity_context is not None:
        files["liquidity_manifest.json"] = args.liquidity_context / "manifest.json"
        files.update({f"liquidity_{name}.csv": args.liquidity_context / f"{name}.csv" for name in LIQUIDITY_ASSETS})
    if args.distribution_context is not None:
        files.update({"distribution.csv": args.distribution_context / "distribution.csv",
                      "distribution_manifest.json": args.distribution_context / "manifest.json"})
    if args.moneyflow_context is not None:
        files.update({"moneyflow.csv": args.moneyflow_context / "moneyflow.csv",
                      "moneyflow_manifest.json": args.moneyflow_context / "manifest.json"})
    if args.etf_context is not None:
        files["etf_manifest.json"] = args.etf_context / "manifest.json"
        files.update({f"etf_{name}.csv": args.etf_context / f"{name}.csv" for name in ETF_ASSETS})
    if args.capital_context is not None:
        files.update({"capital.csv": args.capital_context / "capital.csv",
                      "capital_manifest.json": args.capital_context / "manifest.json"})
        if args.capital_close_prices:
            files.update({"capital_prices.csv": args.capital_context / "prices.csv",
                          "capital_verification.json": args.capital_context / "verification.json"})
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    breadth_manifest = json.loads(files["breadth_manifest.json"].read_text(encoding="utf-8"))
    global_manifest = json.loads(files["global_manifest.json"].read_text(encoding="utf-8"))
    if (breadth_manifest["status"] != "complete"
            or breadth_manifest["contract"]["input_sha256"] != hashes["features.csv"]
            or breadth_manifest["contract"]["baseline_sha256"] != hashes["baseline.csv"]
            or breadth_manifest["breadth_sha256"] != hashes["breadth.csv"]
            or global_manifest["baseline_sha256"] != hashes["baseline.csv"]):
        raise ValueError("Specialist inputs must match the frozen baseline snapshot.")
    for name in ASSETS:
        if hashes[f"{name}.csv"] != global_manifest["assets"][name]["sha256"]:
            raise ValueError(f"Overseas input hash mismatch: {name}.")
    if args.futures_context is not None:
        manifest = json.loads(files["futures_manifest.json"].read_text(encoding="utf-8"))
        if (manifest["status"] != "complete" or manifest["futures_sha256"] != hashes["futures.csv"]
                or manifest["contract"]["input_sha256"] != hashes["features.csv"]
                or manifest["contract"]["baseline_sha256"] != hashes["baseline.csv"]):
            raise ValueError("Futures context must match the frozen baseline snapshot.")
    if args.option_context is not None:
        manifest = json.loads(files["option_manifest.json"].read_text(encoding="utf-8"))
        if (manifest["status"] != "complete" or manifest["options_sha256"] != hashes["options.csv"]
                or manifest["contract"]["input_sha256"] != hashes["features.csv"]
                or manifest["contract"]["baseline_sha256"] != hashes["baseline.csv"]):
            raise ValueError("Option context must match the frozen baseline snapshot.")
        if manifest["contract"].get("position_context", False) != args.option_position_features:
            raise ValueError("Option position feature selection must match its frozen dataset.")
        if args.option_position_features and manifest["contract"]["near_expiry_calendar_days"] != NEAR_EXPIRY_DAYS:
            raise ValueError("Option expiry buckets must match the frozen dataset.")
        if manifest["contract"].get("lag_sessions", 1) != args.option_lag_sessions:
            raise ValueError("Option timing must match the frozen dataset.")
    if args.liquidity_context is not None:
        manifest = json.loads(files["liquidity_manifest.json"].read_text(encoding="utf-8"))
        if (manifest["status"] != "complete"
                or manifest["contract"]["input_sha256"] != hashes["features.csv"]
                or manifest["contract"]["baseline_sha256"] != hashes["baseline.csv"]
                or any(manifest["assets"][name]["sha256"] != hashes[f"liquidity_{name}.csv"] for name in LIQUIDITY_ASSETS)):
            raise ValueError("Liquidity context must match the frozen baseline snapshot.")
    if args.distribution_context is not None:
        manifest = json.loads(files["distribution_manifest.json"].read_text(encoding="utf-8"))
        if (manifest["status"] != "complete"
                or manifest["contract"]["input_sha256"] != hashes["features.csv"]
                or manifest["contract"]["baseline_sha256"] != hashes["baseline.csv"]
                or manifest["distribution_sha256"] != hashes["distribution.csv"]):
            raise ValueError("Distribution context must match the frozen baseline snapshot.")
    if args.moneyflow_context is not None:
        manifest = json.loads(files["moneyflow_manifest.json"].read_text(encoding="utf-8"))
        if (manifest["status"] != "complete"
                or manifest["contract"]["input_sha256"] != hashes["features.csv"]
                or manifest["contract"]["baseline_sha256"] != hashes["baseline.csv"]
                or manifest["moneyflow_sha256"] != hashes["moneyflow.csv"]):
            raise ValueError("Moneyflow context must match the frozen baseline snapshot.")
        if manifest["contract"].get("price_context", False) != args.moneyflow_price_features:
            raise ValueError("Moneyflow price feature selection must match its frozen dataset.")
    if args.etf_context is not None:
        manifest = json.loads(files["etf_manifest.json"].read_text(encoding="utf-8"))
        if (manifest["status"] != "complete"
                or manifest["contract"]["input_sha256"] != hashes["features.csv"]
                or manifest["contract"]["baseline_sha256"] != hashes["baseline.csv"]
                or manifest["contract"]["assets"] != ETF_ASSETS
                or manifest["contract"]["lag_sessions"] != ETF_LAG
                or any(manifest["assets"][name]["sha256"] != hashes[f"etf_{name}.csv"] for name in ETF_ASSETS)):
            raise ValueError("ETF context must match the frozen baseline snapshot.")
    if args.capital_context is not None:
        manifest = json.loads(files["capital_manifest.json"].read_text(encoding="utf-8"))
        if (manifest["status"] != "complete"
                or manifest["contract"]["input_sha256"] != hashes["features.csv"]
                or manifest["contract"]["baseline_sha256"] != hashes["baseline.csv"]
                or manifest["capital_sha256"] != hashes["capital.csv"]
                or manifest["contract"]["lag_sessions"] != CAPITAL_LAG
                or manifest["contract"].get("price_lag_sessions", 1) != (0 if args.capital_close_prices else 1)
                or manifest["contract"]["features"] != list(CAPITAL_COLUMNS)):
            raise ValueError("Capital context must match the frozen baseline and timing contract.")
        if args.capital_close_prices:
            audit = json.loads(files["capital_verification.json"].read_text(encoding="utf-8"))
            if (manifest["prices_sha256"] != hashes["capital_prices.csv"] or not audit["passed"]
                    or audit["context_manifest_sha256"] != hashes["capital_manifest.json"]
                    or manifest["contract"]["publication_hour"] != args.publication_hour):
                raise ValueError("Same-day capital prices require matching audited inputs and publication time.")
        for name, expected in manifest["contract"]["source_sha256"].items():
            if (hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
                    or hashlib.sha256((args.capital_context / "source" / name).read_bytes()).hexdigest() != expected):
                raise ValueError("Capital feature sources changed after data acquisition.")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, path in files.items():
        (args.output / name).write_bytes(path.read_bytes())
    sources = ["\u5faa\u73af\u9a8c\u8bc1\u811a\u672c.py", "return_calibration.py", "regularized_direction.py",
               "tushare_prediction_pipeline.py", "\u6570\u636e\u62c9\u53d6\u811a\u672c_tushare.py",
               "tools/downside_specialist.py", "tools/evaluate_downside_specialist.py",
               "tools/evaluate_selective_context.py", "tools/context_residual_model.py",
               "tools/evaluate_direction_bias.py", "tools/direction_rule_candidates.py",
               "tools/evaluate_prediction_candidate.py", "tools/breadth_features.py", "tools/global_risk_features.py"]
    if args.futures_context is not None:
        sources.extend(["tools/futures_features.py", "tools/fetch_futures_context.py"])
    if args.option_context is not None:
        sources.extend(["tools/option_features.py", "tools/fetch_option_context.py"])
    if args.option_position_features:
        sources.extend(["tools/option_position_features.py", "tools/build_option_positions.py", "tools/liquidity_features.py"])
    if args.calibrate_downside:
        sources.append("tools/downside_probability_calibration.py")
    if args.adaptive_context_ensemble:
        sources.append("tools/downside_ensemble.py")
    if args.liquidity_context is not None:
        sources.extend(["tools/liquidity_features.py", "tools/fetch_liquidity_context.py"])
    if args.distribution_context is not None:
        sources.extend(["tools/distribution_features.py", "tools/build_distribution_context.py", "tools/liquidity_features.py"])
    if args.moneyflow_context is not None:
        sources.extend(["tools/moneyflow_features.py", "tools/fetch_moneyflow_context.py", "tools/liquidity_features.py"])
    if args.etf_context is not None:
        sources.extend(["tools/etf_features.py", "tools/fetch_etf_context.py", "tools/liquidity_features.py"])
    if args.capital_context is not None:
        sources.extend(["tools/capital_features.py", "tools/fetch_capital_context.py", "tools/moneyflow_features.py", "tools/liquidity_features.py"])
        if args.capital_close_prices:
            sources.extend(["tools/capital_close_features.py", "tools/build_capital_close_context.py"])
    if args.bidirectional:
        sources.append("tools/directional_specialist.py")
    if args.magnitude_weighted:
        sources.append("tools/magnitude_downside.py")
    source_hashes = {}
    for name in sources:
        raw = (ROOT / name).read_bytes()
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return hashes, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "baseline", "breadth", "global-context", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--futures-context", type=Path)
    parser.add_argument("--option-context", type=Path)
    parser.add_argument("--option-position-features", action="store_true")
    parser.add_argument("--option-lag-sessions", type=int, choices=(0, 1), default=1)
    parser.add_argument("--liquidity-context", type=Path)
    parser.add_argument("--distribution-context", type=Path)
    parser.add_argument("--moneyflow-context", type=Path)
    parser.add_argument("--etf-context", type=Path)
    parser.add_argument("--capital-context", type=Path)
    parser.add_argument("--capital-close-prices", action="store_true")
    parser.add_argument("--capital-moneyflow-ensemble", action="store_true")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--magnitude-weighted", action="store_true")
    parser.add_argument("--moneyflow-price-features", action="store_true")
    parser.add_argument("--moneyflow-price-ensemble", action="store_true")
    parser.add_argument("--option-moneyflow-ensemble", action="store_true")
    parser.add_argument("--adaptive-context-ensemble", action="store_true")
    parser.add_argument("--normalize-moneyflow", action="store_true")
    parser.add_argument("--futures-lag-sessions", type=int, choices=(0, 1), default=1)
    parser.add_argument("--publication-hour", type=int, default=18)
    parser.add_argument("--calibrate-downside", action="store_true")
    parser.add_argument("--training-policy", choices=tuple(TRAINING_POLICIES), default="uniform")
    parser.add_argument("--model-family", choices=tuple(MODEL_FAMILIES), default="classic")
    args = parser.parse_args()
    magnitude_policy = None
    if args.magnitude_weighted:
        if (args.bidirectional or args.calibrate_downside or args.moneyflow_price_ensemble or args.option_moneyflow_ensemble
                or args.capital_moneyflow_ensemble or args.model_family != "classic" or args.training_policy != "uniform"):
            parser.error("magnitude weighting requires a standalone classic model with uniform temporal weights")
        from tools.magnitude_downside import POLICY as magnitude_policy, magnitude_scores
    if args.bidirectional and (args.calibrate_downside or args.capital_moneyflow_ensemble or args.moneyflow_price_ensemble
                               or args.option_moneyflow_ensemble):
        parser.error("bidirectional models cannot use downside-only calibration or probability ensembles")
    make_predictions = specialist_predictions
    if args.bidirectional:
        from tools.directional_specialist import directional_probabilities, directional_predictions
        make_predictions = directional_predictions
    if args.capital_close_prices and (args.capital_context is None or not 18 <= args.publication_hour <= 23):
        parser.error("--capital-close-prices requires --capital-context and publication at 18:00 or later")
    if args.capital_moneyflow_ensemble and (not args.capital_close_prices or not args.moneyflow_price_ensemble or args.option_moneyflow_ensemble):
        parser.error("--capital-moneyflow-ensemble requires current capital prices and the moneyflow-price ensemble only")
    if args.moneyflow_price_features and args.moneyflow_context is None:
        parser.error("--moneyflow-price-features requires --moneyflow-context")
    if args.moneyflow_price_ensemble and not args.moneyflow_price_features:
        parser.error("--moneyflow-price-ensemble requires --moneyflow-price-features")
    if args.option_moneyflow_ensemble and (args.option_context is None or args.moneyflow_context is None or args.moneyflow_price_ensemble):
        parser.error("--option-moneyflow-ensemble requires both contexts and cannot combine with --moneyflow-price-ensemble")
    if args.adaptive_context_ensemble and not args.option_moneyflow_ensemble:
        parser.error("--adaptive-context-ensemble requires --option-moneyflow-ensemble")
    if args.normalize_moneyflow and args.moneyflow_context is None:
        parser.error("--normalize-moneyflow requires --moneyflow-context")
    if args.option_position_features and args.option_context is None:
        parser.error("--option-position-features requires --option-context")
    if args.option_lag_sessions == 0 and (args.option_context is None or args.publication_hour < 20):
        parser.error("same-day options require --option-context and --publication-hour 20 or later")
    option_columns = ()
    if args.option_context is not None:
        from tools.option_features import FEATURE_COLUMNS as option_columns, option_features
    calibration_parameters = None
    if args.calibrate_downside:
        from tools.downside_probability_calibration import PARAMETERS as calibration_parameters, calibrate_downside_probabilities
    ensemble_parameters = None
    if args.adaptive_context_ensemble:
        from tools.downside_ensemble import PARAMETERS as ensemble_parameters, adaptive_downside_probabilities
    hashes, sources = freeze_inputs(args)
    candidates = {name: CANDIDATES[name] for name in MODEL_FAMILIES[args.model_family]}
    config, _ = pipeline._default_calculation_options()
    core = pipeline.prediction_core
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "inputs": hashes, "sources": sources,
        "config": asdict(config), "common": COMMON, "candidates": candidates, "thresholds": THRESHOLDS,
        "model_family": args.model_family,
        "training_policy": {"name": args.training_policy, **TRAINING_POLICIES[args.training_policy],
                            "weight_normalization": "mean one over eligible prior training samples"},
        "features": ["confidence", "calibrated_confidence", *PRICE_COLUMNS, *BREADTH_COLUMNS, *GLOBAL_COLUMNS,
                     *(FUTURES_COLUMNS if args.futures_context is not None else ()), *option_columns,
                     *(OPTION_POSITION_COLUMNS if args.option_position_features else ()),
                     *(LIQUIDITY_COLUMNS if args.liquidity_context is not None else ()),
                     *(DISTRIBUTION_COLUMNS if args.distribution_context is not None else ()),
                     *(ETF_COLUMNS if args.etf_context is not None else ()),
                     *(CAPITAL_COLUMNS if args.capital_context is not None else ()),
                     *(MONEYFLOW_COLUMNS if args.moneyflow_context is not None else ()),
                     *(MONEYFLOW_PRICE_COLUMNS if args.moneyflow_price_features else ())],
        "target": ("downside share of expected absolute return conditional on incumbent up; not event probability" if args.magnitude_weighted else
                   "P(incumbent error | incumbent direction); separate up/down models, shared threshold; zero return is down"
                   if args.bidirectional else "P(next return <= 0 | incumbent up); train only on prior resolved incumbent up signals"),
        "bidirectional": args.bidirectional,
        "magnitude_weighting": magnitude_policy,
        "candidate_attempts": len(candidates) * len(THRESHOLDS), "selection_years": [2023, 2024],
        "selection": "nondecreasing yearly accuracy and balanced accuracy, at least 5 changes; minimum yearly gain, balanced accuracy, fewer changes",
        "regression_data_previously_observed": True, "automatic_promotion": False,
        "breadth_policy": "same-day close, publication at 18:00 Shanghai; no historical delivery timestamps",
        "global_policy": "US session date strictly before domestic signal date",
        "option_policy": ({"lag_sessions": args.option_lag_sessions, "publication_hour": args.publication_hour,
                           "delivery_caveat": "research timing assumption; original delivery timestamps unavailable"}
                          if args.option_context is not None else None),
        "option_position_features": ({"near_expiry_calendar_days": NEAR_EXPIRY_DAYS,
                                      "changes": "matched adjacent-session contracts only; incomplete features excluded"}
                                     if args.option_position_features else None),
        "liquidity_policy": ("strictly prior domestic session; incomplete features excluded from training; "
                             "incumbent direction retained when unavailable; no original delivery timestamps") if args.liquidity_context is not None else None,
        "distribution_policy": "same-day completed daily data; publication at 18:00 Shanghai" if args.distribution_context is not None else None,
        "etf_policy": ({"lag_sessions": ETF_LAG, "missing": "exclude from training and corrections; never fill",
                        "interpretation": "share-count changes, not cash flows", "original_delivery_timestamps": False}
                       if args.etf_context is not None else None),
        "capital_policy": ({"lag_sessions": CAPITAL_LAG, "price_lag_sessions": 0 if args.capital_close_prices else CAPITAL_LAG,
                            "publication_hour": args.publication_hour,
                            "universe": "historical active Shanghai A-shares, not exact index constituents",
                            "weight": "source-day total market cap", "large_cap_group": "top 10% of source-day capitalization",
                            "missing": "all active stocks require positive cap; matched flow coverage >=98% by count, turnover and cap"}
                           if args.capital_context is not None else None),
        "moneyflow_policy": ("strictly previous domestic session; stock and turnover coverage at least 98%; "
                             "incomplete features excluded from training and corrections") if args.moneyflow_context is not None else None,
        "moneyflow_price_features": args.moneyflow_price_features,
        "moneyflow_normalization": ({**MONEYFLOW_NORMALIZATION,
                                    "policy": "normalize each flow feature against strictly earlier source sessions; no missing fill"}
                                   if args.normalize_moneyflow else None),
        "moneyflow_price_ensemble": ("equal mean of base and price-extended downside probabilities; "
                                     "both models must be trained; same availability mask") if args.moneyflow_price_ensemble else None,
        "capital_moneyflow_ensemble": ("equal mean of capital-close probabilities and the fixed equal moneyflow-price mixture; "
                                        "three models share available dates; no weight search") if args.capital_moneyflow_ensemble else None,
        "option_moneyflow_ensemble": ({"pooling": "causal_brier_weights" if args.adaptive_context_ensemble else "equal_mean",
                                       "policy": "separate option and moneyflow models; shared base features and availability mask; both must be trained",
                                       "adaptive_parameters": ensemble_parameters} if args.option_moneyflow_ensemble else None),
        "futures_policy": ({"lag_sessions": args.futures_lag_sessions, "publication_hour": args.publication_hour,
                            "delivery_caveat": "historical daily data lack original delivery and revision timestamps"}
                           if args.futures_context is not None else None),
        "gate": "unchanged evaluate_direction_bias.compare_frames",
        "calibration": "existing causal rolling calibration for the combined forecast stream",
        "downside_probability_calibration": calibration_parameters,
        "xgboost_version": version("xgboost"), "sklearn_version": version("scikit-learn"),
        "scipy_version": version("scipy") if args.adaptive_context_ensemble else None,
    })
    champion = validate_frame(pd.read_csv(args.output / "baseline.csv", float_precision="round_trip"))
    market = pd.read_csv(args.output / "features.csv", float_precision="round_trip")
    breadth = pd.read_csv(args.output / "breadth.csv", float_precision="round_trip")
    overseas = {name: pd.read_csv(args.output / f"{name}.csv", float_precision="round_trip") for name in ASSETS}
    futures = pd.read_csv(args.output / "futures.csv", float_precision="round_trip") if args.futures_context is not None else None
    options = pd.read_csv(args.output / "options.csv", float_precision="round_trip") if args.option_context is not None else None
    liquidity = ({name: pd.read_csv(args.output / f"liquidity_{name}.csv", float_precision="round_trip") for name in LIQUIDITY_ASSETS}
                 if args.liquidity_context is not None else None)
    distribution = (pd.read_csv(args.output / "distribution.csv", float_precision="round_trip")
                    if args.distribution_context is not None else None)
    moneyflow = (pd.read_csv(args.output / "moneyflow.csv", float_precision="round_trip")
                 if args.moneyflow_context is not None else None)
    etf = ({name: pd.read_csv(args.output / f"etf_{name}.csv", float_precision="round_trip") for name in ETF_ASSETS}
           if args.etf_context is not None else None)
    capital = (pd.read_csv(args.output / "capital.csv", float_precision="round_trip")
               if args.capital_context is not None else None)
    capital_prices = (pd.read_csv(args.output / "capital_prices.csv", float_precision="round_trip")
                      if args.capital_close_prices else None)
    def build_features(market_frame, baseline_frame):
        features = specialist_features(core, market_frame, baseline_frame, config, breadth, overseas)
        available = np.ones(len(features), dtype=bool)
        calendar = pd.to_datetime(market_frame.trade_date).dt.strftime("%Y%m%d").astype(int)
        if futures is not None:
            extra = futures_features(baseline_frame.trade_date, futures, calendar,
                                     lag_sessions=args.futures_lag_sessions, publication_hour=args.publication_hour)
            features = pd.concat([features, extra.drop(columns=["trade_date", "futures_source_date"])], axis=1)
        if options is not None:
            extra = option_features(baseline_frame.trade_date, options, calendar,
                                    lag_sessions=args.option_lag_sessions, publication_hour=args.publication_hour)
            features = pd.concat([features, extra.drop(columns=["trade_date", "option_source_date"])], axis=1)
            if args.option_position_features:
                positions = position_features(baseline_frame.trade_date, options, calendar,
                                              lag_sessions=args.option_lag_sessions, publication_hour=args.publication_hour)
                available &= positions.option_position_available.to_numpy()
                features = pd.concat([features, positions.loc[:, OPTION_POSITION_COLUMNS]], axis=1)
        if liquidity is not None:
            extra = liquidity_features(baseline_frame.trade_date, liquidity, calendar, allow_missing=True)
            available &= extra.liquidity_available.to_numpy()
            features = pd.concat([features, extra.loc[:, LIQUIDITY_COLUMNS]], axis=1)
        if distribution is not None:
            # Truncate source rows to the requested market prefix before validation.
            daily = distribution.loc[distribution.trade_date.le(int(calendar.iloc[-1]))]
            extra = distribution_features(baseline_frame.trade_date, daily, calendar)
            features = pd.concat([features, extra.loc[:, DISTRIBUTION_COLUMNS]], axis=1)
        if etf is not None:
            extra = etf_features(baseline_frame.trade_date, etf, calendar)
            available &= extra.etf_available.to_numpy()
            features = pd.concat([features, extra.loc[:, ETF_COLUMNS]], axis=1)
        if moneyflow is not None:
            extra = moneyflow_features(baseline_frame.trade_date, moneyflow, calendar,
                                       price_context=args.moneyflow_price_features, normalize=args.normalize_moneyflow)
            available &= extra.moneyflow_available.to_numpy()
            columns = (*MONEYFLOW_COLUMNS, *(MONEYFLOW_PRICE_COLUMNS if args.moneyflow_price_features else ()))
            features = pd.concat([features, extra.loc[:, columns]], axis=1)
        if capital is not None:
            extra = (capital_features(baseline_frame.trade_date, capital, calendar) if capital_prices is None else
                     capital_close_features(baseline_frame.trade_date, capital, capital_prices, calendar,
                                            publication_hour=args.publication_hour))
            available &= extra.capital_available.to_numpy()
            features = pd.concat([features, extra.loc[:, CAPITAL_COLUMNS]], axis=1)
        return features, available
    def predict_probabilities(baseline_frame, features, available, name):
        if args.magnitude_weighted:
            return magnitude_scores(baseline_frame, features, name, available=available)
        if args.bidirectional:
            return directional_probabilities(baseline_frame, features, name, available=available,
                                             training_policy=args.training_policy)
        if args.capital_moneyflow_ensemble:
            moneyflow_only = features.drop(columns=list(CAPITAL_COLUMNS))
            capital_only = features.drop(columns=[*MONEYFLOW_COLUMNS, *MONEYFLOW_PRICE_COLUMNS])
            extended = downside_probabilities(baseline_frame, moneyflow_only, name, available=available,
                                               training_policy=args.training_policy)
            base = downside_probabilities(baseline_frame, moneyflow_only.drop(columns=list(MONEYFLOW_PRICE_COLUMNS)), name,
                                           available=available, training_policy=args.training_policy)
            capital_probability = downside_probabilities(baseline_frame, capital_only, name, available=available,
                                                         training_policy=args.training_policy)
            probabilities = mean_downside_probabilities(mean_downside_probabilities(base, extended), capital_probability)
            probabilities = probabilities.rename(columns={"downside_base_probability": "downside_moneyflow_mixture_probability",
                                                           "downside_extended_probability": "downside_capital_probability"})
        elif args.option_moneyflow_ensemble:
            moneyflow_features_only = features.drop(columns=[*option_columns, *(OPTION_POSITION_COLUMNS if args.option_position_features else ())])
            option_features_only = features.drop(columns=[*MONEYFLOW_COLUMNS, *(MONEYFLOW_PRICE_COLUMNS if args.moneyflow_price_features else ())])
            moneyflow_probabilities = downside_probabilities(baseline_frame, moneyflow_features_only, name,
                                                            available=available, training_policy=args.training_policy)
            option_probabilities = downside_probabilities(baseline_frame, option_features_only, name,
                                                         available=available, training_policy=args.training_policy)
            if args.adaptive_context_ensemble:
                probabilities = adaptive_downside_probabilities(baseline_frame, moneyflow_probabilities, option_probabilities)
            else:
                probabilities = mean_downside_probabilities(moneyflow_probabilities, option_probabilities)
            probabilities = probabilities.rename(columns={
                "downside_base_probability": "downside_moneyflow_probability",
                "downside_extended_probability": "downside_option_probability",
                "downside_left_weight": "downside_moneyflow_weight",
            })
        else:
            probabilities = downside_probabilities(baseline_frame, features, name, available=available,
                                                   training_policy=args.training_policy)
            if args.moneyflow_price_ensemble:
                base_features = features.drop(columns=list(MONEYFLOW_PRICE_COLUMNS))
                base_probabilities = downside_probabilities(baseline_frame, base_features, name, available=available,
                                                           training_policy=args.training_policy)
                probabilities = mean_downside_probabilities(base_probabilities, probabilities)
        if args.calibrate_downside:
            probabilities = calibrate_downside_probabilities(baseline_frame, probabilities)
        return probabilities
    base = core._normalize_market_frame(market, config)
    actual = base.close.shift(-1) / base.close - 1
    actual.index = base.date.dt.strftime("%Y%m%d").astype(int)
    if not np.array_equal(champion.real_pct_change, actual.reindex(champion.trade_date)):
        raise ValueError("Frozen outcomes disagree with market prices.")
    development = champion.loc[champion.trade_date.le(20241231)].reset_index(drop=True)
    development_market = market.loc[pd.to_datetime(market.trade_date).lt("2025-01-01")]
    features, available = build_features(development_market, development)
    yearly_baseline = {year: metrics(development.loc[development.trade_date.between(year * 10000, year * 10000 + 1231)])
                       for year in (2023, 2024)}
    directory = args.output / "development"
    directory.mkdir()
    outcomes = []
    for name in candidates:
        probabilities = predict_probabilities(development, features, available, name)
        probabilities.to_csv(directory / f"{name}_probabilities.csv", index=False, float_format="%.17g")
        for threshold in THRESHOLDS:
            result = make_predictions(core, development, probabilities, threshold)
            filename = f"{name}_p{round(threshold * 100)}.csv"
            result.to_csv(directory / filename, index=False, float_format="%.17g")
            yearly = {year: metrics(result.loc[result.trade_date.between(year * 10000, year * 10000 + 1231)])
                      for year in (2023, 2024)}
            gains = [yearly[year][metric] - yearly_baseline[year][metric] for year in (2023, 2024)
                     for metric in ("accuracy", "balanced_accuracy")]
            rows = result.loc[result.trade_date.ge(20230101)]
            changes = int(rows.correction_selected.sum())
            outcomes.append({"candidate": name, "threshold": threshold, "prediction_file": filename,
                             "yearly": yearly, "combined": metrics(rows), "minimum_gain": min(gains),
                             "changed_rows": changes, "eligible": min(gains) >= -1e-12 and changes >= 5})
        print(json.dumps({"candidate": name, "eligible_thresholds": sum(item["eligible"] for item in outcomes if item["candidate"] == name)}), flush=True)
    eligible = [item for item in outcomes if item["eligible"]]
    selected = max(eligible, key=lambda item: (item["minimum_gain"], item["combined"]["balanced_accuracy"], -item["changed_rows"])) if eligible else None
    write_json(args.output / "selection.json", {"selected": selected, "outcomes": outcomes, "baseline_years": yearly_baseline})
    if selected is None:
        report = {"passed": False, "stage": "development", "reason": "No downside specialist passed both development years.", "production_changed": False}
    else:
        features, available = build_features(market, champion)
        probabilities = predict_probabilities(champion, features, available, selected["candidate"])
        result = make_predictions(core, champion, probabilities, selected["threshold"])
        prefix = pd.read_csv(directory / selected["prediction_file"], float_precision="round_trip")
        columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence", *probabilities.columns.drop("trade_date")]
        pd.testing.assert_frame_equal(prefix[columns], result.iloc[:len(prefix)][columns], check_exact=True)
        result.to_csv(args.output / "candidate_predictions.csv", index=False, float_format="%.17g")
        report = {"candidate": selected["candidate"], "threshold": selected["threshold"], **compare_frames(champion, result)}
    write_json(args.output / "report.json", report)
    print(json.dumps({"selected": None if selected is None else (selected["candidate"], selected["threshold"]),
                      "passed": report["passed"], "failed_checks": [item for item in report.get("checks", []) if not item["passed"]]}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
