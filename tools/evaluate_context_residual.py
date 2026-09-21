"""Evaluate frozen context-aware corrections with a development-only choice."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tushare_prediction_pipeline as pipeline
from tools.context_residual_model import ASSETS, CANDIDATES, ERROR_CANDIDATES, FUNDING_CANDIDATES, COMMON, context_features, load_assets, residual_predictions
from tools.evaluate_direction_bias import compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("direction", "correctness", "funding"), default="direction")
    parser.add_argument("--funding", type=Path)
    args = parser.parse_args()
    if args.family == "funding" and args.funding is None:
        parser.error("--family funding requires --funding")
    candidates = {"direction": CANDIDATES, "correctness": ERROR_CANDIDATES, "funding": FUNDING_CANDIDATES}[args.family]
    args.output.mkdir(parents=True, exist_ok=False)
    config, _ = pipeline._default_calculation_options()
    context_manifest = json.loads((args.context / "manifest.json").read_text())
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if input_hash != context_manifest["input_sha256"]:
        raise ValueError("Context was fetched for a different input snapshot.")
    files = {"features.csv": args.input, "baseline.csv": args.baseline}
    for name in ASSETS:
        path = args.context / f"{name}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != context_manifest["assets"][name]["sha256"]:
            raise ValueError(f"Context hash mismatch: {name}.")
        files[path.name] = path
    if args.family == "funding":
        from tools.funding_features import ASSETS as FUNDING_ASSETS
        funding_manifest = json.loads((args.funding / "manifest.json").read_text())
        if funding_manifest["input_sha256"] != input_hash:
            raise ValueError("Funding data were fetched for a different input snapshot.")
        files["funding_manifest.json"] = args.funding / "manifest.json"
        for name in FUNDING_ASSETS:
            path = args.funding / f"{name}.csv"
            if hashlib.sha256(path.read_bytes()).hexdigest() != funding_manifest["assets"][name]["sha256"]:
                raise ValueError(f"Funding hash mismatch: {name}.")
            files[path.name] = path
    hashes = {}
    for name, path in files.items():
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        (args.output / name).write_bytes(raw)
    sources = ["循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
               "数据拉取脚本_tushare.py", "tools/context_residual_model.py", "tools/evaluate_context_residual.py",
               "tools/evaluate_direction_bias.py", "tools/evaluate_prediction_candidate.py", "tools/direction_rule_candidates.py"]
    if args.family == "funding":
        sources += ["tools/funding_features.py", "tools/fetch_direction_funding.py"]
    source_hashes = {}
    for name in sources:
        raw = (ROOT / name).read_bytes()
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
        path = args.output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "inputs": hashes, "sources": source_hashes,
        "context": context_manifest, "config": asdict(config), "common": COMMON, "candidates": candidates,
        "xgboost_version": version("xgboost"), "development": [20230101, 20241231],
        "regression_start": 20250101, "regression_data_previously_observed": True,
        "selection": "development accuracy and balanced accuracy minimum gain, then balanced accuracy, then smaller up-rate gap",
        "information_cutoff": "signal-day domestic close; training labels strictly earlier than prediction signal",
        "confidence_policy": "retain champion raw score on agreement, native probability on changed direction, then fixed causal 300-day calibration",
        "gate": "unchanged evaluate_direction_bias.compare_frames", "automatic_promotion": False,
        "funding_policy": "one domestic trading day lag; missing dates rejected" if args.family == "funding" else None,
        "funding_selection": "nondecreasing accuracy and balanced accuracy separately in 2023 and 2024; at least 5 changes; minimum yearly gain" if args.family == "funding" else None,
    })
    data = pd.read_csv(args.input, float_precision="round_trip")
    champion = validate_frame(pd.read_csv(args.baseline, float_precision="round_trip"))
    assets = load_assets(args.context)
    core = pipeline.prediction_core
    def prepare_features(market):
        base, features = context_features(core, market, config, assets)
        if args.family == "funding":
            from tools.funding_features import funding_features, load_assets as load_funding
            funding = funding_features(base.date, load_funding(args.output))
            features = pd.concat([features, funding.drop(columns="funding_source_date")], axis=1)
        return base, features

    development_mask = pd.to_datetime(data.trade_date) <= "2024-12-31"
    development_data = data.loc[development_mask]
    base, features = prepare_features(development_data)
    development_champion = champion.loc[champion.trade_date <= 20241231].reset_index(drop=True)
    development = args.output / "development"
    development.mkdir()
    results = {"baseline": metrics(development_champion.loc[development_champion.trade_date >= 20230101])}
    baseline_years = {year: metrics(development_champion.loc[development_champion.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
    funding_outcomes = {}
    for name in candidates:
        started = time.monotonic()
        result = residual_predictions(core, development_champion, base, features, name)
        result.to_csv(development / f"{name}.csv", index=False, float_format="%.17g")
        results[name] = metrics(result.loc[result.trade_date >= 20230101])
        if args.family == "funding":
            yearly = {year: metrics(result.loc[result.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
            gains = [yearly[year][metric] - baseline_years[year][metric] for year in (2023, 2024) for metric in ("accuracy", "balanced_accuracy")]
            changes = int(result.loc[result.trade_date >= 20230101, "residual_changed"].sum())
            funding_outcomes[name] = {"yearly": yearly, "minimum_gain": min(gains), "changed_rows": changes, "eligible": min(gains) >= -1e-12 and changes >= 5}
        print(json.dumps({"candidate": name, "seconds": round(time.monotonic() - started, 2), "metrics": results[name]}), flush=True)
    baseline = results["baseline"]
    if args.family == "funding":
        eligible = [name for name in candidates if funding_outcomes[name]["eligible"]]
        selected = max(eligible, key=lambda name: (funding_outcomes[name]["minimum_gain"], results[name]["balanced_accuracy"], -funding_outcomes[name]["changed_rows"])) if eligible else None
        write_json(args.output / "selection.json", {"selected": selected, "metrics": results, "yearly": funding_outcomes, "baseline_years": baseline_years})
        if selected is None:
            report = {"passed": False, "stage": "development", "reason": "No funding candidate passed both development years.", "production_changed": False}
            write_json(args.output / "report.json", report)
            print(json.dumps(report), flush=True)
            return 2
    else:
        selected = max(candidates, key=lambda name: (
            min(results[name]["accuracy"] - baseline["accuracy"], results[name]["balanced_accuracy"] - baseline["balanced_accuracy"]),
            results[name]["balanced_accuracy"], -abs(results[name]["up_rate_gap"])))
        write_json(args.output / "selection.json", {"selected": selected, "metrics": results})
    base, features = prepare_features(data)
    target = base.close.shift(-1) / base.close - 1
    truth = pd.Series(target.to_numpy(), index=base.date.dt.strftime("%Y%m%d").astype(int)).reindex(champion.trade_date)
    import numpy as np
    if not np.array_equal(champion.real_pct_change, truth):
        raise ValueError("Frozen champion outcomes disagree with the current input.")
    result = residual_predictions(core, champion, base, features, selected)
    result.to_csv(args.output / "candidate_predictions.csv", index=False, float_format="%.17g")
    report = {"candidate": selected, **compare_frames(champion, result)}
    write_json(args.output / "report.json", report)
    print(json.dumps({"selected": selected, "passed": report["passed"], "changes": report["paired_direction_changes"],
                      "failed_checks": [item for item in report["checks"] if not item["passed"]]}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
