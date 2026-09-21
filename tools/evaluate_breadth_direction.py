"""Evaluate fixed breadth models and thresholds using development years only."""

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
from tools.breadth_features import breadth_features
from tools.context_residual_model import ASSETS, BREADTH_CANDIDATES, COMMON, context_features, load_assets, residual_predictions
from tools.evaluate_direction_bias import compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame
from tools.evaluate_selective_context import THRESHOLDS, selective_predictions


def add_breadth(core, market, champion, config, assets, breadth, calendar, *, lag_sessions=1, publication_hour=18):
    base, features = context_features(core, market, config, assets)
    dates = base.date.dt.strftime("%Y%m%d").astype(int)
    signal = dates.isin(champion.trade_date)
    extra = breadth_features(dates.loc[signal], breadth, calendar.loc[calendar.le(int(dates.iloc[-1]))],
                             lag_sessions=lag_sessions, publication_hour=publication_hour)
    extra = extra.drop(columns="breadth_source_date")
    extra.index = features.index[signal]
    return base, pd.concat([features, extra], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--breadth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lag-sessions", type=int, choices=(0, 1), default=1)
    parser.add_argument("--publication-hour", type=int, default=18)
    args = parser.parse_args()
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    baseline_hash = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    breadth_manifest = json.loads((args.breadth / "manifest.json").read_text(encoding="utf-8"))
    context_manifest = json.loads((args.context / "manifest.json").read_text(encoding="utf-8"))
    if breadth_manifest["status"] != "complete":
        raise ValueError("Breadth acquisition must complete before model evaluation.")
    if (breadth_manifest["contract"]["input_sha256"] != input_hash
            or breadth_manifest["contract"]["baseline_sha256"] != baseline_hash
            or context_manifest["input_sha256"] != input_hash):
        raise ValueError("Breadth and index inputs must match the frozen baseline snapshot.")
    breadth_path = args.breadth / "breadth.csv"
    if hashlib.sha256(breadth_path.read_bytes()).hexdigest() != breadth_manifest["breadth_sha256"]:
        raise ValueError("Breadth aggregate hash mismatch.")
    # Acquisition sources remain frozen with the input. Feature-builder changes
    # are captured separately in this model's source snapshot.
    files = {"features.csv": args.input, "baseline.csv": args.baseline, "breadth.csv": breadth_path,
             "breadth_manifest.json": args.breadth / "manifest.json", "context_manifest.json": args.context / "manifest.json"}
    for name in ASSETS:
        path = args.context / f"{name}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != context_manifest["assets"][name]["sha256"]:
            raise ValueError(f"Index context hash mismatch: {name}.")
        files[path.name] = path
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, path in files.items():
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        (args.output / name).write_bytes(raw)
    sources = ["循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
               "数据拉取脚本_tushare.py", "tools/context_residual_model.py", "tools/breadth_features.py",
               "tools/fetch_market_breadth.py", "tools/evaluate_breadth_direction.py", "tools/evaluate_direction_bias.py",
               "tools/evaluate_prediction_candidate.py", "tools/evaluate_selective_context.py", "tools/direction_rule_candidates.py"]
    source_hashes = {}
    for name in sources:
        raw = (ROOT / name).read_bytes()
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    config, _ = pipeline._default_calculation_options()
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "inputs": hashes, "sources": source_hashes,
        "config": asdict(config), "common": COMMON, "candidates": BREADTH_CANDIDATES, "thresholds": THRESHOLDS,
        "candidate_attempts": len(BREADTH_CANDIDATES) * len(THRESHOLDS), "selection_years": [2023, 2024],
        "selection": "nondecreasing yearly accuracy and balanced accuracy, at least 5 changes; minimum yearly gain, balanced accuracy, fewer changes",
        "regression_data_previously_observed": True, "automatic_promotion": False,
        "breadth_policy": breadth_manifest["contract"]["availability"],
        "lag_sessions": args.lag_sessions, "publication_hour": args.publication_hour,
        "gate": "unchanged evaluate_direction_bias.compare_frames",
        "xgboost_version": version("xgboost"), "sklearn_version": version("scikit-learn"),
    })
    champion = validate_frame(pd.read_csv(args.output / "baseline.csv", float_precision="round_trip"))
    market = pd.read_csv(args.output / "features.csv", float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    breadth = pd.read_csv(args.output / "breadth.csv", float_precision="round_trip")
    assets = load_assets(args.output)
    core = pipeline.prediction_core
    market_base = core._normalize_market_frame(market, config)
    returns = market_base.close.shift(-1) / market_base.close - 1
    truth = pd.Series(returns.to_numpy(), index=market_base.date.dt.strftime("%Y%m%d").astype(int)).reindex(champion.trade_date)
    if not np.array_equal(champion.real_pct_change, truth):
        raise ValueError("Frozen incumbent outcomes disagree with the market input.")
    development = champion.loc[champion.trade_date.le(20241231)].reset_index(drop=True)
    yearly_baseline = {year: metrics(development.loc[development.trade_date.between(year * 10000, year * 10000 + 1231)])
                       for year in (2023, 2024)}
    development_market = market.loc[calendar.le(20241231)]
    timing = {"lag_sessions": args.lag_sessions, "publication_hour": args.publication_hour}
    base, features = add_breadth(core, development_market, development, config, assets, breadth, calendar, **timing)
    directory = args.output / "development"
    directory.mkdir()
    outcomes = []
    for name in BREADTH_CANDIDATES:
        proposed = residual_predictions(core, development, base, features, name)
        proposed.to_csv(directory / f"{name}_underlying.csv", index=False, float_format="%.17g")
        for threshold in THRESHOLDS:
            result = selective_predictions(core, development, proposed, threshold)
            filename = f"{name}_p{round(threshold * 100)}.csv"
            result.to_csv(directory / filename, index=False, float_format="%.17g")
            yearly = {year: metrics(result.loc[result.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
            gains = [yearly[year][metric] - yearly_baseline[year][metric] for year in (2023, 2024)
                     for metric in ("accuracy", "balanced_accuracy")]
            rows = result.loc[result.trade_date.ge(20230101)]
            changes = int(rows.correction_selected.sum())
            outcomes.append({"candidate": name, "threshold": threshold, "prediction_file": filename,
                             "yearly": yearly, "combined": metrics(rows), "minimum_gain": min(gains),
                             "changed_rows": changes, "eligible": min(gains) >= -1e-12 and changes >= 5})
        print(json.dumps({"candidate": name, "development_eligible": sum(item["eligible"] for item in outcomes if item["candidate"] == name)}), flush=True)
    eligible = [item for item in outcomes if item["eligible"]]
    selected = max(eligible, key=lambda item: (item["minimum_gain"], item["combined"]["balanced_accuracy"], -item["changed_rows"])) if eligible else None
    write_json(args.output / "selection.json", {"selected": selected, "outcomes": outcomes, "baseline_years": yearly_baseline})
    if selected is None:
        report = {"passed": False, "stage": "development", "reason": "No breadth candidate passed both development years.", "production_changed": False}
    else:
        base, features = add_breadth(core, market, champion, config, assets, breadth, calendar, **timing)
        proposed = residual_predictions(core, champion, base, features, selected["candidate"])
        result = selective_predictions(core, champion, proposed, selected["threshold"])
        prefix = pd.read_csv(directory / selected["prediction_file"], float_precision="round_trip")
        columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence"]
        pd.testing.assert_frame_equal(prefix[columns], result.iloc[:len(prefix)][columns], check_exact=True)
        result.to_csv(args.output / "candidate_predictions.csv", index=False, float_format="%.17g")
        report = {"candidate": selected["candidate"], "threshold": selected["threshold"], **compare_frames(champion, result)}
    write_json(args.output / "report.json", report)
    print(json.dumps({"selected": None if selected is None else (selected["candidate"], selected["threshold"]),
                      "passed": report["passed"], "failed_checks": [item for item in report.get("checks", []) if not item["passed"]]}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
