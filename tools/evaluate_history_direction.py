"""Evaluate additional pre-2020 training cycles against the unchanged incumbent."""

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
from tools.context_residual_model import HISTORY_CANDIDATES, COMMON, context_features, load_assets, residual_predictions
from tools.evaluate_direction_bias import calculate, compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.history / "manifest.json").read_text())
    hashes = {}
    for name in ("features.csv", "csi300.csv", "csi500.csv", "chinext.csv"):
        raw = (args.history / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != manifest["sha256"][name]:
            raise ValueError(f"Historical input hash mismatch: {name}.")
        (args.output / name).write_bytes(raw)
        hashes[name] = digest
    raw = args.baseline.read_bytes()
    (args.output / "baseline.csv").write_bytes(raw)
    hashes["baseline.csv"] = hashlib.sha256(raw).hexdigest()
    source_hashes = {}
    for name in ("循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
                 "数据拉取脚本_tushare.py", "tools/context_residual_model.py", "tools/evaluate_history_direction.py",
                 "tools/evaluate_direction_bias.py", "tools/evaluate_prediction_candidate.py", "tools/direction_rule_candidates.py",
                 "tools/fetch_direction_history.py"):
        raw = (ROOT / name).read_bytes()
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
    config, options = pipeline._default_calculation_options()
    options = {key: None if key.endswith("_path") else value for key, value in options.items()}
    options.update(periods=0, include_latest=False, progress=False)
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "inputs": hashes, "history_manifest": manifest,
        "sources": source_hashes, "config": asdict(config), "baseline_loop_options": options,
        "candidates": HISTORY_CANDIDATES, "common": COMMON, "xgboost_version": version("xgboost"),
        "baseline_policy": "retain every frozen incumbent row; only reconstruct older signals absent from that stream",
        "selection_years": [2023, 2024], "regression_data_previously_observed": True,
        "eligibility": "accuracy and balanced accuracy nondecreasing in both development years; at least 5 changes",
        "selection": "minimum yearly gain, combined balanced accuracy, fewer changes",
        "gate": "unchanged compare_frames", "automatic_promotion": False,
    })
    champion = validate_frame(pd.read_csv(args.output / "baseline.csv", float_precision="round_trip"))
    data = pd.read_csv(args.output / "features.csv", float_precision="round_trip")
    dates = pd.to_datetime(data.trade_date)
    first_frozen_signal = pd.to_datetime(str(champion.trade_date.iloc[0]), format="%Y%m%d")
    old_data = data.loc[dates <= first_frozen_signal]
    old_directory = args.output / "older_incumbent"
    old_directory.mkdir()
    older = calculate(old_data, config, options, "baseline", old_directory, None, None)
    if older.trade_date.max() >= champion.trade_date.min():
        raise ValueError("Extended history would replace a frozen incumbent prediction.")
    extended = validate_frame(pd.concat([older, champion], ignore_index=True))
    pd.testing.assert_frame_equal(extended.iloc[-len(champion):].reset_index(drop=True), champion)
    extended.to_csv(args.output / "extended_baseline.csv", index=False, float_format="%.17g")
    core = pipeline.prediction_core
    assets = load_assets(args.output)
    base, features = context_features(core, data, config, assets)
    target = base.close.shift(-1) / base.close - 1
    truth = pd.Series(target.to_numpy(), index=base.date.dt.strftime("%Y%m%d").astype(int)).reindex(extended.trade_date)
    if not np.array_equal(extended.real_pct_change, truth):
        raise ValueError("Historical extension target alignment failed.")
    write_json(args.output / "alignment.json", {"older_prediction_rows": len(older), "frozen_rows_preserved": len(champion),
                                               "target_alignment": True, "incumbent_forecasts_preserved": True})
    development = extended.loc[extended.trade_date <= 20241231].reset_index(drop=True)
    development_base = base.loc[base.date <= "2024-12-31"]
    development_features = features.iloc[:len(development_base)]
    baseline_years = {year: metrics(development.loc[development.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
    (args.output / "development").mkdir()
    outcomes = []
    for name in HISTORY_CANDIDATES:
        result = residual_predictions(core, development, development_base, development_features, name)
        result.to_csv(args.output / "development" / f"{name}.csv", index=False, float_format="%.17g")
        yearly = {year: metrics(result.loc[result.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
        gains = [yearly[year][metric] - baseline_years[year][metric] for year in (2023, 2024) for metric in ("accuracy", "balanced_accuracy")]
        selected_rows = result.loc[result.trade_date >= 20230101]
        changes = int(selected_rows.residual_changed.sum())
        outcome = {"candidate": name, "yearly": yearly, "minimum_gain": min(gains), "combined": metrics(selected_rows),
                   "changed_rows": changes, "eligible": min(gains) >= -1e-12 and changes >= 5}
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)
    eligible = [item for item in outcomes if item["eligible"]]
    selected = max(eligible, key=lambda item: (item["minimum_gain"], item["combined"]["balanced_accuracy"], -item["changed_rows"])) if eligible else None
    write_json(args.output / "selection.json", {"selected": selected, "outcomes": outcomes, "baseline_years": baseline_years})
    if selected is None:
        report = {"passed": False, "stage": "development", "reason": "No historical-context candidate passed both development years.", "production_changed": False}
    else:
        result = residual_predictions(core, extended, base, features, selected["candidate"])
        frozen_development = pd.read_csv(args.output / "development" / f"{selected['candidate']}.csv", float_precision="round_trip")
        columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence", "residual_probability_up"]
        pd.testing.assert_frame_equal(result.iloc[:len(development)][columns], frozen_development[columns])
        evaluated = result.loc[result.trade_date >= champion.trade_date.min()].reset_index(drop=True)
        evaluated.to_csv(args.output / "candidate_predictions.csv", index=False, float_format="%.17g")
        report = {"candidate": selected["candidate"], **compare_frames(champion, evaluated)}
    write_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
