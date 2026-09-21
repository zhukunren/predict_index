"""Evaluate three declared Hong Kong data timing policies for evening forecasts."""

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
from tools.hangseng_timing import CANDIDATES, timed_features
from tools.evaluate_direction_bias import calculate, compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame


PUBLICATION_HOUR = 18


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    hashes = {}
    files = {"features.csv": args.snapshot / "features.csv", "hangseng.csv": args.snapshot / "raw" / "hangseng.csv",
             "baseline.csv": args.baseline}
    for name, source in files.items():
        raw = source.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        expected = manifest["files"].get("raw/hangseng.csv" if name == "hangseng.csv" else name)
        if expected is not None and expected != hashes[name]:
            raise ValueError(f"Frozen snapshot hash mismatch: {name}.")
        (args.output / name).write_bytes(raw)
    sources = {}
    for name in ("循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
                 "数据拉取脚本_tushare.py", "tools/hangseng_timing.py", "tools/evaluate_hangseng_timing.py",
                 "tools/evaluate_direction_bias.py", "tools/evaluate_prediction_candidate.py", "tools/direction_rule_candidates.py"):
        raw = (ROOT / name).read_bytes()
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        sources[name] = hashlib.sha256(raw).hexdigest()
    config, options = pipeline._default_calculation_options()
    options = {key: None if key.endswith("_path") else value for key, value in options.items()}
    options.update(periods=0, include_latest=False, progress=False)
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "inputs": hashes, "sources": sources,
        "config": asdict(config), "loop_options": options, "candidates": CANDIDATES,
        "publication_time": f"{PUBLICATION_HOUR:02}:00 Asia/Shanghai, after Hong Kong closing auction",
        "volume_policy": "prior completed Hong Kong session only",
        "availability_caveat": "historical daily prices are assumed available after close; source delivery timestamps are not present",
        "selection_years": [2023, 2024], "regression_data_previously_observed": True,
        "eligibility": "accuracy and balanced accuracy nondecreasing separately in both years; at least 5 changed directions",
        "selection": "minimum yearly gain, combined balanced accuracy, fewer changes",
        "gate": "unchanged compare_frames", "automatic_promotion": False,
    })
    data = pd.read_csv(args.output / "features.csv", float_precision="round_trip")
    raw_hangseng = pd.read_csv(args.output / "hangseng.csv", float_precision="round_trip")
    champion = validate_frame(pd.read_csv(args.baseline, float_precision="round_trip"))
    dates = pd.to_datetime(data.trade_date)
    development_data = pd.concat([data.loc[dates < "2025-01-01"], data.loc[dates >= "2025-01-01"].head(1)], ignore_index=True)
    baseline = champion.loc[champion.trade_date <= 20241231].reset_index(drop=True)
    baseline_years = {year: metrics(baseline.loc[baseline.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
    development = args.output / "development"
    development.mkdir()
    outcomes = []
    for name in CANDIDATES:
        prepared, availability = timed_features(development_data, raw_hangseng, name, publication_hour=PUBLICATION_HOUR)
        prepared.to_csv(development / f"{name}_features.csv", index=False, float_format="%.17g")
        availability.to_csv(development / f"{name}_availability.csv", index=False)
        result = calculate(prepared, config, options, name, development, None, "20241231", selector_factory=None)
        if not np.array_equal(result.trade_date, baseline.trade_date) or not np.array_equal(result.real_pct_change, baseline.real_pct_change):
            raise ValueError("Hong Kong timing changed domestic prediction dates or targets.")
        yearly = {year: metrics(result.loc[result.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
        gains = [yearly[year][metric] - baseline_years[year][metric] for year in (2023, 2024) for metric in ("accuracy", "balanced_accuracy")]
        changes = int(((result.predicted_label != baseline.predicted_label) & result.trade_date.ge(20230101)).sum())
        outcome = {"candidate": name, "yearly": yearly, "minimum_gain": min(gains), "changed_rows": changes,
                   "combined": metrics(result.loc[result.trade_date >= 20230101]), "eligible": min(gains) >= -1e-12 and changes >= 5}
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)
    eligible = [item for item in outcomes if item["eligible"]]
    selected = max(eligible, key=lambda item: (item["minimum_gain"], item["combined"]["balanced_accuracy"], -item["changed_rows"])) if eligible else None
    write_json(args.output / "selection.json", {"selected": selected, "outcomes": outcomes, "baseline_years": baseline_years})
    if selected is None:
        report = {"passed": False, "stage": "development", "reason": "No timing candidate passed both development years.", "production_changed": False}
    else:
        prepared, availability = timed_features(data, raw_hangseng, selected["candidate"], publication_hour=PUBLICATION_HOUR)
        prepared.to_csv(args.output / "candidate_features.csv", index=False, float_format="%.17g")
        availability.to_csv(args.output / "candidate_availability.csv", index=False)
        comparison = args.output / "comparison"
        comparison.mkdir()
        result = calculate(prepared, config, options, selected["candidate"], comparison, None, None, selector_factory=None)
        frozen = pd.read_csv(development / f"{selected['candidate']}_predictions.csv", float_precision="round_trip")
        columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence"]
        pd.testing.assert_frame_equal(frozen[columns], result.iloc[:len(frozen)][columns])
        report = {"candidate": selected["candidate"], **compare_frames(champion, result)}
    write_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
