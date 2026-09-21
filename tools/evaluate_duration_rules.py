"""Freeze one duration hypothesis, reject on development before regression."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tushare_prediction_pipeline as pipeline
from tools.duration_rule_candidates import CANDIDATE, PARAMETERS, selector
from tools.evaluate_direction_bias import calculate, compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "baseline", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, source in (("features.csv", args.input), ("baseline.csv", args.baseline)):
        raw = source.read_bytes()
        (args.output / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    sources = {}
    names = ("\u5faa\u73af\u9a8c\u8bc1\u811a\u672c.py", "return_calibration.py", "regularized_direction.py",
             "tushare_prediction_pipeline.py", "tools/evaluate_direction_bias.py", "tools/direction_rule_candidates.py",
             "tools/evaluate_prediction_candidate.py", "tools/duration_rule_candidates.py", "tools/evaluate_duration_rules.py")
    for name in names:
        raw = (ROOT / name).read_bytes()
        path = args.output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        sources[name] = hashlib.sha256(raw).hexdigest()
    config, options = pipeline._default_calculation_options()
    options = {key: None if key.endswith("_path") else value for key, value in options.items()}
    options.update(periods=0, include_latest=False, progress=False)
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "inputs": hashes, "sources": sources,
        "candidate": CANDIDATE, "parameters": PARAMETERS, "config": asdict(config), "loop_options": options,
        "hypothesis": "Threshold state duration can change conditional reliability while binary state remains unchanged.",
        "eligibility": "2023 and 2024 separately: accuracy and balanced accuracy nondecreasing; at least 5 development changes",
        "gate": "unchanged evaluate_direction_bias.compare_frames, 26 checks",
        "regression_data_previously_observed": True, "candidate_attempts": 1, "automatic_promotion": False,
    })
    data = pd.read_csv(args.input, float_precision="round_trip")
    baseline = validate_frame(pd.read_csv(args.baseline, float_precision="round_trip"))
    dates = pd.to_datetime(data.trade_date)
    development_data = pd.concat([data.loc[dates < "2025-01-01"], data.loc[dates >= "2025-01-01"].head(1)], ignore_index=True)
    development = args.output / "development"
    development.mkdir()
    result = calculate(development_data, config, options, CANDIDATE, development, None, "20241231", selector_factory=selector)
    reference = baseline.loc[baseline.trade_date <= 20241231].reset_index(drop=True)
    if not result.trade_date.equals(reference.trade_date) or not result.real_pct_change.equals(reference.real_pct_change):
        raise ValueError("Development dates or outcomes disagree with the frozen baseline.")
    yearly = {year: metrics(result.loc[result.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
    baseline_years = {year: metrics(reference.loc[reference.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
    changes = int(((result.predicted_label != reference.predicted_label) & result.trade_date.ge(20230101)).sum())
    gains = [yearly[year][metric] - baseline_years[year][metric]
             for year in (2023, 2024) for metric in ("accuracy", "balanced_accuracy")]
    eligible = bool(min(gains) >= -1e-12 and changes >= 5)
    selection = {"candidate": CANDIDATE, "eligible": eligible, "yearly": yearly, "baseline_years": baseline_years,
                 "minimum_gain": min(gains), "changed_rows": changes}
    write_json(args.output / "selection.json", selection)
    print(json.dumps(selection), flush=True)
    if eligible:
        comparison = args.output / "comparison"
        comparison.mkdir()
        full = calculate(data, config, options, CANDIDATE, comparison, None, None, selector_factory=selector)
        columns = ["trade_date", "predicted_label", "predicted_pct_change", "confidence", "calibrated_confidence"]
        pd.testing.assert_frame_equal(result[columns], full.iloc[:len(result)][columns], check_exact=True)
        report = {"candidate": CANDIDATE, "development_prefix_exact": True, **compare_frames(baseline, full)}
    else:
        report = {"candidate": CANDIDATE, "passed": False, "stage": "development",
                  "reason": "Candidate failed the fixed development requirements.", "production_changed": False}
    write_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
