"""Evaluate one frozen, evidence-gated correction over the three fixed experts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tushare_prediction_pipeline as pipeline
from tools.causal_direction_guard import PARAMETERS, apply_guard
from tools.direction_rule_candidates import CANDIDATES
from tools.evaluate_direction_bias import calculate, compare_frames, metrics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config, options = pipeline._default_calculation_options()
    options = {key: None if key.endswith("_path") else value for key, value in options.items()}
    options.update(periods=0, include_latest=False, progress=False)
    raw_input = args.input.read_bytes()
    (args.output / "features.csv").write_bytes(raw_input)
    sources = ["循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
               "tools/direction_rule_candidates.py", "tools/causal_direction_guard.py",
               "tools/evaluate_prediction_candidate.py", "tools/evaluate_direction_bias.py", "tools/evaluate_direction_guard.py"]
    hashes = {}
    for source in sources:
        raw = (ROOT / source).read_bytes()
        hashes[source] = hashlib.sha256(raw).hexdigest()
        target = args.output / "source" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "input_sha256": hashlib.sha256(raw_input).hexdigest(),
        "sources": hashes, "parameters": PARAMETERS, "challengers": CANDIDATES,
        "config": asdict(config), "loop_options": options,
        "candidate": "causal_disagreement_guard", "candidate_attempts": 1,
        "labels": "strictly completed before the current signal; only same champion direction and disagreement",
        "gate": "same metrics, dates and thresholds as evaluate_direction_bias.compare_frames",
        "regression_data_previously_observed": True, "automatic_promotion": False,
    })
    data = pd.read_csv(args.input, float_precision="round_trip")
    development = args.output / "development"
    development.mkdir()
    dates = pd.to_datetime(data.trade_date)
    development_data = pd.concat([data.loc[dates < "2025-01-01"], data.loc[dates >= "2025-01-01"].head(1)], ignore_index=True)
    development_results = {name: calculate(development_data, config, options, name, development, None, "20241231")
                           for name in ("baseline", *CANDIDATES)}
    corrected = apply_guard(pipeline.prediction_core, development_results["baseline"],
                            {name: development_results[name] for name in CANDIDATES})
    corrected.to_csv(development / "guard_predictions.csv", index=False, float_format="%.17g")
    mask = corrected.trade_date >= 20230101
    summary = {"baseline": metrics(development_results["baseline"].loc[mask]),
               "guard": metrics(corrected.loc[mask]),
               "override_rows": int(corrected.loc[mask, "guard_source"].ne("baseline").sum())}
    write_json(args.output / "development.json", summary)
    print(summary, flush=True)
    comparison = args.output / "comparison"
    comparison.mkdir()
    results = {name: calculate(data, config, options, name, comparison, None, None)
               for name in ("baseline", *CANDIDATES)}
    corrected = apply_guard(pipeline.prediction_core, results["baseline"], {name: results[name] for name in CANDIDATES})
    corrected.to_csv(comparison / "guard_predictions.csv", index=False, float_format="%.17g")
    report = {"candidate": "causal_disagreement_guard", **compare_frames(results["baseline"], corrected),
              "override_rows": int(corrected.guard_source.ne("baseline").sum())}
    write_json(args.output / "report.json", report)
    print({"passed": report["passed"], "override_rows": report["override_rows"],
           "failed_checks": [item for item in report["checks"] if not item["passed"]]}, flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
