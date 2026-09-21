"""Gate the repository's existing alternate direction engines without tuning."""

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
from tools.evaluate_direction_bias import calculate, compare_frames, metrics, write_json
from tools.evaluate_prediction_candidate import validate_frame


CANDIDATES = ("stability_rule", "nested_ml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("existing", "validated-rules", "current-price", "correlation", "prequential"), default="existing")
    args = parser.parse_args()
    candidates = CANDIDATES
    selector_factory = None
    candidate_parameters = {}
    if args.family == "validated-rules":
        from tools.validated_rule_candidates import CANDIDATES as candidates, PARAMETERS, selector
        selector_factory = selector
        candidate_parameters = {"shared": PARAMETERS, "candidates": candidates}
    elif args.family == "current-price":
        from tools.price_direction_rules import CANDIDATES as candidates, PARAMETERS, selector
        selector_factory = selector
        candidate_parameters = {"shared": PARAMETERS, "candidates": candidates}
    elif args.family == "correlation":
        from tools.correlation_rule_candidates import CANDIDATES as candidates, PARAMETERS, selector
        selector_factory = selector
        candidate_parameters = {"shared": PARAMETERS, "candidates": candidates}
    elif args.family == "prequential":
        from tools.prequential_rule_candidates import CANDIDATES as candidates, PARAMETERS, selector
        selector_factory = selector
        candidate_parameters = {"shared": PARAMETERS, "candidates": candidates}
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, source in (("features.csv", args.input), ("baseline.csv", args.baseline)):
        raw = source.read_bytes()
        (args.output / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    sources = {}
    source_files = ["循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
                    "tools/evaluate_direction_bias.py", "tools/evaluate_prediction_candidate.py",
                    "tools/direction_rule_candidates.py", "tools/evaluate_core_direction_engines.py"]
    if args.family == "validated-rules":
        source_files.append("tools/validated_rule_candidates.py")
    elif args.family == "current-price":
        source_files.append("tools/price_direction_rules.py")
    elif args.family == "correlation":
        source_files.append("tools/correlation_rule_candidates.py")
    elif args.family == "prequential":
        source_files.append("tools/prequential_rule_candidates.py")
    for name in source_files:
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
        "config": asdict(config), "loop_options": options, "candidate_engines": list(candidates),
        "family": args.family, "candidate_parameters": candidate_parameters,
        "selection_years": [2023, 2024], "regression_data_previously_observed": True,
        "eligibility": "accuracy and balanced accuracy nondecreasing separately in both years; at least 5 changed directions",
        "selection": "minimum yearly gain, combined balanced accuracy, fewer changes",
        "gate": "unchanged compare_frames", "automatic_promotion": False,
    })
    data = pd.read_csv(args.input, float_precision="round_trip")
    champion = validate_frame(pd.read_csv(args.baseline, float_precision="round_trip"))
    dates = pd.to_datetime(data.trade_date)
    development_data = pd.concat([data.loc[dates < "2025-01-01"], data.loc[dates >= "2025-01-01"].head(1)], ignore_index=True)
    baseline_development = champion.loc[champion.trade_date <= 20241231].reset_index(drop=True)
    baseline_years = {year: metrics(baseline_development.loc[baseline_development.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
    development = args.output / "development"
    development.mkdir()
    outcomes = []
    for name in candidates:
        engine_options = options | {"signal_engine": name} if args.family == "existing" else options
        result = calculate(development_data, config, engine_options, name, development,
                           None, "20241231", selector_factory=selector_factory)
        if not result.trade_date.equals(baseline_development.trade_date) or not result.real_pct_change.equals(baseline_development.real_pct_change):
            raise ValueError("Alternate engine development dates or targets disagree with the frozen incumbent.")
        yearly = {year: metrics(result.loc[result.trade_date.between(year * 10000, year * 10000 + 1231)]) for year in (2023, 2024)}
        gains = [yearly[year][metric] - baseline_years[year][metric] for year in (2023, 2024) for metric in ("accuracy", "balanced_accuracy")]
        changed = (result.predicted_label != baseline_development.predicted_label) & result.trade_date.ge(20230101)
        changes = int(changed.sum())
        outcome = {"candidate": name, "yearly": yearly, "minimum_gain": min(gains), "combined": metrics(result.loc[result.trade_date >= 20230101]),
                   "changed_rows": changes, "eligible": min(gains) >= -1e-12 and changes >= 5}
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)
    eligible = [item for item in outcomes if item["eligible"]]
    selected = max(eligible, key=lambda item: (item["minimum_gain"], item["combined"]["balanced_accuracy"], -item["changed_rows"])) if eligible else None
    write_json(args.output / "selection.json", {"selected": selected, "outcomes": outcomes, "baseline_years": baseline_years})
    if selected is None:
        report = {"passed": False, "stage": "development", "reason": "No candidate passed both development years.", "production_changed": False}
    else:
        comparison = args.output / "comparison"
        comparison.mkdir()
        engine_options = options | {"signal_engine": selected["candidate"]} if args.family == "existing" else options
        result = calculate(data, config, engine_options, selected["candidate"], comparison,
                           None, None, selector_factory=selector_factory)
        prefix = pd.read_csv(development / f"{selected['candidate']}_predictions.csv", float_precision="round_trip")
        columns = ["predicted_label", "predicted_pct_change", "calibrated_confidence"]
        pd.testing.assert_frame_equal(prefix[columns], result.iloc[:len(prefix)][columns], check_exact=True)
        report = {"candidate": selected["candidate"], **compare_frames(champion, result)}
    write_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
