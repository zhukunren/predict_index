"""Freeze three causal candidates on development dates, then check one winner.

The 2025+ data have previously been inspected in this repository. The final
comparison is a regression gate, not a claim of a new untouched holdout.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tushare_prediction_pipeline as pipeline
from tools.direction_rule_candidates import CANDIDATES, selector
from tools.evaluate_prediction_candidate import metric_summary, validate_frame


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def metrics(frame):
    result = metric_summary(frame)
    correct = frame["predicted_label"].eq(frame["real_pct_change"].gt(0))
    actual_up = frame["real_pct_change"].gt(0)
    direction = frame["predicted_label"]
    groups = direction.ne(direction.shift()).cumsum()
    up_runs = frame.loc[direction.eq(1)].groupby(groups).size()
    result.update(
        up_recall=float(correct[actual_up].mean()) if actual_up.any() else None,
        down_recall=float(correct[~actual_up].mean()) if (~actual_up).any() else None,
        up_rate_gap=float(direction.mean() - actual_up.mean()),
        longest_up=int(up_runs.max()) if len(up_runs) else 0,
        switch_rate=float(direction.ne(direction.shift()).iloc[1:].mean()) if len(frame) > 1 else None,
    )
    return result


def calculate(data, config, options, name, output, start, end, *, selector_factory=selector):
    core = pipeline.prediction_core
    original = core._nested_volatility_rule_signal
    if name != "baseline" and selector_factory is not None:
        core._nested_volatility_rule_signal = selector_factory(core, name)
    started = time.monotonic()
    try:
        result = core.loop_validate_prediction_results(
            data, config=config, **(options | {"start_date": start, "end_date": end,
            "diagnostics_output_path": str(output / f"{name}_diagnostics.csv")}),
        )
    finally:
        core._nested_volatility_rule_signal = original
    result = validate_frame(result)
    result.to_csv(output / f"{name}_predictions.csv", index=False, float_format="%.17g")
    print(json.dumps({"candidate": name, "rows": len(result), "seconds": round(time.monotonic() - started, 2)}), flush=True)
    return result


def compare_frames(reference, candidate):
    reference = validate_frame(reference)
    candidate = validate_frame(candidate)
    if not np.array_equal(reference.trade_date, candidate.trade_date) or not np.array_equal(reference.real_pct_change, candidate.real_pct_change):
        raise ValueError("Candidate and baseline date/target alignment failed.")
    groups = {"regression": reference.trade_date >= 20250101}
    groups.update({f"latest_{n}": reference.index >= len(reference) - n for n in (252, 60, 20)})
    checks = []
    reports = {}
    for group, mask in groups.items():
        left, right = metrics(candidate.loc[mask]), metrics(reference.loc[mask])
        reports[group] = {"candidate": left, "baseline": right}
        for metric in ("accuracy", "balanced_accuracy", "down_recall", "return_mae", "return_rmse", "calibrated_confidence_brier"):
            higher = metric in ("accuracy", "balanced_accuracy", "down_recall")
            passed = left[metric] >= right[metric] - 1e-12 if higher else left[metric] <= right[metric] + 1e-12
            checks.append({"group": group, "metric": metric, "candidate": left[metric], "baseline": right[metric], "passed": bool(passed)})
    for group, metric in (("latest_20", "up_rate_gap"), ("latest_60", "longest_up")):
        report = reports[group]
        passed = abs(report["candidate"][metric]) < abs(report["baseline"][metric])
        checks.append({"group": group, "metric": metric, "candidate": report["candidate"][metric], "baseline": report["baseline"][metric], "passed": bool(passed)})
    paired = candidate.loc[groups["regression"], "correct"].astype(int).to_numpy() - reference.loc[groups["regression"], "correct"].astype(int).to_numpy()
    return {"passed": all(check["passed"] for check in checks), "groups": reports, "checks": checks,
            "paired_direction_changes": {"corrected": int((paired > 0).sum()), "damaged": int((paired < 0).sum())},
            "production_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("rank", "probability"), default="rank")
    args = parser.parse_args()
    candidates, selector_factory = CANDIDATES, selector
    if args.family == "probability":
        from tools.probabilistic_rule_candidates import CANDIDATES as candidates, selector as selector_factory
    args.output.mkdir(parents=True, exist_ok=False)
    config, options = pipeline._default_calculation_options()
    options = {key: None if key.endswith("_path") else value for key, value in options.items()}
    options.update(periods=0, include_latest=False, progress=False)
    input_bytes = args.input.read_bytes()
    (args.output / "features.csv").write_bytes(input_bytes)
    sources = ["循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
               "tools/direction_rule_candidates.py", "tools/evaluate_prediction_candidate.py", "tools/evaluate_direction_bias.py"]
    if args.family == "probability":
        sources.append("tools/probabilistic_rule_candidates.py")
    source_hashes = {}
    for name in sources:
        source = (ROOT / name).read_bytes()
        source_hashes[name] = hashlib.sha256(source).hexdigest()
        frozen = args.output / "source" / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(source)
    write_json(args.output / "contract.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(), "source_sha256": source_hashes,
        "config": asdict(config), "loop_options": options, "candidates": candidates,
        "development": [20230101, 20241231], "regression_start": 20250101,
        "regression_data_previously_observed": True,
        "selection": "development only; best minimum gain in accuracy and balanced accuracy; no candidate changes after results",
        "gate": {"windows": ["regression", 252, 60, 20],
                 "nondecreasing": ["accuracy", "balanced_accuracy", "down_recall"],
                 "nonincreasing": ["return_mae", "return_rmse", "calibrated_confidence_brier"],
                 "bias": "strictly lower absolute up-rate gap over 20 days and longest up run over 60 days"},
    })
    data = pd.read_csv(args.input, float_precision="round_trip")
    dates = pd.to_datetime(data["trade_date"])
    development_data = data.loc[dates < "2025-01-01"].copy()
    # The following close may resolve the final development signal, but its
    # own forecast is excluded from candidate selection.
    next_row = data.loc[dates >= "2025-01-01"].head(1)
    development_data = pd.concat([development_data, next_row], ignore_index=True)
    development = args.output / "development"
    development.mkdir()
    outcomes = {}
    for name in ("baseline", *candidates):
        result = calculate(development_data, config, options, name, development, "20230101", "20241231", selector_factory=selector_factory)
        outcomes[name] = metrics(result)
        print(json.dumps({"development": name, "metrics": outcomes[name]}), flush=True)
    baseline = outcomes["baseline"]
    selected = max(candidates, key=lambda name: (
        min(outcomes[name]["accuracy"] - baseline["accuracy"], outcomes[name]["balanced_accuracy"] - baseline["balanced_accuracy"]),
        outcomes[name]["balanced_accuracy"], -abs(outcomes[name]["up_rate_gap"]),
    ))
    write_json(args.output / "selection.json", {"selected": selected, "development": outcomes})
    print(json.dumps({"selected": selected}), flush=True)
    comparison = args.output / "comparison"
    comparison.mkdir()
    results = {name: calculate(data, config, options, name, comparison, "20230101", None, selector_factory=selector_factory)
               for name in ("baseline", selected)}
    reference, candidate = results["baseline"], results[selected]
    report = {"candidate": selected, **compare_frames(reference, candidate)}
    write_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
