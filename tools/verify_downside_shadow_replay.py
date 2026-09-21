"""Independently recompute fixed downside forecasts with future inputs removed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tushare_prediction_pipeline as pipeline
from prediction_service.archive import sha256_file
from prediction_service.service import PredictionService
from tools.fixed_downside_candidate import (
    assert_parity, canonical_json, load_candidate, predict, read_frame,
)


def verify(bundle, archive, output, probes):
    frozen = load_candidate(bundle)
    PredictionService._assert_manifest_integrity(archive / "manifest.json")
    market = read_frame(archive / "features.csv")
    expected = read_frame(archive / "candidate.csv")
    control = read_frame(archive / "baseline.csv")
    sources = {name: read_frame(archive / "raw" / f"context_{name}.csv")
               for name in ("breadth", "moneyflow", "spx", "nasdaq")}
    dates = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    if len(probes) != len(set(probes)) or any(day not in set(expected.trade_date) for day in probes):
        raise ValueError("Probe dates must be distinct archived signal dates.")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "release_id": frozen["release_id"], "verification_kind": "retrospective_prefix_replay",
        "archive_manifest_sha256": sha256_file(archive / "manifest.json"),
        "verifier_sha256": sha256_file(Path(__file__)),
        "created_at": datetime.now(timezone.utc).isoformat(), "probes": [],
        "passed": False, "promotion_allowed": False,
    }
    (output / "report.json").write_bytes(canonical_json(report))
    for day in probes:
        prefix = market.loc[dates.le(day)].drop(columns=["target_next_return", "target_next_direction"], errors="ignore")
        count = int((dates.le(day) & dates.ge(int(control.trade_date.iloc[0]))).sum())
        print(f"Verifying independent prefix through {day} ({count} predictions)", flush=True)
        baseline = pipeline.run_validation_and_prediction(prefix, validation_days=count - 1, progress=False)
        assert_parity(control.loc[control.trade_date.le(day)], baseline)
        context = {name: frame.loc[frame.trade_date.le(day) if name == "breadth" else frame.trade_date.lt(day)]
                   for name, frame in sources.items()}
        candidate = predict(prefix, baseline, context)
        reference = expected.loc[expected.trade_date.le(day)].reset_index(drop=True)
        assert_parity(reference, candidate)
        diagnostic_columns = ["correction_selected", "downside_probability", "downside_base_probability",
                              "downside_extended_probability", "downside_training_rows", "downside_last_training_date"]
        pd.testing.assert_frame_equal(reference[diagnostic_columns], candidate[diagnostic_columns],
                                      check_exact=True, check_dtype=False)
        if pd.notna(candidate.real_pct_change.iloc[-1]) or pd.notna(baseline.real_pct_change.iloc[-1]):
            raise ValueError("The prefix endpoint must have no future outcome.")
        path = output / f"prefix_{day}.csv"
        candidate.to_csv(path, index=False, float_format="%.17g")
        report["probes"].append({"signal_date": day, "rows": count, "passed": True,
                                 "endpoint_unresolved": True, "candidate_sha256": sha256_file(path)})
        (output / "report.json").write_bytes(canonical_json(report))
        print(f"Prefix {day}: exact forecast and training-diagnostic parity", flush=True)
    load_candidate(bundle)
    report["passed"] = True
    (output / "report.json").write_bytes(canonical_json(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", type=int, nargs="+", default=[20250102, 20260902])
    args = parser.parse_args()
    verify(args.bundle, args.archive, args.output, args.probes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
