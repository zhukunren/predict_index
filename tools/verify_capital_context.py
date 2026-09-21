"""Audit capitalization inputs and reproduce feature prefixes without future data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.capital_features import COVERAGE_COLUMNS, capital_day, capital_features


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(args):
    manifest = json.loads((args.context / "manifest.json").read_text(encoding="utf-8"))
    contract = manifest["contract"]
    if manifest["status"] != "complete":
        raise ValueError("Capital audit requires a complete dataset.")
    for key, path in (("input_sha256", args.input), ("baseline_sha256", args.baseline)):
        if digest(path) != contract[key]:
            raise ValueError("Capital audit input differs from the acquisition contract.")
    for name, expected in contract["source_sha256"].items():
        if digest(args.context / "source" / name) != expected or digest(ROOT / name) != expected:
            raise ValueError("Capital acquisition sources changed.")
    parents = {}
    for name in ("breadth", "moneyflow"):
        path = getattr(args, name) / "manifest.json"
        if digest(path) != contract["parents"][name]:
            raise ValueError("Capital parent manifest changed.")
        parents[name] = json.loads(path.read_text(encoding="utf-8"))
    for key, path in (("capital_sha256", args.context / "capital.csv"),
                      ("features_sha256", args.context / "capital_features.csv")):
        if digest(path) != manifest[key]:
            raise ValueError("Capital aggregate or saved features changed.")
    rows = []
    for date in contract["dates"]:
        frames = {}
        for name, directory, daily_manifest in (("basic", args.context, manifest),
                                               ("breadth", args.breadth, parents["breadth"]),
                                               ("moneyflow", args.moneyflow, parents["moneyflow"])):
            path = directory / "raw" / f"{date}.csv.gz"
            if digest(path) != daily_manifest["days"][str(date)]["sha256"]:
                raise ValueError(f"Raw {name} checksum failed for {date}.")
            frames[name] = pd.read_csv(path, float_precision="round_trip")
        rows.append(capital_day(frames["basic"], frames["breadth"], frames["moneyflow"], date))
    daily = pd.DataFrame(rows)
    stored = pd.read_csv(args.context / "capital.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(daily, stored, check_exact=True, check_dtype=False)
    market = pd.read_csv(args.input, float_precision="round_trip")
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    features = capital_features(baseline.trade_date, daily, calendar)
    pd.testing.assert_frame_equal(features, pd.read_csv(args.context / "capital_features.csv", float_precision="round_trip"),
                                  check_exact=True, check_dtype=False)
    probes = []
    for cutoff in (20221230, 20241231, int(baseline.trade_date.iloc[-1])):
        signals = baseline.trade_date[baseline.trade_date.le(cutoff)]
        prefix = capital_features(signals, daily.loc[daily.trade_date.lt(cutoff)], calendar[calendar.le(cutoff)])
        pd.testing.assert_frame_equal(features.loc[features.trade_date.le(cutoff)].reset_index(drop=True), prefix,
                                      check_exact=True, check_dtype=False)
        probes.append({"signal_date": cutoff, "rows": len(prefix), "passed": True})
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(), "passed": True,
        "verifier_sha256": digest(Path(__file__)), "context_manifest_sha256": digest(args.context / "manifest.json"),
        "raw_response_days": len(rows), "capital_raw_rows": sum(day["rows"] for day in manifest["days"].values()),
        "feature_rows": len(features), "unavailable_rows": int((~features.capital_available).sum()),
        "minimum_coverage": {key: float(daily[key].min()) for key in COVERAGE_COLUMNS}, "prefixes": probes,
        "proof_scope": "checksums, exact aggregation, feature serialization and causal date prefixes; not model performance",
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("context", "breadth", "moneyflow", "input", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
