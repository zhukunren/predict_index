"""Freeze stock distribution signals from hash-verified Tushare daily responses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.distribution_features import FEATURE_COLUMNS, distribution_day, distribution_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "baseline", "breadth", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    parent_raw = (args.breadth / "manifest.json").read_bytes()
    parent = json.loads(parent_raw)
    hashes = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest()}
    if parent["status"] != "complete" or any(parent["contract"][key] != value for key, value in hashes.items()):
        raise ValueError("Distribution requires a complete, aligned breadth snapshot.")
    market = pd.read_csv(args.input, float_precision="round_trip")
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    dates = parent["contract"]["dates"]
    locations = pd.Index(calendar).get_indexer(dates)
    if (locations < 0).any() or not pd.Series(locations).diff().dropna().eq(1).all():
        raise ValueError("Distribution source requires consecutive market dates.")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ("tools/distribution_features.py", "tools/build_distribution_context.py", "tools/breadth_features.py", "tools/liquidity_features.py")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "partial",
                "contract": {**hashes, "features": FEATURE_COLUMNS,
                             "availability": "same-day close, publication at 18:00 Shanghai",
                             "universe": parent["contract"]["universe"],
                             "delivery_caveat": parent["contract"]["delivery_caveat"],
                             "parent_manifest_sha256": hashlib.sha256(parent_raw).hexdigest(),
                             "robust_returns": "cross-sectional 1% and 99% winsorization for amount-weighted return only",
                             "transition_denominator": "stocks actively traded on both consecutive domestic sessions",
                             "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}}
    def checkpoint():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checkpoint()
    (args.output / "breadth_manifest.json").write_bytes(parent_raw)
    for name in sources:
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    previous = None
    rows = []
    for index, date in enumerate(dates):
        raw = (args.breadth / "raw" / f"{date}.csv.gz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != parent["days"][str(date)]["sha256"]:
            raise ValueError(f"Distribution raw input hash mismatch for {date}.")
        current = pd.read_csv(io.BytesIO(raw), compression="gzip", float_precision="round_trip")
        if previous is not None:
            rows.append(distribution_day(current, previous, date, dates[index - 1]))
        previous = current
        if (index + 1) % 200 == 0:
            print(json.dumps({"completed_days": index + 1, "total_days": len(dates)}), flush=True)
    daily = pd.DataFrame(rows)
    features = distribution_features(baseline.trade_date, daily, calendar)
    path = args.output / "distribution.csv"
    daily.to_csv(path, index=False, float_format="%.17g")
    features.to_csv(args.output / "distribution_features.csv", index=False, float_format="%.17g")
    manifest.update(status="complete", rows=len(daily), feature_rows=len(features),
                    distribution_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    completed_at=datetime.now(timezone.utc).isoformat())
    checkpoint()
    print(json.dumps({"status": "complete", "rows": len(daily), "feature_rows": len(features)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
