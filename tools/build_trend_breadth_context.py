"""Build multi-session breadth from existing, hash-verified Tushare cache."""

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

from tools.trend_breadth_features import POLICY, FEATURE_COLUMNS, TrendBreadthAccumulator, trend_breadth_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("breadth", "input", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    raw_manifest = (args.breadth / "manifest.json").read_bytes()
    parent = json.loads(raw_manifest)
    hashes = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest()}
    if parent["status"] != "complete" or any(parent["contract"][key] != value for key, value in hashes.items()):
        raise ValueError("Trend context requires the matching frozen breadth and original model inputs.")
    market = pd.read_csv(args.input, float_precision="round_trip")
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    dates = parent["contract"]["dates"]
    positions = pd.Index(calendar).get_indexer(dates)
    if (positions < 0).any() or not pd.Series(positions).diff().dropna().eq(1).all():
        raise ValueError("Raw trend dates must be consecutive domestic sessions.")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ["tools/trend_breadth_features.py", "tools/build_trend_breadth_context.py", "tools/breadth_features.py", "tools/liquidity_features.py"]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "partial",
                "policy": POLICY, "features": FEATURE_COLUMNS,
                "contract": {**hashes, "parent_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                             "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}},
                "raw_directory": str(args.breadth.resolve() / "raw"), "network_requests": 0}
    def checkpoint():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checkpoint()
    (args.output / "parent_manifest.json").write_bytes(raw_manifest)
    for name in sources:
        path = args.output / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    accumulator = TrendBreadthAccumulator(dates)
    rows = []
    source_rows = 0
    for index, date in enumerate(dates):
        raw = (args.breadth / "raw" / f"{date}.csv.gz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != parent["days"][str(date)]["sha256"]:
            raise ValueError(f"Trend raw source hash mismatch: {date}.")
        daily = pd.read_csv(io.BytesIO(raw), compression="gzip", float_precision="round_trip")
        source_rows += len(daily)
        rows.append(accumulator.step(daily, date))
        if (index + 1) % 200 == 0:
            print(json.dumps({"completed_days": index + 1, "total_days": len(dates)}), flush=True)
    daily = pd.DataFrame(rows)
    features = trend_breadth_features(baseline.trade_date, daily, calendar)
    daily.to_csv(args.output / "trend_breadth.csv", index=False, float_format="%.17g")
    features.to_csv(args.output / "features.csv", index=False, float_format="%.17g")
    manifest.update(status="complete", raw_rows=source_rows, rows=len(daily), feature_rows=len(features),
                    available_feature_rows=int(features.trend_available.sum()),
                    trend_sha256=hashlib.sha256((args.output / "trend_breadth.csv").read_bytes()).hexdigest(),
                    feature_sha256=hashlib.sha256((args.output / "features.csv").read_bytes()).hexdigest(),
                    completed_at_utc=datetime.now(timezone.utc).isoformat())
    checkpoint()
    print(json.dumps({k:manifest[k] for k in ("status", "raw_rows", "rows", "available_feature_rows", "network_requests")}), flush=True)


if __name__ == "__main__":
    main()
