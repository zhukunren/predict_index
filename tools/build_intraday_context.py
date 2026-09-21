"""Validate authorized local index bars and freeze an intraday research context."""

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

from tools.intraday_features import POLICY, FEATURE_COLUMNS, intraday_features


def build_context(bar_paths, daily_path, baseline_path, output, *, frequency_minutes=30,
                  volume_unit="shares", amount_unit="yuan", source_description):
    if not source_description.strip():
        raise ValueError("A data source and timestamp convention must be documented.")
    if output.exists():
        raise FileExistsError("Context output already exists; frozen evidence cannot be replaced.")
    bar_bytes = [(Path(path), Path(path).read_bytes()) for path in bar_paths]
    if not bar_bytes:
        raise ValueError("At least one authorized local bar file is required.")
    daily_bytes, baseline_bytes = daily_path.read_bytes(), baseline_path.read_bytes()
    frames = [pd.read_csv(io.BytesIO(raw), compression="gzip" if path.suffix == ".gz" else None, float_precision="round_trip") for path,raw in bar_bytes]
    bars = pd.concat(frames, ignore_index=True)
    daily = pd.read_csv(io.BytesIO(daily_bytes), float_precision="round_trip")
    baseline = pd.read_csv(io.BytesIO(baseline_bytes), float_precision="round_trip")
    features = intraday_features(baseline.trade_date, bars, daily, frequency_minutes=frequency_minutes,
                                 volume_unit=volume_unit, amount_unit=amount_unit)
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    receipts = []
    for index, (path, raw) in enumerate(bar_bytes):
        name = f"bars_{index:03d}.csv" + (".gz" if path.suffix == ".gz" else "")
        (output / "raw" / name).write_bytes(raw)
        receipts.append({"original_path": str(path.resolve()), "stored_file": f"raw/{name}",
                         "sha256": hashlib.sha256(raw).hexdigest()})
    (output / "daily.csv").write_bytes(daily_bytes)
    (output / "baseline.csv").write_bytes(baseline_bytes)
    features.to_csv(output / "features.csv", index=False, float_format="%.17g")
    sources = ["tools/intraday_features.py", "tools/build_intraday_context.py", "tools/liquidity_features.py"]
    for name in sources:
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    manifest = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_description": source_description, "source_paths_are_historical_delivery_evidence": False,
                "policy": POLICY, "frequency_minutes": frequency_minutes, "volume_unit": volume_unit, "amount_unit": amount_unit,
                "features": FEATURE_COLUMNS, "feature_rows": len(features), "raw_rows": len(bars),
                "input_sha256": hashlib.sha256(daily_bytes).hexdigest(), "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
                "features_sha256": hashlib.sha256((output/"features.csv").read_bytes()).hexdigest(),
                "source_sha256": {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                "raw_files": receipts, "production_changed": False, "performance_evaluated": False,
                "network_requests": 0}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, nargs="+", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frequency", type=int, choices=(1,5,15,30), default=30)
    parser.add_argument("--volume-unit", choices=("shares", "hands"), default="shares")
    parser.add_argument("--amount-unit", choices=("yuan", "thousand_yuan"), default="yuan")
    parser.add_argument("--source-description", required=True)
    args = parser.parse_args()
    manifest = build_context(args.bars, args.input, args.baseline, args.output, frequency_minutes=args.frequency,
                             volume_unit=args.volume_unit, amount_unit=args.amount_unit, source_description=args.source_description)
    print(json.dumps({key:manifest[key] for key in ("status", "feature_rows", "raw_rows", "performance_evaluated", "network_requests")}), flush=True)


if __name__ == "__main__":
    main()
