"""Derive option positions from frozen contract and daily activity responses."""

from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.option_position_features import NEAR_EXPIRY_DAYS, aggregate_positions, position_features
from tools.option_features import DAILY_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "baseline", "options", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--include-current", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    args = parser.parse_args()
    parent_raw = (args.options / "manifest.json").read_bytes()
    parent = json.loads(parent_raw)
    hashes = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest()}
    if parent["status"] != "complete" or any(parent["contract"][key] != value for key, value in hashes.items()):
        raise ValueError("Option positions require matching complete frozen option inputs.")
    metadata_raw = (args.options / "contracts.csv").read_bytes()
    if hashlib.sha256(metadata_raw).hexdigest() != parent["contracts_sha256"]:
        raise ValueError("Option position metadata hash mismatch.")
    metadata = pd.read_csv(args.options / "contracts.csv", float_precision="round_trip")
    market = pd.read_csv(args.input, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    positions = pd.Index(calendar).get_indexer(baseline.trade_date)
    if (positions < 20).any() or not pd.Series(positions).diff().dropna().gt(0).all():
        raise ValueError("Option position signals require chronological calendar alignment.")
    dates = calendar.iloc[positions[0] - 20:positions[-1]].tolist()
    if dates != parent["contract"]["dates"]:
        raise ValueError("Option positions require the exact frozen prior-session source range.")
    lag_sessions, publication_hour = (0, 20) if args.include_current else (1, 18)
    parts = []
    for month, item in sorted(parent["months"].items()):
        path = args.options / "raw" / f"{month}.csv.gz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Option position source hash mismatch for {month}.")
        parts.append(pd.read_csv(path, float_precision="round_trip"))
    sources = ("tools/option_position_features.py", "tools/build_option_positions.py",
               "tools/option_features.py", "tools/liquidity_features.py",
               "tools/fetch_option_context.py", "数据拉取脚本_tushare.py")
    contract = {**hashes, "source": "frozen Tushare opt_basic and opt_daily responses",
                "parent_manifest_sha256": hashlib.sha256(parent_raw).hexdigest(),
                "position_context": True, "near_expiry_calendar_days": NEAR_EXPIRY_DAYS,
                "lag_sessions": lag_sessions, "publication_hour": publication_hour,
                "matched_changes": "only contracts present on both adjacent domestic sessions; normalize by matched prior interest",
                "availability": ("same-day complete options, 20:00 research assumption" if args.include_current else "strictly prior domestic session"),
                "delivery_caveat": "historical responses lack original publication and revision timestamps",
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}
    args.output.mkdir(parents=True, exist_ok=False)
    for name in sources:
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (args.output / "parent_manifest.json").write_bytes(parent_raw)
    (args.output / "contracts.csv").write_bytes(metadata_raw)
    manifest = {"status": "partial", "created_at": datetime.now(timezone.utc).isoformat(), "contract": contract}
    def checkpoint():
        temporary = args.output / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output / "manifest.json")
    checkpoint()
    if args.include_current:
        from 数据拉取脚本_tushare import get_pro
        from tools.fetch_option_context import fetch_pages
        config = configparser.ConfigParser(interpolation=None)
        config.read(args.config, encoding="utf-8")
        date = int(baseline.trade_date.iloc[-1])
        extra = fetch_pages(get_pro(config.get("Tushare", "令牌")), "opt_daily",
                            {"exchange": "SSE", "start_date": str(date), "end_date": str(date), "fields": ",".join(DAILY_FIELDS)},
                            page_rows=15000, maximum_rows=150000)
        raw = gzip.compress(extra.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8"), mtime=0)
        (args.output / "current_options.csv.gz").write_bytes(raw)
        manifest["additional_source"] = {"trade_date": date, "rows": len(extra), "sha256": hashlib.sha256(raw).hexdigest()}
        parts.append(extra)
        dates.append(date)
        checkpoint()
    daily = aggregate_positions(pd.concat(parts, ignore_index=True), metadata, dates)
    original = pd.read_csv(args.options / "options.csv", float_precision="round_trip")
    if hashlib.sha256((args.options / "options.csv").read_bytes()).hexdigest() != parent["options_sha256"]:
        raise ValueError("Option position aggregate source hash mismatch.")
    pd.testing.assert_frame_equal(daily.loc[:, original.columns].iloc[:len(original)], original, check_exact=True, check_dtype=False)
    features = position_features(baseline.trade_date, daily, calendar, lag_sessions=lag_sessions, publication_hour=publication_hour)
    options_path, features_path = args.output / "options.csv", args.output / "position_features.csv"
    daily.to_csv(options_path, index=False, float_format="%.17g")
    features.to_csv(features_path, index=False, float_format="%.17g")
    manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), rows=len(daily),
                    feature_rows=len(features), unavailable_signal_dates=features.loc[~features.option_position_available, "trade_date"].tolist(),
                    options_sha256=hashlib.sha256(options_path.read_bytes()).hexdigest(),
                    features_sha256=hashlib.sha256(features_path.read_bytes()).hexdigest())
    checkpoint()
    print(json.dumps({"status": "complete", "days": len(daily), "features": len(features),
                      "unavailable_rows": len(manifest["unavailable_signal_dates"]), "additional_source_days": int(args.include_current)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
