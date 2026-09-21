"""Freeze dated futures contracts and lagged basis context for model research."""

from __future__ import annotations

import argparse
import configparser
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 数据拉取脚本_tushare import get_pro
from tools.evaluate_direction_bias import write_json
from tools.futures_features import FIELDS, PRODUCTS, aggregate_contracts, futures_features


def fetch_month(client, start, end):
    parts = []
    for offset in range(0, 10000, 2000):
        try:
            with redirect_stdout(io.StringIO()):
                part = client.query("fut_daily", exchange="CFFEX", start_date=start, end_date=end,
                                    fields=",".join(FIELDS), limit=2000, offset=offset)
        except Exception:
            raise RuntimeError(f"Futures request failed for {start}; credentials omitted. Resume later.") from None
        if part is None or part.empty:
            break
        parts.append(part)
        if len(part) < 2000:
            break
    else:
        raise ValueError("Futures pagination exceeded the monthly bound.")
    if not parts:
        raise ValueError(f"Empty futures history for {start}.")
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(["ts_code", "trade_date"]).any() or not frame.trade_date.astype(str).between(start, end).all():
        raise ValueError("Futures pagination contains duplicates or unexpected dates.")
    return frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "baseline", "context", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--include-current", action="store_true")
    args = parser.parse_args()
    if args.resume and args.reuse_from is not None:
        parser.error("--resume and --reuse-from cannot be combined")
    market = pd.read_csv(args.input, float_precision="round_trip")
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    positions = pd.Index(calendar).get_indexer(baseline.trade_date)
    if (positions < 20).any() or not pd.Series(positions).diff().dropna().gt(0).all():
        raise ValueError("Futures signals require aligned market dates and 20 warmup days.")
    dates = calendar.iloc[positions[0] - 20:positions[-1] + int(args.include_current)].tolist()
    context_manifest = json.loads((args.context / "manifest.json").read_text(encoding="utf-8"))
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if context_manifest["input_sha256"] != input_hash:
        raise ValueError("Spot data do not match the frozen market snapshot.")
    spot_hashes = {}
    for name in PRODUCTS.values():
        spot_hashes[name] = hashlib.sha256((args.context / f"{name}.csv").read_bytes()).hexdigest()
        if spot_hashes[name] != context_manifest["assets"][name]["sha256"]:
            raise ValueError(f"Spot data hash mismatch for {name}.")
    source_files = ("tools/futures_features.py", "tools/fetch_futures_context.py")
    contract = {
        "input_sha256": input_hash, "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "spot_sha256": spot_hashes, "fields": list(FIELDS), "dates": dates,
        "source": "Tushare fut_daily, all dated IF/IC contracts, continuous synthetic contracts excluded",
        "availability": ("same-day closing quotes for publication at or after 18:00 Asia/Shanghai"
                         if args.include_current else "strictly prior domestic session; no same-day derivatives input"),
        "delivery_caveat": "historical observations lack original publication and revision timestamps",
        "documentation": "https://tushare.pro/document/2?doc_id=138",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
    }
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["contract"] != contract:
            raise ValueError("Futures resume contract changed.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "raw").mkdir()
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "partial", "contract": contract, "months": {}}
        for name in source_files:
            target = args.output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        for name in PRODUCTS.values():
            (args.output / f"{name}.csv").write_bytes((args.context / f"{name}.csv").read_bytes())
        if args.reuse_from is not None:
            parent_raw = (args.reuse_from / "manifest.json").read_bytes()
            parent = json.loads(parent_raw)
            for key in ("input_sha256", "baseline_sha256", "spot_sha256", "fields", "source"):
                if parent["contract"][key] != contract[key]:
                    raise ValueError(f"Reusable futures cache has a different {key}.")
            previous_dates = parent["contract"]["dates"]
            if previous_dates != dates[:len(previous_dates)]:
                raise ValueError("Reusable futures dates must be a prefix of the requested history.")
            # A partial aggregate can still contain complete hashed raw months.
            for key, entry in parent["months"].items():
                month = key.replace("-", "")
                if [date for date in dates if str(date).startswith(month)] != [date for date in previous_dates if str(date).startswith(month)]:
                    continue
                path = args.reuse_from / "raw" / f"{key}.csv.gz"
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Reusable futures cache hash mismatch for {key}.")
                target = args.output / "raw" / path.name
                try:
                    target.hardlink_to(path)
                except OSError:
                    shutil.copyfile(path, target)
                manifest["months"][key] = entry
            manifest["parent_manifest_sha256"] = hashlib.sha256(parent_raw).hexdigest()
    def checkpoint():
        temporary = args.output / "manifest.json.tmp"
        write_json(temporary, manifest)
        temporary.replace(args.output / "manifest.json")
    checkpoint()
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    client = get_pro(config.get("Tushare", "令牌"))
    periods = pd.period_range(pd.to_datetime(str(dates[0])), pd.to_datetime(str(dates[-1])), freq="M")
    parts = []
    for period in periods:
        key = str(period)
        path = args.output / "raw" / f"{key}.csv.gz"
        if key in manifest["months"]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["months"][key]["sha256"]:
                raise ValueError(f"Futures cache hash mismatch for {key}.")
            frame = pd.read_csv(path, dtype={"trade_date": str}, float_precision="round_trip")
        else:
            start = max(period.start_time.strftime("%Y%m%d"), str(dates[0]))
            end = min(period.end_time.strftime("%Y%m%d"), str(dates[-1]))
            frame = fetch_month(client, start, end)
            raw = gzip.compress(frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8"), mtime=0)
            path.write_bytes(raw)
            manifest["months"][key] = {"rows": len(frame), "sha256": hashlib.sha256(raw).hexdigest()}
            checkpoint()
        parts.append(frame)
        if len(parts) % 6 == 0 or len(parts) == len(periods):
            print(json.dumps({"completed_months": len(parts), "total_months": len(periods), "through_month": key}), flush=True)
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.loc[raw.trade_date.astype(int).isin(dates)]
    spots = {name: pd.read_csv(args.output / f"{name}.csv", float_precision="round_trip") for name in PRODUCTS.values()}
    daily = aggregate_contracts(raw, spots)
    if daily.trade_date.tolist() != dates:
        raise ValueError("Futures history does not cover every required domestic session.")
    futures_features(baseline.trade_date, daily, calendar, lag_sessions=0 if args.include_current else 1)
    path = args.output / "futures.csv"
    daily.to_csv(path, index=False, float_format="%.17g")
    manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), rows=len(daily),
                    futures_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    checkpoint()
    print(json.dumps({"status": "complete", "days": len(daily), "raw_rows": len(raw)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
