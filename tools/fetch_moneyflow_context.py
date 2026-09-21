"""Freeze historical order-size flows against the already frozen daily universe."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import threading
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 数据拉取脚本_tushare import get_pro
from tools.moneyflow_features import FIELDS, MIN_COVERAGE, aggregate_moneyflow, moneyflow_features


PAGE_ROWS = 6000
REQUEST_INTERVAL = 0.5


class MoneyflowRequests:
    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self.last_request = 0.0

    def fetch(self, date):
        parts = []
        for offset in range(0, 30000, PAGE_ROWS):
            with self.lock:
                pause = REQUEST_INTERVAL - (time.monotonic() - self.last_request)
                if pause > 0:
                    time.sleep(pause)
                self.last_request = time.monotonic()
            try:
                frame = self.client.query("moneyflow", trade_date=str(date), fields=",".join(FIELDS), limit=PAGE_ROWS, offset=offset)
            except Exception:
                raise RuntimeError(f"Moneyflow request failed for {date}; credentials omitted. Resume later.") from None
            if frame is None or frame.empty:
                if not parts:
                    raise ValueError(f"No moneyflow observations for {date}.")
                break
            parts.append(frame)
            if len(frame) < PAGE_ROWS:
                break
        else:
            raise ValueError("Moneyflow pagination exceeded its declared universe bound.")
        return pd.concat(parts, ignore_index=True).sort_values("ts_code").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "baseline", "breadth", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--price-context", action="store_true")
    args = parser.parse_args()
    if args.resume and args.reuse_from is not None:
        parser.error("--resume and --reuse-from cannot be combined")
    parent_raw = (args.breadth / "manifest.json").read_bytes()
    parent = json.loads(parent_raw)
    hashes = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest()}
    if parent["status"] != "complete" or any(parent["contract"][key] != value for key, value in hashes.items()):
        raise ValueError("Moneyflow requires matching complete frozen daily observations.")
    market = pd.read_csv(args.input, float_precision="round_trip")
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    positions = pd.Index(calendar).get_indexer(baseline.trade_date)
    if (positions < 20).any() or not pd.Series(positions).diff().dropna().gt(0).all():
        raise ValueError("Moneyflow baseline requires aligned dates and warmup history.")
    dates = calendar.iloc[positions[0] - 20:positions[-1]].tolist()
    if any(str(date) not in parent["days"] for date in dates):
        raise ValueError("Frozen daily universe does not cover moneyflow dates.")
    sources = ("tools/moneyflow_features.py", "tools/fetch_moneyflow_context.py", "tools/breadth_features.py", "tools/liquidity_features.py")
    contract = {**hashes, "source": "Tushare moneyflow, historical SH/SZ traded stocks",
                "fields": list(FIELDS), "dates": dates, "minimum_stock_and_amount_coverage": MIN_COVERAGE,
                "denominator": "sum of buy and sell amounts across all four size buckets, matched active stocks",
                "availability": "strictly prior domestic session; incomplete coverage excluded from training and correction",
                "delivery_caveat": "historical data lack original publication and revision timestamps",
                "price_context": args.price_context,
                "price_groups": "same-day price sign and highest-turnover 10% of matched stocks; all features lagged one session" if args.price_context else None,
                "parent_manifest_sha256": hashlib.sha256(parent_raw).hexdigest(),
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["contract"] != contract:
            raise ValueError("Moneyflow resume contract changed.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "raw").mkdir()
        (args.output / "breadth_manifest.json").write_bytes(parent_raw)
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "partial", "contract": contract, "days": {}}
        for name in sources:
            target = args.output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        if args.reuse_from is not None:
            previous_raw = (args.reuse_from / "manifest.json").read_bytes()
            previous = json.loads(previous_raw)
            if previous["status"] != "complete":
                raise ValueError("Moneyflow reuse requires a complete parent dataset.")
            for key in ("input_sha256", "baseline_sha256", "fields", "dates", "minimum_stock_and_amount_coverage", "parent_manifest_sha256"):
                if previous["contract"][key] != contract[key]:
                    raise ValueError(f"Moneyflow reuse requires matching {key}.")
            for date in dates:
                path = args.reuse_from / "raw" / f"{date}.csv.gz"
                if hashlib.sha256(path.read_bytes()).hexdigest() != previous["days"][str(date)]["sha256"]:
                    raise ValueError(f"Reusable moneyflow hash mismatch for {date}.")
                target = args.output / "raw" / path.name
                try:
                    target.hardlink_to(path)
                except OSError:
                    shutil.copyfile(path, target)
                manifest["days"][str(date)] = previous["days"][str(date)]
            manifest["reused_manifest_sha256"] = hashlib.sha256(previous_raw).hexdigest()
            manifest["reused_raw_days"] = len(dates)
    def checkpoint():
        temporary = args.output / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output / "manifest.json")
    def summarize(flow, date):
        raw = (args.breadth / "raw" / f"{date}.csv.gz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != parent["days"][str(date)]["sha256"]:
            raise ValueError(f"Frozen daily universe hash mismatch for {date}.")
        daily = pd.read_csv(io.BytesIO(raw), compression="gzip", float_precision="round_trip")
        return aggregate_moneyflow(flow, daily, date, price_context=args.price_context)
    checkpoint()
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    requests = MoneyflowRequests(get_pro(config.get("Tushare", "令牌")))
    rows, pending = {}, []
    for date in dates:
        key = str(date)
        if key not in manifest["days"]:
            pending.append(date)
            continue
        path = args.output / "raw" / f"{date}.csv.gz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["days"][key]["sha256"]:
            raise ValueError(f"Moneyflow cache hash mismatch for {date}.")
        rows[date] = summarize(pd.read_csv(path, float_precision="round_trip"), date)
    started = time.monotonic()
    print(json.dumps({"total_days": len(dates), "cached_days": len(rows), "pending_days": len(pending)}), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for start in range(0, len(pending), 20):
            jobs = {pool.submit(requests.fetch, date): date for date in pending[start:start + 20]}
            for job in as_completed(jobs):
                date = jobs[job]
                flow = job.result()
                row = summarize(flow, date)
                raw = gzip.compress(flow.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8"), mtime=0)
                (args.output / "raw" / f"{date}.csv.gz").write_bytes(raw)
                manifest["days"][str(date)] = {"rows": len(flow), "sha256": hashlib.sha256(raw).hexdigest(),
                                               "stock_coverage": row["stock_coverage"], "amount_coverage": row["amount_coverage"]}
                rows[date] = row
                checkpoint()
            print(json.dumps({"completed_days": len(rows), "total_days": len(dates), "through_date": max(rows),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    daily = pd.DataFrame([rows[date] for date in dates])
    features = moneyflow_features(baseline.trade_date, daily, calendar, price_context=args.price_context)
    path = args.output / "moneyflow.csv"
    daily.to_csv(path, index=False, float_format="%.17g")
    features.to_csv(args.output / "moneyflow_features.csv", index=False, float_format="%.17g")
    manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), rows=len(daily),
                    raw_rows=int(daily.raw_rows.sum()), moneyflow_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    unavailable_signal_dates=features.loc[~features.moneyflow_available, "trade_date"].tolist())
    checkpoint()
    print(json.dumps({"status": "complete", "days": len(daily), "raw_rows": manifest["raw_rows"],
                      "unavailable_signals": len(manifest["unavailable_signal_dates"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
