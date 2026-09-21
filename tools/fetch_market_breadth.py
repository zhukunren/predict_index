"""Freeze full-market daily observations and breadth with resumable checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
from datetime import datetime, timezone
import gzip
import hashlib
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
from tools.breadth_features import REQUIRED_COLUMNS, aggregate_day, breadth_features
from tools.evaluate_direction_bias import write_json


PAGE_ROWS = 6000
REQUEST_INTERVAL = 0.5


class DailyRequests:
    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self.last_request = 0.0

    def fetch(self, trade_date):
        parts = []
        for offset in range(0, 30000, PAGE_ROWS):
            with self.lock:
                pause = REQUEST_INTERVAL - (time.monotonic() - self.last_request)
                if pause > 0:
                    time.sleep(pause)
                self.last_request = time.monotonic()
            try:
                part = self.client.daily(trade_date=str(trade_date), fields=",".join(REQUIRED_COLUMNS),
                                         limit=PAGE_ROWS, offset=offset)
            except Exception:
                raise RuntimeError(f"Daily request failed for {trade_date}; credentials omitted. Resume later.") from None
            if part is None or part.empty:
                if not parts:
                    raise ValueError(f"No daily observations for {trade_date}.")
                break
            parts.append(part)
            if len(part) < PAGE_ROWS:
                break
        else:
            raise ValueError(f"Daily pagination exceeded the supported universe size for {trade_date}.")
        return pd.concat(parts, ignore_index=True).sort_values("ts_code").reset_index(drop=True)


def reuse_raw_cache(source, output, dates, contract):
    raw_manifest = (source / "manifest.json").read_bytes()
    parent = json.loads(raw_manifest)
    if parent["status"] != "complete":
        raise ValueError("Only complete breadth inputs can supply a reusable raw cache.")
    for key in ("input_sha256", "baseline_sha256", "fields", "universe", "source"):
        if parent["contract"][key] != contract[key]:
            raise ValueError(f"Reusable breadth cache has a different {key}.")
    days = {}
    for date in dates:
        key = str(date)
        if key not in parent["days"]:
            continue
        path = source / "raw" / f"{date}.csv.gz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != parent["days"][key]["sha256"]:
            raise ValueError(f"Reusable daily cache hash mismatch for {date}.")
        target = output / "raw" / path.name
        try:
            target.hardlink_to(path)
        except OSError:
            shutil.copyfile(path, target)
        days[key] = parent["days"][key]
    return days, hashlib.sha256(raw_manifest).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--include-current", action="store_true")
    args = parser.parse_args()
    if args.resume and args.reuse_from is not None:
        parser.error("--resume and --reuse-from cannot be combined")
    market = pd.read_csv(args.input, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    positions = pd.Index(calendar).get_indexer(baseline.trade_date)
    if (positions < 20).any() or (pd.Series(positions).diff().dropna() <= 0).any():
        raise ValueError("Baseline requires chronological market alignment and 20 warmup sessions.")
    dates = calendar.iloc[positions[0] - 20:positions[-1] + int(args.include_current)].tolist()
    contract = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "source": "Tushare daily, historical traded stock universe per date",
        "universe": "SSE/SZSE stocks; BJ-coded records excluded including their pre-exchange OTC history",
        "fields": REQUIRED_COLUMNS, "dates": dates,
        "availability": ("same-day daily close for publication at or after 17:00 Asia/Shanghai, no forward filling"
                         if args.include_current else "strictly previous domestic trading day, no forward filling"),
        "daily_ingestion_documentation": "https://tushare.pro/document/2?doc_id=27; trading days 15:00-16:00 Asia/Shanghai",
        "delivery_caveat": "historical daily observations do not contain original publication/revision timestamps",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("tools/breadth_features.py", "tools/fetch_market_breadth.py")},
    }
    # Normalize tuples so a loaded JSON contract compares exactly on resume.
    contract = json.loads(json.dumps(contract))
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["contract"] != contract:
            raise ValueError("Resume input, source code, or date contract changed.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "raw").mkdir()
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "partial",
                    "contract": contract, "days": {}}
        if args.reuse_from is not None:
            manifest["days"], manifest["parent_manifest_sha256"] = reuse_raw_cache(args.reuse_from, args.output, dates, contract)
            manifest["reused_raw_days"] = len(manifest["days"])
        for name in contract["source_sha256"]:
            target = args.output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
    def checkpoint():
        temporary = args.output / "manifest.json.tmp"
        write_json(temporary, manifest)
        temporary.replace(args.output / "manifest.json")
    checkpoint()
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    requests = DailyRequests(get_pro(config.get("Tushare", "令牌")))
    rows = {}
    pending = []
    for date in dates:
        key = str(date)
        if key not in manifest["days"]:
            pending.append(date)
            continue
        path = args.output / "raw" / f"{date}.csv.gz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["days"][key]["sha256"]:
            raise ValueError(f"Cached stock data hash mismatch for {date}.")
        rows[date] = aggregate_day(pd.read_csv(path, dtype={"trade_date": str}, float_precision="round_trip"), date)
    started = time.monotonic()
    print(json.dumps({"total_days": len(dates), "cached_days": len(rows), "pending_days": len(pending)}), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for start in range(0, len(pending), 20):
            jobs = {pool.submit(requests.fetch, date): date for date in pending[start:start + 20]}
            for job in as_completed(jobs):
                date = jobs[job]
                frame = job.result()
                row = aggregate_day(frame, date)
                if row["source_rows"] < 1000 or row["shanghai_active_rows"] < 500:
                    raise ValueError(f"Possibly incomplete market universe for {date}.")
                raw = gzip.compress(frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8"), mtime=0)
                (args.output / "raw" / f"{date}.csv.gz").write_bytes(raw)
                manifest["days"][str(date)] = {"rows": len(frame), "sha256": hashlib.sha256(raw).hexdigest()}
                rows[date] = row
                checkpoint()
            print(json.dumps({"completed_days": len(rows), "total_days": len(dates),
                              "through_date": max(rows), "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    breadth = pd.DataFrame([rows[date] for date in dates])
    breadth_features(baseline.trade_date, breadth, calendar, lag_sessions=0 if args.include_current else 1)
    path = args.output / "breadth.csv"
    breadth.to_csv(path, index=False, float_format="%.17g")
    manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                    breadth_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(breadth))
    checkpoint()
    print(json.dumps({"status": "complete", "days": len(breadth), "raw_stock_rows": int(breadth.source_rows.sum())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
