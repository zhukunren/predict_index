"""Freeze historical capitalization, reusing hash-verified stock/flow responses."""

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
import sys
import threading
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 数据拉取脚本_tushare import get_pro
from tools.capital_features import FIELDS, COVERAGE_COLUMNS, FEATURE_COLUMNS, LAG_SESSIONS, capital_day, capital_features
from tools.moneyflow_features import MIN_COVERAGE


class CapitalRequests:
    def __init__(self, client):
        self.client, self.lock, self.last_request = client, threading.Lock(), 0.

    def fetch(self, date):
        parts = []
        for offset in range(0, 30000, 6000):
            with self.lock:
                pause = 0.5 - (time.monotonic() - self.last_request)
                if pause > 0:
                    time.sleep(pause)
                self.last_request = time.monotonic()
            try:
                part = self.client.query("daily_basic", ts_code="", trade_date=str(date),
                                         fields=",".join(FIELDS), limit=6000, offset=offset)
            except Exception:
                raise RuntimeError(f"Capitalization request failed for {date}; credentials omitted. Resume later.") from None
            if part is None or part.empty:
                if not parts:
                    raise ValueError(f"No capitalization response for {date}.")
                break
            parts.append(part)
            if len(part) < 6000:
                break
        else:
            raise ValueError("Capitalization pagination exceeded its universe bound.")
        return pd.concat(parts, ignore_index=True).sort_values("ts_code").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "baseline", "breadth", "moneyflow", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    parents = {name: json.loads((getattr(args, name) / "manifest.json").read_text(encoding="utf-8"))
               for name in ("breadth", "moneyflow")}
    hashes = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest()}
    for parent in parents.values():
        if parent["status"] != "complete" or any(parent["contract"][key] != value for key, value in hashes.items()):
            raise ValueError("Capital context requires matching complete daily and flow caches.")
    market = pd.read_csv(args.input, float_precision="round_trip")
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    positions = pd.Index(calendar).get_indexer(baseline.trade_date)
    if (positions < 20).any() or not pd.Series(positions).diff().dropna().gt(0).all():
        raise ValueError("Capital baseline requires aligned dates and warmup history.")
    dates = calendar.iloc[positions[0] - 20:positions[-1]].tolist()
    if any(str(date) not in parent["days"] for parent in parents.values() for date in dates):
        raise ValueError("Frozen daily and flow caches do not cover all required dates.")
    sources = ("tools/capital_features.py", "tools/fetch_capital_context.py", "tools/breadth_features.py",
               "tools/moneyflow_features.py", "tools/liquidity_features.py")
    contract = {
        **hashes, "source": "Tushare daily_basic", "documentation": "https://tushare.pro/document/2?doc_id=32",
        "fields": list(FIELDS), "dates": dates, "lag_sessions": LAG_SESSIONS, "features": list(FEATURE_COLUMNS),
        "universe": "historical active Shanghai A-share codes 6xxxxx.SH, not official index constituents",
        "weight": "historical source-day total_mv in ten thousand yuan; not an exact SSE index reconstruction",
        "coverage": "all active stocks require positive cap; matched flows need 98% of stock count, turnover and capitalization",
        "minimum_flow_coverage": MIN_COVERAGE, "large_cap_group": "largest 10% by source-day total_mv; ties ordered by code",
        "availability": "strictly prior domestic session; no filling; incomplete features excluded",
        "delivery_caveat": "historical responses lack original delivery and revision timestamps",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "parents": {name: hashlib.sha256((getattr(args, name) / "manifest.json").read_bytes()).hexdigest() for name in parents},
    }
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["contract"] != contract:
            raise ValueError("Capitalization resume contract changed.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "raw").mkdir()
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "partial", "contract": contract, "days": {}}
        for name in sources:
            path = args.output / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        for name in parents:
            (args.output / f"{name}_manifest.json").write_bytes((getattr(args, name) / "manifest.json").read_bytes())

    def checkpoint():
        path = args.output / "manifest.json.tmp"
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        path.replace(args.output / "manifest.json")

    def summarize(basic, date):
        frames = {}
        for name, parent in parents.items():
            raw = (getattr(args, name) / "raw" / f"{date}.csv.gz").read_bytes()
            if hashlib.sha256(raw).hexdigest() != parent["days"][str(date)]["sha256"]:
                raise ValueError(f"Cached {name} data hash mismatch for {date}.")
            frames[name] = pd.read_csv(io.BytesIO(raw), compression="gzip", float_precision="round_trip")
        return capital_day(basic, frames["breadth"], frames["moneyflow"], date)

    checkpoint()
    rows, pending = {}, []
    for date in dates:
        if str(date) not in manifest["days"]:
            pending.append(date)
            continue
        path = args.output / "raw" / f"{date}.csv.gz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["days"][str(date)]["sha256"]:
            raise ValueError(f"Capitalization cache hash mismatch for {date}.")
        rows[date] = summarize(pd.read_csv(path, float_precision="round_trip"), date)
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    requests = CapitalRequests(get_pro(config.get("Tushare", "令牌")))
    print(json.dumps({"total_days": len(dates), "cached_days": len(rows), "pending_days": len(pending)}), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for start in range(0, len(pending), 20):
            jobs = {pool.submit(requests.fetch, date): date for date in pending[start:start + 20]}
            for job in as_completed(jobs):
                date, basic = jobs[job], job.result()
                received = datetime.now(timezone.utc).isoformat()
                row = summarize(basic, date)
                raw = gzip.compress(basic.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode(), mtime=0)
                (args.output / "raw" / f"{date}.csv.gz").write_bytes(raw)
                manifest["days"][str(date)] = {"rows": len(basic), "received_at": received,
                                               "sha256": hashlib.sha256(raw).hexdigest()}
                rows[date] = row
                checkpoint()
            print(json.dumps({"completed_days": len(rows), "total_days": len(dates)}), flush=True)
    daily = pd.DataFrame([rows[date] for date in dates])
    features = capital_features(baseline.trade_date, daily, calendar)
    path = args.output / "capital.csv"
    daily.to_csv(path, index=False, float_format="%.17g")
    feature_path = args.output / "capital_features.csv"
    features.to_csv(feature_path, index=False, float_format="%.17g")
    manifest.update(status="complete", rows=len(daily), feature_rows=len(features),
                    capital_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    features_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                    unavailable_signal_dates=features.loc[~features.capital_available, "trade_date"].tolist(),
                    completed_at=datetime.now(timezone.utc).isoformat())
    checkpoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
