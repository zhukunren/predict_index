"""Freeze share counts for four long-established domestic index ETFs."""

from __future__ import annotations

import argparse
import configparser
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 数据拉取脚本_tushare import get_pro
from tools.etf_features import ASSETS, FIELDS, LAG_SESSIONS, WARMUP, etf_features, normalize_shares


def fetch_year(client, name, start, end):
    try:
        with redirect_stdout(io.StringIO()):
            frame = client.query("fund_share", ts_code=ASSETS[name], start_date=start, end_date=end,
                                 fields=",".join(FIELDS), limit=2000)
    except Exception:
        raise RuntimeError(f"ETF share request failed for {name}; credentials omitted.") from None
    if frame is None or frame.empty or len(frame) > 366:
        raise ValueError(f"Empty or unexpectedly large yearly ETF response for {name}.")
    normalized = normalize_shares(name, frame)
    if not normalized.trade_date.between(int(start), int(end)).all():
        raise ValueError(f"Unexpected ETF response dates for {name}.")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    market = pd.read_csv(args.input, float_precision="round_trip")
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    positions = pd.Index(calendar).get_indexer(baseline.trade_date)
    if (positions < WARMUP).any() or not pd.Series(positions).diff().dropna().gt(0).all():
        raise ValueError("ETF signals must align with the frozen calendar.")
    start, end = int(calendar.iloc[positions[0] - WARMUP]), int(calendar.iloc[positions[-1] - LAG_SESSIONS])
    sources = ("tools/fetch_etf_context.py", "tools/etf_features.py", "tools/liquidity_features.py")
    contract = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "start": start, "end": end, "assets": ASSETS, "source": "Tushare fund_share",
        "documentation": "https://tushare.pro/document/2?doc_id=207",
        "fields": list(FIELDS), "share_unit": "ten thousand shares",
        "selection": "four fixed domestic broad-index ETFs established before 2020; no current-size ranking",
        "lag_sessions": LAG_SESSIONS, "warmup_sessions": WARMUP,
        "interpretation": "share-count changes, not cash flows; corporate actions can also change share counts",
        "availability": "two domestic sessions of delay; incomplete features excluded; no stale filling",
        "delivery_caveat": "historical data lack original publication and revision timestamps",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
    }
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["contract"] != contract:
            raise ValueError("ETF resume contract changed.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "raw").mkdir()
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "partial",
                    "contract": contract, "requests": {}, "assets": {}}
        for name in sources:
            target = args.output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
    def checkpoint():
        temporary = args.output / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output / "manifest.json")
    checkpoint()
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    client = get_pro(config.get("Tushare", "令牌"))
    assets = {}
    for name in ASSETS:
        parts = []
        for year in range(start // 10000, end // 10000 + 1):
            key = f"{name}_{year}"
            path = args.output / "raw" / f"{key}.csv"
            lower, upper = str(max(start, year * 10000 + 101)), str(min(end, year * 10000 + 1231))
            if key in manifest["requests"]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["requests"][key]["sha256"]:
                    raise ValueError(f"ETF cache hash mismatch for {key}.")
                frame = pd.read_csv(path, float_precision="round_trip")
            else:
                frame = fetch_year(client, name, lower, upper)
                frame.to_csv(path, index=False, float_format="%.17g")
                manifest["requests"][key] = {"rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                checkpoint()
                time.sleep(0.5)
            parts.append(frame)
        assets[name] = normalize_shares(name, pd.concat(parts, ignore_index=True))
        path = args.output / f"{name}.csv"
        assets[name].to_csv(path, index=False, float_format="%.17g")
        observed = assets[name].trade_date
        expected = calendar[calendar.between(start, end)]
        manifest["assets"][name] = {
            "rows": len(observed), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "missing_domestic_dates": sorted(set(expected) - set(observed)),
            "excluded_non_domestic_dates": sorted(set(observed) - set(expected)),
        }
        checkpoint()
        print(json.dumps({"asset": name, "rows": len(observed), "missing_sessions": len(set(expected) - set(observed))}), flush=True)
    features = etf_features(baseline.trade_date, assets, calendar)
    path = args.output / "etf_features.csv"
    features.to_csv(path, index=False, float_format="%.17g")
    manifest.update(status="complete", feature_rows=len(features),
                    unavailable_signal_dates=features.loc[~features.etf_available, "trade_date"].tolist(),
                    features_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    completed_at=datetime.now(timezone.utc).isoformat())
    checkpoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
