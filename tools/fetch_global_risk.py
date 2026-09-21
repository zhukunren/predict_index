"""Freeze US index closes for an isolated direction-bias experiment."""

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

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 数据拉取脚本_tushare import DataRequest, RetryConfig, fetch_tushare_request, get_pro
from tools.global_risk_features import ASSETS, global_risk_features
from tools.evaluate_direction_bias import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    dates = pd.to_datetime(baseline.trade_date.astype(str), format="%Y%m%d")
    start = dates.min() - pd.Timedelta(days=366)
    end = dates.max() - pd.Timedelta(days=1)
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    client = get_pro(config.get("Tushare", "令牌"))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "source": "Tushare index_global",
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "availability": "US session date strictly before Shanghai signal date; daily close only",
        "delivery_caveat": "historical daily closes lack point-in-time delivery and revision timestamps",
        "assets": {},
    }
    assets = {}
    for name, code in ASSETS.items():
        # Keep provider errors out of logs because they can include credentials.
        with redirect_stdout(io.StringIO()):
            result = fetch_tushare_request(
                client, DataRequest(name, "index_global", code), start.strftime("%Y%m%d"), end.strftime("%Y%m%d"),
                retry_config=RetryConfig(retries=2, sleep_seconds=2, rate_limit_sleep_seconds=60, jitter_seconds=0),
            )
        if result.empty:
            raise RuntimeError(f"Overseas history request failed for {name}; credentials omitted.")
        frame = result[["trade_date", "close"]].sort_values("trade_date").reset_index(drop=True)
        frame["trade_date"] = frame.trade_date.dt.strftime("%Y%m%d")
        if frame.trade_date.duplicated().any():
            raise ValueError(f"Duplicate overseas sessions: {name}.")
        sessions = pd.to_datetime(frame.trade_date.astype(str), format="%Y%m%d")
        if (sessions.iloc[0] - start).days > 7 or (end - sessions.iloc[-1]).days > 7 or sessions.diff().dt.days.gt(7).any():
            raise ValueError(f"Truncated or incomplete overseas coverage: {name}.")
        path = args.output / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        assets[name] = frame
        manifest["assets"][name] = {
            "ts_code": code, "rows": len(frame), "first_date": str(frame.trade_date.iloc[0]),
            "last_date": str(frame.trade_date.iloc[-1]), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(json.dumps({"asset": name, **manifest["assets"][name]}), flush=True)
    global_risk_features(baseline.trade_date, assets)
    write_json(args.output / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
