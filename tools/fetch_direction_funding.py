"""Freeze valuation and financing inputs for causal offline model research."""

from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 数据拉取脚本_tushare import get_pro


REQUESTS = {
    "index_basic": ("index_dailybasic", {"ts_code": "000001.SH", "fields": "trade_date,ts_code,turnover_rate,turnover_rate_f,pe,pe_ttm,pb,total_mv,float_mv"}),
    "margin": ("margin", {"exchange_id": "SSE", "fields": "trade_date,exchange_id,rzye,rzmre,rzche,rqye,rqmcl,rzrqye"}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = pd.read_csv(args.input, float_precision="round_trip")
    dates = pd.to_datetime(inputs.trade_date)
    start = (dates.min() - pd.Timedelta(days=365)).strftime("%Y%m%d")
    end = dates.max().strftime("%Y%m%d")
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    client = get_pro(config.get("Tushare", "令牌"))
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "source": "Tushare",
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "availability_policy": "use strictly previous domestic trading day's record; no same-day funding or valuation input",
                "start": start, "end": end, "assets": {}}
    for name, (api, parameters) in REQUESTS.items():
        try:
            result = client.query(api, start_date=start, end_date=end, **parameters)
        except Exception as exc:
            permission = any(word in str(exc).lower() for word in ("权限", "积分", "permission", "privilege"))
            manifest["assets"][name] = {"status": "unavailable", "reason": "permission" if permission else "request_failed"}
            print(json.dumps({"asset": name, **manifest["assets"][name]}), flush=True)
            continue
        if result is None or result.empty:
            manifest["assets"][name] = {"status": "unavailable", "reason": "empty_response"}
            continue
        if result.trade_date.duplicated().any():
            raise ValueError(f"Duplicate {name} dates.")
        result = result.sort_values("trade_date").reset_index(drop=True)
        path = args.output / f"{name}.csv"
        result.to_csv(path, index=False, float_format="%.17g")
        manifest["assets"][name] = {"status": "available", "api": api, "parameters": parameters,
                                   "rows": len(result), "first_date": str(result.trade_date.iloc[0]),
                                   "last_date": str(result.trade_date.iloc[-1]),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(json.dumps({"asset": name, "rows": len(result), "first_date": str(result.trade_date.iloc[0]),
                          "last_date": str(result.trade_date.iloc[-1])}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["status"] == "available" for item in manifest["assets"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
