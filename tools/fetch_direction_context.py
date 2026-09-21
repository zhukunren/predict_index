"""Freeze additional domestic index context without changing production data."""

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

from 数据拉取脚本_tushare import DataRequest, RetryConfig, fetch_tushare_request, get_pro, _normalize_price_frame


REQUESTS = (
    DataRequest("csi300", "index_daily", "000300.SH"),
    DataRequest("csi500", "index_daily", "000905.SH"),
    DataRequest("chinext", "index_daily", "399006.SZ"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = pd.read_csv(args.input, float_precision="round_trip")
    dates = pd.to_datetime(inputs.trade_date)
    start, end = dates.min().strftime("%Y%m%d"), dates.max().strftime("%Y%m%d")
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    client = get_pro(config.get("Tushare", "令牌"))
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "source": "Tushare index_daily",
                "prediction_time": "after domestic index close", "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "start": start, "end": end, "assets": {}}
    for request in REQUESTS:
        with redirect_stdout(io.StringIO()):
            result = fetch_tushare_request(client, request, start, end, retry_config=RetryConfig(retries=1, rate_limit_sleep_seconds=10))
        if result.empty:
            raise RuntimeError(f"No context data for {request.ts_code}; credentials are not logged.")
        normalized = _normalize_price_frame(result)
        if normalized.trade_date.duplicated().any():
            raise ValueError(f"Duplicate dates for {request.ts_code}.")
        missing = sorted(set(dates) - set(normalized.trade_date))
        if missing:
            raise ValueError(f"Incomplete context for {request.ts_code}: {len(missing)} trading dates missing.")
        path = args.output / f"{request.name}.csv"
        result.to_csv(path, index=False, float_format="%.17g")
        manifest["assets"][request.name] = {"ts_code": request.ts_code, "rows": len(result), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(json.dumps({"asset": request.name, "rows": len(result), "missing_dates": 0}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
