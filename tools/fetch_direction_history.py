"""Prepend older market cycles while retaining the frozen production input rows."""

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

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from 数据拉取脚本_tushare import (
    ApiRateLimiter, DataRequest, RetryConfig, fetch_tushare_request,
    get_pro, make_feature_frame, _normalize_price_frame,
)


REQUESTS = (
    DataRequest("000001_sh", "index_daily", "000001.SH"),
    DataRequest("hangseng", "index_global", "HSI"),
    DataRequest("csi300", "index_daily", "000300.SH"),
    DataRequest("csi500", "index_daily", "000905.SH"),
    DataRequest("chinext", "index_daily", "399006.SZ"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--older-raw", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = pd.read_csv(args.input, float_precision="round_trip")
    frozen["trade_date"] = pd.to_datetime(frozen.trade_date)
    first = frozen.trade_date.min()
    if first.year != 2020:
        raise ValueError("This historical extension is declared for the frozen 2020+ input.")
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    client = get_pro(config.get("Tushare", "令牌"))
    limiter = ApiRateLimiter({"index_global": 6.2})
    assets, hashes = {}, {}
    (args.output / "older_raw").mkdir()
    for request in REQUESTS:
        parts = []
        cached = args.older_raw / f"{request.name}.csv" if args.older_raw else None
        if cached is not None and cached.exists():
            parts.append(_normalize_price_frame(pd.read_csv(cached, float_precision="round_trip")))
            print(json.dumps({"asset": request.name, "cached_rows": len(parts[0])}), flush=True)
        for year in (() if parts else range(2010, 2020, 2)):
            start = f"{year}0101"
            end = f"{year + 1}1231" if year < 2018 else first.strftime("%Y%m%d")
            with redirect_stdout(io.StringIO()):
                part = fetch_tushare_request(client, request, start, end,
                    retry_config=RetryConfig(retries=1, rate_limit_sleep_seconds=10), rate_limiter=limiter)
            if part.empty:
                raise RuntimeError(f"Historical data unavailable for {request.name} from {year}; credentials are not logged.")
            expected_start = max(pd.Timestamp(start), pd.Timestamp("2010-06-01") if request.name == "chinext" else pd.Timestamp(start))
            if pd.to_datetime(part.trade_date).min() > expected_start + pd.Timedelta(days=10):
                raise ValueError(f"Possible truncated historical response for {request.name} from {year}.")
            parts.append(part)
            print(json.dumps({"asset": request.name, "start": start, "end": end, "rows": len(part)}), flush=True)
        raw = pd.concat(parts, ignore_index=True).sort_values("trade_date").reset_index(drop=True)
        if raw.trade_date.duplicated().any():
            raise ValueError(f"Duplicate historical dates for {request.name}.")
        assets[request.name] = raw
        path = args.output / "older_raw" / f"{request.name}.csv"
        raw.to_csv(path, index=False, float_format="%.17g")
        hashes[str(path.relative_to(args.output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    older = make_feature_frame(assets)
    boundary = older.loc[older.trade_date.eq(first)].iloc[0]
    original = frozen.loc[frozen.trade_date.eq(first)].iloc[0]
    price_columns = ["open", "high", "low", "close", "pre_close", "vol", "amount"]
    if not np.allclose(boundary[price_columns].to_numpy(dtype=float), original[price_columns].to_numpy(dtype=float), rtol=0, atol=1e-6):
        raise ValueError("Historical fetch disagrees with the frozen boundary market prices.")
    combined = pd.concat([older.loc[older.trade_date.ge("2011-01-01") & older.trade_date.lt(first)].reindex(columns=frozen.columns), frozen], ignore_index=True)
    pd.testing.assert_frame_equal(combined.iloc[-len(frozen):].reset_index(drop=True), frozen, check_dtype=False)
    combined["trade_date"] = combined.trade_date.dt.strftime("%Y-%m-%d")
    combined.to_csv(args.output / "features.csv", index=False, float_format="%.17g")
    for name in ("csi300", "csi500", "chinext"):
        current = _normalize_price_frame(pd.read_csv(args.context / f"{name}.csv", float_precision="round_trip"))
        past = assets[name].loc[assets[name].trade_date.lt(first)]
        raw = pd.concat([past, current], ignore_index=True).sort_values("trade_date").reset_index(drop=True)
        normalized = _normalize_price_frame(raw)
        if normalized.trade_date.duplicated().any() or len(set(pd.to_datetime(combined.trade_date)) - set(normalized.trade_date)):
            raise ValueError(f"Missing or duplicate extended context dates for {name}.")
        raw.to_csv(args.output / f"{name}.csv", index=False, float_format="%.17g")
    for name in ("features.csv", "csi300.csv", "csi500.csv", "chinext.csv"):
        hashes[name] = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "source": "Tushare index_daily and lagged index_global", "start": str(combined.trade_date.iloc[0]),
                "end": str(combined.trade_date.iloc[-1]), "rows": len(combined), "frozen_rows_preserved": len(frozen),
                "boundary_prices_verified": True, "sha256": hashes,
                "purpose": "additional pre-2020 training cycles; production input and released predictions remain immutable"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(combined), "frozen_rows_preserved": len(frozen), "start": manifest["start"]}), flush=True)


if __name__ == "__main__":
    main()
