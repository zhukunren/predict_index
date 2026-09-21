"""Freeze a small offshore RMB data source for bias research, separately from production."""

import argparse
import configparser
from contextlib import redirect_stderr, redirect_stdout
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
from tools.fx_features import TS_CODE, FIELDS, POLICY, normalize_quotes, fx_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    baseline_bytes = args.baseline.read_bytes()
    baseline = pd.read_csv(io.BytesIO(baseline_bytes), float_precision="round_trip")
    dates = pd.to_datetime(baseline.trade_date.astype(str), format="%Y%m%d")
    start = (dates.min() - pd.Timedelta(days=90)).strftime("%Y%m%d")
    end = (dates.max() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "raw").mkdir()
    sources = ["tools/fx_features.py", "tools/fetch_fx_context.py", "tools/liquidity_features.py", "数据拉取脚本_tushare.py"]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "partial",
                "contract": {"baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(), "start": start, "end": end,
                             "fields": FIELDS, "policy": POLICY,
                             "source_sha256": {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources}},
                "requests": []}
    def checkpoint():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checkpoint()
    for name in sources:
        p = args.output / "source" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT/name).read_bytes())
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    client = get_pro(config.get("Tushare", "令牌"))
    previous = json.loads((args.reuse_from / "manifest.json").read_text()) if args.reuse_from else None
    frames = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        lower, upper = max(start, f"{year}0101"), min(end, f"{year}1231")
        parameters = {"ts_code": TS_CODE, "start_date": lower, "end_date": upper, "fields": ",".join(FIELDS)}
        reusable = next((r for r in previous["requests"] if r["parameters"] == parameters), None) if previous else None
        if reusable is not None:
            raw_bytes = (args.reuse_from / "raw" / f"{year}.csv").read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() != reusable["sha256"]:
                raise ValueError("FX cached response changed.")
            frame = pd.read_csv(io.BytesIO(raw_bytes), float_precision="round_trip")
            receipt = reusable["received_at_utc"]
        else:
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    frame = client.fx_daily(**parameters)
            except Exception:
                manifest.update(status="request_failed", failed_request=parameters)
                checkpoint()
                raise RuntimeError("FX request failed; provider details and credentials omitted.") from None
            receipt = datetime.now(timezone.utc).isoformat()
        if frame is None or frame.empty or len(frame) >= 1000:
            manifest.update(status="incomplete", failed_request=parameters)
            checkpoint()
            raise ValueError("Empty or potentially truncated FX year response.")
        raw = args.output / "raw" / f"{year}.csv"
        if reusable is not None:
            raw.write_bytes(raw_bytes)
        else:
            frame.to_csv(raw, index=False, float_format="%.17g")
        manifest["requests"].append({"parameters": parameters, "received_at_utc": receipt,
                                     "rows": len(frame), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                                     "reused": reusable is not None})
        checkpoint()
        validated = normalize_quotes(frame)
        if not validated.trade_date.between(int(lower), int(upper)).all():
            raise ValueError("FX response contains dates outside the request.")
        frames.append(validated)
        print(json.dumps({"year": year, "rows": len(frame)}), flush=True)
        time.sleep(0.5)
    quotes = normalize_quotes(pd.concat(frames, ignore_index=True))
    observed = pd.to_datetime(quotes.trade_date.astype(str), format="%Y%m%d")
    if (observed.iloc[0] - pd.Timestamp(start)).days > 7 or (pd.Timestamp(end) - observed.iloc[-1]).days > 7:
        raise ValueError("FX requested coverage is incomplete.")
    features = fx_features(baseline.trade_date, quotes)
    quotes.to_csv(args.output / "cnh.csv", index=False, float_format="%.17g")
    features.to_csv(args.output / "features.csv", index=False, float_format="%.17g")
    manifest.update(status="complete", rows=len(quotes), feature_rows=len(features),
                    unused_open_range_anomaly_dates=quotes.loc[quotes.unused_open_range_anomaly, "trade_date"].astype(int).tolist(),
                    first_date=int(quotes.trade_date.iloc[0]), last_date=int(quotes.trade_date.iloc[-1]),
                    cnh_sha256=hashlib.sha256((args.output / "cnh.csv").read_bytes()).hexdigest(),
                    features_sha256=hashlib.sha256((args.output / "features.csv").read_bytes()).hexdigest(),
                    completed_at_utc=datetime.now(timezone.utc).isoformat())
    checkpoint()
    print(json.dumps({k:manifest[k] for k in ("status", "rows", "feature_rows", "first_date", "last_date")}), flush=True)


if __name__ == "__main__":
    main()
