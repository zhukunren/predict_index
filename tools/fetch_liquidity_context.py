"""Freeze SHIBOR and actual exchange repo history for offline research."""

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

from 数据拉取脚本_tushare import get_pro
from tools.liquidity_features import REQUESTS, WARMUP, liquidity_features, normalize_asset


def fetch_year(client, name, start, end):
    api, parameters, columns = REQUESTS[name]
    try:
        with redirect_stdout(io.StringIO()):
            frame = client.query(api, start_date=start, end_date=end, fields=",".join(columns), **parameters)
    except Exception:
        raise RuntimeError(f"Liquidity request failed for {name}; credentials omitted.") from None
    if frame is None or frame.empty or len(frame) >= 1000:
        raise ValueError(f"Empty or potentially truncated yearly liquidity response for {name}.")
    validated = normalize_asset(name, frame)
    if not validated.trade_date.between(int(start), int(end)).all():
        raise ValueError(f"Unexpected liquidity response dates for {name}.")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    market = pd.read_csv(args.input, float_precision="round_trip")
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    positions = pd.Index(calendar).get_indexer(baseline.trade_date)
    if (positions < WARMUP).any() or not pd.Series(positions).diff().dropna().gt(0).all():
        raise ValueError("Liquidity history must align with the frozen calendar.")
    start, end = int(calendar.iloc[positions[0] - WARMUP]), int(calendar.iloc[positions[-1] - 1])
    sources = ("tools/fetch_liquidity_context.py", "tools/liquidity_features.py")
    contract = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "start": start, "end": end, "requests": REQUESTS,
        "source": "Tushare shibor and repo_daily",
        "availability": "strictly previous domestic session; incomplete features marked unavailable; no stale filling",
        "delivery_caveat": "historical responses lack original publication and revision timestamps",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
    }
    contract = json.loads(json.dumps(contract))
    reused = None
    if args.reuse_from is not None:
        reused = json.loads((args.reuse_from / "manifest.json").read_text(encoding="utf-8"))
        for key in ("input_sha256", "baseline_sha256", "start", "end", "requests"):
            if reused["contract"][key] != contract[key]:
                raise ValueError(f"Liquidity reuse requires matching {key}.")
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["contract"] != contract:
            raise ValueError("Liquidity resume contract changed.")
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
    for name in REQUESTS:
        parts = []
        for year in range(start // 10000, end // 10000 + 1):
            key = f"{name}_{year}"
            path = args.output / "raw" / f"{key}.csv"
            lower, upper = str(max(start, year * 10000 + 101)), str(min(end, year * 10000 + 1231))
            if key in manifest["requests"]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["requests"][key]["sha256"]:
                    raise ValueError(f"Liquidity cache hash mismatch for {key}.")
                frame = pd.read_csv(path, float_precision="round_trip")
            else:
                if reused is not None and key in reused["requests"]:
                    raw = (args.reuse_from / "raw" / f"{key}.csv").read_bytes()
                    if hashlib.sha256(raw).hexdigest() != reused["requests"][key]["sha256"]:
                        raise ValueError(f"Liquidity reuse hash mismatch for {key}.")
                    path.write_bytes(raw)
                    frame = pd.read_csv(path, float_precision="round_trip")
                else:
                    frame = fetch_year(client, name, lower, upper)
                    frame.to_csv(path, index=False, float_format="%.17g")
                manifest["requests"][key] = {"rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                checkpoint()
            parts.append(frame)
        assets[name] = normalize_asset(name, pd.concat(parts, ignore_index=True))
        path = args.output / f"{name}.csv"
        # Preserve the provider field name for the common normalizer.
        output = assets[name].rename(columns={"trade_date": "date"}) if name == "shibor" else assets[name]
        output.to_csv(path, index=False, float_format="%.17g")
        assets[name] = output
        observed = normalize_asset(name, output).trade_date
        expected = calendar[calendar.between(start, end)]
        manifest["assets"][name] = {"rows": len(output), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "missing_domestic_dates": sorted(set(expected) - set(observed)),
                                   "excluded_non_domestic_dates": sorted(set(observed) - set(expected))}
        checkpoint()
        print(json.dumps({"asset": name, "rows": len(output), "start": start, "end": end}), flush=True)
    features = liquidity_features(baseline.trade_date, assets, calendar, allow_missing=True)
    path = args.output / "liquidity_features.csv"
    features.to_csv(path, index=False, float_format="%.17g")
    manifest.update(status="complete", feature_rows=len(features),
                    unavailable_signal_dates=features.loc[~features.liquidity_available, "trade_date"].tolist(),
                    features_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    completed_at=datetime.now(timezone.utc).isoformat())
    checkpoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
