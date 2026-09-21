"""Freeze full historical 50ETF option membership and daily activity."""

from __future__ import annotations

import argparse
import configparser
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gzip
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
from tools.evaluate_direction_bias import write_json
from tools.option_features import CONTRACT_FIELDS, DAILY_FIELDS, UNDERLYING, aggregate_options, option_features, validate_contracts


def fetch_pages(client, api, parameters, *, page_rows, maximum_rows):
    parts = []
    for offset in range(0, maximum_rows, page_rows):
        try:
            with redirect_stdout(io.StringIO()):
                part = client.query(api, **parameters, limit=page_rows, offset=offset)
        except Exception:
            raise RuntimeError(f"Option request failed for {api}; credentials omitted. Resume later.") from None
        if part is None or part.empty:
            break
        parts.append(part)
        if len(part) < page_rows:
            break
        time.sleep(0.65)
    else:
        raise ValueError("Option pagination exceeded its declared bound.")
    if not parts:
        raise ValueError(f"Empty option response for {api}.")
    frame = pd.concat(parts, ignore_index=True)
    keys = ["ts_code", "trade_date"] if api == "opt_daily" else ["ts_code"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate option rows from pagination.")
    if api == "opt_daily" and not frame.trade_date.astype(str).between(parameters["start_date"], parameters["end_date"]).all():
        raise ValueError("Unexpected option dates from provider.")
    return frame.sort_values(keys).reset_index(drop=True)


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
    if (positions < 20).any() or not pd.Series(positions).diff().dropna().gt(0).all():
        raise ValueError("Option signals require aligned market dates and 20 warmup days.")
    dates = calendar.iloc[positions[0] - 20:positions[-1]].tolist()
    source_files = ("tools/option_features.py", "tools/fetch_option_context.py")
    contract = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "underlying": UNDERLYING, "dates": dates, "contract_fields": list(CONTRACT_FIELDS), "daily_fields": list(DAILY_FIELDS),
        "source": "Tushare opt_basic and opt_daily, historical 50ETF contract membership",
        "availability": "strictly previous domestic session, no same-day option activity",
        "delivery_caveat": "historical activity has no original delivery or revision timestamps; adjusted strikes and multipliers excluded",
        "documentation": ["https://tushare.pro/document/2?doc_id=158", "https://tushare.pro/document/2?doc_id=159"],
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
    }
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["contract"] != contract:
            raise ValueError("Option resume contract changed.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "raw").mkdir()
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "partial", "contract": contract, "months": {}}
        for name in source_files:
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
    client = get_pro(config.get("Tushare", "令牌"))
    metadata_path = args.output / "contracts.csv"
    if "contracts_sha256" in manifest:
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != manifest["contracts_sha256"]:
            raise ValueError("Option metadata cache hash mismatch.")
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = fetch_pages(client, "opt_basic", {"exchange": "SSE", "opt_code": UNDERLYING,
                                                    "fields": ",".join(CONTRACT_FIELDS)}, page_rows=10000, maximum_rows=50000)
        validate_contracts(metadata)
        metadata.to_csv(metadata_path, index=False)
        manifest.update(contracts_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(), contract_rows=len(metadata))
        checkpoint()
    metadata = validate_contracts(metadata)
    parts = []
    periods = pd.period_range(pd.to_datetime(str(dates[0])), pd.to_datetime(str(dates[-1])), freq="M")
    for period in periods:
        key = str(period)
        path = args.output / "raw" / f"{key}.csv.gz"
        month_dates = [date for date in dates if str(date).startswith(period.strftime("%Y%m"))]
        if key in manifest["months"]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["months"][key]["sha256"]:
                raise ValueError(f"Option daily cache hash mismatch for {key}.")
            frame = pd.read_csv(path, dtype={"trade_date": str}, float_precision="round_trip")
        else:
            frame = fetch_pages(client, "opt_daily", {"exchange": "SSE", "start_date": str(month_dates[0]),
                                                     "end_date": str(month_dates[-1]), "fields": ",".join(DAILY_FIELDS)},
                                page_rows=15000, maximum_rows=150000)
            raw = gzip.compress(frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8"), mtime=0)
            path.write_bytes(raw)
            manifest["months"][key] = {"rows": len(frame), "sha256": hashlib.sha256(raw).hexdigest()}
            checkpoint()
        parts.append(aggregate_options(frame, metadata, month_dates))
        if len(parts) % 6 == 0 or len(parts) == len(periods):
            print(json.dumps({"completed_months": len(parts), "total_months": len(periods), "through_month": key}), flush=True)
    daily = pd.concat(parts, ignore_index=True)
    option_features(baseline.trade_date, daily, calendar)
    path = args.output / "options.csv"
    daily.to_csv(path, index=False, float_format="%.17g")
    manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), rows=len(daily),
                    options_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    checkpoint()
    print(json.dumps({"status": "complete", "days": len(daily), "option_observations": int(daily.contract_rows.sum())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
