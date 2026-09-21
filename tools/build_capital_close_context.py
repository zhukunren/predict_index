"""Freeze current-day capitalization prices using verified historical raw inputs."""

from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.capital_close_features import capital_price_day, capital_close_features
from tools.capital_features import FEATURE_COLUMNS
from tools.fetch_capital_context import CapitalRequests


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(args):
    parent = read_json(args.capital / "manifest.json")
    breadth = read_json(args.breadth / "manifest.json")
    audit = read_json(args.capital / "verification.json")
    hashes = {"input_sha256": digest(args.input), "baseline_sha256": digest(args.baseline)}
    if (not audit["passed"] or audit["context_manifest_sha256"] != digest(args.capital / "manifest.json")
            or any(item["status"] != "complete" or any(item["contract"][k] != v for k, v in hashes.items())
                   for item in (parent, breadth))
            or parent["capital_sha256"] != digest(args.capital / "capital.csv")
            or parent["contract"]["parents"]["breadth"] != digest(args.breadth / "manifest.json")):
        raise ValueError("Same-day prices require audited, matching parent snapshots.")
    for name, expected in parent["contract"]["source_sha256"].items():
        if digest(ROOT / name) != expected or digest(args.capital / "source" / name) != expected:
            raise ValueError("Capital parent sources changed.")
    market = pd.read_csv(args.input, float_precision="round_trip")
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    positions = pd.Index(calendar).get_indexer(baseline.trade_date)
    if (positions < 20).any() or not pd.Series(positions).diff().dropna().gt(0).all():
        raise ValueError("Capital price signals need aligned dates and warmup history.")
    dates = calendar.iloc[positions[0] - 20:positions[-1] + 1].tolist()
    if any(str(date) not in breadth["days"] for date in dates):
        raise ValueError("Daily prices do not cover the complete signal calendar.")
    sources = dict(parent["contract"]["source_sha256"])
    sources.update({name: digest(ROOT / name) for name in ("tools/capital_close_features.py", "tools/build_capital_close_context.py")})
    contract = {**hashes, "dates": dates, "source_sha256": sources, "features": list(FEATURE_COLUMNS),
                "lag_sessions": 1, "price_lag_sessions": 0, "publication_hour": 18,
                "parents": {"capital": digest(args.capital / "manifest.json"),
                            "capital_audit": digest(args.capital / "verification.json"),
                            "breadth": digest(args.breadth / "manifest.json")},
                "price_universe": "all historical active Shanghai A-shares with complete positive total_mv",
                "delivery_caveat": "18:00 research assumption based on daily_basic 15:00-17:00 update documentation; no original delivery times"}
    if args.verify:
        manifest = read_json(args.output / "manifest.json")
        if manifest["status"] != "complete" or manifest["contract"] != contract:
            raise ValueError("Capital close verification contract changed.")
        for name, expected in sources.items():
            if digest(args.output / "source" / name) != expected:
                raise ValueError("Capital close frozen sources changed.")
        for key, filename in (("capital_sha256", "capital.csv"), ("prices_sha256", "prices.csv"),
                              ("features_sha256", "capital_features.csv")):
            if digest(args.output / filename) != manifest[key]:
                raise ValueError("Capital close aggregate or feature checksum failed.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "raw").mkdir()
        (args.output / "capital.csv").write_bytes((args.capital / "capital.csv").read_bytes())
        for name in sources:
            path = args.output / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        manifest = {"status": "partial", "created_at": datetime.now(timezone.utc).isoformat(),
                    "contract": contract, "extra_days": {}, "capital_sha256": parent["capital_sha256"]}
        write_json(args.output / "manifest.json", manifest)
    requests = None
    rows = []
    for index, date in enumerate(dates):
        key = str(date)
        daily_path = args.breadth / "raw" / f"{date}.csv.gz"
        if digest(daily_path) != breadth["days"][key]["sha256"]:
            raise ValueError(f"Daily price checksum failed for {date}.")
        if key in parent["days"]:
            path = args.capital / "raw" / f"{date}.csv.gz"
            expected = parent["days"][key]["sha256"]
        else:
            path = args.output / "raw" / f"{date}.csv.gz"
            if not args.verify:
                if requests is None:
                    from 数据拉取脚本_tushare import get_pro
                    config = configparser.ConfigParser(interpolation=None)
                    config.read(args.config, encoding="utf-8")
                    requests = CapitalRequests(get_pro(config.get("Tushare", "令牌")))
                frame = requests.fetch(date)
                received = datetime.now(timezone.utc).isoformat()
                capital_price_day(frame, pd.read_csv(daily_path, float_precision="round_trip"), date)
                raw = gzip.compress(frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode(), mtime=0)
                path.write_bytes(raw)
                manifest["extra_days"][key] = {"sha256": digest(path), "rows": len(frame), "received_at": received}
                write_json(args.output / "manifest.json", manifest)
            expected = manifest["extra_days"][key]["sha256"]
        if digest(path) != expected:
            raise ValueError(f"Capitalization checksum failed for {date}.")
        rows.append(capital_price_day(pd.read_csv(path, float_precision="round_trip"),
                                      pd.read_csv(daily_path, float_precision="round_trip"), date))
        if (index + 1) % 200 == 0:
            print(json.dumps({"verified_price_days": index + 1, "total_days": len(dates)}), flush=True)
    prices = pd.DataFrame(rows)
    lagged = pd.read_csv(args.capital / "capital.csv", float_precision="round_trip")
    features = capital_close_features(baseline.trade_date, lagged, prices, calendar)
    if args.verify:
        for frame, filename in ((prices, "prices.csv"), (features, "capital_features.csv")):
            pd.testing.assert_frame_equal(frame, pd.read_csv(args.output / filename, float_precision="round_trip"),
                                          check_exact=True, check_dtype=False)
        probes = []
        for cutoff in (20221230, 20241231, int(baseline.trade_date.iloc[-1])):
            signals = baseline.trade_date[baseline.trade_date.le(cutoff)]
            prefix = capital_close_features(signals, lagged.loc[lagged.trade_date.lt(cutoff)],
                                            prices.loc[prices.trade_date.le(cutoff)], calendar[calendar.le(cutoff)])
            pd.testing.assert_frame_equal(features.loc[features.trade_date.le(cutoff)].reset_index(drop=True), prefix,
                                          check_exact=True, check_dtype=False)
            probes.append({"signal_date": cutoff, "rows": len(prefix), "passed": True})
        report = {"passed": True, "verified_at": datetime.now(timezone.utc).isoformat(), "prefixes": probes,
                  "source_days": len(prices), "feature_rows": len(features),
                  "unavailable_rows": int((~features.capital_available).sum()),
                  "context_manifest_sha256": digest(args.output / "manifest.json")}
        with (args.output / "verification.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print(json.dumps(report), flush=True)
    else:
        for frame, filename in ((prices, "prices.csv"), (features, "capital_features.csv")):
            frame.to_csv(args.output / filename, index=False, float_format="%.17g")
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                        prices_sha256=digest(args.output / "prices.csv"), features_sha256=digest(args.output / "capital_features.csv"),
                        feature_rows=len(features), unavailable_signal_dates=features.loc[~features.capital_available, "trade_date"].tolist())
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps({"complete": True, "source_days": len(prices), "feature_rows": len(features),
                          "extra_days": list(manifest["extra_days"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capital", "breadth", "input", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--verify", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
