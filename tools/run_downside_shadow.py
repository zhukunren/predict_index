"""Freeze or run the fixed downside candidate; never publish it to the public API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
from filelock import FileLock
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction_service.archive import frame_to_csv_bytes, sha256_bytes, read_features
from prediction_service.config import Settings
from prediction_service.service import PredictionService
from prediction_service.models import ShadowRun
from prediction_service.downside_shadow import existing_run, prediction_window, record_result, prospective_report, verify_run
from tools.fixed_downside_candidate import (
    ALGORITHM, freeze_candidate, load_candidate, read_frame, predict, assert_parity, baseline_predictions,
)
from tools.breadth_features import aggregate_day, breadth_features
from tools.moneyflow_features import aggregate_moneyflow, moneyflow_features
from tools.global_risk_features import ASSETS, global_risk_features
from tools.fetch_market_breadth import DailyRequests
from tools.fetch_moneyflow_context import MoneyflowRequests


def extend_context(context, market, client):
    """Append missing source dates only and preserve each response's receipt time."""
    from prediction_service.models import utcnow
    from 数据拉取脚本_tushare import ApiRateLimiter, DEFAULT_TUSHARE_MIN_INTERVAL_SECONDS

    context = {name: frame.copy() for name, frame in context.items()}
    raw, receipts, daily_cache = {}, [], {}
    daily_requests, money_requests = DailyRequests(client), MoneyflowRequests(client)
    limiter = ApiRateLimiter(DEFAULT_TUSHARE_MIN_INTERVAL_SECONDS)
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)

    def remember(name, frame):
        raw[name] = frame
        receipts.append({"name": name, "received_at": utcnow().isoformat(), "rows": len(frame),
                         "sha256": sha256_bytes(frame_to_csv_bytes(frame))})

    def daily(date):
        if date not in daily_cache:
            frame = daily_requests.fetch(date)
            summary = aggregate_day(frame, date)
            if summary["source_rows"] < 1000 or summary["shanghai_active_rows"] < 500:
                raise ValueError("Incomplete daily stock universe.")
            remember(f"daily_{date}", frame)
            daily_cache[date] = frame
        return daily_cache[date]

    for date in calendar[calendar.gt(context["breadth"].trade_date.max())]:
        row = aggregate_day(daily(int(date)), int(date))
        context["breadth"] = pd.concat([context["breadth"], pd.DataFrame([row])], ignore_index=True)
    for date in calendar[calendar.gt(context["moneyflow"].trade_date.max()) & calendar.lt(calendar.iloc[-1])]:
        flow = money_requests.fetch(int(date))
        remember(f"moneyflow_{date}", flow)
        row = aggregate_moneyflow(flow, daily(int(date)), int(date), price_context=True)
        context["moneyflow"] = pd.concat([context["moneyflow"], pd.DataFrame([row])], ignore_index=True)
    end = (pd.Timestamp(str(calendar.iloc[-1])) - pd.Timedelta(days=1)).strftime("%Y%m%d")
    for name, code in ASSETS.items():
        start = (pd.Timestamp(str(context[name].trade_date.max())) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if start > end:
            continue
        limiter.wait("index_global")
        try:
            frame = client.index_global(ts_code=code, start_date=start, end_date=end, fields="trade_date,close")
        except Exception:
            raise RuntimeError(f"Overseas request failed for {name}; credentials omitted.") from None
        if frame is None:
            raise ValueError(f"Missing overseas response for {name}.")
        remember(f"global_{name}_{end}", frame)
        if not frame.empty:
            frame = frame.loc[:, ["trade_date", "close"]].copy()
            frame["trade_date"] = pd.to_numeric(frame.trade_date, errors="raise").astype(int)
            if not frame.trade_date.between(int(start), int(end)).all():
                raise ValueError("Overseas response contains dates outside the requested interval.")
            context[name] = pd.concat([context[name], frame], ignore_index=True).sort_values("trade_date").reset_index(drop=True)
    return context, raw, receipts


def run(service, bundle, *, fetch=False, allow_backfill=False):
    frozen = load_candidate(bundle)
    publication, snapshot, control = service._load_active_context()
    service._assert_archive_integrity(snapshot, publication)
    service._assert_runtime_release(control)
    prior = existing_run(service, snapshot.id, frozen["release_id"])
    if prior is not None:
        verify_run(service, prior)
        return prior, prospective_report(service, frozen["release_id"], control.id)
    if fetch:
        service.refresh_calendar()
    prediction_window(service.calendar, snapshot.data_as_of, datetime.now(timezone.utc), allow_backfill=allow_backfill)
    market = read_features(snapshot.features_path)
    with service.database.session() as session:
        previous = session.scalar(select(ShadowRun).where(ShadowRun.release_id == frozen["release_id"],
                                                         ShadowRun.status == "succeeded")
                                  .order_by(ShadowRun.created_at.desc()).limit(1))
    if previous is None:
        context = {name: read_frame(bundle / "seed" / f"{name}.csv") for name in ("breadth", "moneyflow", "spx", "nasdaq")}
    else:
        verify_run(service, previous)
        directory = Path(previous.manifest_path).parent
        context = {name: read_frame(directory / "raw" / f"context_{name}.csv")
                   for name in ("breadth", "moneyflow", "spx", "nasdaq")}
    raw, receipts = {}, []
    if fetch:
        from 数据拉取脚本_tushare import get_pro
        context, raw, receipts = extend_context(context, market, get_pro(service._configured_tushare_token()))
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    first_signal = int(read_frame(bundle / "seed" / "baseline.csv").trade_date.iloc[0])
    signals = calendar[calendar.ge(first_signal)]
    breadth_features(signals, context["breadth"], calendar, lag_sessions=0, publication_hour=18)
    moneyflow_features(signals, context["moneyflow"], calendar, price_context=True)
    global_risk_features(signals, {name: context[name] for name in ASSETS})
    print(json.dumps({"stage": "baseline_replay", "signal_date": snapshot.data_as_of}), flush=True)
    baseline = baseline_predictions(market, bundle / "seed")
    print(json.dumps({"stage": "fixed_candidate", "rows": len(baseline)}), flush=True)
    candidate = predict(market, baseline, context)
    assert_parity(read_frame(bundle / "seed" / "candidate_predictions.csv"), candidate)
    # Re-check pinned sources and dependencies after the potentially long calculation.
    load_candidate(bundle)
    completed = record_result(service, snapshot, control, frozen, candidate, baseline, context,
                              raw_frames=raw, receipts=receipts, allow_backfill=allow_backfill)
    return completed, prospective_report(service, frozen["release_id"], control.id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--seed", type=Path, default=ROOT / "artifacts/evaluation/ensemble_moneyflow_price_downside_v1")
    freeze.add_argument("--bundle", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--bundle", type=Path, required=True)
    execute.add_argument("--config", type=Path, default=ROOT / "config.ini")
    execute.add_argument("--fetch", action="store_true")
    execute.add_argument("--allow-backfill", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps({"release_id": freeze_candidate(args.seed, args.bundle), "promotion_allowed": False}))
        return 0
    service = PredictionService(Settings.from_config(args.config))
    try:
        # Reuse the existing schema, without bootstrap publication or admin creation.
        with FileLock(str(service.settings.root_dir / "downside_shadow.lock"), timeout=0):
            result, report = run(service, args.bundle, fetch=args.fetch, allow_backfill=args.allow_backfill)
            print(json.dumps({"run_id": result.id, "result_csv": result.result_csv_path, **report}), flush=True)
    finally:
        service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
