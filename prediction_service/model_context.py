"""Incremental, shared data collection for the four fixed forecast models."""

from __future__ import annotations

import pandas as pd

from .archive import frame_to_csv_bytes, sha256_bytes
from .models import utcnow
from tools.option_features import CONTRACT_FIELDS, DAILY_FIELDS, UNDERLYING, validate_contracts
from tools.option_position_features import aggregate_positions
from tools.fetch_option_context import fetch_pages


def extend_model_context(context, market, client):
    # Existing collector appends market breadth, prior-session flows and US closes.
    from tools.run_downside_shadow import extend_context

    context, raw, receipts = extend_context(context, market, client)
    sessions = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    missing = sessions[sessions.gt(int(context["options"].trade_date.max()))]
    if missing.empty:
        return context, raw, receipts

    def remember(name, frame):
        raw[name] = frame
        receipts.append({"name": name, "received_at": utcnow().isoformat(),
                         "rows": len(frame), "sha256": sha256_bytes(frame_to_csv_bytes(frame))})

    contracts = fetch_pages(client, "opt_basic", {
        "exchange": "SSE", "opt_code": UNDERLYING, "fields": ",".join(CONTRACT_FIELDS),
    }, page_rows=10000, maximum_rows=50000)
    contracts = validate_contracts(contracts)
    remember("option_contracts", contracts)
    # Matched position changes need the immediately preceding session as well.
    previous = int(context["options"].trade_date.iloc[-1])
    dates = [previous, *missing.astype(int).tolist()]
    frames = []
    for date in dates:
        daily = fetch_pages(client, "opt_daily", {
            "exchange": "SSE", "start_date": str(date), "end_date": str(date),
            "fields": ",".join(DAILY_FIELDS),
        }, page_rows=15000, maximum_rows=150000)
        remember(f"options_{date}", daily)
        frames.append(daily)
    appended = aggregate_positions(pd.concat(frames, ignore_index=True), contracts, dates)
    # A provider revision must not silently alter the boundary used for changes.
    frozen_previous = context["options"].iloc[-1]
    for column in ("P_interest", "C_interest", "P_volume", "C_volume", "P_amount", "C_amount"):
        if float(appended.iloc[0][column]) != float(frozen_previous[column]):
            raise ValueError(f"Frozen option boundary changed: {previous}, {column}.")
    context["options"] = pd.concat([context["options"], appended.iloc[1:]], ignore_index=True)
    return context, raw, receipts
