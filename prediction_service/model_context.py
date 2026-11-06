"""Incremental, shared data collection for the four fixed forecast models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .archive import frame_to_csv_bytes, sha256_bytes, sha256_file
from .engine import HistoricalMarketDataDriftError
from .models import utcnow
from tools.option_features import CONTRACT_FIELDS, DAILY_FIELDS, UNDERLYING, validate_contracts
from tools.option_position_features import aggregate_positions
from tools.fetch_option_context import fetch_pages


@dataclass(frozen=True)
class OptionBoundary:
    snapshot_id: str
    contracts: pd.DataFrame
    daily: pd.DataFrame
    received_at: str | None = None


def option_boundary_for_snapshot(snapshot, bundle_id):
    """Read only manifest-covered inputs from the matching published snapshot."""
    if snapshot is None:
        return None
    directory = Path(snapshot.manifest_path).parent
    manifest = json.loads(Path(snapshot.manifest_path).read_text(encoding="utf-8"))
    if manifest.get("model_bundle_id") != bundle_id:
        return None
    names = ["raw/option_contracts.csv", f"raw/options_{snapshot.data_as_of}.csv"]
    files = manifest["files"]
    if not any(name in files for name in names):
        # Original activation/seed snapshots predate incremental raw collection.
        return None
    for name in names:
        if name not in files or not (directory / name).is_file() or sha256_file(directory / name) != files[name]:
            raise HistoricalMarketDataDriftError(f"Frozen option input missing or corrupt: {name}.")
    frames = [pd.read_csv(directory / name, encoding="utf-8-sig", float_precision="round_trip") for name in names]
    receipt = next((item for item in manifest.get("receipts", [])
                    if item["name"] == f"options_{snapshot.data_as_of}"), {})
    return OptionBoundary(snapshot.id, *frames, received_at=receipt.get("received_at"))


def _verify_boundary(frozen, daily, contracts):
    previous = int(frozen.trade_date)
    replay = aggregate_positions(daily, contracts, [previous]).iloc[0]
    # Changes require an earlier session; all other measures must replay exactly.
    for column in replay.index:
        if column.startswith("matched_"):
            continue
        old, actual = frozen[column], replay[column]
        if not ((pd.isna(old) and pd.isna(actual)) or old == actual):
            raise HistoricalMarketDataDriftError(f"Frozen option boundary changed: {previous}, {column}.")


def _set_boundary_changes(appended, previous_daily, previous_contracts, current_daily, contracts):
    """Apply the frozen position formula across two different known universes.

    The fixed research aggregator handles one contract directory per invocation.
    Only the first appended session crosses an archived directory boundary.
    """
    previous = previous_daily.loc[previous_daily.ts_code.isin(previous_contracts.ts_code)]
    previous = previous.copy()
    previous["oi"] = pd.to_numeric(previous.oi, errors="raise")
    previous = previous.merge(previous_contracts[["ts_code", "call_put"]], on="ts_code", validate="many_to_one")
    current = current_daily.loc[current_daily.ts_code.isin(contracts.ts_code)].copy()
    current["oi"] = pd.to_numeric(current.oi, errors="raise")
    before = previous.set_index("ts_code")[["oi", "call_put"]]
    after = current.set_index("ts_code")[["oi"]]
    common = after.index.intersection(before.index)
    matched_before, matched_after = before.loc[common], after.loc[common]
    denominator = float(matched_before.oi.sum())
    appended.loc[0, "matched_contract_rows"] = len(common)
    appended.loc[0, "matched_prior_interest_coverage"] = denominator / float(before.oi.sum())
    if denominator > 0:
        change = matched_after.oi - matched_before.oi
        put = matched_before.call_put.eq("P")
        put_change, call_change = float(change[put].sum()), float(change[~put].sum())
        for name, mask, difference in (("put", put, put_change), ("call", ~put, call_change)):
            prior = float(matched_before.loc[mask, "oi"].sum())
            if prior > 0:
                appended.loc[0, f"matched_{name}_interest_growth"] = difference / prior
        appended.loc[0, "matched_interest_imbalance"] = (put_change - call_change) / denominator


def extend_model_context(context, market, client, *, option_boundary=None):
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

    previous = int(context["options"].trade_date.iloc[-1])
    dates = missing.astype(int).tolist()
    frames = []
    # Reuse the values actually observed at publication, even if the provider
    # later revises prices or adds backdated contract-directory entries.
    for date in dates if option_boundary is not None else [previous, *dates]:
        daily = fetch_pages(client, "opt_daily", {
            "exchange": "SSE", "start_date": str(date), "end_date": str(date),
            "fields": ",".join(DAILY_FIELDS),
        }, page_rows=15000, maximum_rows=150000)
        remember(f"options_{date}", daily)
        frames.append(daily)
    # Fetch after daily data, and classify every observed SSE contract before
    # selecting 50ETF. Otherwise missing metadata silently drops new listings.
    catalog = fetch_pages(client, "opt_basic", {
        "exchange": "SSE", "fields": ",".join(CONTRACT_FIELDS),
    }, page_rows=10000, maximum_rows=200000)
    if not set(CONTRACT_FIELDS).issubset(catalog.columns) or catalog.ts_code.isna().any() or catalog.opt_code.isna().any() or catalog.ts_code.duplicated().any():
        raise ValueError("Incomplete or duplicate SSE option directory.")
    observed = pd.concat(frames, ignore_index=True)
    unknown = sorted(set(observed.ts_code) - set(catalog.ts_code))
    if unknown:
        raise ValueError(f"Option directory not ready; unclassified contracts: {', '.join(unknown[:10])}.")
    remember("option_catalog", catalog)
    contracts = validate_contracts(catalog.loc[catalog.opt_code.eq(UNDERLYING)])
    remember("option_contracts", contracts)
    if option_boundary is None:
        # Legacy seed snapshots have no per-contract archive. Retain the strict
        # old check rather than inventing a boundary from aggregate totals.
        appended = aggregate_positions(observed, contracts, [previous, *dates])
        _verify_boundary(context["options"].iloc[-1], frames[0], contracts)
        appended = appended.iloc[1:].reset_index(drop=True)
    else:
        prior_contracts = validate_contracts(option_boundary.contracts)
        _verify_boundary(context["options"].iloc[-1], option_boundary.daily, prior_contracts)
        try:
            latest = validate_contracts(catalog.loc[catalog.ts_code.isin(prior_contracts.ts_code)])
            pd.testing.assert_frame_equal(prior_contracts, latest, check_exact=True)
        except (ValueError, AssertionError):
            raise HistoricalMarketDataDriftError("Known option contract metadata changed; frozen membership retained.")
        appended = aggregate_positions(observed, contracts, dates)
        _set_boundary_changes(appended, option_boundary.daily, prior_contracts, frames[0], contracts)
        raw[f"options_{previous}"] = option_boundary.daily
        raw["option_boundary_contracts"] = prior_contracts
        receipts.append({"name": f"options_{previous}", "source": "frozen_archive",
                         "source_snapshot_id": option_boundary.snapshot_id,
                         "received_at": option_boundary.received_at,
                         "reused_at": utcnow().isoformat(), "rows": len(option_boundary.daily),
                         "sha256": sha256_bytes(frame_to_csv_bytes(option_boundary.daily))})
    context["options"] = pd.concat([context["options"], appended], ignore_index=True)
    return context, raw, receipts
