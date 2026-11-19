"""Collector regressions: metadata latency must not redefine published inputs."""

from types import SimpleNamespace
import json

import pandas as pd
import pytest

from prediction_service.archive import frame_to_csv_bytes, sha256_bytes
from prediction_service.engine import HistoricalMarketDataDriftError
from prediction_service.model_context import OptionBoundary, extend_model_context, option_boundary_for_snapshot
from tools.option_features import UNDERLYING
from tools.option_position_features import aggregate_positions


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr("prediction_service.option_directory._active_rows", lambda: [])
    monkeypatch.setattr("prediction_service.option_directory._request", lambda _params: {"result": []})
    monkeypatch.setattr("tools.run_downside_shadow.extend_context",
                        lambda context, *_: ({name: frame.copy() for name, frame in context.items()}, {}, []))
    dates = [20261104, 20261105, 20261106, 20261109]
    contracts = pd.DataFrame([
        {"ts_code": term + side, "opt_code": UNDERLYING, "call_put": side,
         "list_date": start, "delist_date": end}
        for term, start, end in [("near", 20261001, 20261105), ("far", 20261001, 20270224),
                                 ("late", 20261104, 20270224), ("new", 20261106, 20270224)]
        for side in ["P", "C"]
    ])
    rows = []
    for i, date in enumerate(dates):
        for contract in contracts.itertuples():
            if contract.list_date <= date <= contract.delist_date:
                rows.append({"ts_code": contract.ts_code, "trade_date": date, "vol": 20 + i,
                             "amount": 100.25 + i, "oi": (200 if contract.call_put == "P" else 300) + 10 * i})
    raw = pd.DataFrame(rows)

    class Client:
        def __init__(self):
            self.catalog = contracts.copy()
            self.raw = raw.copy()
            self.calls = []

        def query(self, api, **params):
            self.calls.append((api, params))
            frame = self.catalog if api == "opt_basic" else self.raw.loc[self.raw.trade_date.eq(int(params["start_date"]))]
            return frame.iloc[params["offset"]:params["offset"] + params["limit"]].copy()

    old_contracts = contracts.loc[contracts.ts_code.str.startswith(("near", "far"))].reset_index(drop=True)
    old_daily = raw.loc[raw.trade_date.eq(dates[0])].reset_index(drop=True)
    context = {"options": aggregate_positions(old_daily, old_contracts, dates[:1])}
    boundary = OptionBoundary("published-snapshot", old_contracts, old_daily, "2026-11-04T10:31:00Z")
    market = pd.DataFrame({"trade_date": pd.to_datetime(pd.Series(dates).astype(str), format="%Y%m%d")})
    return SimpleNamespace(client=Client(), context=context, boundary=boundary, market=market, dates=dates)


def test_late_metadata_preserves_history_and_matches_only_originally_known_contracts(inputs):
    data = inputs
    before = data.context["options"].copy()
    result, raw, receipts = extend_model_context(data.context, data.market, data.client, option_boundary=data.boundary)
    pd.testing.assert_frame_equal(result["options"].iloc[:1], before, check_exact=True)
    pd.testing.assert_frame_equal(data.context["options"], before, check_exact=True)
    appended = result["options"].iloc[1:].reset_index(drop=True)
    expected = aggregate_positions(data.client.raw.loc[data.client.raw.trade_date.gt(data.dates[0])],
                                   data.client.catalog, data.dates[1:])
    pd.testing.assert_frame_equal(appended.iloc[1:].reset_index(drop=True), expected.iloc[1:].reset_index(drop=True), check_exact=True)
    first = appended.iloc[0]
    assert first.contract_rows == 6 and first.matched_contract_rows == 4
    assert first.P_interest == 630 and first.C_interest == 930
    assert first.matched_put_interest_growth == 20 / 400
    assert first.matched_call_interest_growth == 20 / 600
    assert first.matched_interest_imbalance == 0
    assert first.matched_prior_interest_coverage == 1
    assert all(params.get("start_date") != "20261104" for api, params in data.client.calls if api == "opt_daily")
    pd.testing.assert_frame_equal(raw["options_20261104"], data.boundary.daily)
    receipt = next(item for item in receipts if item["name"] == "options_20261104")
    assert receipt["received_at"] == data.boundary.received_at and receipt["source"] == "frozen_archive"


def test_normal_incremental_collection_matches_fixed_full_replay_exactly(inputs):
    data = inputs
    boundary = OptionBoundary("normal", data.client.catalog, data.boundary.daily)
    context = {"options": aggregate_positions(boundary.daily, boundary.contracts, data.dates[:1])}
    result, _, _ = extend_model_context(context, data.market, data.client, option_boundary=boundary)
    expected = aggregate_positions(data.client.raw, data.client.catalog, data.dates)
    pd.testing.assert_frame_equal(result["options"], expected, check_exact=True)


def test_repeated_one_day_extensions_equal_one_multi_day_catchup(inputs):
    data = inputs
    expected, _, _ = extend_model_context(data.context, data.market, data.client, option_boundary=data.boundary)
    context, boundary = data.context, data.boundary
    for i in range(1, len(data.dates)):
        context, raw, _ = extend_model_context(context, data.market.iloc[:i + 1], data.client, option_boundary=boundary)
        boundary = OptionBoundary("next", raw["option_contracts"], raw[f"options_{data.dates[i]}"])
    pd.testing.assert_frame_equal(context["options"], expected["options"], check_exact=True)
    result, raw, receipts = extend_model_context(context, data.market, data.client, option_boundary=boundary)
    pd.testing.assert_frame_equal(result["options"], context["options"], check_exact=True)
    assert raw == {} and receipts == []


def test_unknown_daily_contract_blocks_publication_until_directory_catches_up(inputs):
    data = inputs
    data.client.catalog = data.client.catalog.loc[~data.client.catalog.ts_code.str.startswith("late")]
    with pytest.raises(ValueError, match="directory not ready"):
        extend_model_context(data.context, data.market, data.client, option_boundary=data.boundary)
    assert len(data.context["options"]) == 1


def test_other_underlyings_are_classified_and_excluded(inputs):
    data = inputs
    other = data.client.catalog.iloc[:1].assign(ts_code="other", opt_code="OP510300.SH")
    data.client.catalog = pd.concat([data.client.catalog, other], ignore_index=True)
    daily = data.client.raw.iloc[:1].assign(ts_code="other", trade_date=data.dates[1], oi=999999)
    data.client.raw = pd.concat([data.client.raw, daily], ignore_index=True)
    result, _, _ = extend_model_context(data.context, data.market, data.client, option_boundary=data.boundary)
    assert result["options"].iloc[1].P_interest == 630


@pytest.mark.parametrize("underlying,expected_interest", [("510050", 730), ("510300", 630)])
def test_exchange_directory_recovers_both_model_and_other_new_listings(inputs, monkeypatch, underlying, expected_interest):
    data = inputs
    exchange = []
    for code, right, label in [("10012843", "C", "认购"), ("10012844", "P", "认沽")]:
        exchange.append({"SECURITY_ID": code, "CONTRACT_ID": f"{underlying}{right}2702M03600",
                         "CALL_OR_PUT": label, "START_DATE": "20261105", "END_DATE": "20270224"})
        for date in data.dates[1:]:
            data.client.raw.loc[len(data.client.raw)] = [code + ".SH", date, 10, 100.0, 100]
    monkeypatch.setattr("prediction_service.option_directory._active_rows", lambda: exchange)
    result, raw, receipts = extend_model_context(data.context, data.market, data.client, option_boundary=data.boundary)
    assert result["options"].iloc[1].P_interest == expected_interest
    assert result["options"].iloc[1].matched_contract_rows == 4
    pd.testing.assert_frame_equal(result["options"].iloc[:1], data.context["options"], check_exact=True)
    assert "option_sse_active" in raw
    assert next(r for r in receipts if r["name"] == "option_sse_active")["provider"] == "SSE"


@pytest.mark.parametrize("change", ["daily", "metadata", "missing_metadata", "missing_daily"])
def test_corrupt_boundary_or_conflicting_known_metadata_is_rejected(inputs, change):
    data = inputs
    boundary = data.boundary
    if change == "daily":
        daily = boundary.daily.copy()
        daily.loc[daily.ts_code.eq("nearP"), "oi"] += 1
        boundary = OptionBoundary("corrupt", boundary.contracts, daily)
    elif change == "metadata":
        data.client.catalog.loc[data.client.catalog.ts_code.eq("nearP"), "call_put"] = "C"
    elif change == "missing_metadata":
        data.client.catalog = data.client.catalog.loc[~data.client.catalog.ts_code.eq("nearP")]
        data.client.raw = data.client.raw.loc[~data.client.raw.ts_code.eq("nearP")]
    else:
        data.client.raw = data.client.raw.loc[~(data.client.raw.ts_code.eq("nearP") & data.client.raw.trade_date.eq(data.dates[1]))]
    with pytest.raises((HistoricalMarketDataDriftError, ValueError)):
        extend_model_context(data.context, data.market, data.client, option_boundary=boundary)
    assert len(data.context["options"]) == 1


def test_provider_revisions_cannot_replace_the_archived_boundary(inputs):
    data = inputs
    data.client.raw.loc[data.client.raw.trade_date.eq(data.dates[0]), "oi"] *= 10
    result, _, _ = extend_model_context(data.context, data.market, data.client, option_boundary=data.boundary)
    assert result["options"].iloc[0].P_interest == 400
    assert result["options"].iloc[1].matched_put_interest_growth == 0.05


def test_legacy_boundary_retains_strict_drift_check(inputs):
    with pytest.raises(HistoricalMarketDataDriftError, match="Frozen option boundary changed"):
        extend_model_context(inputs.context, inputs.market, inputs.client)
    inputs.client.catalog = inputs.client.catalog.loc[~inputs.client.catalog.ts_code.str.startswith("late")]
    inputs.client.raw = inputs.client.raw.loc[~inputs.client.raw.ts_code.str.startswith("late")]
    result, _, _ = extend_model_context(inputs.context, inputs.market, inputs.client)
    expected = aggregate_positions(inputs.client.raw, inputs.client.catalog, inputs.dates)
    pd.testing.assert_frame_equal(result["options"], expected, check_exact=True)


def test_boundary_loader_requires_manifest_hashes_and_matching_bundle(tmp_path, inputs):
    data = inputs
    directory = tmp_path / "raw"
    directory.mkdir()
    files = {}
    for name, frame in [("option_contracts.csv", data.boundary.contracts), ("options_20261104.csv", data.boundary.daily)]:
        content = frame_to_csv_bytes(frame)
        (directory / name).write_bytes(content)
        files[f"raw/{name}"] = sha256_bytes(content)
    manifest = {"model_bundle_id": "bundle", "files": files}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    snapshot = SimpleNamespace(id="frozen", manifest_path=str(path), data_as_of="20261104")
    assert option_boundary_for_snapshot(snapshot, "other") is None
    assert option_boundary_for_snapshot(None, "bundle") is None
    boundary = option_boundary_for_snapshot(snapshot, "bundle")
    pd.testing.assert_frame_equal(boundary.daily, data.boundary.daily)
    resolved = frame_to_csv_bytes(data.boundary.contracts)
    (directory / "option_resolved_catalog.csv").write_bytes(resolved)
    files["raw/option_resolved_catalog.csv"] = sha256_bytes(resolved)
    path.write_text(json.dumps(manifest))
    pd.testing.assert_frame_equal(option_boundary_for_snapshot(snapshot, "bundle").catalog, data.boundary.contracts)
    (directory / "option_resolved_catalog.csv").write_bytes(b"corrupt")
    with pytest.raises(HistoricalMarketDataDriftError, match="directory is missing or corrupt"):
        option_boundary_for_snapshot(snapshot, "bundle")
    del files["raw/option_resolved_catalog.csv"]
    path.write_text(json.dumps(manifest))
    (directory / "options_20261104.csv").write_bytes(b"corrupt")
    with pytest.raises(HistoricalMarketDataDriftError, match="missing or corrupt"):
        option_boundary_for_snapshot(snapshot, "bundle")
    files.clear()
    path.write_text(json.dumps(manifest))
    assert option_boundary_for_snapshot(snapshot, "bundle") is None
    files["raw/option_contracts.csv"] = sha256_bytes((directory / "option_contracts.csv").read_bytes())
    path.write_text(json.dumps(manifest))
    with pytest.raises(HistoricalMarketDataDriftError, match="missing or corrupt"):
        option_boundary_for_snapshot(snapshot, "bundle")
