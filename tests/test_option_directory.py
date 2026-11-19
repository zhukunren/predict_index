import json

import pandas as pd
import pytest

from prediction_service import option_directory as directory
from tools.option_features import CONTRACT_FIELDS


def contract(code="10012843", underlying="510300", right="C"):
    return {"SECURITY_ID": code, "CONTRACT_ID": f"{underlying}{right}2611M03600",
            "UNDERLYING_SECURITY_ID": underlying, "CALL_OR_PUT": "认购" if right == "C" else "认沽",
            "START_DATE": "20261118", "END_DATE": "20261125", "ADJUST_DATE": "20261118"}


def resolve(monkeypatch, active, listings=()):
    monkeypatch.setattr(directory, "_active_rows", lambda: active)
    monkeypatch.setattr(directory, "_request", lambda _params: {"result": list(listings)})
    catalog = pd.DataFrame([{"ts_code": "10000001.SH", "opt_code": "OP510050.SH", "call_put": "C",
                             "list_date": "20200101", "delist_date": "20270101"}], columns=CONTRACT_FIELDS)
    observed = pd.DataFrame({"ts_code": ["10012843.SH"], "trade_date": ["20261118"]})
    receipts = []
    result = directory.complete_option_directory(catalog, observed, lambda *args, **kwargs: receipts.append((args, kwargs)))
    pd.testing.assert_frame_equal(result.iloc[:1].reset_index(drop=True), catalog)
    return result, receipts


def test_adds_exchange_metadata_and_keeps_original_provider_rows(monkeypatch):
    result, receipts = resolve(monkeypatch, [contract()])
    assert result.iloc[-1].to_dict() == {"ts_code": "10012843.SH", "opt_code": "OP510300.SH", "call_put": "C",
                                       "list_date": 20261118, "delist_date": 20261125}
    assert receipts[0][1]["provider"] == "SSE" and receipts[0][1]["url"] == directory.SSE_URL


def test_expired_contract_can_be_resolved_from_dated_listing(monkeypatch):
    result, receipts = resolve(monkeypatch, [], [contract()])
    assert len(result) == 2
    assert receipts[-1][0][0] == "option_sse_listing_20261118"


@pytest.mark.parametrize("field,value", [("SECURITY_ID", "bad"), ("CONTRACT_ID", "garbage"),
    ("UNDERLYING_SECURITY_ID", "510050"), ("CALL_OR_PUT", "认沽"), ("START_DATE", "20261119"),
    ("START_DATE", "20260230"), ("END_DATE", "20261117"), ("START_DATE", "")])
def test_ambiguous_or_invalid_exchange_metadata_is_rejected(monkeypatch, field, value):
    row = contract();row[field] = value
    with pytest.raises(ValueError):
        resolve(monkeypatch, [row])


def test_missing_exchange_record_still_blocks_publication(monkeypatch):
    with pytest.raises(ValueError, match="unclassified contracts"):
        resolve(monkeypatch, [])


def test_wrong_listing_date_and_duplicate_identities_are_rejected(monkeypatch):
    row = contract();row["ADJUST_DATE"] = "20261117"
    with pytest.raises(ValueError, match="Unexpected date"):
        resolve(monkeypatch, [], [row])
    with pytest.raises(ValueError, match="Duplicate"):
        resolve(monkeypatch, [contract(), contract()])


def test_no_network_dependency_when_primary_directory_is_complete(monkeypatch):
    monkeypatch.setattr(directory, "_active_rows", lambda: pytest.fail("unnecessary fallback"))
    catalog = pd.DataFrame({"ts_code": ["known"]})
    assert directory.complete_option_directory(catalog, catalog, None) is catalog


def test_hash_verified_archive_metadata_survives_exchange_outage_and_expiry(monkeypatch):
    row = directory._contract(contract())
    cached = pd.DataFrame([row])
    observed = pd.DataFrame({"ts_code": ["10012843.SH"], "trade_date": ["20261124"]})
    primary = cached.iloc[:0]
    monkeypatch.setattr(directory, "_active_rows", lambda: pytest.fail("archive should resolve this contract"))
    receipts = []
    result = directory.complete_option_directory(primary, observed, lambda *a, **kw: receipts.append(kw),
                                                  cached_catalog=cached, source_snapshot_id="published")
    assert result.ts_code.tolist() == ["10012843.SH"]
    assert receipts == [{"provider": "frozen_archive", "source_snapshot_id": "published"}]
    observed["trade_date"] = "20261126"
    with pytest.raises(ValueError, match="Archived option dates"):
        directory.complete_option_directory(primary, observed, None, cached_catalog=cached)


def test_pagination_uses_declared_total_even_when_server_caps_page_size(monkeypatch):
    responses = [{"pageHelp": {"total": 2}, "result": [contract()]},
                 {"pageHelp": {"total": 2}, "result": [contract("10012844", right="P")]}]
    calls = []
    def request(params):
        calls.append(params)
        return responses.pop(0)
    monkeypatch.setattr(directory, "_request", request)
    assert len(directory._active_rows()) == 2
    assert [c["pageHelp.pageNo"] for c in calls] == [1, 2]


@pytest.mark.parametrize("second", [[], [contract()], [contract("10012844", right="P"), contract("10012845")]])
def test_incomplete_duplicate_or_overflow_page_is_rejected(monkeypatch, second):
    responses = [{"pageHelp": {"total": 2}, "result": [contract()]},
                 {"pageHelp": {"total": 2}, "result": second}]
    monkeypatch.setattr(directory, "_request", lambda _params: responses.pop(0))
    with pytest.raises(ValueError):
        directory._active_rows()


@pytest.mark.parametrize("payload", [{"result": [], "fieldErrors": {"sqlId": "error"}}, {}, {"result": "invalid"}, []])
def test_http_and_payload_errors_are_safe_failures(monkeypatch, payload):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, _limit): return json.dumps(payload).encode()
    monkeypatch.setattr(directory, "urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(RuntimeError, match="SSE option directory request failed"):
        directory._request({"sqlId": directory.ACTIVE_QUERY})


def test_network_failure_does_not_invent_contracts(monkeypatch):
    def fail(*_args, **_kwargs):
        raise TimeoutError("network timeout")
    monkeypatch.setattr(directory, "urlopen", fail)
    with pytest.raises(RuntimeError):
        directory._request({})
