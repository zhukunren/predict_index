from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tools.breadth_features import GROUPS, MEASURES, aggregate_day, breadth_features
from tools import fetch_market_breadth as fetcher


def _stocks():
    return pd.DataFrame({
        "ts_code": ["600000.SH", "600001.SH", "000001.SZ", "000002.SZ"],
        "trade_date": "20240102", "pct_chg": [1.0, -3.0, 0.0, 5.0],
        "amount": [10.0, 30.0, 60.0, 0.0], "vol": [1.0, 2.0, 3.0, 0.0],
    })


def test_breadth_uses_traded_universe_with_correct_percent_units_and_weights():
    result = aggregate_day(_stocks(), 20240102)
    assert result["source_rows"] == 4
    assert result["all_active_rows"] == 3
    assert result["shanghai_active_rows"] == 2
    assert result["all_up_fraction"] == pytest.approx(1 / 3)
    assert result["all_down_fraction"] == pytest.approx(1 / 3)
    assert result["all_mean_return"] == pytest.approx(-0.02 / 3)
    assert result["all_median_return"] == 0.0
    assert result["shanghai_median_return"] == pytest.approx(-0.01)
    assert result["all_down_amount_fraction"] == pytest.approx(0.3)
    assert result["shanghai_down_amount_fraction"] == pytest.approx(0.75)
    assert result["all_large_down_fraction"] == pytest.approx(1 / 3)


def test_bj_coded_pre_exchange_history_does_not_enter_sh_sz_breadth():
    stocks = _stocks()
    otc = pd.DataFrame({"ts_code": ["920527.BJ"], "trade_date": ["20240102"],
                        "pct_chg": [np.nan], "amount": [7.11], "vol": [16.0]})
    result = aggregate_day(pd.concat([stocks, otc], ignore_index=True), 20240102)
    expected = aggregate_day(stocks, 20240102)
    assert result["source_rows"] == 5
    assert result["excluded_exchange_rows"] == 1
    for key in expected:
        if key not in ("source_rows", "excluded_exchange_rows"):
            assert result[key] == expected[key]


@pytest.mark.parametrize("kind", ["date", "duplicate", "nan", "negative_volume"])
def test_bad_stock_observations_are_rejected(kind):
    stocks = _stocks()
    if kind == "date":
        stocks.loc[0, "trade_date"] = "20240103"
    elif kind == "duplicate":
        stocks.loc[0, "ts_code"] = stocks.ts_code.iloc[1]
    elif kind == "nan":
        stocks.loc[0, "pct_chg"] = np.nan
    else:
        stocks.loc[0, "vol"] = -1
    with pytest.raises(ValueError):
        aggregate_day(stocks, 20240102)


def _breadth():
    dates = pd.bdate_range("2023-01-03", periods=60).strftime("%Y%m%d").astype(int)
    data = pd.DataFrame({"trade_date": dates})
    for group in GROUPS:
        for measure in MEASURES:
            data[f"{group}_{measure}"] = 0.2 if "fraction" in measure else 0.001
        data[f"{group}_up_fraction"] = 0.4 + np.arange(60) / 1000
    return dates, data


def test_breadth_uses_prior_session_and_has_exact_prefix_replay():
    dates, data = _breadth()
    full = breadth_features(dates[25:], data, dates)
    assert full.breadth_source_date.iloc[0] == dates[24]
    assert full.breadth_all_up_fraction.iloc[0] == data.all_up_fraction.iloc[24]
    prefix = breadth_features(dates[25:41], data.iloc[:40], dates[:41])
    pd.testing.assert_frame_equal(full.iloc[:16], prefix, check_exact=True)
    changed = data.copy()
    changed.loc[40:, "all_up_fraction"] = 0.8
    altered = breadth_features(dates[25:], changed, dates)
    pd.testing.assert_frame_equal(full.iloc[:16], altered.iloc[:16], check_exact=True)
    assert not full.iloc[16:].equals(altered.iloc[16:])


def test_breadth_rejects_missing_sessions_and_incomplete_warmup():
    dates, data = _breadth()
    with pytest.raises(ValueError, match="Missing lagged breadth"):
        breadth_features(dates[25:], data.drop(index=30), dates)
    with pytest.raises(ValueError, match="Missing lagged breadth"):
        breadth_features(dates[10:], data, dates)


def test_same_day_breadth_requires_after_ingestion_publication_and_excludes_future():
    dates, data = _breadth()
    with pytest.raises(ValueError, match="publication after"):
        breadth_features(dates[25:], data, dates, lag_sessions=0, publication_hour=15)
    full = breadth_features(dates[25:], data, dates, lag_sessions=0, publication_hour=18)
    assert full.breadth_source_date.iloc[0] == dates[25]
    assert full.breadth_all_up_fraction.iloc[0] == data.all_up_fraction.iloc[25]
    prefix = breadth_features(dates[25:41], data.iloc[:41], dates[:41], lag_sessions=0)
    pd.testing.assert_frame_equal(full.iloc[:16], prefix, check_exact=True)
    data.loc[41:, "all_up_fraction"] = 0.9
    changed = breadth_features(dates[25:], data, dates, lag_sessions=0)
    pd.testing.assert_frame_equal(full.iloc[:16], changed.iloc[:16], check_exact=True)


def test_daily_fetch_paginates_without_losing_stock_observations(monkeypatch):
    monkeypatch.setattr(fetcher, "PAGE_ROWS", 2)
    monkeypatch.setattr(fetcher, "REQUEST_INTERVAL", 0)
    stocks = _stocks()
    class Client:
        def __init__(self):
            self.offsets = []

        def daily(self, **kwargs):
            self.offsets.append(kwargs["offset"])
            return stocks.iloc[kwargs["offset"]:kwargs["offset"] + kwargs["limit"]].copy()
    client = Client()
    result = fetcher.DailyRequests(client).fetch(20240102)
    assert client.offsets == [0, 2, 4]
    pd.testing.assert_frame_equal(result, stocks.sort_values("ts_code").reset_index(drop=True))


@pytest.mark.parametrize("corrupt", [False, True])
def test_raw_cache_reuse_verifies_hash_and_preserves_source(tmp_path, corrupt):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "raw").mkdir(parents=True)
    (output / "raw").mkdir(parents=True)
    raw = gzip.compress(_stocks().to_csv(index=False).encode(), mtime=0)
    path = source / "raw" / "20240102.csv.gz"
    path.write_bytes(raw)
    contract = {"input_sha256": "market", "baseline_sha256": "baseline", "fields": list(fetcher.REQUIRED_COLUMNS),
                "universe": "SH/SZ", "source": "Tushare daily"}
    parent = {"status": "complete", "contract": contract,
              "days": {"20240102": {"rows": 4, "sha256": hashlib.sha256(raw).hexdigest()}}}
    (source / "manifest.json").write_text(json.dumps(parent), encoding="utf-8")
    if corrupt:
        path.write_bytes(raw + b"altered")
        with pytest.raises(ValueError, match="hash mismatch"):
            fetcher.reuse_raw_cache(source, output, [20240102, 20240103], contract)
        assert not (output / "raw" / path.name).exists()
    else:
        days, _ = fetcher.reuse_raw_cache(source, output, [20240102, 20240103], contract)
        assert list(days) == ["20240102"]
        assert path.read_bytes() == raw
        assert (output / "raw" / path.name).read_bytes() == raw
