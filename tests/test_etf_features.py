from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.etf_features import ASSETS, etf_features, normalize_shares
from tools.fetch_etf_context import fetch_year


def inputs():
    dates = pd.Series(pd.bdate_range("2023-01-02", periods=90).strftime("%Y%m%d").astype(int))
    assets = {name: pd.DataFrame({"trade_date": dates, "ts_code": code, "fd_share": 100 * 1.01 ** np.arange(len(dates))})
              for name, code in ASSETS.items()}
    return dates, assets


def test_shares_are_scale_invariant_and_lagged_two_sessions():
    dates, assets = inputs()
    full = etf_features(dates.iloc[22:], assets, dates)
    assert full.etf_source_date.iloc[0] == dates.iloc[20]
    assert full.etf_available.all()
    np.testing.assert_allclose(full.etf_sse50_share_change_1, 0.01)
    np.testing.assert_allclose(full.etf_sse50_share_change_5, 1.01 ** 5 - 1)
    np.testing.assert_allclose(full.etf_sse50_share_change_mean_20, 0.01)
    for frame in assets.values():
        frame["fd_share"] *= 10000
    scaled = etf_features(dates.iloc[22:], assets, dates)
    pd.testing.assert_frame_equal(full, scaled)


def test_full_history_matches_prefix_and_future_data_cannot_change_past():
    dates, assets = inputs()
    full = etf_features(dates.iloc[22:], assets, dates)
    prefix_assets = {name: frame.iloc[:49] for name, frame in assets.items()}
    prefix = etf_features(dates.iloc[22:51], prefix_assets, dates.iloc[:51])
    pd.testing.assert_frame_equal(full.iloc[:29], prefix, check_exact=True)
    for frame in assets.values():
        frame.loc[49:, "fd_share"] *= 10
    changed = etf_features(dates.iloc[22:], assets, dates)
    pd.testing.assert_frame_equal(full.iloc[:29], changed.iloc[:29], check_exact=True)


def test_missing_session_is_not_an_unchanged_share_count():
    dates, assets = inputs()
    assets["sse50"] = assets["sse50"].drop(index=35)
    full = etf_features(dates.iloc[22:], assets, dates)
    assert full.loc[~full.etf_available, "trade_date"].tolist() == dates.iloc[37:58].tolist()
    assert full.loc[full.trade_date.eq(dates.iloc[37]), "etf_sse50_share_change_1"].isna().all()


def test_non_session_disclosures_do_not_enter_trading_day_windows():
    dates, assets = inputs()
    original = etf_features(dates.iloc[22:], assets, dates)
    for name, frame in assets.items():
        extra = frame.iloc[:1].assign(trade_date=20230107, fd_share=1e10)
        assets[name] = pd.concat([frame, extra], ignore_index=True)
    result = etf_features(dates.iloc[22:], assets, dates)
    pd.testing.assert_frame_equal(result, original, check_exact=True)


@pytest.mark.parametrize("problem", ["duplicate", "instrument", "negative", "nonfinite"])
def test_invalid_observations_are_rejected(problem):
    _, assets = inputs()
    frame = assets["sse50"].copy()
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "instrument":
        frame.loc[0, "ts_code"] = "000001.SH"
    elif problem == "negative":
        frame.loc[0, "fd_share"] = -1
    else:
        frame.loc[0, "fd_share"] = np.nan
    with pytest.raises(ValueError):
        normalize_shares("sse50", frame)


def test_share_request_contract_and_error_sanitizing():
    dates, assets = inputs()
    class Client:
        def query(self, api, **kwargs):
            assert api == "fund_share"
            assert kwargs["ts_code"] == ASSETS["sse50"]
            assert kwargs["limit"] == 2000
            return assets["sse50"]
    result = fetch_year(Client(), "sse50", str(dates.iloc[0]), str(dates.iloc[-1]))
    assert len(result) == len(dates)
    with pytest.raises(ValueError, match="dates"):
        fetch_year(Client(), "sse50", str(dates.iloc[10]), str(dates.iloc[-1]))
    class Failed:
        def query(self, *args, **kwargs):
            raise RuntimeError("SECRET_TOKEN")
    with pytest.raises(RuntimeError, match="credentials omitted") as failure:
        fetch_year(Failed(), "sse50", str(dates.iloc[0]), str(dates.iloc[-1]))
    assert "SECRET_TOKEN" not in str(failure.value)
