from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.funding_features import funding_features


def _assets():
    dates = pd.bdate_range("2023-01-03", periods=140)
    positions = np.arange(len(dates), dtype=float)
    return dates, {
        "index_basic": pd.DataFrame({
            "trade_date": dates.strftime("%Y%m%d"), "turnover_rate": 0.5 + positions / 1000,
            "turnover_rate_f": 1 + positions / 1000, "pe_ttm": 12 + np.sin(positions),
            "pb": 1.2, "float_mv": 1e12,
        }),
        "margin": pd.DataFrame({
            "trade_date": dates.strftime("%Y%m%d"), "rzye": 1e9 + positions * 1e6,
            "rzmre": 1e7 + positions * 1e4, "rzche": 9e6 + positions * 1e4,
            "rqye": 1e7, "rzrqye": 1.01e9 + positions * 1e6,
        }),
    }


def test_funding_uses_only_previous_session_and_has_prefix_parity():
    dates, assets = _assets()
    full = funding_features(dates[65:], assets)
    assert full.funding_source_date.iloc[0] == int(dates[64].strftime("%Y%m%d"))
    prefix_assets = {name: frame.iloc[:100].copy() for name, frame in assets.items()}
    prefix = funding_features(dates[65:101], prefix_assets)
    pd.testing.assert_frame_equal(full.iloc[:36], prefix)
    assets["margin"].loc[100:, ["rzye", "rzmre", "rzche"]] *= 1.8
    assets["index_basic"].loc[100:, ["turnover_rate", "turnover_rate_f", "pe_ttm"]] *= 2
    changed = funding_features(dates[65:], assets)
    pd.testing.assert_frame_equal(full.iloc[:36], changed.iloc[:36])
    assert not full.iloc[36:].equals(changed.iloc[36:])


@pytest.mark.parametrize("name", ["index_basic", "margin"])
def test_funding_does_not_bridge_missing_dates(name):
    dates, assets = _assets()
    assets[name] = assets[name].drop(index=80)
    with pytest.raises(ValueError, match="Missing lagged funding"):
        funding_features(dates[65:], assets)
