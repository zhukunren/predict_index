from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.option_features import UNDERLYING, aggregate_options
from tools.option_position_features import aggregate_positions, position_features


def inputs():
    dates = pd.Series(pd.bdate_range("2023-01-02", periods=40).strftime("%Y%m%d").astype(int))
    contracts = pd.DataFrame([
        {"ts_code": f"{term}{right}", "opt_code": UNDERLYING, "call_put": right,
         "list_date": start, "delist_date": end}
        for term, start, end in (("old", 20221201, 20230104), ("new", 20230105, 20230222), ("far", 20221201, 20230426))
        for right in ("P", "C")
    ])
    rows = []
    for date in dates:
        active = contracts.loc[contracts.list_date.le(date) & contracts.delist_date.ge(date)]
        for contract in active.itertuples():
            interest = 100 if contract.ts_code.startswith("far") else (1000 if contract.ts_code.startswith("old") else 2000)
            rows.append({"ts_code": contract.ts_code, "trade_date": date, "vol": 10, "amount": 100, "oi": interest})
    return dates, contracts, pd.DataFrame(rows)


def test_rollover_does_not_count_expired_and_new_contracts_as_matched_changes():
    dates, contracts, raw = inputs()
    raw.loc[raw.ts_code.eq("farP") & raw.trade_date.eq(20230105), "oi"] = 120
    full = aggregate_positions(raw, contracts, dates)
    expected = aggregate_options(raw, contracts, dates)
    pd.testing.assert_frame_equal(full.loc[:, expected.columns], expected, check_exact=True)
    rollover = full.loc[full.trade_date.eq(20230105)].iloc[0]
    assert rollover.matched_contract_rows == 2
    assert rollover.matched_prior_interest_coverage == pytest.approx(200 / 2200)
    assert rollover.matched_put_interest_growth == pytest.approx(0.2)
    assert rollover.matched_call_interest_growth == 0
    assert rollover.matched_interest_imbalance == pytest.approx(0.1)
    # The newly listed contracts expire in 48 days, so all current interest is far.
    assert rollover.near_interest_share == 0
    assert np.isnan(rollover.near_put_interest_share)


def test_expiry_buckets_use_each_historical_session_date():
    dates, contracts, raw = inputs()
    full = aggregate_positions(raw, contracts, dates)
    first = full.iloc[0]
    assert first.near_interest_share == pytest.approx(2000 / 2200)
    assert first.near_put_interest_share == first.far_put_interest_share == 0.5
    # The February contracts move into the 30-calendar-day bucket on January 23.
    before = full.loc[full.trade_date.eq(20230120)].iloc[0]
    after = full.loc[full.trade_date.eq(20230123)].iloc[0]
    assert before.near_interest_share == 0
    assert after.near_interest_share == pytest.approx(4000 / 4200)


@pytest.mark.parametrize("lag_sessions", [0, 1])
def test_aggregation_and_features_match_prefixes_and_ignore_future_activity(lag_sessions):
    dates, contracts, raw = inputs()
    full = aggregate_positions(raw, contracts, dates)
    timing = {"lag_sessions": lag_sessions, "publication_hour": 20}
    features = position_features(dates.iloc[6:], full, dates, **timing)
    source_rows = 26 - lag_sessions
    prefix_daily = aggregate_positions(raw.loc[raw.trade_date.le(dates.iloc[source_rows - 1])], contracts, dates.iloc[:source_rows])
    pd.testing.assert_frame_equal(full.iloc[:source_rows], prefix_daily, check_exact=True)
    prefix = position_features(dates.iloc[6:26], prefix_daily, dates.iloc[:26], **timing)
    pd.testing.assert_frame_equal(features.iloc[:20], prefix, check_exact=True)
    raw.loc[raw.trade_date.ge(dates.iloc[source_rows]), "oi"] *= 10
    changed = aggregate_positions(raw, contracts, dates)
    changed_features = position_features(dates.iloc[6:], changed, dates, **timing)
    pd.testing.assert_frame_equal(features.iloc[:20], changed_features.iloc[:20], check_exact=True)
    assert features.option_position_source_date.iloc[0] == dates.iloc[6 - lag_sessions]


def test_same_day_positions_reject_early_or_invalid_publication():
    dates, contracts, raw = inputs()
    daily = aggregate_positions(raw, contracts, dates)
    with pytest.raises(ValueError, match="20:00"):
        position_features(dates.iloc[6:], daily, dates, lag_sessions=0, publication_hour=19)
    with pytest.raises(ValueError, match="timing"):
        position_features(dates.iloc[6:], daily, dates, lag_sessions=-1, publication_hour=20)


def test_missing_position_measures_are_not_filled_and_propagate_to_rolling_means():
    dates, contracts, raw = inputs()
    daily = aggregate_positions(raw, contracts, dates)
    daily.loc[25, "matched_interest_imbalance"] = np.nan
    full = position_features(dates.iloc[6:], daily, dates)
    assert not full.loc[full.trade_date.isin(dates.iloc[26:31]), "option_position_available"].any()
    assert full.loc[full.trade_date.eq(dates.iloc[31]), "option_position_available"].item()


def test_incomplete_live_contract_set_is_rejected():
    dates, contracts, raw = inputs()
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate_positions(raw.iloc[1:], contracts, dates)


def test_zero_interest_bucket_remains_unavailable():
    dates, contracts, raw = inputs()
    raw.loc[raw.trade_date.eq(dates.iloc[25]) & raw.ts_code.str.startswith("new"), "oi"] = 0
    daily = aggregate_positions(raw, contracts, dates)
    full = position_features(dates.iloc[6:], daily, dates)
    assert not full.loc[full.trade_date.eq(dates.iloc[26]), "option_position_available"].item()
    assert full.loc[full.trade_date.eq(dates.iloc[26]), "option_position_near_put_interest_share"].isna().all()
