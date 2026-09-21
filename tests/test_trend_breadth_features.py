import numpy as np
import pandas as pd
import pytest

from tools.trend_breadth_features import (
    FEATURE_COLUMNS, TrendBreadthAccumulator, trend_breadth_features,
)


def dates(n=80):
    return pd.bdate_range("2022-01-03", periods=n).strftime("%Y%m%d").astype(int).to_numpy()


def day(date, returns=(1.0, -1.0), amounts=(3.0, 1.0)):
    return pd.DataFrame({"ts_code": ["600000.SH", "000001.SZ"], "trade_date": date,
                         "pct_chg": returns, "amount": amounts, "vol": 100.0})


def build(calendar, overrides=None):
    state = TrendBreadthAccumulator(calendar)
    return pd.DataFrame([state.step((overrides or {}).get(int(d), day(d)), d) for d in calendar])


def test_returns_are_compounded_per_stock_then_aggregated():
    calendar = dates()
    result = build(calendar)
    assert not result.trend_available.iloc[:59].any()
    assert result.trend_available.iloc[59:].all()
    last = result.iloc[-1]
    for window in (5, 20, 60):
        assert last[f"trend_up_{window}_fraction"] == 0.5
        assert last[f"eligible_{window}_rows"] == 2
    assert last.trend_negative_20_amount_fraction == 0.25
    assert last.trend_return_20_median == pytest.approx(((1.01**20 - 1) + (0.99**20 - 1)) / 2)
    assert last.trend_shanghai_relative_20 == 0.5
    assert last.trend_weakening_fraction == 0
    assert last.trend_up_20_change_5 == 0


def test_compounding_not_sum_of_daily_percentages():
    calendar = dates(20)
    state = TrendBreadthAccumulator(calendar)
    for i, d in enumerate(calendar):
        row = state.step(day(d, returns=((10 if i % 2 == 0 else -10), 0)), d)
    assert row["trend_up_20_fraction"] == 0
    assert row["trend_negative_20_amount_fraction"] == 0.75


def test_suspension_and_new_listing_do_not_inherit_or_fill_history():
    calendar = dates(125)
    inactive = day(calendar[60]); inactive.loc[1, "vol"] = 0
    # Keep Shanghai represented while the other stock is temporarily absent.
    missing = day(calendar[62]).iloc[:1].copy()
    new = day(calendar[65]); new.loc[len(new)] = ["300001.SZ", calendar[65], 1, 1, 100]
    result = build(calendar, {int(calendar[60]): inactive, int(calendar[62]): missing, int(calendar[65]): new})
    assert result.eligible_20_rows.iloc[61] == 1
    assert result.eligible_20_rows.iloc[63] == 1
    assert result.eligible_20_rows.iloc[65] == 1
    assert not result.trend_available.iloc[65]
    assert result.trend_available.iloc[-1]


def test_future_stock_membership_and_returns_do_not_change_prefix():
    calendar = dates()
    original = build(calendar)
    future = day(calendar[70], returns=(-70, 100))
    future.loc[2] = ["300002.SZ", calendar[70], 30, 1000, 100]
    changed = build(calendar, {int(calendar[70]): future})
    prefix = build(calendar[:70])
    pd.testing.assert_frame_equal(original.iloc[:70], changed.iloc[:70], check_exact=True)
    pd.testing.assert_frame_equal(original.iloc[:70], prefix, check_exact=True)
    pd.testing.assert_frame_equal(
        trend_breadth_features(calendar[60:70], original, calendar),
        trend_breadth_features(calendar[60:70], prefix, calendar[:70]), check_exact=True,
    )


def test_missing_market_session_or_wrong_publication_time_is_rejected():
    calendar = dates()
    state = TrendBreadthAccumulator(calendar)
    with pytest.raises(ValueError, match="consecutive"):
        state.step(day(calendar[1]), calendar[1])
    result = build(calendar)
    with pytest.raises(ValueError, match="Missing trend"):
        trend_breadth_features(calendar[65:], result.drop(62), calendar)
    with pytest.raises(ValueError, match="18:00"):
        trend_breadth_features(calendar[65:], result, calendar, publication_hour=15)


@pytest.mark.parametrize("invalid", ["duplicate", "date", "return"])
def test_invalid_raw_observations_are_rejected(invalid):
    calendar = dates(1); frame = day(calendar[0])
    if invalid == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    if invalid == "date": frame.loc[0, "trade_date"] = 20210101
    if invalid == "return": frame.loc[0, "pct_chg"] = -100
    with pytest.raises(ValueError): TrendBreadthAccumulator(calendar).step(frame, calendar[0])


def test_unavailable_stock_history_is_not_filled_into_model_features():
    calendar = dates(); source = build(calendar)
    expected = source.copy(deep=True)
    result = trend_breadth_features(calendar, source, calendar)
    assert result.loc[0, list(FEATURE_COLUMNS)].isna().all()
    assert not result.trend_available.iloc[0]
    assert np.isfinite(result.loc[result.trend_available, FEATURE_COLUMNS].to_numpy(float)).all()
    pd.testing.assert_frame_equal(source, expected, check_exact=True)


def test_falsely_claimed_coverage_is_rejected():
    calendar = dates(); source = build(calendar)
    source.loc[70, "stock_coverage_60"] = 0.5
    with pytest.raises(ValueError, match="Incomplete"):
        trend_breadth_features(calendar[65:], source, calendar)
