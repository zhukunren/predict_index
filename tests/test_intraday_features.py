import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.build_intraday_context import build_context
from tools.intraday_features import FEATURE_COLUMNS, expected_bar_times, intraday_features


def fixture(day=20240102, frequency=30, opening_row=False):
    times = expected_bar_times(day, frequency)
    anchors = np.array([100., 103., 104., 105., 106., 104., 102., 101., 100.])
    positions = np.arange(frequency, 241, frequency)
    close = np.interp(positions, np.arange(0, 241, 30), anchors)
    open_ = np.r_[100., close[:-1]]
    block_amount = np.array([10.,10.,20.,20.,30.,30.,40.,40.])*1000
    amount = np.repeat(block_amount/(30/frequency), 30//frequency)
    bars = pd.DataFrame({"ts_code":"000001.SH", "trade_time":times,
                         "open":open_, "close":close, "high":np.maximum(open_,close)+.5,
                         "low":np.minimum(open_,close)-.5, "vol":1000.*frequency/30, "amount":amount})
    if opening_row:
        opening = {"ts_code":"000001.SH", "trade_time":pd.Timestamp(str(day))+pd.Timedelta(hours=9,minutes=30),
                   "open":100.,"high":100.,"low":100.,"close":100.,"vol":100.,"amount":1000.}
        bars = pd.concat([pd.DataFrame([opening]),bars],ignore_index=True)
    daily = pd.DataFrame({"trade_date":[day],"open":[100.],"close":[100.],
                          "high":[bars.high.max()],"low":[bars.low.min()],
                          "vol":[bars.vol.sum()/100],"amount":[bars.amount.sum()/1000]})
    return bars, daily


def test_late_price_path_and_turnover_shares_have_expected_meaning():
    bars, daily = fixture()
    result = intraday_features(daily.trade_date, bars, daily).iloc[0]
    assert result.intraday_last_30_return == pytest.approx(100/101-1)
    assert result.intraday_afternoon_return == pytest.approx(100/106-1)
    assert result.intraday_late_minus_early_return == pytest.approx(100/101-1-.03)
    assert result.intraday_last_30_amount_share == pytest.approx(.2)
    assert result.intraday_afternoon_amount_share == pytest.approx(.7)
    moves = np.diff(np.log([100,103,104,105,106,104,102,101,100]))
    assert result.intraday_realized_vol_30 == pytest.approx(np.sqrt((moves**2).sum()))
    assert result.intraday_downside_variance_share == pytest.approx((moves[moves<0]**2).sum()/(moves**2).sum())
    assert result.intraday_last_bar == "2024-01-02T15:00:00+08:00"


@pytest.mark.parametrize("frequency", [1,5,15,30])
def test_thirty_minute_feature_definition_is_independent_of_input_frequency(frequency):
    bars, daily = fixture(frequency=frequency)
    coarse, reference = fixture()
    result = intraday_features(daily.trade_date, bars, daily, frequency_minutes=frequency)
    expected = intraday_features(reference.trade_date, coarse, reference)
    np.testing.assert_allclose(result.loc[:, FEATURE_COLUMNS], expected.loc[:, FEATURE_COLUMNS], rtol=0, atol=1e-14)


def test_future_and_target_columns_are_excluded_from_requested_features():
    first, ref1 = fixture()
    second, ref2 = fixture(20240103)
    bars, daily = pd.concat([first,second],ignore_index=True),pd.concat([ref1,ref2],ignore_index=True)
    daily["target_next_return"] = [999,-999]
    all_features = intraday_features(daily.trade_date, bars, daily)
    expected = intraday_features([20240102], first, ref1)
    pd.testing.assert_frame_equal(all_features.iloc[:1], expected, check_exact=True)
    bars.loc[bars.trade_time.dt.day.eq(3), "close"] = np.nan
    daily.loc[daily.trade_date.eq(20240103), "close"] = -1
    daily["target_next_return"] *= -1
    pd.testing.assert_frame_equal(intraday_features([20240102], bars, daily), expected, check_exact=True)


@pytest.mark.parametrize("problem", ["missing_close", "duplicate", "lunch", "wrong_code", "wrong_frequency"])
def test_missing_or_ambiguous_bar_structure_is_rejected(problem):
    bars, daily = fixture()
    frequency = 30
    if problem == "missing_close": bars=bars.iloc[:-1]
    if problem == "duplicate": bars=pd.concat([bars,bars.iloc[-1:]],ignore_index=True)
    if problem == "lunch": bars.loc[4,"trade_time"]=pd.Timestamp("2024-01-02 12:30")
    if problem == "wrong_code": bars.loc[0,"ts_code"]="000300.SH"
    if problem == "wrong_frequency": frequency=5
    with pytest.raises(ValueError): intraday_features(daily.trade_date,bars,daily,frequency_minutes=frequency)


def test_daily_price_and_quantity_unit_mismatches_are_rejected():
    bars, daily = fixture()
    bad = daily.copy();bad["close"] += .02
    with pytest.raises(ValueError,match="prices disagree"): intraday_features(daily.trade_date,bars,bad)
    with pytest.raises(ValueError,match="totals/units"): intraday_features(daily.trade_date,bars,daily,volume_unit="hands")
    scaled = bars.copy();scaled["vol"] /= 100;scaled["amount"] /= 1000
    pd.testing.assert_frame_equal(intraday_features(daily.trade_date,bars,daily),
                                  intraday_features(daily.trade_date,scaled,daily,volume_unit="hands",amount_unit="thousand_yuan"), check_exact=True)


def test_optional_opening_observation_contributes_to_daily_turnover():
    bars,daily=fixture(opening_row=True)
    result=intraday_features(daily.trade_date,bars,daily).iloc[0]
    assert result.intraday_bar_rows==9
    assert result.intraday_last_30_amount_share==pytest.approx(40000/201000)


def test_timezone_and_incomplete_daily_coverage_are_explicit():
    bars,daily=fixture()
    utc=bars.copy();utc["trade_time"]=utc.trade_time.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    pd.testing.assert_frame_equal(intraday_features(daily.trade_date,utc,daily),intraday_features(daily.trade_date,bars,daily),check_exact=True)
    with pytest.raises(ValueError,match="18:00"): intraday_features(daily.trade_date,bars,daily,publication_hour=15)
    with pytest.raises(ValueError,match="Missing whole"): intraday_features([20240102,20240103],bars,daily)


def test_flat_path_has_zero_realized_volatility():
    bars,daily=fixture()
    bars.loc[:,["open","high","low","close"]]=100.
    daily.loc[:,["open","high","low","close"]]=100.
    result=intraday_features(daily.trade_date,bars,daily).iloc[0]
    assert result.intraday_realized_vol_30==0
    assert result.intraday_downside_variance_share==0


def test_builder_freezes_real_inputs_and_never_labels_them_as_performance_evidence(tmp_path:Path):
    bars,daily=fixture()
    raw=tmp_path/'bars.csv';day=tmp_path/'daily.csv';base=tmp_path/'baseline.csv';out=tmp_path/'context'
    bars.to_csv(raw,index=False);daily.to_csv(day,index=False);daily[['trade_date']].to_csv(base,index=False)
    manifest=build_context([raw],day,base,out,source_description='Synthetic test fixture only; Shanghai interval-end timestamps')
    assert manifest['status']=='complete' and manifest['performance_evaluated'] is False
    assert manifest['network_requests']==0 and manifest['production_changed'] is False
    assert (out/'raw/bars_000.csv').read_bytes()==raw.read_bytes()
    assert json.loads((out/'manifest.json').read_text())['feature_rows']==1
    with pytest.raises(FileExistsError):build_context([raw],day,base,out,source_description='fixture')
    bars.iloc[:-1].to_csv(raw,index=False)
    invalid=tmp_path/'invalid'
    with pytest.raises(ValueError):build_context([raw],day,base,invalid,source_description='fixture')
    assert not invalid.exists()
