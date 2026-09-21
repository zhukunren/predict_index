import numpy as np
import pandas as pd
import pytest

from tools.fx_features import TS_CODE, FEATURE_COLUMNS, fx_features, normalize_quotes


def quotes(rows=65):
    dates = pd.bdate_range("2023-01-02", periods=rows).strftime("%Y%m%d").astype(int)
    close = 6.5 * np.exp(np.arange(rows) * 0.001)
    return pd.DataFrame({"trade_date": dates, "ts_code": TS_CODE,
                         "bid_open": close * .999, "bid_high": close * 1.002, "bid_low": close * .998,
                         "bid_close": close, "ask_close": close * 1.0001, "tick_qty": 1000})


def test_same_gmt_day_close_is_excluded_and_returns_use_mid_quotes():
    frame = quotes()
    signal = int(frame.trade_date.iloc[40])
    result = fx_features([signal], frame)
    assert result.fx_source_date.iloc[0] == frame.trade_date.iloc[39]
    assert result.fx_cnh_return_1.iloc[0] == pytest.approx(np.expm1(.001))
    assert result.fx_cnh_return_5.iloc[0] == pytest.approx(np.expm1(.005))
    assert result.fx_cnh_return_20.iloc[0] == pytest.approx(np.expm1(.020))
    assert result.fx_cnh_close_spread.iloc[0] == pytest.approx(.0001 / 1.00005)
    assert result.fx_cnh_bid_range.iloc[0] == pytest.approx(.004 * np.exp(.001))


def test_same_day_and_future_prices_do_not_change_prior_features():
    frame = quotes(); original = frame.copy(deep=True)
    signals = frame.trade_date.iloc[30:].reset_index(drop=True)
    full = fx_features(signals, frame)
    through = int(frame.trade_date.iloc[45])
    prefix = fx_features(signals.loc[signals.le(through)], frame.loc[frame.trade_date.lt(through)])
    pd.testing.assert_frame_equal(full.loc[full.trade_date.le(through)].reset_index(drop=True), prefix, check_exact=True)
    price_columns = [c for c in frame if c.startswith(("bid_", "ask_"))]
    frame.loc[frame.trade_date.ge(through), price_columns] *= 1.5
    changed = fx_features(signals, frame)
    pd.testing.assert_frame_equal(full.loc[full.trade_date.le(through)], changed.loc[changed.trade_date.le(through)], check_exact=True)
    assert not np.array_equal(full.loc[:, FEATURE_COLUMNS], changed.loc[:, FEATURE_COLUMNS])
    pd.testing.assert_frame_equal(original, quotes(), check_exact=True)


def test_stale_quote_and_truncated_warmup_are_rejected():
    frame = quotes()
    with pytest.raises(ValueError, match="stale"):
        fx_features([20240501], frame)
    with pytest.raises(ValueError, match="21"):
        fx_features([int(frame.trade_date.iloc[10])], frame)
    with pytest.raises(ValueError, match="long missing"):
        fx_features([int(frame.trade_date.iloc[-1])], frame.drop(index=range(25,35)))


def test_unused_open_range_anomaly_is_audited_without_changing_features():
    frame = quotes()
    signals = frame.trade_date.iloc[30:]
    expected = fx_features(signals, frame)
    frame.loc[10, "bid_open"] = frame.bid_high.iloc[10] * 1.1
    assert normalize_quotes(frame).unused_open_range_anomaly.sum() == 1
    pd.testing.assert_frame_equal(expected, fx_features(signals, frame), check_exact=True)


@pytest.mark.parametrize("defect", ["duplicate", "code", "negative", "range", "spread"])
def test_quote_integrity_defects_are_rejected(defect):
    frame = quotes()
    if defect == "duplicate": frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    if defect == "code": frame.loc[0, "ts_code"] = "EURUSD.FXCM"
    if defect == "negative": frame.loc[0, "bid_close"] = -1
    if defect == "range": frame.loc[0, "bid_low"] = frame.bid_high.iloc[0] + .1
    if defect == "spread": frame.loc[0, "ask_close"] = frame.bid_close.iloc[0] - .1
    with pytest.raises(ValueError): normalize_quotes(frame)


def test_out_of_order_signals_are_rejected_and_nondefault_index_aligns():
    frame = quotes(); frame.index = np.arange(len(frame))*3+10
    signals = frame.trade_date.iloc[30:]
    result = fx_features(signals, frame)
    expected = fx_features(signals.reset_index(drop=True), frame.reset_index(drop=True))
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    with pytest.raises(ValueError, match="chronological"):
        fx_features(signals.iloc[::-1], frame)
