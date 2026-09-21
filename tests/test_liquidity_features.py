from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.fetch_liquidity_context import fetch_year
from tools.liquidity_features import FEATURE_COLUMNS, liquidity_features


def inputs(rows=90):
    dates = pd.Series(pd.bdate_range("2023-01-02", periods=rows).strftime("%Y%m%d").astype(int))
    step = np.arange(rows)
    bank = pd.DataFrame({"date": dates, "on": 1 + step / 100, "1w": 2, "3m": 3})
    assets = {"shibor": bank}
    for name, code, premium in (("gc001", "204001.SH", 0.1), ("gc007", "204007.SH", 0.2)):
        rate = bank["on"] + premium
        assets[name] = pd.DataFrame({"trade_date": dates, "ts_code": code, "close": rate,
                                    "high": rate + 0.1, "low": rate - 0.1, "amount": 100 + step})
    return dates, assets


def test_rates_units_lag_spreads_and_rolling_volume():
    dates, assets = inputs()
    result = liquidity_features(dates.iloc[20:], assets, dates)
    assert result.columns.tolist() == ["trade_date", "liquidity_source_date", *FEATURE_COLUMNS]
    first = result.iloc[0]
    assert first.liquidity_source_date == dates.iloc[19]
    assert first.liquidity_overnight == pytest.approx(0.0119)
    assert first.liquidity_overnight_change_1 == pytest.approx(0.0001)
    assert first.liquidity_overnight_change_5 == pytest.approx(0.0005)
    assert first.liquidity_exchange_bank_overnight_spread == pytest.approx(0.001)
    assert first.liquidity_exchange_volume_ratio_20 == pytest.approx(119 / np.mean(np.arange(100, 120)))
    assert first.liquidity_exchange_short_volume_share == pytest.approx(0.5)


def test_future_data_and_current_session_do_not_change_prefix():
    dates, assets = inputs()
    original = {name: frame.copy(deep=True) for name, frame in assets.items()}
    full = liquidity_features(dates.iloc[20:], assets, dates)
    prefix = liquidity_features(dates.iloc[20:51], {name: frame.iloc[:50] for name, frame in assets.items()}, dates.iloc[:51])
    pd.testing.assert_frame_equal(full.iloc[:31], prefix, check_exact=True)
    for name in assets:
        pd.testing.assert_frame_equal(assets[name], original[name])
        columns = [column for column in assets[name] if column not in ("date", "trade_date", "ts_code")]
        assets[name].loc[50:, columns] *= 2
    altered = liquidity_features(dates.iloc[20:], assets, dates)
    pd.testing.assert_frame_equal(full.iloc[:31], altered.iloc[:31], check_exact=True)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_code", "nonfinite", "invalid_range", "calendar_gap"])
def test_incomplete_or_invalid_liquidity_is_rejected(problem):
    dates, assets = inputs()
    if problem == "missing":
        assets["shibor"] = assets["shibor"].drop(index=35)
    elif problem == "duplicate":
        assets["gc001"] = pd.concat([assets["gc001"], assets["gc001"].iloc[:1]])
    elif problem == "wrong_code":
        assets["gc007"].loc[10, "ts_code"] = "204001.SH"
    elif problem == "nonfinite":
        assets["shibor"].loc[10, "on"] = np.nan
    elif problem == "invalid_range":
        assets["gc001"].loc[10, "close"] = 99
    else:
        dates = pd.concat([dates.iloc[:22], dates.iloc[21:]])
    with pytest.raises(ValueError):
        liquidity_features(dates.iloc[20:], assets, dates)


def test_fetch_year_accepts_provider_fields_and_sanitizes_errors():
    _, assets = inputs()
    class Client:
        def query(self, api, **kwargs):
            assert api == "shibor"
            assert kwargs["fields"] == "date,on,1w,3m"
            return assets["shibor"].iloc[::-1]
    result = fetch_year(Client(), "shibor", "20230101", "20231231")
    assert len(result) == 90
    class Failing:
        def query(self, *args, **kwargs):
            raise RuntimeError("SECRET_TOKEN")
    with pytest.raises(RuntimeError, match="credentials omitted") as error:
        fetch_year(Failing(), "shibor", "20230101", "20231231")
    assert "SECRET_TOKEN" not in str(error.value)


def test_irregular_row_indices_and_extra_bank_holiday_do_not_shift_features():
    dates, assets = inputs()
    original = liquidity_features(dates.iloc[20:], assets, dates)
    assets["shibor"] = pd.concat([assets["shibor"], pd.DataFrame({"date": [20230107], "on": [9], "1w": [9], "3m": [9]})])
    for frame in assets.values():
        frame.index = np.arange(len(frame)) * 3
    altered = liquidity_features(dates.iloc[20:], assets, dates)
    pd.testing.assert_frame_equal(original, altered, check_exact=True)


def test_missing_repo_session_disables_rolling_features_without_stale_fill():
    dates, assets = inputs()
    assets["gc001"] = assets["gc001"].drop(index=35)
    result = liquidity_features(dates.iloc[20:], assets, dates, allow_missing=True)
    unavailable = result.loc[~result.liquidity_available, "trade_date"]
    assert unavailable.tolist() == dates.iloc[36:56].tolist()
    assert result.loc[result.trade_date.eq(dates.iloc[36]), "liquidity_exchange_bank_overnight_spread"].isna().all()
    prefix = liquidity_features(dates.iloc[20:51], {name: frame.loc[frame.index < 50] for name, frame in assets.items()},
                                dates.iloc[:51], allow_missing=True)
    pd.testing.assert_frame_equal(result.iloc[:31], prefix, check_exact=True)


@pytest.mark.parametrize("name", ["downside_logistic", "downside_shallow"])
def test_unavailable_liquidity_cannot_train_or_change_direction(name):
    from tools.downside_specialist import downside_probabilities
    dates, assets = inputs(400)
    assets["gc001"] = assets["gc001"].drop(index=335)
    extra = liquidity_features(dates.iloc[20:], assets, dates, allow_missing=True)
    features = extra.drop(columns=["liquidity_available", "liquidity_source_date"])
    available = extra.liquidity_available.to_numpy()
    champion = pd.DataFrame({"trade_date": extra.trade_date, "predicted_label": 1,
                             "real_pct_change": np.where(np.arange(len(extra)) % 3 == 0, -0.01, 0.01)})
    full = downside_probabilities(champion, features, name, available=available)
    assert full.loc[~available, "downside_training_rows"].eq(0).all()
    assert full.loc[~available, "downside_probability"].eq(0).all()
    assert full.downside_training_rows.iloc[-1] >= 120
    modified = champion.copy()
    modified.loc[~available, "real_pct_change"] *= -1
    altered = downside_probabilities(modified, features, name, available=available)
    pd.testing.assert_frame_equal(full, altered, check_exact=True)
    prefix = champion.iloc[:345].copy()
    prefix.loc[344, "real_pct_change"] = np.nan
    replay = downside_probabilities(prefix, features.iloc[:345], name, available=available[:345])
    pd.testing.assert_frame_equal(full.iloc[:345], replay, check_exact=True)
