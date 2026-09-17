from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import 循环验证脚本 as core
import 预测脚本 as predictor
import return_calibration
from return_calibration import calibrate_returns


def _market_frame(rows: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=rows)
    index = np.arange(rows)
    returns = 0.006 * np.sin(index * 0.73) + 0.003 * np.cos(index * 0.19)
    close = 100.0 * np.cumprod(1.0 + returns)
    volume = 1_000_000.0 * (1.2 + 0.1 * np.sin(index * 0.13))
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close * 0.999,
            "high": close * 1.006,
            "low": close * 0.994,
            "close": close,
            "pre_close": np.r_[100.0, close[:-1]],
            "vol": volume,
            "amount": close * volume,
        }
    )


def _config() -> core.DirectionPredictionConfig:
    return core.DirectionPredictionConfig(
        lookback=5,
        neutral_band=0.0,
        min_train_sequences=20,
        external_feature_mode="none",
        drop_zero_volume=True,
    )


def _bilstm_causal_config() -> core.DirectionPredictionConfig:
    return core.DirectionPredictionConfig(
        lookback=5,
        neutral_band=0.0,
        min_train_sequences=20,
        hidden_size=4,
        dense_size=4,
        dropout=0.0,
        epochs=1,
        batch_size=16,
        patience=1,
        device="cpu",
        external_feature_mode="none",
        drop_zero_volume=True,
        use_timeseries_cv=False,
    )


def _options() -> dict[str, object]:
    return {
        "signal_engine": "volatility_rule",
        "nested_rule_threshold_window": 20,
        "nested_rule_calibration_window": 10,
        "nested_rule_min_threshold_rows": 5,
        "nested_rule_min_calibration_rows": 5,
        "state_veto_window": 10,
        "state_veto_state_window": 50,
        "state_veto_min_rows": 2,
        "state_veto_bad_accuracy": 0.5,
        "state_veto_quantiles": (0.4,),
        "return_magnitude_mode": "directional_median",
        "return_calibration_window": 20,
        "return_calibration_min_rows": 5,
        "confidence_calibration_window": 20,
        "confidence_calibration_min_rows": 5,
        "confidence_calibration_compare_windows": (),
        "recent_failure_guard": False,
        "regime_postprocess": False,
        "output_path": None,
        "diagnostics_output_path": None,
        "confidence_output_path": None,
        "confidence_summary_path": None,
        "confidence_calibration_output_path": None,
        "confidence_calibration_summary_path": None,
        "rolling_confidence_output_path": None,
        "rolling_confidence_comparison_output_path": None,
        "regime_postprocess_diagnostics_output_path": None,
        "progress": False,
    }


@pytest.mark.parametrize("signal_engine", ["volatility_rule", "state_veto_rule"])
@pytest.mark.parametrize("recent_failure_guard", [False, True])
def test_live_prediction_matches_same_date_in_longer_replay(
    signal_engine, recent_failure_guard
):
    data = _market_frame()
    options = _options()
    options.update(
        signal_engine=signal_engine, recent_failure_guard=recent_failure_guard
    )
    cutoff = 130
    live = core.predict_next_day(data.iloc[: cutoff + 1], config=_config(), **options)
    replay = core.loop_validate_prediction_results(
        data,
        config=_config(),
        start_date=data["trade_date"].iloc[100],
        end_date=data["trade_date"].iloc[145],
        **options,
    )
    trade_date = int(data["trade_date"].iloc[cutoff].strftime("%Y%m%d"))
    historical = replay.loc[replay["trade_date"].eq(trade_date)].iloc[0]

    assert live["predicted_label"] == historical["predicted_label"]
    assert live["estimated_next_return"] == pytest.approx(
        historical["predicted_pct_change"], abs=1e-12
    )
    assert live["estimated_next_close"] == pytest.approx(historical["predicted_close"])
    assert live["confidence"] == pytest.approx(historical["calibrated_confidence"])
    assert live["raw_confidence"] == pytest.approx(historical["confidence"])


def test_include_latest_retains_unknown_outcome_and_public_csv():
    data = _market_frame()
    result = core.loop_validate_prediction_results(
        data, config=_config(), periods=3, include_latest=True, **_options()
    )
    latest = result.iloc[-1]
    public = core._format_result_frame_for_csv(result)

    assert latest["trade_date"] == int(data["trade_date"].iloc[-1].strftime("%Y%m%d"))
    assert pd.isna(latest["real_pct_change"])
    assert pd.isna(latest["correct"])
    assert result.iloc[:-1]["real_pct_change"].notna().all()
    assert result.iloc[:-1]["correct"].notna().all()
    assert pd.isna(public.iloc[-1]["次日实际涨跌幅"])
    assert pd.isna(public.iloc[-1]["方向预测正确"])


def test_future_market_changes_cannot_rewrite_prior_predictions():
    data = _market_frame()
    changed = data.copy()
    cutoff = 130
    later_rows = changed.index > cutoff
    changed.loc[later_rows, ["open", "high", "low", "close", "pre_close"]] *= 1.7
    changed.loc[later_rows, ["vol", "amount"]] *= 2.0
    options = _options()
    options.update(
        start_date=data["trade_date"].iloc[110],
        end_date=data["trade_date"].iloc[145],
        recent_failure_guard=True,
    )

    original = core.loop_validate_prediction_results(data, config=_config(), **options)
    altered = core.loop_validate_prediction_results(changed, config=_config(), **options)
    cutoff_date = int(data["trade_date"].iloc[cutoff].strftime("%Y%m%d"))
    prediction_columns = [
        "trade_date",
        "predicted_label",
        "predicted_pct_change",
        "predicted_close",
        "confidence",
        "calibrated_confidence",
        "return_calibration_scale",
    ]
    pd.testing.assert_frame_equal(
        original.loc[original["trade_date"].le(cutoff_date), prediction_columns],
        altered.loc[altered["trade_date"].le(cutoff_date), prediction_columns],
    )


def test_bilstm_live_prediction_matches_causal_replay_and_ignores_future_rows():
    data = _market_frame(90)
    cutoff = 61
    config = _bilstm_causal_config()
    options = _options()
    options.update(
        signal_engine="bilstm_causal",
        bilstm_refit_interval=5,
        recent_failure_guard=False,
        confidence_calibration_window=None,
        return_calibration_window=0,
    )
    live = core.predict_next_day(data.iloc[: cutoff + 1], config=config, **options)
    replay = core.loop_validate_prediction_results(
        data,
        config=config,
        start_date=data["trade_date"].iloc[cutoff],
        end_date=data["trade_date"].iloc[cutoff],
        **options,
    )
    changed = data.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close", "pre_close"]] *= 1.7
    changed.loc[changed.index > cutoff, ["vol", "amount"]] *= 2.0
    replay_with_future_changed = core.loop_validate_prediction_results(
        changed,
        config=config,
        start_date=data["trade_date"].iloc[cutoff],
        end_date=data["trade_date"].iloc[cutoff],
        **options,
    )

    row = replay.iloc[0]
    assert live["predicted_label"] == row["predicted_label"]
    assert live["estimated_next_return"] == pytest.approx(
        row["predicted_pct_change"], abs=1e-12
    )
    assert live["estimated_next_close"] == pytest.approx(row["predicted_close"])
    assert live["raw_confidence"] == pytest.approx(row["confidence"])
    assert live["calibrated_confidence"] == pytest.approx(row["confidence"])
    pd.testing.assert_frame_equal(
        replay.loc[:, ["predicted_label", "predicted_pct_change", "predicted_close", "confidence"]],
        replay_with_future_changed.loc[:, ["predicted_label", "predicted_pct_change", "predicted_close", "confidence"]],
    )


def _return_frame() -> pd.DataFrame:
    predicted = np.asarray([0.01, 0.01, 0.08, 0.04, -0.02])
    return pd.DataFrame(
        {
            "trade_date": np.arange(1, 6),
            "predicted_pct_change": predicted,
            "predicted_label": (predicted > 0).astype(int),
            "predicted_close": 100.0 * (1.0 + predicted),
            "real_pct_change": [0.001, 0.002, 0.04, -0.04, np.nan],
            "confidence": [0.6, 0.7, 0.65, 0.8, 0.55],
            "correct": pd.array([True, True, True, False, pd.NA], dtype="boolean"),
        }
    )


def test_return_calibration_uses_weighted_median_and_updates_price():
    frame = _return_frame()

    calibrated = calibrate_returns(frame, window=3, min_rows=2)

    # Ratios 0.1, 0.2, 0.5 have weights 0.01, 0.01, 0.08.
    assert calibrated.loc[3, "return_calibration_scale"] == pytest.approx(0.5)
    assert calibrated.loc[3, "return_calibration_rows"] == 3
    assert calibrated.loc[3, "predicted_pct_change"] == pytest.approx(0.02)
    assert calibrated.loc[3, "predicted_close"] == pytest.approx(102.0)
    pd.testing.assert_series_equal(calibrated["predicted_label"], frame["predicted_label"])
    pd.testing.assert_series_equal(calibrated["correct"], frame["correct"])
    assert calibrated.loc[3, "uncalibrated_predicted_return"] == pytest.approx(0.04)


def test_return_calibration_does_not_read_current_or_future_outcomes():
    frame = _return_frame()
    changed = frame.copy()
    changed.loc[3:, "real_pct_change"] = [0.5, 0.9]
    changed.loc[4, "predicted_pct_change"] = 0.9

    original = calibrate_returns(frame, window=3, min_rows=2)
    altered = calibrate_returns(changed, window=3, min_rows=2)

    columns = [
        "predicted_pct_change",
        "predicted_close",
        "return_calibration_scale",
        "return_calibration_rows",
    ]
    pd.testing.assert_frame_equal(original.loc[:3, columns], altered.loc[:3, columns])


def test_return_calibration_clamps_weighted_median_index(monkeypatch):
    frame = _return_frame()
    original_searchsorted = return_calibration.np.searchsorted

    def oversized_index(*args, **kwargs):
        return len(args[0])

    monkeypatch.setattr(return_calibration.np, "searchsorted", oversized_index)
    calibrated = calibrate_returns(frame, window=3, min_rows=2)

    monkeypatch.setattr(return_calibration.np, "searchsorted", original_searchsorted)
    assert calibrated.loc[3, "return_calibration_scale"] == pytest.approx(0.5)


def test_return_calibration_preserves_finite_close_for_negative_one_return():
    frame = _return_frame()
    frame.loc[3, "predicted_pct_change"] = -1.0
    frame.loc[3, "predicted_close"] = 0.0

    calibrated = calibrate_returns(frame, window=3, min_rows=2)

    assert calibrated.loc[3, "return_calibration_close_fallback"] == 1
    assert calibrated.loc[3, "predicted_close"] == pytest.approx(0.0)
    assert np.isfinite(calibrated["predicted_close"]).all()


def test_zero_calibrated_return_keeps_explicit_up_direction():
    frame = _return_frame()
    frame.loc[:2, "real_pct_change"] *= -1

    calibrated = calibrate_returns(frame, window=3, min_rows=2)
    public = core._format_result_frame_for_csv(calibrated)

    assert calibrated.loc[3, "return_calibration_scale"] == 0.0
    assert calibrated.loc[3, "predicted_pct_change"] == 0.0
    assert calibrated.loc[3, "predicted_label"] == 1
    assert calibrated.loc[3, "predicted_close"] == pytest.approx(100.0)
    assert public.loc[3, "预测方向"] == "上涨"


def _cli_arguments() -> list[str]:
    return [
        "--mode", "predict",
        "--signal-engine", "volatility_rule",
        "--rule-top-k", "7",
        "--confidence-calibration-window", "30",
        "--confidence-calibration-compare-windows", "",
        "--return-calibration-window", "40",
        "--return-calibration-min-rows", "8",
        "--no-recent-failure-guard",
    ]


@pytest.mark.parametrize("standalone", [False, True])
def test_prediction_cli_forwards_engine_and_calibration_options(
    monkeypatch, capsys, standalone
):
    data = _market_frame(5)
    captured = {}

    def predict(frame, config, **kwargs):
        assert frame is data
        captured.update(kwargs)
        return {"signal_engine": kwargs["signal_engine"]}

    if standalone:
        args = predictor.parse_args(_cli_arguments() + ["--json"])
        monkeypatch.setattr(predictor, "parse_args", lambda: args)
        monkeypatch.setattr(predictor, "resolve_csv_path", lambda _: Path("market.csv"))
        monkeypatch.setattr(predictor.pd, "read_csv", lambda *args, **kwargs: data)
        monkeypatch.setattr(predictor, "predict_next_day", predict)
        monkeypatch.setattr(predictor, "save_prediction_csv", lambda *args: None)
        predictor.main()
    else:
        args = core._build_argument_parser().parse_args(_cli_arguments())
        monkeypatch.setattr(core, "predict_next_day", predict)
        core._run_predict_cli(data, _config(), args)

    assert captured["signal_engine"] == "volatility_rule"
    assert captured["rule_top_k"] == 7
    assert captured["confidence_calibration_window"] == 30
    assert captured["confidence_calibration_compare_windows"] == ()
    assert captured["return_calibration_window"] == 40
    assert captured["return_calibration_min_rows"] == 8
    assert captured["recent_failure_guard"] is False
    assert json.loads(capsys.readouterr().out)["signal_engine"] == "volatility_rule"
