from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tushare_prediction_pipeline as pipeline


def _result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [20260105, 20260106, 20260107],
            "predicted_pct_change": [0.002, -0.003, 0.001],
            "predicted_label": [1, 0, 1],
            "predicted_close": [100.2, 99.7, 100.1],
            "confidence": [0.61, 0.62, 0.63],
            "calibrated_confidence": [0.60, 0.61, 0.62],
            "real_pct_change": [0.004, -0.001, None],
            "correct": pd.array([True, True, pd.NA], dtype="boolean"),
            "confidence_calibration_window": [300, 300, 300],
            "confidence_calibration_method": ["platt", "platt", "platt"],
            "confidence_calibration_rows": [60, 61, 62],
            "confidence_calibration_fallback": [0, 0, 0],
        }
    )


def _market_frame(rows: int) -> pd.DataFrame:
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


def _fast_config() -> pipeline.prediction_core.DirectionPredictionConfig:
    return pipeline.prediction_core.DirectionPredictionConfig(
        lookback=5,
        neutral_band=0.0,
        min_train_sequences=20,
        external_feature_mode="none",
        drop_zero_volume=True,
    )


def _fast_loop_overrides() -> dict[str, object]:
    return {
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
    }


def test_requested_validation_days_excludes_the_live_prediction(monkeypatch):
    captured: dict[str, object] = {}

    def fake_loop(feature_frame, config, **kwargs):
        captured.update(kwargs)
        return _result_frame()

    monkeypatch.setattr(
        pipeline.prediction_core,
        "loop_validate_prediction_results",
        fake_loop,
    )

    result = pipeline.run_validation_and_prediction(
        pd.DataFrame(),
        validation_days=2,
        signal_engine="state_veto_rule",
        progress=False,
    )

    assert len(result) == 3
    assert captured["periods"] == 3
    assert captured["include_latest"] is True
    assert captured["signal_engine"] == "state_veto_rule"
    assert captured["output_path"] is None


def test_combined_csv_marks_validation_and_prediction_rows(monkeypatch):
    captured: dict[str, object] = {}

    def capture_to_csv(frame, path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", capture_to_csv)

    public = pipeline.save_combined_csv(_result_frame(), "combined.csv")

    assert public.columns[0] == "结果类型"
    assert public["结果类型"].tolist() == ["循环验证", "循环验证", "次日预测"]
    assert public["预测方向"].tolist() == ["上涨", "下跌", "上涨"]
    assert pd.isna(public.loc[2, "次日实际涨跌幅"])
    assert str(captured["path"]) == "combined.csv"
    assert captured["index"] is False
    assert captured["encoding"] == "utf-8-sig"


def test_validation_days_must_be_nonnegative():
    with pytest.raises(ValueError, match="nonnegative"):
        pipeline.run_validation_and_prediction(
            pd.DataFrame(),
            validation_days=-1,
            progress=False,
        )


def test_cli_validation_day_alias():
    args = pipeline.parse_args(["--loop-days", "17", "--no-progress"])

    assert args.validation_days == 17
    assert args.no_progress is True


def test_appending_market_days_preserves_predictions_and_completes_outcome():
    initial_data = _market_frame(130)
    updated_data = _market_frame(138)
    options = _fast_loop_overrides()
    config = _fast_config()

    initial = pipeline.run_validation_and_prediction(
        initial_data,
        validation_days=10,
        signal_engine="state_veto_rule",
        progress=False,
        config=config,
        loop_overrides=options,
    )
    updated = pipeline.run_validation_and_prediction(
        updated_data,
        # Keep the same output start date so every original result overlaps.
        validation_days=18,
        signal_engine="state_veto_rule",
        progress=False,
        config=config,
        loop_overrides=options,
    )

    prediction_columns = [
        "trade_date",
        "predicted_label",
        "predicted_pct_change",
        "predicted_close",
        "confidence",
        "calibrated_confidence",
        "return_calibration_scale",
    ]
    latest_signal_date = int(initial.iloc[-1]["trade_date"])
    updated_overlap = updated.loc[
        updated["trade_date"].le(latest_signal_date), prediction_columns
    ]
    pd.testing.assert_frame_equal(
        initial.loc[:, prediction_columns].reset_index(drop=True),
        updated_overlap.reset_index(drop=True),
    )

    initial_live = initial.iloc[-1]
    completed_live = updated.loc[
        updated["trade_date"].eq(latest_signal_date)
    ].iloc[0]
    expected_return = (
        updated_data["close"].iloc[130] / updated_data["close"].iloc[129] - 1.0
    )
    assert pd.isna(initial_live["real_pct_change"])
    assert pd.isna(initial_live["correct"])
    assert completed_live["real_pct_change"] == pytest.approx(expected_return)
    assert pd.notna(completed_live["correct"])
