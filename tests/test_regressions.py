from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import 循环验证脚本 as core


ROOT = Path(__file__).resolve().parents[1]
DATA_MODULES = (
    ("tushare_data", "数据拉取脚本_tushare.py", "tushare"),
    ("akshare_data", "scripts/fetch_akshare.py", "akshare"),
    ("wind_data", "scripts/fetch_wind.py", "WindPy"),
)


@pytest.fixture(params=DATA_MODULES, ids=lambda item: item[0])
def data_module(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    module_name, filename, dependency = request.param
    dependency_stub = types.ModuleType(dependency)
    if dependency == "WindPy":
        dependency_stub.w = object()
    monkeypatch.setitem(sys.modules, dependency, dependency_stub)

    import_name = f"_test_{module_name}"
    spec = importlib.util.spec_from_file_location(import_name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, import_name, module)
    spec.loader.exec_module(module)
    return module


def _target_frame(rows: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=rows)
    close = np.arange(10.0, 10.0 + rows)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "vol": np.arange(100.0, 100.0 + rows),
            "amount": close * 100.0,
        }
    )


def _hangseng_name(module: types.ModuleType) -> str:
    if hasattr(module, "HANGSENG_NAME"):
        return str(module.HANGSENG_NAME)
    return str(module.HANGSENG_REQUEST.name)


def _target_name(module: types.ModuleType) -> str:
    if hasattr(module, "TARGET_NAME"):
        return str(module.TARGET_NAME)
    return str(module.TARGET_REQUEST.name)


def _market_frame(rows: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=rows)
    daily_returns = np.where(np.arange(rows) % 3 == 0, -0.004, 0.003)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    pre_close = np.r_[close[0] / 1.001, close[:-1]]
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close * 0.999,
            "high": close * 1.006,
            "low": close * 0.994,
            "close": close,
            "pre_close": pre_close,
            "vol": 1_000_000.0 + np.arange(rows) * 100.0,
            "amount": close * (1_000_000.0 + np.arange(rows) * 100.0),
        }
    )


def _loop_config() -> core.DirectionPredictionConfig:
    return core.DirectionPredictionConfig(
        lookback=5,
        neutral_band=0.0,
        min_train_sequences=20,
        external_feature_mode="none",
        drop_zero_volume=True,
    )


def _fast_rule_kwargs() -> dict[str, object]:
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
        "return_calibration_window": 0,
        "output_path": None,
        "diagnostics_output_path": None,
        "confidence_output_path": None,
        "confidence_summary_path": None,
        "confidence_calibration_window": None,
        "confidence_calibration_compare_windows": (),
        "rolling_confidence_output_path": None,
        "rolling_confidence_comparison_output_path": None,
        "progress": False,
    }


def test_cli_helpers_preserve_loop_argument_overrides():
    args = core._build_argument_parser().parse_args(
        [
            "--mode",
            "predict",
            "--loop-validate",
            "--latest-periods",
            "7",
            "--start-date",
            "20240101",
            "--end-date",
            "20241231",
            "--confidence-calibration-window",
            "0",
            "--confidence-calibration-compare-windows",
            "120, 300,600",
            "--state-veto-quantiles",
            "0.2,0.4",
            "--high-confidence-max-veto-state-rows",
            "-1",
            "--recent-failure-threshold",
            "0.37",
            "--regime-postprocess-state-columns",
            "vol20_bucket, streak_state",
            "--no-progress",
            "--epochs",
            "3",
        ]
    )

    kwargs = core._loop_validation_kwargs_from_cli_args(
        args,
        output_path="result.csv",
    )

    assert core._resolve_cli_mode(args) == "loop_validate"
    assert core._config_from_cli_args(args).epochs == 3
    assert kwargs["start_date"] is None
    assert kwargs["end_date"] is None
    assert kwargs["periods"] == 7
    assert kwargs["confidence_calibration_window"] is None
    assert kwargs["confidence_calibration_compare_windows"] == (120, 300, 600)
    assert kwargs["state_veto_quantiles"] == (0.2, 0.4)
    assert kwargs["high_confidence_max_veto_state_rows"] is None
    assert kwargs["recent_failure_degrade_threshold"] == pytest.approx(0.37)
    assert kwargs["regime_postprocess_state_columns"] == (
        "vol20_bucket",
        "streak_state",
    )
    assert kwargs["progress"] is False


def test_data_builders_keep_unknown_target_and_apply_timing_lag(data_module):
    target = _target_frame()
    hangseng = _target_frame().drop(columns="amount")
    target_name = _target_name(data_module)
    hangseng_name = _hangseng_name(data_module)

    target_features = data_module._target_feature_frame(target)
    assert pd.isna(target_features["target_next_return"].iloc[-1])
    assert pd.isna(target_features["target_next_direction"].iloc[-1])

    assert data_module.default_feature_lag(hangseng_name, "after_close") == 1
    assert data_module.default_feature_lag(hangseng_name, "before_open") == 2
    after_close = data_module.make_feature_frame(
        {target_name: target, hangseng_name: hangseng},
        prediction_time="after_close",
    )
    before_open = data_module.make_feature_frame(
        {target_name: target, hangseng_name: hangseng},
        prediction_time="before_open",
    )
    assert "hangseng_ret1_lag1" in after_close.columns
    assert "hangseng_ret1_lag2" in before_open.columns
    assert not after_close.equals(before_open)


def test_existing_hangseng_features_stay_aligned_when_sorted(data_module):
    source = pd.DataFrame(
        {
            "trade_date": ["2026-01-07", "2026-01-06", "2026-01-05"],
            "hangseng_ret1_lag1": [30.0, 20.0, 10.0],
        }
    )

    result = data_module._existing_hangseng_feature_frame(source)

    assert result["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
    ]
    assert result["hangseng_ret1_lag1"].tolist() == [10.0, 20.0, 30.0]


def test_lag2_external_features_are_rule_candidates():
    features = pd.DataFrame({"ext_hangseng_ret1_lag2": [0.0, 0.1]})

    assert "ext_hangseng_ret1_lag2" in core._rule_candidate_columns(features)


def test_calibrated_rule_threshold_mask_excludes_future_rows(monkeypatch):
    config = _loop_config()
    base = core._normalize_market_frame(_market_frame(), config)
    features = pd.DataFrame(
        {"volatility_20": np.linspace(0.0, 1.0, len(base))},
        index=base.index,
    )
    idx = 50
    captured: dict[str, pd.Series] = {}

    def capture_candidates(
        candidate_features: pd.DataFrame,
        threshold_mask: pd.Series,
    ) -> list[tuple[str, np.ndarray]]:
        captured["mask"] = threshold_mask.copy()
        return []

    monkeypatch.setattr(core, "_candidate_rule_predictions", capture_candidates)
    core._calibrated_rule_signal(
        base=base,
        features=features,
        idx=idx,
        config=config,
        threshold_end_date=base["date"].iloc[80],
        calibration_start_date=None,
        calibration_end_date=None,
        top_k=1,
    )

    selected = np.flatnonzero(captured["mask"].to_numpy())
    assert len(selected) > 0
    assert int(selected.max()) < idx


def test_historical_selector_warms_history_without_failure_guard(monkeypatch):
    history_lengths: list[int] = []
    original = core._historical_selector_decision

    def track_history(**kwargs):
        history_lengths.append(len(kwargs["history"]))
        return original(**kwargs)

    monkeypatch.setattr(core, "_historical_selector_decision", track_history)
    core.loop_validate_prediction_results(
        _market_frame(90),
        config=_loop_config(),
        periods=1,
        signal_engine="historical_selector",
        recent_failure_guard=False,
        selector_window=7,
        selector_min_history=2,
        selector_disagreement_window=5,
        selector_disagreement_min_history=1,
        **_fast_rule_kwargs(),
    )

    assert history_lengths[-1] >= 7


def test_loop_confidence_is_returned_saved_and_shown(tmp_path: Path, capsys):
    output_path = tmp_path / "prediction_results.csv"
    calibration_path = tmp_path / "confidence_calibration.csv"
    calibration_summary_path = tmp_path / "confidence_calibration_summary.csv"
    loop_kwargs = _fast_rule_kwargs()
    loop_kwargs.update(
        {
            "output_path": str(output_path),
            "confidence_calibration_output_path": str(calibration_path),
            "confidence_calibration_summary_path": str(calibration_summary_path),
            "progress": True,
        }
    )

    result = core.loop_validate_prediction_results(
        _market_frame(90),
        config=_loop_config(),
        periods=2,
        signal_engine="volatility_rule",
        recent_failure_guard=False,
        **loop_kwargs,
    )

    saved = pd.read_csv(output_path)
    calibration = pd.read_csv(calibration_path)
    calibration_summary = pd.read_csv(calibration_summary_path)
    assert "confidence" in result.columns
    assert result["confidence"].between(0.0, 1.0).all()
    assert list(saved.columns) == list(core._PUBLIC_RESULT_COLUMNS)
    assert "置信度" in saved.columns
    assert np.allclose(saved["置信度"], result["confidence"])
    assert "预测方向" in saved.columns
    expected_directions = np.where(
        result["predicted_pct_change"].to_numpy() > 0,
        "上涨",
        "下跌",
    )
    assert saved["预测方向"].tolist() == expected_directions.tolist()
    assert int(calibration["rows"].sum()) == len(result)
    assert calibration_summary.loc[0, "rows_used"] == len(result)
    progress_output = capsys.readouterr().err
    assert "confidence=" in progress_output
    assert "%" in progress_output


def test_confidence_tracks_direction_reversals():
    veto_signal = core.RuleSignal(
        predicted_return=0.003,
        predicted_label=1,
        rule_names=["test"],
        calibration_accuracy=0.35,
        calibration_rows=20,
        diagnostics={"veto_applied": 1},
    )
    assert core._rule_signal_confidence(veto_signal) == pytest.approx(0.65)

    guard_diagnostics = {
        "recent_failure_guard_applied": 1,
        "recent_failure_guard_accuracy": 0.4,
    }
    assert core._confidence_after_failure_guard(
        0.65,
        guard_diagnostics,
    ) == pytest.approx(0.6)


def test_prediction_script_writes_shared_csv_schema(tmp_path: Path):
    from scripts.predict import save_prediction_csv

    output_path = tmp_path / "next_day_prediction.csv"
    result = {
        "last_date": "2026-09-14",
        "estimated_next_return": 0.0125,
        "estimated_next_close": 101.25,
        "confidence": 0.42,
        "raw_confidence": 0.30,
        "calibrated_confidence": 0.42,
        "confidence_calibration_status": "已校准",
        "confidence_calibration_method": "platt",
        "confidence_calibration_rows": 80,
        "confidence_calibration_fallback": 0,
    }

    save_prediction_csv(result, output_path)
    saved = pd.read_csv(output_path)

    assert list(saved.columns) == list(core._PUBLIC_RESULT_COLUMNS)
    assert saved.loc[0, "信号日期"] == 20260914
    assert saved.loc[0, "预测方向"] == "上涨"
    assert saved.loc[0, "预测次日涨跌幅"] == pytest.approx(0.0125)
    assert saved.loc[0, "原始边界分数"] == pytest.approx(0.30)
    assert saved.loc[0, "置信度"] == pytest.approx(0.42)
    assert saved.loc[0, "置信度校准状态"] == "已校准"
    assert pd.isna(saved.loc[0, "次日实际涨跌幅"])


def test_confidence_calibration_report_matches_bin_accuracy():
    frame = pd.DataFrame(
        {
            "confidence": [0.60, 0.60, 0.60, 0.60, 2.0 / 3.0, 2.0 / 3.0],
            "correct": [True, True, True, False, True, False],
        }
    )

    by_bin, summary = core.confidence_calibration_report(
        frame,
        bin_edges=(0.0, 0.60, 0.65, 1.0),
    )

    first = by_bin.loc[by_bin["rows"] == 4].iloc[0]
    second = by_bin.loc[by_bin["rows"] == 2].iloc[0]
    assert first["direction_accuracy"] == pytest.approx(0.75)
    assert second["direction_accuracy"] == pytest.approx(0.5)
    assert summary.loc[0, "rows_used"] == 6
    assert summary.loc[0, "direction_accuracy"] == pytest.approx(4.0 / 6.0)
    assert summary.loc[0, "expected_calibration_error"] == pytest.approx(
        (4 / 6) * 0.15 + (2 / 6) * (2.0 / 3.0 - 0.5)
    )


def test_rolling_confidence_calibration_uses_prior_rows_only():
    frame = pd.DataFrame(
        {
            "trade_date": np.arange(1, 41),
            "confidence": np.linspace(0.50, 0.85, 40),
            "correct": (np.arange(40) % 3 != 0),
        }
    )
    changed = frame.copy()
    changed.loc[20, "correct"] = not bool(changed.loc[20, "correct"])

    original = core._apply_rolling_confidence_calibration(
        frame,
        window=20,
        min_rows=10,
        method="platt",
    )
    changed_result = core._apply_rolling_confidence_calibration(
        changed,
        window=20,
        min_rows=10,
        method="platt",
    )

    assert original.loc[20, "confidence_calibration_rows"] == 20
    assert original.loc[20, "confidence_calibration_fallback"] == 0
    assert original.loc[20, "calibrated_confidence"] == pytest.approx(
        changed_result.loc[20, "calibrated_confidence"]
    )
    comparison, _ = core.compare_rolling_confidence_calibration(
        frame,
        windows=(10, 20),
        min_rows=5,
        evaluation_start_trade_date=21,
    )
    assert comparison["selected"].astype(bool).sum() == 1


def test_loop_can_compare_rolling_calibration_windows(tmp_path: Path):
    loop_kwargs = _fast_rule_kwargs()
    loop_kwargs.update(
        {
            "confidence_calibration_window": 20,
            "confidence_calibration_min_rows": 5,
            "confidence_calibration_compare_windows": (10, 20),
            "rolling_confidence_output_path": str(
                tmp_path / "rolling_confidence.csv"
            ),
            "rolling_confidence_comparison_output_path": str(
                tmp_path / "rolling_comparison.csv"
            ),
        }
    )

    result = core.loop_validate_prediction_results(
        _market_frame(90),
        config=_loop_config(),
        periods=2,
        signal_engine="volatility_rule",
        recent_failure_guard=False,
        **loop_kwargs,
    )

    comparison = pd.read_csv(tmp_path / "rolling_comparison.csv")
    assert "calibrated_confidence" in result.columns
    assert result["calibrated_confidence"].between(0.0, 1.0).all()
    assert set(comparison["window"]) == {10, 20}
    assert comparison["evaluation_rows"].ge(0).all()


def test_regime_states_use_full_market_history():
    config = _loop_config()
    base = core._normalize_market_frame(_market_frame(), config)
    indexes = np.arange(len(base) - 6, len(base) - 1)
    result = pd.DataFrame(
        {
            "trade_date": base["date"].iloc[indexes].dt.strftime("%Y%m%d").astype(int),
            "predicted_pct_change": 0.003,
            "predicted_close": base["close"].iloc[indexes].to_numpy() * 1.003,
            "real_pct_change": (base["close"].shift(-1) / base["close"] - 1.0)
            .iloc[indexes]
            .to_numpy(),
            "correct": True,
        }
    )

    states = core._build_regime_postprocess_frame(
        result_frame=result,
        base=base,
        diagnostics_frame=pd.DataFrame(),
    )

    assert states["sh_vol20_pct"].notna().all()
    assert not states["vol20_bucket"].isin(["missing", "insufficient_history"]).any()
    assert states["streak_state"].iloc[0] != "streak_flat_or_missing"


def test_regime_warms_predictions_and_diagnostic_label_is_final(
    monkeypatch,
    tmp_path: Path,
):
    captured: dict[str, pd.DataFrame] = {}
    original = core._apply_regime_postprocess

    def capture_frame(frame: pd.DataFrame, **kwargs):
        captured["frame"] = frame.copy()
        return original(frame, **kwargs)

    def force_flip(**kwargs):
        if kwargs["idx"] == 0:
            return None
        return {
            "state_column": "forced",
            "state_value": "forced",
            "side": "long",
            "rows": 1,
            "accuracy": 0.0,
        }

    monkeypatch.setattr(core, "_apply_regime_postprocess", capture_frame)
    monkeypatch.setattr(core, "_select_regime_flip_key", force_flip)
    loop_kwargs = _fast_rule_kwargs()
    diagnostics_path = tmp_path / "diagnostics.csv"
    loop_kwargs["diagnostics_output_path"] = str(diagnostics_path)
    result = core.loop_validate_prediction_results(
        _market_frame(90),
        config=_loop_config(),
        periods=2,
        signal_engine="volatility_rule",
        recent_failure_guard=True,
        recent_failure_window=1,
        recent_failure_degrade_threshold=1.0,
        recent_failure_invert_threshold=1.0,
        recent_failure_short_window=1,
        recent_failure_short_threshold=1.0,
        regime_postprocess=True,
        regime_postprocess_diagnostics_output_path=None,
        regime_postprocess_state_columns=("streak_state",),
        regime_postprocess_history_window=5,
        regime_postprocess_min_history=2,
        regime_postprocess_max_flip_rate=1.0,
        **loop_kwargs,
    )

    frame = captured["frame"]
    diagnostics = pd.read_csv(diagnostics_path)
    assert len(frame) == 12
    assert len(result) == 2
    assert diagnostics["regime_postprocess_flipped"].astype(int).eq(1).all()
    assert np.allclose(
        diagnostics["predicted_pct_change"],
        result["predicted_pct_change"],
    )
    expected_label = (result["predicted_pct_change"] > 0).astype(int)
    assert diagnostics["predicted_label"].astype(int).equals(expected_label)
    assert diagnostics["correct"].astype(bool).equals(result["correct"].astype(bool))
    assert frame["recent_failure_guard_applied"].astype(int).eq(1).any()
