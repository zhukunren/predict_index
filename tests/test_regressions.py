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
    ("akshare_data", "data_akshare.py", "akshare"),
    ("wind_data", "数据拉取脚本_wind.py", "WindPy"),
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
        "output_path": None,
        "diagnostics_output_path": None,
        "confidence_output_path": None,
        "confidence_summary_path": None,
        "progress": False,
    }


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
