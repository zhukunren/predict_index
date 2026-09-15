from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import 循环验证脚本 as core


def _signal(predictions: np.ndarray, labels: np.ndarray, idx: int):
    return core._nested_volatility_rule_signal(
        base=pd.DataFrame({"close": np.linspace(100.0, 105.0, len(labels))}),
        cache=core.NestedRuleCache(
            predictions=np.asarray(predictions, dtype=np.int8),
            rule_names=[f"rule_{i}" for i in range(len(predictions))],
            labels=np.asarray(labels, dtype=float),
            valid_mask=np.ones(len(labels), dtype=bool),
        ),
        idx=idx,
        calibration_window=10,
        min_calibration_rows=10,
        top_k=3,
    )


def test_historical_rule_accuracy_uses_current_weighted_vote():
    predictions = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        ]
    )

    signal = _signal(predictions, np.ones(11), 10)

    # The 90%-accurate rule outweighs both 60%-accurate rules together.
    assert signal.predicted_label == 1
    assert signal.calibration_accuracy == pytest.approx(0.9)
    assert signal.diagnostics["selected_rule_count"] == 3
    assert (predictions[:, :10].mean(axis=0) >= 0.5).mean() == pytest.approx(0.7)


def test_reversed_complement_is_one_expert_with_combined_rank_weight():
    strong = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1])
    weaker = np.asarray([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    repeated_signal = _signal(
        np.vstack([strong, 1 - strong, weaker]), np.ones(11), 10
    )

    assert repeated_signal.diagnostics["selected_rule_count"] == 2
    assert repeated_signal.diagnostics["selected_ranked_rule_count"] == 3
    assert repeated_signal.diagnostics["selected_expert_count"] == 2
    assert len(repeated_signal.rule_names) == 2
    assert "+" in repeated_signal.rule_names[0]


def test_rule_deduplication_and_signal_do_not_read_future_rows():
    history = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1])
    predictions = np.vstack([np.r_[history, 0, 0], np.r_[history, 1, 1]])
    labels = np.ones(13)
    original = _signal(predictions, labels, 10)
    changed_predictions = predictions.copy()
    changed_predictions[:, 11:] = 1 - changed_predictions[:, 11:]
    changed_labels = labels.copy()
    changed_labels[10:] = 0
    changed = _signal(changed_predictions, changed_labels, 10)
    prefix = _signal(predictions[:, :11], labels[:11], 10)

    assert original.diagnostics["selected_rule_count"] == 1
    assert changed.predicted_label == original.predicted_label
    assert changed.rule_names == original.rule_names
    assert changed.calibration_accuracy == original.calibration_accuracy
    assert prefix.predicted_label == original.predicted_label
    assert prefix.rule_names == original.rule_names
    assert prefix.calibration_accuracy == original.calibration_accuracy


def _confidence_frame() -> pd.DataFrame:
    rng = np.random.default_rng(47)
    return pd.DataFrame(
        {
            "trade_date": np.arange(1, 41),
            "confidence": rng.uniform(0.4, 0.9, 40),
            "correct": rng.random(40) < 0.6,
        }
    )


def _calibrate(frame: pd.DataFrame, primary_window: int | None):
    return core._apply_loop_confidence_calibration(
        result_frame=frame,
        validation_range=core._LoopValidationRange(0, 20, 39, 21, 40),
        options=core._ConfidenceCalibrationOptions(
            bin_edges=(0.0, 0.5, 0.7, 1.0),
            window=primary_window,
            min_rows=5,
            method="platt",
            compare_windows=(5, 12),
            rolling_windows=core._normalize_confidence_calibration_windows(
                primary_window, (5, 12)
            ),
            rolling_output_path=None,
            comparison_output_path=None,
        ),
    )


def test_comparison_winner_cannot_replace_configured_calibration_window():
    frame = _confidence_frame()
    comparison, _ = core.compare_rolling_confidence_calibration(
        frame,
        windows=(5, 12),
        min_rows=5,
        evaluation_start_trade_date=21,
        evaluation_end_trade_date=40,
        bin_edges=(0.0, 0.5, 0.7, 1.0),
    )
    winner = int(comparison.loc[comparison["selected"], "window"].iloc[0])
    primary_window = 12 if winner == 5 else 5

    actual, diagnostics = _calibrate(frame, primary_window)
    expected = core._apply_rolling_confidence_calibration(
        frame, window=primary_window, min_rows=5, method="platt"
    )

    pd.testing.assert_frame_equal(actual, expected)
    assert diagnostics.loc[diagnostics["used_for_output"], "window"].tolist() == [
        primary_window
    ]
    assert diagnostics.loc[diagnostics["selected"], "window"].tolist() == [winner]


@pytest.mark.parametrize("primary_window", [None, 0])
def test_comparison_cannot_enable_disabled_calibration(primary_window):
    frame = _confidence_frame()

    actual, diagnostics = _calibrate(frame, primary_window)

    pd.testing.assert_frame_equal(actual, frame)
    assert set(diagnostics["window"]) == {5, 12}
    assert not diagnostics["used_for_output"].any()


def test_loop_confidence_with_comparison_is_causal():
    frame = _confidence_frame()
    changed = frame.copy()
    changed.loc[20:, "correct"] = ~changed.loc[20:, "correct"]
    changed.loc[21:, "confidence"] = 1.0 - changed.loc[21:, "confidence"]

    original, _ = _calibrate(frame, 12)
    altered, _ = _calibrate(changed, 12)

    pd.testing.assert_frame_equal(
        original.loc[:20].filter(regex="calibrat"),
        altered.loc[:20].filter(regex="calibrat"),
    )
