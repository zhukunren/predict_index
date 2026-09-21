from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import 循环验证脚本 as core
from tools.causal_direction_guard import apply_guard, wilson_lower
from tools.evaluate_direction_bias import compare_frames


def _frame(label, actual):
    raw = 0.003 if label else -0.003
    return pd.DataFrame({
        "trade_date": pd.bdate_range("2024-01-02", periods=len(actual)).strftime("%Y%m%d").astype(int),
        "predicted_label": [label] * len(actual),
        "predicted_pct_change": [raw] * len(actual),
        "uncalibrated_predicted_return": [raw] * len(actual),
        "predicted_close": [100 * (1 + raw)] * len(actual),
        "confidence": [0.6] * len(actual),
        "calibrated_confidence": [0.6] * len(actual),
        "real_pct_change": actual,
        "correct": [bool(label == (value > 0)) for value in actual],
    })


def test_guard_requires_evidence_and_preserves_champion_history():
    actual = [-0.01] * 80
    champion, challenger = _frame(1, actual), _frame(0, actual)
    original = champion.copy(deep=True)
    result = apply_guard(core, champion, {"alternative": challenger})
    assert result.loc[:19, "guard_source"].eq("baseline").all()
    assert result.loc[20:, "guard_source"].eq("alternative").all()
    assert result.loc[30, "guard_history_rows"] == 30
    assert result.loc[70, "predicted_label"] == 0
    pd.testing.assert_frame_equal(champion, original)
    assert wilson_lower(15, 20) > 0.5
    assert wilson_lower(12, 20) < 0.5


def test_current_and_future_outcomes_cannot_affect_a_correction():
    actual = [-0.01] * 100
    champion, challenger = _frame(1, actual), _frame(0, actual)
    before = apply_guard(core, champion, {"alternative": challenger})
    champion.loc[70:, "real_pct_change"] = 0.5
    challenger.loc[70:, "real_pct_change"] = 0.5
    after = apply_guard(core, champion, {"alternative": challenger})
    columns = ["predicted_label", "predicted_pct_change", "predicted_close", "confidence",
               "calibrated_confidence", "guard_source", "guard_history_rows", "guard_lower_bound"]
    pd.testing.assert_frame_equal(before.loc[:70, columns], after.loc[:70, columns])
    prefix = apply_guard(core, champion.iloc[:71], {"alternative": challenger.iloc[:71]})
    pd.testing.assert_frame_equal(after.loc[:70, columns], prefix[columns])


def test_recent_evidence_and_matching_dates_are_required():
    actual = [-0.01] * 70 + [0.01] * 70
    champion, challenger = _frame(1, actual), _frame(0, actual)
    result = apply_guard(core, champion, {"alternative": challenger})
    assert result.iloc[-1]["guard_source"] == "baseline"
    with pytest.raises(ValueError, match="ordered signal dates"):
        apply_guard(core, champion, {"alternative": challenger.iloc[::-1]})
    challenger.loc[0, "real_pct_change"] = 0.5
    with pytest.raises(ValueError, match="identical outcomes"):
        apply_guard(core, champion, {"alternative": challenger})


def test_non_regression_gate_rejects_no_correction_and_missing_dates():
    actual = np.tile([-0.01, 0.01], 160)
    baseline = _frame(1, actual)
    report = compare_frames(baseline, baseline.copy())
    assert report["passed"] is False
    failed = {item["metric"] for item in report["checks"] if not item["passed"]}
    assert failed == {"up_rate_gap", "longest_up"}
    with pytest.raises(ValueError, match="alignment"):
        compare_frames(baseline, baseline.iloc[1:])
