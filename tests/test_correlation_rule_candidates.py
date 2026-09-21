from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import matthews_corrcoef

import tushare_prediction_pipeline as pipeline
from tools.correlation_rule_candidates import correlations, selector


core = pipeline.prediction_core


def compute(predictions, labels, idx=100):
    cache = core.NestedRuleCache(predictions=np.asarray(predictions), labels=np.asarray(labels),
                                rule_names=[f"rule_{i}" for i in range(len(predictions))],
                                valid_mask=np.ones(len(labels), dtype=bool))
    return selector(core, "mcc_ranked_rules")(
        base=pd.DataFrame({"close": 100 + np.arange(len(labels)) * 0.01}), cache=cache,
        idx=idx, calibration_window=100, min_calibration_rows=30, top_k=5,
    )


def test_vectorized_correlation_matches_standard_estimator_with_missing_rules():
    rng = np.random.default_rng(842)
    predictions = rng.integers(-1, 2, size=(20, 130))
    labels = rng.integers(0, 2, size=130)
    scores, counts = correlations(predictions, labels)
    for index, prediction in enumerate(predictions):
        valid = prediction >= 0
        assert counts[index] == valid.sum()
        assert scores[index] == pytest.approx(matthews_corrcoef(labels[valid], prediction[valid]))


def test_constant_majority_predictions_have_no_directional_correlation():
    labels = np.tile([1] * 8 + [0] * 2, 11)
    constant = np.ones(len(labels), dtype=int)
    useful = labels.copy()
    useful[100] = 0
    scores, _ = correlations(np.vstack([constant, useful]), labels)
    assert scores[0] == 0
    assert scores[1] > 0
    result = compute(np.vstack([constant, useful]), labels)
    assert result.predicted_label == 0
    assert result.diagnostics["selected_ranked_rule_count"] == 1


def test_signal_excludes_current_outcome_and_future_predictions():
    rng = np.random.default_rng(86)
    labels = rng.integers(0, 2, size=130)
    predictions = rng.integers(0, 2, size=(8, 130))
    original = predictions.copy()
    result = compute(predictions, labels)
    prefix = compute(predictions[:, :101], labels[:101])
    assert result == prefix
    np.testing.assert_array_equal(predictions, original)
    labels[100:] = 1 - labels[100:]
    predictions[:, 101:] = 1 - predictions[:, 101:]
    assert compute(predictions, labels) == result


def test_negative_correlation_reverses_its_direction_without_changing_score():
    labels = np.array([0, 1] * 60)
    result = compute(np.vstack([labels]), labels)
    opposite = compute(np.vstack([1 - labels]), labels)
    assert result.predicted_label == opposite.predicted_label
    assert result.predicted_return == opposite.predicted_return
    assert result.calibration_accuracy == opposite.calibration_accuracy
    assert result.diagnostics["top_rule_correlation"] == opposite.diagnostics["top_rule_correlation"] == 1
