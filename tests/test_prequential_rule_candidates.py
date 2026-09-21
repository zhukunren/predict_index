from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tushare_prediction_pipeline as pipeline
from tools.prequential_rule_candidates import CANDIDATES, ensemble_history, prequential_paths, prior_sum, selector


core = pipeline.prediction_core


def cache_for(predictions, labels, mask=None):
    labels = np.asarray(labels, dtype=np.int8)
    return core.NestedRuleCache(predictions=np.asarray(predictions, dtype=np.int8), labels=labels,
                                valid_mask=np.ones(len(labels), dtype=bool) if mask is None else mask,
                                rule_names=[f"feature_{i}>rolling_q0.5" for i in range(len(predictions))])


def test_rolling_sum_excludes_current_observation():
    values = np.arange(20).reshape(2, 10)
    actual = prior_sum(values, 3)
    expected = np.array([[row[max(0, i - 3):i].sum() for i in range(10)] for row in values])
    np.testing.assert_array_equal(actual, expected)


def test_rule_orientation_is_saved_at_each_past_signal():
    labels = np.zeros(12, dtype=int)
    labels[6:] = 1
    cache = cache_for([np.ones(12)], labels)
    paths = prequential_paths(cache, 4, 3)
    np.testing.assert_array_equal(paths[0], [-1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    # Today's orientation cannot rewrite the predictions before the regime changed.
    assert paths[0, 3:6].tolist() == [0, 0, 0]
    cache.valid_mask[3:7] = False
    unavailable = prequential_paths(cache, 4, 3)
    assert unavailable[0, 7] == -1


@pytest.mark.parametrize("name", CANDIDATES)
def test_predictions_and_rankings_match_prefix_and_ignore_future_outcomes(name):
    rng = np.random.default_rng(925)
    labels = rng.integers(0, 2, 330)
    predictions = np.vstack([np.where(rng.random(330) < chance, labels, 1 - labels) for chance in (0.6, 0.7, 0.8)])
    cache = cache_for(predictions, labels)
    original_predictions = cache.predictions.copy()
    full = ensemble_history(cache, name, 120, 30, 5)
    current = 270
    prefix_labels = labels[:current + 1].copy()
    prefix_labels[-1] = 1 - prefix_labels[-1]
    prefix_mask = np.ones(current + 1, dtype=bool)
    prefix_mask[-1] = False
    prefix = ensemble_history(cache_for(predictions[:, :current + 1], prefix_labels, prefix_mask), name, 120, 30, 5)
    for key in full:
        np.testing.assert_array_equal(full[key][..., :current + 1], prefix[key])
    changed_labels, changed_predictions = labels.copy(), predictions.copy()
    changed_labels[current:] = 1 - changed_labels[current:]
    changed_predictions[:, current + 1:] = 1 - changed_predictions[:, current + 1:]
    changed = ensemble_history(cache_for(changed_predictions, changed_labels), name, 120, 30, 5)
    for key in full:
        np.testing.assert_array_equal(full[key][..., :current + 1], changed[key][..., :current + 1])
    np.testing.assert_array_equal(cache.predictions, original_predictions)


@pytest.mark.parametrize("name", CANDIDATES)
def test_complete_signal_calibrates_from_prior_ensemble_and_preserves_rank_weights(name):
    labels = np.tile([0, 1], 170)
    cache = cache_for([labels, 1 - labels], labels)
    base = pd.DataFrame({"close": 100 + np.arange(len(labels)) * 0.1})
    kwargs = dict(base=base, cache=cache, idx=280, calibration_window=120, min_calibration_rows=30, top_k=5)
    signal = selector(core, name)(**kwargs)
    assert signal.diagnostics["rule_mode"] == name
    assert signal.diagnostics["selected_ranked_rule_count"] == 2
    assert signal.diagnostics["selected_expert_count"] == 1
    assert signal.diagnostics["prequential_last_calibration_index"] == 279
    expected = ensemble_history(cache, name, 120, 30, 5)
    history = np.arange(160, 280)
    assert signal.calibration_accuracy == float((expected["prediction"][history] == labels[history]).mean())
    prefix = dict(kwargs, base=base.iloc[:281], cache=cache_for(cache.predictions[:, :281], labels[:281]))
    assert selector(core, name)(**prefix) == signal


def test_balanced_ranking_does_not_reward_constant_up_expert():
    labels = np.tile([1] * 8 + [0] * 2, 35)
    cache = cache_for([np.ones(len(labels)), labels.copy()], labels)
    results = ensemble_history(cache, "prequential_balanced_rules", 120, 30, 5)
    selected = results["ranked"][:, -1][results["weights"][:, -1] > 0]
    assert selected.tolist() == [1]


def test_first_ensemble_predictions_are_emitted_before_confidence_is_fitted():
    labels = np.tile([0, 1], 80)
    cache = cache_for([labels.copy()], labels)
    base = pd.DataFrame({"close": 100 + np.arange(len(labels)) * 0.1})
    compute = selector(core, "prequential_accuracy_rules")
    for idx in (60, 75):
        signal = compute(base=base, cache=cache, idx=idx, calibration_window=120, min_calibration_rows=30, top_k=5)
        assert signal.diagnostics["rule_mode"] == "prequential_accuracy_rules"
        assert signal.calibration_accuracy == 0.5
        assert signal.calibration_rows == idx - 60


@pytest.mark.parametrize("name", CANDIDATES)
def test_warmup_uses_original_selector(name):
    labels = np.tile([0, 1], 30)
    cache = cache_for([labels.copy()], labels)
    base = pd.DataFrame({"close": 100 + np.arange(len(labels)) * 0.1})
    kwargs = dict(base=base, cache=cache, idx=40, calibration_window=120, min_calibration_rows=30, top_k=5)
    assert selector(core, name)(**kwargs) == core._nested_volatility_rule_signal(**kwargs)
