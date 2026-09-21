from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tushare_prediction_pipeline as pipeline
from tools.validated_rule_candidates import CANDIDATES, rule_plan, selector


core = pipeline.prediction_core


def cache_for(predictions, labels):
    return core.NestedRuleCache(
        predictions=np.asarray(predictions, dtype=np.int8), labels=np.asarray(labels, dtype=np.int8),
        rule_names=[f"feature_{i}>rolling_q0.5" for i in range(len(predictions))],
        valid_mask=np.ones(len(labels), dtype=bool),
    )


def plan(cache, name, *, idx=180, top_k=5):
    return rule_plan(cache, idx, 120, 30, top_k, name)


@pytest.mark.parametrize("name", CANDIDATES)
def test_rules_must_keep_their_learned_orientation_in_later_history(name):
    labels = np.array([0, 1] * 96)
    failed = labels.copy()
    failed[120:] = 1 - failed[120:]
    stable = labels.copy()
    stable[np.arange(0, 120, 4)] = 1 - stable[np.arange(0, 120, 4)]
    cache = cache_for([failed, stable], labels)
    original = cache.predictions.copy()
    result = plan(cache, name)
    assert result["ranked_rules"] == [0, 1]
    assert result["accepted_rules"] == [1]
    assert result["reverse"] == [False]
    assert result["ranking_end_index"] == 119
    assert result["validation_start_index"] == 120
    assert result["validation_end_index"] == 179
    assert result["predicted_label"] == 0
    np.testing.assert_array_equal(cache.predictions, original)
    # A failed rule cannot be reversed using the validation outcomes.
    assert plan(cache_for([failed], labels), name) is None
    labels[120:180] = 1 - labels[120:180]
    changed = plan(cache_for([failed, stable], labels), name)
    assert changed["ranked_rules"] == [0, 1]
    assert changed["accepted_rules"] == [0]


@pytest.mark.parametrize("name", CANDIDATES)
def test_majority_only_rule_does_not_pass_directional_validation(name):
    labels = np.tile([1] * 8 + [0] * 2, 20)
    constant = np.ones(200, dtype=int)
    useful = labels.copy()
    useful[np.arange(0, 120, 4)] = 1 - useful[np.arange(0, 120, 4)]
    useful[180] = 0
    result = plan(cache_for([constant, useful], labels), name)
    assert result["accepted_rules"] == [1]
    assert result["predicted_label"] == 0
    assert plan(cache_for([constant], labels), name) is None


@pytest.mark.parametrize("name", CANDIDATES)
def test_complete_signal_has_prefix_parity_and_excludes_future_outcomes(name):
    labels = np.array([0, 1] * 101)
    predictions = np.vstack([labels.copy(), 1 - labels])
    cache = cache_for(predictions, labels)
    base = pd.DataFrame({"close": 100 + np.arange(len(labels)) * 0.01})
    compute = selector(core, name)
    kwargs = {"idx": 180, "calibration_window": 120, "min_calibration_rows": 30, "top_k": 5}
    original = compute(base=base, cache=cache, **kwargs)
    assert original.diagnostics["selected_expert_count"] == 1
    prefix = compute(base=base.iloc[:181], cache=cache_for(predictions[:, :181], labels[:181]), **kwargs)
    assert original == prefix
    labels[180:] = 1 - labels[180:]
    predictions[:, 181:] = 1 - predictions[:, 181:]
    base.loc[181:, "close"] *= 10
    changed = compute(base=base, cache=cache_for(predictions, labels), **kwargs)
    assert original == changed


@pytest.mark.parametrize("name", CANDIDATES)
def test_insufficient_validation_classes_and_warmup_use_original_selector(name):
    labels = np.array([0, 1] * 100)
    cache = cache_for([labels.copy()], labels)
    cache.valid_mask[120:180] = cache.labels[120:180] == 1
    assert plan(cache, name) is None
    assert plan(cache, name, idx=80) is None
    base = pd.DataFrame({"close": 100 + np.arange(len(labels)) * 0.01})
    kwargs = dict(base=base, cache=cache, idx=80, calibration_window=120, min_calibration_rows=30, top_k=5)
    assert selector(core, name)(**kwargs) == core._nested_volatility_rule_signal(**kwargs)
