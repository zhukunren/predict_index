from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tushare_prediction_pipeline as pipeline
from tools.duration_rule_candidates import conditional_probability, selector, state_durations


def test_state_durations_reset_on_transition_and_missing():
    actual = state_durations(np.array([[-1, 1, 1, 0, 0, -1, 0, 0, 0]]))
    np.testing.assert_array_equal(actual, [[0, 1, 2, 1, 2, 0, 1, 2, 3]])


def test_state_duration_is_prefix_causal_and_complement_invariant():
    states = np.array([[0, 0, 1, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0]])
    full = state_durations(states)
    np.testing.assert_array_equal(full[:, :6], state_durations(states[:, :6]))
    np.testing.assert_array_equal(full[0], full[1])


def test_reliability_can_change_within_the_same_binary_state():
    states = np.array([1] * 25 + [0, 1])
    ages = state_durations(states[None, :])[0]
    labels = np.array([1] * 10 + [0] * 17)
    probability, rows, ordinary = conditional_probability(states, ages, labels, np.arange(26), 26)
    assert rows == 1
    assert probability > ordinary
    assert probability == pytest.approx((1 + 10 * 11 / 27) / 11)


def test_conditional_estimate_rejects_unresolved_history():
    with pytest.raises(ValueError, match="current or future"):
        conditional_probability(np.ones(5), np.arange(1, 6), np.ones(5), np.arange(5), 4)


def _signal(predictions, labels, idx):
    core = pipeline.prediction_core
    cache = core.NestedRuleCache(
        predictions=np.asarray(predictions, dtype=np.int8), labels=np.asarray(labels, dtype=np.int8),
        rule_names=[f"feature_{j}>rolling_q0.5" for j in range(len(predictions))],
        valid_mask=np.ones(len(labels), dtype=bool),
    )
    return selector(core)(base=pd.DataFrame({"close": 100 + np.arange(len(labels))}), cache=cache,
                          idx=idx, calibration_window=30, min_calibration_rows=10, top_k=5)


def test_selector_ignores_current_and_future_outcomes_and_future_states():
    rng = np.random.default_rng(1251)
    states = rng.integers(0, 2, size=(8, 50))
    labels = rng.integers(0, 2, size=50)
    original = _signal(states, labels, 35)
    labels[35:] = 1 - labels[35:]
    states[:, 36:] = 1 - states[:, 36:]
    changed = _signal(states, labels, 35)
    prefix = _signal(states[:, :36], labels[:36], 35)
    assert original == changed == prefix


def test_complements_preserve_probability_and_collapsed_expert_count():
    states = np.tile([1, 1, 1, 0, 0], 9)
    labels = np.tile([1, 0, 1, 0, 1], 9)
    one = _signal([states], labels, 40)
    pair = _signal([states, 1 - states], labels, 40)
    assert one.predicted_label == pair.predicted_label
    assert one.diagnostics["probability_up"] == pytest.approx(pair.diagnostics["probability_up"])
    assert pair.diagnostics["selected_expert_count"] == 1
