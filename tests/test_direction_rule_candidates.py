from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import 循环验证脚本 as core
from tools.direction_rule_candidates import CANDIDATES, selector


def _signal(name, predictions, labels, *, idx=10, top_k=3, names=None):
    predictions = np.asarray(predictions, dtype=np.int8)
    return selector(core, name)(
        base=pd.DataFrame({"close": np.linspace(100.0, 105.0, len(labels))}),
        cache=core.NestedRuleCache(
            predictions=predictions,
            rule_names=names or [f"feature_{rule}>rolling_q0.5" for rule in range(len(predictions))],
            labels=np.asarray(labels, dtype=np.int8),
            valid_mask=np.ones(len(labels), dtype=bool),
        ),
        idx=idx, calibration_window=10, min_calibration_rows=10, top_k=top_k,
    )


def test_balanced_score_does_not_reward_the_majority_prior_as_skill():
    labels = [1] * 8 + [0, 0, 0]
    predictions = [[1] * 11, [1] * 5 + [0] * 6]
    assert _signal("accuracy_unique", predictions, labels, top_k=1).predicted_label == 1
    assert _signal("balanced_unique", predictions, labels, top_k=1).predicted_label == 0


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_duplicates_and_reversed_complements_do_not_add_weight(candidate):
    labels = np.array([1, 0] * 5 + [0])
    strong = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    other = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    original = _signal(candidate, [strong, other], labels)
    duplicated = _signal(candidate, [strong, other, strong, 1 - strong, 1 - other], labels)
    assert duplicated.predicted_label == original.predicted_label
    assert duplicated.predicted_return == original.predicted_return
    assert duplicated.calibration_accuracy == original.calibration_accuracy
    assert duplicated.diagnostics["selected_expert_count"] == 2


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_selection_does_not_use_current_outcome_or_future_features(candidate):
    rng = np.random.default_rng(82)
    labels = rng.integers(0, 2, size=16)
    predictions = rng.integers(0, 2, size=(8, 16))
    original = _signal(candidate, predictions, labels)
    labels[10:] = 1 - labels[10:]
    predictions[:, 11:] = 1 - predictions[:, 11:]
    changed = _signal(candidate, predictions, labels)
    prefix = _signal(candidate, predictions[:, :11], labels[:11])
    assert changed == original
    assert prefix.predicted_label == original.predicted_label
    assert prefix.rule_names == original.rule_names
    assert prefix.calibration_accuracy == original.calibration_accuracy


def test_feature_limit_prevents_multiple_thresholds_of_one_feature():
    labels = np.array([1, 0] * 5 + [0])
    predictions = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
    ])
    names = ["volatility>rolling_q0.5", "volatility>rolling_q0.6", "position>rolling_q0.5"]
    result = _signal("balanced_features", predictions, labels, names=names)
    assert result.diagnostics["selected_expert_count"] == 2
    assert sum("volatility" in name for name in result.rule_names) == 1
