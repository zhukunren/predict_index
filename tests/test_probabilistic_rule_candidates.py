from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import 循环验证脚本 as core
from tools.probabilistic_rule_candidates import CANDIDATES, selector


def _signal(candidate, predictions, labels, *, idx=10, top_k=3):
    cache = core.NestedRuleCache(
        predictions=np.asarray(predictions, dtype=np.int8),
        labels=np.asarray(labels, dtype=np.int8),
        rule_names=[f"feature_{number}>rolling_q0.5" for number in range(len(predictions))],
        valid_mask=np.ones(len(labels), dtype=bool),
    )
    return selector(core, candidate)(
        base=pd.DataFrame({"close": 100 + np.arange(len(labels))}),
        cache=cache, idx=idx, calibration_window=10, min_calibration_rows=10, top_k=top_k,
    )


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_conditional_evidence_can_correct_a_bullish_rule(candidate):
    labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    predictions = [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]]
    # This rule is 60% correct overall, but its bullish bucket is 3/7 correct.
    result = _signal(candidate, predictions, labels, top_k=1)
    assert result.predicted_label == 0
    assert result.diagnostics["probability_up"] < 0.5
    assert 0.5 <= result.calibration_accuracy <= 1


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_probability_aggregation_ignores_current_outcome_and_future_rows(candidate):
    random = np.random.default_rng(531)
    predictions = random.integers(0, 2, size=(7, 18))
    labels = random.integers(0, 2, size=18)
    original = _signal(candidate, predictions, labels)
    labels[10:] = 1 - labels[10:]
    predictions[:, 11:] = 1 - predictions[:, 11:]
    changed = _signal(candidate, predictions, labels)
    prefix = _signal(candidate, predictions[:, :11], labels[:11])
    assert original == changed == prefix


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_complementary_rules_keep_the_same_conditional_probability(candidate):
    labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    rule = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1])
    original = _signal(candidate, [rule], labels)
    complement = _signal(candidate, [rule, 1 - rule], labels)
    assert original.predicted_label == complement.predicted_label
    assert original.diagnostics["probability_up"] == pytest.approx(complement.diagnostics["probability_up"])
    assert complement.diagnostics["selected_expert_count"] == 1
