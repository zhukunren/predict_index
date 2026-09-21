from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tools import nested_downside as nested
from test_nested_downside import CONFIG, inputs


def both_directions(rows=145):
    champion, features = inputs(rows)
    champion.loc[1::2, "predicted_label"] = 0
    champion.loc[::13, "real_pct_change"] = 0
    return champion, features


@pytest.mark.parametrize("paired", [False, True])
def test_shared_error_learner_uses_both_sides_and_has_causal_prefixes(paired):
    champion, features = both_directions()
    config = replace(CONFIG, require_paired_nonregression=paired, correction_threshold=.5)
    untouched = champion.copy(deep=True)
    full, attempts = nested.nested_error_probabilities(champion, features, ("trend",), config=config)
    assert full.error_training_rows.max() == CONFIG.train_window
    assert full.loc[champion.predicted_label.eq(0), "error_training_rows"].max() > 0
    assert full.loc[champion.predicted_label.eq(1), "error_training_rows"].max() > 0
    current = 100
    prefix = champion.iloc[:current+1].copy()
    prefix.loc[current, "real_pct_change"] = np.nan
    replay, short_attempts = nested.nested_error_probabilities(prefix, features.iloc[:current+1], ("trend",), config=config)
    pd.testing.assert_frame_equal(full.iloc[:current+1], replay, check_exact=True)
    cutoff = int(champion.trade_date.iloc[current])
    assert short_attempts == [a for a in attempts if a["signal_date"] <= cutoff]
    champion.loc[current:, "real_pct_change"] = .1
    features.loc[current+1:, ["price", "trend"]] *= -10
    changed, changed_attempts = nested.nested_error_probabilities(champion, features, ("trend",), config=config)
    pd.testing.assert_frame_equal(full.iloc[:current+1], changed.iloc[:current+1], check_exact=True)
    assert short_attempts == [a for a in changed_attempts if a["signal_date"] <= cutoff]
    pd.testing.assert_frame_equal(prefix.iloc[:-1], untouched.iloc[:current], check_exact=True)


def test_error_training_labels_treat_flat_return_as_down_and_exclude_unknown(monkeypatch):
    champion, features = both_directions(70)
    champion.loc[51, "real_pct_change"] = np.nan
    targets = []
    real_fit = nested._fit
    def record(values, target, c, max_iter):
        targets.append(target.copy())
        return real_fit(values, target, c, max_iter)
    monkeypatch.setattr(nested, "_fit", record)
    config = replace(CONFIG, refit_interval=100, inverse_penalties=(.1,), correction_threshold=.5)
    nested.nested_error_probabilities(champion, features, config=config)
    expected = champion.predicted_label.ne(champion.real_pct_change.gt(0)).astype(int).to_numpy()
    # One fit at row 50: both original directions supply the 30/40/50 rows.
    assert [len(target) for target in targets] == [30, 40, 50]
    for target, count in zip(targets, (30, 40, 50), strict=True):
        np.testing.assert_array_equal(target, expected[:count])
    assert expected[13] == 0  # Original down plus zero actual return is correct.
    assert expected[26] == 1  # Original up plus zero actual return is wrong.


def test_direction_interactions_preserve_features_and_do_not_read_targets():
    champion, features = both_directions()
    before = features.copy(deep=True)
    extended, optional = nested.directional_error_features(champion, features, ("trend",))
    np.testing.assert_array_equal(extended.incumbent_direction, 2*champion.predicted_label-1)
    np.testing.assert_array_equal(extended.price__direction, features.price*(2*champion.predicted_label-1))
    assert optional == ("trend", "trend__direction")
    champion["real_pct_change"] = np.nan
    after, _ = nested.directional_error_features(champion, features, ("trend",))
    pd.testing.assert_frame_equal(extended, after, check_exact=True)
    pd.testing.assert_frame_equal(before, features, check_exact=True)


def test_bidirectional_validation_assigns_hits_to_the_correct_actual_class():
    actual = np.array([-1., 1., 1., -1., -1., 0.])
    incumbent = np.array([1, 0, 1, 0, 1, 0])
    result = nested.paired_validation(actual, incumbent, np.arange(6), np.array([.9,.9,.1,.1,.1,.9]), .5)
    assert result["corrected"] == 2 and result["damaged"] == 1
    assert result["up_hit_delta"] == 1 and result["down_hit_delta"] == 0
    assert result["accuracy_delta"] == pytest.approx(1/6)
    assert result["balanced_accuracy_delta"] == pytest.approx(.25)


def test_both_direction_predictions_and_unknown_outcomes_are_consistent():
    import tushare_prediction_pipeline as pipeline
    champion, _ = both_directions(85)
    champion["predicted_pct_change"] = np.where(champion.predicted_label.eq(1), .002, -.002)
    champion["uncalibrated_predicted_return"] = champion.predicted_pct_change
    champion["predicted_close"] = 100*(1+champion.predicted_pct_change)
    champion["confidence"] = .6; champion["calibrated_confidence"] = .6
    champion["correct"] = champion.predicted_label.eq(champion.real_pct_change.gt(0)).astype("boolean")
    champion.loc[84, ["real_pct_change", "correct"]] = [np.nan, pd.NA]
    probabilities = pd.DataFrame({"trade_date": champion.trade_date, "error_probability": .7,
                                  "error_training_rows": 0, "error_last_training_date": 0,
                                  "nested_last_validation_date": 0, "nested_selected_threshold": 0.,
                                  "incumbent_label": champion.predicted_label})
    probabilities.loc[75:76, ["error_training_rows", "error_last_training_date", "nested_last_validation_date", "nested_selected_threshold"]] = [50, int(champion.trade_date.iloc[70]), int(champion.trade_date.iloc[70]), .5]
    result = nested.selected_error_predictions(pipeline.prediction_core, champion, probabilities)
    assert result.predicted_label.iloc[75] == 1
    assert result.predicted_label.iloc[76] == 0
    assert result.correction_selected.sum() == 2
    assert pd.isna(result.correct.iloc[-1])
    nonzero = result.predicted_pct_change.ne(0)
    assert result.loc[nonzero, "predicted_pct_change"].gt(0).eq(result.loc[nonzero, "predicted_label"].eq(1)).all()
    probabilities.loc[75, "incumbent_label"] = 1
    with pytest.raises(ValueError, match="incumbent direction"):
        nested.selected_error_predictions(pipeline.prediction_core, champion, probabilities)
