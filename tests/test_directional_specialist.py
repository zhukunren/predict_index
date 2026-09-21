import numpy as np
import pandas as pd
import pytest

import tushare_prediction_pipeline as pipeline
from tools import downside_specialist as downside
from tools.directional_specialist import directional_probabilities, directional_predictions
from test_downside_specialist import inputs


@pytest.mark.parametrize("name", ("downside_logistic", "downside_shallow"))
def test_real_directional_models_change_both_sides_and_have_causal_prefixes(name, monkeypatch):
    monkeypatch.setattr(downside, "COMMON", downside.COMMON | {"min_train_rows": 20, "train_window": 100})
    frame, features = inputs(240)
    original = frame.copy(deep=True)
    probabilities = directional_probabilities(frame, features, name)
    result = directional_predictions(pipeline.prediction_core, frame, probabilities, .55)
    assert (result.correction_selected & frame.predicted_label.eq(1)).sum() > 5
    assert (result.correction_selected & frame.predicted_label.eq(0)).sum() > 5
    reference = downside.downside_probabilities(frame, features, name)
    for suffix in ("probability", "training_rows", "last_training_date"):
        up = frame.predicted_label.eq(1)
        np.testing.assert_array_equal(probabilities.loc[up, f"error_{suffix}"], reference.loc[up, f"downside_{suffix}"])
    assert probabilities.loc[frame.predicted_label.eq(0), "error_training_rows"].max() == 25
    columns = ["predicted_label", "predicted_pct_change", "predicted_close", "calibrated_confidence",
               "correction_selected", *probabilities.columns.drop("trade_date")]
    for current in (130, 171):
        prefix = frame.iloc[:current + 1].copy()
        prefix.loc[current, "real_pct_change"] = np.nan
        prefix["correct"] = prefix.correct.astype("boolean")
        prefix.loc[current, "correct"] = None
        p = directional_probabilities(prefix, features.iloc[:current + 1], name)
        replay = directional_predictions(pipeline.prediction_core, prefix, p, .55)
        pd.testing.assert_frame_equal(result.loc[:current, columns], replay[columns], check_exact=True)
    changed = frame.copy()
    changed.loc[130:, "real_pct_change"] *= -1
    changed_features = features.copy()
    changed_features.loc[131:, "x"] *= -100
    p = directional_probabilities(changed, changed_features, name)
    pd.testing.assert_frame_equal(probabilities.iloc[:131], p.iloc[:131], check_exact=True)
    # Opposite-direction samples must not enter a branch's training set.
    changed = frame.copy()
    changed.loc[frame.predicted_label.eq(1), "real_pct_change"] *= -1
    p = directional_probabilities(changed, features, name)
    pd.testing.assert_frame_equal(probabilities.loc[frame.predicted_label.eq(0)], p.loc[frame.predicted_label.eq(0)], check_exact=True)
    pd.testing.assert_frame_equal(frame, original, check_exact=True)


def test_zero_returns_are_correct_for_down_branch_and_unknown_rows_stay_unknown(monkeypatch):
    import tools.directional_specialist as module

    frame, features = inputs(5)
    frame["real_pct_change"] = [0., .01, -.01, np.nan, np.inf]
    captured = []

    def capture(champion, features, name, **kwargs):
        captured.append(champion.copy())
        return pd.DataFrame({"trade_date": champion.trade_date, "downside_probability": 0.,
                             "downside_training_rows": 0, "downside_last_training_date": 0})

    monkeypatch.setattr(module, "downside_probabilities", capture)
    directional_probabilities(frame, features, "downside_logistic")
    assert captured[1].real_pct_change.iloc[:3].tolist() == [1., -1., 1.]
    assert captured[1].real_pct_change.iloc[3:].isna().all()
    assert captured[1].predicted_label.tolist() == (1 - frame.predicted_label).tolist()


def test_missing_features_do_not_train_or_correct_and_opposite_branch_is_not_a_fallback(monkeypatch):
    monkeypatch.setattr(downside, "COMMON", downside.COMMON | {"min_train_rows": 20, "train_window": 100})
    frame, features = inputs(240)
    available = frame.predicted_label.eq(1).to_numpy()
    features.loc[~available, "x"] = np.nan
    probabilities = directional_probabilities(frame, features, "downside_logistic", available=available)
    assert probabilities.loc[~available, "error_training_rows"].eq(0).all()
    assert probabilities.loc[~available, "error_probability"].eq(0).all()
    result = directional_predictions(pipeline.prediction_core, frame, probabilities, .55)
    assert not result.loc[~available, "correction_selected"].any()


def test_shared_threshold_respects_both_directions_fitted_state_and_input_identity():
    frame, _ = inputs(5)
    probability = pd.DataFrame({"trade_date": frame.trade_date, "incumbent_label": frame.predicted_label,
                                "error_probability": [.65, .65, .64, .99, .64],
                                "error_training_rows": [120, 120, 120, 0, 120],
                                "error_last_training_date": 20221230})
    result = directional_predictions(pipeline.prediction_core, frame, probability, .65)
    assert result.predicted_label.tolist() == [1, 0, 1, 1, 0]
    assert result.correction_selected.tolist() == [True, True, False, False, False]
    with pytest.raises(ValueError, match="identical"):
        directional_predictions(pipeline.prediction_core, frame, probability.assign(incumbent_label=1), .65)
    bad = probability.copy()
    bad.loc[0, "error_last_training_date"] = bad.trade_date.iloc[0]
    with pytest.raises(ValueError, match="precede"):
        directional_predictions(pipeline.prediction_core, frame, bad, .65)
    with pytest.raises(ValueError, match="finite"):
        directional_predictions(pipeline.prediction_core, frame, probability.assign(error_probability=np.nan), .65)
