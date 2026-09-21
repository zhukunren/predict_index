import numpy as np
import pandas as pd
import pytest

from tools import nested_downside as nested


CONFIG = nested.NestedDownsideConfig(train_window=120, min_train_rows=30,
                                   validation_block_rows=10, validation_blocks=2,
                                   refit_interval=15, inverse_penalties=(0.01, 0.1))


def inputs(rows=125):
    rng = np.random.default_rng(817)
    base = rng.normal(size=rows)
    extra = rng.normal(size=rows)
    dates = pd.bdate_range("2023-01-03", periods=rows).strftime("%Y%m%d").astype(int)
    actual = np.where(base + 0.5 * extra > 0, -0.01, 0.01)
    champion = pd.DataFrame({"trade_date": dates, "predicted_label": np.ones(rows, dtype=int), "real_pct_change": actual})
    features = pd.DataFrame({"trade_date": dates, "price": base, "trend": extra})
    return champion, features


def test_validation_blocks_follow_training_and_exclude_later_observations():
    history = np.arange(70)
    blocks = nested.validation_blocks(history, CONFIG)
    assert len(blocks) == 2
    for train, validation in blocks:
        assert train[-1] < validation[0]
        assert len(train) >= CONFIG.min_train_rows
        assert len(validation) == CONFIG.validation_block_rows
    assert set(blocks[0][1]).isdisjoint(blocks[1][1])
    assert blocks[-1][1][-1] == history[-1]


@pytest.mark.parametrize("policy", ["probability", "paired", "threshold"])
def test_full_and_unresolved_prefix_predictions_and_selections_match(policy):
    config = replace(CONFIG, require_paired_nonregression=policy != "probability",
                     threshold_candidates=(.5, .6) if policy == "threshold" else ())
    champion, features = inputs()
    before_champion, before_features = champion.copy(deep=True), features.copy(deep=True)
    full, attempts = nested.nested_downside_probabilities(champion, features, ("trend",), config=config)
    current = 90
    prefix = champion.iloc[:current+1].copy()
    prefix.loc[current, "real_pct_change"] = np.nan
    replay, prefix_attempts = nested.nested_downside_probabilities(prefix, features.iloc[:current+1], ("trend",), config=config)
    pd.testing.assert_frame_equal(full.iloc[:current+1], replay, check_exact=True)
    assert prefix_attempts == [a for a in attempts if a["signal_date"] <= int(champion.trade_date.iloc[current])]
    assert full.downside_training_rows.max() > 0
    assert all(a["last_training_signal_date"] < a["signal_date"] for a in attempts)
    for a in attempts:
        assert len(a["choices"]) == (8 if policy == "threshold" else 4)
        assert all(f["train_last_signal_date"] < f["validation_first_signal_date"] <= f["validation_last_signal_date"] < a["signal_date"] for f in a["folds"])
    pd.testing.assert_frame_equal(champion, before_champion, check_exact=True)
    pd.testing.assert_frame_equal(features, before_features, check_exact=True)


@pytest.mark.parametrize("policy", ["probability", "paired", "threshold"])
def test_changing_current_and_future_outcomes_cannot_change_prediction_or_selection(policy):
    config = replace(CONFIG, require_paired_nonregression=policy != "probability",
                     threshold_candidates=(.5, .6) if policy == "threshold" else ())
    champion, features = inputs()
    original, attempts = nested.nested_downside_probabilities(champion, features, ("trend",), config=config)
    champion.loc[90:, "real_pct_change"] *= -1
    features.loc[91:, ["price", "trend"]] *= 100
    changed, changed_attempts = nested.nested_downside_probabilities(champion, features, ("trend",), config=config)
    pd.testing.assert_frame_equal(original.iloc[:91], changed.iloc[:91], check_exact=True)
    cutoff = int(champion.trade_date.iloc[90])
    assert [a for a in attempts if a["signal_date"] <= cutoff] == [a for a in changed_attempts if a["signal_date"] <= cutoff]


def test_transform_is_fitted_separately_on_each_training_block(monkeypatch):
    champion, features = inputs(70)
    observations = []
    original = nested._fit

    def record(values, target, c, max_iter):
        observations.append(values.copy())
        return original(values, target, c, max_iter)

    monkeypatch.setattr(nested, "_fit", record)
    config = nested.NestedDownsideConfig(train_window=120, min_train_rows=30, validation_block_rows=10,
                                        validation_blocks=2, refit_interval=100, inverse_penalties=(0.1,))
    _, attempts = nested.nested_downside_probabilities(champion, features.drop(columns="trend"), config=config)
    assert len(attempts) == 1
    assert [len(v) for v in observations] == [30, 40, 50]
    for values, count in zip(observations, (30, 40, 50), strict=True):
        np.testing.assert_array_equal(values[:, 0], features.price.iloc[:count])


def test_missing_features_and_original_down_signals_never_request_corrections():
    champion, features = inputs()
    champion.loc[::7, "predicted_label"] = 0
    available = np.ones(len(champion), dtype=bool); available[::11] = False
    features.loc[~available, "trend"] = np.nan
    result, _ = nested.nested_downside_probabilities(champion, features, ("trend",), available=available, config=CONFIG)
    excluded = ~available | champion.predicted_label.eq(0)
    assert result.loc[excluded, "downside_training_rows"].eq(0).all()
    assert result.loc[excluded, "downside_probability"].eq(0).all()
    assert result.downside_training_rows.max() > 0


def test_requires_inner_validation_warmup_and_rejects_misalignment():
    champion, features = inputs(45)
    result, attempts = nested.nested_downside_probabilities(champion, features, ("trend",), config=CONFIG)
    assert not attempts and result.downside_training_rows.eq(0).all()
    with pytest.raises(ValueError, match="aligned"):
        nested.nested_downside_probabilities(champion, features.iloc[::-1], ("trend",), config=CONFIG)
    with pytest.raises(ValueError, match="Optional"):
        nested.nested_downside_probabilities(champion, features, ("future_target",), config=CONFIG)


def test_paired_balanced_accuracy_uses_all_dates_including_incumbent_down():
    actual = np.array([-.01, .01, .01, -.01, -.01, .01])
    incumbent = np.array([1, 0, 1, 0, 1, 1])
    report = nested.paired_validation(actual, incumbent, np.array([0, 2, 4, 5]), np.array([.8, .1, .9, .2]), .6)
    assert report["rows"] == 6
    assert report["actual_up_rows"] == report["actual_down_rows"] == 3
    assert report["corrected"] == 2 and report["damaged"] == 0
    assert report["accuracy_delta"] == pytest.approx(1/3)
    assert report["balanced_accuracy_delta"] == pytest.approx(1/3)
    assert report["nondecreasing"]


def test_auxiliary_confidence_does_not_override_losing_validation_results(monkeypatch):
    champion, features = inputs()
    champion["real_pct_change"] = .01
    champion.loc[::5, "real_pct_change"] = -.01
    class ConfidentDown:
        def predict_proba(self, values):
            return np.tile([.01, .99], (len(values), 1))
    monkeypatch.setattr(nested, "_fit", lambda *args: ConfidentDown())
    result, attempts = nested.nested_downside_probabilities(
        champion, features, ("trend",), config=replace(CONFIG, require_paired_nonregression=True),
    )
    assert attempts
    assert all(a["selected"]["features"] == "incumbent" for a in attempts)
    assert all(not c["admissible"] for a in attempts for c in a["choices"])
    assert result.downside_training_rows.eq(0).all()
    assert result.downside_probability.eq(0).all()
    dates = [a["signal_date"] for a in attempts]
    assert len(dates) < 10  # A rejected refit still respects the scheduled interval.


def test_unchanged_validation_decisions_are_not_evidence_of_improvement(monkeypatch):
    champion, features = inputs()
    class BelowThreshold:
        def predict_proba(self, values):
            return np.tile([.45, .55], (len(values), 1))
    monkeypatch.setattr(nested, "_fit", lambda *args: BelowThreshold())
    _, attempts = nested.nested_downside_probabilities(
        champion, features, ("trend",), config=replace(CONFIG, require_paired_nonregression=True),
    )
    assert all(a["selected"]["features"] == "incumbent" for a in attempts)


def test_learned_threshold_selection_compares_actual_decision_improvement():
    common = {"features": "base", "inverse_penalty": .1, "admissible": True,
              "changed_validation_rows": 10, "pooled_balanced_gain": .02}
    choices = [common | {"correction_threshold": .5, "pooled_accuracy_gain": .02, "log_loss": .68},
               common | {"correction_threshold": .6, "pooled_accuracy_gain": .01, "log_loss": .66},
               common | {"correction_threshold": .7, "pooled_accuracy_gain": .04, "log_loss": .64, "admissible": False}]
    assert nested.select_choice(choices, learn_threshold=True)["correction_threshold"] == .5
    assert nested.select_choice(choices, learn_threshold=False)["correction_threshold"] == .6


def test_selected_thresholds_control_only_market_model_corrections():
    import tushare_prediction_pipeline as pipeline
    champion, _ = inputs(85)
    champion["predicted_pct_change"] = .002
    champion["uncalibrated_predicted_return"] = .002
    champion["predicted_close"] = 100.2
    champion["confidence"] = .6
    champion["calibrated_confidence"] = .6
    champion["correct"] = champion.predicted_label.eq(champion.real_pct_change.gt(0)).astype("boolean")
    champion.loc[77, ["predicted_label", "predicted_pct_change", "uncalibrated_predicted_return", "predicted_close"]] = [0, -.002, -.002, 99.8]
    champion.loc[84, ["real_pct_change", "correct"]] = [np.nan, None]
    probabilities = pd.DataFrame({"trade_date": champion.trade_date, "downside_probability": .56,
                                  "downside_training_rows": 0, "downside_last_training_date": 0,
                                  "nested_last_validation_date": 0, "nested_selected_threshold": 0.0})
    probabilities.loc[75:77, ["downside_training_rows", "downside_last_training_date", "nested_last_validation_date"]] = [50, int(champion.trade_date.iloc[70]), int(champion.trade_date.iloc[70])]
    probabilities.loc[75:77, "nested_selected_threshold"] = [.5, .6, .5]
    result = nested.selected_threshold_predictions(pipeline.prediction_core, champion, probabilities)
    assert result.predicted_label.iloc[75] == 0
    assert result.predicted_label.iloc[76] == 1
    assert result.predicted_label.iloc[77] == 0
    assert result.correction_selected.sum() == 1
    assert pd.isna(result.real_pct_change.iloc[-1])
    assert pd.isna(result.correct.iloc[-1])
    probabilities.loc[75, "nested_last_validation_date"] = champion.trade_date.iloc[75]
    with pytest.raises(ValueError, match="precede"):
        nested.selected_threshold_predictions(pipeline.prediction_core, champion, probabilities)
from dataclasses import replace
