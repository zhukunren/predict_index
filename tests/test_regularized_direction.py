from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier

import regularized_direction as direction


MODEL_OPTIONS = {"train_window": 80, "min_train_rows": 64, "refit_interval": 5}


def _market_rows(rows: int = 111) -> tuple[pd.DataFrame, pd.Series]:
    index = np.arange(rows, dtype=float)
    daily_returns = np.where(np.sin(index * 0.47) + 0.2 * np.cos(index * 0.13) > 0, 0.004, -0.003)
    close = pd.Series(100.0 * np.cumprod(1.0 + daily_returns))
    features = pd.DataFrame(
        {
            "row_id": index,
            "cycle": np.sin(index * 0.47),
            "slow_cycle": np.cos(index * 0.13),
            "return_1": close.pct_change().fillna(0.0),
        }
    )
    return features, close


def test_real_training_matches_every_requested_prefix():
    features, close = _market_rows()
    full_probabilities, full_training_rows = direction.rolling_probabilities(
        features, close, **MODEL_OPTIONS
    )
    assert np.isfinite(full_probabilities).all()
    assert np.unique(full_probabilities[80:]).size > 1
    for last_row in (64, 68, 90, 108):
        probabilities, training_rows = direction.rolling_probabilities(
            features.iloc[: last_row + 1], close.iloc[: last_row + 1], **MODEL_OPTIONS
        )
        np.testing.assert_array_equal(probabilities, full_probabilities[: last_row + 1])
        np.testing.assert_array_equal(training_rows, full_training_rows[: last_row + 1])


def test_unknown_current_label_and_future_features_do_not_change_current_probability():
    features, close = _market_rows()
    current_row = 90
    probabilities, training_rows = direction.rolling_probabilities(features, close, **MODEL_OPTIONS)
    mutated_features = features.copy()
    mutated_close = close.copy()
    mutated_features.loc[current_row + 1 :] = 100_000.0
    # The next close determines the current unknown target; it is not training information.
    mutated_close.iloc[current_row + 1 :] = close.iloc[current_row] * np.cumprod(
        np.full(len(close) - current_row - 1, 1.10)
    )
    assert (close.iloc[current_row + 1] > close.iloc[current_row]) != (
        mutated_close.iloc[current_row + 1] > mutated_close.iloc[current_row]
    )
    mutated_probabilities, mutated_training_rows = direction.rolling_probabilities(
        mutated_features, mutated_close, **MODEL_OPTIONS
    )
    np.testing.assert_array_equal(mutated_probabilities[: current_row + 1], probabilities[: current_row + 1])
    np.testing.assert_array_equal(mutated_training_rows[: current_row + 1], training_rows[: current_row + 1])


def test_refit_anchor_and_actual_training_labels_are_strictly_historical(monkeypatch: pytest.MonkeyPatch):
    features, close = _market_rows()
    labels = (close.shift(-1) / close - 1.0).gt(0).to_numpy(dtype=int)
    fit_records: list[tuple[np.ndarray, np.ndarray]] = []
    prediction_records: list[tuple[int, np.ndarray]] = []

    class RecordingExtraTrees(ExtraTreesClassifier):
        def fit(self, x, y, sample_weight=None):
            self.recorded_rows = np.asarray(x)[:, 0].astype(int)
            fit_records.append((self.recorded_rows.copy(), np.asarray(y).copy()))
            return super().fit(x, y, sample_weight=sample_weight)

        def predict_proba(self, x):
            prediction_records.append((int(np.asarray(x)[0, 0]), self.recorded_rows.copy()))
            return super().predict_proba(x)

    monkeypatch.setattr(direction, "ExtraTreesClassifier", RecordingExtraTrees)
    _, training_rows = direction.rolling_probabilities(features, close, **MODEL_OPTIONS)
    actual_fit_dates = [int(rows[-1] + 1) for rows, _ in fit_records]
    assert actual_fit_dates == [64, *range(65, len(features), 5)]
    for rows, actual_labels in fit_records:
        np.testing.assert_array_equal(actual_labels, labels[rows])
        assert len(rows) <= MODEL_OPTIONS["train_window"]
    for signal_row, fitted_rows in prediction_records:
        assert fitted_rows.max() < signal_row
        assert training_rows[signal_row] == len(fitted_rows)
    assert training_rows[64] == 64
    np.testing.assert_array_equal(training_rows[65:70], np.full(5, 65))


@pytest.mark.parametrize("daily_return", [-0.01, 0.0, 0.01])
def test_single_class_fallback_is_smoothed_and_uses_only_completed_labels(daily_return: float):
    features = pd.DataFrame({"row_id": np.arange(15, dtype=float)})
    close = pd.Series(100.0 * np.cumprod(np.full(15, 1.0 + daily_return)))
    probabilities, training_rows = direction.rolling_probabilities(
        features, close, train_window=8, min_train_rows=4, refit_interval=5
    )
    history_count = np.minimum(np.arange(15), 8)
    expected_up_count = history_count if daily_return > 0 else np.zeros(15)
    np.testing.assert_array_equal(training_rows, history_count)
    np.testing.assert_allclose(probabilities, (expected_up_count + 1.0) / (history_count + 2.0))
    assert probabilities[0] == 0.5
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))


def test_nonfinite_feature_row_is_not_predicted_or_used_for_training():
    features, close = _market_rows(rows=25)
    features.loc[7, "cycle"] = np.nan
    probabilities, training_rows = direction.rolling_probabilities(
        features, close, train_window=10, min_train_rows=4, refit_interval=5
    )
    assert np.isnan(probabilities[7])
    assert training_rows[7] == 0
    assert training_rows[10] == 9
    assert np.isfinite(probabilities[np.arange(len(features)) != 7]).all()


@pytest.mark.parametrize(
    "options",
    [
        {"train_window": 10, "min_train_rows": 1, "refit_interval": 5},
        {"train_window": 3, "min_train_rows": 4, "refit_interval": 5},
        {"train_window": 10, "min_train_rows": 4, "refit_interval": 0},
    ],
)
def test_invalid_window_settings_are_rejected(options):
    features, close = _market_rows(rows=20)
    with pytest.raises(ValueError, match="Invalid rolling model"):
        direction.rolling_probabilities(features, close, **options)
