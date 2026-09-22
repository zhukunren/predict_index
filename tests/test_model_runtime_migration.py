import pandas as pd
import pytest

from tools.migrate_model_runtime import compatibility_report


def forecasts():
    return pd.DataFrame({"trade_date": [20260105, 20260106], "predicted_label": [1, 0],
                         "predicted_pct_change": [.003, -.002], "predicted_close": [100.3, 99.8],
                         "confidence": [.6, .7], "calibrated_confidence": [.58, .62],
                         "return_calibration_scale": [.8, .9]})


def test_runtime_migration_accepts_only_bounded_probability_noise():
    old = forecasts()
    new = old.copy()
    new.loc[0, "confidence"] += 1e-13
    report = compatibility_report(old, new)
    assert report["fields"]["predicted_label"]["exact"]
    assert report["fields"]["confidence"]["max_abs"] < 1e-12


@pytest.mark.parametrize("column", ["predicted_label", "predicted_pct_change", "predicted_close", "return_calibration_scale"])
def test_runtime_migration_cannot_change_non_probability_outputs(column):
    old = forecasts()
    new = old.copy()
    new.loc[0, column] += 1 if column == "predicted_label" else 1e-13
    with pytest.raises(ValueError, match="frozen forecast"):
        compatibility_report(old, new)


@pytest.mark.parametrize("value", [.601, float("nan"), float("inf"), 1.01])
def test_runtime_migration_rejects_large_or_invalid_probability_changes(value):
    old = forecasts()
    new = old.copy()
    new.loc[0, "confidence"] = value
    with pytest.raises(ValueError):
        compatibility_report(old, new)


def test_runtime_migration_rejects_missing_or_duplicate_dates():
    old = forecasts()
    with pytest.raises(ValueError, match="Missing"):
        compatibility_report(old, old.iloc[:1])
    with pytest.raises(ValueError, match="unique"):
        compatibility_report(old, pd.concat([old, old.iloc[:1]], ignore_index=True))
