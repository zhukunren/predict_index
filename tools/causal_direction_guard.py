"""Correct a champion only with prior evidence on matching disagreements."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


PARAMETERS = {
    "history_window": 252,
    "minimum_disagreements": 20,
    "wilson_z": 1.96,
    "recent_window": 60,
    "minimum_recent_disagreements": 5,
    "minimum_recent_accuracy": 0.5,
}


def wilson_lower(correct: int, rows: int, z: float = 1.96) -> float:
    if not rows:
        return 0.0
    p = correct / rows
    return (p + z * z / (2 * rows) - z * math.sqrt(p * (1 - p) / rows + z * z / (4 * rows * rows))) / (1 + z * z / rows)


def apply_guard(core, champion: pd.DataFrame, challengers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    for challenger in challengers.values():
        if not np.array_equal(champion.trade_date, challenger.trade_date):
            raise ValueError("Direction guard requires identical, ordered signal dates.")
        if not np.allclose(champion.real_pct_change, challenger.real_pct_change, atol=0, rtol=0, equal_nan=True):
            raise ValueError("Direction guard requires identical outcomes.")
    frame = champion.copy().reset_index(drop=True)
    labels = frame.predicted_label.to_numpy(dtype=int).copy()
    actual = frame.real_pct_change.to_numpy(dtype=float)
    known = np.isfinite(actual)
    candidate_labels = {name: values.predicted_label.to_numpy(dtype=int) for name, values in challengers.items()}
    diagnostics = []
    for index in range(len(frame)):
        candidates = []
        start = max(0, index - PARAMETERS["history_window"])
        recent_start = max(start, index - PARAMETERS["recent_window"])
        for name, predicted in candidate_labels.items():
            if predicted[index] == labels[index]:
                continue
            matching = (labels[start:index] == labels[index]) & (predicted[start:index] != labels[start:index]) & known[start:index]
            rows = int(matching.sum())
            if rows < PARAMETERS["minimum_disagreements"]:
                continue
            correct = int((predicted[start:index][matching] == (actual[start:index][matching] > 0)).sum())
            lower = wilson_lower(correct, rows, PARAMETERS["wilson_z"])
            if lower <= 0.5:
                continue
            recent = (labels[recent_start:index] == labels[index]) & (predicted[recent_start:index] != labels[recent_start:index]) & known[recent_start:index]
            recent_rows = int(recent.sum())
            if recent_rows < PARAMETERS["minimum_recent_disagreements"]:
                continue
            recent_accuracy = float((predicted[recent_start:index][recent] == (actual[recent_start:index][recent] > 0)).mean())
            if recent_accuracy < PARAMETERS["minimum_recent_accuracy"]:
                continue
            candidates.append((lower, rows, name, recent_accuracy))
        source = "baseline"
        lower, rows, recent_accuracy = None, 0, None
        if candidates:
            lower, rows, source, recent_accuracy = max(candidates)
        selected = champion if source == "baseline" else challengers[source]
        selected_row = selected.iloc[index]
        raw_return = float(selected_row["uncalibrated_predicted_return"])
        current_close = float(selected_row["predicted_close"]) / (1 + float(selected_row["predicted_pct_change"]))
        label = int(selected_row["predicted_label"])
        frame.loc[index, "predicted_label"] = label
        frame.loc[index, "predicted_pct_change"] = raw_return
        frame.loc[index, "predicted_close"] = current_close * (1 + raw_return)
        frame.loc[index, "confidence"] = selected_row["confidence"]
        frame.loc[index, "correct"] = bool(label == (actual[index] > 0)) if known[index] else None
        diagnostics.append({"guard_source": source, "guard_lower_bound": lower,
                            "guard_history_rows": rows, "guard_recent_accuracy": recent_accuracy})
    # Fit the mixed forecast's calibration from its own previous outcomes.
    frame = core._apply_rolling_confidence_calibration(frame, window=300, min_rows=60, method="platt")
    from return_calibration import calibrate_returns
    frame = calibrate_returns(frame, window=252, min_rows=60)
    return pd.concat([frame, pd.DataFrame(diagnostics)], axis=1)
