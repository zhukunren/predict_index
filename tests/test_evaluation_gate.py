from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prediction_evaluation", ROOT / "tools" / "evaluate_prediction_candidate.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def _frame(labels: list[int]) -> pd.DataFrame:
    actual = [0.01, -0.01, 0.01, -0.01]
    predicted = [0.01 if label else -0.01 for label in labels]
    return pd.DataFrame(
        {
            "trade_date": [20240102, 20240103, 20250102, 20250103],
            "predicted_label": labels,
            "predicted_pct_change": predicted,
            "real_pct_change": actual,
            "correct": [label == int(value > 0) for label, value in zip(labels, actual)],
            # Deliberately hostile legacy values. The comparison must ignore them.
            "confidence": [0.2, 0.8, 0.3, 0.7],
            "calibrated_confidence": [1.0, 1.0, 1.0, 1.0],
        }
    )


def _causal_stub(frame: pd.DataFrame, **_: object) -> pd.DataFrame:
    result = frame.copy()
    result["calibrated_confidence"] = 0.5
    return result


def test_champion_gate_requires_exact_direction_parity(monkeypatch):
    monkeypatch.setattr(evaluation, "TEST_START", 20250101)
    baseline = _frame([1, 0, 1, 0])
    candidate = _frame([1, 0, 0, 0])

    outcome = evaluation.compare_candidate(
        baseline,
        candidate,
        confidence_calibrator=_causal_stub,
        require_direction_parity=True,
    )

    parity = next(check for check in outcome["checks"] if check["metric"] == "exact_direction_parity")
    assert outcome["direction_differences"] == 1
    assert parity["passed"] is False
    assert outcome["passed"] is False


def test_gate_uses_shared_causal_confidence_and_exact_dates(monkeypatch):
    monkeypatch.setattr(evaluation, "TEST_START", 20250101)
    baseline = _frame([1, 0, 1, 0])
    candidate = _frame([1, 0, 1, 0])

    outcome = evaluation.compare_candidate(
        baseline,
        candidate,
        confidence_calibrator=_causal_stub,
        require_direction_parity=True,
    )

    brier = next(
        check
        for check in outcome["checks"]
        if check["metric"] == "causal_fixed_300_confidence_brier"
    )
    assert brier["candidate"] == 0.25
    assert brier["baseline"] == 0.25
    assert outcome["passed"] is True

    incomplete = evaluation.compare_candidate(
        baseline,
        candidate.iloc[:-1],
        confidence_calibrator=_causal_stub,
        require_direction_parity=True,
    )
    coverage = next(check for check in incomplete["checks"] if check["metric"] == "exact_date_coverage")
    assert coverage["passed"] is False
    assert incomplete["passed"] is False


def test_direction_changer_cannot_trade_balanced_accuracy_for_hit_ratio(monkeypatch):
    monkeypatch.setattr(evaluation, "TEST_START", 20240101)
    actual = [0.01, 0.01, 0.01, -0.01]

    def make(labels: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": [20240102, 20240103, 20240104, 20240105],
                "predicted_label": labels,
                "predicted_pct_change": [0.01 if label else -0.01 for label in labels],
                "real_pct_change": actual,
                "correct": [label == int(value > 0) for label, value in zip(labels, actual)],
                "confidence": [0.5] * 4,
                "calibrated_confidence": [0.5] * 4,
            }
        )

    # Both have three correct days, but the challenger never predicts down.
    outcome = evaluation.compare_candidate(
        make([1, 0, 1, 0]),
        make([1, 1, 1, 1]),
        confidence_calibrator=_causal_stub,
        require_direction_parity=False,
    )

    accuracy = next(check for check in outcome["checks"] if check["metric"] == "accuracy")
    balanced = next(check for check in outcome["checks"] if check["metric"] == "balanced_accuracy")
    assert accuracy["passed"] is True
    assert balanced["passed"] is False
    assert outcome["passed"] is False


def test_extra_candidate_date_fails_coverage_gate(monkeypatch):
    monkeypatch.setattr(evaluation, "TEST_START", 20250101)
    baseline = _frame([1, 0, 1, 0])
    extra = pd.concat(
        [
            baseline,
            pd.DataFrame(
                {
                    "trade_date": [20250104],
                    "predicted_label": [1],
                    "predicted_pct_change": [0.01],
                    "real_pct_change": [0.01],
                    "correct": [True],
                    "confidence": [0.5],
                    "calibrated_confidence": [0.5],
                }
            ),
        ],
        ignore_index=True,
    )

    outcome = evaluation.compare_candidate(
        baseline,
        extra,
        confidence_calibrator=_causal_stub,
        require_direction_parity=False,
    )

    coverage = next(check for check in outcome["checks"] if check["metric"] == "exact_date_coverage")
    assert coverage["passed"] is False
    assert outcome["candidate_only_dates"] == [20250104]
    assert outcome["passed"] is False
