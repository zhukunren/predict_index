from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_test_shadow_live_replay",
    ROOT / "tools" / "verify_shadow_live_replay.py",
)
assert SPEC is not None and SPEC.loader is not None
parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity)


def _replay_row() -> pd.Series:
    return pd.Series(
        {
            "predicted_label": 1,
            "predicted_pct_change": 0.00125,
            "predicted_close": 101.25,
            "confidence": 0.61,
            "calibrated_confidence": 0.59,
            "return_calibration_scale": 0.5,
        }
    )


def _live() -> dict[str, float | int]:
    return {
        "predicted_label": 1,
        "estimated_next_return": 0.00125,
        "estimated_next_close": 101.25,
        "raw_confidence": 0.61,
        "calibrated_confidence": 0.59,
        "return_calibration_scale": 0.5,
    }


def test_live_replay_comparison_accepts_matching_shadow_values():
    checks, values = parity.compare_live_to_replay(_live(), _replay_row())

    assert all(checks.values())
    assert values["predicted_label"] == {"live": 1, "historical_replay": 1}


def test_live_replay_comparison_rejects_prediction_drift():
    live = _live()
    live["estimated_next_return"] = 0.002

    checks, _ = parity.compare_live_to_replay(live, _replay_row())

    assert checks["predicted_pct_change"] is False


def test_probe_selection_uses_multiple_recent_real_dates():
    frame = pd.DataFrame(
        {"trade_date": pd.bdate_range("2026-01-05", periods=10)}
    )

    probes = parity._resolve_probe_dates(
        frame,
        requested_date=None,
        requested_dates=None,
        probe_count=3,
        probe_window_days=5,
    )

    assert [item.strftime("%Y%m%d") for item in probes] == [
        "20260109",
        "20260113",
        "20260115",
    ]
    assert probes[-1] < pd.Timestamp(frame["trade_date"].iloc[-1])
