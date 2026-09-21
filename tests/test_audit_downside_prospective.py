from datetime import datetime, timezone

import pytest

from tools.audit_downside_prospective import build_audit, pre_open


def row(date, *, release="candidate", target="20260115", created="2026-01-14T10:30:00+00:00",
        settled=True, actual=-.01, label=1):
    return {"signal_date": str(date), "target_date": target, "generated_at": created,
            "predicted_label": label, "predicted_pct_change": .01 if label else -.01,
            "confidence": .6, "calibrated_confidence": .6, "prospective": True,
            "settled": settled, "actual_return": actual if settled else None,
            "correct": bool(label == int(actual > 0)) if settled else None}


def test_pre_open_uses_target_session_open_and_timezone():
    assert pre_open(datetime(2026, 1, 14, 10, 30, tzinfo=timezone.utc), "20260115")
    assert not pre_open(datetime(2026, 1, 15, 1, 30, tzinfo=timezone.utc), "20260115")


def test_audit_lists_pairs_windows_and_missing_dates():
    candidate = [row(20260114)]
    control = [row(20260114, release="control"), row(20260113, release="control", target="20260114")]
    report = build_audit(candidate, control, candidate_release="candidate", control_release="control")
    assert report["prospective_paired_rows"] == 1
    assert report["candidate_only_dates"] == []
    assert report["control_only_dates"] == ["20260113"]
    assert report["candidate"]["20"]["sufficient"] is False
    assert report["candidate"]["20"]["metrics"]["rows"] == 1
    assert report["promotion_allowed"] is False


def test_audit_rejects_target_or_actual_mismatch():
    candidate = [row(20260114)]
    control = [row(20260114, target="20260116")]
    with pytest.raises(ValueError, match="targets differ"):
        build_audit(candidate, control, candidate_release="candidate", control_release="control")


def test_unsettled_rows_are_not_counted_as_pairs():
    candidate = [row(20260114, settled=False)]
    control = [row(20260114, settled=False)]
    report = build_audit(candidate, control, candidate_release="candidate", control_release="control")
    assert report["candidate_prospective_rows"] == 1
    assert report["candidate_prospective_settled_rows"] == 0
    assert report["prospective_paired_rows"] == 0
