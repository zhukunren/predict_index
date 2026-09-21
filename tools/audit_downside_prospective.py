"""Audit real pre-open shadow evidence without changing the prediction ledger."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction_service.config import Settings
from prediction_service.models import Base, ModelRelease, OutcomeResolution, PredictionLedger, Publication, ShadowRun
from tools.evaluate_direction_bias import metrics
from tools.fixed_downside_candidate import load_candidate


SHANGHAI = ZoneInfo("Asia/Shanghai")
WINDOWS = (20, 60, 252)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def pre_open(created_at: datetime, target_date: str) -> bool:
    opening = datetime.strptime(str(target_date), "%Y%m%d").replace(hour=9, minute=30, tzinfo=SHANGHAI)
    return as_utc(created_at) < opening.astimezone(timezone.utc)


def _row(prediction, outcome):
    target_date = outcome.target_trade_date if outcome is not None else prediction.target_date
    generated_at = as_utc(prediction.created_at)
    return {
        "signal_date": str(prediction.signal_date),
        "target_date": None if target_date is None else str(target_date),
        "generated_at": generated_at.isoformat(),
        "predicted_label": int(prediction.predicted_label),
        "predicted_pct_change": float(prediction.predicted_return),
        "confidence": float(prediction.raw_confidence),
        "calibrated_confidence": float(prediction.calibrated_confidence),
        "prospective": bool(target_date is not None and pre_open(generated_at, str(target_date))),
        "settled": outcome is not None,
        "actual_return": None if outcome is None else float(outcome.actual_return),
        "correct": None if outcome is None else bool(outcome.correct),
    }


def window_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [row for row in rows if row["settled"]]
    result = {"all": {"rows": len(settled), "sufficient_for_20": len(settled) >= 20,
                       "sufficient_for_60": len(settled) >= 60, "sufficient_for_252": len(settled) >= 252}}
    for size in WINDOWS:
        tail = settled[-size:]
        if not tail:
            result[str(size)] = {"rows": 0, "sufficient": False, "metrics": None,
                                 "start_date": None, "end_date": None}
            continue
        frame = pd.DataFrame({
            "trade_date": [int(row["signal_date"]) for row in tail],
            "predicted_label": [row["predicted_label"] for row in tail],
            "predicted_pct_change": [row["predicted_pct_change"] for row in tail],
            "real_pct_change": [row["actual_return"] for row in tail],
            "confidence": [row["confidence"] for row in tail],
            "calibrated_confidence": [row["calibrated_confidence"] for row in tail],
            "correct": [row["correct"] for row in tail],
        })
        result[str(size)] = {"rows": len(tail), "sufficient": len(tail) >= size,
                             "start_date": tail[0]["signal_date"], "end_date": tail[-1]["signal_date"],
                             "metrics": metrics(frame)}
    return result


def build_audit(candidate_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]], *,
                candidate_release: str, control_release: str) -> dict[str, Any]:
    candidate = {row["signal_date"]: row for row in candidate_rows}
    control = {row["signal_date"]: row for row in control_rows}
    candidate_pre = {date: row for date, row in candidate.items() if row["prospective"]}
    control_pre = {date: row for date, row in control.items() if row["prospective"]}
    candidate_settled = {date: row for date, row in candidate_pre.items() if row["settled"]}
    control_settled = {date: row for date, row in control_pre.items() if row["settled"]}
    common = sorted(set(candidate_settled) & set(control_settled))
    candidate_only = sorted(set(candidate_pre) - set(control_pre))
    control_only = sorted(set(control_pre) - set(candidate_pre))
    target_mismatches = []
    for date in common:
        left, right = candidate_settled[date], control_settled[date]
        if left["target_date"] != right["target_date"] or left["actual_return"] != right["actual_return"]:
            target_mismatches.append(date)
    if target_mismatches:
        raise ValueError(f"Prospective targets differ on {target_mismatches[:5]}.")
    paired_candidate = [candidate_settled[date] for date in common]
    paired_control = [control_settled[date] for date in common]
    return {
        "candidate_release": candidate_release,
        "control_release": control_release,
        "candidate_rows": len(candidate_rows), "control_rows": len(control_rows),
        "candidate_prospective_rows": len(candidate_pre), "control_prospective_rows": len(control_pre),
        "candidate_prospective_settled_rows": len(candidate_settled),
        "control_prospective_settled_rows": len(control_settled),
        "prospective_paired_rows": len(common), "paired_signal_dates": common,
        "candidate_only_dates": candidate_only, "control_only_dates": control_only,
        "target_mismatch_dates": target_mismatches,
        "candidate": window_report(paired_candidate), "control": window_report(paired_control),
        "historical_gate_passed": False, "promotion_allowed": False,
        "evidence_scope": "pre-open ledger rows with immutable settled outcomes; not the historical 26-check gate",
    }


def audit(args):
    frozen = load_candidate(args.bundle)
    candidate_release = frozen["release_id"]
    settings = Settings.from_config(args.config)
    engine = create_engine(settings.database_url, future=True)
    try:
        with Session(engine) as session:
            publication = session.scalar(select(Publication).where(Publication.is_active.is_(True)))
            if publication is None:
                raise ValueError("No active publication exists.")
            control_release = publication.release_id
            prediction_rows = session.scalars(
                select(PredictionLedger).where(PredictionLedger.release_id.in_((candidate_release, control_release)))
                .order_by(PredictionLedger.release_id, PredictionLedger.signal_date)
            ).all()
            prediction_ids = [row.id for row in prediction_rows]
            outcomes = {}
            if prediction_ids:
                outcomes = {row.prediction_id: row for row in session.scalars(
                    select(OutcomeResolution).where(OutcomeResolution.prediction_id.in_(prediction_ids))
                ).all()}
            rows = [_row(row, outcomes.get(row.id)) for row in prediction_rows]
            grouped = {candidate_release: [], control_release: []}
            for row, item in zip(prediction_rows, rows, strict=True):
                grouped[row.release_id].append(item)
            report = build_audit(grouped[candidate_release], grouped[control_release],
                                 candidate_release=candidate_release, control_release=control_release)
            report.update({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "active_publication_id": publication.id,
                "active_snapshot_id": publication.snapshot_id,
                "bundle": str(args.bundle.resolve()),
                "bundle_frozen_sha256": hashlib.sha256((args.bundle / "frozen.json").read_bytes()).hexdigest(),
                "audit_source": "prediction_ledger + outcome_resolutions + active publication",
            })
    finally:
        engine.dispose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps(report, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
