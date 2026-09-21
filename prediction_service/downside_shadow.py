"""Record fixed downside candidates in the existing immutable shadow ledger."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from sqlalchemy import select

from .archive import frame_to_csv_bytes, read_features, sha256_bytes, sha256_file, write_daily_archive
from .calendar import SHANGHAI
from .engine import feature_close_by_date, prediction_payload, payload_fingerprint
from .models import ModelRelease, PredictionLedger, OutcomeResolution, ShadowRun, utcnow
from .service import PredictionDriftError
from tools.fixed_downside_candidate import ALGORITHM, canonical_json


def prediction_window(calendar, signal_date, now, *, allow_backfill=False):
    if now.tzinfo is None:
        raise ValueError("Prediction time must be timezone-aware.")
    target = calendar.next_session(signal_date)
    if target is None or calendar.sessions(signal_date, signal_date) != [signal_date]:
        raise ValueError("A complete exchange calendar is required for the signal and target dates.")
    close_ready = datetime.strptime(signal_date, "%Y%m%d").replace(hour=18, tzinfo=SHANGHAI)
    opening = datetime.strptime(target, "%Y%m%d").replace(hour=9, minute=30, tzinfo=SHANGHAI)
    if now < close_ready:
        raise ValueError("Same-day market breadth requires actual generation at or after 18:00 Shanghai.")
    if now >= opening and not allow_backfill:
        raise ValueError("Target session has already opened; use explicit backfill mode for replay only.")
    return target, opening


def existing_run(service, snapshot_id, release_id):
    with service.database.session() as session:
        return session.scalar(select(ShadowRun).where(
            ShadowRun.snapshot_id == snapshot_id, ShadowRun.release_id == release_id,
        ))


def verify_run(service, run):
    if (run.status != "succeeded" or not run.result_csv_path or not run.manifest_path
            or sha256_file(Path(run.result_csv_path)) != run.result_csv_sha256):
        raise PredictionDriftError("Shadow result differs from the registered archive.")
    service._assert_manifest_integrity(Path(run.manifest_path))


def record_result(service, snapshot, baseline_release, frozen, candidate, baseline, context, *,
                  raw_frames=None, receipts=(), allow_backfill=False, actor="cli"):
    """No code path here changes Publication or accepts a caller-supplied clock."""
    service._assert_archive_integrity(snapshot, None)
    market = read_features(snapshot.features_path)
    if (candidate.empty or candidate.trade_date.duplicated().any()
            or not candidate.trade_date.is_monotonic_increasing
            or not np.array_equal(candidate.trade_date, baseline.trade_date)
            or str(int(candidate.trade_date.iloc[-1])) != snapshot.data_as_of
            or candidate.real_pct_change.iloc[:-1].isna().any()
            or pd.notna(candidate.real_pct_change.iloc[-1])):
        raise ValueError("Shadow results must end in exactly one unresolved snapshot-date prediction.")
    numbers = candidate.loc[:, ["predicted_pct_change", "predicted_close", "confidence", "calibrated_confidence"]]
    if (not np.isfinite(numbers.to_numpy()).all() or not candidate.predicted_label.isin((0, 1)).all()
            or not candidate.confidence.between(0, 1).all()
            or not candidate.calibrated_confidence.between(0, 1).all()):
        raise ValueError("Shadow forecast values are invalid.")
    target, opening = prediction_window(service.calendar, snapshot.data_as_of, utcnow(), allow_backfill=allow_backfill)
    close_by_date = feature_close_by_date(market)
    prices = pd.Series(close_by_date)
    actual = (prices.shift(-1) / prices - 1).reindex(candidate.trade_date.astype(str)).to_numpy()
    if not np.allclose(candidate.real_pct_change, actual, rtol=0, atol=1e-14, equal_nan=True):
        raise ValueError("Shadow outcomes disagree with the immutable market snapshot.")
    base_prices = prices.reindex(candidate.trade_date.astype(str)).to_numpy()
    if not np.allclose(candidate.predicted_close, base_prices * (1 + candidate.predicted_pct_change), rtol=1e-14, atol=1e-12):
        raise ValueError("Shadow forecast prices disagree with their base close and return.")
    nonzero = candidate.predicted_pct_change.ne(0)
    if not candidate.loc[nonzero, "predicted_pct_change"].gt(0).eq(candidate.loc[nonzero, "predicted_label"].eq(1)).all():
        raise ValueError("Shadow return signs disagree with direction labels.")
    settled = candidate.real_pct_change.notna()
    if not candidate.loc[settled, "correct"].eq(candidate.loc[settled, "predicted_label"].eq(candidate.loc[settled, "real_pct_change"].gt(0))).all():
        raise ValueError("Shadow correctness disagrees with the resolved direction.")
    # Anchor the control to already published immutable predictions, not only a replay.
    with service.database.session() as session:
        stored = session.scalars(select(PredictionLedger).where(
            PredictionLedger.release_id == baseline_release.id,
            PredictionLedger.signal_date.in_([str(int(day)) for day in baseline.trade_date]),
        )).all()
    if not any(row.signal_date == snapshot.data_as_of for row in stored):
        raise ValueError("The target snapshot has no published control prediction.")
    by_date = baseline.set_index(baseline.trade_date.astype(str))
    for row in stored:
        replay = by_date.loc[row.signal_date]
        actual = payload_fingerprint(prediction_payload(replay, base_close=close_by_date[row.signal_date]))
        if actual != row.prediction_fingerprint:
            raise PredictionDriftError("Control replay differs from the published immutable ledger.")
    config = frozen["configuration"]
    identifier = frozen["release_id"]
    if sha256_bytes(canonical_json(config)) != identifier:
        raise ValueError("Candidate release identity mismatch.")
    release = ModelRelease(id=identifier, algorithm_id=ALGORITHM,
                           source_bundle_sha256=sha256_bytes(canonical_json(config["sources"])),
                           config_sha256=sha256_bytes(canonical_json(config)),
                           config_json=canonical_json(config).decode())
    with service._operation_lock:
        previous = existing_run(service, snapshot.id, identifier)
        if previous is not None:
            verify_run(service, previous)
            return previous
        candidate, plan = service._candidate_plan(features=market, release=release, snapshot_id=snapshot.id,
                                                   allow_new=True, result=candidate, inherit_provenance=False)
        for item in plan:
            if item.is_new and item.outcome is not None and item.origin == "live":
                raise ValueError("Settled replay data cannot be recorded as a prospective forecast.")
        artifact = service._render_artifact(result=candidate.tail(61), plan=plan, snapshot_id=snapshot.id,
                                            release_id=identifier, as_of=snapshot.data_as_of)
        run_id = str(uuid.uuid4())
        context_frames = {f"context_{name}": frame for name, frame in context.items()}
        archive = write_daily_archive(
            archive_root=service.settings.shadow_archive_dir, snapshot_id=run_id,
            data_as_of=snapshot.data_as_of, features=market, public_csv=artifact.csv_bytes,
            raw_frames={**context_frames, **(raw_frames or {})},
            extra_files={"candidate.csv": frame_to_csv_bytes(candidate), "baseline.csv": frame_to_csv_bytes(baseline),
                         "candidate_contract.json": canonical_json(frozen)},
            manifest={"source": ALGORITHM, "market_snapshot_id": snapshot.id,
                      "market_snapshot_features_sha256": snapshot.features_sha256,
                      "release_id": identifier, "control_release_id": baseline_release.id,
                      "target_date": target, "receipts": list(receipts),
                      "mode": "backfill" if utcnow() >= opening else "prospective",
                      "automatic_promotion": False, "historical_gate_passed": False},
        )
        # Do not let slow calculation or archival turn a late run into live evidence.
        finished = utcnow()
        prediction_window(service.calendar, snapshot.data_as_of, finished, allow_backfill=allow_backfill)
        if finished >= opening and any(item.is_new and item.origin == "live" for item in plan):
            raise ValueError("Archival crossed the target opening; retry as a backfill.")
        with service.database.session() as session:
            registered = session.get(ModelRelease, identifier)
            if registered is None:
                session.add(release)
                session.flush()
            elif registered.config_json != release.config_json:
                raise ValueError("Stored candidate configuration changed.")
            service._persist_prediction_plan(session, plan=plan, release=release, snapshot_id=snapshot.id)
            run = ShadowRun(id=run_id, engine=ALGORITHM, release_id=identifier, snapshot_id=snapshot.id,
                            status="succeeded", requested_by=actor, result_csv_path=str(archive.results_path),
                            result_csv_sha256=artifact.csv_sha256, manifest_path=str(archive.manifest_path),
                            row_count=len(artifact.public_frame), started_at=min(item.generated_at for item in plan if item.is_new)
                            if any(item.is_new for item in plan) else utcnow(), finished_at=utcnow(),
                            message="Fixed research candidate; public model unchanged.")
            session.add(run)
            session.flush()
        return run


def prospective_report(service, release_id, control_release_id):
    from tools.evaluate_direction_bias import metrics

    with service.database.session() as session:
        pairs = session.execute(select(PredictionLedger, OutcomeResolution).join(
            OutcomeResolution, OutcomeResolution.prediction_id == PredictionLedger.id,
        ).where(PredictionLedger.release_id.in_((release_id, control_release_id)))).all()
    series = {release_id: {}, control_release_id: {}}
    for prediction, outcome in pairs:
        generated = prediction.created_at
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        opening = datetime.strptime(outcome.target_trade_date, "%Y%m%d").replace(hour=9, minute=30, tzinfo=SHANGHAI)
        if generated >= opening:
            continue
        series[prediction.release_id][prediction.signal_date] = {
            "trade_date": int(prediction.signal_date), "predicted_label": prediction.predicted_label,
            "predicted_pct_change": prediction.predicted_return, "confidence": prediction.raw_confidence,
            "calibrated_confidence": prediction.calibrated_confidence, "real_pct_change": outcome.actual_return,
            "correct": outcome.correct, "target_date": outcome.target_trade_date,
        }
    dates = sorted(set(series[release_id]) & set(series[control_release_id]))
    for date in dates:
        left, right = series[release_id][date], series[control_release_id][date]
        if left["target_date"] != right["target_date"] or left["real_pct_change"] != right["real_pct_change"]:
            raise PredictionDriftError("Prospective comparison targets differ.")
    return {"release_id": release_id, "control_release_id": control_release_id,
            "prospective_paired_rows": len(dates), "historical_gate_passed": False, "promotion_allowed": False,
            "candidate": metrics(pd.DataFrame([series[release_id][date] for date in dates])) if dates else None,
            "baseline": metrics(pd.DataFrame([series[control_release_id][date] for date in dates])) if dates else None}
