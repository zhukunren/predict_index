"""Poll immutable publications and record one fixed candidate before market open."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

from filelock import FileLock, Timeout

from .calendar import SHANGHAI
from .downside_shadow import existing_run, prospective_report, verify_run
from tools.fixed_downside_candidate import load_candidate
from tools.run_downside_shadow import run as run_shadow


RETRY_INTERVAL = timedelta(minutes=10)
MAX_ATTEMPTS = 6


class FixedDownsideWatcher:
    def __init__(self, service, bundle: Path):
        self.service = service
        self.bundle = Path(bundle).resolve()
        self.state_path = service.settings.root_dir / "downside_watcher.json"

    def _save(self, state, now, status, **values):
        state.update(values, status=status, checked_at=now.isoformat(), pid=os.getpid(),
                     bundle=str(self.bundle), watcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     automatic_promotion=False, backfill_allowed=False)
        path = self.state_path.with_suffix(".json.tmp")
        path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        path.replace(self.state_path)
        return state

    def run_once(self, now=None):
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("Watcher time must be timezone-aware.")
        now = now.astimezone(timezone.utc)
        previous = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        state = dict(previous)
        try:
            frozen = load_candidate(self.bundle)
            _, snapshot, control = self.service._load_active_context()
        except Exception as exc:
            # Provider and configuration errors may embed credentials in their text.
            return self._save(state, now, "blocked", error_type=type(exc).__name__)
        key = (snapshot.id, frozen["release_id"])
        if key != (previous.get("snapshot_id"), previous.get("release_id")):
            state = {"snapshot_id": snapshot.id, "release_id": frozen["release_id"],
                     "signal_date": snapshot.data_as_of, "attempts": 0}
        state.pop("error_type", None)
        state.pop("next_attempt_at", None)
        try:
            completed = existing_run(self.service, *key)
            if completed is not None:
                verify_run(self.service, completed)
                report = prospective_report(self.service, frozen["release_id"], control.id)
                return self._save(state, now, "up_to_date", run_id=completed.id, report=report)
            ready_at = datetime.strptime(snapshot.data_as_of, "%Y%m%d").replace(hour=18, tzinfo=SHANGHAI)
            if now < ready_at:
                return self._save(state, now, "waiting_window", next_attempt_at=ready_at.isoformat())
            target = self.service.calendar.next_session(snapshot.data_as_of)
            if target is not None:
                opening = datetime.strptime(target, "%Y%m%d").replace(hour=9, minute=30, tzinfo=SHANGHAI)
                state["target_date"] = target
                if now >= opening:
                    return self._save(state, now, "missed_window")
            # A missing calendar is refreshed by the runner before any prediction.
            # That runner independently checks real time again when archiving.
            if state["attempts"] >= MAX_ATTEMPTS:
                return self._save(state, now, "retry_exhausted", error_type=previous.get("error_type"))
            if state.get("last_attempt_at"):
                due = datetime.fromisoformat(state["last_attempt_at"]) + RETRY_INTERVAL
                if now < due:
                    return self._save(state, now, "retry_wait", next_attempt_at=due.isoformat(),
                                      error_type=previous.get("error_type"))
            try:
                lock = FileLock(str(self.service.settings.root_dir / "downside_shadow.lock"), timeout=0)
                lock.acquire()
            except Timeout:
                return self._save(state, now, "busy")
            try:
                # The active publication can advance between polling and locking.
                _, current_snapshot, _ = self.service._load_active_context()
                if current_snapshot.id != snapshot.id:
                    return self._save(state, now, "publication_changed")
                self._save(state, now, "running", attempts=state["attempts"] + 1, last_attempt_at=now.isoformat())
                completed, report = run_shadow(self.service, self.bundle, fetch=True, allow_backfill=False)
                if completed.snapshot_id != snapshot.id:
                    return self._save(state, datetime.now(timezone.utc), "publication_changed")
                return self._save(state, datetime.now(timezone.utc), "succeeded", run_id=completed.id, report=report)
            finally:
                lock.release()
        except Exception as exc:
            return self._save(state, now, "failed", error_type=type(exc).__name__)
