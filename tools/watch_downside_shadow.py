"""Run or inspect the fixed research watcher without changing public forecasts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from filelock import FileLock, Timeout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction_service.config import Settings
from prediction_service.service import PredictionService
from prediction_service.downside_watcher import FixedDownsideWatcher


def read_status(settings):
    path = settings.root_dir / "downside_watcher.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not_started"}
    lock = FileLock(str(settings.root_dir / "downside_watcher.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        state["watcher_running"] = True
    else:
        state["watcher_running"] = False
        lock.release()
    observed = datetime.now(timezone.utc)
    state["observed_at"] = observed.isoformat()
    if state.get("checked_at"):
        state["heartbeat_age_seconds"] = (observed - datetime.fromisoformat(state["checked_at"])).total_seconds()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "once", "status", "stop"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.command in ("run", "once") and args.bundle is None:
        parser.error("run and once require --bundle for an explicitly fixed candidate")
    if not 20 <= args.poll_seconds <= 3600:
        parser.error("poll interval must be between 20 and 3600 seconds")
    settings = Settings.from_config(args.config)
    state_path = settings.root_dir / "downside_watcher.json"
    stop_path = settings.root_dir / "downside_watcher.stop"
    if args.command == "status":
        print(json.dumps(read_status(settings)))
        return 0
    if args.command == "stop":
        stop_path.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        print('{"stop_requested":true}')
        return 0
    with FileLock(str(settings.root_dir / "downside_watcher.lock"), timeout=0):
        if args.command == "run":
            stop_path.unlink(missing_ok=True)
        service = PredictionService(settings)
        watcher = FixedDownsideWatcher(service, args.bundle)
        last_status = None
        try:
            while True:
                state = watcher.run_once()
                identity = (state["status"], state.get("snapshot_id"), state.get("attempts"), state.get("error_type"))
                if identity != last_status or args.command == "once":
                    print(json.dumps(state), flush=True)
                    last_status = identity
                if args.command == "once":
                    return 1 if state["status"] in {"failed", "blocked", "retry_exhausted"} else 0
                # Check the stop request during idle waits; never interrupt a ledger write.
                until = time.monotonic() + args.poll_seconds
                while time.monotonic() < until and not stop_path.exists():
                    time.sleep(min(1, max(0, until - time.monotonic())))
                if stop_path.exists():
                    watcher._save(state, datetime.now(timezone.utc), "stopped")
                    return 0
        except KeyboardInterrupt:
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            watcher._save(state, datetime.now(timezone.utc), "stopped")
            return 0
        finally:
            service.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
