"""Prepare, activate or verify the explicitly selected production/shadow models."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction_service.archive import read_features
from prediction_service.config import Settings
from prediction_service.model_registry import prepare_bundle, ModelBundle
from prediction_service.portfolio import publish_portfolio, activation_features
from prediction_service.service import PredictionService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "activate", "verify"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.ini")
    parser.add_argument("--fetch", action="store_true", help="Incrementally fetch missing shared signals.")
    parser.add_argument("--data-dir", type=Path, help="Use an isolated service directory for rehearsal.")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_bundle(args.bundle, ROOT / "artifacts/evaluation")
        print(json.dumps({"bundle_id": result["bundle_id"], "parity": result["parity"]}), flush=True)
        return 0
    bundle = ModelBundle(args.bundle)
    settings = replace(Settings.from_config(args.config), model_bundle_dir=args.bundle.resolve(),
                       scheduled_refresh_hour=18, scheduled_refresh_minute=30)
    if args.data_dir:
        directory = args.data_dir.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        settings = replace(settings, root_dir=directory, database_url=f"sqlite:///{(directory / 'prediction_service.db').as_posix()}",
                           archive_dir=directory / "archives", shadow_archive_dir=directory / "shadow_archives",
                           scheduled_refresh_enabled=False)
    with FileLock(str(settings.root_dir / "service.lock"), timeout=0):
        service = PredictionService(settings)
        try:
            service.initialize(bootstrap=False)
            if args.command == "activate":
                service.refresh_calendar()
                _, snapshot, _ = service._load_active_context()
                market = activation_features(bundle, read_features(snapshot.features_path))
                result = publish_portfolio(service, market, source="model_activation",
                                           actor="user_requested_cli", fetch_context=args.fetch, activate=True)
            else:
                result = service.recompute_active()
            print(json.dumps({"snapshot_id": result.snapshot_id, "release_id": result.release_id,
                              "data_as_of": result.data_as_of, "csv_sha256": result.csv_sha256,
                              "rows": len(result.public_frame)}), flush=True)
        finally:
            service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
