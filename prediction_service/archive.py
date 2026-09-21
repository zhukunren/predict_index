"""Write self-describing daily archives without overwriting a prior snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(
        index=False,
        lineterminator="\n",
        # 17 significant digits preserve IEEE-754 values across a CSV round trip.
        float_format="%.17g",
    )
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def read_features(path: str | Path) -> pd.DataFrame:
    """Keep legacy parsing stable; new snapshots use exact float round trips."""
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    version = 2
    if manifest_path.exists():
        version = json.loads(manifest_path.read_text(encoding="utf-8")).get("format_version", 1)
    return pd.read_csv(
        path, encoding="utf-8-sig",
        float_precision="round_trip" if version >= 2 else None,
    )


@dataclass(frozen=True, slots=True)
class ArchiveArtifact:
    directory: Path
    features_path: Path
    results_path: Path
    manifest_path: Path
    diagnostics_path: Path | None
    features_sha256: str
    results_sha256: str


def write_daily_archive(
    *,
    archive_root: Path,
    snapshot_id: str,
    data_as_of: str,
    features: pd.DataFrame,
    public_csv: bytes,
    manifest: dict[str, Any],
    raw_frames: dict[str, pd.DataFrame] | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> ArchiveArtifact:
    """Stage an archive then atomically make the immutable directory visible."""

    parsed_date = datetime.strptime(data_as_of, "%Y%m%d")
    day_dir = archive_root / parsed_date.strftime("%Y") / parsed_date.strftime("%m") / parsed_date.strftime("%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    destination = day_dir / snapshot_id
    if destination.exists():
        raise FileExistsError(f"归档目录已经存在：{destination}")

    staging = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=day_dir))
    try:
        features_path = staging / "features.csv"
        features_path.write_bytes(frame_to_csv_bytes(features))
        results_path = staging / "results.csv"
        results_path.write_bytes(public_csv)

        if raw_frames:
            raw_dir = staging / "raw"
            raw_dir.mkdir()
            for name, frame in raw_frames.items():
                safe_name = "".join(
                    character if character.isalnum() or character in "._-" else "_"
                    for character in name
                )
                (raw_dir / f"{safe_name}.csv").write_bytes(frame_to_csv_bytes(frame))

        if extra_files:
            for relative_name, content in extra_files.items():
                relative_path = Path(relative_name)
                if relative_path.name != relative_name or relative_path.is_absolute():
                    raise ValueError(f"不允许的归档附加文件名：{relative_name}")
                (staging / relative_path).write_bytes(content)

        files = {
            "features.csv": sha256_file(features_path),
            "results.csv": sha256_file(results_path),
        }
        if raw_frames:
            for path in sorted((staging / "raw").glob("*.csv")):
                files[str(path.relative_to(staging)).replace("\\", "/")] = sha256_file(path)
        if extra_files:
            for relative_name in extra_files:
                files[relative_name] = sha256_file(staging / relative_name)

        stored_manifest = {
            "format_version": 2,
            "snapshot_id": snapshot_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": files,
            **manifest,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(stored_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()
        raise

    return ArchiveArtifact(
        directory=destination,
        features_path=destination / "features.csv",
        results_path=destination / "results.csv",
        manifest_path=destination / "manifest.json",
        diagnostics_path=(destination / "diagnostics.csv") if extra_files and "diagnostics.csv" in extra_files else None,
        features_sha256=sha256_file(destination / "features.csv"),
        results_sha256=sha256_file(destination / "results.csv"),
    )
