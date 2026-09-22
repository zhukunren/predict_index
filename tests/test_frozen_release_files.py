"""Frozen assets must survive a checkout without line-ending conversion."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile


def test_committed_portfolio_files_match_their_original_manifest():
    root = Path(__file__).resolve().parents[1]
    bundle = root / "artifacts/releases/four_model_portfolio_20260919"
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["seed_files"].items():
        assert sha256((bundle / "seed" / name).read_bytes()).hexdigest() == expected, name
    with zipfile.ZipFile(bundle / "sources.zip") as archive:
        for name, expected in manifest["runtime"]["sources"].items():
            assert sha256(archive.read(name)).hexdigest() == expected, name
    assert sha256((bundle / "sources.zip").read_bytes()).hexdigest() == manifest["source_archive_sha256"]
