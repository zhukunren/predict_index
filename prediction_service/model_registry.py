"""Content-addressed bundles for the explicitly selected four-model portfolio."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
import io
import json
from pathlib import Path
import zipfile

import pandas as pd

import tushare_prediction_pipeline as pipeline
from .archive import sha256_bytes, sha256_file
from .config import PROJECT_ROOT
from .forecast_models import MODELS, PRODUCTION_KEY, CONTEXT_NAMES, calculate_models, assert_parity
from tools.downside_specialist import COMMON, CANDIDATES


SOURCE_FILES = (
    "循环验证脚本.py", "return_calibration.py", "regularized_direction.py", "tushare_prediction_pipeline.py",
    "数据拉取脚本_tushare.py", "prediction_service/forecast_models.py", "prediction_service/model_registry.py",
    "tools/downside_specialist.py", "tools/moneyflow_features.py", "tools/breadth_features.py",
    "tools/global_risk_features.py", "tools/liquidity_features.py", "tools/option_features.py",
    "tools/option_position_features.py", "tools/evaluate_selective_context.py", "tools/context_residual_model.py",
    "tools/evaluate_direction_bias.py", "tools/direction_rule_candidates.py", "tools/evaluate_prediction_candidate.py",
)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def read_frame(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


def runtime_contract():
    config, options = pipeline._default_calculation_options()
    return {
        "schema_version": 1,
        "models": [asdict(model) for model in MODELS],
        "production_key": PRODUCTION_KEY,
        "common": COMMON, "classifier": CANDIDATES["downside_logistic"],
        "training_policy": "uniform", "ensemble_weights": [0.5, 0.5],
        "publication_hour_shanghai": 18, "publication_minute_shanghai": 30,
        "option_lag_sessions": 0,
        "option_positions": True, "moneyflow_lag_sessions": 1,
        "direction_config": asdict(config),
        "baseline_options": {k: v for k, v in options.items() if not k.endswith("_path")},
        "sources": {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES},
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "torch")},
    }


def prepare_bundle(output: Path, evaluations: Path):
    """Freeze selected experiments without changing their failed research gates."""
    if output.exists():
        raise FileExistsError(f"Model bundle already exists: {output}")
    option = evaluations / "option_moneyflow_ensemble_v1"
    price = evaluations / "ensemble_moneyflow_price_downside_v1"
    files = {}
    for model in MODELS:
        if model.experiment is None:
            continue
        directory = evaluations / model.experiment
        contract = json.loads((directory / "contract.json").read_text(encoding="utf-8"))
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))["selected"]
        if selection["candidate"] != "downside_logistic" or selection["threshold"] != model.threshold:
            raise ValueError(f"Selected experiment changed: {model.key}")
        for name, digest in contract["inputs"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Research input hash mismatch: {model.key}/{name}")
        for name in ("contract.json", "selection.json", "report.json", "candidate_predictions.csv"):
            files[f"{model.key}_{name}"] = (directory / name).read_bytes()
    for name in ("features", "baseline", "breadth", "options", "spx", "nasdaq"):
        files[f"{name}.csv"] = (option / f"{name}.csv").read_bytes()
    files["moneyflow.csv"] = (price / "moneyflow.csv").read_bytes()
    frames = {name: pd.read_csv(io.BytesIO(files[f"{name}.csv"]), float_precision="round_trip")
              for name in ("features", "baseline", *CONTEXT_NAMES)}
    original_flow = read_frame(option / "moneyflow.csv")
    pd.testing.assert_frame_equal(original_flow, frames["moneyflow"].loc[:, original_flow.columns], check_exact=True)
    results = calculate_models(frames["features"], frames["baseline"], frames)
    for model in MODELS:
        expected = frames["baseline"] if model.key == "baseline" else read_frame(evaluations / model.experiment / "candidate_predictions.csv")
        assert_parity(expected, results[model.key])
    runtime = runtime_contract()
    seeds = {name: sha256_bytes(content) for name, content in files.items()}
    releases = {}
    for model in MODELS:
        config = {"algorithm_id": f"fixed_{model.key}_v1", "model_key": model.key, "model_name": model.name,
                  "runtime": runtime, "seed_files": seeds,
                  "historical_gate_passed": model.key == "baseline",
                  "activation_basis": "explicit_user_selection"}
        releases[model.key] = {"release_id": sha256_bytes(canonical_json(config)), "configuration": config}
    source_buffer = io.BytesIO()
    with zipfile.ZipFile(source_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            archive.writestr(name, (PROJECT_ROOT / name).read_bytes())
    manifest = {"runtime": runtime, "seed_files": seeds, "releases": releases,
                "source_archive_sha256": sha256_bytes(source_buffer.getvalue()),
                "parity": {model.key: {"rows": len(results[model.key]), "exact_match": True} for model in MODELS}}
    manifest["bundle_id"] = sha256_bytes(canonical_json(manifest))
    output.mkdir(parents=True)
    (output / "seed").mkdir()
    for name, content in files.items():
        (output / "seed" / name).write_bytes(content)
    (output / "sources.zip").write_bytes(source_buffer.getvalue())
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    return manifest


class ModelBundle:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.verify()

    def verify(self):
        manifest = self.manifest
        if sha256_bytes(canonical_json({k: v for k, v in manifest.items() if k != "bundle_id"})) != manifest["bundle_id"]:
            raise ValueError("Model bundle identity changed.")
        if manifest["runtime"] != json.loads(canonical_json(runtime_contract())):
            raise ValueError("Frozen model code, parameters or dependencies changed; prepare a new version explicitly.")
        if sha256_file(self.directory / "sources.zip") != manifest["source_archive_sha256"]:
            raise ValueError("Frozen model source archive changed.")
        for name, digest in manifest["seed_files"].items():
            if Path(name).name != name or sha256_file(self.directory / "seed" / name) != digest:
                raise ValueError(f"Frozen model input changed: {name}")
        for key, release in manifest["releases"].items():
            if release["release_id"] != sha256_bytes(canonical_json(release["configuration"])):
                raise ValueError(f"Model release identity changed: {key}")

    def context(self):
        return {name: read_frame(self.directory / "seed" / f"{name}.csv") for name in CONTEXT_NAMES}

    def calculate(self, market, context):
        seed = self.directory / "seed"
        expected = read_frame(seed / "baseline.csv")
        dates = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
        count = int(dates.ge(int(expected.trade_date.iloc[0])).sum())
        if count < len(expected):
            raise ValueError("Market data do not cover the frozen research history.")
        baseline = pipeline.run_validation_and_prediction(market, validation_days=count - 1, progress=False)
        assert_parity(expected, baseline)
        results = calculate_models(market, baseline, context)
        for model in MODELS:
            if model.experiment:
                assert_parity(read_frame(seed / f"{model.key}_candidate_predictions.csv"), results[model.key])
        self.verify()
        return results
