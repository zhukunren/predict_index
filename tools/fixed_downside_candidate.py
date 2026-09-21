"""One frozen research candidate, with no selection on subsequent observations."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

import tushare_prediction_pipeline as pipeline
from prediction_service.archive import sha256_bytes, sha256_file
from prediction_service.config import PROJECT_ROOT
from tools.downside_specialist import (
    COMMON, CANDIDATES, specialist_features, downside_probabilities,
    mean_downside_probabilities, specialist_predictions,
)
from tools.moneyflow_features import FEATURE_COLUMNS, PRICE_FEATURE_COLUMNS, moneyflow_features


ALGORITHM = "moneyflow_price_ensemble_shadow_v1"
MODEL = "downside_logistic"
THRESHOLD = 0.55
SEED_FILES = (
    "features.csv", "baseline.csv", "breadth.csv", "moneyflow.csv", "spx.csv", "nasdaq.csv",
    "contract.json", "selection.json", "candidate_predictions.csv", "report.json",
)
PARITY_COLUMNS = (
    "trade_date", "predicted_label", "predicted_pct_change", "predicted_close",
    "confidence", "calibrated_confidence", "return_calibration_scale",
)
SOURCE_FILES = (
    "\u5faa\u73af\u9a8c\u8bc1\u811a\u672c.py", "return_calibration.py", "regularized_direction.py",
    "tushare_prediction_pipeline.py", "\u6570\u636e\u62c9\u53d6\u811a\u672c_tushare.py",
    "tools/fixed_downside_candidate.py", "tools/run_downside_shadow.py",
    "tools/downside_specialist.py", "tools/moneyflow_features.py", "tools/breadth_features.py",
    "tools/global_risk_features.py", "tools/liquidity_features.py",
    "tools/fetch_market_breadth.py", "tools/fetch_moneyflow_context.py",
    "tools/evaluate_selective_context.py", "tools/context_residual_model.py",
    "tools/funding_features.py", "tools/evaluate_direction_bias.py",
    "tools/direction_rule_candidates.py", "tools/evaluate_prediction_candidate.py",
    "prediction_service/downside_shadow.py", "prediction_service/service.py",
    "prediction_service/engine.py", "prediction_service/archive.py", "prediction_service/models.py",
    "prediction_service/calendar.py", "prediction_service/database.py", "prediction_service/config.py",
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()


def runtime_contract():
    config, options = pipeline._default_calculation_options()
    options = {name: value for name, value in options.items() if not name.endswith("_path")}
    return {
        "algorithm_id": ALGORITHM, "model": MODEL, "threshold": THRESHOLD,
        "common": COMMON, "parameters": CANDIDATES[MODEL], "training_policy": "uniform",
        "ensemble_weights": [0.5, 0.5], "publication_hour_shanghai": 18,
        "direction_config": asdict(config), "baseline_options": options,
        "sources": {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES},
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "torch")},
        "automatic_promotion": False, "historical_gate_passed": False,
    }


def read_frame(path):
    return pd.read_csv(path, float_precision="round_trip")


def predict(market, champion, context):
    config, _ = pipeline._default_calculation_options()
    core = pipeline.prediction_core
    features = specialist_features(core, market, champion, config, context["breadth"],
                                   {name: context[name] for name in ("spx", "nasdaq")})
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    flow = moneyflow_features(champion.trade_date, context["moneyflow"], calendar, price_context=True)
    features = pd.concat([features, flow.loc[:, (*FEATURE_COLUMNS, *PRICE_FEATURE_COLUMNS)]], axis=1)
    available = flow.moneyflow_available.to_numpy()
    base = downside_probabilities(champion, features.drop(columns=list(PRICE_FEATURE_COLUMNS)), MODEL, available=available)
    extended = downside_probabilities(champion, features, MODEL, available=available)
    return specialist_predictions(core, champion, mean_downside_probabilities(base, extended), THRESHOLD)


def assert_parity(expected, actual):
    rows = actual.set_index("trade_date").reindex(expected.trade_date).reset_index()
    pd.testing.assert_frame_equal(expected.loc[:, PARITY_COLUMNS].reset_index(drop=True),
                                  rows.loc[:, PARITY_COLUMNS], check_exact=True, check_dtype=False)


def freeze_candidate(seed: Path, output: Path):
    selection = json.loads((seed / "selection.json").read_text(encoding="utf-8"))["selected"]
    contract = json.loads((seed / "contract.json").read_text(encoding="utf-8"))
    if (selection["candidate"] != MODEL or selection["threshold"] != THRESHOLD
            or not contract.get("moneyflow_price_ensemble") or not contract.get("moneyflow_price_features")):
        raise ValueError("This runner only accepts the fixed equal-weight moneyflow/price candidate.")
    files = {name: (seed / name).read_bytes() for name in SEED_FILES}
    for name in ("features.csv", "baseline.csv", "breadth.csv", "moneyflow.csv", "spx.csv", "nasdaq.csv"):
        if sha256_bytes(files[name]) != contract["inputs"][name]:
            raise ValueError(f"Frozen seed hash mismatch: {name}.")
    frames = {name: pd.read_csv(io.BytesIO(files[f"{name}.csv"]), float_precision="round_trip")
              for name in ("features", "baseline", "breadth", "moneyflow", "spx", "nasdaq")}
    result = predict(frames["features"], frames["baseline"], frames)
    assert_parity(read_frame(seed / "candidate_predictions.csv"), result)
    report = json.loads(files["report.json"])
    if report["passed"] or len(report["checks"]) != 26 or sum(item["passed"] for item in report["checks"]) != 25:
        raise ValueError("Seed must retain its original failed 25/26 research gate.")
    frozen = runtime_contract()
    frozen["seed_files"] = {name: sha256_bytes(content) for name, content in files.items()}
    identifier = sha256_bytes(canonical_json(frozen))
    source_zip = io.BytesIO()
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            archive.writestr(name, (PROJECT_ROOT / name).read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    (output / "seed").mkdir()
    for name, content in files.items():
        (output / "seed" / name).write_bytes(content)
    (output / "sources.zip").write_bytes(source_zip.getvalue())
    (output / "frozen.json").write_bytes(canonical_json({
        "release_id": identifier, "configuration": frozen,
        "source_archive_sha256": sha256_bytes(source_zip.getvalue()),
    }))
    return identifier


def load_candidate(bundle):
    bundle = Path(bundle)
    frozen = json.loads((bundle / "frozen.json").read_text(encoding="utf-8"))
    config = frozen["configuration"]
    if frozen["release_id"] != sha256_bytes(canonical_json(config)):
        raise ValueError("Frozen candidate identity mismatch.")
    if {k: v for k, v in config.items() if k != "seed_files"} != json.loads(canonical_json(runtime_contract())):
        raise ValueError("Frozen candidate code, dependencies or parameters changed; freeze a new version explicitly.")
    if sha256_file(bundle / "sources.zip") != frozen["source_archive_sha256"]:
        raise ValueError("Frozen source archive changed.")
    for name, digest in config["seed_files"].items():
        if name not in SEED_FILES or sha256_file(bundle / "seed" / name) != digest:
            raise ValueError("Frozen candidate seed changed.")
    return frozen


def baseline_predictions(market, seed):
    dates = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    expected = read_frame(seed / "baseline.csv")
    count = int(dates.ge(int(expected.trade_date.iloc[0])).sum())
    if count <= len(expected):
        raise ValueError("The live market snapshot must include the entire research seed.")
    result = pipeline.run_validation_and_prediction(market, validation_days=count - 1, progress=False)
    assert_parity(expected, result)
    return result
