from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict

import pytest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_test_bilstm_causal_evaluation",
    ROOT / "tools" / "evaluate_bilstm_causal.py",
)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def _frame(rows: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=rows)
    actual = np.where(np.arange(rows) % 3 == 0, -0.004, 0.003)
    labels = (np.arange(rows) % 4 != 0).astype(int)
    return pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y%m%d").astype(int),
            "predicted_label": labels,
            "predicted_pct_change": np.where(labels == 1, 0.002, -0.002),
            "predicted_close": 100.0,
            "confidence": 0.60,
            "calibrated_confidence": 0.58,
            "real_pct_change": actual,
            "correct": labels == (actual > 0),
        }
    )


def test_git_blob_is_the_declared_independent_frozen_input():
    blob = evaluation.frozen_input_bytes()
    data = evaluation.load_frozen_input()

    assert evaluation.sha256(blob) == evaluation.FROZEN_INPUT_SHA256
    assert len(data) == 1625
    assert str(data["trade_date"].iloc[-1]) == "2026-09-14"


def test_metrics_and_stability_have_fixed_schema():
    frame = _frame()
    validated = evaluation.validate_frame(frame)
    metrics = evaluation.metric_summary(validated)
    stability, monthly = evaluation.stability_summary(validated)

    assert metrics["rows"] == len(frame)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert metrics["brier"] >= 0.0
    assert metrics["return_mae"] >= 0.0
    assert metrics["return_rmse"] >= metrics["return_mae"]
    assert stability["monthly_periods"] >= 1
    assert stability["rolling_20_accuracy_std"] is not None
    assert list(monthly.columns[:2]) == ["month", "rows"]


def test_bilstm_contract_is_pure_and_uses_fixed_refit_interval():
    assert evaluation.BILSTM_OVERRIDES["signal_engine"] == "bilstm_causal"
    assert evaluation.BILSTM_OVERRIDES["bilstm_refit_interval"] == 5
    assert evaluation.BILSTM_OVERRIDES["recent_failure_guard"] is False
    assert evaluation.DEFAULT_OVERRIDES["signal_engine"] == "state_veto_rule"


@pytest.mark.research_artifacts
def test_v1_candidate_is_compatible_with_production_bilstm_contract():
    import 循环验证脚本 as core

    data = evaluation.load_frozen_input()
    config, production_options = evaluation.production_config_and_options(core)
    options = evaluation._candidate_options(
        core,
        data,
        config,
        production_options,
        evaluation.BILSTM_OVERRIDES,
    )
    frame, parity, provenance = evaluation.load_reusable_bilstm_candidate(
        evaluation.V1_CANDIDATE_DIR,
        config=config,
        options=options,
    )

    assert len(frame) == 412
    assert parity["passed"] is True
    assert provenance["predictions_sha256"] == evaluation.sha256(
        (evaluation.V1_CANDIDATE_DIR / "bilstm_causal_predictions.csv").read_bytes()
    )


@pytest.fixture
def reusable_candidate(tmp_path, monkeypatch):
    """Synthetic loader fixture, never an archived model-performance claim."""
    import 循环验证脚本 as core

    frozen = evaluation.load_frozen_input()
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    config, defaults = evaluation.production_config_and_options(core)
    options = evaluation._candidate_options(core, frozen, config,
                                            defaults, evaluation.BILSTM_OVERRIDES)
    directory = tmp_path / "synthetic_candidate"
    directory.mkdir()
    predictions = _frame(40)
    path = directory / "bilstm_causal_predictions.csv"
    predictions.to_csv(path, index=False)
    contract = {"frozen_input": {"sha256": evaluation.FROZEN_INPUT_SHA256},
                "independent_test_period": {"start_date": evaluation.TEST_START, "end_date": evaluation.TEST_END},
                "model_config": asdict(config), "bilstm_options": options}
    summary = {"parity_passed": True, "bilstm_causal": {"predictions_sha256": evaluation.sha256(path.read_bytes())}}
    for name, value in (("contract.json", contract), ("summary.json", summary),
                        ("bilstm_live_replay_parity.json", {"passed": True})):
        (directory / name).write_text(json.dumps(value))
    return directory, config, options


def test_candidate_loader_accepts_matching_contract_and_hashes(reusable_candidate):
    directory, config, options = reusable_candidate
    frame, parity, provenance = evaluation.load_reusable_bilstm_candidate(directory, config=config, options=options)
    assert len(frame) == 40 and parity["passed"]
    assert provenance["predictions_sha256"] == evaluation.sha256((directory / "bilstm_causal_predictions.csv").read_bytes())


@pytest.mark.parametrize("filename", ["contract.json", "summary.json", "bilstm_causal_predictions.csv", "bilstm_live_replay_parity.json"])
def test_candidate_loader_rejects_missing_evidence(reusable_candidate, filename):
    directory, config, options = reusable_candidate
    (directory / filename).unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        evaluation.load_reusable_bilstm_candidate(directory, config=config, options=options)


@pytest.mark.parametrize("change", ["input", "options", "parity", "predictions"])
def test_candidate_loader_rejects_tampered_evidence(reusable_candidate, change):
    directory, config, options = reusable_candidate
    if change == "predictions":
        path = directory / "bilstm_causal_predictions.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path = directory / ("bilstm_live_replay_parity.json" if change == "parity" else "contract.json")
        content = json.loads(path.read_text())
        if change == "input":
            content["frozen_input"]["sha256"] = "wrong-input"
        elif change == "options":
            content["bilstm_options"]["bilstm_refit_interval"] = 999
        else:
            content["passed"] = False
        path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        evaluation.load_reusable_bilstm_candidate(directory, config=config, options=options)
