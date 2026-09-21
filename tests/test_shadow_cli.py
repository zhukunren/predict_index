from __future__ import annotations

import importlib
from pathlib import Path
import threading
import time


def test_shadow_loop_validation_cli_forces_fixed_engine_and_output():
    module = importlib.import_module("scripts.shadow_validation")

    args = module.parse_args(
        [
            "--mode",
            "predict",
            "--walk-forward",
            "--signal-engine",
            "state_veto_rule",
            "--bilstm-refit-interval",
            "99",
            "--recent-failure-guard",
        ]
    )

    assert args.mode == "loop_validate"
    assert args.loop_validate is True
    assert args.walk_forward is False
    assert args.signal_engine == "bilstm_causal"
    assert args.bilstm_refit_interval == 5
    assert args.recent_failure_guard is False
    assert args.periods == 60
    assert Path(args.output) == module.DEFAULT_OUTPUT_PATH


def test_shadow_next_day_cli_forces_fixed_engine_and_output():
    module = importlib.import_module("scripts.shadow_predict")

    args = module.parse_args(
        [
            "--mode",
            "loop_validate",
            "--loop-validate",
            "--signal-engine",
            "state_veto_rule",
            "--bilstm-refit-interval",
            "99",
            "--recent-failure-guard",
        ]
    )

    assert args.mode == "predict"
    assert args.loop_validate is False
    assert args.walk_forward is False
    assert args.signal_engine == "bilstm_causal"
    assert args.bilstm_refit_interval == 5
    assert args.recent_failure_guard is False
    assert args.verbose is False
    assert Path(args.output) == module.DEFAULT_OUTPUT_PATH


def test_shadow_next_day_cli_keeps_json_machine_readable_and_supports_quiet_mode():
    module = importlib.import_module("scripts.shadow_predict")

    assert module.parse_args(["--json"]).verbose is False
    assert module.parse_args(["--quiet"]).verbose is False
    assert module.parse_args(["--verbose"]).verbose is True


def test_shadow_training_heartbeat_reports_long_running_work(capsys, monkeypatch):
    module = importlib.import_module("scripts.shadow_predict")
    monkeypatch.setattr(module, "HEARTBEAT_SECONDS", 0.01)
    completed = threading.Event()
    reporter = threading.Thread(
        target=module._training_heartbeat,
        args=(completed, time.monotonic() - 1.0),
    )

    reporter.start()
    time.sleep(0.03)
    completed.set()
    reporter.join()

    assert "影子模型仍在训练与历史校准" in capsys.readouterr().err
