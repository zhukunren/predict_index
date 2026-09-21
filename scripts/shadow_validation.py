"""Standalone fixed-contract entry point for the causal BiLSTM shadow mode.

The prediction implementation remains in :mod:`循环验证脚本`.  This entry
point keeps the shadow contract explicit and prevents a generic CLI override
from silently running the production engine or its recent-failure guard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import 循环验证脚本 as core


SHADOW_SIGNAL_ENGINE = "bilstm_causal"
SHADOW_BILSTM_REFIT_INTERVAL = 5
SHADOW_VALIDATION_PERIODS = 60
DEFAULT_CSV_PATH = Path("market_data/merged_features.csv")
DEFAULT_OUTPUT_PATH = Path("artifacts/run/bilstm_causal_validation.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the shared CLI while enforcing the standalone shadow contract."""

    parser = core._build_argument_parser()
    parser.description = "使用固定因果 BiLSTM 引擎执行影子模式循环验证。"
    parser.set_defaults(
        csv=str(DEFAULT_CSV_PATH),
        mode="loop_validate",
        periods=SHADOW_VALIDATION_PERIODS,
        output=str(DEFAULT_OUTPUT_PATH),
    )
    args = parser.parse_args(argv)

    # Keep this entry point safe even when callers pass generic engine flags.
    args.mode = "loop_validate"
    args.loop_validate = True
    args.walk_forward = False
    args.signal_engine = SHADOW_SIGNAL_ENGINE
    args.bilstm_refit_interval = SHADOW_BILSTM_REFIT_INTERVAL
    args.recent_failure_guard = False
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到指定 CSV：{csv_path}")

    data = pd.read_csv(csv_path, encoding=args.encoding)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = core._config_from_cli_args(args)
    core._run_loop_validation_cli(data, config, args)


if __name__ == "__main__":
    main()
