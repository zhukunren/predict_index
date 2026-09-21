"""Standalone fixed-contract next-day predictor for the BiLSTM shadow mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import pandas as pd

from scripts import predict as prediction_cli
import 循环验证脚本 as core


SHADOW_SIGNAL_ENGINE = "bilstm_causal"
SHADOW_BILSTM_REFIT_INTERVAL = 5
DEFAULT_OUTPUT_PATH = Path("artifacts/run/bilstm_causal_next_day_prediction.csv")
HEARTBEAT_SECONDS = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the shared predictor CLI while forcing shadow-mode settings."""

    parser = core._build_argument_parser()
    parser.description = "使用固定因果 BiLSTM 引擎预测下一交易日。"
    parser.set_defaults(csv=None, output=str(DEFAULT_OUTPUT_PATH), mode="predict")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON。")
    parser.add_argument("--quiet", action="store_true", help="不显示训练过程。")
    args = parser.parse_args(argv)

    # Keep this entry point safe even when callers pass generic engine flags.
    args.mode = "predict"
    args.loop_validate = False
    args.walk_forward = False
    args.signal_engine = SHADOW_SIGNAL_ENGINE
    args.bilstm_refit_interval = SHADOW_BILSTM_REFIT_INTERVAL
    args.recent_failure_guard = False
    # JSON callers need one machine-readable document on stdout.
    args.verbose = bool(args.verbose) and not args.quiet and not args.json
    return args


def _training_heartbeat(completed: threading.Event, started_at: float) -> None:
    """Report long-running causal replay without changing core-model behavior."""

    while not completed.wait(HEARTBEAT_SECONDS):
        elapsed_seconds = int(time.monotonic() - started_at)
        print(
            f"影子模型仍在训练与历史校准，已耗时 {elapsed_seconds} 秒。",
            file=sys.stderr,
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    csv_path = prediction_cli.resolve_csv_path(args.csv)
    device = core._resolve_device(args.device)
    show_status = not args.quiet and not args.no_progress
    if show_status:
        print(
            "影子模式启动："
            f"引擎={SHADOW_SIGNAL_ENGINE}，重训间隔={SHADOW_BILSTM_REFIT_INTERVAL}，"
            f"设备={device}。",
            file=sys.stderr,
            flush=True,
        )
        print(f"正在读取特征文件：{csv_path}", file=sys.stderr, flush=True)
    data = pd.read_csv(csv_path, encoding=args.encoding)
    if show_status:
        print(
            f"已载入 {len(data)} 行特征；正在进行因果训练与历史校准。",
            file=sys.stderr,
            flush=True,
        )
    options = core._loop_validation_kwargs_from_cli_args(args, output_path=None)
    options["progress"] = False
    completed = threading.Event()
    started_at = time.monotonic()
    heartbeat = (
        threading.Thread(
            target=_training_heartbeat,
            args=(completed, started_at),
            daemon=True,
        )
        if show_status
        else None
    )
    if heartbeat is not None:
        heartbeat.start()
    try:
        result = core.predict_next_day(
            data,
            config=core._config_from_cli_args(args),
            **options,
        )
    finally:
        completed.set()
        if heartbeat is not None:
            heartbeat.join()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_cli.save_prediction_csv(result, output_path)
    if show_status:
        print(f"预测已完成，正在写入：{output_path}", file=sys.stderr, flush=True)
    if args.json:
        print(
            json.dumps(
                core._json_sanitize(result),
                ensure_ascii=False,
                indent=2,
                default=core._json_default,
                allow_nan=False,
            )
        )
    else:
        prediction_cli.print_summary(result, csv_path)
        print(f"结果 CSV: {output_path}")


if __name__ == "__main__":
    main()
