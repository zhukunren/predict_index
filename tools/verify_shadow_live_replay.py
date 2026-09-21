"""Verify causal BiLSTM shadow parity on one real or local feature snapshot.

The historical replay receives the complete input frame, while the independent
live prediction receives only rows through the probe date.  A match therefore
checks that the shadow engine neither changes its causal refit anchor nor reads
future rows when producing the same signal.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import 循环验证脚本 as core


SHADOW_SIGNAL_ENGINE = "bilstm_causal"
SHADOW_BILSTM_REFIT_INTERVAL = 5
HEARTBEAT_SECONDS = 30
PARITY_FIELDS = (
    "predicted_label",
    "predicted_pct_change",
    "predicted_close",
    "raw_confidence",
    "calibrated_confidence",
    "return_calibration_scale",
)
T = TypeVar("T")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "market_data" / "merged_features.csv",
        help="用于比较的 merged_features.csv 路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "run" / "shadow_live_replay_parity.json",
        help="JSON 比对报告路径。",
    )
    parser.add_argument(
        "--probe-date",
        default=None,
        help="要比较的信号日期，格式 YYYYMMDD；默认使用倒数第二个交易日。",
    )
    parser.add_argument(
        "--probe-dates",
        default=None,
        help="逗号分隔的多个信号日期；不能与 --probe-date 同时使用。",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=1,
        help="未指定探针日期时，从最近窗口中均匀选取的真实交易日数量。",
    )
    parser.add_argument(
        "--probe-window-days",
        type=int,
        default=60,
        help="自动选取探针时使用最近多少个可回测交易日。",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_shadow_config(*, device: str, epochs: int) -> core.DirectionPredictionConfig:
    defaults = core._build_argument_parser().parse_args([])
    return replace(
        core._config_from_cli_args(defaults),
        device=device,
        epochs=epochs,
        verbose=False,
    )


def _build_shadow_options() -> dict[str, Any]:
    defaults = core._build_argument_parser().parse_args([])
    options = core._loop_validation_kwargs_from_cli_args(defaults, output_path=None)
    for name in tuple(options):
        if name.endswith("_path"):
            options[name] = None
    options.update(
        signal_engine=SHADOW_SIGNAL_ENGINE,
        bilstm_refit_interval=SHADOW_BILSTM_REFIT_INTERVAL,
        recent_failure_guard=False,
        progress=False,
    )
    return options


def _resolve_probe_dates(
    frame: pd.DataFrame,
    *,
    requested_date: str | None,
    requested_dates: str | None,
    probe_count: int,
    probe_window_days: int,
) -> list[pd.Timestamp]:
    dates = pd.to_datetime(frame["trade_date"], errors="raise")
    unique_dates = dates.drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_dates) < 2:
        raise ValueError("至少需要两个交易日才能验证实时前缀不读取未来行情。")
    if probe_count < 1:
        raise ValueError("--probe-count 必须至少为 1。")
    if probe_window_days < 1:
        raise ValueError("--probe-window-days 必须至少为 1。")
    if requested_date and requested_dates:
        raise ValueError("--probe-date 不能与 --probe-dates 同时使用。")

    if requested_date:
        requested = [requested_date]
    elif requested_dates:
        requested = [part.strip() for part in requested_dates.split(",") if part.strip()]
        if not requested:
            raise ValueError("--probe-dates 至少需要一个有效日期。")
    else:
        candidates = unique_dates.iloc[:-1]
        window = candidates.iloc[-min(len(candidates), probe_window_days) :]
        positions = np.linspace(
            0,
            len(window) - 1,
            min(probe_count, len(window)),
            dtype=int,
        )
        return [pd.Timestamp(window.iloc[index]) for index in np.unique(positions)]

    probes = sorted({pd.Timestamp(str(value)) for value in requested})
    for probe in probes:
        if not dates.eq(probe).any():
            raise ValueError(f"输入特征不存在探针日期：{probe:%Y%m%d}")
        if probe >= unique_dates.iloc[-1]:
            raise ValueError("探针日期必须早于输入中的最后一个交易日。")
    return probes


def _heartbeat(completed: threading.Event, label: str, started_at: float) -> None:
    while not completed.wait(HEARTBEAT_SECONDS):
        elapsed = int(time.monotonic() - started_at)
        print(f"{label}仍在计算，已耗时 {elapsed} 秒。", file=sys.stderr, flush=True)


def _run_with_heartbeat(label: str, operation: Callable[[], T]) -> T:
    print(f"开始{label}...", file=sys.stderr, flush=True)
    completed = threading.Event()
    started_at = time.monotonic()
    reporter = threading.Thread(
        target=_heartbeat,
        args=(completed, label, started_at),
        daemon=True,
    )
    reporter.start()
    try:
        return operation()
    finally:
        completed.set()
        reporter.join()
        elapsed = time.monotonic() - started_at
        print(f"{label}完成，耗时 {elapsed:.1f} 秒。", file=sys.stderr, flush=True)


def _numeric_check(actual: float, expected: float, *, atol: float = 1e-10) -> bool:
    return bool(np.isclose(actual, expected, rtol=0.0, atol=atol, equal_nan=True))


def compare_live_to_replay(
    live: dict[str, Any],
    replay_row: pd.Series,
) -> tuple[dict[str, bool], dict[str, dict[str, float | int]]]:
    """Compare all externally meaningful shadow prediction fields."""

    expected = {
        "predicted_label": int(replay_row["predicted_label"]),
        "predicted_pct_change": float(replay_row["predicted_pct_change"]),
        "predicted_close": float(replay_row["predicted_close"]),
        "raw_confidence": float(replay_row["confidence"]),
        "calibrated_confidence": float(replay_row["calibrated_confidence"]),
        "return_calibration_scale": float(replay_row["return_calibration_scale"]),
    }
    actual = {
        "predicted_label": int(live["predicted_label"]),
        "predicted_pct_change": float(live["estimated_next_return"]),
        "predicted_close": float(live["estimated_next_close"]),
        "raw_confidence": float(live["raw_confidence"]),
        "calibrated_confidence": float(live["calibrated_confidence"]),
        "return_calibration_scale": float(live["return_calibration_scale"]),
    }
    checks = {
        "predicted_label": actual["predicted_label"] == expected["predicted_label"],
        **{
            field: _numeric_check(float(actual[field]), float(expected[field]))
            for field in PARITY_FIELDS
            if field != "predicted_label"
        },
    }
    values = {
        field: {"live": actual[field], "historical_replay": expected[field]}
        for field in PARITY_FIELDS
    }
    return checks, values


def verify(
    frame: pd.DataFrame,
    *,
    config: core.DirectionPredictionConfig,
    options: dict[str, Any],
    probe_dates: list[pd.Timestamp],
) -> dict[str, Any]:
    source_dates = pd.to_datetime(frame["trade_date"], errors="raise")
    if not probe_dates:
        raise ValueError("至少需要一个探针日期。")
    historical_options = options | {
        "start_date": min(probe_dates).strftime("%Y%m%d"),
        "end_date": max(probe_dates).strftime("%Y%m%d"),
        "periods": 0,
        "include_latest": False,
    }
    replay = _run_with_heartbeat(
        "历史因果回放",
        lambda: core.loop_validate_prediction_results(
            frame,
            config=config,
            **historical_options,
        ),
    )
    replay_by_date = replay.set_index("trade_date", drop=False)
    records: list[dict[str, Any]] = []
    for probe_date in probe_dates:
        date_number = int(probe_date.strftime("%Y%m%d"))
        if date_number not in replay_by_date.index:
            raise RuntimeError(f"历史回放缺少探针日期：{date_number}")
        prefix = frame.loc[source_dates.le(probe_date)].copy()
        live = _run_with_heartbeat(
            f"独立实时前缀预测 {date_number}",
            lambda prefix=prefix: core.predict_next_day(
                prefix,
                config=config,
                **options,
            ),
        )
        checks, values = compare_live_to_replay(live, replay_by_date.loc[date_number])
        records.append(
            {
                "probe_date": str(date_number),
                "prefix_last_date": pd.Timestamp(prefix["trade_date"].iloc[-1]).strftime("%Y%m%d"),
                "future_rows_excluded_from_live": int(len(frame) - len(prefix)),
                "passed": all(checks.values()),
                "checks": checks,
                "values": values,
            }
        )
    return {
        "method": "full_frame_historical_replay_vs_independent_live_prefix",
        "passed": all(record["passed"] for record in records),
        "probe_count": len(records),
        "historical_replay_start_date": min(probe_dates).strftime("%Y%m%d"),
        "historical_replay_end_date": max(probe_dates).strftime("%Y%m%d"),
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"找不到特征文件：{input_path}")
    if args.epochs < 1:
        raise ValueError("--epochs 必须至少为 1。")

    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    probe_dates = _resolve_probe_dates(
        frame,
        requested_date=args.probe_date,
        requested_dates=args.probe_dates,
        probe_count=args.probe_count,
        probe_window_days=args.probe_window_days,
    )
    config = _build_shadow_config(device=args.device, epochs=args.epochs)
    options = _build_shadow_options()
    result = verify(frame, config=config, options=options, probe_dates=probe_dates)
    report = {
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "rows": len(frame),
            "data_as_of": pd.Timestamp(frame["trade_date"].iloc[-1]).strftime("%Y%m%d"),
        },
        "shadow_contract": {
            "signal_engine": SHADOW_SIGNAL_ENGINE,
            "bilstm_refit_interval": SHADOW_BILSTM_REFIT_INTERVAL,
            "recent_failure_guard": False,
            "config": asdict(config),
        },
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"parity_passed={report['passed']}")
    print(f"report={args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
