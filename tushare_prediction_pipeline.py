"""One-command Tushare fetch, rolling validation, and next-day prediction.

The latest source row has no realized next-day return.  It is intentionally
kept as the final ``次日预测`` row, while the preceding rows are completed
``循环验证`` records.  This lets one CSV contain both backtest results and the
current live prediction without treating the latter as an observed outcome.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import 循环验证脚本 as prediction_core


DEFAULT_DATA_START_DATE = "20200101"
DEFAULT_DATA_DIR = Path("market_data")
DEFAULT_OUTPUT_CSV = Path("tushare_validation_prediction.csv")


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是非负整数。") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是非负整数。")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the small, workflow-focused command-line interface."""

    parser = argparse.ArgumentParser(
        description="从 Tushare 拉取指数数据，并输出循环验证和次日预测的合并 CSV。"
    )
    parser.add_argument(
        "--validation-days",
        "--loop-days",
        dest="validation_days",
        type=_nonnegative_int,
        default=60,
        help="写入 CSV 的已完成循环验证交易日数；0 表示只输出次日预测。",
    )
    parser.add_argument(
        "--start-date",
        dest="data_start_date",
        default=DEFAULT_DATA_START_DATE,
        help="Tushare 拉取数据的开始日期，格式 YYYYMMDD。",
    )
    parser.add_argument(
        "--end-date",
        dest="data_end_date",
        default=pd.Timestamp.today().strftime("%Y%m%d"),
        help="Tushare 拉取数据的结束日期，格式 YYYYMMDD。",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Tushare token；未传入时读取 TUSHARE_TOKEN 等环境变量。",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="--save-market-data 的保存目录，也是 --skip-fetch 的读取目录。",
    )
    parser.add_argument(
        "--save-market-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="同时保存 Tushare 原始行情和 merged_features.csv，默认只写最终结果。",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="不调用 Tushare，改用 --data-dir/merged_features.csv 重跑计算。",
    )
    parser.add_argument(
        "--prediction-time",
        choices=("after_close", "before_open"),
        default="after_close",
        help="预测时点；默认在当日收盘后生成下一交易日预测。",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help="包含循环验证和次日预测的最终 CSV 路径。",
    )
    parser.add_argument(
        "--signal-engine",
        choices=prediction_core._SIGNAL_ENGINES,
        default=prediction_core.SCRIPT_SIGNAL_ENGINE,
        help="方向信号引擎；默认使用项目冻结验收的 state_veto_rule。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不输出逐交易日计算进度。",
    )
    parser.add_argument(
        "--retries",
        type=_nonnegative_int,
        default=3,
        help="Tushare 接口失败后的重试次数。",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=1.0,
        help="普通接口失败后的首次重试等待秒数。",
    )
    parser.add_argument(
        "--rate-limit-sleep-seconds",
        type=float,
        default=65.0,
        help="检测到 Tushare 限流后的等待秒数。",
    )
    parser.add_argument(
        "--index-global-min-interval",
        type=float,
        default=6.2,
        help="index_global 接口连续调用的最小间隔秒数。",
    )
    return parser.parse_args(argv)


def fetch_feature_frame(args: argparse.Namespace) -> pd.DataFrame:
    """Fetch the minimal Tushare dataset and return its merged feature frame."""

    # Delay the optional tushare import so --skip-fetch remains usable without it.
    from 数据拉取脚本_tushare import (
        DEFAULT_TUSHARE_MIN_INTERVAL_SECONDS,
        RetryConfig,
        fetch_all,
    )

    retry_config = RetryConfig(
        retries=args.retries,
        sleep_seconds=args.retry_sleep_seconds,
        rate_limit_sleep_seconds=args.rate_limit_sleep_seconds,
    )
    datasets = fetch_all(
        args.data_start_date,
        args.data_end_date,
        token=args.token,
        output_dir=args.data_dir,
        save=args.save_market_data,
        prediction_time=args.prediction_time,
        retry_config=retry_config,
        tushare_min_interval_by_api={
            **DEFAULT_TUSHARE_MIN_INTERVAL_SECONDS,
            "index_global": args.index_global_min_interval,
        },
    )
    feature_frame = datasets["merged_features"]
    if feature_frame.empty:
        raise ValueError("Tushare 未返回可用于预测的合并行情数据。")
    return feature_frame


def load_or_fetch_feature_frame(args: argparse.Namespace) -> pd.DataFrame:
    """Use an existing feature CSV when requested, otherwise fetch from Tushare."""

    if not args.skip_fetch:
        return fetch_feature_frame(args)

    feature_path = Path(args.data_dir) / "merged_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"--skip-fetch 需要已存在的特征文件：{feature_path}"
        )
    return pd.read_csv(feature_path, encoding="utf-8-sig")


def _default_calculation_options() -> tuple[
    prediction_core.DirectionPredictionConfig,
    dict[str, Any],
]:
    """Reuse the accepted predictor defaults instead of copying them here."""

    defaults = prediction_core._build_argument_parser().parse_args([])
    config = prediction_core._config_from_cli_args(defaults)
    options = prediction_core._loop_validation_kwargs_from_cli_args(
        defaults,
        output_path=None,
    )
    return config, options


def run_validation_and_prediction(
    feature_frame: pd.DataFrame,
    *,
    validation_days: int,
    signal_engine: str = prediction_core.SCRIPT_SIGNAL_ENGINE,
    progress: bool = True,
    config: prediction_core.DirectionPredictionConfig | None = None,
    loop_overrides: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return completed validation rows followed by exactly one live prediction.

    ``include_latest=True`` keeps the latest feature date.  Its next-day return
    is unknown, so it is the live prediction.  Requesting one extra period is
    what makes ``validation_days`` equal the number of completed rows instead
    of the total number of output rows.
    """

    if validation_days < 0:
        raise ValueError("validation_days must be nonnegative.")

    default_config, options = _default_calculation_options()
    if loop_overrides:
        options.update(loop_overrides)
    # This script owns the only result file; suppress auxiliary output paths
    # inherited from the main predictor's command-line defaults.
    for name in tuple(options):
        if name.endswith("_path"):
            options[name] = None
    options.update(
        start_date=None,
        end_date=None,
        periods=validation_days + 1,
        include_latest=True,
        output_path=None,
        progress=progress,
        signal_engine=signal_engine,
    )
    result = prediction_core.loop_validate_prediction_results(
        feature_frame,
        config=config or default_config,
        **options,
    )

    expected_rows = validation_days + 1
    if len(result) != expected_rows:
        available_validation_days = max(0, len(result) - 1)
        raise ValueError(
            f"请求 {validation_days} 个循环验证日，但可用数据仅生成 "
            f"{available_validation_days} 个；请扩大 --start-date 的历史范围。"
        )
    if result.empty or pd.notna(result["real_pct_change"].iloc[-1]):
        raise RuntimeError("最新一行必须是没有实际收益的次日预测。")
    if result.iloc[:-1]["real_pct_change"].isna().any():
        raise RuntimeError("循环验证结果中出现缺少实际收益的历史行。")
    return result


def build_combined_csv_frame(result_frame: pd.DataFrame) -> pd.DataFrame:
    """Format the shared public schema and mark validation versus live rows."""

    if result_frame.empty:
        raise ValueError("没有可写入 CSV 的预测结果。")
    record_type = np.where(
        result_frame["real_pct_change"].notna().to_numpy(),
        "循环验证",
        "次日预测",
    )
    public_frame = prediction_core._format_result_frame_for_csv(result_frame)
    public_frame.insert(0, "结果类型", record_type)
    return public_frame


def save_combined_csv(result_frame: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Write the one final CSV and return the public frame for callers/tests."""

    public_frame = build_combined_csv_frame(result_frame)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    public_frame.to_csv(path, index=False, encoding="utf-8-sig")
    return public_frame


def print_summary(
    result_frame: pd.DataFrame,
    *,
    output_path: str | Path,
    signal_engine: str,
) -> None:
    """Print a compact handoff summary without exposing the Tushare token."""

    validation = result_frame.iloc[:-1]
    latest = result_frame.iloc[-1]
    direction = "上涨" if int(latest["predicted_label"]) else "下跌"
    confidence = float(latest.get("calibrated_confidence", latest["confidence"]))
    validation_accuracy = validation["correct"].astype("boolean").mean()

    print(f"循环验证记录: {len(validation)}")
    if len(validation):
        print(f"循环验证方向准确率: {float(validation_accuracy):.2%}")
    print(f"最新信号日期: {int(latest['trade_date'])}")
    print(f"预测方向: {direction}")
    print(f"预测次日涨跌幅: {float(latest['predicted_pct_change']):+.2%}")
    print(f"方向正确概率: {confidence:.2%}")
    print(f"信号引擎: {signal_engine}")
    print(f"结果 CSV: {Path(output_path)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    feature_frame = load_or_fetch_feature_frame(args)
    result_frame = run_validation_and_prediction(
        feature_frame,
        validation_days=args.validation_days,
        signal_engine=args.signal_engine,
        progress=not args.no_progress,
    )
    save_combined_csv(result_frame, args.output)
    print_summary(
        result_frame,
        output_path=args.output,
        signal_engine=args.signal_engine,
    )


if __name__ == "__main__":
    main()
