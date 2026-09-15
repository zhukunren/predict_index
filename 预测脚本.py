"""Predict only the next trading day's direction from local market features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from 循环验证脚本 import (
    DirectionPredictionConfig,
    _json_default,
    _json_sanitize,
    _format_result_frame_for_csv,
    predict_next_day,
)


DEFAULT_CANDIDATE_CSVS = (
    Path("market_data_akshare/merged_features.csv"),
    Path("market_data/merged_features.csv"),
    Path("merged_features.csv"),
)
DEFAULT_OUTPUT_CSV = Path("drp_feim_next_day_prediction.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅预测下一交易日结果并生成一行结果 CSV，不跑回测。"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="本地 merged_features.csv 路径；不填则自动查找最新的默认文件。",
    )
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV 文件编码。")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help="预测结果 CSV 输出路径，默认使用与循环验证相同的中文列名格式。",
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数。")
    parser.add_argument("--lookback", type=int, default=30, help="序列回看交易日数。")
    parser.add_argument(
        "--neutral-band",
        type=float,
        default=0.001,
        help="方向标签中性区间，默认 +/-0.1%。",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="训练设备。",
    )
    parser.add_argument(
        "--external-feature-mode",
        choices=["none", "core", "all"],
        default="core",
        help="是否使用合并特征中的外部市场特征。",
    )
    parser.add_argument(
        "--technical-feature-mode",
        choices=["none", "v1", "v1_core"],
        default="none",
        help="是否额外生成技术指标特征。",
    )
    parser.add_argument("--json", action="store_true", help="输出完整 JSON。")
    return parser.parse_args()


def resolve_csv_path(csv_arg: str | None) -> Path:
    if csv_arg:
        path = Path(csv_arg)
        if not path.exists():
            raise FileNotFoundError(f"找不到指定 CSV：{path}")
        return path

    existing = [path for path in DEFAULT_CANDIDATE_CSVS if path.exists()]
    if not existing:
        choices = ", ".join(str(path) for path in DEFAULT_CANDIDATE_CSVS)
        raise FileNotFoundError(f"未找到默认特征文件，请指定 CSV。默认查找：{choices}")

    return max(existing, key=lambda path: path.stat().st_mtime)


def build_config(args: argparse.Namespace) -> DirectionPredictionConfig:
    return DirectionPredictionConfig(
        epochs=args.epochs,
        lookback=args.lookback,
        neutral_band=args.neutral_band,
        device=args.device,
        external_feature_mode=args.external_feature_mode,
        technical_feature_mode=args.technical_feature_mode,
        verbose=False,
    )


def build_prediction_csv_frame(result: dict[str, Any]) -> pd.DataFrame:
    """Build one next-day prediction row with the loop-validation CSV schema."""

    signal_date = int(pd.Timestamp(result["last_date"]).strftime("%Y%m%d"))
    raw_confidence = float(result.get("raw_confidence", result["confidence"]))
    calibrated_confidence = float(
        result.get("calibrated_confidence", result["confidence"])
    )
    internal_frame = pd.DataFrame(
        [
            {
                "trade_date": signal_date,
                "predicted_pct_change": float(result["estimated_next_return"]),
                "predicted_close": float(result["estimated_next_close"]),
                "confidence": raw_confidence,
                "calibrated_confidence": calibrated_confidence,
                # 次日尚未发生，因此实际收益和方向是否正确暂时为空。
                "real_pct_change": pd.NA,
                "correct": pd.NA,
                "confidence_calibration_status": result.get(
                    "confidence_calibration_status", pd.NA
                ),
                "confidence_calibration_window": pd.NA,
                "confidence_calibration_method": result.get(
                    "confidence_calibration_method", pd.NA
                ),
                "confidence_calibration_rows": result.get(
                    "confidence_calibration_rows", pd.NA
                ),
                "confidence_calibration_fallback": result.get(
                    "confidence_calibration_fallback", pd.NA
                ),
            }
        ]
    )
    return _format_result_frame_for_csv(internal_frame)


def save_prediction_csv(result: dict[str, Any], output_path: Path) -> pd.DataFrame:
    """Save the next-day prediction as a one-row, loop-compatible CSV."""

    frame = build_prediction_csv_frame(result)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame


def print_summary(result: dict[str, Any], csv_path: Path) -> None:
    direction = "上涨" if result["predicted_direction"] == "up" else "下跌"
    sign = "+" if result["estimated_next_pct_change"] >= 0 else ""

    print("下一交易日预测")
    print(f"数据文件: {csv_path}")
    print(f"最后交易日: {result['last_date']}")
    print(f"最后收盘价: {result['last_close']:.4f}")
    print(f"预测方向: {direction}")
    print(f"上涨概率: {result['probability_up']:.2%}")
    print(f"决策阈值: {result['decision_threshold']:.2%}")
    if "raw_confidence" in result:
        print(f"原始边界分数: {result['raw_confidence']:.2%}")
    print(f"置信度: {result['confidence']:.2%}")
    if "confidence_calibration_status" in result:
        print(f"置信度状态: {result['confidence_calibration_status']}")
        print(
            "置信度校准样本数: "
            f"{result['confidence_calibration_rows']}"
        )
    print(f"估计涨跌幅: {sign}{result['estimated_next_pct_change']:.2f}%")
    print(f"估计收盘价: {result['estimated_next_close']:.4f}")


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    data = pd.read_csv(csv_path, encoding=args.encoding)
    result = predict_next_day(data, config=build_config(args))
    output_path = Path(args.output)
    save_prediction_csv(result, output_path)

    if args.json:
        print(
            json.dumps(
                _json_sanitize(result),
                ensure_ascii=False,
                indent=2,
                default=_json_default,
                allow_nan=False,
            )
        )
    else:
        print_summary(result, csv_path)
        print(f"结果 CSV: {output_path}")


if __name__ == "__main__":
    main()
