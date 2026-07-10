"""当前 DRP-FEIM 预测器使用的 Wind 数据构建脚本。

脚本从 Wind 拉取预测器所需的最小行情输入：

* Wind ``wsd`` 的 000001.SH -> ``market_data/000001_sh.csv``
* Wind ``wsd`` 的 HSI.HI -> ``market_data/hangseng.csv``
* 合并后的特征表 -> ``market_data/merged_features.csv``

输出列结构保持与现有预测脚本兼容。
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from WindPy import w


DEFAULT_OUTPUT_DIR = Path("market_data")
PredictionTime = Literal["after_close", "before_open"]


class ChineseArgumentParser(argparse.ArgumentParser):
    """将 argparse 自动生成的 usage 前缀改为中文。"""

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法:", 1)

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法:", 1)


TARGET_NAME = "000001_sh"
HANGSENG_NAME = "hangseng"
HANGSENG_FEATURES_KEY = "_existing_hangseng_features"
TARGET_CANDIDATE_FILES = (
    "000001_sh.csv",
    "sh000001.csv",
    "wind.csv",
)

DEFAULT_TARGET_CODE = "000001.SH"
DEFAULT_HANGSENG_CODE = "HSI.HI"
WIND_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "chg",
    "pct_chg",
    "volume",
    "amt",
)


@dataclass(frozen=True)
class RetryConfig:
    retries: int = 3
    sleep_seconds: float = 1.0
    backoff_factor: float = 2.0
    jitter_seconds: float = 0.25


DEFAULT_RETRY_CONFIG = RetryConfig()


def normalize_date(value: str | int | pd.Timestamp) -> str:
    text = str(value).strip()
    if text.isdigit() and len(text) == 8:
        return text
    return pd.Timestamp(value).strftime("%Y%m%d")


def wind_date(value: str | int | pd.Timestamp) -> str:
    return pd.Timestamp(normalize_date(value)).strftime("%Y-%m-%d")


def ensure_output_dir(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_call(
    fetcher: Callable[[], pd.DataFrame],
    *,
    name: str,
    retry_config: RetryConfig = DEFAULT_RETRY_CONFIG,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retry_config.retries + 1):
        try:
            df = fetcher()
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as exc:
            last_error = exc
            if attempt < retry_config.retries:
                sleep_seconds = retry_config.sleep_seconds * (
                    retry_config.backoff_factor**attempt
                )
                if retry_config.jitter_seconds > 0:
                    sleep_seconds += random.uniform(0, retry_config.jitter_seconds)
                print(
                    f"[等待] {name} 第 {attempt + 1} 次尝试失败；"
                    f"{sleep_seconds:.1f}s 后重试：{exc}"
                )
                time.sleep(sleep_seconds)
    print(f"[警告] {name} 调用失败：{last_error}")
    return pd.DataFrame()


def start_wind() -> None:
    status = w.start()
    error_code = int(getattr(status, "ErrorCode", 0) or 0)
    if error_code != 0:
        raise RuntimeError(f"WindPy 启动失败，ErrorCode={error_code}")
    if hasattr(w, "isconnected") and not bool(w.isconnected()):
        raise RuntimeError("WindPy 未连接。请检查 Wind 终端是否已启动并登录。")


def fetch_wind_daily(
    code: str,
    start_date: str | int,
    end_date: str | int,
    *,
    fields: tuple[str, ...] = WIND_FIELDS,
    options: str = "PriceAdj=F",
    retry_config: RetryConfig = DEFAULT_RETRY_CONFIG,
) -> pd.DataFrame:
    """通过 ``w.wsd`` 拉取单个 Wind 证券/指数的日线数据。"""

    start = wind_date(start_date)
    end = wind_date(end_date)
    field_text = ",".join(fields)

    def call() -> pd.DataFrame:
        wind_data = w.wsd(code, field_text, start, end, options)
        error_code = int(getattr(wind_data, "ErrorCode", 0) or 0)
        if error_code != 0:
            raise RuntimeError(
                f"Wind wsd 拉取 {code} 失败，ErrorCode={error_code}"
            )
        times = list(getattr(wind_data, "Times", []) or [])
        data = list(getattr(wind_data, "Data", []) or [])
        if not times or not data:
            return pd.DataFrame()
        frame = pd.DataFrame({"trade_date": pd.to_datetime(times, errors="coerce")})
        for field, values in zip(fields, data, strict=False):
            frame[field] = values
        return frame

    raw = safe_call(
        call,
        name=f"wind:wsd:{code}",
        retry_config=retry_config,
    )
    if raw.empty:
        return raw
    return _sort_output_frame(raw)


def fetch_shanghai_composite(
    start_date: str | int,
    end_date: str | int,
    *,
    code: str = DEFAULT_TARGET_CODE,
    retry_config: RetryConfig = DEFAULT_RETRY_CONFIG,
) -> pd.DataFrame:
    raw = fetch_wind_daily(
        code,
        start_date,
        end_date,
        retry_config=retry_config,
    )
    if raw.empty:
        return raw
    return _to_target_raw_frame(raw, ts_code=code)


def fetch_hangseng(
    start_date: str | int,
    end_date: str | int,
    *,
    code: str = DEFAULT_HANGSENG_CODE,
    retry_config: RetryConfig = DEFAULT_RETRY_CONFIG,
) -> pd.DataFrame:
    raw = fetch_wind_daily(
        code,
        start_date,
        end_date,
        retry_config=retry_config,
    )
    if raw.empty:
        return raw
    return _to_hangseng_raw_frame(raw)


def fetch_all(
    start_date: str | int,
    end_date: str | int,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save: bool = True,
    prediction_time: PredictionTime = "after_close",
    feature_lags: dict[str, int] | None = None,
    retry_config: RetryConfig = DEFAULT_RETRY_CONFIG,
    target_code: str = DEFAULT_TARGET_CODE,
    hangseng_code: str = DEFAULT_HANGSENG_CODE,
    sleep_between_calls_seconds: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """拉取 Wind 数据并生成 ``merged_features.csv``。"""

    start_wind()
    out_dir = ensure_output_dir(output_dir)
    datasets: dict[str, pd.DataFrame] = {}

    target = fetch_shanghai_composite(
        start_date,
        end_date,
        code=target_code,
        retry_config=retry_config,
    )
    datasets[TARGET_NAME] = target
    if save and not target.empty:
        target.to_csv(out_dir / f"{TARGET_NAME}.csv", index=False, encoding="utf-8-sig")

    if sleep_between_calls_seconds > 0:
        time.sleep(sleep_between_calls_seconds)

    hangseng = fetch_hangseng(
        start_date,
        end_date,
        code=hangseng_code,
        retry_config=retry_config,
    )
    datasets[HANGSENG_NAME] = hangseng
    if save and not hangseng.empty:
        hangseng.to_csv(
            out_dir / f"{HANGSENG_NAME}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    feature = make_feature_frame(
        datasets,
        prediction_time=prediction_time,
        feature_lags=feature_lags,
    )
    datasets["merged_features"] = feature
    if save:
        feature.to_csv(out_dir / "merged_features.csv", index=False, encoding="utf-8-sig")
    return datasets


def load_saved_datasets(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    include_merged: bool = False,
) -> dict[str, pd.DataFrame]:
    """读取已保存的最小行情 CSV，不调用 Wind。"""

    out_dir = Path(output_dir)
    datasets: dict[str, pd.DataFrame] = {}
    merged_path = out_dir / "merged_features.csv"
    merged_frame = pd.read_csv(merged_path) if merged_path.exists() else None

    for filename in TARGET_CANDIDATE_FILES:
        path = out_dir / filename
        if path.exists():
            datasets[TARGET_NAME] = pd.read_csv(path)
            break
    if TARGET_NAME not in datasets and merged_frame is not None:
        datasets[TARGET_NAME] = merged_frame

    hangseng_path = out_dir / f"{HANGSENG_NAME}.csv"
    if hangseng_path.exists():
        datasets[HANGSENG_NAME] = pd.read_csv(hangseng_path)
    elif merged_frame is not None and any(
        str(column).startswith("hangseng_") for column in merged_frame.columns
    ):
        datasets[HANGSENG_FEATURES_KEY] = merged_frame

    if include_merged and merged_frame is not None:
        datasets["merged_features"] = merged_frame

    return datasets


def rebuild_merged_features(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    target_name: str = TARGET_NAME,
    prediction_time: PredictionTime = "after_close",
    feature_lags: dict[str, int] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """从已保存的原始 CSV 重建 ``merged_features.csv``。"""

    out_dir = ensure_output_dir(output_dir)
    datasets = load_saved_datasets(out_dir)
    feature = make_feature_frame(
        datasets,
        target_name=target_name,
        prediction_time=prediction_time,
        feature_lags=feature_lags,
    )
    if save:
        feature.to_csv(out_dir / "merged_features.csv", index=False, encoding="utf-8-sig")
    return feature


def make_feature_frame(
    datasets: dict[str, pd.DataFrame],
    *,
    target_name: str = TARGET_NAME,
    prediction_time: PredictionTime = "after_close",
    feature_lags: dict[str, int] | None = None,
) -> pd.DataFrame:
    """构建当前预测器所需的最小合并特征表。"""

    if target_name not in datasets or datasets[target_name].empty:
        raise ValueError(
            "缺少目标指数数据。请在输出目录中提供以下文件之一："
            f"{', '.join(TARGET_CANDIDATE_FILES)}。"
        )

    feature = _target_feature_frame(datasets[target_name])

    hangseng = datasets.get(HANGSENG_NAME)
    if hangseng is not None and not hangseng.empty:
        lag = (
            feature_lags[HANGSENG_NAME]
            if feature_lags is not None and HANGSENG_NAME in feature_lags
            else default_feature_lag(HANGSENG_NAME, prediction_time)
        )
        part = _asset_feature_frame(HANGSENG_NAME, hangseng, lag=lag)
        if not part.empty:
            feature = feature.merge(part, on="trade_date", how="left")
    elif HANGSENG_FEATURES_KEY in datasets:
        part = _existing_hangseng_feature_frame(datasets[HANGSENG_FEATURES_KEY])
        if not part.empty:
            feature = feature.merge(part, on="trade_date", how="left")

    return feature.sort_values("trade_date").reset_index(drop=True)


def default_feature_lag(
    name: str,
    prediction_time: PredictionTime = "after_close",
) -> int:
    if prediction_time not in {"after_close", "before_open"}:
        raise ValueError("prediction_time 必须是 'after_close' 或 'before_open'。")
    if name == HANGSENG_NAME:
        return 1
    return 0


def _to_target_raw_frame(df: pd.DataFrame, *, ts_code: str) -> pd.DataFrame:
    source = _normalize_price_frame(df)
    required = ["trade_date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"目标指数数据缺少必要列：{', '.join(missing)}")

    out = pd.DataFrame(
        {
            "ts_code": ts_code,
            "trade_date": source["trade_date"],
            "open": source["open"],
            "close": source["close"],
            "high": source["high"],
            "low": source["low"],
        }
    )
    out["pre_close"] = source.get("pre_close", out["close"].shift(1))
    out["pre_close"] = out["pre_close"].where(out["pre_close"].gt(0), out["close"].shift(1))
    out["change"] = source.get("change", out["close"] - out["pre_close"])
    out["pct_chg"] = source.get(
        "pct_chg",
        (out["close"] / out["pre_close"] - 1.0) * 100.0,
    )
    out["vol"] = source.get("vol", pd.Series(np.nan, index=source.index))
    out["amount"] = source.get("amount", pd.Series(np.nan, index=source.index))
    return _sort_output_frame(out)


def _to_hangseng_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    source = _normalize_price_frame(df)
    required = ["trade_date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"恒生指数数据缺少必要列：{', '.join(missing)}")

    out = pd.DataFrame(
        {
            "ts_code": "HSI",
            "trade_date": source["trade_date"],
            "open": source["open"],
            "close": source["close"],
            "high": source["high"],
            "low": source["low"],
        }
    )
    out["pre_close"] = source.get("pre_close", out["close"].shift(1))
    out["pre_close"] = out["pre_close"].where(out["pre_close"].gt(0), out["close"].shift(1))
    out["change"] = source.get("change", out["close"] - out["pre_close"])
    out["pct_chg"] = source.get(
        "pct_chg",
        (out["close"] / out["pre_close"] - 1.0) * 100.0,
    )
    out["swing"] = source.get(
        "swing",
        (out["high"] - out["low"]) / out["pre_close"] * 100.0,
    )
    out["vol"] = source.get("vol", pd.Series(np.nan, index=source.index))
    return _sort_output_frame(out)


def _target_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    source = _normalize_price_frame(df)
    required = ["trade_date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"目标指数数据缺少必要列：{', '.join(missing)}")

    out = source[["trade_date", "open", "high", "low", "close"]].copy()
    out["pre_close"] = source.get("pre_close", out["close"].shift(1))
    out["pre_close"] = out["pre_close"].where(out["pre_close"].gt(0), out["close"].shift(1))
    out["vol"] = source.get("vol", pd.Series(np.nan, index=source.index))
    out["amount"] = source.get("amount", pd.Series(np.nan, index=source.index))
    out["pct_chg"] = source.get(
        "pct_chg",
        (out["close"] / out["pre_close"] - 1.0) * 100.0,
    )
    out["target_next_return"] = out["close"].shift(-1) / out["close"] - 1.0
    out["target_next_direction"] = (out["target_next_return"] > 0).astype(float)
    return out.dropna(subset=["trade_date", "open", "high", "low", "close"])


def _asset_feature_frame(name: str, df: pd.DataFrame, *, lag: int = 0) -> pd.DataFrame:
    if lag < 0:
        raise ValueError("特征滞后期必须大于等于 0。")

    source = _normalize_price_frame(df)
    if "trade_date" not in source.columns or "close" not in source.columns:
        return pd.DataFrame()

    close = source["close"].astype(float)
    out = pd.DataFrame({"trade_date": source["trade_date"]})
    out[f"{name}_ret1"] = close.pct_change()
    out[f"{name}_ret5"] = close / close.shift(5) - 1.0
    out[f"{name}_vol20"] = close.pct_change().rolling(20).std()

    if "open" in source.columns:
        open_ = source["open"].astype(float)
        out[f"{name}_gap"] = open_ / close.shift(1) - 1.0
        out[f"{name}_intraday"] = close / open_ - 1.0
    if "high" in source.columns and "low" in source.columns:
        high = source["high"].astype(float)
        low = source["low"].astype(float)
        out[f"{name}_range"] = (high - low) / close.shift(1)
    if "vol" in source.columns:
        vol = source["vol"].astype(float).replace(0, np.nan)
        out[f"{name}_vol_chg"] = np.log(vol).diff()
    if "pct_chg" in source.columns:
        out[f"{name}_pct_chg"] = source["pct_chg"].astype(float)

    if lag > 0:
        value_cols = [column for column in out.columns if column != "trade_date"]
        out[value_cols] = out[value_cols].shift(lag)
        out = out.rename(columns={column: f"{column}_lag{lag}" for column in value_cols})
    return out


def _existing_hangseng_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    source = _normalize_price_frame(df)
    if "trade_date" not in source.columns:
        return pd.DataFrame()

    raw_dates = source["trade_date"]
    selected = [
        column
        for column in df.columns
        if str(column).startswith("hangseng_")
    ]
    if not selected:
        return pd.DataFrame()

    out = pd.DataFrame({"trade_date": raw_dates})
    for column in selected:
        out[str(column)] = pd.to_numeric(df[column], errors="coerce")
    return out


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    aliases = {
        "trade_date": ("trade_date", "trade_dt", "date", "datetime", "time", "opdate"),
        "open": ("open", "s_dq_open", "s_open", "adj_open", "开盘"),
        "high": ("high", "s_dq_high", "s_high", "adj_high", "最高"),
        "low": ("low", "s_dq_low", "s_low", "adj_low", "最低"),
        "close": ("close", "s_dq_close", "s_close", "adj_close", "收盘"),
        "pre_close": (
            "pre_close",
            "preclose",
            "prev_close",
            "previous_close",
            "s_dq_preclose",
            "昨收",
        ),
        "change": ("change", "chg", "涨跌额"),
        "vol": ("vol", "volume", "s_dq_volume", "s_volume", "成交量"),
        "amount": ("amount", "amt", "turnover", "s_dq_amount", "s_amount", "成交额"),
        "pct_chg": ("pct_chg", "pct_change", "s_dq_pctchange", "change_pct", "涨跌幅"),
        "swing": ("swing", "振幅"),
    }
    columns_by_lower = {str(column).strip().lower(): column for column in df.columns}

    out = pd.DataFrame(index=df.index)
    for canonical, names in aliases.items():
        source = None
        for name in names:
            source = columns_by_lower.get(name.lower())
            if source is not None:
                break
        if source is None:
            continue
        if canonical == "trade_date":
            out[canonical] = _parse_trade_dates(df[source])
        else:
            out[canonical] = pd.to_numeric(df[source], errors="coerce")

    if "trade_date" not in out.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            out["trade_date"] = pd.to_datetime(df.index, errors="coerce")
        else:
            raise ValueError("未找到交易日期列。")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["trade_date"])
    out = out.sort_values("trade_date")
    out = out.drop_duplicates(subset=["trade_date"], keep="last")
    return out.reset_index(drop=True)


def _parse_trade_dates(values: pd.Series) -> pd.Series:
    as_text = values.astype(str).str.strip()
    yyyymmdd = as_text.str.fullmatch(r"\d{8}")
    if len(as_text) and float(yyyymmdd.mean()) > 0.8:
        return pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")
    iso_ymd = as_text.str.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    if len(as_text) and float(iso_ymd.mean()) > 0.8:
        normalized = as_text.str.replace("/", "-", regex=False)
        return pd.to_datetime(normalized, format="%Y-%m-%d", errors="coerce")
    dmy = as_text.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}")
    if len(as_text) and float(dmy.mean()) > 0.8:
        return pd.to_datetime(as_text, dayfirst=True, errors="coerce")
    return pd.to_datetime(values, dayfirst=True, errors="coerce")


def _sort_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"])
    out = out.sort_values("trade_date")
    return out.reset_index(drop=True)


def _main() -> None:
    parser = ChineseArgumentParser(
        description="从 Wind 构建 DRP-FEIM 预测器所需的最小行情数据。"
    )
    parser._optionals.title = "可选参数"
    for action in parser._actions:
        if action.dest == "help":
            action.help = "显示帮助信息并退出。"
    parser.add_argument(
        "--start-date",
        default="20200101",
        help="数据开始日期，格式为 YYYYMMDD。",
    )
    parser.add_argument(
        "--end-date",
        default=pd.Timestamp.today().strftime("%Y%m%d"),
        help="数据结束日期，格式为 YYYYMMDD。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="原始 CSV 和 merged_features.csv 的保存目录。",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="只在内存中生成结果，不写入 CSV 文件。",
    )
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="仅使用已保存的原始 CSV 重建 merged_features.csv，不调用 Wind。",
    )
    parser.add_argument(
        "--prediction-time",
        choices=["after_close", "before_open"],
        default="after_close",
        help="预测时点：after_close 使用当日收盘后可见特征，before_open 额外滞后一日。",
    )
    parser.add_argument(
        "--target-code",
        default=DEFAULT_TARGET_CODE,
        help="目标指数 Wind 代码。",
    )
    parser.add_argument(
        "--hangseng-code",
        default=DEFAULT_HANGSENG_CODE,
        help="恒生指数 Wind 代码。",
    )
    parser.add_argument(
        "--sleep-between-calls-seconds",
        type=float,
        default=0.0,
        help="不同接口调用之间的固定等待秒数。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRY_CONFIG.retries,
        help="接口调用失败后的重试次数。",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=DEFAULT_RETRY_CONFIG.sleep_seconds,
        help="普通失败后的基础等待秒数。",
    )
    parser.add_argument(
        "--retry-backoff-factor",
        type=float,
        default=DEFAULT_RETRY_CONFIG.backoff_factor,
        help="重试等待时间的指数退避倍数。",
    )
    parser.add_argument(
        "--retry-jitter-seconds",
        type=float,
        default=DEFAULT_RETRY_CONFIG.jitter_seconds,
        help="每次重试额外加入的随机抖动秒数。",
    )
    args = parser.parse_args()

    retry_config = RetryConfig(
        retries=args.retries,
        sleep_seconds=args.retry_sleep_seconds,
        backoff_factor=args.retry_backoff_factor,
        jitter_seconds=args.retry_jitter_seconds,
    )

    if args.rebuild_only:
        feature = rebuild_merged_features(
            args.output_dir,
            prediction_time=args.prediction_time,
            save=not args.no_save,
        )
        print(f"合并特征表: {feature.shape}")
        return

    datasets = fetch_all(
        args.start_date,
        args.end_date,
        output_dir=args.output_dir,
        save=not args.no_save,
        prediction_time=args.prediction_time,
        retry_config=retry_config,
        target_code=args.target_code,
        hangseng_code=args.hangseng_code,
        sleep_between_calls_seconds=args.sleep_between_calls_seconds,
    )
    for name, df in datasets.items():
        print(f"{name} 数据: {df.shape}")


if __name__ == "__main__":
    _main()
