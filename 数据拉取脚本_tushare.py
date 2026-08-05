"""当前 DRP-FEIM 预测器使用的 Tushare 数据构建脚本。

默认预测器只使用上证指数 OHLCV 派生特征，以及存在时的恒生指数滞后特征。
本脚本只拉取并重建这些最小输入：

* Tushare ``index_daily`` 的 000001.SH -> ``market_data/000001_sh.csv``
* Tushare ``index_global`` 的 HSI -> ``market_data/hangseng.csv``
* 合并后的特征表 -> ``market_data/merged_features.csv``

不使用 AkShare 替代源。
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import tushare as ts


DEFAULT_OUTPUT_DIR = Path("market_data")
PredictionTime = Literal["after_close", "before_open"]


class ChineseArgumentParser(argparse.ArgumentParser):
    """将 argparse 自动生成的 usage 前缀改为中文。"""

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法:", 1)

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法:", 1)


@dataclass(frozen=True)
class DataRequest:
    name: str
    api: str
    ts_code: str


@dataclass(frozen=True)
class RetryConfig:
    retries: int = 3
    sleep_seconds: float = 1.0
    backoff_factor: float = 2.0
    jitter_seconds: float = 0.25
    rate_limit_sleep_seconds: float = 65.0


DEFAULT_RETRY_CONFIG = RetryConfig()
DEFAULT_TUSHARE_MIN_INTERVAL_SECONDS = {
    "index_global": 6.2,
}
RATE_LIMIT_ERROR_KEYWORDS = (
    "频率超限",
    "每分钟",
    "rate limit",
    "too many requests",
    "too frequent",
)

TARGET_REQUEST = DataRequest("000001_sh", "index_daily", "000001.SH")
HANGSENG_REQUEST = DataRequest("hangseng", "index_global", "HSI")
ALL_REQUESTS = (TARGET_REQUEST, HANGSENG_REQUEST)
HANGSENG_FEATURES_KEY = "_existing_hangseng_features"

TARGET_CANDIDATE_FILES = (
    "000001_sh.csv",
    "sh000001.csv",
    "wind.csv",
)


class ApiRateLimiter:
    """限制已知低频接口的连续调用速度。"""

    def __init__(self, min_interval_by_api: dict[str, float] | None = None) -> None:
        self.min_interval_by_api = min_interval_by_api or {}
        self._last_call_at: dict[str, float] = {}

    def wait(self, api: str) -> None:
        min_interval = self.min_interval_by_api.get(api, 0.0)
        if min_interval <= 0:
            return
        now = time.monotonic()
        last_call_at = self._last_call_at.get(api)
        if last_call_at is not None:
            wait_seconds = min_interval - (now - last_call_at)
            if wait_seconds > 0:
                print(f"[等待] {api} 限速：休眠 {wait_seconds:.1f}s")
                time.sleep(wait_seconds)
        self._last_call_at[api] = time.monotonic()


def get_pro(token: str | None = None):
    """创建 Tushare Pro 客户端。"""

    token = token or _resolve_tushare_token()
    if not token:
        raise RuntimeError(
            "缺少 Tushare token。请设置 TUSHARE_TOKEN、TUSHARE_PRO_TOKEN、"
            "tushare_token，或通过 --token 传入。"
        )
    return ts.pro_api(token)


def _resolve_tushare_token() -> str | None:
    for name in (
        "TUSHARE_TOKEN",
        "TUSHARE_PRO_TOKEN",
        "tushare_token",
        "tushre_token",
        "TUSHRE_TOKEN",
        "TUSAHRE_TOKEN",
    ):
        value = os.getenv(name)
        if value:
            return value
    return None


def normalize_date(value: str | int | pd.Timestamp) -> str:
    text = str(value).strip()
    if text.isdigit() and len(text) == 8:
        return text
    return pd.Timestamp(value).strftime("%Y%m%d")


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
                if is_rate_limit_error(exc):
                    sleep_seconds = retry_config.rate_limit_sleep_seconds
                else:
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


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(keyword.lower() in text for keyword in RATE_LIMIT_ERROR_KEYWORDS)


def fetch_tushare_request(
    pro,
    request: DataRequest,
    start_date: str | int,
    end_date: str | int,
    *,
    retry_config: RetryConfig = DEFAULT_RETRY_CONFIG,
    rate_limiter: ApiRateLimiter | None = None,
) -> pd.DataFrame:
    start = normalize_date(start_date)
    end = normalize_date(end_date)

    def call() -> pd.DataFrame:
        if rate_limiter is not None:
            rate_limiter.wait(request.api)
        api_func = getattr(pro, request.api)
        return api_func(
            ts_code=request.ts_code,
            start_date=start,
            end_date=end,
        )

    df = safe_call(
        call,
        name=f"{request.api}:{request.ts_code}",
        retry_config=retry_config,
    )
    if df.empty:
        return df
    return _sort_by_trade_date(df)


def fetch_all(
    start_date: str | int,
    end_date: str | int,
    *,
    token: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save: bool = True,
    prediction_time: PredictionTime = "after_close",
    feature_lags: dict[str, int] | None = None,
    retry_config: RetryConfig = DEFAULT_RETRY_CONFIG,
    tushare_min_interval_by_api: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """拉取最小 Tushare 数据集并生成 ``merged_features.csv``。"""

    pro = get_pro(token)
    out_dir = ensure_output_dir(output_dir)
    rate_limiter = ApiRateLimiter(
        tushare_min_interval_by_api or DEFAULT_TUSHARE_MIN_INTERVAL_SECONDS
    )
    datasets: dict[str, pd.DataFrame] = {}

    for request in ALL_REQUESTS:
        df = fetch_tushare_request(
            pro,
            request,
            start_date,
            end_date,
            retry_config=retry_config,
            rate_limiter=rate_limiter,
        )
        datasets[request.name] = df
        if save and not df.empty:
            df.to_csv(out_dir / f"{request.name}.csv", index=False, encoding="utf-8-sig")

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
    """读取已保存的最小行情 CSV，不调用外部接口。"""

    out_dir = Path(output_dir)
    datasets: dict[str, pd.DataFrame] = {}
    merged_path = out_dir / "merged_features.csv"
    merged_frame = pd.read_csv(merged_path) if merged_path.exists() else None

    for filename in TARGET_CANDIDATE_FILES:
        path = out_dir / filename
        if path.exists():
            datasets[TARGET_REQUEST.name] = pd.read_csv(path)
            break
    if TARGET_REQUEST.name not in datasets and merged_frame is not None:
        datasets[TARGET_REQUEST.name] = merged_frame

    hangseng_path = out_dir / f"{HANGSENG_REQUEST.name}.csv"
    if hangseng_path.exists():
        datasets[HANGSENG_REQUEST.name] = pd.read_csv(hangseng_path)
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
    target_name: str = TARGET_REQUEST.name,
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
    target_name: str = TARGET_REQUEST.name,
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

    hangseng = datasets.get(HANGSENG_REQUEST.name)
    if hangseng is not None and not hangseng.empty:
        lag = (
            feature_lags[HANGSENG_REQUEST.name]
            if feature_lags is not None and HANGSENG_REQUEST.name in feature_lags
            else default_feature_lag(HANGSENG_REQUEST.name, prediction_time)
        )
        part = _asset_feature_frame(HANGSENG_REQUEST.name, hangseng, lag=lag)
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
    market_lag = 1 if name == HANGSENG_REQUEST.name else 0
    timing_lag = 1 if prediction_time == "before_open" else 0
    return market_lag + timing_lag


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
    out["target_next_direction"] = (
        (out["target_next_return"] > 0)
        .astype(float)
        .where(out["target_next_return"].notna())
    )
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
    selected = [
        column
        for column in df.columns
        if str(column).startswith("hangseng_")
    ]
    if not selected:
        return pd.DataFrame()

    columns_by_lower = {str(column).strip().lower(): column for column in df.columns}
    date_column = next(
        (
            columns_by_lower[name]
            for name in ("trade_date", "trade_dt", "date", "datetime", "time", "opdate")
            if name in columns_by_lower
        ),
        None,
    )
    out = pd.DataFrame(index=df.index)
    if date_column is not None:
        out["trade_date"] = _parse_trade_dates(df[date_column])
    elif isinstance(df.index, pd.DatetimeIndex):
        out["trade_date"] = pd.to_datetime(df.index, errors="coerce")
    else:
        return pd.DataFrame()
    out["_source_order"] = np.arange(len(out))
    for column in selected:
        out[str(column)] = pd.to_numeric(df[column], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["trade_date"])
    out = out.sort_values(["trade_date", "_source_order"])
    out = out.drop_duplicates(subset=["trade_date"], keep="last")
    return out.drop(columns="_source_order").reset_index(drop=True)


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
        ),
        "vol": ("vol", "volume", "s_dq_volume", "s_volume", "成交量"),
        "amount": ("amount", "turnover", "s_dq_amount", "s_amount", "成交额"),
        "pct_chg": ("pct_chg", "pct_change", "s_dq_pctchange", "change_pct", "涨跌幅"),
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


def _sort_by_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    if "trade_date" not in df.columns:
        return df.reset_index(drop=True)
    out = df.copy()
    out["trade_date"] = _parse_trade_dates(out["trade_date"])
    return out.sort_values("trade_date").reset_index(drop=True)


def _main() -> None:
    parser = ChineseArgumentParser(
        description="从 Tushare 构建 DRP-FEIM 预测器所需的最小行情数据。"
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
        "--token",
        default=None,
        help="Tushare token；未传入时从环境变量读取。",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="只在内存中生成结果，不写入 CSV 文件。",
    )
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="仅使用已保存的原始 CSV 重建 merged_features.csv，不调用 Tushare。",
    )
    parser.add_argument(
        "--prediction-time",
        choices=["after_close", "before_open"],
        default="after_close",
        help="预测时点：after_close 使用当日收盘后可见特征，before_open 额外滞后一日。",
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
    parser.add_argument(
        "--rate-limit-sleep-seconds",
        type=float,
        default=DEFAULT_RETRY_CONFIG.rate_limit_sleep_seconds,
        help="遇到 Tushare 限流时的等待秒数。",
    )
    parser.add_argument(
        "--index-global-min-interval",
        type=float,
        default=DEFAULT_TUSHARE_MIN_INTERVAL_SECONDS["index_global"],
        help="index_global 接口两次调用之间的最小间隔秒数。",
    )
    args = parser.parse_args()

    retry_config = RetryConfig(
        retries=args.retries,
        sleep_seconds=args.retry_sleep_seconds,
        backoff_factor=args.retry_backoff_factor,
        jitter_seconds=args.retry_jitter_seconds,
        rate_limit_sleep_seconds=args.rate_limit_sleep_seconds,
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
        token=args.token,
        output_dir=args.output_dir,
        save=not args.no_save,
        prediction_time=args.prediction_time,
        retry_config=retry_config,
        tushare_min_interval_by_api={
            **DEFAULT_TUSHARE_MIN_INTERVAL_SECONDS,
            "index_global": args.index_global_min_interval,
        },
    )
    for name, df in datasets.items():
        print(f"{name} 数据: {df.shape}")


if __name__ == "__main__":
    _main()
