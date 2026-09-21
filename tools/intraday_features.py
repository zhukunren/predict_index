"""Same-session index path features with strict bar and daily-data alignment."""

import numpy as np
import pandas as pd

from tools.liquidity_features import date_values


TS_CODE = "000001.SH"
FIELDS = ("ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount")
FEATURE_COLUMNS = (
    "intraday_last_30_return", "intraday_afternoon_return", "intraday_late_minus_early_return",
    "intraday_realized_vol_30", "intraday_downside_variance_share",
    "intraday_last_30_amount_share", "intraday_afternoon_amount_share", "intraday_close_vs_mean_30",
)
PRICE_TOLERANCE = .01
TOTAL_RELATIVE_TOLERANCE = 1e-4
POLICY = {
    "instrument": TS_CODE, "timezone": "Asia/Shanghai", "bar_labels": "interval end",
    "supported_minutes": [1, 5, 15, 30], "optional_opening_observation": "09:30",
    "sessions": ["09:30-11:30", "13:00-15:00"],
    "feature_sampling": "eight 30-minute interval-end closes, independent of input frequency",
    "information_time": "same signal-day close, after 18:00 Shanghai",
    "missing_data": "reject incomplete sessions; no resampling fill or estimated quotes",
    "daily_price_tolerance_points": PRICE_TOLERANCE,
    "daily_total_relative_tolerance": TOTAL_RELATIVE_TOLERANCE,
    "frozen_daily_units": {"vol": "hands (100 shares)", "amount": "thousand yuan"},
    "minute_totals": "per-bar incremental quantities; opening row included if supplied",
    "zero_path_variance": "downside variance share is defined as zero for a flat path",
    "historical_delivery_timestamps_available": False,
}


def local_times(values):
    try:
        times = pd.to_datetime(values, format="mixed", errors="raise")
        if times.dt.tz is not None:
            times = times.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Bar times must have a consistent declared timezone; naive times mean Shanghai.") from exc
    if times.isna().any():
        raise ValueError("Missing intraday timestamps.")
    return times


def expected_bar_times(day, frequency_minutes):
    if frequency_minutes not in (1, 5, 15, 30):
        raise ValueError("Intraday frequency must be 1, 5, 15, or 30 minutes.")
    step = pd.Timedelta(minutes=frequency_minutes)
    day = pd.Timestamp(str(day))
    return pd.date_range(day + pd.Timedelta(hours=9, minutes=30) + step,
                         day + pd.Timedelta(hours=11, minutes=30), freq=step).append(
        pd.date_range(day + pd.Timedelta(hours=13) + step, day + pd.Timedelta(hours=15), freq=step))


def intraday_features(signal_dates, bars, daily, *, frequency_minutes=30,
                      volume_unit="shares", amount_unit="yuan", publication_hour=18):
    if not 18 <= publication_hour <= 23:
        raise ValueError("Intraday research requires completed daily data after 18:00 Shanghai.")
    if volume_unit not in ("shares", "hands") or amount_unit not in ("yuan", "thousand_yuan"):
        raise ValueError("Minute volume and amount units must be explicitly supported.")
    signals = date_values(signal_dates)
    if signals.empty or signals.duplicated().any() or not signals.is_monotonic_increasing:
        raise ValueError("Intraday signal dates must be unique and chronological.")
    if not set(FIELDS) <= set(bars.columns) or not {"trade_date", "open", "high", "low", "close", "vol", "amount"} <= set(daily.columns):
        raise ValueError("Missing intraday or daily reference fields.")
    source = bars.loc[:, FIELDS].copy()
    source["trade_time"] = local_times(source.trade_time)
    source["trade_date"] = source.trade_time.dt.strftime("%Y%m%d").astype(int)
    source = source.loc[source.trade_date.isin(signals)]
    reference = daily.copy()
    strings = reference.trade_date.astype(str).str.replace(r"\.0$", "", regex=True)
    reference["trade_date"] = pd.to_datetime(strings, format="mixed", errors="raise").dt.strftime("%Y%m%d").astype(int)
    reference = reference.loc[reference.trade_date.isin(signals)]
    if (reference.trade_date.duplicated().any() or source.trade_time.duplicated().any()
            or not source.ts_code.eq(TS_CODE).all()):
        raise ValueError("Duplicate timestamps/reference dates or unexpected index code.")
    if not signals.isin(source.trade_date).all() or not signals.isin(reference.trade_date).all():
        raise ValueError("Missing whole intraday or daily reference sessions.")
    numeric = ["open", "high", "low", "close", "vol", "amount"]
    for frame in (source, reference):
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Requested intraday and daily values must be finite.")
        if (not frame[["open", "high", "low", "close"]].gt(0).all().all()
                or frame.vol.lt(0).any() or frame.amount.lt(0).any()
                or frame.high.lt(frame.low).any()
                or frame.open.lt(frame.low - PRICE_TOLERANCE).any() or frame.open.gt(frame.high + PRICE_TOLERANCE).any()
                or frame.close.lt(frame.low - PRICE_TOLERANCE).any() or frame.close.gt(frame.high + PRICE_TOLERANCE).any()):
            raise ValueError("Invalid OHLC range or nonnegative turnover quantities.")
    reference = reference.set_index("trade_date")
    source = source.sort_values("trade_time")
    rows = []
    for signal in signals:
        day = pd.Timestamp(str(signal))
        group = source.loc[source.trade_date.eq(signal)].set_index("trade_time")
        required = expected_bar_times(signal, frequency_minutes)
        opening = day + pd.Timedelta(hours=9, minutes=30)
        if len(required.difference(group.index)) or len(group.index.difference(required.append(pd.DatetimeIndex([opening])))):
            raise ValueError(f"Incomplete or unexpected interval-end bars for {signal}.")
        ref = reference.loc[signal]
        if (abs(group.close.iloc[-1] - ref.close) > PRICE_TOLERANCE
                or abs(group.high.max() - ref.high) > PRICE_TOLERANCE
                or abs(group.low.min() - ref.low) > PRICE_TOLERANCE):
            raise ValueError(f"Intraday prices disagree with frozen daily prices for {signal}.")
        vol = float(group.vol.sum()) / (100 if volume_unit == "shares" else 1)
        amount = float(group.amount.sum()) / (1000 if amount_unit == "yuan" else 1)
        if ref.vol <= 0 or ref.amount <= 0 or not np.isclose(vol, ref.vol, rtol=TOTAL_RELATIVE_TOLERANCE, atol=.01) or not np.isclose(amount, ref.amount, rtol=TOTAL_RELATIVE_TOLERANCE, atol=.001):
            raise ValueError(f"Minute totals/units disagree with frozen daily totals for {signal}.")
        closes = group.loc[expected_bar_times(signal, 30), "close"].to_numpy(dtype=float)
        log_moves = np.diff(np.log(np.r_[float(ref.open), closes]))
        squared = log_moves**2
        last_30 = float(closes[-1]/closes[-2]-1)
        early_30 = float(closes[0]/float(ref.open)-1)
        total_amount = float(group.amount.sum())
        row = {
            "trade_date": int(signal), "intraday_source_date": int(signal),
            "intraday_last_bar": group.index[-1].tz_localize("Asia/Shanghai").isoformat(),
            "intraday_bar_rows": len(group),
            "intraday_last_30_return": last_30,
            "intraday_afternoon_return": float(closes[-1]/closes[3]-1),
            "intraday_late_minus_early_return": last_30-early_30,
            "intraday_realized_vol_30": float(np.sqrt(squared.sum())),
            "intraday_downside_variance_share": float(squared[log_moves < 0].sum()/squared.sum()) if squared.sum() > 0 else 0.0,
            "intraday_last_30_amount_share": float(group.loc[group.index > day + pd.Timedelta(hours=14, minutes=30), "amount"].sum()/total_amount),
            "intraday_afternoon_amount_share": float(group.loc[group.index > day + pd.Timedelta(hours=11, minutes=30), "amount"].sum()/total_amount),
            "intraday_close_vs_mean_30": float(closes[-1]/closes.mean()-1),
        }
        if not np.isfinite([row[name] for name in FEATURE_COLUMNS]).all():
            raise ValueError("Nonfinite derived intraday features.")
        rows.append(row)
    return pd.DataFrame(rows)
