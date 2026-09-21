"""Shared statistics for settled predictions and configurable trading-day windows."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


CONCENTRATION_MIN_ROWS = 20
CONCENTRATION_RATE = 0.8
CONCENTRATION_GAP = 0.2
LONG_STREAK_ROWS = 10


def _direction_diagnostics(settled: pd.DataFrame, correct: pd.Series, dates: pd.Series) -> dict[str, Any]:
    count = len(settled)
    predicted_up = settled["预测方向"].eq("上涨")
    actual_up = settled["次日实际涨跌幅"].gt(0)
    up_rate = float(predicted_up.mean()) if count else None
    actual_up_rate = float(actual_up.mean()) if count else None
    gap = up_rate - actual_up_rate if count else None

    # A missing outcome or a known gap in daily forecasts ends a streak.
    breaks = predicted_up.ne(predicted_up.shift()) | settled["_sequence"].diff().ne(1)
    if "预测目标交易日" in settled:
        previous_target = pd.to_numeric(settled["预测目标交易日"], errors="coerce").shift()
        breaks |= previous_target.notna() & previous_target.ne(pd.to_numeric(settled["信号日期"]))
    runs = []
    for _, group in settled.groupby(breaks.cumsum(), sort=False):
        runs.append({
            "direction": str(group["预测方向"].iloc[0]),
            "rows": len(group),
            "start_date": int(dates.loc[group.index].iloc[0]),
            "end_date": int(dates.loc[group.index].iloc[-1]),
            "correct_rows": int(correct.loc[group.index].sum()),
            "accuracy": float(correct.loc[group.index].mean()),
        })

    longest_up = max((run for run in runs if run["direction"] == "上涨"), key=lambda run: run["rows"], default=None)
    longest_down = max((run for run in runs if run["direction"] == "下跌"), key=lambda run: run["rows"], default=None)
    longest = max(runs, key=lambda run: run["rows"], default=None)
    adjacent = settled["_sequence"].diff().eq(1)
    if "预测目标交易日" in settled:
        adjacent &= previous_target.isna() | previous_target.eq(pd.to_numeric(settled["信号日期"]))
    transitions = int(adjacent.sum())
    switches = int((predicted_up.ne(predicted_up.shift()) & adjacent).sum())
    alerts = []
    concentrated_up = count >= CONCENTRATION_MIN_ROWS and up_rate >= CONCENTRATION_RATE and gap >= CONCENTRATION_GAP - 1e-12
    concentrated_down = count >= CONCENTRATION_MIN_ROWS and 1 - up_rate >= CONCENTRATION_RATE and -gap >= CONCENTRATION_GAP - 1e-12
    if concentrated_up or concentrated_down:
        direction = "上涨" if concentrated_up else "下跌"
        predicted_rows = int(predicted_up.sum()) if concentrated_up else int((~predicted_up).sum())
        actual_rows = int(actual_up.sum()) if concentrated_up else int((~actual_up).sum())
        alerts.append({
            "code": "direction_concentration",
            "direction": direction,
            "message": f"近 {count} 条已结算预测中，{predicted_rows} 次预测{direction}，实际{direction} {actual_rows} 次。",
        })
    if longest and longest["rows"] >= LONG_STREAK_ROWS:
        alerts.append({
            "code": "long_direction_streak",
            "direction": longest["direction"],
            "message": f"窗口内最长连续预测{longest['direction']} {longest['rows']} 次，命中 {longest['correct_rows']} 次。",
        })
    return {
        "predicted_up_rate": up_rate,
        "actual_up_rate": actual_up_rate,
        "up_rate_gap": gap,
        "switch_rows": switches,
        "transition_rows": transitions,
        "switch_rate": switches / transitions if transitions else None,
        "longest_up": longest_up,
        "longest_down": longest_down,
        "alerts": alerts,
    }


def statistics(frame: pd.DataFrame, days: int | None = None) -> dict[str, Any]:
    ordered = frame.sort_values("信号日期").reset_index(drop=True).copy()
    ordered["_sequence"] = np.arange(len(ordered))
    settled = ordered.loc[ordered["次日实际涨跌幅"].notna()].copy()
    available = len(settled)
    if days is not None:
        settled = settled.tail(days)
    actual = pd.to_numeric(settled["次日实际涨跌幅"])
    predicted = pd.to_numeric(settled["预测次日涨跌幅"])
    predicted_up = settled["预测方向"].eq("上涨")
    actual_up = actual.gt(0)
    correct = predicted_up.eq(actual_up)

    def rate(mask: pd.Series) -> float | None:
        return float(correct.loc[mask].mean()) if mask.any() else None

    up_recall, down_recall = rate(actual_up), rate(~actual_up)
    count = len(settled)
    accuracy = float(correct.mean()) if count else None
    use_target_dates = "预测目标交易日" in settled and settled["预测目标交易日"].notna().all()
    dates = settled["预测目标交易日"] if use_target_dates else settled["信号日期"]
    baseline_accuracy = float(actual_up.mean()) if count else None
    return {
        "rows": count,
        "available_rows": available,
        "requested_days": days,
        "correct_rows": int(correct.sum()),
        "accuracy": accuracy,
        "balanced_accuracy": (up_recall + down_recall) / 2 if up_recall is not None and down_recall is not None else None,
        "up_rows": int(predicted_up.sum()),
        "down_rows": int((~predicted_up).sum()),
        "up_accuracy": rate(predicted_up),
        "down_accuracy": rate(~predicted_up),
        "actual_up_rows": int(actual_up.sum()),
        "actual_down_rows": int((~actual_up).sum()),
        "up_correct_rows": int((predicted_up & correct).sum()),
        "down_correct_rows": int((~predicted_up & correct).sum()),
        "up_recall": up_recall,
        "down_recall": down_recall,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_lift": accuracy - baseline_accuracy if count else None,
        "mean_confidence": float(pd.to_numeric(settled["置信度"]).mean()) if count else None,
        "mae": float((predicted - actual).abs().mean()) if count else None,
        "start_date": int(dates.iloc[0]) if count else None,
        "end_date": int(dates.iloc[-1]) if count else None,
        "date_basis": "target_trade_date" if use_target_dates else "signal_date",
        "live_rows": int(settled.get("记录来源", pd.Series(index=settled.index, dtype=str)).eq("live").sum()),
        "direction": _direction_diagnostics(settled, correct, dates),
        "records": clean_records(settled.drop(columns="_sequence").assign(hit=correct)),
    }


def clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.to_dict("records")
    for row in records:
        for key, value in row.items():
            if pd.isna(value) or (isinstance(value, (float, np.floating)) and not math.isfinite(value)):
                row[key] = None
    return records


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
