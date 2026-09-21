"""Persisted SSE calendar. Request handlers never depend on a network lookup."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select

from .models import TradingDay, utcnow


SHANGHAI = ZoneInfo("Asia/Shanghai")


class MarketCalendar:
    def __init__(self, database) -> None:
        self.database = database

    def store(self, frame: pd.DataFrame) -> None:
        if frame.empty or not {"cal_date", "is_open"}.issubset(frame.columns):
            raise ValueError("交易日历返回为空或缺少必要字段。")
        dates = pd.to_datetime(frame["cal_date"].astype(str), format="%Y%m%d", errors="raise")
        flags = pd.to_numeric(frame["is_open"], errors="raise")
        if not flags.isin([0, 1]).all() or dates.duplicated().any():
            raise ValueError("交易日历包含无效状态或重复日期。")
        with self.database.session() as session:
            for date, is_open in zip(dates.dt.strftime("%Y%m%d"), flags, strict=True):
                session.merge(TradingDay(date=date, is_open=bool(is_open), fetched_at=utcnow()))

    def sessions(self, start: str, end: str) -> list[str] | None:
        with self.database.session() as session:
            rows = session.scalars(select(TradingDay).where(TradingDay.date.between(start, end)).order_by(TradingDay.date)).all()
        expected = pd.date_range(start, end).strftime("%Y%m%d").tolist()
        if [row.date for row in rows] != expected:
            return None
        return [row.date for row in rows if row.is_open]

    def next_session(self, day: str) -> str | None:
        start = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        with self.database.session() as session:
            next_day = session.scalar(select(TradingDay.date).where(TradingDay.date >= start, TradingDay.is_open.is_(True)).order_by(TradingDay.date).limit(1))
        if next_day and self.sessions(start, next_day) is not None:
            return next_day
        return None

    def expected_as_of(self, now: datetime, hour: int, minute: int) -> str | None:
        local = now.astimezone(SHANGHAI)
        day = local.date() if (local.hour, local.minute) >= (hour, minute) else local.date() - timedelta(days=1)
        dates = self.sessions((day - timedelta(days=30)).strftime("%Y%m%d"), day.strftime("%Y%m%d"))
        return dates[-1] if dates else None
