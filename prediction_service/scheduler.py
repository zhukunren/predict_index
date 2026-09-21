"""A small single-process scheduler for the daily refresh worker."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timedelta
import logging
from zoneinfo import ZoneInfo

from .config import Settings
from .service import RefreshManager


class DailyRefreshScheduler:
    def __init__(
        self,
        settings: Settings,
        refresh_manager: RefreshManager,
        *,
        is_trading_day: Callable[[str], bool] | None = None,
        expected_date: Callable[[datetime], str | None] | None = None,
    ) -> None:
        self.settings = settings
        self.refresh_manager = refresh_manager
        self.is_trading_day = is_trading_day or (lambda _date: True)
        self.expected_date = expected_date
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_check: datetime | None = None

    def start(self) -> None:
        if not self.settings.scheduled_refresh_enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="prediction-daily-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        timezone = ZoneInfo("Asia/Shanghai")
        while not self._stop_event.is_set():
            try:
                self.run_once(datetime.now(timezone))
            except Exception:
                logging.getLogger(__name__).exception("Daily refresh scheduling failed")
            self._stop_event.wait(20)

    def run_once(self, now: datetime) -> bool:
        """Retry delayed data and catch up missed sessions using persisted job state."""

        timezone = ZoneInfo("Asia/Shanghai")
        localized = now.astimezone(timezone)
        business_date = localized.strftime("%Y%m%d")
        if self._next_check is not None and localized < self._next_check:
            return False
        expected = self.expected_date(localized) if self.expected_date else None
        due = (localized.hour, localized.minute) >= (
            self.settings.scheduled_refresh_hour,
            self.settings.scheduled_refresh_minute,
        )
        if expected is None and (
            localized.weekday() >= 5
            or not due
            or not self.is_trading_day(business_date)
        ):
            return False
        _, created = self.refresh_manager.submit(
            trigger="scheduled",
            actor=None,
            idempotency_key=expected or business_date,
        )
        self._next_check = localized + timedelta(minutes=1)
        return created
