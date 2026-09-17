"""A small single-process scheduler for the daily refresh worker."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime
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
    ) -> None:
        self.settings = settings
        self.refresh_manager = refresh_manager
        self.is_trading_day = is_trading_day or (lambda _date: True)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_submission_date: str | None = None

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
            self.run_once(datetime.now(timezone))
            self._stop_event.wait(20)

    def run_once(self, now: datetime) -> bool:
        """Submit at most one eligible scheduled job for a Shanghai business date."""

        timezone = ZoneInfo("Asia/Shanghai")
        localized = now.astimezone(timezone)
        business_date = localized.strftime("%Y%m%d")
        due = (localized.hour, localized.minute) >= (
            self.settings.scheduled_refresh_hour,
            self.settings.scheduled_refresh_minute,
        )
        if (
            localized.weekday() >= 5
            or not due
            or self._last_submission_date == business_date
            or not self.is_trading_day(business_date)
        ):
            return False
        _, created = self.refresh_manager.submit(
            trigger="scheduled",
            actor=None,
            idempotency_key=business_date,
        )
        # An existing terminal job is also enough to prevent a cold-start loop.
        self._last_submission_date = business_date
        return created
