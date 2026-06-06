"""Daily P&L summary scheduler."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .logging_setup import get_logger

log = get_logger(__name__)


class DailyScheduler:
    def __init__(self, hour_utc: int, callback: Callable[[], Awaitable[None]]) -> None:
        self.hour_utc = hour_utc
        self.callback = callback
        self._scheduler = AsyncIOScheduler(timezone="UTC")

    def start(self) -> None:
        self._scheduler.add_job(
            self.callback,
            CronTrigger(hour=self.hour_utc, minute=0, timezone="UTC"),
            id="daily_pnl_summary",
            replace_existing=True,
        )
        self._scheduler.start()
        log.info("Daily summary scheduled", extra={"extra": {"hour_utc": self.hour_utc}})

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
