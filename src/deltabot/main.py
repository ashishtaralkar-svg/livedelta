"""Async live entrypoint with graceful signal handling."""

from __future__ import annotations

import asyncio
import signal

from .config import Settings, load_settings
from .core.revbreak_trader import RevBreakEngine
from .core.trader import TradingEngine
from .enums import NotifyEvent
from .exchange.rest_client import RestClient
from .logging_setup import get_logger, setup_logging
from .notify.telegram import TelegramNotifier
from .scheduler import DailyScheduler

log = get_logger(__name__)


async def run(settings: Settings) -> None:
    rest = RestClient(
        base_url=settings.rest_base_url,
        api_key=settings.api_key.get_secret_value(),
        api_secret=settings.api_secret.get_secret_value(),
    )
    notifier = TelegramNotifier(settings)
    await notifier.start()

    if settings.strategy == "revbreak":
        engine: TradingEngine | RevBreakEngine = RevBreakEngine(settings, rest, notifier)
    else:
        engine = TradingEngine(settings, rest, notifier)
    scheduler = DailyScheduler(settings.daily_summary_hour_utc, engine.daily_summary)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        log.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Windows: add_signal_handler may be unavailable for SIGTERM.
            signal.signal(sig, lambda *_: _request_stop())

    scheduler.start()
    engine_task = asyncio.create_task(engine.start())
    stop_task = asyncio.create_task(stop_event.wait())

    try:
        done, _ = await asyncio.wait(
            {engine_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if engine_task in done and (exc := engine_task.exception()):
            log.error("Engine crashed", extra={"extra": {"error": str(exc)}})
            await notifier.notify(NotifyEvent.API_ERROR, detail=f"engine crashed: {exc}")
    finally:
        log.info("Stopping bot")
        await engine.stop()
        engine_task.cancel()
        scheduler.shutdown()
        await notifier.stop()
        rest.close()


def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    if not settings.api_key.get_secret_value() or not settings.api_secret.get_secret_value():
        raise SystemExit("DELTA_API_KEY and DELTA_API_SECRET must be set for live trading")
    log.info(
        "Booting deltabot",
        extra={"extra": {"symbol": settings.symbol, "testnet": settings.testnet, "region": settings.region}},
    )
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
