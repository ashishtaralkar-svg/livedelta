"""Async, non-blocking Telegram notifier.

Messages are enqueued and delivered by a background worker so a slow or failing
Telegram API never blocks or crashes the trading loop. If Telegram is not
configured the notifier degrades to logging only.
"""

from __future__ import annotations

import asyncio

import httpx

from ..config import Settings
from ..enums import NotifyEvent
from ..logging_setup import get_logger

log = get_logger(__name__)

_EMOJI = {
    NotifyEvent.ENTRY_LONG: "🟢",
    NotifyEvent.ENTRY_SHORT: "🔴",
    NotifyEvent.EXIT: "⚪",
    NotifyEvent.REVERSAL: "🔁",
    NotifyEvent.API_ERROR: "⚠️",
    NotifyEvent.RESTART: "🚀",
    NotifyEvent.DAILY_PNL: "📊",
}


def _num(x, digits: int = 1) -> str:
    """None-safe number format for message fields."""
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "n/a"


def _format(event: NotifyEvent, ctx: dict) -> str:
    emoji = _EMOJI.get(event, "ℹ️")
    if event in (NotifyEvent.ENTRY_LONG, NotifyEvent.ENTRY_SHORT):
        # Options entry: show the contract sold and the premium received.
        if ctx.get("contract"):
            msg = (
                f"{emoji} <b>SELL {ctx.get('direction')}</b> {ctx.get('contract')}\n"
                f"Sell premium: {_num(ctx.get('premium'))}\n"
                f"BTC: {_num(ctx.get('btc_price'))}"
            )
            if ctx.get("sl_level") is not None:
                msg += f"\nBTC stop: {_num(ctx.get('sl_level'))}  |  Opt TP: {_num(ctx.get('tp_price'))}"
            return msg
        return f"{emoji} <b>ENTRY {ctx.get('direction')}</b> {ctx.get('symbol')} @ {ctx.get('price'):.2f}"
    if event == NotifyEvent.REVERSAL:
        return (
            f"{emoji} <b>REVERSAL</b> {ctx.get('symbol')} {ctx.get('from_state')} → "
            f"{ctx.get('direction')} @ {ctx.get('price'):.2f}"
        )
    if event == NotifyEvent.EXIT:
        # Options exit: show contract, buy/sell premiums and per-trade PnL.
        if ctx.get("contract"):
            pnl = ctx.get("pnl")
            pnl_mark = "🟢" if isinstance(pnl, (int, float)) and pnl >= 0 else "🔴"
            return (
                f"{emoji} <b>EXIT</b> ({ctx.get('reason', '')}) {ctx.get('contract')}\n"
                f"Sell: {_num(ctx.get('entry_premium'))} → Buy: {_num(ctx.get('exit_premium'))}\n"
                f"{pnl_mark} PnL: {_num(pnl, 2)} USD"
            )
        return f"{emoji} <b>EXIT</b> ({ctx.get('reason', '')}) size={ctx.get('size')}"
    if event == NotifyEvent.API_ERROR:
        return f"{emoji} <b>API ERROR</b>\n{ctx.get('detail')}"
    if event == NotifyEvent.RESTART:
        return f"{emoji} <b>Bot started</b> ({ctx.get('mode')})"
    if event == NotifyEvent.DAILY_PNL:
        return (
            f"{emoji} <b>Daily P&amp;L</b>\n"
            f"Trades: {ctx.get('trades')} (W:{ctx.get('wins')} / L:{ctx.get('losses')})\n"
            f"PnL: {ctx.get('pnl', 0):.4f}\nCumulative: {ctx.get('cumulative_pnl', 0):.4f}"
        )
    return f"{emoji} {event.value}: {ctx}"


class TelegramNotifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.telegram_enabled
        self._token = settings.telegram_token.get_secret_value() if settings.telegram_token else ""
        self._chat_id = settings.telegram_chat_id or ""
        self._queue: asyncio.Queue[tuple[NotifyEvent, dict]] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if not self.enabled:
            log.info("Telegram disabled (no token/chat id) — notifications logged only")
            return
        self._client = httpx.AsyncClient(timeout=10.0)
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        # Drain remaining messages, then shut down.
        if self._worker:
            await self._queue.join()
            self._worker.cancel()
        if self._client:
            await self._client.aclose()

    async def notify(self, event: NotifyEvent, **ctx) -> None:
        log.info("notify", extra={"extra": {"event": event.value, **ctx}})
        if not self.enabled:
            return
        await self._queue.put((event, ctx))

    async def _run(self) -> None:
        assert self._client is not None
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        while True:
            event, ctx = await self._queue.get()
            try:
                await self._client.post(
                    url,
                    json={
                        "chat_id": self._chat_id,
                        "text": _format(event, ctx),
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — never let Telegram crash the bot
                log.warning("Telegram send failed", extra={"extra": {"error": str(exc)}})
            finally:
                self._queue.task_done()
