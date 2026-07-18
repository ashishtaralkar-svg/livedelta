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
    NotifyEvent.SKIPPED: "⏭️",
    NotifyEvent.PAPER_ENTRY: "📝",
    NotifyEvent.PAPER_EXIT: "📝",
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
        # Options entry: show the contract traded and the premium paid/received.
        # side="buy" (DCv3 etc.) BUYS the option; default/"sell" SELLS it (all
        # other bots). A rollover (tag=ROLL) re-trades an already-open position
        # at 17:30 -- label it so it isn't mistaken for a brand-new signal.
        if ctx.get("contract"):
            is_buy_side = ctx.get("side") == "buy"
            verb = "BUY" if is_buy_side else "SELL"
            head = f"ROLLOVER {verb}" if ctx.get("tag") == "ROLL" else f"{verb} {ctx.get('direction')}"
            prem_label = "Buy premium" if is_buy_side else "Sell premium"
            msg = (
                f"{emoji} <b>{head}</b> {ctx.get('contract')}\n"
                f"{prem_label}: {_num(ctx.get('premium'))}\n"
                f"BTC: {_num(ctx.get('btc_price'))}"
            )
            if ctx.get("sl_level") is not None:
                msg += f"\nBTC stop: {_num(ctx.get('sl_level'))}  |  Opt TP: {_num(ctx.get('tp_price'))}"
            return msg
        return f"{emoji} <b>ENTRY {ctx.get('direction')}</b> {ctx.get('symbol')} @ {ctx.get('price'):.2f}"
    if event == NotifyEvent.SKIPPED:
        return (
            f"{emoji} <b>ENTRY SKIPPED</b> ({ctx.get('reason')})\n"
            f"BTC: {_num(ctx.get('btc_price'))}\n"
            f"SL distance: {ctx.get('sl_distance')} pts"
        )
    if event == NotifyEvent.PAPER_ENTRY:
        return (
            f"{emoji} <b>PAPER TRADE OPENED</b> — no real order\n"
            f"Why: {ctx.get('reason')}\n"
            f"Would sell {ctx.get('direction')} (notional premium {_num(ctx.get('premium'))})\n"
            f"BTC: {_num(ctx.get('btc_price'))}  |  BTC stop: {_num(ctx.get('sl_level'))} "
            f"({ctx.get('sl_distance')} pts)\n"
            f"Monitoring until SL / EOD"
        )
    if event == NotifyEvent.PAPER_EXIT:
        return (
            f"{emoji} <b>PAPER TRADE CLOSED</b> ({ctx.get('reason')}) — no real order was open\n"
            f"BTC exit: {_num(ctx.get('btc_price'))}\n"
            f"Bot is flat again and watching for the next signal"
        )
    if event == NotifyEvent.REVERSAL:
        return (
            f"{emoji} <b>REVERSAL</b> {ctx.get('symbol')} {ctx.get('from_state')} → "
            f"{ctx.get('direction')} @ {ctx.get('price'):.2f}"
        )
    if event == NotifyEvent.EXIT:
        # Options exit: show contract, entry/exit premiums and per-trade PnL.
        if ctx.get("contract"):
            pnl = ctx.get("pnl")
            pnl_mark = "🟢" if isinstance(pnl, (int, float)) and pnl >= 0 else "🔴"
            open_verb, close_verb = ("Buy", "Sell") if ctx.get("side") == "buy" else ("Sell", "Buy")
            return (
                f"{emoji} <b>EXIT</b> ({ctx.get('reason', '')}) {ctx.get('contract')}\n"
                f"{open_verb}: {_num(ctx.get('entry_premium'))} → {close_verb}: {_num(ctx.get('exit_premium'))}\n"
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
        self._label_prefix = f"[{settings.bot_label}] " if settings.bot_label else ""
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
                        "text": self._label_prefix + _format(event, ctx),
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — never let Telegram crash the bot
                log.warning("Telegram send failed", extra={"extra": {"error": str(exc)}})
            finally:
                self._queue.task_done()
