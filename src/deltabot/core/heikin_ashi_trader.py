"""HeikinAshi live trading engine.

Runs HeikinAshiStrategy on 1-minute BTC candles; executes option trades via
OptionsExecutor in SELL mode (BUY signal -> sell PUT, SELL signal -> sell
CALL). No profit target -- exits are exactly the fixed pattern SL, the ASAP
EMA-trail exit, or the 17:25 IST EOD square-off.

ASAP (real-price) execution: entry trigger, fixed SL, and the EMA trail are
all checked on every forming-candle update (i.e. every WS tick for the
currently-forming 1-minute bar), firing the instant real price crosses the
relevant level -- not waiting for the bar to close. The closed-candle path
(``HeikinAshiStrategy.update``) is a fallback that normally no-ops because the
intracandle path already flattened/opened the strategy's in-memory state.

Designed as an independent Docker container, normally on its OWN Delta
sub-account (separate capital from any other bot) so position ownership never
needs to be arbitrated between strategies. Still tracks a state file
(``DELTA_STATE_FILE``) so a restart reconciles cleanly.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from ..config import Settings
from ..enums import NotifyEvent, OptionType, PositionState, SignalDir
from ..exchange.rest_client import RestClient
from ..exchange.ws_manager import WebSocketManager
from ..logging_setup import get_logger
from ..models import Candle
from ..strategy.heikin_ashi import HeikinAshiStrategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 60  # HeikinAshi always runs on 1-minute BTC candles

log = get_logger(__name__)


class HeikinAshiEngine:
    """Live trading engine wired to HeikinAshiStrategy."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = HeikinAshiStrategy(
            st_period=settings.atr_period,
            st_multiplier=settings.st_multiplier,
            ema_length=settings.ema_length,
            ema200_length=settings.ema200_length,
            day_tz=settings.day_tz,
            day_start_hour=settings.day_start_hour,
            day_start_minute=settings.day_start_minute,
            square_off_hour=settings.square_off_hour,
            square_off_minute=settings.square_off_minute,
        )
        self.executor = OptionsExecutor(rest, settings)
        self.aggregator = CandleAggregator(
            on_closed=self._on_closed_candle, on_forming=self._on_forming_candle
        )
        self.ws: WebSocketManager | None = None
        self._last_closed_start: int | None = None
        self._tasks: set[asyncio.Task] = set()
        self._sq_off_task: asyncio.Task | None = None
        self._sq_off_date: date | None = None

        self._entry_premium: float | None = None
        # Re-entrancy guard so intracandle + closed-candle paths can't double-open.
        self._entry_in_progress = False
        # Shared guard so the intracandle exit and the closed-candle fallback
        # can never double-close the same position.
        self._closing = False

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        mode = "TESTNET" if self.settings.testnet else "LIVE"
        await self.notifier.notify(NotifyEvent.RESTART, mode=mode)

        await self._warmup()
        await self._sync_options_to_exchange()

        self.ws = WebSocketManager(
            ws_url=self.settings.ws_url,
            symbol=self.settings.symbol,
            resolution="1m",
            api_key=self.settings.api_key.get_secret_value() or None,
            api_secret=self.settings.api_secret.get_secret_value() or None,
            on_candle=self.aggregator.ingest,
            on_reconnect=self._on_reconnect,
            heartbeat_timeout_s=self.settings.heartbeat_timeout_s,
        )
        self._sq_off_task = asyncio.create_task(self._square_off_scheduler())
        log.info("HeikinAshiEngine: starting live")
        await self.ws.run()

    async def stop(self) -> None:
        if self.ws:
            self.ws.stop()
        if self._sq_off_task is not None:
            self._sq_off_task.cancel()
        if self.settings.close_on_shutdown and self.executor.has_open_position:
            try:
                await self.executor.close_option()
                if self.settings.state_file:
                    position_state.clear(self.settings.state_file)
                await self.notifier.notify(NotifyEvent.EXIT, reason="shutdown",
                                           size=self.settings.option_contracts)
                log.info("HeikinAshi: closed option on shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("HeikinAshi: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass  # no ledger in this engine; skip daily PnL summary

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        bars_needed = max(
            self.settings.ema200_length + self.settings.atr_period + 50,
            self.settings.warmup_days * 86400 // _BAR_SECONDS,
        )
        start = last_closed_end - bars_needed * _BAR_SECONDS
        candles = await self._fetch_history_paged(start, last_closed_end)
        current_bar = (now // _BAR_SECONDS) * _BAR_SECONDS
        closed = [c for c in candles if c.start_time < current_bar]
        for c in closed:
            self.strategy.update(c)
        if closed:
            self._last_closed_start = closed[-1].start_time
        log.info("HeikinAshi warmup done", extra={"extra": {"candles": len(closed)}})

    async def _fetch_history_paged(self, start: int, end: int) -> list[Candle]:
        page_span = 2000 * _BAR_SECONDS
        out: list[Candle] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + page_span, end)
            page = await asyncio.to_thread(
                self.rest.get_candles, self.settings.symbol, "1m", cursor, chunk_end
            )
            out.extend(page)
            cursor = chunk_end
        seen: set[int] = set()
        unique: list[Candle] = []
        for c in sorted(out, key=lambda c: c.start_time):
            if c.start_time not in seen:
                seen.add(c.start_time)
                unique.append(c)
        return unique

    async def _on_reconnect(self) -> None:
        await self._sync_options_to_exchange()
        await self._maybe_reseed_after_gap()

    async def _maybe_reseed_after_gap(self) -> None:
        if self._last_closed_start is None:
            return
        now = int(time.time())
        current_bar = (now // _BAR_SECONDS) * _BAR_SECONDS
        if current_bar - self._last_closed_start > _BAR_SECONDS:
            log.warning("HeikinAshi: candle gap detected — re-seeding")
            await self._warmup()

    # ------------------------------------------------------------------ #
    def _on_closed_candle(self, candle: Candle) -> None:
        task = asyncio.create_task(self._handle_closed_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_closed_candle(self, candle: Candle) -> None:
        if self._last_closed_start is not None:
            gap = candle.start_time - self._last_closed_start
            if gap > _BAR_SECONDS:
                log.warning("HeikinAshi: candle gap — re-seeding")
                await self._warmup()
        self._last_closed_start = candle.start_time

        if self._entries_blocked() and not self.executor.has_open_position:
            # After EOD square-off: keep feeding the strategy but suppress new entries.
            self.strategy.update(candle)
            return

        # Closed-bar fallback: the intracandle path normally fires first for
        # both exits and entries (see _handle_forming_candle).
        dec = self.strategy.update(candle)
        if dec is None:
            return

        if dec.has_exit and self.executor.has_open_position:
            await self._close(dec.exit_reason or "SL")

        if dec.has_entry and not self.executor.has_open_position and not self._entries_blocked():
            signal_dir = SignalDir.LONG.value if dec.buy_signal else SignalDir.SHORT.value
            await self._open_entry(signal_dir, dec.sl_level, candle.close)

    # ------------------------------------------------------------------ #
    def _on_forming_candle(self, candle: Candle) -> None:
        task = asyncio.create_task(self._handle_forming_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_forming_candle(self, candle: Candle) -> None:
        """ASAP: fires the entry trigger, fixed SL, and EMA trail the instant
        REAL price crosses the relevant level, instead of waiting for the
        1-minute candle to close."""
        if not self.strategy.ready:
            return

        if self.executor.has_open_position:
            for price in (candle.low, candle.high):
                long_sl, short_sl, sl_level = self.strategy.check_intracandle_sl(price)
                if long_sl or short_sl:
                    direction = SignalDir.LONG.value if long_sl else SignalDir.SHORT.value
                    log.info("HeikinAshi: intracandle SL touched",
                             extra={"extra": {"price": price, "sl": sl_level}})
                    await self._close("SL")
                    self.strategy.notify_exit(direction, "SL")
                    return
                long_trail, short_trail, trail_level = self.strategy.check_intracandle_trail(price)
                if long_trail or short_trail:
                    direction = SignalDir.LONG.value if long_trail else SignalDir.SHORT.value
                    log.info("HeikinAshi: intracandle TRAIL crossed",
                             extra={"extra": {"price": price, "trail_level": trail_level}})
                    await self._close("TRAIL")
                    self.strategy.notify_exit(direction, "TRAIL")
                    return
            return

        # Flat -> watch the armed setup for an ASAP breakout entry.
        if self._entry_in_progress or not self.strategy.has_pending or self._entries_blocked():
            return
        confirmed, invalidated, entry_price = self.strategy.apply_intracandle_pending(candle)
        if invalidated:
            log.info("HeikinAshi: setup invalidated intracandle (SL crossed before trigger)")
            return
        if confirmed:
            signal_dir = (SignalDir.LONG.value
                          if self.strategy.position_state == PositionState.LONG
                          else SignalDir.SHORT.value)
            log.info("HeikinAshi: intracandle breakout — entering ASAP",
                     extra={"extra": {"trigger": entry_price}})
            await self._open_entry(signal_dir, self.strategy.sl_level, entry_price)

    # ------------------------------------------------------------------ #
    async def _open_entry(self, signal_dir: int, sl_level: float | None, btc_price: float) -> None:
        """Open a short option for a new HeikinAshi signal. Callable from both
        the intracandle (ASAP) and closed-candle (fallback) paths; guarded so
        the two can never double-open. ``signal_dir`` is a SignalDir value;
        bullish sells a PUT, bearish sells a CALL."""
        if self._entry_in_progress or self.executor.has_open_position or self._entry_premium is not None:
            return
        self._entry_in_progress = True
        try:
            is_buy = signal_dir == SignalDir.LONG.value
            try:
                fill, symbol = await self.executor.open_option_by_premium(
                    signal_dir, self.settings.target_premium
                )
            except OptionsMarginError as exc:
                log.error("HeikinAshi: margin error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Margin: {exc}")
                self.strategy.notify_exit(signal_dir, "SL")  # unblock + flatten strategy
                return
            except Exception as exc:  # noqa: BLE001
                log.error("HeikinAshi: open_option_by_premium failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                self.strategy.notify_exit(signal_dir, "SL")
                return

            if fill is None:
                log.warning("HeikinAshi: no option fill — skipping entry")
                self.strategy.notify_exit(signal_dir, "SL")
                return

            self._entry_premium = fill

            if self.settings.state_file:
                position_state.save(
                    self.settings.state_file,
                    symbol=symbol or "",
                    product_id=self.executor.tracked_product_id,
                    size=self.settings.option_contracts,
                    entry_premium=fill,
                    direction=signal_dir,
                )

            direction = "PUT" if is_buy else "CALL"
            log.info("HeikinAshi entry", extra={"extra": {
                "direction": direction, "symbol": symbol, "fill": fill, "sl_level": sl_level,
            }})
            event = NotifyEvent.ENTRY_LONG if is_buy else NotifyEvent.ENTRY_SHORT
            await self.notifier.notify(
                event,
                direction=direction,
                contract=symbol or "?",
                premium=fill,
                btc_price=btc_price,
                sl_level=sl_level,
            )
        finally:
            self._entry_in_progress = False

    async def _close(self, reason: str) -> None:
        if self._closing or not self.executor.has_open_position:
            return  # already closing (e.g. intracandle path beat the closed-bar fallback)
        self._closing = True
        try:
            await self._do_close(reason)
        finally:
            self._closing = False

    async def _do_close(self, reason: str) -> None:
        contract = self.executor.tracked_symbol
        try:
            fill = await self.executor.close_option()
        except Exception as exc:  # noqa: BLE001
            log.error("HeikinAshi: close failed", extra={"extra": {"error": str(exc)}})
            await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close: {exc}")
            return

        if self.settings.state_file:
            position_state.clear(self.settings.state_file)

        entry_prem = self._entry_premium
        lots = self.settings.option_contracts
        gross = ((entry_prem - fill) * lots * 0.001
                 if (entry_prem is not None and fill is not None) else 0.0)
        self._entry_premium = None

        log.info("HeikinAshi exit", extra={"extra": {"reason": reason, "contract": contract}})
        await self.notifier.notify(
            NotifyEvent.EXIT,
            reason=reason,
            contract=contract or "?",
            entry_premium=entry_prem,
            exit_premium=fill,
            pnl=round(gross, 2),
            size=lots,
        )

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile the open option position with the exchange on start/reconnect.

        Normally runs on its OWN sub-account (no other bot shares it), so any
        open short option found belongs to this engine. Ownership is decided
        from THREE signals so no single failure (state-file write, flaky
        fetch) can cause a double-open:
          A. the exchange currently reports an open short  -> adopt it
          B. we are already tracking a position in memory  -> keep it (a WS
             reconnect must never drop the live position we opened)
          C. the state file names a position               -> re-adopt from it
        Only when NONE of these hold do we treat ourselves as genuinely FLAT.
        """
        state_file = self.settings.state_file
        saved = position_state.load(state_file) if state_file else None
        owned_symbol = saved.get("symbol") if saved else None
        believe_owned = owned_symbol is not None or self.executor.has_open_position

        shorts: list[dict] = []
        for attempt in range(3):
            try:
                positions = await asyncio.to_thread(
                    self.rest.get_option_positions, self.executor.underlying
                )
            except Exception as exc:  # noqa: BLE001
                log.error("HeikinAshi reconcile: fetch failed",
                          extra={"extra": {"error": str(exc), "attempt": attempt}})
                positions = []
            shorts = [p for p in positions if p["size"] < 0]
            if shorts or not believe_owned:
                break
            log.warning("HeikinAshi reconcile: expected a position but fetch is empty — retrying",
                        extra={"extra": {"owned": owned_symbol,
                                         "tracked": self.executor.tracked_symbol, "attempt": attempt}})
            await asyncio.sleep(1.5)

        if shorts:
            match = next((p for p in shorts if p.get("symbol") == owned_symbol), shorts[0])
            if saved and match.get("symbol") == owned_symbol:
                self._entry_premium = saved.get("entry_premium")
            opt_type = OptionType.CALL if match["symbol"].startswith("C-") else OptionType.PUT
            self.executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
            log.info("HeikinAshi reconcile: adopted open short",
                     extra={"extra": {"symbol": match["symbol"],
                                      "matched_state_file": match.get("symbol") == owned_symbol}})
            return

        if believe_owned:
            if not self.executor.has_open_position and saved and saved.get("product_id"):
                self._entry_premium = saved.get("entry_premium")
                opt_type = OptionType.CALL if str(owned_symbol).startswith("C-") else OptionType.PUT
                self.executor.adopt(int(saved["product_id"]), int(saved.get("size") or 0),
                                    opt_type, owned_symbol)
            log.warning("HeikinAshi reconcile: position not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"owned": owned_symbol, "tracked": self.executor.tracked_symbol}})
            return

        self.executor.clear()
        self._entry_premium = None
        self.strategy.force_flat()
        self._closing = False
        log.info("HeikinAshi reconcile: no owned position — state FLAT")

    # ------------------------------------------------------------------ #
    # EOD wall-clock square-off -- a defensive fallback in case the WS feed
    # stalls right at 17:25; the primary EOD exit comes from strategy.update()
    # itself on the 17:25 bar (via the closed-candle path above).
    # ------------------------------------------------------------------ #
    def _entries_blocked(self) -> bool:
        now = datetime.now(_IST)
        if self._sq_off_date != now.date():
            return False
        resume = now.replace(
            hour=self.settings.entry_resume_hour,
            minute=self.settings.entry_resume_minute,
            second=0, microsecond=0,
        )
        return now < resume

    async def _square_off_scheduler(self) -> None:
        while True:
            now = datetime.now(_IST)
            target = now.replace(
                hour=self.settings.square_off_hour,
                minute=self.settings.square_off_minute,
                second=0, microsecond=0,
            )
            if now >= target:
                target += timedelta(days=1)
            wait_s = (target - now).total_seconds()
            log.info("HeikinAshi: next EOD square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off_all()
            except Exception as exc:  # noqa: BLE001
                log.error("HeikinAshi: EOD square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off_all(self) -> None:
        self._sq_off_date = datetime.now(_IST).date()
        log.info("HeikinAshi: EOD square-off firing", extra={"extra": {"date": str(self._sq_off_date)}})
        if not self.executor.has_open_position:
            return
        try:
            await self._close("EOD")
        except Exception as exc:  # noqa: BLE001
            log.error("HeikinAshi: EOD close failed", extra={"extra": {"error": str(exc)}})
            await self._sync_options_to_exchange()
