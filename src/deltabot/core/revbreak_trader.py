"""RevBreak live trading engine.

Runs RevBreakStrategy on 5-minute BTC candles; executes option trades via
OptionsExecutor in SELL mode; checks the option's mark price on every closed bar
for the -N% take-profit; manages BTC-stop-based SL and wall-clock EOD square-off.

Designed as a second Docker container alongside the PineStrategy bot. Position
ownership is tracked via a state file (``DELTA_STATE_FILE``) so each bot only
reconciles its own position on restart.

Data flow per closed 5m BTC candle:
    1. If position open: fetch option mark price → check TP
    2. RevBreakStrategy.update(candle) → exits (BTC SL / EOD) + new entries
    3. If BTC exit fires: close option via REST
    4. If new entry signal: open option nearest to target_premium via option chain
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from ..config import Settings
from ..enums import NotifyEvent, OptionType, PositionState, SignalDir
from ..exchange.rest_client import RestClient
from ..exchange.ws_manager import WebSocketManager
from ..logging_setup import get_logger
from ..models import Candle
from ..strategy.revbreak import RevBreakStrategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 300  # 5 minutes — RevBreak always runs on 5m BTC candles

log = get_logger(__name__)


class RevBreakEngine:
    """Live trading engine wired to RevBreakStrategy."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = RevBreakStrategy(
            atr_period=settings.atr_period,
            st_multiplier=settings.st_multiplier,
            gate=settings.revbreak_gate,
            st_entry_filter=settings.revbreak_st_filter,
            reentry_block=settings.revbreak_reentry_block,
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

        # Option TP tracking (set on entry, cleared on close).
        self._entry_premium: float | None = None
        self._tp_price: float | None = None
        self._current_dir: int | None = None  # SignalDir value of open position
        self._tp_mult = 1.0 - settings.take_profit_pct / 100.0
        # Re-entrancy guard so intracandle + closed-candle paths can't double-open.
        self._entry_in_progress = False

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        mode = "TESTNET" if self.settings.testnet else "LIVE"
        await self.notifier.notify(NotifyEvent.RESTART, mode=f"{mode} [RevBreak]")

        # Warmup strategy with historical 5m candles.
        await self._warmup()

        # Reconcile: re-adopt own position if state file exists.
        await self._sync_options_to_exchange()

        self.ws = WebSocketManager(
            ws_url=self.settings.ws_url,
            symbol=self.settings.symbol,
            resolution="5m",
            api_key=self.settings.api_key.get_secret_value() or None,
            api_secret=self.settings.api_secret.get_secret_value() or None,
            on_candle=self.aggregator.ingest,
            on_reconnect=self._on_reconnect,
            heartbeat_timeout_s=self.settings.heartbeat_timeout_s,
        )
        self._sq_off_task = asyncio.create_task(self._square_off_scheduler())
        log.info("RevBreakEngine: starting live")
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
                log.info("RevBreak: closed option on shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("RevBreak: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass  # no ledger in RevBreak engine; skip daily PnL summary

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        bars_needed = max(
            self.settings.warmup_candles + self.settings.atr_period + 5,
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
        log.info("RevBreak warmup done", extra={"extra": {"candles": len(closed)}})

    async def _fetch_history_paged(self, start: int, end: int) -> list[Candle]:
        page_span = 2000 * _BAR_SECONDS
        out: list[Candle] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + page_span, end)
            page = await asyncio.to_thread(
                self.rest.get_candles, self.settings.symbol, "5m", cursor, chunk_end
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
            log.warning("RevBreak: candle gap detected — re-seeding")
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
                log.warning("RevBreak: candle gap — re-seeding")
                await self._warmup()
        self._last_closed_start = candle.start_time

        if self._entries_blocked() and not self.executor.has_open_position:
            # After EOD square-off: keep feeding the strategy but suppress new entries.
            self.strategy.update(candle)
            return

        # 1. Option TP check (before strategy update, mirrors backtest order).
        if self.executor.has_open_position and self._tp_price is not None:
            try:
                mark = await asyncio.to_thread(
                    self.rest.get_mark_price, self.executor.tracked_symbol or ""
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("RevBreak: get_mark_price failed", extra={"extra": {"error": str(exc)}})
                mark = None
            if mark is not None and mark <= self._tp_price:
                await self._close_tp(mark, candle)
                # Still call strategy.update so Supertrend/day state advances.
                self.strategy.update(candle)
                return

        # 2. Strategy update → exits and entries.
        dec = self.strategy.update(candle)
        if dec is None:
            return

        # 3. BTC exit (SL / EOD).
        if dec.has_exit and self.executor.has_open_position:
            exit_price = dec.long_exit_price if dec.long_exit else dec.short_exit_price
            await self._close_btc_exit(dec.exit_reason or "SL", exit_price, candle)

        # 4. New entry (only if now flat and not in settlement window). The
        #    intracandle path usually fires first; this is the closed-bar fallback.
        if dec.has_entry and not self.executor.has_open_position and not self._entries_blocked():
            signal_dir = SignalDir.LONG.value if dec.buy_signal else SignalDir.SHORT.value
            await self._open_entry(signal_dir, dec.sl_level, candle.close)

    # ------------------------------------------------------------------ #
    def _on_forming_candle(self, candle: Candle) -> None:
        task = asyncio.create_task(self._handle_forming_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_forming_candle(self, candle: Candle) -> None:
        """Intracandle updates: enter ASAP when price crosses the pattern trigger,
        and exit ASAP when the BTC pattern-extreme stop is touched — instead of
        waiting for the 5m candle to close."""
        if not self.strategy.ready:
            return

        # Open position → check the BTC stop against the running low/high.
        if self.executor.has_open_position:
            for price in (candle.low, candle.high):
                long_sl, short_sl, level = self.strategy.check_intracandle_sl(price)
                if long_sl or short_sl:
                    direction = SignalDir.LONG.value if long_sl else SignalDir.SHORT.value
                    log.info("RevBreak: intracandle SL touched", extra={"extra": {"price": price, "sl": level}})
                    await self._close_btc_exit("SL", level if level is not None else price, candle)
                    self.strategy.notify_exit(direction, "SL")
                    return
            return

        # Flat → watch the armed setup for an ASAP breakout entry.
        if self._entry_in_progress or not self.strategy.has_pending or self._entries_blocked():
            return
        confirmed, invalidated, entry_price = self.strategy.apply_intracandle_pending(candle)
        if invalidated:
            log.info("RevBreak: setup invalidated intracandle (SL crossed before trigger)")
            return
        if confirmed:
            signal_dir = (SignalDir.LONG.value
                          if self.strategy.position_state == PositionState.LONG
                          else SignalDir.SHORT.value)
            log.info("RevBreak: intracandle breakout — entering ASAP",
                     extra={"extra": {"trigger": entry_price}})
            await self._open_entry(signal_dir, self.strategy.sl_level, entry_price)

    # ------------------------------------------------------------------ #
    async def _close_tp(self, mark: float, candle: Candle) -> None:
        """Close position because option mark price hit the −N% take-profit."""
        contract = self.executor.tracked_symbol
        try:
            fill = await self.executor.close_option()
        except Exception as exc:  # noqa: BLE001
            log.error("RevBreak: TP close failed", extra={"extra": {"error": str(exc)}})
            await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"TP close: {exc}")
            return

        if self.settings.state_file:
            position_state.clear(self.settings.state_file)

        exit_prem = fill if fill is not None else mark
        lots = self.settings.option_contracts
        entry_prem = self._entry_premium
        gross = (entry_prem - exit_prem) * lots * 0.001 if entry_prem else 0.0

        if self._current_dir is not None:
            self.strategy.notify_exit(self._current_dir, "TP")
        self._entry_premium = self._tp_price = self._current_dir = None

        log.info("RevBreak TP hit", extra={"extra": {"contract": contract, "exit_prem": exit_prem}})
        await self.notifier.notify(
            NotifyEvent.EXIT,
            reason="TP",
            contract=contract or "?",
            entry_premium=entry_prem,
            exit_premium=exit_prem,
            pnl=round(gross, 2),
            size=lots,
        )

    async def _close_btc_exit(self, reason: str, btc_exit_price: float, candle: Candle) -> None:
        """Close position because BTC hit the pattern stop-loss or EOD fired."""
        if not self.executor.has_open_position:
            return  # already closed (e.g. intracandle SL beat the closed-candle exit)
        contract = self.executor.tracked_symbol
        try:
            fill = await self.executor.close_option()
        except Exception as exc:  # noqa: BLE001
            log.error("RevBreak: BTC-exit close failed", extra={"extra": {"error": str(exc)}})
            await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close: {exc}")
            return

        if self.settings.state_file:
            position_state.clear(self.settings.state_file)

        exit_prem = fill
        entry_prem = self._entry_premium
        lots = self.settings.option_contracts
        gross = ((entry_prem - exit_prem) * lots * 0.001
                 if (entry_prem is not None and exit_prem is not None) else 0.0)

        self._entry_premium = self._tp_price = self._current_dir = None

        log.info("RevBreak exit", extra={"extra": {
            "reason": reason, "contract": contract, "btc_exit": btc_exit_price,
        }})
        await self.notifier.notify(
            NotifyEvent.EXIT,
            reason=reason,
            contract=contract or "?",
            entry_premium=entry_prem,
            exit_premium=exit_prem,
            pnl=round(gross, 2),
            size=lots,
        )

    async def _open_entry(self, signal_dir: int, sl_level: float | None, btc_price: float) -> None:
        """Open a short option for a new RevBreak signal. Callable from both the
        intracandle (ASAP) and closed-candle (fallback) paths; guarded so the two
        can never double-open. ``signal_dir`` is a SignalDir value; bullish sells a
        PUT, bearish sells a CALL."""
        if self._entry_in_progress or self.executor.has_open_position:
            return
        self._entry_in_progress = True
        try:
            is_buy = signal_dir == SignalDir.LONG.value
            try:
                fill, symbol = await self.executor.open_option_by_premium(
                    signal_dir, self.settings.target_premium
                )
            except OptionsMarginError as exc:
                log.error("RevBreak: margin error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Margin: {exc}")
                self.strategy.notify_exit(signal_dir, "SL")  # unblock + flatten strategy
                return
            except Exception as exc:  # noqa: BLE001
                log.error("RevBreak: open_option_by_premium failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                self.strategy.notify_exit(signal_dir, "SL")
                return

            if fill is None:
                log.warning("RevBreak: no option fill — skipping entry")
                self.strategy.notify_exit(signal_dir, "SL")
                return

            self._entry_premium = fill
            self._tp_price = fill * self._tp_mult
            self._current_dir = signal_dir

            if self.settings.state_file:
                position_state.save(
                    self.settings.state_file,
                    symbol=symbol or "",
                    entry_premium=fill,
                    tp_price=self._tp_price,
                    direction=signal_dir,
                )

            direction = "PUT" if is_buy else "CALL"
            log.info("RevBreak entry", extra={"extra": {
                "direction": direction, "symbol": symbol, "fill": fill,
                "tp_price": round(self._tp_price, 1), "sl_level": sl_level,
            }})
            event = NotifyEvent.ENTRY_LONG if is_buy else NotifyEvent.ENTRY_SHORT
            await self.notifier.notify(
                event,
                direction=direction,
                contract=symbol or "?",
                premium=fill,
                btc_price=btc_price,
                sl_level=sl_level,
                tp_price=round(self._tp_price, 1),
            )
        finally:
            self._entry_in_progress = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile open option position against the exchange on start/reconnect.

        Loads the state file (if configured) to identify which symbol this bot
        owns. Only adopts a position matching that symbol; ignores all others.
        """
        try:
            positions = await asyncio.to_thread(
                self.rest.get_option_positions, self.executor.underlying
            )
        except Exception as exc:  # noqa: BLE001
            log.error("RevBreak reconcile: failed to fetch positions", extra={"extra": {"error": str(exc)}})
            return

        shorts = [p for p in positions if p["size"] < 0]

        state_file = self.settings.state_file
        if state_file:
            saved = position_state.load(state_file)
            if saved:
                owned_symbol = saved.get("symbol")
                shorts = [p for p in shorts if p.get("symbol") == owned_symbol]
                if shorts:
                    # Restore TP tracking from state file.
                    self._entry_premium = saved.get("entry_premium")
                    self._tp_price = saved.get("tp_price")
                    self._current_dir = saved.get("direction")
            else:
                shorts = []  # No state file → start flat.

        if not shorts:
            self.executor.clear()
            self._entry_premium = self._tp_price = self._current_dir = None
            log.info("RevBreak reconcile: no owned position — state FLAT")
            return

        pos = shorts[0]
        opt_type = OptionType.CALL if pos["symbol"].startswith("C-") else OptionType.PUT
        self.executor.adopt(pos["product_id"], pos["size"], opt_type, pos.get("symbol"))
        log.info("RevBreak reconcile: adopted own position",
                 extra={"extra": {"symbol": pos["symbol"], "tp_price": self._tp_price}})

    # ------------------------------------------------------------------ #
    # EOD wall-clock square-off (mirrors TradingEngine._square_off_scheduler)
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
            log.info("RevBreak: next EOD square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off_all()
            except Exception as exc:  # noqa: BLE001
                log.error("RevBreak: EOD square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off_all(self) -> None:
        self._sq_off_date = datetime.now(_IST).date()
        log.info("RevBreak: EOD square-off firing", extra={"extra": {"date": str(self._sq_off_date)}})
        if not self.executor.has_open_position:
            return
        try:
            contract = self.executor.tracked_symbol
            fill = await self.executor.close_option()
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            entry_prem = self._entry_premium
            lots = self.settings.option_contracts
            gross = ((entry_prem - fill) * lots * 0.001
                     if (entry_prem is not None and fill is not None) else 0.0)
            self._entry_premium = self._tp_price = self._current_dir = None
            log.info("RevBreak: EOD square-off complete")
            await self.notifier.notify(
                NotifyEvent.EXIT,
                reason="EOD",
                contract=contract or "?",
                entry_premium=entry_prem,
                exit_premium=fill,
                pnl=round(gross, 2),
                size=lots,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("RevBreak: EOD close failed", extra={"extra": {"error": str(exc)}})
            await self._sync_options_to_exchange()
