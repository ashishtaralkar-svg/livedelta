"""EMA21 Breakdown live trading engine (option BUY).

Runs Ema21BreakdownStrategy on 15-minute BTC candles, BUYING options (the
mirror of DCv2Engine's sell -- follows DCv3Engine's pattern, see that file's
module docstring for the full buy-side rationale): bullish signal -> buy a
CALL near ``target_premium``, bearish signal -> buy a PUT. Profit comes from
the premium RISING (rally take-profit, ``take_profit_pct``), not decaying.
This is the "15m + buy + premium~100 + 300% rally TP + ema_sl + 12-17h IST
entry window" config validated in scripts/backtest_ema21_breakdown.py
(3mo: +$90.69 net / 79 legs at 25 lots).

DIFFERENCES FROM DCv3Engine (dcv3_trader.py), which this otherwise mirrors
(state persistence, order placement via OptionsExecutor, self-heal, reconcile):
  * CLOSED-BAR ONLY -- Ema21BreakdownStrategy has no check_intracandle_sl()/
    apply_intracandle_pending() (DCv2Strategy's ASAP-intrabar-fill machinery),
    and its own backtest is closed-bar-only. Rather than invent new,
    never-validated intrabar behavior, this engine only acts on
    strategy.update() at candle close -- exactly what was backtested.
  * NO ROLLOVER -- Ema21BreakdownStrategy has no "still open directionally
    but option flat" concept (unlike DCv2/DCv3's directional-carry design):
    a closed setup just ends; a fresh one re-forms if conditions recur. So
    square-off is a plain daily close, no weekend-flat/continuous-roll
    branching.
  * NO RESUME-HOUR TRACKING -- entry_start_hour/entry_end_hour gating lives
    INSIDE the strategy itself (added when the backtest script gained
    --entry-start-hour/--entry-end-hour); _entries_blocked() here only
    checks the weekday skip list.

HONESTY: this strategy has the THINNEST live/backtest history of any bot in
the fleet (max 79 legs over 3 months on the winning config, never executed a
real order) -- start at DELTA_OPTION_CONTRACTS=1 and watch closely.

Runs as its own Docker container on a SEPARATE sub-account; position
ownership is tracked via its own ``DELTA_STATE_FILE``. Never touches any
other bot.
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
from ..strategy.ema21_breakdown import Ema21BreakdownStrategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 900  # 15 minutes

log = get_logger(__name__)


class Ema21BreakdownEngine:
    """Live engine wired to Ema21BreakdownStrategy, executed via BUYING
    options. See module docstring for how this differs from DCv3Engine."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = Ema21BreakdownStrategy(
            ema_len=settings.ema21_ema_len,
            max_wait=settings.ema21_max_wait,
            target_rr=settings.ema21_target_rr,
            trade_ce=settings.ema21_trade_ce,
            trade_pe=settings.ema21_trade_pe,
            trend_filter=settings.ema21_trend_filter,
            ema200_filter=settings.ema21_ema200_filter,
            ema50_len=settings.ema21_ema50_len,
            ema200_len=settings.ema21_ema200_len,
            ema_sl=settings.ema21_ema_sl,
            entry_start_hour=settings.ema21_entry_start_hour,
            entry_end_hour=settings.ema21_entry_end_hour,
            day_tz=settings.day_tz,
        )
        self.executor = OptionsExecutor(rest, settings)
        # No on_forming -- this engine is closed-bar only, see module docstring.
        self.aggregator = CandleAggregator(on_closed=self._on_closed_candle)
        self.ws: WebSocketManager | None = None
        self._last_closed_start: int | None = None
        self._tasks: set[asyncio.Task] = set()
        self._sq_off_task: asyncio.Task | None = None
        self._tp_poll_task: asyncio.Task | None = None
        self._sq_off_date: date | None = None

        self._entry_premium: float | None = None
        self._tp_price: float | None = None
        self._current_dir: int | None = None
        # BUY side: TP is a RALLY target -- 300% (default) -> the option must 4x.
        self._tp_mult = 1.0 + settings.take_profit_pct / 100.0
        self._entry_in_progress = False
        self._closing = False
        self._verify_misses = 0
        self._last_verify = 0.0

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        mode = "TESTNET" if self.settings.testnet else "LIVE"
        await self.notifier.notify(NotifyEvent.RESTART, mode=mode)
        await self._warmup()
        await self._sync_options_to_exchange()

        self.ws = WebSocketManager(
            ws_url=self.settings.ws_url,
            symbol=self.settings.symbol,
            resolution="15m",
            api_key=self.settings.api_key.get_secret_value() or None,
            api_secret=self.settings.api_secret.get_secret_value() or None,
            on_candle=self.aggregator.ingest,
            on_reconnect=self._on_reconnect,
            heartbeat_timeout_s=self.settings.heartbeat_timeout_s,
        )
        self._sq_off_task = asyncio.create_task(self._square_off_scheduler())
        if self.settings.ema21_tp_poll_seconds > 0:
            self._tp_poll_task = asyncio.create_task(self._tp_poll_loop())
        log.info("Ema21BreakdownEngine: starting live (BUY side)")
        await self.ws.run()

    async def stop(self) -> None:
        if self.ws:
            self.ws.stop()
        for t in (self._sq_off_task, self._tp_poll_task):
            if t is not None:
                t.cancel()
        if self.settings.close_on_shutdown and self.executor.has_open_position:
            try:
                await self.executor.close_option()
                if self.settings.state_file:
                    position_state.clear(self.settings.state_file)
                await self.notifier.notify(NotifyEvent.EXIT, reason="shutdown",
                                           size=self.settings.option_contracts)
                log.info("Ema21: closed option on shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("Ema21: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        # Mirrors Ema21BreakdownStrategy.ready: only needs EMA200 warmup if a
        # 200-EMA filter is actually enabled, otherwise just EMA(ema_len).
        needs_ema200 = self.settings.ema21_trend_filter or self.settings.ema21_ema200_filter
        longest_ema = max(self.settings.ema21_ema_len,
                          self.settings.ema21_ema200_len if needs_ema200 else self.settings.ema21_ema_len)
        bars_needed = max(
            self.settings.warmup_candles + longest_ema + 50,
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
        log.info("Ema21 warmup done",
                 extra={"extra": {"candles": len(closed), "ready": self.strategy.ready}})

    async def _fetch_history_paged(self, start: int, end: int) -> list[Candle]:
        page_span = 2000 * _BAR_SECONDS
        out: list[Candle] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + page_span, end)
            page = await asyncio.to_thread(
                self.rest.get_candles, self.settings.symbol, "15m", cursor, chunk_end
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
            log.warning("Ema21: candle gap detected — re-seeding")
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
                log.warning("Ema21: candle gap — re-seeding")
                await self._warmup()
        self._last_closed_start = candle.start_time

        dec = self.strategy.update(candle)

        if self.settings.ema21_debug_state:
            log.info("Ema21 state", extra={"extra": {
                "candle": candle.start_time, "o": candle.open, "h": candle.high,
                "l": candle.low, "c": candle.close, "blocked": self._entries_blocked(),
                "has_option": self.executor.has_open_position, **self.strategy.debug_state()}})

        # 1. Closed-bar exit (SL -- fixed anchor or dynamic EMA-close, per
        #    ema21_ema_sl -- or TARGET, only if ema21_target_rr > 0) closes
        #    the option.
        if dec is not None and dec.has_exit and self.executor.has_open_position:
            exit_price = dec.long_exit_price if dec.long_exit else dec.short_exit_price
            await self._close_leg(dec.exit_reason or "SL", btc_exit_price=exit_price)

        # 2. Rally TP: mark has RISEN to/above tp_price.
        if self.executor.has_open_position and self._tp_price is not None:
            symbol = self.executor.tracked_symbol
            mark = None
            if symbol:
                try:
                    mark = await asyncio.to_thread(self.rest.get_mark_price, symbol)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Ema21: get_mark_price failed", extra={"extra": {"error": str(exc)}})
            if mark is not None and mark >= self._tp_price:
                await self._close_tp(mark)
                return

        # 3. Closed-bar entry (the strategy's own entry_start_hour/
        #    entry_end_hour gate already applies inside update() above --
        #    only the weekday skip list is checked here).
        if (dec is not None and dec.has_entry
                and not self.executor.has_open_position and not self._entries_blocked()):
            signal_dir = SignalDir.LONG.value if dec.buy_signal else SignalDir.SHORT.value
            await self._open_entry(signal_dir, dec.sl_level, candle.close)

    # ------------------------------------------------------------------ #
    async def _tp_poll_loop(self) -> None:
        interval = self.settings.ema21_tp_poll_seconds
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            await self._maybe_verify_position()
            if self._closing or self._tp_price is None or not self.executor.has_open_position:
                continue
            symbol = self.executor.tracked_symbol
            if not symbol:
                continue
            try:
                mark = await asyncio.to_thread(self.rest.get_mark_price, symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning("Ema21: TP-poll mark fetch failed", extra={"extra": {"error": str(exc)}})
                continue
            if mark is not None and self._tp_price is not None and mark >= self._tp_price:
                log.info("Ema21: rally TP hit (poll)",
                         extra={"extra": {"mark": mark, "tp": self._tp_price}})
                await self._close_tp(mark)

    # ------------------------------------------------------------------ #
    async def _maybe_verify_position(self) -> None:
        """Self-heal: confirm the tracked LONG option still exists on the exchange."""
        iv = self.settings.position_verify_seconds
        if iv <= 0 or self._closing or self._entry_in_progress or not self.executor.has_open_position:
            self._verify_misses = 0
            return
        now = time.time()
        if now - self._last_verify < iv:
            return
        self._last_verify = now
        tracked = self.executor.tracked_product_id
        try:
            positions = await asyncio.to_thread(
                self.rest.get_option_positions, self.executor.underlying
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Ema21: position-verify fetch failed", extra={"extra": {"error": str(exc)}})
            return
        if any(p["size"] > 0 and p.get("product_id") == tracked for p in positions):
            self._verify_misses = 0
            return
        self._verify_misses += 1
        if self._verify_misses < 2:
            log.warning("Ema21: tracked position not on exchange (1st miss) — rechecking",
                        extra={"extra": {"contract": self.executor.tracked_symbol}})
            return
        contract = self.executor.tracked_symbol
        log.warning("Ema21: position closed OUTSIDE the bot — self-healing to FLAT",
                    extra={"extra": {"contract": contract}})
        self.executor.clear()
        if self.settings.state_file:
            position_state.clear(self.settings.state_file)
        self._entry_premium = self._tp_price = self._current_dir = None
        self._verify_misses = 0
        self.strategy.force_flat()
        await self.notifier.notify(
            NotifyEvent.EXIT, reason="closed outside the bot (self-healed)",
            contract=contract or "?", size=self.settings.option_contracts, side="buy",
        )

    # ------------------------------------------------------------------ #
    async def _close_tp(self, mark: float) -> None:
        if self._closing or not self.executor.has_open_position:
            return
        self._closing = True
        try:
            contract = self.executor.tracked_symbol
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("Ema21: TP close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"TP close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            exit_prem = fill if fill is not None else mark
            entry_prem = self._entry_premium
            lots = self.settings.option_contracts
            gross = (exit_prem - entry_prem) * lots * 0.001 if entry_prem is not None else 0.0
            self.strategy.force_flat()
            self._entry_premium = self._tp_price = self._current_dir = None
            log.info("Ema21 TP hit", extra={"extra": {"contract": contract, "exit_prem": exit_prem}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason="TP", contract=contract or "?",
                entry_premium=entry_prem, exit_premium=exit_prem,
                pnl=round(gross, 2), size=lots, side="buy",
            )
        finally:
            self._closing = False

    async def _close_leg(self, reason: str, btc_exit_price: float) -> None:
        if self._closing or not self.executor.has_open_position:
            return
        self._closing = True
        try:
            contract = self.executor.tracked_symbol
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("Ema21: leg close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            entry_prem = self._entry_premium
            lots = self.settings.option_contracts
            gross = ((fill - entry_prem) * lots * 0.001
                     if (entry_prem is not None and fill is not None) else 0.0)
            self._entry_premium = self._tp_price = self._current_dir = None
            log.info("Ema21 exit", extra={"extra": {
                "reason": reason, "contract": contract, "btc_exit": btc_exit_price}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                entry_premium=entry_prem, exit_premium=fill,
                pnl=round(gross, 2), size=lots, side="buy",
            )
        finally:
            self._closing = False

    async def _open_entry(self, signal_dir: int, sl_level: float | None, btc_price: float) -> None:
        """BUY the option for a new signal. Bullish -> buy CALL, bearish ->
        buy PUT (the OptionsExecutor picks the side; this just labels the
        notification correctly)."""
        if self._entry_in_progress or self.executor.has_open_position or self._entry_premium is not None:
            return
        self._entry_in_progress = True
        try:
            is_buy_signal = signal_dir == SignalDir.LONG.value
            try:
                fill, symbol = await self.executor.open_option_by_premium(
                    signal_dir, self.settings.target_premium
                )
            except OptionsMarginError as exc:
                log.error("Ema21: margin/balance error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Balance: {exc}")
                self.strategy.force_flat()
                return
            except Exception as exc:  # noqa: BLE001
                log.error("Ema21: open_option_by_premium failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                self.strategy.force_flat()
                return
            if fill is None:
                log.warning("Ema21: no option fill — flattening to stay in sync")
                self.strategy.force_flat()
                return

            self._entry_premium = fill
            self._tp_price = fill * self._tp_mult
            self._current_dir = signal_dir
            if self.settings.state_file:
                position_state.save(
                    self.settings.state_file, symbol=symbol or "",
                    product_id=self.executor.tracked_product_id,
                    size=self.settings.option_contracts, entry_premium=fill,
                    tp_price=self._tp_price, direction=signal_dir,
                )
            direction = "CALL" if is_buy_signal else "PUT"
            log.info("Ema21 entry", extra={"extra": {
                "direction": direction, "symbol": symbol, "fill": fill,
                "tp_price": round(self._tp_price, 1), "sl_level": sl_level}})
            event = NotifyEvent.ENTRY_LONG if is_buy_signal else NotifyEvent.ENTRY_SHORT
            await self.notifier.notify(
                event, direction=direction, contract=symbol or "?",
                premium=fill, btc_price=btc_price, sl_level=sl_level,
                tp_price=round(self._tp_price, 1), side="buy",
            )
        finally:
            self._entry_in_progress = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile the open option with the exchange -- looks for a LONG
        position (size > 0), same as DCv3 (buy-mode, strict single trade)."""
        state_file = self.settings.state_file
        saved = position_state.load(state_file) if state_file else None
        owned_symbol = saved.get("symbol") if saved else None
        believe_owned = owned_symbol is not None or self.executor.has_open_position

        longs: list[dict] = []
        for attempt in range(3):
            try:
                positions = await asyncio.to_thread(
                    self.rest.get_option_positions, self.executor.underlying
                )
            except Exception as exc:  # noqa: BLE001
                log.error("Ema21 reconcile: fetch failed",
                          extra={"extra": {"error": str(exc), "attempt": attempt}})
                positions = []
            longs = [p for p in positions if p["size"] > 0]
            if longs or not believe_owned:
                break
            log.warning("Ema21 reconcile: expected a position but fetch is empty — retrying",
                        extra={"extra": {"owned": owned_symbol, "attempt": attempt}})
            await asyncio.sleep(1.5)

        if longs:
            match = next((p for p in longs if p.get("symbol") == owned_symbol), longs[0])
            if saved and match.get("symbol") == owned_symbol:
                self._entry_premium = saved.get("entry_premium")
                self._tp_price = saved.get("tp_price")
                self._current_dir = saved.get("direction")
            opt_type = OptionType.CALL if match["symbol"].startswith("C-") else OptionType.PUT
            self.executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
            log.info("Ema21 reconcile: adopted open long",
                     extra={"extra": {"symbol": match["symbol"]}})
            return

        if believe_owned:
            if not self.executor.has_open_position and saved and saved.get("product_id"):
                self._entry_premium = saved.get("entry_premium")
                self._tp_price = saved.get("tp_price")
                self._current_dir = saved.get("direction")
                opt_type = OptionType.CALL if str(owned_symbol).startswith("C-") else OptionType.PUT
                self.executor.adopt(int(saved["product_id"]), int(saved.get("size") or 0),
                                    opt_type, owned_symbol)
            log.warning("Ema21 reconcile: position not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"owned": owned_symbol}})
            return

        self.executor.clear()
        self._entry_premium = self._tp_price = self._current_dir = None
        if self.strategy.position_state != PositionState.FLAT:
            self.strategy.force_flat()
        self._closing = False
        log.info("Ema21 reconcile: no owned position — state FLAT")

    # ------------------------------------------------------------------ #
    # Daily 17:25 square-off -- plain close, no rollover (see module docstring).
    # ------------------------------------------------------------------ #
    def _entries_blocked(self) -> bool:
        return datetime.now(_IST).weekday() in self.settings.skip_weekday_ints

    async def _square_off_scheduler(self) -> None:
        while True:
            now = datetime.now(_IST)
            target = now.replace(hour=self.settings.square_off_hour,
                                 minute=self.settings.square_off_minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait_s = (target - now).total_seconds()
            log.info("Ema21: next 17:25 square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off()
            except Exception as exc:  # noqa: BLE001
                log.error("Ema21: square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off(self) -> None:
        now = datetime.now(_IST)
        self._sq_off_date = now.date()
        log.info("Ema21: 17:25 square-off firing", extra={"extra": {"date": str(self._sq_off_date)}})
        if self.executor.has_open_position:
            try:
                contract = self.executor.tracked_symbol
                fill = await self.executor.close_option()
                if self.settings.state_file:
                    position_state.clear(self.settings.state_file)
                entry_prem = self._entry_premium
                lots = self.settings.option_contracts
                gross = ((fill - entry_prem) * lots * 0.001
                         if (entry_prem is not None and fill is not None) else 0.0)
                self._entry_premium = self._tp_price = self._current_dir = None
                await self.notifier.notify(
                    NotifyEvent.EXIT, reason="EOD", contract=contract or "?",
                    entry_premium=entry_prem, exit_premium=fill, pnl=round(gross, 2), size=lots,
                    side="buy",
                )
            except Exception as exc:  # noqa: BLE001
                log.error("Ema21: square-off close failed", extra={"extra": {"error": str(exc)}})
                await self._sync_options_to_exchange()
                return
        # No rollover concept -- a closed setup just ends; force_flat() lets a
        # fresh signal re-form from scratch (also harmless no-op if already flat).
        self.strategy.force_flat()
