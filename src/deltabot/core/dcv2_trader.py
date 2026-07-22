"""DCv2 live trading engine (option SELL, daily square-off + rollover).

Runs DCv2Strategy on 5-minute BTC candles (synthetic Heikin Ashi computed
internally) and sells options as the execution vehicle -- BUY signal -> SELL a
PUT near ``target_premium`` (900), SELL signal -> SELL a CALL. This is the
"5m + 70% TP + fri-flat + 25 lots" config validated in scripts/backtest_dcv2.py
(6mo: +$1,056 net at 25 lots).

EXECUTION TIMING (per request):
  * ENTRY (signal triggered): ASAP intracandle -- the instant REAL price breaks
    the signal-range trigger, sell the option (does not wait for the 5m close).
  * TARGET (70% premium-decay TP): ASAP -- a short poll of the option mark buys
    it back the moment premium <= 30% of entry, then the strategy is flattened
    so it hunts a FRESH signal (no rollover of a booked trade).
  * SL (fixed signal-range level): ASAP intracandle -- buy back the instant
    REAL price touches the range low (long) / high (short).
  * TRAILING SL (TRAIL) and the EMA-reversal exit: CLOSED-BAR only -- evaluated
    when the 5m candle closes beyond/against both EMAs (never intracandle).
  Intracandle entry+SL are gated by ``dcv2_intracandle_enabled`` (default on);
  turn it off to run strictly closed-bar and exactly match the backtest.

DAILY LIFECYCLE:
  * 17:25 IST square-off closes the OPTION leg (the directional trade keeps
    running inside the strategy). On the last session before a skip-weekday
    (Friday, when ``dcv2_weekend_flat``), it also FLATTENS the trade so nothing
    is carried over the weekend.
  * 17:30 IST: if the directional trade is still open, SELL a fresh
    ~target_premium option (ROLLOVER), preserving the strategy's SL/trail.

Runs as its own Docker container on a SEPARATE sub-account; position ownership
is tracked via its own ``DELTA_STATE_FILE``. Never touches the other bots.
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
from ..strategy.dcv2 import DCv2Strategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError, PaperExecutor

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 300  # 5 minutes — DCv2's validated timeframe

log = get_logger(__name__)


class DCv2Engine:
    """Live engine wired to DCv2Strategy (option SELL + daily square-off/rollover)."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = DCv2Strategy(
            dc_period=settings.dcv2_dc_period,
            ema_trend_length=settings.dcv2_ema_trend_length,
            ema_long_length=settings.dcv2_ema_long_length,
            skip_weekdays=settings.skip_weekday_ints,
            day_tz=settings.day_tz,
            day_start_hour=settings.day_start_hour,
            day_start_minute=settings.day_start_minute,
            square_off_hour=settings.square_off_hour,
            square_off_minute=settings.square_off_minute,
        )
        self.executor = (PaperExecutor(rest, settings) if settings.paper_mode
                         else OptionsExecutor(rest, settings))
        self.aggregator = CandleAggregator(
            on_closed=self._on_closed_candle, on_forming=self._on_forming_candle
        )
        self.ws: WebSocketManager | None = None
        self._last_closed_start: int | None = None
        self._tasks: set[asyncio.Task] = set()
        self._sq_off_task: asyncio.Task | None = None
        self._tp_poll_task: asyncio.Task | None = None
        self._sq_off_date: date | None = None

        # Open-option tracking (set on entry/rollover, cleared on close).
        self._entry_premium: float | None = None
        self._tp_price: float | None = None
        self._current_dir: int | None = None
        self._tp_mult = 1.0 - settings.take_profit_pct / 100.0   # 70% -> 0.30
        # Guards so poll / forming / closed-candle paths can't double open/close.
        self._entry_in_progress = False
        self._closing = False
        # Self-heal.
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
            resolution="5m",
            api_key=self.settings.api_key.get_secret_value() or None,
            api_secret=self.settings.api_secret.get_secret_value() or None,
            on_candle=self.aggregator.ingest,
            on_reconnect=self._on_reconnect,
            heartbeat_timeout_s=self.settings.heartbeat_timeout_s,
        )
        self._sq_off_task = asyncio.create_task(self._square_off_scheduler())
        if self.settings.dcv2_tp_poll_seconds > 0:
            self._tp_poll_task = asyncio.create_task(self._tp_poll_loop())
        log.info("DCv2Engine: starting live")
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
                log.info("DCv2: closed option on shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("DCv2: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        bars_needed = max(
            self.settings.warmup_candles + self.settings.dcv2_ema_long_length + 50,
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
        log.info("DCv2 warmup done",
                 extra={"extra": {"candles": len(closed), "ready": self.strategy.ready}})

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
            log.warning("DCv2: candle gap detected — re-seeding")
            await self._warmup()

    # ------------------------------------------------------------------ #
    # Intracandle: ASAP entry + ASAP SL (TRAIL/EMA-cross stay closed-bar).
    # ------------------------------------------------------------------ #
    def _on_forming_candle(self, candle: Candle) -> None:
        if not self.settings.dcv2_intracandle_enabled:
            return
        task = asyncio.create_task(self._handle_forming_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_forming_candle(self, candle: Candle) -> None:
        if not self.strategy.ready:
            return
        # ASAP SL: REAL price touching the fixed range level closes the leg NOW.
        if self.executor.has_open_position and not self._closing:
            long_sl, short_sl, sl = self.strategy.check_intracandle_sl(candle.close)
            if long_sl or short_sl:
                log.info("DCv2: intracandle SL touch — closing ASAP",
                         extra={"extra": {"sl": sl, "price": candle.close}})
                self.strategy.force_flat()
                await self._close_leg("SL", btc_exit_price=sl if sl is not None else candle.close)
                return
        # ASAP entry: the instant REAL price breaks the pending trigger.
        if self.executor.has_open_position or self._entry_in_progress:
            return
        if not self.strategy.has_pending or self._entries_blocked():
            return
        confirmed, invalidated, entry_price = self.strategy.apply_intracandle_pending(candle)
        if invalidated:
            log.info("DCv2: setup invalidated intracandle (SL side hit before trigger)")
            return
        if confirmed:
            signal_dir = (SignalDir.LONG.value
                          if self.strategy.position_state == PositionState.LONG
                          else SignalDir.SHORT.value)
            log.info("DCv2: intracandle breakout — entering ASAP",
                     extra={"extra": {"trigger": entry_price}})
            await self._open_entry(signal_dir, self.strategy.sl_level, entry_price, tag="ENTRY")

    def _on_closed_candle(self, candle: Candle) -> None:
        task = asyncio.create_task(self._handle_closed_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_closed_candle(self, candle: Candle) -> None:
        if self._last_closed_start is not None:
            gap = candle.start_time - self._last_closed_start
            if gap > _BAR_SECONDS:
                log.warning("DCv2: candle gap — re-seeding")
                await self._warmup()
        self._last_closed_start = candle.start_time

        # Strategy update -> closed-bar exits (SL / EMA_CROSS / TRAIL) + entries.
        dec = self.strategy.update(candle)

        # Optional per-candle diagnostic snapshot (DELTA_DCV2_DEBUG_STATE=true).
        if self.settings.dcv2_debug_state:
            log.info("DCv2 state", extra={"extra": {
                "candle": candle.start_time, "o": candle.open, "h": candle.high,
                "l": candle.low, "c": candle.close, "blocked": self._entries_blocked(),
                "has_option": self.executor.has_open_position, **self.strategy.debug_state()}})

        # 1. Closed-bar exit closes the option if we hold one. (Intracandle SL
        #    may already have handled it; guard on has_open_position.)
        if dec is not None and dec.has_exit and self.executor.has_open_position:
            exit_price = dec.long_exit_price if dec.long_exit else dec.short_exit_price
            await self._close_leg(dec.exit_reason or "SL", btc_exit_price=exit_price)

        # 2. 70% decay TP (mark check on the closed bar; the poll also runs).
        if self.executor.has_open_position and self._tp_price is not None:
            symbol = self.executor.tracked_symbol
            mark = None
            if symbol:
                try:
                    mark = await asyncio.to_thread(self.rest.get_mark_price, symbol)
                except Exception as exc:  # noqa: BLE001
                    log.warning("DCv2: get_mark_price failed", extra={"extra": {"error": str(exc)}})
            if mark is not None and mark <= self._tp_price:
                await self._close_tp(mark)
                return

        # 3. Closed-bar entry fallback (when intracandle is disabled, or a signal
        #    completes exactly on the close).
        if (dec is not None and dec.has_entry
                and not self.executor.has_open_position and not self._entries_blocked()):
            signal_dir = SignalDir.LONG.value if dec.buy_signal else SignalDir.SHORT.value
            await self._open_entry(signal_dir, dec.sl_level, candle.close, tag="ENTRY")
            return

        # 4. Rollover: directional trade still open but the option is flat (it was
        #    squared off at 17:25) -> re-sell in the same direction after the gap.
        if (self.settings.dcv2_rollover_enabled
                and not self.executor.has_open_position and not self._entry_in_progress
                and not self._entries_blocked()
                and self.strategy.position_state != PositionState.FLAT):
            signal_dir = (SignalDir.LONG.value
                          if self.strategy.position_state == PositionState.LONG
                          else SignalDir.SHORT.value)
            log.info("DCv2: rollover — re-selling option for the still-open trade")
            await self._open_entry(signal_dir, self.strategy.sl_level, candle.close, tag="ROLL")

    # ------------------------------------------------------------------ #
    async def _tp_poll_loop(self) -> None:
        interval = self.settings.dcv2_tp_poll_seconds
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
                log.warning("DCv2: TP-poll mark fetch failed", extra={"extra": {"error": str(exc)}})
                continue
            if mark is not None and self._tp_price is not None and mark <= self._tp_price:
                log.info("DCv2: 70% decay TP hit (poll)",
                         extra={"extra": {"mark": mark, "tp": self._tp_price}})
                await self._close_tp(mark)

    # ------------------------------------------------------------------ #
    async def _maybe_verify_position(self) -> None:
        if self.settings.paper_mode:
            return   # paper positions are not on the exchange; nothing to verify
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
            log.warning("DCv2: position-verify fetch failed", extra={"extra": {"error": str(exc)}})
            return
        if any(p["size"] < 0 and p.get("product_id") == tracked for p in positions):
            self._verify_misses = 0
            return
        self._verify_misses += 1
        if self._verify_misses < 2:
            log.warning("DCv2: tracked position not on exchange (1st miss) — rechecking",
                        extra={"extra": {"contract": self.executor.tracked_symbol}})
            return
        contract = self.executor.tracked_symbol
        log.warning("DCv2: position closed OUTSIDE the bot — self-healing to FLAT",
                    extra={"extra": {"contract": contract}})
        self.executor.clear()
        if self.settings.state_file:
            position_state.clear(self.settings.state_file)
        self._entry_premium = self._tp_price = self._current_dir = None
        self._verify_misses = 0
        self.strategy.force_flat()
        await self.notifier.notify(
            NotifyEvent.EXIT, reason="closed outside the bot (self-healed)",
            contract=contract or "?", size=self.settings.option_contracts,
        )

    # ------------------------------------------------------------------ #
    async def _close_tp(self, mark: float) -> None:
        """70% decay TP: book the profit and FLATTEN the strategy so it hunts a
        fresh signal (no rollover of a booked trade)."""
        if self._closing or not self.executor.has_open_position:
            return
        self._closing = True
        try:
            contract = self.executor.tracked_symbol
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("DCv2: TP close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"TP close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            exit_prem = fill if fill is not None else mark
            entry_prem = self._entry_premium
            lots = self.settings.option_contracts
            gross = (entry_prem - exit_prem) * lots * 0.001 if entry_prem is not None else 0.0
            self.strategy.force_flat()
            self._entry_premium = self._tp_price = self._current_dir = None
            log.info("DCv2 TP hit", extra={"extra": {"contract": contract, "exit_prem": exit_prem}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason="TP", contract=contract or "?",
                entry_premium=entry_prem, exit_premium=exit_prem,
                pnl=round(gross, 2), size=lots,
            )
        finally:
            self._closing = False

    async def _close_leg(self, reason: str, btc_exit_price: float) -> None:
        """Close the option because a strategy exit (SL / EMA_CROSS / TRAIL) fired
        or an intracandle SL was hit. The strategy is already/also flattened."""
        if self._closing or not self.executor.has_open_position:
            return
        self._closing = True
        try:
            contract = self.executor.tracked_symbol
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("DCv2: leg close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            entry_prem = self._entry_premium
            lots = self.settings.option_contracts
            gross = ((entry_prem - fill) * lots * 0.001
                     if (entry_prem is not None and fill is not None) else 0.0)
            self._entry_premium = self._tp_price = self._current_dir = None
            log.info("DCv2 exit", extra={"extra": {
                "reason": reason, "contract": contract, "btc_exit": btc_exit_price}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                entry_premium=entry_prem, exit_premium=fill,
                pnl=round(gross, 2), size=lots,
            )
        finally:
            self._closing = False

    async def _open_entry(self, signal_dir: int, sl_level: float | None,
                          btc_price: float, tag: str) -> None:
        """SELL the option for a new signal (or a rollover). Bullish -> sell PUT,
        bearish -> sell CALL."""
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
                log.error("DCv2: margin error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Margin: {exc}")
                self.strategy.force_flat()
                return
            except Exception as exc:  # noqa: BLE001
                log.error("DCv2: open_option_by_premium failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                self.strategy.force_flat()
                return
            if fill is None:
                log.warning("DCv2: no option fill — flattening to stay in sync")
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
            direction = "PUT" if is_buy else "CALL"
            log.info("DCv2 entry", extra={"extra": {
                "tag": tag, "direction": direction, "symbol": symbol, "fill": fill,
                "tp_price": round(self._tp_price, 1), "sl_level": sl_level}})
            event = NotifyEvent.ENTRY_LONG if is_buy else NotifyEvent.ENTRY_SHORT
            await self.notifier.notify(
                event, direction=direction, contract=symbol or "?",
                premium=fill, btc_price=btc_price, sl_level=sl_level,
                tp_price=round(self._tp_price, 1), tag=tag,
            )
        finally:
            self._entry_in_progress = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile the open option with the exchange on start/reconnect (this
        sub-account runs only dcv2bot, so any open short is ours)."""
        if self.settings.paper_mode:
            # No real positions to reconcile; a paper leg never survives a
            # restart. Start flat and let the strategy re-hunt.
            self.executor.clear()
            self._entry_premium = self._tp_price = self._current_dir = None
            self.strategy.force_flat()
            self._closing = False
            log.info("DCv2 (paper): reconcile skipped — starting FLAT")
            return
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
                log.error("DCv2 reconcile: fetch failed",
                          extra={"extra": {"error": str(exc), "attempt": attempt}})
                positions = []
            shorts = [p for p in positions if p["size"] < 0]
            if shorts or not believe_owned:
                break
            log.warning("DCv2 reconcile: expected a position but fetch is empty — retrying",
                        extra={"extra": {"owned": owned_symbol, "attempt": attempt}})
            await asyncio.sleep(1.5)

        if shorts:
            match = next((p for p in shorts if p.get("symbol") == owned_symbol), shorts[0])
            if saved and match.get("symbol") == owned_symbol:
                self._entry_premium = saved.get("entry_premium")
                self._tp_price = saved.get("tp_price")
                self._current_dir = saved.get("direction")
            opt_type = OptionType.CALL if match["symbol"].startswith("C-") else OptionType.PUT
            self.executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
            log.info("DCv2 reconcile: adopted open short",
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
            log.warning("DCv2 reconcile: position not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"owned": owned_symbol}})
            return

        self.executor.clear()
        self._entry_premium = self._tp_price = self._current_dir = None
        self.strategy.force_flat()
        self._closing = False
        log.info("DCv2 reconcile: no owned position — state FLAT")

    # ------------------------------------------------------------------ #
    # Daily 17:25 square-off + Friday-flat; 17:30 rollover is in the closed-bar
    # handler (re-sells while the directional trade is still open).
    # ------------------------------------------------------------------ #
    def _entries_blocked(self) -> bool:
        now = datetime.now(_IST)
        if now.weekday() in self.settings.skip_weekday_ints:
            return True
        if self._sq_off_date != now.date():
            return False
        resume = now.replace(hour=self.settings.entry_resume_hour,
                             minute=self.settings.entry_resume_minute, second=0, microsecond=0)
        return now < resume

    def _weekend_flat_today(self, now: datetime) -> bool:
        """True on the last session before a skip day: flatten (don't roll) when
        today OR tomorrow is a blocked weekday (so Friday's 17:25 ends the trade
        before Saturday). Matches the backtest's fri-flat mode."""
        if not self.settings.dcv2_weekend_flat:
            return False
        skip = self.settings.skip_weekday_ints
        return now.weekday() in skip or ((now.weekday() + 1) % 7) in skip

    async def _square_off_scheduler(self) -> None:
        while True:
            now = datetime.now(_IST)
            target = now.replace(hour=self.settings.square_off_hour,
                                 minute=self.settings.square_off_minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait_s = (target - now).total_seconds()
            log.info("DCv2: next 17:25 square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off()
            except Exception as exc:  # noqa: BLE001
                log.error("DCv2: square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off(self) -> None:
        now = datetime.now(_IST)
        self._sq_off_date = now.date()
        weekend_flat = self._weekend_flat_today(now)
        reason = "WEEKEND" if weekend_flat else "EOD"
        log.info("DCv2: 17:25 square-off firing",
                 extra={"extra": {"date": str(self._sq_off_date), "weekend_flat": weekend_flat}})
        if self.executor.has_open_position:
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
                await self.notifier.notify(
                    NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                    entry_premium=entry_prem, exit_premium=fill, pnl=round(gross, 2), size=lots,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("DCv2: square-off close failed", extra={"extra": {"error": str(exc)}})
                await self._sync_options_to_exchange()
                return
        # Friday: end the directional trade entirely (no weekend rollover).
        # Mon-Thu: leave the trade open; the 17:30 rollover re-sells it.
        if weekend_flat:
            self.strategy.force_flat()
            log.info("DCv2: weekend-flat — directional trade closed, no rollover")
