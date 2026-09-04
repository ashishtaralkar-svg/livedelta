"""Range Engulfing Fade (Sell Only, Intracandle) live trading engine.

Runs RangeEngulfingFadeSellStrategy on 15-minute BTC candles: a red-then-
green range-engulfing pattern arms a trigger at the green candle's high;
the INSTANT a later candle's real price reaches that trigger, sells an
AT-THE-MONEY CALL (requires DELTA_OPTION_OFFSET=0 -- see
.env.range_fade.example) -- fixed 1:1 risk:reward, SL/TARGET are the
pattern's own height/low. No premium target, no rally TP: the only exits
are that frozen SL/TARGET or the daily square-off.

KEY DIFFERENCE FROM EVERY OTHER ENGINE IN THIS REPO: this is the first
genuinely INTRACANDLE-live engine in the fleet. Every other engine only
reacts to CLOSED candles (CandleAggregator's on_closed); this one ALSO
wires on_forming, which the exchange's candlestick websocket channel
already emits continuously for the currently-forming bar (see
candle_aggregator.py's own docstring). Concretely:
  * on_closed  -> strategy.arm_from_closed_candle() ONLY, called
    SYNCHRONOUSLY (no asyncio.create_task -- arming is pure in-memory
    pattern math, no I/O). This matters: CandleAggregator fires the new
    candle's first on_forming callback immediately after on_closed, in
    the SAME synchronous call, so arming must complete first or that
    first forming tick could check a stale, not-yet-expired trigger.
    Pattern detection itself needs a full closed candle to confirm the
    red/green engulf+range+close conditions (matching the .pine chart's
    own bullEngulf, likewise only evaluated once per closed bar since
    calc_on_every_tick=false there too) -- so no entry/exit logic here.
  * on_forming -> strategy.check_intracandle_entry()/check_intracandle_
    exit() on EVERY tick of the live forming candle -- this is what makes
    "trade should be triggered immediate" real: the SELL (or the SL/
    TARGET close) fires the instant real price crosses the level, not up
    to 15 minutes later at the next candle close.
Both check_intracandle_*() methods mutate the strategy's state the
INSTANT a condition is detected (before the async order call even
starts), so repeated forming ticks arriving while an order is still
in-flight are safe no-ops by construction -- see the strategy module's
own docstring for why. A failed order (network error, exchange reject)
is caught by the periodic self-heal verify loop below, same as every
other engine.

DIFFERENCES FROM ema21_trader.py (which this otherwise mirrors -- state
persistence via OptionsExecutor, self-heal, reconcile):
  * SELL-mode (option sold, not bought): P&L is (entry - exit), decay is
    profit -- the mirror of ema21's buy-side sign, same as supertrend's.
  * NO TP POLL LOOP -- like supertrend_trader.py, this strategy has no
    premium target at all (fixed SL/TARGET only), so a dedicated periodic
    self-heal loop replaces the piggyback used by TP-poll bots.
  * ATM EXECUTION, not premium-target: uses OptionsExecutor.open_option()
    (BTC price +/- option_offset, snapped to the nearest LISTED strike)
    with DELTA_OPTION_OFFSET=0 in this bot's own .env for pure ATM --
    NOT open_option_by_premium(), which every premium-target bot uses.
    The BTC price passed in is the exact trigger level the strategy fired
    at (a real, precise price), not the candle's open/close.
  * SELL-ONLY, single leg -- only ever sells a CALL (the strategy has no
    long/buy side at all, per its own module docstring), so reconcile
    only ever looks for ONE short leg, no CE/PE disambiguation needed
    (unlike supertrend's dual-leg reconcile).

HONESTY: this strategy has NEVER executed a real order, and this is the
first live engine to use on_forming at all -- start at
DELTA_OPTION_CONTRACTS=1 and watch closely before trusting the configured
lot size.

Runs as its own Docker container on a SEPARATE sub-account; position
ownership is tracked via its own DELTA_STATE_FILE. Never touches any
other bot.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from ..config import Settings
from ..enums import NotifyEvent, OptionType, SignalDir
from ..exchange.rest_client import RestClient
from ..exchange.ws_manager import WebSocketManager
from ..logging_setup import get_logger
from ..models import Candle
from ..strategy.range_engulfing_fade_sell import RangeEngulfingFadeSellStrategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 900  # 15 minutes

log = get_logger(__name__)


class RangeEngulfingFadeSellEngine:
    """Live engine wired to RangeEngulfingFadeSellStrategy: sell-only,
    intracandle, ATM, 1:1 R:R. See module docstring."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = RangeEngulfingFadeSellStrategy()
        self.executor = OptionsExecutor(rest, settings)
        self.aggregator = CandleAggregator(
            on_closed=self._on_closed_candle, on_forming=self._on_forming_candle
        )
        self.ws: WebSocketManager | None = None
        self._last_closed_start: int | None = None
        self._tasks: set[asyncio.Task] = set()
        self._sq_off_task: asyncio.Task | None = None
        self._verify_task: asyncio.Task | None = None
        self._sq_off_date: date | None = None

        self._entry_premium: float | None = None
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
        if self.settings.position_verify_seconds > 0:
            self._verify_task = asyncio.create_task(self._verify_loop())
        log.info("RangeEngulfingFadeSellEngine: starting live (sell-only, intracandle)")
        await self.ws.run()

    async def stop(self) -> None:
        if self.ws:
            self.ws.stop()
        for t in (self._sq_off_task, self._verify_task):
            if t is not None:
                t.cancel()
        if self.settings.close_on_shutdown and self.executor.has_open_position:
            try:
                lots = self.executor.tracked_size   # captured BEFORE close_option() clears tracked state
                await self.executor.close_option()
                if self.settings.state_file:
                    position_state.clear(self.settings.state_file)
                await self.notifier.notify(NotifyEvent.EXIT, reason="shutdown", size=lots, side="sell CE")
                log.info("RangeFade: closed option on shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("RangeFade: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        bars_needed = max(
            self.settings.warmup_candles, self.settings.warmup_days * 86400 // _BAR_SECONDS
        )
        start = last_closed_end - bars_needed * _BAR_SECONDS
        candles = await self._fetch_history_paged(start, last_closed_end)
        current_bar = (now // _BAR_SECONDS) * _BAR_SECONDS
        closed = [c for c in candles if c.start_time < current_bar]
        for c in closed:
            self.strategy.arm_from_closed_candle(c)
        if closed:
            self._last_closed_start = closed[-1].start_time
        log.info("RangeFade warmup done", extra={"extra": {
            "candles": len(closed), **self.strategy.debug_state()}})

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
            log.warning("RangeFade: candle gap detected — re-seeding")
            await self._warmup()

    # ------------------------------------------------------------------ #
    # Closed candle: pattern arming/expiry ONLY, run SYNCHRONOUSLY (pure
    # in-memory math, no I/O) -- see module docstring for why this must
    # complete before the new candle's first on_forming callback, which
    # CandleAggregator fires immediately after this, in the same
    # synchronous call.
    # ------------------------------------------------------------------ #
    def _on_closed_candle(self, candle: Candle) -> None:
        gap = (candle.start_time - self._last_closed_start) if self._last_closed_start is not None else 0
        self._last_closed_start = candle.start_time

        self.strategy.arm_from_closed_candle(candle)

        if self.settings.range_fade_debug_state:
            log.info("RangeFade state", extra={"extra": {
                "candle": candle.start_time, "o": candle.open, "h": candle.high,
                "l": candle.low, "c": candle.close, "blocked": self._entries_blocked(),
                "has_option": self.executor.has_open_position, **self.strategy.debug_state()}})

        if gap > _BAR_SECONDS:
            log.warning("RangeFade: candle gap — re-seeding")
            task = asyncio.create_task(self._warmup())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------------ #
    # Forming candle: the REAL-TIME reaction -- fires the instant price
    # crosses a level, not at candle close. See module docstring.
    # ------------------------------------------------------------------ #
    def _on_forming_candle(self, candle: Candle) -> None:
        if self.strategy.in_short:
            result = self.strategy.check_intracandle_exit(candle)
            if result is not None:
                reason, exit_price = result
                task = asyncio.create_task(self._close_leg(reason, exit_price))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        elif not self._entries_blocked():
            result = self.strategy.check_intracandle_entry(candle)
            if result is not None:
                entry_trigger, sl_level, _target = result
                task = asyncio.create_task(self._open_entry(entry_trigger, sl_level))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------------ #
    async def _verify_loop(self) -> None:
        interval = self.settings.position_verify_seconds
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            await self._maybe_verify_position()

    async def _maybe_verify_position(self) -> None:
        """Self-heal: confirm the tracked SHORT option still exists on the exchange."""
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
            log.warning("RangeFade: position-verify fetch failed", extra={"extra": {"error": str(exc)}})
            return
        if any(p["size"] < 0 and p.get("product_id") == tracked for p in positions):
            self._verify_misses = 0
            return
        self._verify_misses += 1
        if self._verify_misses < 2:
            log.warning("RangeFade: tracked position not on exchange (1st miss) — rechecking",
                        extra={"extra": {"contract": self.executor.tracked_symbol}})
            return
        contract = self.executor.tracked_symbol
        lots = self.executor.tracked_size   # captured BEFORE executor.clear() wipes tracked state
        log.warning("RangeFade: position closed OUTSIDE the bot — self-healing to FLAT",
                    extra={"extra": {"contract": contract}})
        self.executor.clear()
        if self.settings.state_file:
            position_state.clear(self.settings.state_file)
        self._entry_premium = None
        self._verify_misses = 0
        self.strategy.force_flat()
        await self.notifier.notify(
            NotifyEvent.EXIT, reason="closed outside the bot (self-healed)",
            contract=contract or "?", size=lots, side="sell CE",
        )

    # ------------------------------------------------------------------ #
    async def _close_leg(self, reason: str, btc_exit_price: float) -> None:
        if self._closing or not self.executor.has_open_position:
            return
        self._closing = True
        try:
            contract = self.executor.tracked_symbol
            lots = self.executor.tracked_size   # captured BEFORE close_option() clears tracked state
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("RangeFade: leg close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            entry_prem = self._entry_premium
            # SELL side: profit when the premium DECAYED (entry - exit).
            gross = ((entry_prem - fill) * lots * 0.001
                     if (entry_prem is not None and fill is not None) else 0.0)
            self._entry_premium = None
            log.info("RangeFade exit", extra={"extra": {
                "reason": reason, "contract": contract, "btc_exit": btc_exit_price}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                entry_premium=entry_prem, exit_premium=fill,
                pnl=round(gross, 2), size=lots, side="sell CE",
            )
        finally:
            self._closing = False

    async def _open_entry(self, entry_trigger: float, sl_level: float | None) -> None:
        """SELL an ATM CALL the instant the trigger fires -- sell-only, so
        signal_dir is always SHORT (see OptionsExecutor._option_type_for:
        sell-side SHORT -> CALL). ``entry_trigger`` is the exact BTC price
        the strategy fired at, used directly as the ATM reference (see
        OptionsExecutor._calc_strike, which needs DELTA_OPTION_OFFSET=0
        for this to be pure ATM rather than ITM)."""
        if self._entry_in_progress or self.executor.has_open_position or self._entry_premium is not None:
            return
        self._entry_in_progress = True
        try:
            try:
                fill = await self.executor.open_option(SignalDir.SHORT.value, entry_trigger)
            except OptionsMarginError as exc:
                log.error("RangeFade: margin/balance error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Balance: {exc}")
                self.strategy.force_flat()
                return
            except Exception as exc:  # noqa: BLE001
                log.error("RangeFade: open_option failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                self.strategy.force_flat()
                return
            if fill is None:
                log.warning("RangeFade: no option fill — flattening to stay in sync")
                self.strategy.force_flat()
                return

            self._entry_premium = fill
            symbol = self.executor.tracked_symbol
            if self.settings.state_file:
                position_state.save(
                    self.settings.state_file, symbol=symbol or "",
                    product_id=self.executor.tracked_product_id,
                    size=self.executor.tracked_size, entry_premium=fill,
                    direction=SignalDir.SHORT.value,
                )
            log.info("RangeFade entry", extra={"extra": {
                "direction": "CALL", "symbol": symbol, "fill": fill,
                "entry_trigger": entry_trigger, "sl_level": sl_level}})
            await self.notifier.notify(
                NotifyEvent.ENTRY_SHORT, direction="CALL", contract=symbol or "?",
                premium=fill, btc_price=entry_trigger, sl_level=sl_level, side="sell",
            )
        finally:
            self._entry_in_progress = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile the open option with the exchange -- looks for a
        SHORT position (size < 0). Sell-only, single leg, so no CE/PE
        disambiguation is needed (unlike supertrend's dual-leg reconcile)
        -- this bot only ever sells calls."""
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
                log.error("RangeFade reconcile: fetch failed",
                          extra={"extra": {"error": str(exc), "attempt": attempt}})
                positions = []
            shorts = [p for p in positions if p["size"] < 0]
            if shorts or not believe_owned:
                break
            log.warning("RangeFade reconcile: expected a position but fetch is empty — retrying",
                        extra={"extra": {"owned": owned_symbol, "attempt": attempt}})
            await asyncio.sleep(1.5)

        if shorts:
            match = next((p for p in shorts if p.get("symbol") == owned_symbol), shorts[0])
            if saved and match.get("symbol") == owned_symbol:
                self._entry_premium = saved.get("entry_premium")
            opt_type = OptionType.CALL if str(match.get("symbol", "")).startswith("C-") else OptionType.PUT
            self.executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
            log.info("RangeFade reconcile: adopted open short",
                     extra={"extra": {"symbol": match["symbol"]}})
            return

        if believe_owned:
            if not self.executor.has_open_position and saved and saved.get("product_id"):
                self._entry_premium = saved.get("entry_premium")
                opt_type = OptionType.CALL if str(owned_symbol).startswith("C-") else OptionType.PUT
                self.executor.adopt(int(saved["product_id"]), int(saved.get("size") or 0),
                                    opt_type, owned_symbol)
            log.warning("RangeFade reconcile: position not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"owned": owned_symbol}})
            return

        self.executor.clear()
        self._entry_premium = None
        self.strategy.force_flat()
        self._closing = False
        log.info("RangeFade reconcile: no owned position — state FLAT")

    # ------------------------------------------------------------------ #
    # Daily 17:25 square-off -- plain close, no rollover.
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
            log.info("RangeFade: next 17:25 square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off()
            except Exception as exc:  # noqa: BLE001
                log.error("RangeFade: square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off(self) -> None:
        now = datetime.now(_IST)
        self._sq_off_date = now.date()
        log.info("RangeFade: 17:25 square-off firing", extra={"extra": {"date": str(self._sq_off_date)}})
        if self.executor.has_open_position:
            await self._close_leg("EOD", now.timestamp())
        # No rollover concept -- a closed setup just ends; force_flat() lets a
        # fresh pattern re-arm from scratch (also harmless no-op if already flat).
        self.strategy.force_flat()
