"""Supertrend Stop-And-Reverse (SAR) live trading engine (option SELL).

Runs SupertrendSarStrategy on 1-MINUTE BTC candles, SELLING options: the
strategy's own entry_is_short flag picks the side directly (True -> sell a
CALL, False -> sell a PUT) -- see src/deltabot/strategy/supertrend_sar.py's
module docstring for the full v1-v5 rule history. This is the winning
config found by comparing against the ORIGINAL 5-minute/05:35-start/TP-50%
baseline across matched 1wk/1mo/3mo windows: "1m candles + sell +
premium~1400 + TP 70%->roll + min-SL 1.0xATR + evening restart, with the
day's-first-entry time moved to 17:35 (i.e. right next to the 17:30
evening-restart trigger, effectively collapsing both into one entry
window)" -- validated in scripts/backtest_supertrend_sar.py:
  3mo: +$331.42 net / 492 legs at 10 lots, 55.7% win rate ($11.05/lot/mo)
  1mo: +$183.38 net / 179 legs at 10 lots, 59.8% win rate ($18.34/lot/mo)
  1wk: +$73.21 net / 37 legs at 10 lots, 70.3% win rate ($7.32/lot/wk)
vs. the original 5m/05:35/TP-50% baseline's 3mo figure of only
+$150.22/324 legs at 10 lots, 48.8% win rate ($5.01/lot/mo) -- roughly 2x
better per lot per month, and the ONLY config whose win rate stayed above
50% across all three windows. See config.py's sar_* block for the full
before/after comparison table this decision was based on.

ARCHITECTURE: unlike SupertrendFixedSlEngine (strategy="supertrend"), this
strategy is STRICT SINGLE POSITION -- a CE and a PE are never open at the
same time (closing one and opening the other happens atomically within the
same closed-bar Decision). So this engine uses ONE OptionsExecutor, the
same shape as Ema21BreakdownEngine (dcv3_trader.py's pattern), not two.

TWO DISTINCT "reopen" PATHS, both landing on the SAME _open_entry():
  * A normal Decision from strategy.update() -- the day's first entry, a
    stop-and-reverse (SL hit -> new leg, same Decision), or the v5 evening
    restart. All three are indistinguishable to this engine: just another
    dec.has_entry with a direction (dec.entry_is_short) and an SL level for
    the notify (dec.sl_level). The strategy itself decides which of the
    three it is.
  * The TP-roll (sar_tp_pct, default 50%): a PURELY option-level mechanic
    this engine drives on its own via a mark-price poll (mirrors
    Ema21BreakdownEngine's own rally-TP poll, but the SELL-side mirror:
    closes on DECAY to sar_tp_pct% of the entry premium, not a rally).
    strategy.update() is never told about a roll -- its own _active_sl/
    _is_short/session state is completely untouched by one, exactly
    matching scripts/backtest_supertrend_sar.py's own --tp-pct design.
    Reopens the SAME direction, tracked here via self._current_is_short
    (not re-derived from the strategy).

ASAP (v6) SL/REVERSAL, CLOSED-BAR ENTRIES: the frozen SL is checked against
REAL price on every forming-candle WS tick (_handle_forming_candle), firing
the stop-and-reverse the instant it's touched instead of waiting up to a
full 1-minute bar for the close -- same established pattern as
HeikinAshiEngine's own intracandle SL/trail check
(strategy.check_intracandle_sl() is a pure check; strategy.
apply_intracandle_reversal() mutates state only after this engine has
confirmed the close actually succeeded, mirroring HeikinAshi's
check-then-notify_exit ordering). The closed-bar path
(_handle_closed_candle) still runs every bar as a FALLBACK -- normally a
no-op, since by the time the bar closes the intracandle path has already
moved the strategy's SL past whatever touched it; it only actually fires
when the ASAP path was skipped (e.g. a WS gap/reconnect). Entries (the
day's first entry, the v5 evening restart) are NOT ASAP -- both need the
closing candle's own color/close-vs-open, unknowable before the bar
actually closes, so they stay on the closed-bar path exactly as before.
Timing (sar_start_hour/minute, sar_restart_hour/minute) is expressed in
terms of a candle's start_time, so the actual live/backtest fill for THOSE
still lands on the candle that CLOSES after the configured threshold (one
bar's lag -- see the sar_restart_hour/minute comment in config.py).

Square-off is wall-clock driven (a wait-until scheduler, not candle-driven,
so it fires at the exact configured second rather than lagging up to one
bar) -- same as every other bot here. The v5 evening restart needs NO
special handling in this engine at all: it's just another entry Decision
from the strategy, arriving naturally on the first closed candle after
sar_restart_hour:minute.

HONESTY: never executed a real order -- start cautious and watch closely.

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
from ..enums import NotifyEvent, OptionType, SignalDir
from ..exchange.rest_client import RestClient
from ..exchange.ws_manager import WebSocketManager
from ..logging_setup import get_logger
from ..models import Candle
from ..strategy.supertrend_sar import SupertrendSarStrategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 60  # 1 minute -- see module docstring for why this beat 5m

log = get_logger(__name__)


class SupertrendSarEngine:
    """Live engine wired to SupertrendSarStrategy, executed via SELLING
    options. See module docstring for the two distinct reopen paths."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = SupertrendSarStrategy(
            atr_period=settings.sar_atr_period, factor=settings.sar_factor,
            day_tz=settings.day_tz,
            start_hour=settings.sar_start_hour, start_minute=settings.sar_start_minute,
            reset_hour=settings.sar_reset_hour, reset_minute=settings.sar_reset_minute,
            min_sl_atr_mult=settings.sar_min_sl_atr_mult,
            restart_hour=settings.sar_restart_hour, restart_minute=settings.sar_restart_minute,
        )
        self.executor = OptionsExecutor(rest, settings)
        self.aggregator = CandleAggregator(
            on_closed=self._on_closed_candle, on_forming=self._on_forming_candle
        )
        self.ws: WebSocketManager | None = None
        self._last_closed_start: int | None = None
        self._last_btc_close: float | None = None
        self._tasks: set[asyncio.Task] = set()
        self._sq_off_task: asyncio.Task | None = None
        self._tp_poll_task: asyncio.Task | None = None
        self._sq_off_date: date | None = None

        self._entry_premium: float | None = None
        self._tp_price: float | None = None   # SELL side: decay TARGET (below entry), not a rally target
        self._current_is_short: bool | None = None
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
            resolution="1m",
            api_key=self.settings.api_key.get_secret_value() or None,
            api_secret=self.settings.api_secret.get_secret_value() or None,
            on_candle=self.aggregator.ingest,
            on_reconnect=self._on_reconnect,
            heartbeat_timeout_s=self.settings.heartbeat_timeout_s,
        )
        self._sq_off_task = asyncio.create_task(self._square_off_scheduler())
        if self.settings.sar_tp_poll_seconds > 0:
            self._tp_poll_task = asyncio.create_task(self._tp_poll_loop())
        log.info("SupertrendSarEngine: starting live (SELL side)")
        await self.ws.run()

    async def stop(self) -> None:
        if self.ws:
            self.ws.stop()
        for t in (self._sq_off_task, self._tp_poll_task):
            if t is not None:
                t.cancel()
        if self.settings.close_on_shutdown and self.executor.has_open_position:
            try:
                lots = self.executor.tracked_size   # captured BEFORE close_option() clears tracked state
                await self.executor.close_option()
                if self.settings.state_file:
                    position_state.clear(self.settings.state_file)
                await self.notifier.notify(NotifyEvent.EXIT, reason="shutdown", size=lots, side="sell")
                log.info("SAR: closed option on shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("SAR: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        bars_needed = max(
            self.settings.warmup_candles + self.settings.sar_atr_period + 50,
            self.settings.warmup_days * 86400 // _BAR_SECONDS,
        )
        start = last_closed_end - bars_needed * _BAR_SECONDS
        candles = await self._fetch_history_paged(start, last_closed_end)
        current_bar = (now // _BAR_SECONDS) * _BAR_SECONDS
        closed = [c for c in candles if c.start_time < current_bar]
        for c in closed:
            self.strategy.update(c)
            if c.close:
                self._last_btc_close = c.close
        if closed:
            self._last_closed_start = closed[-1].start_time
        log.info("SAR warmup done",
                 extra={"extra": {"candles": len(closed), "ready": self.strategy.ready}})

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
            log.warning("SAR: candle gap detected — re-seeding")
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
                log.warning("SAR: candle gap — re-seeding")
                await self._warmup()
        self._last_closed_start = candle.start_time
        self._last_btc_close = candle.close

        dec = self.strategy.update(candle)

        if self.settings.sar_debug_state:
            log.info("SAR state", extra={"extra": {
                "candle": candle.start_time, "o": candle.open, "h": candle.high,
                "l": candle.low, "c": candle.close, "blocked": self._entries_blocked(),
                "has_option": self.executor.has_open_position, **self.strategy.debug_state()}})

        # 1. Closed-bar exit: the frozen SL was hit. FALLBACK -- v6's own
        #    ASAP intracandle path (_handle_forming_candle) normally
        #    already caught this touch in real time; this only actually
        #    fires when it didn't (e.g. a gap/reconnect swallowed the WS
        #    ticks). The SAME Decision may also carry a reversal entry --
        #    handled in step 2 right after, same call, same bar (executor
        #    is flat again by then).
        if dec is not None and dec.has_exit and self.executor.has_open_position:
            await self._close_leg("SL", btc_exit_price=dec.exit_price)

        # 2. Closed-bar entry: the day's first entry, a stop-and-reverse, or
        #    the v5 evening restart -- all indistinguishable here, just
        #    another Decision with a direction. (_entries_blocked() only
        #    gates the strategy's OWN first-entry/restart branches in
        #    principle -- a reversal must never be blocked by the weekday
        #    filter, since it's just closing out an already-open risk, not
        #    opening new exposure. Weekday gating is therefore checked
        #    against whether this is a reversal via dec.has_exit.)
        if (dec is not None and dec.has_entry and not self.executor.has_open_position
                and (dec.has_exit or not self._entries_blocked())):
            await self._open_entry(dec.entry_is_short, dec.sl_level, candle.close)

    # ------------------------------------------------------------------ #
    def _on_forming_candle(self, candle: Candle) -> None:
        task = asyncio.create_task(self._handle_forming_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_forming_candle(self, candle: Candle) -> None:
        """v6 ASAP: fires the SL/reversal the instant REAL price crosses the
        frozen SL, instead of waiting for the 1-minute bar to close. Checks
        both the forming candle's running low AND high (either could have
        touched it at different points within the still-building bar) --
        same pattern as HeikinAshiEngine's own intracandle SL check. The
        entry side (day's-first-entry, evening restart) stays closed-bar
        only -- both need the closing candle's own color, which can't be
        known before the bar actually closes; see the strategy's own
        module docstring."""
        if not self.strategy.ready or not self.executor.has_open_position:
            return
        if self._closing or self._entry_in_progress:
            return   # a close/open is already mid-flight (TP-roll, or we already beat ourselves to it)
        for price in (candle.low, candle.high):
            hit, level = self.strategy.check_intracandle_sl(price)
            if not hit:
                continue
            log.info("SAR: intracandle SL touched", extra={"extra": {"price": price, "sl": level}})
            await self._close_leg("SL", btc_exit_price=level if level is not None else price)
            if self.executor.has_open_position:
                return   # close failed (still tracked) -- do NOT reverse; retry next tick/closed-bar fallback
            new_is_short, new_sl = self.strategy.apply_intracandle_reversal(price)
            await self._open_entry(new_is_short, new_sl, price)
            return

    # ------------------------------------------------------------------ #
    async def _tp_poll_loop(self) -> None:
        interval = self.settings.sar_tp_poll_seconds
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
                log.warning("SAR: TP-poll mark fetch failed", extra={"extra": {"error": str(exc)}})
                continue
            if mark is not None and self._tp_price is not None and mark <= self._tp_price:
                log.info("SAR: TP-roll hit (poll)",
                         extra={"extra": {"mark": mark, "tp": self._tp_price}})
                await self._close_and_roll(mark)

    # ------------------------------------------------------------------ #
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
            log.warning("SAR: position-verify fetch failed", extra={"extra": {"error": str(exc)}})
            return
        if any(p["size"] < 0 and p.get("product_id") == tracked for p in positions):
            self._verify_misses = 0
            return
        self._verify_misses += 1
        if self._verify_misses < 2:
            log.warning("SAR: tracked position not on exchange (1st miss) — rechecking",
                        extra={"extra": {"contract": self.executor.tracked_symbol}})
            return
        contract = self.executor.tracked_symbol
        lots = self.executor.tracked_size   # captured BEFORE executor.clear() wipes tracked state
        log.warning("SAR: position closed OUTSIDE the bot — self-healing to FLAT",
                    extra={"extra": {"contract": contract}})
        self.executor.clear()
        if self.settings.state_file:
            position_state.clear(self.settings.state_file)
        self._entry_premium = self._tp_price = self._current_is_short = None
        self._verify_misses = 0
        self.strategy.force_flat()
        await self.notifier.notify(
            NotifyEvent.EXIT, reason="closed outside the bot (self-healed)",
            contract=contract or "?", size=lots, side="sell",
        )

    # ------------------------------------------------------------------ #
    def _pnl(self, entry_prem: float | None, exit_prem: float | None, lots: int) -> float:
        # SELL side: profit when premium DECAYS (entry - exit), the mirror
        # of a buy-side engine's (exit - entry).
        if entry_prem is None or exit_prem is None:
            return 0.0
        return (entry_prem - exit_prem) * lots * 0.001

    async def _close_and_roll(self, mark: float) -> None:
        """TP-roll: book the profit, then IMMEDIATELY resell a fresh
        contract in the SAME direction at target_premium again -- purely an
        option-level mechanic, strategy.update() is never told about this
        (its own _active_sl/_is_short/session state is untouched), same as
        scripts/backtest_supertrend_sar.py's own --tp-pct design."""
        if self._closing or not self.executor.has_open_position:
            return
        was_short = self._current_is_short
        self._closing = True
        try:
            contract = self.executor.tracked_symbol
            lots = self.executor.tracked_size   # captured BEFORE close_option() clears tracked state
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("SAR: TP-roll close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"TP-roll close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            exit_prem = fill if fill is not None else mark
            entry_prem = self._entry_premium
            gross = self._pnl(entry_prem, exit_prem, lots)
            self._entry_premium = self._tp_price = self._current_is_short = None
            log.info("SAR TP-roll hit", extra={"extra": {"contract": contract, "exit_prem": exit_prem}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason="TP", contract=contract or "?",
                entry_premium=entry_prem, exit_premium=exit_prem,
                pnl=round(gross, 2), size=lots, side="sell",
            )
        finally:
            self._closing = False
        if was_short is not None:
            await self._open_entry(was_short, sl_level=None, btc_price=self._last_btc_close or 0.0)

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
                log.error("SAR: leg close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            entry_prem = self._entry_premium
            gross = self._pnl(entry_prem, fill, lots)
            self._entry_premium = self._tp_price = self._current_is_short = None
            log.info("SAR exit", extra={"extra": {
                "reason": reason, "contract": contract, "btc_exit": btc_exit_price}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                entry_premium=entry_prem, exit_premium=fill,
                pnl=round(gross, 2), size=lots, side="sell",
            )
        finally:
            self._closing = False

    async def _open_entry(self, is_short: bool, sl_level: float | None, btc_price: float) -> None:
        """SELL the option for a new/reversed/restarted/rolled position.
        entry_is_short=True -> sell a CALL (CE); False -> sell a PUT (PE) --
        the OptionsExecutor's sell-side _option_type_for maps SignalDir.SHORT
        -> CALL, SignalDir.LONG -> PUT."""
        if self._entry_in_progress or self.executor.has_open_position:
            return
        self._entry_in_progress = True
        try:
            signal_dir = SignalDir.SHORT.value if is_short else SignalDir.LONG.value
            try:
                fill, symbol = await self.executor.open_option_by_premium(
                    signal_dir, self.settings.target_premium
                )
            except OptionsMarginError as exc:
                log.error("SAR: margin/balance error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Balance: {exc}")
                self.strategy.force_flat()
                return
            except Exception as exc:  # noqa: BLE001
                log.error("SAR: open_option_by_premium failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                self.strategy.force_flat()
                return
            if fill is None:
                log.warning("SAR: no option fill — flattening to stay in sync")
                self.strategy.force_flat()
                return

            self._entry_premium = fill
            # sar_tp_pct<=0 means "no TP-roll at all" (pure SAR) -- mirrors
            # ema21's own take_profit_pct<=0 convention.
            tp_frac = self.settings.sar_tp_pct / 100.0 if self.settings.sar_tp_pct > 0 else None
            self._tp_price = fill * tp_frac if tp_frac is not None else None
            self._current_is_short = is_short
            if self.settings.state_file:
                position_state.save(
                    self.settings.state_file, symbol=symbol or "",
                    product_id=self.executor.tracked_product_id,
                    size=self.executor.tracked_size, entry_premium=fill,
                    tp_price=self._tp_price, is_short=is_short,
                )
            tp_display = round(self._tp_price, 1) if self._tp_price is not None else None
            direction = "CALL" if is_short else "PUT"
            log.info("SAR entry", extra={"extra": {
                "direction": direction, "symbol": symbol, "fill": fill,
                "tp_price": tp_display, "sl_level": sl_level}})
            event = NotifyEvent.ENTRY_SHORT if is_short else NotifyEvent.ENTRY_LONG
            await self.notifier.notify(
                event, direction=direction, contract=symbol or "?",
                premium=fill, btc_price=btc_price, sl_level=sl_level,
                tp_price=tp_display, side="sell",
            )
        finally:
            self._entry_in_progress = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile the open option with the exchange -- looks for a SHORT
        position (size < 0), same as every other sell-mode single-leg bot."""
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
                log.error("SAR reconcile: fetch failed",
                          extra={"extra": {"error": str(exc), "attempt": attempt}})
                positions = []
            shorts = [p for p in positions if p["size"] < 0]
            if shorts or not believe_owned:
                break
            log.warning("SAR reconcile: expected a position but fetch is empty — retrying",
                        extra={"extra": {"owned": owned_symbol, "attempt": attempt}})
            await asyncio.sleep(1.5)

        if shorts:
            match = next((p for p in shorts if p.get("symbol") == owned_symbol), shorts[0])
            if saved and match.get("symbol") == owned_symbol:
                self._entry_premium = saved.get("entry_premium")
                self._tp_price = saved.get("tp_price")
                self._current_is_short = saved.get("is_short")
            opt_type = OptionType.CALL if match["symbol"].startswith("C-") else OptionType.PUT
            self.executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
            log.info("SAR reconcile: adopted open short",
                     extra={"extra": {"symbol": match["symbol"]}})
            return

        if believe_owned:
            if not self.executor.has_open_position and saved and saved.get("product_id"):
                self._entry_premium = saved.get("entry_premium")
                self._tp_price = saved.get("tp_price")
                self._current_is_short = saved.get("is_short")
                opt_type = OptionType.CALL if str(owned_symbol).startswith("C-") else OptionType.PUT
                self.executor.adopt(int(saved["product_id"]), int(saved.get("size") or 0),
                                    opt_type, owned_symbol)
            log.warning("SAR reconcile: position not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"owned": owned_symbol}})
            return

        self.executor.clear()
        self._entry_premium = self._tp_price = self._current_is_short = None
        if self.strategy.in_position:
            self.strategy.force_flat()
        self._closing = False
        log.info("SAR reconcile: no owned position — state FLAT")

    # ------------------------------------------------------------------ #
    # Daily 17:25 square-off -- plain close, then the v5 evening restart
    # naturally re-arms via the NEXT closed candle's Decision, no special
    # handling needed here.
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
            log.info("SAR: next 17:25 square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off()
            except Exception as exc:  # noqa: BLE001
                log.error("SAR: square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off(self) -> None:
        now = datetime.now(_IST)
        self._sq_off_date = now.date()
        log.info("SAR: 17:25 square-off firing", extra={"extra": {"date": str(self._sq_off_date)}})
        if self.executor.has_open_position:
            try:
                contract = self.executor.tracked_symbol
                lots = self.executor.tracked_size   # captured BEFORE close_option() clears tracked state
                fill = await self.executor.close_option()
                if self.settings.state_file:
                    position_state.clear(self.settings.state_file)
                entry_prem = self._entry_premium
                gross = self._pnl(entry_prem, fill, lots)
                self._entry_premium = self._tp_price = self._current_is_short = None
                await self.notifier.notify(
                    NotifyEvent.EXIT, reason="EOD", contract=contract or "?",
                    entry_premium=entry_prem, exit_premium=fill, pnl=round(gross, 2), size=lots,
                    side="sell",
                )
            except Exception as exc:  # noqa: BLE001
                log.error("SAR: square-off close failed", extra={"extra": {"error": str(exc)}})
                await self._sync_options_to_exchange()
                return
        # force_flat() leaves _active_session/_active_restart_session/day-
        # high/day-low untouched (see the strategy's own docstring) -- the
        # v5 evening restart re-arms on its own via the next Decision.
        self.strategy.force_flat()
