"""Straddle4pm live trading engine (option BUY, dual independent legs).

Runs StraddleStrategy, which has NO BTC-price signal at all -- it only
tracks a fixed daily entry TIME (default 16:00 IST). On that trigger this
engine BUYS a CALL and a PUT near ``target_premium`` (e.g. ~100 each) as two
independent legs, same "dual OptionsExecutor" pattern as
supertrend_trader.py (see that module's own docstring for why: both legs
can be open simultaneously as genuinely independent contracts, so a single
shared OptionsExecutor's single-leg semantics don't fit).

EXIT: whichever leg's premium first reaches ``settings.straddle_exit_target``
(a FIXED ABSOLUTE value, e.g. 250 -- NOT a % of entry, unlike every other TP
in this repo) is closed at that mark, and the OTHER leg is closed alongside
it immediately at whatever it's currently worth -- once one side has paid
off, the straddle's whole thesis is resolved. If NEITHER leg reaches
target, the normal square_off_hour/minute closes both. Neither leg has its
own SL -- max loss per leg is capped at the premium paid (buying, not
selling), same risk profile as ema21bot's buy side.

DIFFERENCES FROM supertrend_trader.py (which this otherwise mirrors --
closed-bar only, no intracandle, no rollover):
  * ENTRY IS ONE EVENT, NOT TWO INDEPENDENT SIGNALS -- supertrend's CE/PE
    legs open on separate, independent flips; here both legs open TOGETHER
    off the SAME daily trigger. A failure opening either leg unfires
    today's trigger (via strategy.unfire_today()) so a LATER candle still
    inside the grace window can retry BOTH legs from scratch (never just
    one -- a lone leg is not a straddle).
  * NO SL, NO TREND EXIT -- the only exits are the shared TP poll and the
    daily square-off.
  * WARMUP CAN FALSELY "FIRE" TODAY -- replaying today's own entry-window
    candle through strategy.update() during warmup (e.g. after a restart
    inside the grace window) marks it fired even though nothing was ever
    bought. Reconcile corrects this: if neither leg is actually open after
    reconciling with the exchange, it calls strategy.unfire_today() so a
    later candle can genuinely retry.
  * RECONCILE must disambiguate CALL vs PUT by SYMBOL PREFIX, not sign --
    both legs are LONG (size > 0) on the exchange (buy side), so sign alone
    only tells "ours vs. some other bot's SELL", not which of our own two
    legs is which.

Runs as its own Docker container on a SEPARATE sub-account; position
ownership is tracked via TWO state files (one per leg), both derived from
``DELTA_STATE_FILE``. Never touches any other bot.

HONESTY: this strategy has NEVER executed a real order. Treat live results
with real caution for the first several weeks.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ..config import Settings
from ..enums import NotifyEvent, OptionType, SignalDir
from ..exchange.rest_client import RestClient
from ..exchange.ws_manager import WebSocketManager
from ..logging_setup import get_logger
from ..models import Candle
from ..strategy.straddle import StraddleStrategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 300  # 5 minutes

log = get_logger(__name__)


def _leg_path(base: str, suffix: str) -> str:
    """``state/straddle_pos.json`` -> ``state/straddle_pos_call.json``.
    Empty base (state persistence off) stays empty for both legs."""
    if not base:
        return ""
    p = Path(base)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


class _Leg:
    """Per-leg tracking state -- one instance for the CALL leg, one for the
    PUT leg."""

    def __init__(self, name: str, executor: OptionsExecutor, state_file: str) -> None:
        self.name = name                       # "call" or "put"
        self.executor = executor
        self.state_file = state_file
        self.entry_premium: float | None = None
        self.entry_in_progress = False
        self.closing = False
        self.verify_misses = 0
        self.last_verify = 0.0


class StraddleEngine:
    """Live engine wired to StraddleStrategy, buying a CALL and a PUT
    together at a fixed daily time and tracking them as independent legs.
    See module docstring."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = StraddleStrategy(
            entry_hour=settings.straddle_entry_hour,
            entry_minute=settings.straddle_entry_minute,
            entry_grace_minutes=settings.straddle_entry_grace_minutes,
            day_tz=settings.day_tz,
        )
        self.executor_call = OptionsExecutor(rest, settings)
        self.executor_put = OptionsExecutor(rest, settings)
        self.call = _Leg("call", self.executor_call, _leg_path(settings.state_file, "call"))
        self.put = _Leg("put", self.executor_put, _leg_path(settings.state_file, "put"))

        # No on_forming -- closed-bar only, see module docstring.
        self.aggregator = CandleAggregator(on_closed=self._on_closed_candle)
        self.ws: WebSocketManager | None = None
        self._last_closed_start: int | None = None
        self._tasks: set[asyncio.Task] = set()
        self._sq_off_task: asyncio.Task | None = None
        self._verify_task: asyncio.Task | None = None
        self._tp_poll_task: asyncio.Task | None = None
        self._sq_off_date: date | None = None
        self._entering_both = False   # guards against the TP-poll racing a same-tick entry

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
        if self.settings.position_verify_seconds > 0:
            self._verify_task = asyncio.create_task(self._verify_loop())
        if self.settings.straddle_tp_poll_seconds > 0:
            self._tp_poll_task = asyncio.create_task(self._tp_poll_loop())
        log.info("StraddleEngine: starting live (dual-leg BUY)")
        await self.ws.run()

    async def stop(self) -> None:
        if self.ws:
            self.ws.stop()
        for t in (self._sq_off_task, self._verify_task, self._tp_poll_task):
            if t is not None:
                t.cancel()
        if self.settings.close_on_shutdown:
            for leg in (self.call, self.put):
                if leg.executor.has_open_position:
                    try:
                        await self._close_leg(leg, "shutdown")
                    except Exception as exc:  # noqa: BLE001
                        log.error("Straddle: failed to close on shutdown",
                                 extra={"extra": {"leg": leg.name, "error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        bars_needed = max(self.settings.warmup_candles, self.settings.warmup_days * 86400 // _BAR_SECONDS)
        start = last_closed_end - bars_needed * _BAR_SECONDS
        candles = await self._fetch_history_paged(start, last_closed_end)
        current_bar = (now // _BAR_SECONDS) * _BAR_SECONDS
        closed = [c for c in candles if c.start_time < current_bar]
        for c in closed:
            self.strategy.update(c)   # may falsely "fire" today -- reconcile corrects this below
        if closed:
            self._last_closed_start = closed[-1].start_time
        log.info("Straddle warmup done", extra={"extra": {"candles": len(closed)}})

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
            log.warning("Straddle: candle gap detected — re-seeding")
            await self._warmup()
            await self._sync_options_to_exchange()

    # ------------------------------------------------------------------ #
    def _on_closed_candle(self, candle: Candle) -> None:
        task = asyncio.create_task(self._handle_closed_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_closed_candle(self, candle: Candle) -> None:
        if self._last_closed_start is not None:
            gap = candle.start_time - self._last_closed_start
            if gap > _BAR_SECONDS:
                log.warning("Straddle: candle gap — re-seeding")
                await self._warmup()
        self._last_closed_start = candle.start_time

        should_fire = self.strategy.update(candle)

        if self.settings.straddle_debug_state:
            log.info("Straddle state", extra={"extra": {
                "candle": candle.start_time, "close": candle.close, "should_fire": should_fire,
                "has_call": self.executor_call.has_open_position,
                "has_put": self.executor_put.has_open_position}})

        # TP check on the closed bar; the poll also runs independently.
        await self._maybe_tp_pair()

        if should_fire and not self.executor_call.has_open_position and not self.executor_put.has_open_position:
            await self._enter_both(candle.close)

    # ------------------------------------------------------------------ #
    async def _enter_both(self, btc_price: float) -> None:
        """Buy the CALL and the PUT together. If EITHER leg fails, close
        whatever DID open (a lone leg is not a straddle) and unfire today
        so a later candle inside the grace window can retry both."""
        if self._entering_both:
            return
        self._entering_both = True
        try:
            call_fill = await self._open_leg(self.call, SignalDir.LONG, btc_price)
            put_fill = await self._open_leg(self.put, SignalDir.SHORT, btc_price)
            if call_fill is None or put_fill is None:
                log.warning("Straddle: one leg failed to open — unwinding the pair",
                           extra={"extra": {"call_fill": call_fill, "put_fill": put_fill}})
                if call_fill is not None:
                    await self._close_leg(self.call, "PARTIAL_FAIL")
                if put_fill is not None:
                    await self._close_leg(self.put, "PARTIAL_FAIL")
                self.strategy.unfire_today()
        finally:
            self._entering_both = False

    async def _open_leg(self, leg: _Leg, signal_dir: int, btc_price: float) -> float | None:
        """BUY the option for this leg. LONG -> buy CALL, SHORT -> buy PUT
        (OptionsExecutor's BUY-side mapping, see options_executor.py)."""
        if leg.entry_in_progress or leg.executor.has_open_position or leg.entry_premium is not None:
            return leg.entry_premium
        leg.entry_in_progress = True
        try:
            try:
                fill, symbol = await leg.executor.open_option_by_premium(
                    signal_dir, self.settings.target_premium
                )
            except OptionsMarginError as exc:
                log.error("Straddle: margin/balance error",
                         extra={"extra": {"leg": leg.name, "error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Balance: {exc}")
                return None
            except Exception as exc:  # noqa: BLE001
                log.error("Straddle: open_option_by_premium failed",
                         extra={"extra": {"leg": leg.name, "error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                return None
            if fill is None:
                log.warning("Straddle: no option fill", extra={"extra": {"leg": leg.name}})
                return None

            leg.entry_premium = fill
            if leg.state_file:
                position_state.save(
                    leg.state_file, symbol=symbol or "",
                    product_id=leg.executor.tracked_product_id,
                    size=self.settings.option_contracts, entry_premium=fill,
                    direction=signal_dir,
                )
            direction = "CALL" if signal_dir == SignalDir.LONG else "PUT"
            log.info("Straddle entry", extra={"extra": {
                "leg": leg.name, "direction": direction, "symbol": symbol, "fill": fill}})
            event = NotifyEvent.ENTRY_LONG if signal_dir == SignalDir.LONG else NotifyEvent.ENTRY_SHORT
            await self.notifier.notify(
                event, direction=direction, contract=symbol or "?",
                premium=fill, btc_price=btc_price,
                tp_price=self.settings.straddle_exit_target, side="buy",
            )
            return fill
        finally:
            leg.entry_in_progress = False

    # ------------------------------------------------------------------ #
    async def _tp_poll_loop(self) -> None:
        interval = self.settings.straddle_tp_poll_seconds
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            await self._maybe_tp_pair()

    async def _maybe_tp_pair(self) -> None:
        """If EITHER leg's mark has reached straddle_exit_target, close BOTH
        legs -- the straddle's thesis is resolved the instant one side pays
        off, regardless of what the other side is worth right now."""
        target = self.settings.straddle_exit_target
        for leg in (self.call, self.put):
            if leg.closing or not leg.executor.has_open_position:
                continue
            symbol = leg.executor.tracked_symbol
            if not symbol:
                continue
            try:
                mark = await asyncio.to_thread(self.rest.get_mark_price, symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning("Straddle: TP mark fetch failed",
                           extra={"extra": {"leg": leg.name, "error": str(exc)}})
                continue
            if mark is not None and mark >= target:
                log.info("Straddle: target hit — closing both legs",
                         extra={"extra": {"winning_leg": leg.name, "mark": mark, "target": target}})
                other = self.put if leg is self.call else self.call
                await self._close_leg(leg, "TARGET")
                if other.executor.has_open_position:
                    await self._close_leg(other, "PAIR")
                return

    # ------------------------------------------------------------------ #
    async def _verify_loop(self) -> None:
        interval = self.settings.position_verify_seconds
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            await self._maybe_verify_leg(self.call)
            await self._maybe_verify_leg(self.put)

    async def _maybe_verify_leg(self, leg: _Leg) -> None:
        """Self-heal: confirm the tracked LONG option for this leg still
        exists on the exchange (both legs are long -- sign alone can't
        disambiguate CALL from PUT, so this only checks the tracked
        product_id)."""
        iv = self.settings.position_verify_seconds
        if iv <= 0 or leg.closing or leg.entry_in_progress or not leg.executor.has_open_position:
            leg.verify_misses = 0
            return
        now = time.time()
        if now - leg.last_verify < iv:
            return
        leg.last_verify = now
        tracked = leg.executor.tracked_product_id
        try:
            positions = await asyncio.to_thread(
                self.rest.get_option_positions, leg.executor.underlying
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Straddle: position-verify fetch failed",
                       extra={"extra": {"leg": leg.name, "error": str(exc)}})
            return
        if any(p["size"] > 0 and p.get("product_id") == tracked for p in positions):
            leg.verify_misses = 0
            return
        leg.verify_misses += 1
        if leg.verify_misses < 2:
            log.warning("Straddle: tracked leg not on exchange (1st miss) — rechecking",
                        extra={"extra": {"leg": leg.name, "contract": leg.executor.tracked_symbol}})
            return
        contract = leg.executor.tracked_symbol
        log.warning("Straddle: leg closed OUTSIDE the bot — self-healing to FLAT",
                    extra={"extra": {"leg": leg.name, "contract": contract}})
        leg.executor.clear()
        if leg.state_file:
            position_state.clear(leg.state_file)
        leg.entry_premium = None
        leg.verify_misses = 0
        await self.notifier.notify(
            NotifyEvent.EXIT, reason="closed outside the bot (self-healed)",
            contract=contract or "?", size=self.settings.option_contracts,
            side=f"buy {'CE' if leg.name == 'call' else 'PE'}",
        )

    # ------------------------------------------------------------------ #
    async def _close_leg(self, leg: _Leg, reason: str) -> None:
        if leg.closing or not leg.executor.has_open_position:
            return
        leg.closing = True
        try:
            contract = leg.executor.tracked_symbol
            try:
                fill = await leg.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("Straddle: leg close failed",
                         extra={"extra": {"leg": leg.name, "error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close: {exc}")
                return
            if leg.state_file:
                position_state.clear(leg.state_file)
            entry_prem = leg.entry_premium
            lots = self.settings.option_contracts
            # BUY side: profit when the premium ROSE (exit - entry).
            gross = ((fill - entry_prem) * lots * 0.001
                     if (entry_prem is not None and fill is not None) else 0.0)
            leg.entry_premium = None
            side_label = "buy CE" if leg.name == "call" else "buy PE"
            log.info("Straddle exit", extra={"extra": {
                "leg": leg.name, "reason": reason, "contract": contract, "fill": fill}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                entry_premium=entry_prem, exit_premium=fill,
                pnl=round(gross, 2), size=lots, side=side_label,
            )
        finally:
            leg.closing = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile BOTH legs with the exchange. Both are long (size > 0)
        on the exchange, so disambiguate by symbol prefix (C-/P-), not sign.
        If neither leg turns out to actually be open, unfire today's trigger
        in case warmup's blind candle replay falsely marked it fired (see
        module docstring)."""
        positions: list[dict] = []
        try:
            positions = await asyncio.to_thread(
                self.rest.get_option_positions, self.executor_call.underlying
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Straddle reconcile: fetch failed", extra={"extra": {"error": str(exc)}})

        call_longs = [p for p in positions if p["size"] > 0 and p.get("symbol", "").startswith("C-")]
        put_longs = [p for p in positions if p["size"] > 0 and p.get("symbol", "").startswith("P-")]
        self._reconcile_leg(self.call, call_longs, OptionType.CALL)
        self._reconcile_leg(self.put, put_longs, OptionType.PUT)

        today = datetime.now(_IST).date()
        if (not self.call.executor.has_open_position and not self.put.executor.has_open_position
                and self.strategy._last_fired_date == today):
            log.info("Straddle reconcile: today marked fired but neither leg is open — unfiring")
            self.strategy.unfire_today()

    def _reconcile_leg(self, leg: _Leg, found: list[dict], opt_type: OptionType) -> None:
        saved = position_state.load(leg.state_file) if leg.state_file else None
        owned_symbol = saved.get("symbol") if saved else None
        believe_owned = owned_symbol is not None or leg.executor.has_open_position

        if found:
            match = next((p for p in found if p.get("symbol") == owned_symbol), found[0])
            if saved and match.get("symbol") == owned_symbol:
                leg.entry_premium = saved.get("entry_premium")
            leg.executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
            log.info("Straddle reconcile: adopted open leg",
                     extra={"extra": {"leg": leg.name, "symbol": match["symbol"]}})
            return

        if believe_owned:
            if not leg.executor.has_open_position and saved and saved.get("product_id"):
                leg.entry_premium = saved.get("entry_premium")
                leg.executor.adopt(int(saved["product_id"]), int(saved.get("size") or 0),
                                   opt_type, owned_symbol)
            log.warning("Straddle reconcile: leg not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"leg": leg.name, "owned": owned_symbol}})
            return

        leg.executor.clear()
        leg.entry_premium = None
        leg.closing = False
        log.info("Straddle reconcile: no owned position — leg FLAT",
                 extra={"extra": {"leg": leg.name}})

    # ------------------------------------------------------------------ #
    # Daily 17:25 square-off -- plain close of both legs, no rollover.
    # ------------------------------------------------------------------ #
    async def _square_off_scheduler(self) -> None:
        while True:
            now = datetime.now(_IST)
            target = now.replace(hour=self.settings.square_off_hour,
                                 minute=self.settings.square_off_minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait_s = (target - now).total_seconds()
            log.info("Straddle: next square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off()
            except Exception as exc:  # noqa: BLE001
                log.error("Straddle: square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off(self) -> None:
        now = datetime.now(_IST)
        self._sq_off_date = now.date()
        log.info("Straddle: square-off firing", extra={"extra": {"date": str(self._sq_off_date)}})
        for leg in (self.call, self.put):
            if leg.executor.has_open_position:
                await self._close_leg(leg, "EOD")
