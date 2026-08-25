"""Supertrend Fixed-SL live trading engine (option SELL, dual independent legs).

Runs SupertrendFixedSlStrategy on 5-minute BTC candles: a bearish Supertrend
(10,3) flip SELLS a CALL near ``target_premium``, a bullish flip SELLS a PUT.
SL = Supertrend's own value AT THE FLIP, frozen for the life of that leg --
never re-derived from Supertrend's still-updating value. Exits are that
frozen SL, the daily square-off, or the premium-decay TP (see
``settings.supertrend_take_profit_pct``): once EITHER leg's option premium
decays by that %, that leg is flattened and BOTH legs are blocked from new
entries until ``supertrend_tp_block_hour:minute`` (default 17:30) that same
day -- a fresh flip after that time re-arms normally.

KEY DIFFERENCE FROM EVERY OTHER ENGINE IN THIS REPO (dcv2/dcv3/dchannel/
revbreak/tcp/ema21): a CE (short) leg and a PE (long) leg can be open AT THE
SAME TIME, as genuinely independent contracts (see the strategy module's own
docstring). Every other engine assumes exactly one open leg via a single
OptionsExecutor. This engine uses TWO independent OptionsExecutor instances
-- ``executor_short`` (CE) and ``executor_long`` (PE) -- rather than modify
the shared OptionsExecutor class (which the other 5 live bots depend on for
single-leg semantics). OptionsExecutor has no singleton/shared state, so two
instances against the same rest/settings is safe.

OTHER DIFFERENCES FROM ema21_trader.py (which this otherwise mirrors --
closed-bar only, no intracandle, no rollover, entry gating lives inside the
strategy itself):
  * TP POLL LOOP COVERS BOTH LEGS -- unlike every other bot's single-leg TP
    poll, ``_tp_poll_loop`` here checks the short and long leg's option mark
    independently each tick (either, both, or neither may have a tp_price
    armed at a given moment). A separate self-heal loop still runs
    alongside it (see ``_verify_loop``), unlike ema21bot/dcv2bot/dcv3bot
    which piggyback self-heal on their single TP-poll timer.
  * RECONCILE must disambiguate CE vs PE by SYMBOL PREFIX, not sign -- both
    legs are short (size < 0) on the exchange, so sign alone can't tell them
    apart (every other engine only ever has one leg to find, so sign alone
    suffices there).
  * Always SELL-mode (option_side stays at its default "sell") -- unlike
    ema21/dcv3, this bot never buys, so take_profit_pct/option_side are
    unused config surface here.

HONESTY: this strategy has NEVER executed a real order. Treat live results
with real caution for the first several weeks.

Runs as its own Docker container on a SEPARATE sub-account; position
ownership is tracked via TWO state files (one per leg), both derived from
``DELTA_STATE_FILE``. Never touches any other bot.
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
from ..strategy.supertrend_fixed_sl import SupertrendFixedSlStrategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 300  # 5 minutes

log = get_logger(__name__)


def _leg_path(base: str, suffix: str) -> str:
    """``state/supertrend_pos.json`` -> ``state/supertrend_pos_short.json``.
    Empty base (state persistence off) stays empty for both legs."""
    if not base:
        return ""
    p = Path(base)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


class _Leg:
    """Per-leg tracking state -- one instance for the short/CE leg, one for
    the long/PE leg."""

    def __init__(self, name: str, executor: OptionsExecutor, state_file: str) -> None:
        self.name = name                       # "short" or "long"
        self.executor = executor
        self.state_file = state_file
        self.entry_premium: float | None = None
        self.tp_price: float | None = None
        self.entry_in_progress = False
        self.closing = False
        self.verify_misses = 0
        self.last_verify = 0.0


class SupertrendFixedSlEngine:
    """Live engine wired to SupertrendFixedSlStrategy, tracking a short (CE)
    and long (PE) leg as independent contracts. See module docstring."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = SupertrendFixedSlStrategy(
            atr_period=settings.supertrend_atr_period,
            factor=settings.supertrend_factor,
            day_tz=settings.day_tz,
            gap_start_hour=settings.supertrend_gap_start_hour,
            gap_start_minute=settings.supertrend_gap_start_minute,
            gap_end_hour=settings.supertrend_gap_end_hour,
            gap_end_minute=settings.supertrend_gap_end_minute,
            trade_ce=settings.supertrend_trade_ce,
            trade_pe=settings.supertrend_trade_pe,
            trend_filter=settings.supertrend_trend_filter,
            trend_fast_len=settings.supertrend_trend_fast_len,
            trend_slow_len=settings.supertrend_trend_slow_len,
            trend_flip_exit=settings.supertrend_trend_flip_exit,
        )
        self.executor_short = OptionsExecutor(rest, settings)
        self.executor_long = OptionsExecutor(rest, settings)
        self.short = _Leg("short", self.executor_short, _leg_path(settings.state_file, "short"))
        self.long = _Leg("long", self.executor_long, _leg_path(settings.state_file, "long"))

        # No on_forming -- closed-bar only, see module docstring.
        self.aggregator = CandleAggregator(on_closed=self._on_closed_candle)
        self.ws: WebSocketManager | None = None
        self._last_closed_start: int | None = None
        self._tasks: set[asyncio.Task] = set()
        self._sq_off_task: asyncio.Task | None = None
        self._verify_task: asyncio.Task | None = None
        self._tp_poll_task: asyncio.Task | None = None
        self._sq_off_date: date | None = None
        # Epoch seconds; new entries (both legs) blocked while time.time() is
        # below this. Set whenever EITHER leg's premium-decay TP fires.
        self._block_entries_until: float = 0.0

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
        if self.settings.supertrend_tp_poll_seconds > 0:
            self._tp_poll_task = asyncio.create_task(self._tp_poll_loop())
        log.info("SupertrendFixedSlEngine: starting live (dual-leg SELL)")
        await self.ws.run()

    async def stop(self) -> None:
        if self.ws:
            self.ws.stop()
        for t in (self._sq_off_task, self._verify_task, self._tp_poll_task):
            if t is not None:
                t.cancel()
        if self.settings.close_on_shutdown:
            for leg in (self.short, self.long):
                if leg.executor.has_open_position:
                    try:
                        await self._close_leg(leg, "shutdown", 0.0)
                    except Exception as exc:  # noqa: BLE001
                        log.error("Supertrend: failed to close on shutdown",
                                 extra={"extra": {"leg": leg.name, "error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        bars_needed = max(
            self.settings.warmup_candles + self.settings.supertrend_atr_period + 50,
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
        log.info("Supertrend warmup done",
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
            log.warning("Supertrend: candle gap detected — re-seeding")
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
                log.warning("Supertrend: candle gap — re-seeding")
                await self._warmup()
        self._last_closed_start = candle.start_time

        dec = self.strategy.update(candle)

        if self.settings.supertrend_debug_state:
            log.info("Supertrend state", extra={"extra": {
                "candle": candle.start_time, "o": candle.open, "h": candle.high,
                "l": candle.low, "c": candle.close, "blocked": self._entries_blocked(),
                "has_short": self.executor_short.has_open_position,
                "has_long": self.executor_long.has_open_position, **self.strategy.debug_state()}})

        if dec is None:
            return

        # Exits and entries are INDEPENDENT per leg -- both can legitimately
        # fire on the same bar (e.g. CE stops out while PE enters).
        if dec.short_exit and self.executor_short.has_open_position:
            reason = "TREND" if dec.short_trend_exit else "SL"
            await self._close_leg(self.short, reason, dec.short_exit_price)
        if dec.long_exit and self.executor_long.has_open_position:
            reason = "TREND" if dec.long_trend_exit else "SL"
            await self._close_leg(self.long, reason, dec.long_exit_price)

        # Premium-decay TP (mark check on the closed bar; the poll also runs
        # independently between candles). Checked before entries so a TP this
        # same bar can't be immediately masked by a fresh flip.
        for leg in (self.short, self.long):
            await self._maybe_tp_leg(leg)

        if (dec.sell_signal and not self.executor_short.has_open_position
                and not self._entries_blocked()):
            await self._open_leg(self.short, SignalDir.SHORT.value, dec.short_sl, candle.close)
        if (dec.buy_signal and not self.executor_long.has_open_position
                and not self._entries_blocked()):
            await self._open_leg(self.long, SignalDir.LONG.value, dec.long_sl, candle.close)

    # ------------------------------------------------------------------ #
    async def _tp_poll_loop(self) -> None:
        interval = self.settings.supertrend_tp_poll_seconds
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            for leg in (self.short, self.long):
                await self._maybe_tp_leg(leg)

    async def _maybe_tp_leg(self, leg: _Leg) -> None:
        """Premium-decay TP: if this leg's option mark has fallen to/through
        its tp_price, flatten it and block NEW entries on BOTH legs until
        supertrend_tp_block_hour:minute today (see module docstring)."""
        if leg.closing or leg.tp_price is None or not leg.executor.has_open_position:
            return
        symbol = leg.executor.tracked_symbol
        if not symbol:
            return
        try:
            mark = await asyncio.to_thread(self.rest.get_mark_price, symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("Supertrend: TP mark fetch failed",
                       extra={"extra": {"leg": leg.name, "error": str(exc)}})
            return
        if mark is None or mark > leg.tp_price:
            return
        log.info("Supertrend: premium-decay TP hit",
                 extra={"extra": {"leg": leg.name, "mark": mark, "tp": leg.tp_price}})
        await self._close_leg(leg, "TP", mark)
        if leg.name == "short":
            self.strategy.force_flat_short()
        else:
            self.strategy.force_flat_long()
        self._block_entries_until = self._compute_block_until()
        log.info("Supertrend: TP hit — new entries blocked until block time",
                 extra={"extra": {"leg": leg.name,
                                   "until_ist": datetime.fromtimestamp(
                                       self._block_entries_until, _IST).isoformat()}})

    def _compute_block_until(self) -> float:
        now = datetime.now(_IST)
        target = now.replace(hour=self.settings.supertrend_tp_block_hour,
                             minute=self.settings.supertrend_tp_block_minute, second=0, microsecond=0)
        if now >= target:
            return time.time()  # already past today's cutoff (square-off already ran) -- no-op block
        return target.timestamp()

    # ------------------------------------------------------------------ #
    async def _verify_loop(self) -> None:
        interval = self.settings.position_verify_seconds
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            await self._maybe_verify_leg(self.short)
            await self._maybe_verify_leg(self.long)

    async def _maybe_verify_leg(self, leg: _Leg) -> None:
        """Self-heal: confirm the tracked SHORT option for this leg still
        exists on the exchange (both legs are short -- sign alone can't
        disambiguate, so this only checks the tracked product_id)."""
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
            log.warning("Supertrend: position-verify fetch failed",
                       extra={"extra": {"leg": leg.name, "error": str(exc)}})
            return
        if any(p["size"] < 0 and p.get("product_id") == tracked for p in positions):
            leg.verify_misses = 0
            return
        leg.verify_misses += 1
        if leg.verify_misses < 2:
            log.warning("Supertrend: tracked leg not on exchange (1st miss) — rechecking",
                        extra={"extra": {"leg": leg.name, "contract": leg.executor.tracked_symbol}})
            return
        contract = leg.executor.tracked_symbol
        log.warning("Supertrend: leg closed OUTSIDE the bot — self-healing to FLAT",
                    extra={"extra": {"leg": leg.name, "contract": contract}})
        leg.executor.clear()
        if leg.state_file:
            position_state.clear(leg.state_file)
        leg.entry_premium = None
        leg.tp_price = None
        leg.verify_misses = 0
        if leg.name == "short":
            self.strategy.force_flat_short()
        else:
            self.strategy.force_flat_long()
        await self.notifier.notify(
            NotifyEvent.EXIT, reason="closed outside the bot (self-healed)",
            contract=contract or "?", size=self.settings.option_contracts,
            side=f"sell {'CE' if leg.name == 'short' else 'PE'}",
        )

    # ------------------------------------------------------------------ #
    async def _close_leg(self, leg: _Leg, reason: str, btc_exit_price: float) -> None:
        if leg.closing or not leg.executor.has_open_position:
            return
        leg.closing = True
        try:
            contract = leg.executor.tracked_symbol
            try:
                fill = await leg.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("Supertrend: leg close failed",
                         extra={"extra": {"leg": leg.name, "error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close: {exc}")
                return
            if leg.state_file:
                position_state.clear(leg.state_file)
            entry_prem = leg.entry_premium
            lots = self.settings.option_contracts
            # SELL side: profit when the premium DECAYED (entry - exit).
            gross = ((entry_prem - fill) * lots * 0.001
                     if (entry_prem is not None and fill is not None) else 0.0)
            leg.entry_premium = None
            leg.tp_price = None
            side_label = "sell CE" if leg.name == "short" else "sell PE"
            log.info("Supertrend exit", extra={"extra": {
                "leg": leg.name, "reason": reason, "contract": contract, "btc_exit": btc_exit_price}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                entry_premium=entry_prem, exit_premium=fill,
                pnl=round(gross, 2), size=lots, side=side_label,
            )
        finally:
            leg.closing = False

    async def _open_leg(self, leg: _Leg, signal_dir: int, sl_level: float | None,
                        btc_price: float) -> None:
        """SELL the option for a new flip on this leg. Short(CE)/bearish
        signal_dir sells a CALL, long(PE)/bullish sells a PUT (the
        OptionsExecutor picks the side; this just labels the notification)."""
        if leg.entry_in_progress or leg.executor.has_open_position or leg.entry_premium is not None:
            return
        leg.entry_in_progress = True
        try:
            is_buy_signal = signal_dir == SignalDir.LONG.value  # bullish -> PE leg
            try:
                fill, symbol = await leg.executor.open_option_by_premium(
                    signal_dir, self.settings.target_premium
                )
            except OptionsMarginError as exc:
                log.error("Supertrend: margin/balance error",
                         extra={"extra": {"leg": leg.name, "error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Balance: {exc}")
                if leg.name == "short":
                    self.strategy.force_flat_short()
                else:
                    self.strategy.force_flat_long()
                return
            except Exception as exc:  # noqa: BLE001
                log.error("Supertrend: open_option_by_premium failed",
                         extra={"extra": {"leg": leg.name, "error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                if leg.name == "short":
                    self.strategy.force_flat_short()
                else:
                    self.strategy.force_flat_long()
                return
            if fill is None:
                log.warning("Supertrend: no option fill — flattening leg to stay in sync",
                           extra={"extra": {"leg": leg.name}})
                if leg.name == "short":
                    self.strategy.force_flat_short()
                else:
                    self.strategy.force_flat_long()
                return

            leg.entry_premium = fill
            tp_pct = self.settings.supertrend_take_profit_pct
            leg.tp_price = fill * (1.0 - tp_pct / 100.0) if tp_pct > 0 else None
            if leg.state_file:
                position_state.save(
                    leg.state_file, symbol=symbol or "",
                    product_id=leg.executor.tracked_product_id,
                    size=self.settings.option_contracts, entry_premium=fill,
                    tp_price=leg.tp_price, direction=signal_dir,
                )
            direction = "PUT" if is_buy_signal else "CALL"   # SELL side mapping
            log.info("Supertrend entry", extra={"extra": {
                "leg": leg.name, "direction": direction, "symbol": symbol, "fill": fill,
                "sl_level": sl_level, "tp_price": leg.tp_price}})
            event = NotifyEvent.ENTRY_LONG if is_buy_signal else NotifyEvent.ENTRY_SHORT
            await self.notifier.notify(
                event, direction=direction, contract=symbol or "?",
                premium=fill, btc_price=btc_price, sl_level=sl_level,
                tp_price=leg.tp_price, side="sell",
            )
        finally:
            leg.entry_in_progress = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile BOTH legs with the exchange. Both are short (size < 0)
        on the exchange, so disambiguate by symbol prefix (C-/P-), not sign."""
        positions: list[dict] = []
        try:
            positions = await asyncio.to_thread(
                self.rest.get_option_positions, self.executor_short.underlying
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Supertrend reconcile: fetch failed", extra={"extra": {"error": str(exc)}})

        ce_shorts = [p for p in positions if p["size"] < 0 and p.get("symbol", "").startswith("C-")]
        pe_shorts = [p for p in positions if p["size"] < 0 and p.get("symbol", "").startswith("P-")]
        self._reconcile_leg(self.short, ce_shorts, OptionType.CALL)
        self._reconcile_leg(self.long, pe_shorts, OptionType.PUT)

    def _reconcile_leg(self, leg: _Leg, found: list[dict], opt_type: OptionType) -> None:
        saved = position_state.load(leg.state_file) if leg.state_file else None
        owned_symbol = saved.get("symbol") if saved else None
        believe_owned = owned_symbol is not None or leg.executor.has_open_position

        if found:
            match = next((p for p in found if p.get("symbol") == owned_symbol), found[0])
            if saved and match.get("symbol") == owned_symbol:
                leg.entry_premium = saved.get("entry_premium")
                leg.tp_price = saved.get("tp_price")
            leg.executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
            log.info("Supertrend reconcile: adopted open leg",
                     extra={"extra": {"leg": leg.name, "symbol": match["symbol"]}})
            return

        if believe_owned:
            if not leg.executor.has_open_position and saved and saved.get("product_id"):
                leg.entry_premium = saved.get("entry_premium")
                leg.tp_price = saved.get("tp_price")
                leg.executor.adopt(int(saved["product_id"]), int(saved.get("size") or 0),
                                   opt_type, owned_symbol)
            log.warning("Supertrend reconcile: leg not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"leg": leg.name, "owned": owned_symbol}})
            return

        leg.executor.clear()
        leg.entry_premium = None
        leg.tp_price = None
        # force_flat_short()/force_flat_long() are no-ops if that side was
        # already flat, so it's always safe to call unconditionally here.
        if leg.name == "short":
            self.strategy.force_flat_short()
        else:
            self.strategy.force_flat_long()
        leg.closing = False
        log.info("Supertrend reconcile: no owned position — leg FLAT",
                 extra={"extra": {"leg": leg.name}})

    # ------------------------------------------------------------------ #
    # Daily 17:25 square-off -- plain close of both legs, no rollover.
    # ------------------------------------------------------------------ #
    def _entries_blocked(self) -> bool:
        if datetime.now(_IST).weekday() in self.settings.skip_weekday_ints:
            return True
        return time.time() < self._block_entries_until

    async def _square_off_scheduler(self) -> None:
        while True:
            now = datetime.now(_IST)
            target = now.replace(hour=self.settings.square_off_hour,
                                 minute=self.settings.square_off_minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait_s = (target - now).total_seconds()
            log.info("Supertrend: next 17:25 square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off()
            except Exception as exc:  # noqa: BLE001
                log.error("Supertrend: square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off(self) -> None:
        now = datetime.now(_IST)
        self._sq_off_date = now.date()
        log.info("Supertrend: 17:25 square-off firing", extra={"extra": {"date": str(self._sq_off_date)}})
        for leg in (self.short, self.long):
            if leg.executor.has_open_position:
                await self._close_leg(leg, "EOD", now.timestamp())
        # No rollover concept -- closed legs just end; force_flat() lets a
        # fresh flip re-arm from scratch (also harmless no-op if already flat).
        self.strategy.force_flat()
