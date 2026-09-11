"""Supertrend 15m-Filter Fixed-SL live trading engine (option SELL).

Runs Supertrend15mFilterFixedSlStrategy on 1-MINUTE BTC candles, SELLING
options: the strategy's own entry_is_short flag picks the side directly
(True -> sell a CALL, False -> sell a PUT) -- see
src/deltabot/strategy/supertrend_15m_filter_fixed_sl.py's module docstring
for the full rule set. REAL (non-HA) candles -- Heikin Ashi was only ever
applied to the separate supertrend_flip strategy, never this one.

Best validated backtest (1m, 15m filter, frozen SL, target 70%, premium
~1400, 25 lots): 1wk +$32.67/38 legs/26.3% win, 1mo +$243.25/150 legs/30.0%
win, 3mo +$635.26/483 legs/28.8% win -- the most consistent pace (~5.4
legs/day) and win rate across all three windows of any strategy tested this
session. Never executed a real order -- treat live results with real caution.

ARCHITECTURE: STRICT SINGLE POSITION (a CE and a PE are never open at the
same time -- unlike strategy="supertrend"/SupertrendFixedSlEngine), so this
engine uses ONE OptionsExecutor, the same shape as Ema21BreakdownEngine /
SupertrendSarEngine.

CLOSED-BAR ONLY -- Supertrend15mFilterFixedSlStrategy has no intracandle
methods and its own backtest is closed-bar-only (both the 1m entry/SL and
the premium target are evaluated once per closed 1m candle), so this engine
only acts at candle close -- exactly what was backtested. The 15-minute
confirmation filter is handled ENTIRELY INSIDE the strategy itself (its own
_feed_15m aggregation) -- this engine only ever feeds it 1-minute candles.

PROFIT TARGET (st15f_target_pct) is checked once per closed 1-minute candle
via a single mark-price fetch, NOT a continuous sub-minute poll -- matches
the validated backtest's own per-candle check exactly (unlike Ema21/SAR's
rally/decay TP, which poll continuously since their backtests do too).
REDUCTION convention -- "target 70% reduce price of option means if sold at
100 then target is 30" -- target = entry_premium * (1 - target_pct/100),
the OPPOSITE of SAR's decay-TO sar_tp_pct. On a target hit this engine calls
ONLY strategy.notify_target_hit() (no separate force_flat()) -- deliberately
relying on notify_target_hit() having been made self-sufficient after a real
bug (see the strategy's own docstring): it blocks new entries until the
15-minute Supertrend has its own next fresh flip, exactly matching "once
target is done then no new trade until 15 min supertrend is not change".

NO AUTO-REVERSE on a stop-out (unlike strategy="sar") -- a plain close.
NO ROLLOVER. NO DAILY SQUARE-OFF BY DEFAULT (st15f_eod_square_off=False) --
neither was described and the validated backtest has neither; a still-open
leg just runs until its own SL/target fires, or (live-only, not modeled in
the backtest) the exchange's own settlement plus this bot's regular
self-heal/reconcile below catches an expired contract and force-flattens.

OPT-IN EOD SQUARE-OFF (st15f_eod_square_off=True, on-request comparison
variant, backtested 2026-09-08): force-closes whatever's open at
square_off_hour:square_off_minute IST (the shared field, default 17:25)
every day, reason "EOD" -- pair with st15f_target_pct=0 to fully disable
the premium target and hold to end of day instead. Backtested statistically
a WASH vs target 70% on real candles (1mo $242.36 vs $243.25, 3mo $629.86
vs $635.26 at 25 lots) with MORE variance and capital held longer per
trade -- not a proven improvement, just a variant the user asked to run
live too. See config.py's st15f_eod_square_off comment for the full number.

OPT-IN COMPOUNDING LOT-SIZING (st15f_compound_capital=True, on-request
variant, added 2026-09-11): resizes every NEW entry to
floor(real available balance / st15f_capital_per_lot) lots, capped at
st15f_max_lots (a HARD SAFETY CEILING, not a target -- see config.py's
st15f_compound_capital comment for why: an uncapped 3-month backtest of
this exact ratio reached 3,447 lots / +31,860%, a number driven by
unconstrained compounding, not proven edge, with a genuine 58% single-day
drawdown baked into that same run). Recomputed once per day, piggybacking
on the st15f_eod_square_off checkpoint (requires that to be True too, or
the lot size never updates -- logged as a startup warning if
misconfigured). Uses the REAL, LIVE account balance (RestClient.
get_available_balance) -- unlike the backtest script's own --start-capital,
which has to simulate a running figure since it has no real account.

EXPIRY needed MINUTE precision ("if at 17:26 you should take trade in next
day option") that the shared OptionsExecutor._select_expiry() can't express
(hour-only, via option_expiry_cutoff_hour) -- see _MinutePreciseOptionsExecutor
below, a small subclass used ONLY by this engine. OptionsExecutor has no
shared/singleton state, so this cannot affect any other bot.

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
from ..strategy.supertrend_15m_filter_fixed_sl import Supertrend15mFilterFixedSlStrategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 60  # 1 minute

log = get_logger(__name__)


class _MinutePreciseOptionsExecutor(OptionsExecutor):
    """OptionsExecutor whose expiry cutoff has MINUTE precision (e.g. 17:26
    IST), not the shared class's hour-only ``option_expiry_cutoff_hour``.
    Overrides ONLY ``_select_expiry()`` -- every other method (margin
    checks, order placement, adopt/clear) is inherited unchanged. Used ONLY
    by this engine; OptionsExecutor has no shared/singleton state, so this
    cannot affect any other bot's behavior."""

    def __init__(self, rest: RestClient, settings: Settings, cutoff_hour: int, cutoff_minute: int) -> None:
        super().__init__(rest, settings)
        self._cutoff_hour = cutoff_hour
        self._cutoff_minute = cutoff_minute

    def _select_expiry(self) -> date:
        now_ist = datetime.now(tz=_IST)
        cutoff = now_ist.replace(hour=self._cutoff_hour, minute=self._cutoff_minute,
                                  second=0, microsecond=0)
        if now_ist >= cutoff:
            return (now_ist + timedelta(days=1)).date()
        return now_ist.date()


class Supertrend15mFilterFixedSlEngine:
    """Live engine wired to Supertrend15mFilterFixedSlStrategy, executed via
    SELLING options. See module docstring for the full architecture."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = Supertrend15mFilterFixedSlStrategy(
            atr_period=settings.st15f_atr_period,
            factor=settings.st15f_factor,
            atr_period_15m=settings.st15f_atr_period_15m,
            factor_15m=settings.st15f_factor_15m,
        )
        self.executor = _MinutePreciseOptionsExecutor(
            rest, settings, settings.st15f_expiry_cutoff_hour, settings.st15f_expiry_cutoff_minute,
        )
        # No on_forming -- this engine is closed-bar only, see module docstring.
        self.aggregator = CandleAggregator(on_closed=self._on_closed_candle)
        self.ws: WebSocketManager | None = None
        self._last_closed_start: int | None = None
        self._tasks: set[asyncio.Task] = set()
        self._selfheal_task: asyncio.Task | None = None
        self._sq_off_task: asyncio.Task | None = None
        self._sq_off_date: date | None = None

        self._entry_premium: float | None = None
        self._current_is_short: bool | None = None
        # target_pct<=0 means "no target at all" -- mirrors sar_tp_pct's own convention.
        self._target_frac = (
            1.0 - settings.st15f_target_pct / 100.0 if settings.st15f_target_pct > 0 else None
        )
        self._entry_in_progress = False
        self._closing = False
        self._verify_misses = 0
        self._last_verify = 0.0
        # Compounding lot-sizing (st15f_compound_capital) -- see config.py's
        # own comment for the full rationale/safety-cap discussion. Falls
        # back to the static option_contracts until the first real-balance
        # fetch (in start(), below) succeeds.
        self._current_lots = settings.option_contracts

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        mode = "TESTNET" if self.settings.testnet else "LIVE"
        await self.notifier.notify(NotifyEvent.RESTART, mode=mode)
        if self.settings.st15f_compound_capital and not self.settings.st15f_eod_square_off:
            log.warning("St15f: st15f_compound_capital=True but st15f_eod_square_off=False -- "
                        "the lot size will NEVER update (compounding piggybacks on the daily "
                        "square-off checkpoint). Enable st15f_eod_square_off too if this is "
                        "unintentional.")
        await self._warmup()
        await self._sync_options_to_exchange()
        await self._maybe_recompute_compounded_lots()

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
        if self.settings.position_verify_seconds > 0:
            self._selfheal_task = asyncio.create_task(self._selfheal_loop())
        if self.settings.st15f_eod_square_off:
            self._sq_off_task = asyncio.create_task(self._square_off_scheduler())
        log.info("Supertrend15mFilterFixedSlEngine: starting live (SELL side)")
        await self.ws.run()

    async def stop(self) -> None:
        if self.ws:
            self.ws.stop()
        if self._selfheal_task is not None:
            self._selfheal_task.cancel()
        if self._sq_off_task is not None:
            self._sq_off_task.cancel()
        if self.settings.close_on_shutdown and self.executor.has_open_position:
            try:
                lots = self.executor.tracked_size   # captured BEFORE close_option() clears tracked state
                await self.executor.close_option()
                if self.settings.state_file:
                    position_state.clear(self.settings.state_file)
                await self.notifier.notify(NotifyEvent.EXIT, reason="shutdown", size=lots, side="sell")
                log.info("St15f: closed option on shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("St15f: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        # Needs enough 1-minute bars for BOTH the 1m Supertrend's own ATR
        # warmup AND enough completed 15-minute buckets for the 15m
        # Supertrend's ATR warmup (atr_period_15m * 15 minutes) -- warmup_days
        # (default 3) comfortably covers either floor in practice.
        bars_needed = max(
            self.settings.warmup_candles + self.settings.st15f_atr_period_15m * 15 + 50,
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
        log.info("St15f warmup done",
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
            log.warning("St15f: candle gap detected — re-seeding")
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
                log.warning("St15f: candle gap — re-seeding")
                await self._warmup()
        self._last_closed_start = candle.start_time

        dec = self.strategy.update(candle)

        if self.settings.st15f_debug_state:
            log.info("St15f state", extra={"extra": {
                "candle": candle.start_time, "o": candle.open, "h": candle.high,
                "l": candle.low, "c": candle.close, "blocked": self._entries_blocked(),
                "has_option": self.executor.has_open_position, **self.strategy.debug_state()}})

        # 1. Closed-bar exit: the frozen SL was crossed. The strategy has
        #    already cleared its own _is_short/_active_sl internally, so no
        #    force_flat() call is needed here.
        if dec is not None and dec.has_exit and self.executor.has_open_position:
            await self._close_leg("SL", btc_exit_price=dec.exit_price)

        # 2. Profit target: a single mark-price check once per closed candle
        #    (not a continuous poll -- see module docstring).
        if (self.executor.has_open_position and self._target_frac is not None
                and not self._closing):
            symbol = self.executor.tracked_symbol
            mark = None
            if symbol:
                try:
                    mark = await asyncio.to_thread(self.rest.get_mark_price, symbol)
                except Exception as exc:  # noqa: BLE001
                    log.warning("St15f: get_mark_price failed", extra={"extra": {"error": str(exc)}})
            if (mark is not None and self._entry_premium is not None
                    and mark <= self._entry_premium * self._target_frac):
                await self._close_target(mark)
                return

        # 3. Closed-bar entry.
        if (dec is not None and dec.has_entry and not self.executor.has_open_position
                and not self._entries_blocked()):
            await self._open_entry(dec.entry_is_short, dec.sl_level, candle.close)

    # ------------------------------------------------------------------ #
    async def _selfheal_loop(self) -> None:
        interval = self.settings.position_verify_seconds
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            await self._maybe_verify_position()

    async def _maybe_verify_position(self) -> None:
        """Self-heal: confirm the tracked SHORT option still exists on the
        exchange (also the only thing that notices an expired contract, since
        this engine has no bespoke expiry-tracking -- see module docstring)."""
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
            log.warning("St15f: position-verify fetch failed", extra={"extra": {"error": str(exc)}})
            return
        if any(p["size"] < 0 and p.get("product_id") == tracked for p in positions):
            self._verify_misses = 0
            return
        self._verify_misses += 1
        if self._verify_misses < 2:
            log.warning("St15f: tracked position not on exchange (1st miss) — rechecking",
                        extra={"extra": {"contract": self.executor.tracked_symbol}})
            return
        contract = self.executor.tracked_symbol
        lots = self.executor.tracked_size   # captured BEFORE executor.clear() wipes tracked state
        log.warning("St15f: position closed OUTSIDE the bot (settlement/manual/liquidation) — "
                    "self-healing to FLAT", extra={"extra": {"contract": contract}})
        self.executor.clear()
        if self.settings.state_file:
            position_state.clear(self.settings.state_file)
        self._entry_premium = self._current_is_short = None
        self._verify_misses = 0
        self.strategy.force_flat()
        await self.notifier.notify(
            NotifyEvent.EXIT, reason="closed outside the bot (self-healed)",
            contract=contract or "?", size=lots, side="sell",
        )

    # ------------------------------------------------------------------ #
    def _pnl(self, entry_prem: float | None, exit_prem: float | None, lots: int) -> float:
        # SELL side: profit when premium DECAYS (entry - exit).
        if entry_prem is None or exit_prem is None:
            return 0.0
        return (entry_prem - exit_prem) * lots * 0.001

    async def _close_target(self, mark: float) -> None:
        if self._closing or not self.executor.has_open_position:
            return
        self._closing = True
        try:
            contract = self.executor.tracked_symbol
            lots = self.executor.tracked_size   # captured BEFORE close_option() clears tracked state
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("St15f: target close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"TARGET close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            exit_prem = fill if fill is not None else mark
            entry_prem = self._entry_premium
            gross = self._pnl(entry_prem, exit_prem, lots)
            self._entry_premium = self._current_is_short = None
            # Deliberately ONLY notify_target_hit() -- it is self-sufficient
            # (also clears the strategy's own _is_short/_active_sl) by design,
            # specifically so this call site can't repeat the exact missing-
            # force_flat() bug the backtest hit. See the strategy's own
            # notify_target_hit() docstring.
            self.strategy.notify_target_hit()
            log.info("St15f target hit", extra={"extra": {"contract": contract, "exit_prem": exit_prem}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason="TARGET", contract=contract or "?",
                entry_premium=entry_prem, exit_premium=exit_prem,
                pnl=round(gross, 2), size=lots, side="sell",
            )
        finally:
            self._closing = False

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
                log.error("St15f: leg close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            entry_prem = self._entry_premium
            gross = self._pnl(entry_prem, fill, lots)
            self._entry_premium = self._current_is_short = None
            log.info("St15f exit", extra={"extra": {
                "reason": reason, "contract": contract, "btc_exit": btc_exit_price}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                entry_premium=entry_prem, exit_premium=fill,
                pnl=round(gross, 2), size=lots, side="sell",
            )
        finally:
            self._closing = False

    async def _open_entry(self, is_short: bool, sl_level: float | None, btc_price: float) -> None:
        """SELL the option for a new entry. entry_is_short=True -> sell a
        CALL (CE); False -> sell a PUT (PE) -- the OptionsExecutor's
        sell-side _option_type_for maps SignalDir.SHORT -> CALL,
        SignalDir.LONG -> PUT."""
        if self._entry_in_progress or self.executor.has_open_position:
            return
        if self.settings.st15f_compound_capital:
            if self._current_lots <= 0:
                log.warning("St15f: compounding lot size is 0 — balance too small to trade, skipping entry")
                self.strategy.force_flat()
                return
            # OptionsExecutor.open_option_by_premium always sizes off
            # settings.option_contracts (no explicit-lots parameter exists)
            # -- this engine owns its own Settings instance exclusively (one
            # per container), so overwriting it here is safe and takes
            # effect immediately on the very next call below.
            self.settings.option_contracts = self._current_lots
        self._entry_in_progress = True
        try:
            signal_dir = SignalDir.SHORT.value if is_short else SignalDir.LONG.value
            try:
                fill, symbol = await self.executor.open_option_by_premium(
                    signal_dir, self.settings.target_premium
                )
            except OptionsMarginError as exc:
                log.error("St15f: margin/balance error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Balance: {exc}")
                self.strategy.force_flat()
                return
            except Exception as exc:  # noqa: BLE001
                log.error("St15f: open_option_by_premium failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                self.strategy.force_flat()
                return
            if fill is None:
                log.warning("St15f: no option fill — flattening to stay in sync")
                self.strategy.force_flat()
                return

            self._entry_premium = fill
            self._current_is_short = is_short
            if self.settings.state_file:
                position_state.save(
                    self.settings.state_file, symbol=symbol or "",
                    product_id=self.executor.tracked_product_id,
                    size=self.executor.tracked_size, entry_premium=fill,
                    is_short=is_short,
                )
            direction = "CALL" if is_short else "PUT"
            leverage_ok = self.executor.last_leverage_ok   # None unless DELTA_OPTION_LEVERAGE>0
            log.info("St15f entry", extra={"extra": {
                "direction": direction, "symbol": symbol, "fill": fill,
                "sl_level": sl_level, "leverage_ok": leverage_ok}})
            event = NotifyEvent.ENTRY_SHORT if is_short else NotifyEvent.ENTRY_LONG
            await self.notifier.notify(
                event, direction=direction, contract=symbol or "?",
                premium=fill, btc_price=btc_price, sl_level=sl_level,
                side="sell", leverage=self.settings.option_leverage, leverage_ok=leverage_ok,
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
                log.error("St15f reconcile: fetch failed",
                          extra={"extra": {"error": str(exc), "attempt": attempt}})
                positions = []
            shorts = [p for p in positions if p["size"] < 0]
            if shorts or not believe_owned:
                break
            log.warning("St15f reconcile: expected a position but fetch is empty — retrying",
                        extra={"extra": {"owned": owned_symbol, "attempt": attempt}})
            await asyncio.sleep(1.5)

        if shorts:
            match = next((p for p in shorts if p.get("symbol") == owned_symbol), shorts[0])
            if saved and match.get("symbol") == owned_symbol:
                self._entry_premium = saved.get("entry_premium")
                self._current_is_short = saved.get("is_short")
            opt_type = OptionType.CALL if match["symbol"].startswith("C-") else OptionType.PUT
            self.executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
            log.info("St15f reconcile: adopted open short",
                     extra={"extra": {"symbol": match["symbol"]}})
            return

        if believe_owned:
            if not self.executor.has_open_position and saved and saved.get("product_id"):
                self._entry_premium = saved.get("entry_premium")
                self._current_is_short = saved.get("is_short")
                opt_type = OptionType.CALL if str(owned_symbol).startswith("C-") else OptionType.PUT
                self.executor.adopt(int(saved["product_id"]), int(saved.get("size") or 0),
                                    opt_type, owned_symbol)
            log.warning("St15f reconcile: position not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"owned": owned_symbol}})
            return

        self.executor.clear()
        self._entry_premium = self._current_is_short = None
        if self.strategy.in_position:
            self.strategy.force_flat()
        self._closing = False
        log.info("St15f reconcile: no owned position — state FLAT")

    # ------------------------------------------------------------------ #
    def _entries_blocked(self) -> bool:
        return datetime.now(_IST).weekday() in self.settings.skip_weekday_ints

    # ------------------------------------------------------------------ #
    # OPT-IN (st15f_eod_square_off) -- see module/config.py docstrings. Off
    # by default; the base strategy has no square-off concept at all.
    # ------------------------------------------------------------------ #
    async def _square_off_scheduler(self) -> None:
        while True:
            now = datetime.now(_IST)
            target = now.replace(hour=self.settings.square_off_hour,
                                 minute=self.settings.square_off_minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait_s = (target - now).total_seconds()
            log.info("St15f: next EOD square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off()
            except Exception as exc:  # noqa: BLE001
                log.error("St15f: square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off(self) -> None:
        now = datetime.now(_IST)
        self._sq_off_date = now.date()
        log.info("St15f: EOD square-off firing", extra={"extra": {"date": str(self._sq_off_date)}})
        if self._closing or not self.executor.has_open_position:
            self.strategy.force_flat()
            await self._maybe_recompute_compounded_lots()
            return
        self._closing = True
        try:
            contract = self.executor.tracked_symbol
            lots = self.executor.tracked_size   # captured BEFORE close_option() clears tracked state
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("St15f: square-off close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"EOD close: {exc}")
                await self._sync_options_to_exchange()
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            entry_prem = self._entry_premium
            gross = self._pnl(entry_prem, fill, lots)
            self._entry_premium = self._current_is_short = None
            await self.notifier.notify(
                NotifyEvent.EXIT, reason="EOD", contract=contract or "?",
                entry_premium=entry_prem, exit_premium=fill, pnl=round(gross, 2), size=lots,
                side="sell",
            )
        finally:
            self._closing = False
        self.strategy.force_flat()
        # "at every day close check deposit and update lots" -- checked here,
        # AFTER the close above realizes today's last trade (if any) into the
        # real account balance, so the fetch below reflects the true
        # end-of-day deposit. See config.py's st15f_compound_capital comment.
        await self._maybe_recompute_compounded_lots()

    async def _maybe_recompute_compounded_lots(self) -> None:
        if not self.settings.st15f_compound_capital:
            return
        try:
            balance = await asyncio.to_thread(
                self.rest.get_available_balance, self.settings.option_margin_asset or None
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("St15f: compounding balance fetch failed — keeping current lot size",
                        extra={"extra": {"error": str(exc), "current_lots": self._current_lots}})
            return
        raw_lots = int(balance // self.settings.st15f_capital_per_lot)
        new_lots = max(0, min(self.settings.st15f_max_lots, raw_lots))
        if new_lots != self._current_lots:
            log.info("St15f: compounding lot-size update", extra={"extra": {
                "balance": round(balance, 2), "old_lots": self._current_lots, "new_lots": new_lots,
                "capped": raw_lots > self.settings.st15f_max_lots}})
        self._current_lots = new_lots
