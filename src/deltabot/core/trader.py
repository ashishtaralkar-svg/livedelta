"""TradingEngine — orchestrates the live event loop.

Data flow:
    WS candlestick -> CandleAggregator.ingest -> (rollover) closed candle
        -> PineStrategy.update -> StrategyDecision (exits + entries)
        -> plan_actions -> OrderEngine.execute_plan
        -> update state, record PnL, notify

The strategy mirrors ``ashish.pine``: it enters only when price clears the
prev-day open/close and the 50-EMA(high/low) with the Supertrend aligned, exits
on a previous-candle stop-loss, and force-squares-off at the configured cut-off
time. Exits are evaluated on closed candles (the closed-candle approximation),
so backtest and live share identical fill semantics.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ..config import Settings
from ..enums import NotifyEvent, OptionType, PositionState, SignalDir
from ..exchange.rest_client import RestClient
from ..exchange.ws_manager import WebSocketManager
from ..logging_setup import get_logger
from ..models import Candle
from ..pnl import TradeLedger
from ..strategy.pine_strategy import PineStrategy, StrategyDecision, IntracandelSLCheck
from . import reconciler
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError
from .order_engine import OrderEngine, OrderExecutionError
from .state_machine import PositionStateMachine, plan_actions

_IST = ZoneInfo("Asia/Kolkata")

log = get_logger(__name__)

_RESOLUTION_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}


class TradingEngine:
    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier
        self.strategy = PineStrategy(
            atr_period=settings.atr_period,
            st_multiplier=settings.st_multiplier,
            ema_length=settings.ema_length,
            day_tz=settings.day_tz,
            day_start_hour=settings.day_start_hour,
            day_start_minute=settings.day_start_minute,
            square_off_hour=settings.square_off_hour,
            square_off_minute=settings.square_off_minute,
            use_close=settings.use_close,
        )
        self.sm = PositionStateMachine(contracts=settings.contracts)
        self.ledger = TradeLedger(contract_value=0.001, contracts=settings.contracts)
        self.order_engine = OrderEngine(rest, settings)
        self.options_executor = OptionsExecutor(rest, settings) if settings.options_mode else None
        self.aggregator = CandleAggregator(on_closed=self._on_closed_candle, on_forming=self._on_forming_candle)
        self.ws: WebSocketManager | None = None
        self._last_closed_start: int | None = None
        self._bar_seconds = _RESOLUTION_SECONDS.get(settings.resolution, 60)
        self._tasks: set[asyncio.Task] = set()

        # Wall-clock EOD square-off (independent of candle closes): fires at
        # square_off_hour:minute IST so positions flatten BEFORE the daily options
        # settle at 17:30 — the candle-driven square-off only fires when the
        # crossing bar closes, which can lag past settlement.
        self._sq_off_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        mode = "TESTNET" if self.settings.testnet else "LIVE"
        await self.notifier.notify(NotifyEvent.RESTART, mode=mode)

        # 1. Resolve product id (env override wins).
        product_id = self.settings.product_id or await asyncio.to_thread(
            self.rest.resolve_product_id, self.settings.symbol
        )
        self.settings.product_id = product_id
        self.order_engine.set_product_id(product_id)
        log.info(
            "Resolved product",
            extra={"extra": {"symbol": self.settings.symbol, "product_id": product_id}},
        )

        # 2. Leverage (not applicable to options — margin is set per option contract).
        if not self.settings.options_mode:
            try:
                await asyncio.to_thread(self.rest.set_leverage, product_id, self.settings.leverage)
                log.info("Leverage set", extra={"extra": {"leverage": self.settings.leverage}})
            except Exception as exc:  # noqa: BLE001
                log.warning("set_leverage failed (continuing)", extra={"extra": {"error": str(exc)}})

        # 3. Warmup the Supertrend from history.
        await self._warmup(product_id)

        # 4. Reconcile state with the exchange.
        await reconciler.reconcile(self.rest, product_id, self.sm, context="startup")
        await self._sync_strategy_to_exchange()
        await self._sync_ledger_to_exchange()

        # 5/6. Wire and run the WebSocket.
        self.ws = WebSocketManager(
            ws_url=self.settings.ws_url,
            symbol=self.settings.symbol,
            resolution=self.settings.resolution,
            api_key=self.settings.api_key.get_secret_value() or None,
            api_secret=self.settings.api_secret.get_secret_value() or None,
            on_candle=self.aggregator.ingest,
            on_reconnect=self._on_reconnect,
            heartbeat_timeout_s=self.settings.heartbeat_timeout_s,
        )
        # 7. Wall-clock EOD square-off scheduler (independent of candle closes).
        self._sq_off_task = asyncio.create_task(self._square_off_scheduler())

        log.info("Starting live engine")
        await self.ws.run()

    async def stop(self) -> None:
        if self.ws:
            self.ws.stop()
        if self._sq_off_task is not None:
            self._sq_off_task.cancel()
        if self.settings.close_on_shutdown:
            await self._close_on_shutdown()

    # ------------------------------------------------------------------ #
    async def _warmup(self, product_id: int) -> None:
        now = int(time.time())
        # Align to the last fully-closed bar boundary.
        last_closed_end = (now // self._bar_seconds) * self._bar_seconds
        # Need enough history for the Supertrend/EMA warmup AND at least a couple of
        # custom days so the previous-day open/close levels exist before we trade.
        indicator_bars = (
            self.settings.warmup_candles + self.settings.atr_period + self.settings.ema_length + 5
        )
        day_bars = self.settings.warmup_days * 86400 // self._bar_seconds
        bars_needed = max(indicator_bars, day_bars)
        start = last_closed_end - bars_needed * self._bar_seconds
        candles = await self._fetch_history_paged(start, last_closed_end)
        # Drop any in-progress final bar (start_time == current bar).
        current_bar = (now // self._bar_seconds) * self._bar_seconds
        closed = [c for c in candles if c.start_time < current_bar]
        self.strategy.seed(closed)
        if closed:
            self._last_closed_start = closed[-1].start_time
        log.info(
            "Warmup complete",
            extra={
                "extra": {
                    "candles": len(closed),
                    "ready": self.strategy.ready,
                    "state": self.strategy.position_state.value,
                    "pd_open": self.strategy.pd_open,
                    "pd_close": self.strategy.pd_close,
                }
            },
        )
        # seed() reset the strategy to flat; re-adopt any live exchange position so a
        # mid-session gap reseed cannot desync us from the real position.
        await self._sync_strategy_to_exchange()

    async def _fetch_history_paged(self, start: int, end: int) -> list[Candle]:
        """Fetch closed candles in ``[start, end]`` in <=2000-bar pages (REST cap)."""
        page_span = 2000 * self._bar_seconds
        out: list[Candle] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + page_span, end)
            page = await asyncio.to_thread(
                self.rest.get_candles, self.settings.symbol, self.settings.resolution, cursor, chunk_end
            )
            out.extend(page)
            cursor = chunk_end
        # De-dup overlapping page boundaries and keep ascending order.
        seen: set[int] = set()
        unique: list[Candle] = []
        for c in sorted(out, key=lambda c: c.start_time):
            if c.start_time not in seen:
                seen.add(c.start_time)
                unique.append(c)
        return unique

    async def _on_reconnect(self) -> None:
        if self.settings.product_id is None:
            return
        await reconciler.reconcile(self.rest, self.settings.product_id, self.sm, context="reconnect")
        await self._sync_strategy_to_exchange()
        await self._sync_ledger_to_exchange()
        # Backfill any candles missed while disconnected, then resume.
        await self._maybe_reseed_after_gap()

    async def _maybe_reseed_after_gap(self) -> None:
        if self._last_closed_start is None:
            return
        now = int(time.time())
        current_bar = (now // self._bar_seconds) * self._bar_seconds
        expected_next = self._last_closed_start + self._bar_seconds
        if current_bar - expected_next > self._bar_seconds:
            log.warning("Detected candle gap — re-seeding strategy from REST")
            await self._warmup(self.settings.product_id or 0)

    # ------------------------------------------------------------------ #
    def _on_closed_candle(self, candle: Candle) -> None:
        """Sync callback from the aggregator; bridge to the async handler."""
        task = asyncio.create_task(self._handle_closed_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_forming_candle(self, candle: Candle) -> None:
        """Sync callback for forming (intracandle) updates to check SL."""
        task = asyncio.create_task(self._handle_forming_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_closed_candle(self, candle: Candle) -> None:
        # Gap detection across closed bars.
        if self._last_closed_start is not None:
            gap = candle.start_time - self._last_closed_start
            if gap > self._bar_seconds:
                log.warning(
                    "Candle gap at close — re-seeding strategy",
                    extra={"extra": {"prev": self._last_closed_start, "now": candle.start_time}},
                )
                await self._warmup(self.settings.product_id or 0)
        self._last_closed_start = candle.start_time

        decision = self.strategy.update(candle)
        if decision is not None and (decision.has_exit or decision.has_entry):
            log.info(
                "Closed candle",
                extra={
                    "extra": {
                        "t": candle.start_time,
                        "close": candle.close,
                        "long_exit": decision.long_exit,
                        "short_exit": decision.short_exit,
                        "buy": decision.buy_signal,
                        "sell": decision.sell_signal,
                        "target": decision.target_state.value,
                    }
                },
            )
            await self._act_on_decision(decision)

    async def _handle_forming_candle(self, candle: Candle) -> None:
        """Check intracandle price for pending entry confirmation and stop-loss."""
        if not self.strategy.ready:
            return

        # --- Pending entry: confirm or invalidate intracandle ---
        confirmed, invalidated, entry_price = self.strategy.apply_intracandle_pending(candle)
        if confirmed:
            is_long = self.strategy.position_state == PositionState.LONG
            log.info(
                "Intracandle pending entry confirmed",
                extra={"extra": {"t": candle.start_time, "price": entry_price, "side": "LONG" if is_long else "SHORT"}},
            )
            dec = StrategyDecision(
                candle=candle,
                long_exit=False, short_exit=False,
                long_exit_sl=False, short_exit_sl=False,
                long_sq_off=False, short_sq_off=False,
                long_exit_price=candle.close,
                short_exit_price=candle.close,
                buy_signal=is_long,
                sell_signal=not is_long,
                entry_price=entry_price,
                target_state=self.strategy.position_state,
            )
            await self._act_on_decision(dec)
            return

        if invalidated:
            log.info(
                "Intracandle pending entry invalidated — SL crossed before trigger",
                extra={"extra": {"t": candle.start_time}},
            )
            return

        # --- Existing position: check intracandle SL ---
        for price in [candle.low, candle.high]:
            sl_check = self.strategy.check_intracandle_sl(price)
            if not (sl_check.long_exit_sl or sl_check.short_exit_sl):
                continue
            log.info(
                "Intracandle SL triggered",
                extra={"extra": {"t": candle.start_time, "price": price,
                                 "long_sl": sl_check.long_exit_sl, "short_sl": sl_check.short_exit_sl}},
            )
            if sl_check.long_exit_sl:
                dec = StrategyDecision(
                    candle=candle,
                    long_exit=True, short_exit=False,
                    long_exit_sl=True, short_exit_sl=False,
                    long_sq_off=False, short_sq_off=False,
                    long_exit_price=sl_check.long_exit_price or price,
                    short_exit_price=price,
                    buy_signal=False, sell_signal=False,
                    entry_price=price,
                    target_state=PositionState.FLAT,
                )
                self.strategy._in_long = False
                self.strategy._long_prev_low = None
                self.strategy._long_entry = None
            else:
                dec = StrategyDecision(
                    candle=candle,
                    long_exit=False, short_exit=True,
                    long_exit_sl=False, short_exit_sl=True,
                    long_sq_off=False, short_sq_off=False,
                    long_exit_price=price,
                    short_exit_price=sl_check.short_exit_price or price,
                    buy_signal=False, sell_signal=False,
                    entry_price=price,
                    target_state=PositionState.FLAT,
                )
                self.strategy._in_short = False
                self.strategy._short_prev_high = None
                self.strategy._short_entry = None
            await self._act_on_decision(dec)
            break

    async def _act_on_decision(self, dec: StrategyDecision) -> None:
        was = self.sm.state

        # ---- Options execution path ----
        if self.settings.options_mode:
            assert self.options_executor is not None
            try:
                # Exit before entry so a same-bar reversal buys back the old short first.
                if dec.has_exit:
                    fill = await self.options_executor.close_option()
                    if self.ledger.has_open:
                        self.ledger.close(
                            fill if fill is not None else dec.candle.close, dec.candle.start_time
                        )
                if dec.has_entry:
                    signal_dir = SignalDir.LONG if dec.buy_signal else SignalDir.SHORT
                    fill = await self.options_executor.open_option(signal_dir, dec.candle.close)
                    # We are SHORT the option leg regardless of the BTC direction; track
                    # it in the ledger with the option-lot quantity so PnL/summary work.
                    qty = self.settings.option_contracts * 0.001
                    self.ledger.open(
                        SignalDir.SHORT.value,
                        fill if fill is not None else dec.candle.close,
                        dec.candle.start_time,
                        qty_btc=qty,
                    )
            except OptionsMarginError as exc:
                log.error("Insufficient margin to sell option", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Margin: {exc}")
                return
            except Exception as exc:  # noqa: BLE001
                log.error("Options execution failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                # Re-sync the tracked option leg to the true exchange state.
                await self._sync_options_to_exchange()
                return
            self.sm.set_state(dec.target_state)
            await self._notify_decision(was, dec)
            return

        # ---- Futures execution path (unchanged) ----
        actions = plan_actions(
            long_exit=dec.long_exit,
            short_exit=dec.short_exit,
            buy_signal=dec.buy_signal,
            sell_signal=dec.sell_signal,
        )
        if not actions:
            return
        pos = await self.order_engine.current_position()
        try:
            await self.order_engine.execute_plan(actions, pos)
        except OrderExecutionError as exc:
            log.error("Order execution failed", extra={"extra": {"error": str(exc)}})
            await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
            # Re-sync to the true state rather than assuming.
            await reconciler.reconcile(
                self.rest, self.settings.product_id, self.sm, context="post-failure"
            )
            await self._sync_strategy_to_exchange()
            return

        # Mirror the Pine fills in the ledger (closed-candle approximation): exits
        # fill at the stop level or bar close, entries at the bar close.
        if dec.long_exit:
            self.ledger.close(dec.long_exit_price, dec.candle.start_time)
        if dec.short_exit:
            self.ledger.close(dec.short_exit_price, dec.candle.start_time)
        if dec.buy_signal:
            self.ledger.open(SignalDir.LONG.value, dec.entry_price, dec.candle.start_time)
        if dec.sell_signal:
            self.ledger.open(SignalDir.SHORT.value, dec.entry_price, dec.candle.start_time)

        self.sm.set_state(dec.target_state)
        await self._notify_decision(was, dec)

    async def _notify_decision(self, was: PositionState, dec: StrategyDecision) -> None:
        if dec.has_exit and not dec.has_entry:
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=dec.exit_reason or "EXIT", size=self.settings.contracts
            )
            return
        is_long = dec.buy_signal
        if was == PositionState.FLAT or not dec.has_exit:
            event = NotifyEvent.ENTRY_LONG if is_long else NotifyEvent.ENTRY_SHORT
        else:
            event = NotifyEvent.REVERSAL  # a same-bar exit + opposite entry (flip)
        await self.notifier.notify(
            event,
            symbol=self.settings.symbol,
            direction="LONG" if is_long else "SHORT",
            price=dec.entry_price,
            from_state=was.value,
        )

    # ------------------------------------------------------------------ #
    async def _sync_strategy_to_exchange(self) -> None:
        """Align the strategy's in-memory position with the exchange.

        On startup/reconnect the exchange is the source of truth. If it holds a
        position the strategy did not open (so the stop level is unknown), seed a
        best-effort stop from the last seen candle's low/high.

        In options mode, reconciliation runs against the live option chain via
        :meth:`_sync_options_to_exchange`.
        """
        if self.settings.options_mode:
            await self._sync_options_to_exchange()
            return
        if self.settings.product_id is None:
            return
        pos = await self.order_engine.current_position()
        if pos.size == 0:
            self.strategy.sync_position(PositionState.FLAT)
            return
        state = PositionState.LONG if pos.size > 0 else PositionState.SHORT
        self.strategy.sync_position(state, entry_price=pos.entry_price)
        log.info(
            "Strategy position synced to exchange",
            extra={"extra": {"state": state.value, "entry": pos.entry_price}},
        )

    async def _sync_options_to_exchange(self) -> None:
        """Reconcile the tracked option leg and strategy state against the exchange.

        On startup/reconnect, scan for any open SHORT option on our underlying. A
        short PUT corresponds to a BTC-bullish (LONG) strategy state; a short CALL
        to a BTC-bearish (SHORT) state. If none is open, reset everything to flat.

        Note: the BTC stop level cannot be recovered after a restart (it is unknown
        for an adopted position), so the price-based SL will not fire on it — only
        the EOD square-off will close an adopted leg. The strategy will trade
        normally again once the adopted position closes.
        """
        if self.options_executor is None:
            return
        try:
            positions = await asyncio.to_thread(
                self.rest.get_option_positions, self.options_executor.underlying
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failed to fetch option positions for reconcile (leaving state unchanged)",
                extra={"extra": {"error": str(exc)}},
            )
            return

        shorts = [p for p in positions if p["size"] < 0]
        if not shorts:
            self.options_executor.clear()
            self.strategy.sync_position(PositionState.FLAT)
            self.sm.set_state(PositionState.FLAT)
            if self.ledger.has_open:
                self.ledger.close(0.0)
            log.info("Options reconcile: no open short option — state FLAT")
            return

        if len(shorts) > 1:
            log.warning(
                "Options reconcile: multiple open short options found — adopting the first",
                extra={"extra": {"symbols": [p["symbol"] for p in shorts]}},
            )
        pos = shorts[0]
        opt_type = OptionType.CALL if pos["symbol"].startswith("C-") else OptionType.PUT
        self.options_executor.adopt(pos["product_id"], pos["size"], opt_type)
        # Short PUT => BTC-bullish (LONG); short CALL => BTC-bearish (SHORT).
        state = PositionState.LONG if opt_type == OptionType.PUT else PositionState.SHORT
        self.strategy.sync_position(state, entry_price=pos["entry_price"])
        self.sm.set_state(state)
        if not self.ledger.has_open:
            self.ledger.open(
                SignalDir.SHORT.value,
                pos["entry_price"] or 0.0,
                qty_btc=self.settings.option_contracts * 0.001,
            )
        log.info(
            "Options reconcile: adopted open short option",
            extra={
                "extra": {
                    "symbol": pos["symbol"],
                    "state": state.value,
                    "product_id": pos["product_id"],
                }
            },
        )

    async def _sync_ledger_to_exchange(self) -> None:
        """Align the ledger's open position with the exchange (best-effort)."""
        if self.settings.options_mode:
            return
        if self.settings.product_id is None:
            return
        pos = await self.order_engine.current_position()
        if pos.size == 0:
            if self.ledger.has_open:
                self.ledger.close(pos.entry_price or 0.0)
            return
        direction = SignalDir.LONG.value if pos.size > 0 else SignalDir.SHORT.value
        if not self.ledger.has_open:
            self.ledger.open(direction, pos.entry_price or 0.0)

    # ------------------------------------------------------------------ #
    # Wall-clock EOD square-off
    # ------------------------------------------------------------------ #
    async def _square_off_scheduler(self) -> None:
        """Fire the EOD square-off at square_off_hour:minute IST, every day.

        This is a wall-clock timer, deliberately independent of candle closes: the
        strategy's candle-driven square-off only triggers when the crossing bar
        *closes*, which on a 5m series lands at/after the 17:30 option settlement.
        Firing on the clock guarantees we flatten before settlement.
        """
        while True:
            now = datetime.now(_IST)
            target = now.replace(
                hour=self.settings.square_off_hour,
                minute=self.settings.square_off_minute,
                second=0,
                microsecond=0,
            )
            if now >= target:
                target += timedelta(days=1)
            wait_s = (target - now).total_seconds()
            log.info(
                "Next EOD square-off scheduled",
                extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}},
            )
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off_all()
            except Exception as exc:  # noqa: BLE001 — never let the scheduler die
                log.error("EOD square-off failed", extra={"extra": {"error": str(exc)}})
            # Step past the firing minute so the next loop schedules tomorrow.
            await asyncio.sleep(60)

    async def _square_off_all(self) -> None:
        """Force-close any open position now. New entries resume on the next signal
        (matching the Pine strategy's re-entry after square-off)."""
        ts = int(time.time())
        log.info("EOD square-off firing", extra={"extra": {"date": str(datetime.now(_IST).date())}})

        # --- Strategy 1: options leg ---
        if self.settings.options_mode and self.options_executor is not None:
            try:
                if self.options_executor.has_open_position:
                    fill = await self.options_executor.close_option()
                    if self.ledger.has_open:
                        self.ledger.close(fill if fill is not None else 0.0, ts)
                    self.strategy.sync_position(PositionState.FLAT)
                    self.sm.set_state(PositionState.FLAT)
                    log.info("EOD square-off: closed short-option leg")
                    await self.notifier.notify(
                        NotifyEvent.EXIT, reason="EOD", size=self.settings.option_contracts
                    )
            except Exception as exc:  # noqa: BLE001
                log.error("EOD square-off failed for option leg", extra={"extra": {"error": str(exc)}})
                await self._sync_options_to_exchange()
            return

        # --- Strategy 1: futures leg ---
        try:
            pos = await self.order_engine.current_position()
            if pos.size == 0:
                return
            from ..enums import Side

            side = Side.SELL if pos.size > 0 else Side.BUY
            await asyncio.to_thread(
                self.rest.place_market_order, self.settings.product_id, pos.abs_size, side, True
            )
            if self.ledger.has_open:
                self.ledger.close(0.0, ts)
            self.strategy.sync_position(PositionState.FLAT)
            self.sm.set_state(PositionState.FLAT)
            log.info("EOD square-off: closed futures position")
            await self.notifier.notify(NotifyEvent.EXIT, reason="EOD", size=pos.abs_size)
        except Exception as exc:  # noqa: BLE001
            log.error("EOD square-off failed for futures position", extra={"extra": {"error": str(exc)}})
            await reconciler.reconcile(
                self.rest, self.settings.product_id, self.sm, context="post-eod"
            )
            await self._sync_strategy_to_exchange()

    async def _close_on_shutdown(self) -> None:
        # In options mode, buy back the tracked short option rather than touching
        # the perpetual product_id (which would find a flat position and no-op).
        if self.settings.options_mode and self.options_executor is not None:
            try:
                if self.options_executor.has_open_position:
                    await self.options_executor.close_option()
                    await self.notifier.notify(
                        NotifyEvent.EXIT, reason="shutdown", size=self.settings.option_contracts
                    )
                    log.info("Closed option position on shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failed to close option on shutdown", extra={"extra": {"error": str(exc)}}
                )
            return
        try:
            pos = await self.order_engine.current_position()
            if pos.size == 0:
                return
            from ..enums import Side

            side = Side.SELL if pos.size > 0 else Side.BUY
            await asyncio.to_thread(
                self.rest.place_market_order, self.settings.product_id, pos.abs_size, side, True
            )
            await self.notifier.notify(NotifyEvent.EXIT, reason="shutdown", size=pos.abs_size)
            log.info("Closed position on shutdown")
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        summary = self.ledger.daily_summary()
        await self.notifier.notify(NotifyEvent.DAILY_PNL, **summary)
