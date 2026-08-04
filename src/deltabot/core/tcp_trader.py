"""TCP (Three Candle Pattern) live trading engine (option BUY).

Runs TCPStrategy on 5-minute BTC candles and BUYS options: a bullish
pattern trigger (SELL of a break above the pattern high, HA open==low confirm
sequence) buys a CALL near ``target_premium`` (e.g. 400), a bearish trigger
buys a PUT. Profit comes from the premium RISING (or the SL/target level
being hit), not decaying -- this is the option-BUY vehicle, mirroring
DCv3Engine's relationship to DCv2Engine (same infra, opposite vehicle).

STRATEGY (see src/deltabot/strategy/tcp.py for the full rule set):
  SELL pattern (3 exact consecutive HA bars) -> buys a CALL on breakout.
  BUY pattern (mirror) -> buys a PUT on breakout.
  No EMA gate; both directions independently hunted at all times.

TAKE-PROFIT: a straightforward PREMIUM-GAIN target, tp_price = fill * (1 +
take_profit_pct/100) -- e.g. take_profit_pct=50 means exit once the bought
option's premium is up 50% from entry (validated in scripts/backtest_dcv2.py
--variant tcp --side buy --target-premium 400 --tp-gain-pct 50:
1mo +$84.87 net at 25 lots, 68 legs, 52.9% win rate).

EXECUTION TIMING (same convention as DCv2Engine/DCv3Engine):
  * ENTRY (signal triggered): ASAP intracandle.
  * TARGET (premium-gain TP): ASAP via a short poll of the option mark.
  * SL (fixed pattern-range level): ASAP intracandle.
  Intracandle entry+SL gated by ``dcv2_intracandle_enabled`` (shared config
  field with the other DC bots -- each bot's own env file sets it
  independently).

DAILY LIFECYCLE: same 17:25 square-off + 17:30 rollover machinery as
DCv2Engine/DCv3Engine (Mon-Sat trading, Saturday-flat, is the validated
config for the underlying signal -- see dcv2-mon-sat-weekend in project
memory; set DELTA_SKIP_WEEKDAYS/DELTA_DCV2_WEEKEND_FLAT accordingly per bot).

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
from ..strategy.tcp import TCPStrategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 300  # 5 minutes

log = get_logger(__name__)


class TCPEngine:
    """Live engine wired to TCPStrategy, executed via BUYING options
    (the same option-BUY vehicle as DCv3Engine). See module docstring."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = TCPStrategy(
            dc_period=settings.dcv2_dc_period,
            skip_weekdays=settings.skip_weekday_ints,
            day_tz=settings.day_tz,
        )
        self.executor = OptionsExecutor(rest, settings)
        self.aggregator = CandleAggregator(
            on_closed=self._on_closed_candle, on_forming=self._on_forming_candle
        )
        self.ws: WebSocketManager | None = None
        self._last_closed_start: int | None = None
        self._last_btc_close: float | None = None   # for the continuous-roll notify message
        self._tasks: set[asyncio.Task] = set()
        self._sq_off_task: asyncio.Task | None = None
        self._tp_poll_task: asyncio.Task | None = None
        self._sq_off_date: date | None = None

        self._entry_premium: float | None = None
        self._tp_price: float | None = None
        self._current_dir: int | None = None
        # BUY side: TP is a RALLY target -- 50% -> exit once premium is up 50%.
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
        log.info("TCPEngine: starting live (BUY side)")
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
                log.info("TCP: closed option on shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("TCP: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        bars_needed = max(
            self.settings.warmup_candles + self.settings.dcv2_dc_period + 50,
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
        log.info("TCP warmup done",
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
            log.warning("TCP: candle gap detected — re-seeding")
            await self._warmup()

    # ------------------------------------------------------------------ #
    # Intracandle: ASAP entry + ASAP SL.
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
        if self.executor.has_open_position and not self._closing:
            long_sl, short_sl, sl = self.strategy.check_intracandle_sl(candle.close)
            if long_sl or short_sl:
                log.info("TCP: intracandle SL touch — closing ASAP",
                         extra={"extra": {"sl": sl, "price": candle.close}})
                self.strategy.force_flat()
                await self._close_leg("SL", btc_exit_price=sl if sl is not None else candle.close)
                return
        if self.executor.has_open_position or self._entry_in_progress:
            return
        if not self.strategy.has_pending or self._entries_blocked():
            return
        confirmed, invalidated, entry_price = self.strategy.apply_intracandle_pending(candle)
        if invalidated:
            log.info("TCP: setup invalidated intracandle (SL side hit before trigger)")
            return
        if confirmed:
            signal_dir = (SignalDir.LONG.value
                          if self.strategy.position_state == PositionState.LONG
                          else SignalDir.SHORT.value)
            log.info("TCP: intracandle breakout — entering ASAP",
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
                log.warning("TCP: candle gap — re-seeding")
                await self._warmup()
        self._last_closed_start = candle.start_time
        self._last_btc_close = candle.close

        dec = self.strategy.update(candle)

        if self.settings.dcv2_debug_state:
            log.info("TCP state", extra={"extra": {
                "candle": candle.start_time, "o": candle.open, "h": candle.high,
                "l": candle.low, "c": candle.close, "blocked": self._entries_blocked(),
                "has_option": self.executor.has_open_position, **self.strategy.debug_state()}})

        # 1. Closed-bar exit (SL) closes the option.
        if dec is not None and dec.has_exit and self.executor.has_open_position:
            exit_price = dec.long_exit_price if dec.long_exit else dec.short_exit_price
            await self._close_leg(dec.exit_reason or "SL", btc_exit_price=exit_price)

        # 2. Premium-gain TP: mark has RISEN to/above tp_price.
        if self.executor.has_open_position and self._tp_price is not None:
            symbol = self.executor.tracked_symbol
            mark = None
            if symbol:
                try:
                    mark = await asyncio.to_thread(self.rest.get_mark_price, symbol)
                except Exception as exc:  # noqa: BLE001
                    log.warning("TCP: get_mark_price failed", extra={"extra": {"error": str(exc)}})
            if mark is not None and mark >= self._tp_price:
                await self._close_tp(mark)
                return

        # 3. Closed-bar entry fallback.
        if (dec is not None and dec.has_entry
                and not self.executor.has_open_position and not self._entries_blocked()):
            signal_dir = SignalDir.LONG.value if dec.buy_signal else SignalDir.SHORT.value
            await self._open_entry(signal_dir, dec.sl_level, candle.close, tag="ENTRY")
            return

        # 4. Rollover: directional trade still open but the option is flat.
        if (self.settings.dcv2_rollover_enabled
                and not self.executor.has_open_position and not self._entry_in_progress
                and not self._entries_blocked()
                and self.strategy.position_state != PositionState.FLAT):
            signal_dir = (SignalDir.LONG.value
                          if self.strategy.position_state == PositionState.LONG
                          else SignalDir.SHORT.value)
            log.info("TCP: rollover — re-buying option for the still-open trade")
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
                log.warning("TCP: TP-poll mark fetch failed", extra={"extra": {"error": str(exc)}})
                continue
            if mark is not None and self._tp_price is not None and mark >= self._tp_price:
                log.info("TCP: premium-gain TP hit (poll)",
                         extra={"extra": {"mark": mark, "tp": self._tp_price}})
                await self._close_tp(mark)

    # ------------------------------------------------------------------ #
    async def _maybe_verify_position(self) -> None:
        """Self-heal: confirm the tracked LONG option still exists on the
        exchange (this is the option-BUY vehicle -- size > 0 is ours)."""
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
            log.warning("TCP: position-verify fetch failed", extra={"extra": {"error": str(exc)}})
            return
        if any(p["size"] > 0 and p.get("product_id") == tracked for p in positions):
            self._verify_misses = 0
            return
        self._verify_misses += 1
        if self._verify_misses < 2:
            log.warning("TCP: tracked position not on exchange (1st miss) — rechecking",
                        extra={"extra": {"contract": self.executor.tracked_symbol}})
            return
        contract = self.executor.tracked_symbol
        log.warning("TCP: position closed OUTSIDE the bot — self-healing to FLAT",
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
                log.error("TCP: TP close failed", extra={"extra": {"error": str(exc)}})
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
            log.info("TCP TP hit", extra={"extra": {"contract": contract, "exit_prem": exit_prem}})
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
                log.error("TCP: leg close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            entry_prem = self._entry_premium
            lots = self.settings.option_contracts
            gross = ((fill - entry_prem) * lots * 0.001
                     if (entry_prem is not None and fill is not None) else 0.0)
            self._entry_premium = self._tp_price = self._current_dir = None
            log.info("TCP exit", extra={"extra": {
                "reason": reason, "contract": contract, "btc_exit": btc_exit_price}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                entry_premium=entry_prem, exit_premium=fill,
                pnl=round(gross, 2), size=lots, side="buy",
            )
        finally:
            self._closing = False

    async def _open_entry(self, signal_dir: int, sl_level: float | None,
                          btc_price: float, tag: str) -> None:
        """BUY the option for a new signal (or a rollover). Bullish -> buy CALL,
        bearish -> buy PUT (the OptionsExecutor picks the side per settings.option_side)."""
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
                log.error("TCP: margin/balance error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Balance: {exc}")
                self.strategy.force_flat()
                return
            except Exception as exc:  # noqa: BLE001
                log.error("TCP: open_option_by_premium failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                self.strategy.force_flat()
                return
            if fill is None:
                log.warning("TCP: no option fill — flattening to stay in sync")
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
            log.info("TCP entry", extra={"extra": {
                "tag": tag, "direction": direction, "symbol": symbol, "fill": fill,
                "tp_price": round(self._tp_price, 1), "sl_level": sl_level}})
            event = NotifyEvent.ENTRY_LONG if is_buy_signal else NotifyEvent.ENTRY_SHORT
            await self.notifier.notify(
                event, direction=direction, contract=symbol or "?",
                premium=fill, btc_price=btc_price, sl_level=sl_level,
                tp_price=round(self._tp_price, 1), tag=tag, side="buy",
            )
        finally:
            self._entry_in_progress = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile the open option with the exchange -- looks for a LONG
        position (size > 0), the same convention as DCv3Engine."""
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
                log.error("TCP reconcile: fetch failed",
                          extra={"extra": {"error": str(exc), "attempt": attempt}})
                positions = []
            longs = [p for p in positions if p["size"] > 0]
            if longs or not believe_owned:
                break
            log.warning("TCP reconcile: expected a position but fetch is empty — retrying",
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
            log.info("TCP reconcile: adopted open long",
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
            log.warning("TCP reconcile: position not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"owned": owned_symbol}})
            return

        self.executor.clear()
        self._entry_premium = self._tp_price = self._current_dir = None
        # Only reset the strategy on a genuine position DESYNC. When it is
        # already flat, a reconnect reconcile must NOT force_flat() -- that
        # would wipe the in-progress hunt state, dropping setups that were
        # mid-formation when the WebSocket reconnected.
        if self.strategy.position_state != PositionState.FLAT:
            self.strategy.force_flat()
        self._closing = False
        log.info("TCP reconcile: no owned position — state FLAT")

    # ------------------------------------------------------------------ #
    # Daily 17:25 square-off + optional weekend-flat; 17:30 rollover is in the
    # closed-bar handler.
    # ------------------------------------------------------------------ #
    def _entries_blocked(self) -> bool:
        now = datetime.now(_IST)
        if now.weekday() in self.settings.skip_weekday_ints:
            return True
        if self.settings.dcv2_continuous_roll:
            return False   # no same-day settlement gap -- _square_off() rolls immediately
        if self._sq_off_date != now.date():
            return False
        resume = now.replace(hour=self.settings.entry_resume_hour,
                             minute=self.settings.entry_resume_minute, second=0, microsecond=0)
        return now < resume

    def _weekend_flat_today(self, now: datetime) -> bool:
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
            log.info("TCP: next 17:25 square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off()
            except Exception as exc:  # noqa: BLE001
                log.error("TCP: square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off(self) -> None:
        now = datetime.now(_IST)
        self._sq_off_date = now.date()
        weekend_flat = self._weekend_flat_today(now)
        still_open = self.strategy.position_state != PositionState.FLAT
        continuous_roll = self.settings.dcv2_continuous_roll and not weekend_flat and still_open
        reason = "WEEKEND" if weekend_flat else ("ROLL" if continuous_roll else "EOD")
        log.info("TCP: 17:25 square-off firing",
                 extra={"extra": {"date": str(self._sq_off_date), "weekend_flat": weekend_flat,
                                  "continuous_roll": continuous_roll}})
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
                    NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                    entry_premium=entry_prem, exit_premium=fill, pnl=round(gross, 2), size=lots,
                    side="buy",
                )
            except Exception as exc:  # noqa: BLE001
                log.error("TCP: square-off close failed", extra={"extra": {"error": str(exc)}})
                await self._sync_options_to_exchange()
                return
        if weekend_flat:
            self.strategy.force_flat()
            log.info("TCP: weekend-flat — directional trade closed, no rollover")
            return
        if continuous_roll:
            state = self.strategy.position_state
            if state == PositionState.FLAT:
                log.warning("TCP: continuous-roll skipped — strategy went FLAT "
                            "during square-off (candle-close race)")
                return
            signal_dir = SignalDir.LONG.value if state == PositionState.LONG else SignalDir.SHORT.value
            btc_price = self._last_btc_close if self._last_btc_close is not None else 0.0
            log.info("TCP: continuous-roll — immediately re-buying at square-off")
            await self._open_entry(signal_dir, self.strategy.sl_level, btc_price, tag="ROLL")
