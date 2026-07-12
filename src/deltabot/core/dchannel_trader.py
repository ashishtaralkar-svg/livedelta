"""Dchannel live trading engine.

Runs DchannelStrategy on 5-minute BTC candles (it computes synthetic Heikin Ashi
internally); executes option trades via OptionsExecutor in SELL mode; checks the
option's mark price for the premium-decay take-profit; manages the strategy's
BTC-price stop-loss and wall-clock EOD square-off.

This is deliberately a SIMPLER sibling of RevBreakSellEngine:
  * No paper trades and no SL-band classification -- the validated Dchannel
    backtest (5m, WR OFF, EMA200, sell ~1000 prem, 70% decay TP -> +$546/6mo)
    used neither. Every signal is a real trade.
  * Closed-bar only. The backtest evaluates signals at closed 5m boundaries and
    never intracandle, so this engine does too (no forming-candle entries),
    which is also what the RevBreak intracandle post-mortem recommended.
  * Same premium-decay TP as RevBreak (``take_profit_pct``), same BTC-stop SL
    (from the strategy decision), same EOD square-off, same reconcile/state model.

Runs as its own Docker container on a SEPARATE sub-account, in parallel with the
existing bots which it never touches. Position ownership is tracked via its own
``DELTA_STATE_FILE``.

Data flow per closed 5m BTC candle:
    1. DchannelStrategy.update(candle) -> exits (BTC SL / EOD) + new entries
    2. If a BTC exit fires and we hold the option: buy it back
    3. If still open: fetch option mark -> premium-decay TP check
    4. If a new entry signal and we are flat: sell the option nearest target_premium
A short poll also checks the option mark between bars so the TP fires ASAP.
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
from ..strategy.dchannel import DchannelStrategy
from . import position_state
from .candle_aggregator import CandleAggregator
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_BAR_SECONDS = 300  # 5 minutes — Dchannel's validated live timeframe

log = get_logger(__name__)

# Large multiple so the strategy's OWN internal BTC-price TP is unreachable: the
# engine drives TP via the option premium-decay target instead (matches the
# backtest's --tp-mode premium, which forces the internal RR TP out of range).
_INTERNAL_TP_DISABLED = 10_000.0


class DchannelEngine:
    """Live trading engine wired to DchannelStrategy (option SELL mode)."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.strategy = DchannelStrategy(
            dc_period=settings.dchannel_dc_period,
            wr_period=settings.dchannel_wr_period,
            wr_level=settings.dchannel_wr_level,
            ema_length=settings.dchannel_ema_length,
            ma_length=settings.dchannel_ma_length,
            wr_enabled=settings.dchannel_wr_enabled,
            anchor_mode=settings.dchannel_anchor_mode,
            rr_multiple=_INTERNAL_TP_DISABLED,
            tp_pct=None,
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
        self._tp_poll_task: asyncio.Task | None = None
        self._sq_off_date: date | None = None

        # Option TP tracking (set on entry, cleared on close).
        self._entry_premium: float | None = None
        self._tp_price: float | None = None
        self._current_dir: int | None = None  # SignalDir value of the open position
        self._tp_mult = 1.0 - settings.take_profit_pct / 100.0
        # Guards so the poll and closed-candle paths can't double open/close.
        self._entry_in_progress = False
        self._closing = False
        # Self-heal: detect a position closed OUTSIDE the bot (manual close,
        # settlement). Two consecutive empty checks are required before acting.
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
        if self.settings.dchannel_tp_poll_seconds > 0:
            self._tp_poll_task = asyncio.create_task(self._tp_poll_loop())
        log.info("DchannelEngine: starting live")
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
                log.info("Dchannel: closed option on shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("Dchannel: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass  # no ledger in this engine

    # ------------------------------------------------------------------ #
    async def _warmup(self) -> None:
        now = int(time.time())
        last_closed_end = (now // _BAR_SECONDS) * _BAR_SECONDS
        bars_needed = max(
            self.settings.warmup_candles + self.settings.dchannel_ema_length + 50,
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
        log.info("Dchannel warmup done",
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
            log.warning("Dchannel: candle gap detected — re-seeding")
            await self._warmup()

    # ------------------------------------------------------------------ #
    def _on_forming_candle(self, candle: Candle) -> None:
        if not self.settings.dchannel_intracandle_enabled:
            return  # closed-bar only
        task = asyncio.create_task(self._handle_forming_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_forming_candle(self, candle: Candle) -> None:
        """ASAP entry: the instant REAL price breaks the signal-range trigger
        (mid-bar, not waiting for the 5m close), open the option. This matches
        the backtest, which fills at the trigger price the moment a bar crosses
        it. Entry only -- SL/TP still resolve on the closed bar / TP poll."""
        if not self.strategy.ready:
            return
        if self.executor.has_open_position or self._entry_in_progress:
            return
        if not self.strategy.has_pending or self._entries_blocked():
            return
        confirmed, invalidated, entry_price = self.strategy.apply_intracandle_pending(candle)
        if invalidated:
            log.info("Dchannel: setup invalidated intracandle (opposite extreme hit first)")
            return
        if confirmed:
            signal_dir = (SignalDir.LONG.value
                          if self.strategy.position_state == PositionState.LONG
                          else SignalDir.SHORT.value)
            log.info("Dchannel: intracandle breakout — entering ASAP",
                     extra={"extra": {"trigger": entry_price}})
            await self._open_entry(signal_dir, self.strategy.sl_level, entry_price)

    def _on_closed_candle(self, candle: Candle) -> None:
        task = asyncio.create_task(self._handle_closed_candle(candle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_closed_candle(self, candle: Candle) -> None:
        if self._last_closed_start is not None:
            gap = candle.start_time - self._last_closed_start
            if gap > _BAR_SECONDS:
                log.warning("Dchannel: candle gap — re-seeding")
                await self._warmup()
        self._last_closed_start = candle.start_time

        if self._entries_blocked() and not self.executor.has_open_position:
            # In the settlement window: keep feeding the strategy, suppress entries.
            self.strategy.update(candle)
            return

        # 1. Strategy update -> exits (BTC SL / EOD) and new entries. SL is
        #    checked here BEFORE the premium TP, matching the backtest order.
        dec = self.strategy.update(candle)

        # 2. BTC exit (SL / EOD) closes the real option if we hold one.
        if dec is not None and dec.has_exit and self.executor.has_open_position:
            exit_price = dec.long_exit_price if dec.long_exit else dec.short_exit_price
            await self._close_btc_exit(dec.exit_reason or "SL", exit_price)

        # 3. Premium-decay TP check (only if still open after the BTC-exit step).
        if self.executor.has_open_position and self._tp_price is not None:
            symbol = self.executor.tracked_symbol
            mark = None
            if symbol:
                try:
                    mark = await asyncio.to_thread(self.rest.get_mark_price, symbol)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Dchannel: get_mark_price failed", extra={"extra": {"error": str(exc)}})
            if mark is not None and self._tp_price is not None and mark <= self._tp_price:
                await self._close_tp(mark)
                return

        # 4. New entry (only when flat and outside the settlement window).
        if (dec is not None and dec.has_entry
                and not self.executor.has_open_position and not self._entries_blocked()):
            signal_dir = SignalDir.LONG.value if dec.buy_signal else SignalDir.SHORT.value
            await self._open_entry(signal_dir, dec.sl_level, candle.close)

    # ------------------------------------------------------------------ #
    async def _tp_poll_loop(self) -> None:
        """Poll the option mark on a short interval so the premium-decay TP fires
        ASAP rather than only at the 5m close (mirrors the backtest, which sees
        intra-bar option lows via the option candles)."""
        interval = self.settings.dchannel_tp_poll_seconds
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
                log.warning("Dchannel: TP-poll mark fetch failed", extra={"extra": {"error": str(exc)}})
                continue
            if mark is not None and self._tp_price is not None and mark <= self._tp_price:
                log.info("Dchannel: premium-decay TP hit (poll)",
                         extra={"extra": {"mark": mark, "tp": self._tp_price}})
                await self._close_tp(mark)

    # ------------------------------------------------------------------ #
    async def _maybe_verify_position(self) -> None:
        """Self-heal: confirm the tracked short still exists on the exchange.

        If it vanished (closed manually in the UI, settled at expiry, any exit the
        bot did not make), the engine would otherwise poll a dead position forever
        and never trade again. Two CONSECUTIVE empty fetches are required before
        acting, and a fetch error is never treated as "gone", so a flaky API call
        can never drop a live position.
        """
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
        except Exception as exc:  # noqa: BLE001 — transient: conclude nothing
            log.warning("Dchannel: position-verify fetch failed",
                        extra={"extra": {"error": str(exc)}})
            return
        if any(p["size"] < 0 and p.get("product_id") == tracked for p in positions):
            self._verify_misses = 0
            return

        self._verify_misses += 1
        if self._verify_misses < 2:
            log.warning("Dchannel: tracked position not on exchange (1st miss) — rechecking",
                        extra={"extra": {"contract": self.executor.tracked_symbol}})
            return

        contract = self.executor.tracked_symbol
        log.warning("Dchannel: position closed OUTSIDE the bot — self-healing to FLAT",
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
        if self._closing or not self.executor.has_open_position:
            return
        self._closing = True
        try:
            contract = self.executor.tracked_symbol
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("Dchannel: TP close failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"TP close: {exc}")
                return
            if self.settings.state_file:
                position_state.clear(self.settings.state_file)
            exit_prem = fill if fill is not None else mark
            entry_prem = self._entry_premium
            lots = self.settings.option_contracts
            gross = (entry_prem - exit_prem) * lots * 0.001 if entry_prem is not None else 0.0
            self.strategy.notify_exit("TP")  # flatten the strategy so it hunts again
            self._entry_premium = self._tp_price = self._current_dir = None
            log.info("Dchannel TP hit", extra={"extra": {"contract": contract, "exit_prem": exit_prem}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason="TP", contract=contract or "?",
                entry_premium=entry_prem, exit_premium=exit_prem,
                pnl=round(gross, 2), size=lots,
            )
        finally:
            self._closing = False

    async def _close_btc_exit(self, reason: str, btc_exit_price: float) -> None:
        """Close the real option because the strategy's BTC SL or EOD fired. The
        strategy already flattened itself inside update() for SL/EOD, so we only
        close the exchange leg and clear our tracking."""
        if self._closing or not self.executor.has_open_position:
            return
        self._closing = True
        try:
            contract = self.executor.tracked_symbol
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("Dchannel: BTC-exit close failed", extra={"extra": {"error": str(exc)}})
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
            log.info("Dchannel exit", extra={"extra": {
                "reason": reason, "contract": contract, "btc_exit": btc_exit_price}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=contract or "?",
                entry_premium=entry_prem, exit_premium=exit_prem,
                pnl=round(gross, 2), size=lots,
            )
        finally:
            self._closing = False

    async def _open_entry(self, signal_dir: int, sl_level: float | None, btc_price: float) -> None:
        """Sell the option for a new Dchannel signal. Bullish -> sell PUT, bearish
        -> sell CALL (matching the backtest's SELL side)."""
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
                log.error("Dchannel: margin error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"Margin: {exc}")
                self.strategy.notify_exit("SL")  # unblock + flatten the strategy
                return
            except Exception as exc:  # noqa: BLE001
                log.error("Dchannel: open_option_by_premium failed", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=str(exc))
                self.strategy.notify_exit("SL")
                return
            if fill is None:
                log.warning("Dchannel: no option fill — skipping entry")
                self.strategy.notify_exit("SL")
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
            log.info("Dchannel entry", extra={"extra": {
                "direction": direction, "symbol": symbol, "fill": fill,
                "tp_price": round(self._tp_price, 1), "sl_level": sl_level}})
            event = NotifyEvent.ENTRY_LONG if is_buy else NotifyEvent.ENTRY_SHORT
            await self.notifier.notify(
                event, direction=direction, contract=symbol or "?",
                premium=fill, btc_price=btc_price, sl_level=sl_level,
                tp_price=round(self._tp_price, 1),
            )
        finally:
            self._entry_in_progress = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile the open option with the exchange on start/reconnect.

        SUB-ACCOUNT SAFETY MODEL (this sub-account runs only dchannelbot): any open
        short option on it is ours, so adopt it and do NOT open a second while one
        exists. Ownership is decided from three signals so no single failure can
        double-open: (A) exchange reports a short, (B) we already track one in
        memory, (C) the state file names one. Only when NONE hold are we FLAT."""
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
                log.error("Dchannel reconcile: fetch failed",
                          extra={"extra": {"error": str(exc), "attempt": attempt}})
                positions = []
            shorts = [p for p in positions if p["size"] < 0]
            if shorts or not believe_owned:
                break
            log.warning("Dchannel reconcile: expected a position but fetch is empty — retrying",
                        extra={"extra": {"owned": owned_symbol, "attempt": attempt}})
            await asyncio.sleep(1.5)

        # A) Exchange reports a short -> adopt it (prefer the state-file one so TP/
        #    direction tracking is restored).
        if shorts:
            match = next((p for p in shorts if p.get("symbol") == owned_symbol), shorts[0])
            if saved and match.get("symbol") == owned_symbol:
                self._entry_premium = saved.get("entry_premium")
                self._tp_price = saved.get("tp_price")
                self._current_dir = saved.get("direction")
            opt_type = OptionType.CALL if match["symbol"].startswith("C-") else OptionType.PUT
            self.executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
            log.info("Dchannel reconcile: adopted open short",
                     extra={"extra": {"symbol": match["symbol"],
                                      "matched_state_file": match.get("symbol") == owned_symbol}})
            return

        # B/C) Fetch empty but we believe we own one -> preserve/re-adopt, never
        #      clear-and-trade (that path orphans shorts / double-opens).
        if believe_owned:
            if not self.executor.has_open_position and saved and saved.get("product_id"):
                self._entry_premium = saved.get("entry_premium")
                self._tp_price = saved.get("tp_price")
                self._current_dir = saved.get("direction")
                opt_type = OptionType.CALL if str(owned_symbol).startswith("C-") else OptionType.PUT
                self.executor.adopt(int(saved["product_id"]), int(saved.get("size") or 0),
                                    opt_type, owned_symbol)
            log.warning("Dchannel reconcile: position not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"owned": owned_symbol, "tracked": self.executor.tracked_symbol}})
            return

        # Genuinely flat.
        self.executor.clear()
        self._entry_premium = self._tp_price = self._current_dir = None
        self.strategy.force_flat()
        self._closing = False
        log.info("Dchannel reconcile: no owned position — state FLAT")

    # ------------------------------------------------------------------ #
    # EOD wall-clock square-off (belt-and-suspenders for the strategy's own EOD)
    # ------------------------------------------------------------------ #
    def _entries_blocked(self) -> bool:
        now = datetime.now(_IST)
        if self._sq_off_date != now.date():
            return False
        resume = now.replace(hour=self.settings.entry_resume_hour,
                             minute=self.settings.entry_resume_minute, second=0, microsecond=0)
        return now < resume

    async def _square_off_scheduler(self) -> None:
        while True:
            now = datetime.now(_IST)
            target = now.replace(hour=self.settings.square_off_hour,
                                 minute=self.settings.square_off_minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait_s = (target - now).total_seconds()
            log.info("Dchannel: next EOD square-off",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._square_off_all()
            except Exception as exc:  # noqa: BLE001
                log.error("Dchannel: EOD square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _square_off_all(self) -> None:
        self._sq_off_date = datetime.now(_IST).date()
        log.info("Dchannel: EOD square-off firing", extra={"extra": {"date": str(self._sq_off_date)}})
        if not self.executor.has_open_position:
            self.strategy.notify_exit("EOD")
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
            self.strategy.notify_exit("EOD")
            self._entry_premium = self._tp_price = self._current_dir = None
            log.info("Dchannel: EOD square-off complete")
            await self.notifier.notify(
                NotifyEvent.EXIT, reason="EOD", contract=contract or "?",
                entry_premium=entry_prem, exit_premium=fill, pnl=round(gross, 2), size=lots,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Dchannel: EOD close failed", extra={"extra": {"error": str(exc)}})
            await self._sync_options_to_exchange()
