"""Daily 9PM IST Short Strangle -- live trading engine (option SELL, two
SYNCHRONIZED legs).

Pure TIME-based entry, no BTC candle signal at all: once a day at
``strangle9pm_entry_hour:minute`` IST (default 21:00), sells a CALL and a
PUT simultaneously, each at the listed strike closest to
``strangle9pm_otm_pct``%% OTM from the current BTC spot price (default 2%).
Ported from scripts/backtest_daily_strangle_9pm.py, which validated this
across 1wk/1mo/3mo windows before this live engine was built.

KEY DIFFERENCE FROM supertrend_trader.py (the other dual-leg engine in this
fleet): there, CE and PE are genuinely INDEPENDENT positions that can open/
close at different times. HERE, the two legs are always opened together and
always closed together as ONE combined position -- target/SL are evaluated
on the SUM of both legs' current buyback premiums against the SUM of both
legs' entry premiums, exactly like the backtest. Still uses TWO independent
OptionsExecutor instances (one per leg) since each is a genuinely separate
exchange contract, but there is no per-leg state machine: `_maybe_enter`/
`_close_strangle` always act on both at once.

EXITS (checked by a periodic poll loop, since premium is not something a
BTC candle feed reports):
  1. TARGET: combined premium decayed to <= entry * (1 - target_pct/100).
  2. SL: combined premium rose to >= entry * (1 + sl_pct/100).
  3. FALLBACK: neither fired by ``strangle9pm_exit_hour:minute`` IST the
     NEXT day (default 17:00) -- force-closed there, matching the
     backtest's own "exit next-day 17:00 IST" rule.

No candle feed, no warmup, no WebSocket -- this bot only needs REST calls
(current spot price at entry time, mark price while polling), so `start()`
just launches its two schedulers + poll loop and blocks on a stop event.

HONESTY: this strategy has never executed a real order before this engine
was built. Treat live results with real caution for the first several
weeks, same as every other bot's first deployment in this fleet.

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
from ..logging_setup import get_logger
from . import position_state
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")

log = get_logger(__name__)


def _leg_path(base: str, suffix: str) -> str:
    """``state/strangle9pm_pos.json`` -> ``state/strangle9pm_pos_ce.json``.
    Empty base (state persistence off) stays empty for both legs."""
    if not base:
        return ""
    p = Path(base)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _next_occurrence(hour: int, minute: int) -> datetime:
    now = datetime.now(_IST)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return target


class DailyStrangle9pmEngine:
    """Live engine: sells a CE+PE strangle once daily at a fixed IST time,
    closes both legs together on a combined premium target/SL or the
    next-day fallback square-off. See module docstring."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        self.executor_ce = OptionsExecutor(rest, settings)
        self.executor_pe = OptionsExecutor(rest, settings)
        self.state_file_ce = _leg_path(settings.state_file, "ce")
        self.state_file_pe = _leg_path(settings.state_file, "pe")

        self.entry_premium_ce: float | None = None
        self.entry_premium_pe: float | None = None
        self.entry_in_progress = False
        self.closing = False

        self._entry_task: asyncio.Task | None = None
        self._fallback_task: asyncio.Task | None = None
        self._poll_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------ #
    @property
    def has_open_strangle(self) -> bool:
        return self.executor_ce.has_open_position or self.executor_pe.has_open_position

    @property
    def combined_entry_premium(self) -> float | None:
        if self.entry_premium_ce is None or self.entry_premium_pe is None:
            return None
        return self.entry_premium_ce + self.entry_premium_pe

    async def start(self) -> None:
        mode = "TESTNET" if self.settings.testnet else "LIVE"
        await self.notifier.notify(NotifyEvent.RESTART, mode=mode)
        await self._sync_options_to_exchange()

        self._entry_task = asyncio.create_task(self._entry_scheduler())
        self._fallback_task = asyncio.create_task(self._fallback_scheduler())
        if self.settings.strangle9pm_poll_seconds > 0:
            self._poll_task = asyncio.create_task(self._poll_loop())
        log.info("DailyStrangle9pmEngine: starting live (SELL CE+PE strangle)")
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._stop_event.set()
        for t in (self._entry_task, self._fallback_task, self._poll_task):
            if t is not None:
                t.cancel()
        if self.settings.close_on_shutdown and self.has_open_strangle:
            try:
                await self._close_strangle("shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("Strangle9pm: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    async def _get_spot_price(self) -> float | None:
        now = int(time.time())
        try:
            candles = await asyncio.to_thread(
                self.rest.get_candles, self.settings.symbol, "1m", now - 300, now
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Strangle9pm: spot price fetch failed", extra={"extra": {"error": str(exc)}})
            return None
        if not candles:
            return None
        return candles[-1].close

    # ------------------------------------------------------------------ #
    # Daily 21:00 IST entry
    # ------------------------------------------------------------------ #
    async def _entry_scheduler(self) -> None:
        while True:
            target = _next_occurrence(
                self.settings.strangle9pm_entry_hour, self.settings.strangle9pm_entry_minute
            )
            wait_s = (target - datetime.now(_IST)).total_seconds()
            log.info("Strangle9pm: next entry window",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._maybe_enter()
            except Exception as exc:  # noqa: BLE001
                log.error("Strangle9pm: entry failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)   # clear the boundary before recomputing "next"

    async def _maybe_enter(self) -> None:
        if self.entry_in_progress or self.has_open_strangle:
            log.info("Strangle9pm: entry window fired but already positioned/in-progress — skipping")
            return
        self.entry_in_progress = True
        try:
            spot = await self._get_spot_price()
            if spot is None:
                log.error("Strangle9pm: no spot price — skipping today's entry")
                return

            try:
                ce_fill, ce_symbol = await self.executor_ce.open_option_by_otm_pct(
                    SignalDir.SHORT.value, spot, self.settings.strangle9pm_otm_pct
                )
            except OptionsMarginError as exc:
                log.error("Strangle9pm: CE margin error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"CE balance: {exc}")
                return
            if ce_fill is None:
                log.error("Strangle9pm: CE leg failed to fill — aborting entry (no PE opened)")
                return

            try:
                pe_fill, pe_symbol = await self.executor_pe.open_option_by_otm_pct(
                    SignalDir.LONG.value, spot, self.settings.strangle9pm_otm_pct
                )
            except OptionsMarginError as exc:
                log.error("Strangle9pm: PE margin error — CE leg is now UNHEDGED, closing it",
                         extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"PE balance: {exc}")
                await self._close_ce_only("PE_FAILED")
                return
            if pe_fill is None:
                log.error("Strangle9pm: PE leg failed to fill — CE leg is now UNHEDGED, closing it")
                await self._close_ce_only("PE_FAILED")
                return

            self.entry_premium_ce = ce_fill
            self.entry_premium_pe = pe_fill
            if self.state_file_ce:
                position_state.save(
                    self.state_file_ce, symbol=ce_symbol or "",
                    product_id=self.executor_ce.tracked_product_id,
                    size=self.settings.option_contracts, entry_premium=ce_fill,
                    direction=SignalDir.SHORT.value,
                )
            if self.state_file_pe:
                position_state.save(
                    self.state_file_pe, symbol=pe_symbol or "",
                    product_id=self.executor_pe.tracked_product_id,
                    size=self.settings.option_contracts, entry_premium=pe_fill,
                    direction=SignalDir.LONG.value,
                )
            log.info("Strangle9pm entry", extra={"extra": {
                "ce_symbol": ce_symbol, "ce_fill": ce_fill,
                "pe_symbol": pe_symbol, "pe_fill": pe_fill,
                "combined_entry": ce_fill + pe_fill, "spot": spot}})
            await self.notifier.notify(
                NotifyEvent.ENTRY_SHORT, direction="CALL", contract=ce_symbol or "?",
                premium=ce_fill, btc_price=spot, side="sell",
            )
            await self.notifier.notify(
                NotifyEvent.ENTRY_LONG, direction="PUT", contract=pe_symbol or "?",
                premium=pe_fill, btc_price=spot, side="sell",
            )
        finally:
            self.entry_in_progress = False

    async def _close_ce_only(self, reason: str) -> None:
        """The PE leg failed to open after the CE leg already filled -- close
        the now-unhedged CE leg rather than run a naked short. Does not
        touch entry_premium_ce/pe (both stay None, no strangle was formed)."""
        if not self.executor_ce.has_open_position:
            return
        symbol = self.executor_ce.tracked_symbol
        try:
            fill = await self.executor_ce.close_option()
        except Exception as exc:  # noqa: BLE001
            log.error("Strangle9pm: failed to unwind unhedged CE leg",
                     extra={"extra": {"error": str(exc)}})
            await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"unhedged CE close failed: {exc}")
            return
        if self.state_file_ce:
            position_state.clear(self.state_file_ce)
        await self.notifier.notify(NotifyEvent.EXIT, reason=reason, contract=symbol or "?",
                                   entry_premium=self.entry_premium_ce, exit_premium=fill,
                                   size=self.settings.option_contracts, side="sell CE")

    # ------------------------------------------------------------------ #
    # Combined premium TARGET / SL poll loop
    # ------------------------------------------------------------------ #
    async def _poll_loop(self) -> None:
        interval = self.settings.strangle9pm_poll_seconds
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            try:
                await self._check_target_sl()
            except Exception as exc:  # noqa: BLE001
                log.error("Strangle9pm: poll check failed", extra={"extra": {"error": str(exc)}})

    async def _check_target_sl(self) -> None:
        if self.closing or not self.has_open_strangle or self.combined_entry_premium is None:
            return
        ce_symbol = self.executor_ce.tracked_symbol
        pe_symbol = self.executor_pe.tracked_symbol
        if not ce_symbol or not pe_symbol:
            return
        try:
            ce_mark = await asyncio.to_thread(self.rest.get_mark_price, ce_symbol)
            pe_mark = await asyncio.to_thread(self.rest.get_mark_price, pe_symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("Strangle9pm: mark price fetch failed", extra={"extra": {"error": str(exc)}})
            return
        if ce_mark is None or pe_mark is None:
            return
        combined = ce_mark + pe_mark
        entry = self.combined_entry_premium
        target_level = entry * (1 - self.settings.strangle9pm_target_pct / 100.0)
        sl_level = entry * (1 + self.settings.strangle9pm_sl_pct / 100.0)
        if combined <= target_level:
            await self._close_strangle("TARGET")
        elif combined >= sl_level:
            await self._close_strangle("SL")

    # ------------------------------------------------------------------ #
    # Next-day 17:00 IST fallback square-off
    # ------------------------------------------------------------------ #
    async def _fallback_scheduler(self) -> None:
        while True:
            target = _next_occurrence(
                self.settings.strangle9pm_exit_hour, self.settings.strangle9pm_exit_minute
            )
            wait_s = (target - datetime.now(_IST)).total_seconds()
            log.info("Strangle9pm: next fallback square-off window",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                if self.has_open_strangle:
                    await self._close_strangle("EOD")
            except Exception as exc:  # noqa: BLE001
                log.error("Strangle9pm: fallback square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    # ------------------------------------------------------------------ #
    async def _close_strangle(self, reason: str) -> None:
        if self.closing:
            return
        self.closing = True
        try:
            ce_symbol = self.executor_ce.tracked_symbol
            pe_symbol = self.executor_pe.tracked_symbol
            ce_fill = pe_fill = None
            if self.executor_ce.has_open_position:
                try:
                    ce_fill = await self.executor_ce.close_option()
                except Exception as exc:  # noqa: BLE001
                    log.error("Strangle9pm: CE close failed", extra={"extra": {"error": str(exc)}})
                    await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} CE close: {exc}")
            if self.executor_pe.has_open_position:
                try:
                    pe_fill = await self.executor_pe.close_option()
                except Exception as exc:  # noqa: BLE001
                    log.error("Strangle9pm: PE close failed", extra={"extra": {"error": str(exc)}})
                    await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} PE close: {exc}")

            if self.state_file_ce:
                position_state.clear(self.state_file_ce)
            if self.state_file_pe:
                position_state.clear(self.state_file_pe)

            lots = self.settings.option_contracts
            entry_ce, entry_pe = self.entry_premium_ce, self.entry_premium_pe
            gross_ce = ((entry_ce - ce_fill) * lots * 0.001
                       if (entry_ce is not None and ce_fill is not None) else 0.0)
            gross_pe = ((entry_pe - pe_fill) * lots * 0.001
                       if (entry_pe is not None and pe_fill is not None) else 0.0)
            self.entry_premium_ce = None
            self.entry_premium_pe = None

            log.info("Strangle9pm exit", extra={"extra": {
                "reason": reason, "ce_symbol": ce_symbol, "pe_symbol": pe_symbol,
                "ce_fill": ce_fill, "pe_fill": pe_fill, "pnl": round(gross_ce + gross_pe, 2)}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=ce_symbol or "?",
                entry_premium=entry_ce, exit_premium=ce_fill, pnl=round(gross_ce, 2),
                size=lots, side="sell CE",
            )
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=pe_symbol or "?",
                entry_premium=entry_pe, exit_premium=pe_fill, pnl=round(gross_pe, 2),
                size=lots, side="sell PE",
            )
        finally:
            self.closing = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile BOTH legs with the exchange on startup. Both are short
        (size < 0) -- disambiguate CE vs PE by symbol prefix (C-/P-)."""
        positions: list[dict] = []
        try:
            positions = await asyncio.to_thread(
                self.rest.get_option_positions, self.executor_ce.underlying
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Strangle9pm reconcile: fetch failed", extra={"extra": {"error": str(exc)}})

        ce_shorts = [p for p in positions if p["size"] < 0 and p.get("symbol", "").startswith("C-")]
        pe_shorts = [p for p in positions if p["size"] < 0 and p.get("symbol", "").startswith("P-")]
        self.entry_premium_ce = self._reconcile_leg(
            self.executor_ce, self.state_file_ce, ce_shorts, OptionType.CALL, "CE")
        self.entry_premium_pe = self._reconcile_leg(
            self.executor_pe, self.state_file_pe, pe_shorts, OptionType.PUT, "PE")

    def _reconcile_leg(
        self, executor: OptionsExecutor, state_file: str, found: list[dict],
        opt_type: OptionType, label: str,
    ) -> float | None:
        saved = position_state.load(state_file) if state_file else None
        owned_symbol = saved.get("symbol") if saved else None
        believe_owned = owned_symbol is not None or executor.has_open_position

        if found:
            match = next((p for p in found if p.get("symbol") == owned_symbol), found[0])
            entry_prem = saved.get("entry_premium") if (saved and match.get("symbol") == owned_symbol) else None
            executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
            log.info("Strangle9pm reconcile: adopted open leg",
                     extra={"extra": {"leg": label, "symbol": match["symbol"]}})
            return entry_prem

        if believe_owned:
            entry_prem = None
            if not executor.has_open_position and saved and saved.get("product_id"):
                entry_prem = saved.get("entry_premium")
                executor.adopt(int(saved["product_id"]), int(saved.get("size") or 0), opt_type, owned_symbol)
            log.warning("Strangle9pm reconcile: leg not returned by exchange — preserving "
                        "tracked/state position, will NOT open new trades. If it was closed "
                        "manually, clear the state file and restart.",
                        extra={"extra": {"leg": label, "owned": owned_symbol}})
            return entry_prem

        executor.clear()
        log.info("Strangle9pm reconcile: no owned position — leg FLAT", extra={"extra": {"leg": label}})
        return None
