"""Daily 9PM IST LADDERED Short Strangle -- live trading engine (option
SELL, SIX synchronized legs: one CALL + one PUT at EACH of
strangle9pm_laddered_otm_pcts, default 1%%/1.5%%/2%%).

Generalization of daily_strangle_9pm_trader.py (2 legs) to N OTM tiers,
ported from scripts/backtest_daily_strangle_9pm_laddered.py, which
validated this at 1mo (two separate 30-lot and 10-lot-per-leg windows,
both positive, laddered beating the 2-leg base strategy at matched TOTAL
exposure). Same daily entry timing, same combined target/SL rule as the
2-leg engine -- the only structural difference is 6 legs instead of 2, all
opened together and all closed together as ONE combined position.

SIZING (on request, 2026-10-02): "keep lot size as per live 9pm, divide by
3" -- EACH leg gets floor(the normal dynamic st15f-style lot count / 3),
so total exposure across all 6 legs matches what the 2-leg bot would have
deployed across its 2 legs at the SAME real balance (3 tiers x 2 legs each
= the 2-leg bot's own total x 1, not x3 -- see _maybe_recompute_lots).

EXITS (checked by a periodic poll loop, closed-1-minute-candle trade price,
same as the 2-leg engine's own 2026-09-29 alignment fix):
  1. TARGET: COMBINED premium (sum of all 6 legs) decayed to
     <= entry * (1 - target_pct/100).
  2. SL: COMBINED premium rose to >= entry * (1 + sl_pct/100).
  3. FALLBACK: neither fired by exit_hour:minute IST the NEXT day --
     force-closed there, matching the 2-leg engine's own fallback.

Runs as its own Docker container on a SEPARATE sub-account; position
ownership is tracked via SIX state files (one per leg/tier), all derived
from DELTA_STATE_FILE. Never touches any other bot.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
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


class _Leg:
    def __init__(self, name: str, is_call: bool, otm_pct: float, executor: OptionsExecutor, state_file: str) -> None:
        self.name = name                 # e.g. "ce_1", "pe_1_5", "ce_2"
        self.is_call = is_call
        self.otm_pct = otm_pct
        self.executor = executor
        self.state_file = state_file
        self.entry_premium: float | None = None


class DailyStrangle9pmLadderedEngine:
    """Live engine: sells a CE+PE pair at EACH configured OTM%% tier once
    daily, closes ALL legs together on a COMBINED premium target/SL or the
    next-day fallback square-off. See module docstring."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier

        otm_pcts = [float(x.strip()) for x in settings.strangle9pm_laddered_otm_pcts.split(",") if x.strip()]
        self.legs: list[_Leg] = []
        for pct in otm_pcts:
            tag = str(pct).rstrip("0").rstrip(".").replace(".", "_")
            for name, is_call in ((f"ce_{tag}", True), (f"pe_{tag}", False)):
                executor = OptionsExecutor(rest, settings)
                state_file = _leg_path(settings.state_file, name)
                self.legs.append(_Leg(name, is_call, pct, executor, state_file))

        self.entry_in_progress = False
        self.closing = False
        self._entry_task: asyncio.Task | None = None
        self._fallback_task: asyncio.Task | None = None
        self._poll_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------ #
    @property
    def has_open_strangle(self) -> bool:
        return any(leg.executor.has_open_position for leg in self.legs)

    @property
    def combined_entry_premium(self) -> float | None:
        prems = [leg.entry_premium for leg in self.legs]
        if any(p is None for p in prems):
            return None
        return sum(prems)

    async def start(self) -> None:
        mode = "TESTNET" if self.settings.testnet else "LIVE"
        await self.notifier.notify(NotifyEvent.RESTART, mode=mode)
        await self._sync_options_to_exchange()

        self._entry_task = asyncio.create_task(self._entry_scheduler())
        self._fallback_task = asyncio.create_task(self._fallback_scheduler())
        if self.settings.strangle9pm_poll_seconds > 0:
            self._poll_task = asyncio.create_task(self._poll_loop())
        log.info("DailyStrangle9pmLadderedEngine: starting live",
                 extra={"extra": {"legs": [leg.name for leg in self.legs]}})
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
                log.error("Strangle9pmLadder: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

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
            log.error("Strangle9pmLadder: spot price fetch failed", extra={"extra": {"error": str(exc)}})
            return None
        if not candles:
            return None
        return candles[-1].close

    # ------------------------------------------------------------------ #
    async def _entry_scheduler(self) -> None:
        while True:
            target = _next_occurrence(
                self.settings.strangle9pm_entry_hour, self.settings.strangle9pm_entry_minute
            )
            wait_s = (target - datetime.now(_IST)).total_seconds()
            log.info("Strangle9pmLadder: next entry window",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                await self._maybe_enter()
            except Exception as exc:  # noqa: BLE001
                log.error("Strangle9pmLadder: entry failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _maybe_recompute_lots(self) -> int:
        """Same dynamic-sizing formula as the 2-leg engine (balance /
        capital_per_lot, capped at max_lots, floored at 1), then divided by
        the number of OTM TIERS (not the number of legs -- a tier is one
        CE+PE pair) so total exposure across all tiers matches what the
        2-leg bot would deploy at the same real balance, per request
        2026-10-02 ("keep lot size as per live 9pm, divide by 3")."""
        if not self.settings.strangle9pm_compound_capital:
            return self.settings.option_contracts
        try:
            balance = await asyncio.to_thread(
                self.rest.get_available_balance, self.settings.option_margin_asset or None
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Strangle9pmLadder: dynamic-lot balance fetch failed — keeping current lot size",
                       extra={"extra": {"error": str(exc)}})
            return self.settings.option_contracts
        num_tiers = len(self.legs) // 2
        raw_lots = int(balance // self.settings.strangle9pm_capital_per_lot)
        base_lots = max(1, min(self.settings.strangle9pm_max_lots, raw_lots))
        per_tier_lots = max(1, base_lots // num_tiers)
        log.info("Strangle9pmLadder: dynamic lot-size recompute", extra={"extra": {
            "balance": round(balance, 2), "base_lots": base_lots, "num_tiers": num_tiers,
            "per_tier_lots": per_tier_lots, "old_lots": self.settings.option_contracts}})
        self.settings.option_contracts = per_tier_lots
        return per_tier_lots

    async def _maybe_enter(self) -> None:
        if self.entry_in_progress or self.has_open_strangle:
            log.info("Strangle9pmLadder: entry window fired but already positioned/in-progress — skipping")
            return
        self.entry_in_progress = True
        try:
            spot = await self._get_spot_price()
            if spot is None:
                log.error("Strangle9pmLadder: no spot price — skipping today's entry")
                return
            await self._maybe_recompute_lots()

            opened: list[_Leg] = []
            for leg in self.legs:
                signal_dir = SignalDir.SHORT.value if leg.is_call else SignalDir.LONG.value
                try:
                    fill, symbol = await leg.executor.open_option_by_otm_pct(signal_dir, spot, leg.otm_pct)
                except OptionsMarginError as exc:
                    log.error("Strangle9pmLadder: margin error — unwinding all opened legs",
                             extra={"extra": {"leg": leg.name, "error": str(exc)}})
                    await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{leg.name} balance: {exc}")
                    await self._unwind(opened, "LEG_FAILED")
                    return
                if fill is None:
                    log.error("Strangle9pmLadder: leg failed to fill — unwinding all opened legs",
                             extra={"extra": {"leg": leg.name}})
                    await self._unwind(opened, "LEG_FAILED")
                    return
                leg.entry_premium = fill
                if leg.state_file:
                    position_state.save(
                        leg.state_file, symbol=symbol or "", product_id=leg.executor.tracked_product_id,
                        size=self.settings.option_contracts, entry_premium=fill, direction=signal_dir,
                    )
                opened.append(leg)

            combined = sum(leg.entry_premium for leg in self.legs)
            log.info("Strangle9pmLadder entry", extra={"extra": {
                "legs": {leg.name: {"symbol": leg.executor.tracked_symbol, "fill": leg.entry_premium}
                        for leg in self.legs},
                "combined_entry": combined, "spot": spot}})
            for leg in self.legs:
                event = NotifyEvent.ENTRY_SHORT if leg.is_call else NotifyEvent.ENTRY_LONG
                await self.notifier.notify(
                    event, direction="CALL" if leg.is_call else "PUT",
                    contract=leg.executor.tracked_symbol or "?",
                    premium=leg.entry_premium, btc_price=spot, side=f"sell ({leg.name})",
                )
        finally:
            self.entry_in_progress = False

    async def _unwind(self, opened: list[_Leg], reason: str) -> None:
        """Entry failed partway through -- close whatever already filled
        rather than run a partial, unhedged ladder."""
        for leg in opened:
            if not leg.executor.has_open_position:
                continue
            symbol = leg.executor.tracked_symbol
            try:
                fill = await leg.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                log.error("Strangle9pmLadder: failed to unwind leg",
                         extra={"extra": {"leg": leg.name, "error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"unwind {leg.name} failed: {exc}")
                continue
            if leg.state_file:
                position_state.clear(leg.state_file)
            entry_prem = leg.entry_premium
            leg.entry_premium = None
            await self.notifier.notify(NotifyEvent.EXIT, reason=reason, contract=symbol or "?",
                                       entry_premium=entry_prem, exit_premium=fill,
                                       size=self.settings.option_contracts, side=f"sell ({leg.name})")

    # ------------------------------------------------------------------ #
    async def _poll_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise
            try:
                await self._check_target_sl()
            except Exception as exc:  # noqa: BLE001
                log.error("Strangle9pmLadder: poll check failed", extra={"extra": {"error": str(exc)}})

    async def _closed_candle_premium(self, symbol: str, max_age_sec: int = 120) -> float | None:
        now = int(time.time())
        try:
            candles = await asyncio.to_thread(
                self.rest.get_candles, symbol, "1m", now - max_age_sec, now
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Strangle9pmLadder: closed-candle fetch failed — falling back to mark price",
                       extra={"extra": {"symbol": symbol, "error": str(exc)}})
            candles = None
        if candles:
            latest = candles[-1]
            if now - latest.start_time <= max_age_sec:
                return latest.close
        try:
            return await asyncio.to_thread(self.rest.get_mark_price, symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("Strangle9pmLadder: mark price fallback fetch failed",
                       extra={"extra": {"symbol": symbol, "error": str(exc)}})
            return None

    async def _check_target_sl(self) -> None:
        if self.closing or not self.has_open_strangle or self.combined_entry_premium is None:
            return
        symbols = [leg.executor.tracked_symbol for leg in self.legs]
        if any(s is None for s in symbols):
            return
        prices = []
        for sym in symbols:
            p = await self._closed_candle_premium(sym)
            if p is None:
                return
            prices.append(p)
        combined = sum(prices)
        entry = self.combined_entry_premium
        target_level = entry * (1 - self.settings.strangle9pm_target_pct / 100.0)
        sl_level = entry * (1 + self.settings.strangle9pm_sl_pct / 100.0)
        if combined <= target_level:
            await self._close_strangle("TARGET")
        elif combined >= sl_level:
            await self._close_strangle("SL")

    # ------------------------------------------------------------------ #
    async def _fallback_scheduler(self) -> None:
        while True:
            target = _next_occurrence(
                self.settings.strangle9pm_exit_hour, self.settings.strangle9pm_exit_minute
            )
            wait_s = (target - datetime.now(_IST)).total_seconds()
            log.info("Strangle9pmLadder: next fallback square-off window",
                     extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                raise
            try:
                if self.has_open_strangle:
                    await self._close_strangle("EOD")
            except Exception as exc:  # noqa: BLE001
                log.error("Strangle9pmLadder: fallback square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    # ------------------------------------------------------------------ #
    async def _close_strangle(self, reason: str) -> None:
        if self.closing:
            return
        self.closing = True
        try:
            lots = self.settings.option_contracts
            for leg in self.legs:
                if not leg.executor.has_open_position:
                    continue
                symbol = leg.executor.tracked_symbol
                try:
                    fill = await leg.executor.close_option()
                except Exception as exc:  # noqa: BLE001
                    log.error("Strangle9pmLadder: leg close failed",
                             extra={"extra": {"leg": leg.name, "error": str(exc)}})
                    await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} {leg.name} close: {exc}")
                    continue
                if leg.state_file:
                    position_state.clear(leg.state_file)
                entry_prem = leg.entry_premium
                gross = ((entry_prem - fill) * lots * 0.001
                        if (entry_prem is not None and fill is not None) else 0.0)
                leg.entry_premium = None
                log.info("Strangle9pmLadder exit", extra={"extra": {
                    "leg": leg.name, "reason": reason, "symbol": symbol, "fill": fill, "pnl": round(gross, 2)}})
                await self.notifier.notify(
                    NotifyEvent.EXIT, reason=reason, contract=symbol or "?",
                    entry_premium=entry_prem, exit_premium=fill, pnl=round(gross, 2),
                    size=lots, side=f"sell ({leg.name})",
                )
        finally:
            self.closing = False

    # ------------------------------------------------------------------ #
    async def _sync_options_to_exchange(self) -> None:
        """Reconcile ALL legs with the exchange on startup. All are short
        (size < 0) -- first match each leg to its OWN state file's exact
        remembered symbol (unambiguous across restarts); any exchange
        position left unclaimed after that pass is adopted by whichever
        same-type (CALL/PUT) leg doesn't already believe it owns something,
        first-come-first-served (only relevant if state files are missing,
        e.g. a fresh deploy adopting a manually-opened ladder)."""
        positions: list[dict] = []
        try:
            positions = await asyncio.to_thread(
                self.rest.get_option_positions, self.legs[0].executor.underlying
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Strangle9pmLadder reconcile: fetch failed", extra={"extra": {"error": str(exc)}})

        shorts = [p for p in positions if p["size"] < 0]
        claimed: set[int] = set()

        # Pass 1: exact symbol match against each leg's own state file.
        for leg in self.legs:
            saved = position_state.load(leg.state_file) if leg.state_file else None
            owned_symbol = saved.get("symbol") if saved else None
            if owned_symbol:
                match = next((p for i, p in enumerate(shorts)
                             if p.get("symbol") == owned_symbol and i not in claimed), None)
                if match:
                    idx = shorts.index(match)
                    claimed.add(idx)
                    opt_type = OptionType.CALL if leg.is_call else OptionType.PUT
                    leg.executor.adopt(match["product_id"], match["size"], opt_type, match.get("symbol"))
                    leg.entry_premium = saved.get("entry_premium")
                    log.info("Strangle9pmLadder reconcile: adopted leg (state match)",
                             extra={"extra": {"leg": leg.name, "symbol": match["symbol"]}})

        # Pass 2: unclaimed legs -- adopt any remaining same-type position,
        # first-come-first-served; otherwise preserve a believed-owned
        # symbol with no exchange match, or go flat.
        for leg in self.legs:
            if leg.executor.has_open_position:
                continue
            saved = position_state.load(leg.state_file) if leg.state_file else None
            owned_symbol = saved.get("symbol") if saved else None
            prefix = "C-" if leg.is_call else "P-"
            candidate = next((p for i, p in enumerate(shorts)
                              if i not in claimed and p.get("symbol", "").startswith(prefix)), None)
            opt_type = OptionType.CALL if leg.is_call else OptionType.PUT
            if candidate is not None:
                idx = shorts.index(candidate)
                claimed.add(idx)
                leg.executor.adopt(candidate["product_id"], candidate["size"], opt_type, candidate.get("symbol"))
                leg.entry_premium = saved.get("entry_premium") if (saved and candidate.get("symbol") == owned_symbol) else None
                log.info("Strangle9pmLadder reconcile: adopted leg (unmatched pool)",
                         extra={"extra": {"leg": leg.name, "symbol": candidate["symbol"]}})
            elif owned_symbol is not None:
                leg.executor.adopt(int(saved["product_id"]), int(saved.get("size") or 0), opt_type, owned_symbol)
                leg.entry_premium = saved.get("entry_premium")
                log.warning("Strangle9pmLadder reconcile: leg not returned by exchange — preserving "
                            "tracked/state position, will NOT open new trades. If it was closed "
                            "manually, clear the state file and restart.",
                            extra={"extra": {"leg": leg.name, "owned": owned_symbol}})
            else:
                leg.executor.clear()
                leg.entry_premium = None
                log.info("Strangle9pmLadder reconcile: no owned position — leg FLAT",
                         extra={"extra": {"leg": leg.name}})
