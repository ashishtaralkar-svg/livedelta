"""Weekly short strangle + breakeven-triggered hedge -- live engine (strategy ``weeklyhedge``).

Port of scripts/backtest_weekly_strangle_breakeven_hedge.py --hedge-offset 2000:
  * Every ``weekly_entry_weekday`` (Fri) at ``weekly_entry_hour:minute`` IST, SELL a CALL
    and a PUT ~``weekly_otm_pct``%% away on NEXT Friday's weekly expiry.
  * Breakevens: upper = CE strike + combined premium, lower = PE strike - combined premium.
  * Once per minute, a CLOSED 1m BTC candle beyond a breakeven BUYS one hedge on that side
    (strike = breakeven +/- ``weekly_hedge_offset``, same expiry, same lots). At most one
    hedge per side per week; hedges are held to expiry.
  * Everything is closed at ``weekly_exit_hour:minute`` IST on expiry Friday. No target / SL.

Runs on its own sub-account. Each leg has its own state file plus a ``_meta`` file holding
the week's expiry / breakevens / hedged sides, all derived from ``DELTA_STATE_FILE``. New
entries are refused while ANY option position the bot does not own is open on the account.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ..config import Settings
from ..enums import NotifyEvent, OptionType
from ..exchange.rest_client import RestClient
from ..logging_setup import get_logger
from . import position_state
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_FRIDAY = 4

log = get_logger(__name__)


def _suffix_path(base: str, suffix: str) -> str:
    if not base:
        return ""
    p = Path(base)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def next_friday_expiry(now_ist: datetime) -> date:
    """The next Friday strictly after today (a Friday entry sells the following Friday)."""
    return (now_ist + timedelta(days=(_FRIDAY - now_ist.weekday()) % 7 or 7)).date()


def expiry_from_symbol(symbol: str) -> date | None:
    try:
        return datetime.strptime(symbol.split("-")[-1], "%d%m%y").date()
    except (ValueError, IndexError):
        return None


@dataclass
class _Leg:
    name: str           # ce / pe (sold) or hce / hpe (hedge)
    side: str           # CE / PE
    opt_type: OptionType
    sell: bool
    executor: OptionsExecutor
    state_file: str
    entry_premium: float | None = None


class WeeklyStrangleHedgeEngine:
    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier
        base = settings.state_file
        self.legs: dict[str, _Leg] = {}
        for name, side, otype, sell in (("ce", "CE", OptionType.CALL, True),
                                        ("pe", "PE", OptionType.PUT, True),
                                        ("hce", "CE", OptionType.CALL, False),
                                        ("hpe", "PE", OptionType.PUT, False)):
            self.legs[name] = _Leg(name, side, otype, sell,
                                   OptionsExecutor(rest, settings, side="sell" if sell else "buy"),
                                   _suffix_path(base, name))
        self.meta_file = _suffix_path(base, "meta")
        self.meta: dict | None = None
        self.busy = False
        self._hedge_fail_notified: set[str] = set()
        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------ #
    @property
    def has_position(self) -> bool:
        return any(leg.executor.has_open_position for leg in self.legs.values())

    def _close_dt(self) -> datetime | None:
        if not self.meta or not self.meta.get("expiry"):
            return None
        d = date.fromisoformat(self.meta["expiry"])
        return datetime(d.year, d.month, d.day, self.settings.weekly_exit_hour,
                        self.settings.weekly_exit_minute, tzinfo=_IST)

    def _save_meta(self) -> None:
        if self.meta_file and self.meta is not None:
            position_state.save(self.meta_file, **self.meta)

    def _next_entry(self, now: datetime) -> datetime:
        s = self.settings
        target = now.replace(hour=s.weekly_entry_hour, minute=s.weekly_entry_minute, second=0, microsecond=0)
        target += timedelta(days=(s.weekly_entry_weekday - now.weekday()) % 7)
        if target <= now:
            target += timedelta(days=7)
        return target

    async def start(self) -> None:
        mode = "TESTNET" if self.settings.testnet else "LIVE"
        await self.notifier.notify(NotifyEvent.RESTART, mode=mode)
        await self._reconcile()
        self._tasks = [asyncio.create_task(self._entry_scheduler()),
                       asyncio.create_task(self._poll_loop())]
        log.info("WeeklyStrangleHedgeEngine: starting live", extra={"extra": {
            "meta": self.meta, "open_legs": [n for n, l in self.legs.items() if l.executor.has_open_position]}})
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._stop_event.set()
        for t in self._tasks:
            t.cancel()
        if self.settings.close_on_shutdown and self.has_position:
            try:
                await self._close_all("shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("WeeklyHedge: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    async def _closed_btc_close(self) -> float | None:
        now = int(time.time())
        try:
            candles = await asyncio.to_thread(self.rest.get_candles, self.settings.symbol, "1m", now - 300, now)
        except Exception as exc:  # noqa: BLE001
            log.error("WeeklyHedge: BTC candle fetch failed", extra={"extra": {"error": str(exc)}})
            return None
        closed = [c for c in candles or [] if c.start_time + 60 <= now]
        return closed[-1].close if closed else None

    # ------------------------------------------------------------------ #
    async def _entry_scheduler(self) -> None:
        while True:
            target = self._next_entry(datetime.now(_IST))
            wait_s = (target - datetime.now(_IST)).total_seconds()
            log.info("WeeklyHedge: next entry window", extra={"extra": {"at": target.isoformat(), "in_s": int(wait_s)}})
            await asyncio.sleep(max(0.0, wait_s))
            try:
                await self._maybe_enter()
            except Exception as exc:  # noqa: BLE001
                log.error("WeeklyHedge: entry failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _maybe_enter(self) -> None:
        if self.busy or self.has_position:
            log.info("WeeklyHedge: entry window fired but already positioned/busy — skipping")
            return
        self.busy = True
        try:
            try:
                foreign = await asyncio.to_thread(self.rest.get_option_positions, self.legs["ce"].executor.underlying)
            except Exception as exc:  # noqa: BLE001
                log.error("WeeklyHedge: position check failed — skipping entry", extra={"extra": {"error": str(exc)}})
                return
            if foreign:
                syms = [p["symbol"] for p in foreign]
                log.error("WeeklyHedge: untracked option positions on account — NOT entering",
                          extra={"extra": {"symbols": syms}})
                await self.notifier.notify(NotifyEvent.API_ERROR,
                                           detail=f"weeklyhedge skipped entry: untracked positions {syms}")
                return
            spot = await self._closed_btc_close()
            if spot is None:
                log.error("WeeklyHedge: no BTC price — skipping this week's entry")
                return
            expiry = next_friday_expiry(datetime.now(_IST))
            off = spot * self.settings.weekly_otm_pct / 100.0
            ce, pe = self.legs["ce"], self.legs["pe"]

            ce_fill = await self._open(ce, expiry, spot + off)
            if ce_fill is None:
                return
            pe_fill = await self._open(pe, expiry, spot - off)
            if pe_fill is None:
                log.error("WeeklyHedge: PE failed after CE filled — unwinding CE")
                await self._close_leg(ce, "PE_FAILED")
                return

            combined = ce_fill + pe_fill
            self.meta = {
                "expiry": expiry.isoformat(), "entry_spot": spot, "combined": combined,
                "upper": ce.executor.tracked_strike + combined,
                "lower": pe.executor.tracked_strike - combined,
                "hedged": [],
            }
            self._save_meta()
            self._hedge_fail_notified.clear()
            log.info("WeeklyHedge entry", extra={"extra": {
                "ce": ce.executor.tracked_symbol, "ce_fill": ce_fill,
                "pe": pe.executor.tracked_symbol, "pe_fill": pe_fill, **self.meta}})
            for leg, fill, ev in ((ce, ce_fill, NotifyEvent.ENTRY_SHORT), (pe, pe_fill, NotifyEvent.ENTRY_LONG)):
                await self.notifier.notify(ev, direction="CALL" if leg.side == "CE" else "PUT",
                                           contract=leg.executor.tracked_symbol or "?",
                                           premium=fill, btc_price=spot, side="sell")
        finally:
            self.busy = False

    async def _open(self, leg: _Leg, expiry: date, target_strike: float) -> float | None:
        try:
            fill, symbol = await leg.executor.open_option_at_strike(leg.opt_type, expiry, target_strike)
        except OptionsMarginError as exc:
            log.error("WeeklyHedge: margin error", extra={"extra": {"leg": leg.name, "error": str(exc)}})
            await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"weeklyhedge {leg.name} balance: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001
            log.error("WeeklyHedge: open failed", extra={"extra": {"leg": leg.name, "error": str(exc)}})
            await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"weeklyhedge {leg.name} open: {exc}")
            return None
        if fill is None:
            return None
        leg.entry_premium = fill
        if leg.state_file:
            position_state.save(leg.state_file, symbol=symbol or "",
                                product_id=leg.executor.tracked_product_id,
                                size=leg.executor.tracked_size, entry_premium=fill)
        return fill

    # ------------------------------------------------------------------ #
    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                await self._tick()
            except Exception as exc:  # noqa: BLE001
                log.error("WeeklyHedge: poll failed", extra={"extra": {"error": str(exc)}})

    async def _tick(self) -> None:
        if self.busy or not self.has_position:
            return
        close_dt = self._close_dt()
        if close_dt is not None and datetime.now(_IST) >= close_dt:
            await self._close_all("EXPIRY")
            return
        if not self.meta or "upper" not in self.meta:
            return
        btc = await self._closed_btc_close()
        if btc is None:
            return
        for side, hit in (("CE", btc > self.meta["upper"]), ("PE", btc < self.meta["lower"])):
            if hit and side not in self.meta["hedged"]:
                await self._open_hedge(side, btc)

    async def _open_hedge(self, side: str, btc: float) -> None:
        leg = self.legs["hce" if side == "CE" else "hpe"]
        offset = self.settings.weekly_hedge_offset
        strike = self.meta["upper"] + offset if side == "CE" else self.meta["lower"] - offset
        expiry = date.fromisoformat(self.meta["expiry"])
        self.busy = True
        try:
            try:
                fill, symbol = await leg.executor.open_option_at_strike(leg.opt_type, expiry, strike)
            except Exception as exc:  # noqa: BLE001
                log.error("WeeklyHedge: hedge buy failed — will retry next minute",
                          extra={"extra": {"side": side, "error": str(exc)}})
                if side not in self._hedge_fail_notified:
                    self._hedge_fail_notified.add(side)
                    await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"weeklyhedge {side} hedge buy failed: {exc}")
                return
            if fill is None:
                return
            leg.entry_premium = fill
            if leg.state_file:
                position_state.save(leg.state_file, symbol=symbol or "",
                                    product_id=leg.executor.tracked_product_id,
                                    size=leg.executor.tracked_size, entry_premium=fill)
            self.meta["hedged"].append(side)
            self._save_meta()
            log.info("WeeklyHedge: hedge bought", extra={"extra": {
                "side": side, "symbol": symbol, "fill": fill, "btc": btc, "target_strike": strike}})
            await self.notifier.notify(NotifyEvent.ENTRY_LONG, direction=f"{'CALL' if side == 'CE' else 'PUT'} hedge",
                                       contract=symbol or "?", premium=fill, btc_price=btc, side="buy")
        finally:
            self.busy = False

    # ------------------------------------------------------------------ #
    async def _close_leg(self, leg: _Leg, reason: str) -> bool:
        symbol = leg.executor.tracked_symbol
        size = leg.executor.tracked_size
        try:
            fill = await leg.executor.close_option()
        except Exception as exc:  # noqa: BLE001
            log.error("WeeklyHedge: close failed", extra={"extra": {"leg": leg.name, "error": str(exc)}})
            await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"weeklyhedge {reason} {leg.name} close: {exc}")
            return False
        if leg.state_file:
            position_state.clear(leg.state_file)
        entry = leg.entry_premium
        pnl = 0.0
        if entry is not None and fill is not None:
            pnl = ((entry - fill) if leg.sell else (fill - entry)) * size * 0.001
        leg.entry_premium = None
        log.info("WeeklyHedge exit", extra={"extra": {
            "leg": leg.name, "reason": reason, "symbol": symbol, "fill": fill, "pnl": round(pnl, 2)}})
        await self.notifier.notify(NotifyEvent.EXIT, reason=reason, contract=symbol or "?",
                                   entry_premium=entry, exit_premium=fill, pnl=round(pnl, 2), size=size,
                                   side=f"{'sell' if leg.sell else 'buy'} {leg.side}")
        return True

    async def _close_all(self, reason: str) -> None:
        self.busy = True
        try:
            for leg in self.legs.values():
                if leg.executor.has_open_position:
                    await self._close_leg(leg, reason)
            if not self.has_position:
                self.meta = None
                if self.meta_file:
                    position_state.clear(self.meta_file)
        finally:
            self.busy = False

    # ------------------------------------------------------------------ #
    async def _reconcile(self) -> None:
        """Adopt only positions whose symbol is in this bot's own state files."""
        try:
            positions = await asyncio.to_thread(self.rest.get_option_positions,
                                                self.legs["ce"].executor.underlying)
        except Exception as exc:  # noqa: BLE001
            log.error("WeeklyHedge reconcile: fetch failed", extra={"extra": {"error": str(exc)}})
            positions = []
        for leg in self.legs.values():
            saved = position_state.load(leg.state_file) if leg.state_file else None
            sym = saved.get("symbol") if saved else None
            match = next((p for p in positions if sym and p["symbol"] == sym
                          and (p["size"] < 0) == leg.sell), None)
            if match:
                leg.executor.adopt(match["product_id"], match["size"], leg.opt_type, sym)
                leg.entry_premium = saved.get("entry_premium")
                log.info("WeeklyHedge reconcile: adopted leg", extra={"extra": {"leg": leg.name, "symbol": sym}})
            else:
                if sym:
                    log.warning("WeeklyHedge reconcile: state-file leg not on exchange — treating as closed",
                                extra={"extra": {"leg": leg.name, "symbol": sym}})
                    position_state.clear(leg.state_file)
                leg.executor.clear()
                leg.entry_premium = None

        self.meta = position_state.load(self.meta_file) if self.meta_file else None
        if not self.has_position:
            self.meta = None
            if self.meta_file:
                position_state.clear(self.meta_file)
            log.info("WeeklyHedge reconcile: FLAT")
            return
        if not self.meta:
            sym = next(l.executor.tracked_symbol for l in self.legs.values() if l.executor.has_open_position)
            exp = expiry_from_symbol(sym or "")
            self.meta = {"expiry": exp.isoformat() if exp else None, "hedged": ["CE", "PE"]}
            log.warning("WeeklyHedge reconcile: meta file missing — holding to expiry, hedging DISABLED",
                        extra={"extra": {"meta": self.meta}})
