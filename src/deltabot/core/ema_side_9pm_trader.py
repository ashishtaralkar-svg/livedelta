"""9PM EMA50/EMA200 one-side option SELL -- live trading engine (DELTA_STRATEGY=emaside9pm).

Directional version of the 9PM strangle: sells ONE leg picked by the 5m BTC trend.
Ported from scripts/backtest_9pm_ema_side_sell.py (``--stacked --sl-pct 100``):

* SESSION: starts at ``emaside_entry_hour:minute`` IST (21:00) and runs until
  ``emaside_exit_hour:minute`` the NEXT day (17:00). Sessions starting on a
  DELTA_SKIP_WEEKDAYS day are skipped. At most ONE trade per session.
* ENTRY: on every closed ``emaside_resolution`` (5m) BTC candle from the one that
  closed at the session start:
      close > EMA_fast > EMA_slow  -> SELL a PUT
      close < EMA_fast < EMA_slow  -> SELL a CALL
  anything else -> keep waiting for a later close. Strike = listed strike nearest
  ``emaside_otm_pct``%% from that candle's close (0 = ATM), on the session's
  next-day 17:30 expiry.
* EXITS (single leg, checked once a minute on the option's latest CLOSED 1m trade
  candle, same grain as the backtest): TARGET when the premium has decayed
  ``emaside_target_pct``%%; SL when it has risen ``emaside_sl_pct``%%; otherwise
  bought back at the session's exit time.

Own Delta sub-account. Position ownership: ``DELTA_STATE_FILE`` (the open leg) plus
``<stem>_session.json`` (the last session that already traded, so a restart never
re-enters the same session).
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
from ..models import Candle
from . import position_state
from .options_executor import OptionsExecutor, OptionsMarginError

_IST = ZoneInfo("Asia/Kolkata")
_RES_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900}

log = get_logger(__name__)


# ---------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------- #
def ema(values: list[float], length: int) -> list[float | None]:
    """Pine ta.ema: seeded with the SMA of the first ``length`` values."""
    out: list[float | None] = [None] * len(values)
    if len(values) < length:
        return out
    a = 2.0 / (length + 1)
    v = sum(values[:length]) / length
    out[length - 1] = v
    for i in range(length, len(values)):
        v = a * values[i] + (1 - a) * v
        out[i] = v
    return out


def stacked_signal(closes: list[float], fast: int, slow: int) -> tuple[int | None, float | None, float | None]:
    """Signal on the LAST close: LONG (sell PUT) if close > EMA_fast > EMA_slow, SHORT
    (sell CALL) if close < EMA_fast < EMA_slow, else None. Returns (dir, ema_fast, ema_slow)."""
    ef, es = ema(closes, fast), ema(closes, slow)
    if not closes or ef[-1] is None or es[-1] is None:
        return None, None, None
    c, f, s = closes[-1], ef[-1], es[-1]
    if c > f > s:
        return SignalDir.LONG.value, f, s
    if c < f < s:
        return SignalDir.SHORT.value, f, s
    return None, f, s


def session_window(now: datetime, entry_h: int, entry_m: int, exit_h: int, exit_m: int
                   ) -> tuple[datetime, datetime] | None:
    """The (start, end) of the session active at ``now`` (IST), or None between the
    exit time and the next entry time. A session starts on day D at entry_h:entry_m and
    ends on D+1 at exit_h:exit_m."""
    today_start = now.replace(hour=entry_h, minute=entry_m, second=0, microsecond=0)
    if now >= today_start:
        start = today_start
    else:
        start = today_start - timedelta(days=1)
    end = (start + timedelta(days=1)).replace(hour=exit_h, minute=exit_m)
    return (start, end) if start <= now < end else None


def _session_path(base: str) -> str:
    if not base:
        return ""
    p = Path(base)
    return str(p.with_name(f"{p.stem}_session{p.suffix}"))


def _next_occurrence(hour: int, minute: int) -> datetime:
    now = datetime.now(_IST)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return target


class EmaSide9pmEngine:
    """Live engine: one EMA-trend-picked option sale per 21:00 session. See module docstring."""

    def __init__(self, settings: Settings, rest: RestClient, notifier) -> None:
        self.settings = settings
        self.rest = rest
        self.notifier = notifier
        self.executor = OptionsExecutor(rest, settings)
        self.state_file = settings.state_file
        self.session_file = _session_path(settings.state_file)
        self.bar = _RES_SECONDS[settings.emaside_resolution]

        self.entry_premium: float | None = None
        self.direction: int | None = None
        self.entry_in_progress = False
        self.closing = False
        self._pending_exit: str | None = None       # a close that failed -- retried every poll
        self._foreign_symbols: set[str] = set()     # untracked positions on the account -> no entries
        self._missing_checks = 0
        self._last_traded_session = self._load_session()

        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        mode = "TESTNET" if self.settings.testnet else "LIVE"
        await self.notifier.notify(NotifyEvent.RESTART, mode=mode)
        if self.state_file and (err := position_state.check_writable(self.state_file)):
            log.error("EmaSide9pm: state dir NOT writable — fix ownership (chown 10001) or a restart "
                      "may re-enter a session / lose the open leg", extra={"extra": {"error": err}})
            await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"state dir not writable: {err}")
        await self._reconcile()
        self._tasks = [
            asyncio.create_task(self._hunt_loop()),
            asyncio.create_task(self._poll_loop()),
            asyncio.create_task(self._fallback_scheduler()),
        ]
        log.info("EmaSide9pmEngine: starting live", extra={"extra": {
            "otm_pct": self.settings.emaside_otm_pct, "target_pct": self.settings.emaside_target_pct,
            "sl_pct": self.settings.emaside_sl_pct, "lots": self.settings.option_contracts,
            "ema": f"{self.settings.emaside_fast}/{self.settings.emaside_slow}",
            "resolution": self.settings.emaside_resolution,
            "last_traded_session": self._last_traded_session}})
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._stop_event.set()
        for t in self._tasks:
            t.cancel()
        if self.settings.close_on_shutdown and self.executor.has_open_position:
            try:
                await self._close("shutdown")
            except Exception as exc:  # noqa: BLE001
                log.error("EmaSide9pm: failed to close on shutdown", extra={"extra": {"error": str(exc)}})

    async def daily_summary(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Session bookkeeping
    # ------------------------------------------------------------------ #
    def _load_session(self) -> str | None:
        saved = position_state.load(self.session_file) if self.session_file else None
        return saved.get("session") if saved else None

    def _mark_session_traded(self, session: str) -> None:
        self._last_traded_session = session
        if self.session_file:
            position_state.save(self.session_file, session=session)

    # ------------------------------------------------------------------ #
    # Entry hunt: once per closed signal candle
    # ------------------------------------------------------------------ #
    async def _hunt_loop(self) -> None:
        while True:
            now = time.time()
            wake = (int(now) // self.bar + 1) * self.bar + 5     # 5s after the next candle close
            await asyncio.sleep(max(1.0, wake - now))
            try:
                await self._hunt_once()
            except Exception as exc:  # noqa: BLE001
                log.error("EmaSide9pm: hunt failed", extra={"extra": {"error": str(exc)}})

    async def _closed_signal_candles(self, boundary: int) -> list[Candle] | None:
        """BTC candles up to the one that closed at ``boundary`` (retries while the
        exchange has not published it yet)."""
        for _ in range(4):
            try:
                candles = await asyncio.to_thread(
                    self.rest.get_candles, self.settings.symbol, self.settings.emaside_resolution,
                    boundary - 4 * 86400, boundary)
            except Exception as exc:  # noqa: BLE001
                log.warning("EmaSide9pm: candle fetch failed", extra={"extra": {"error": str(exc)}})
                candles = []
            closed = [c for c in candles if c.start_time + self.bar <= boundary]
            if closed and closed[-1].start_time == boundary - self.bar:
                return closed
            await asyncio.sleep(5)
        log.warning("EmaSide9pm: latest closed candle not available — skipping this bar",
                    extra={"extra": {"boundary": boundary}})
        return None

    async def _hunt_once(self) -> None:
        s = self.settings
        if self.executor.has_open_position or self.entry_in_progress or self.closing:
            return
        now_ist = datetime.now(_IST)
        win = session_window(now_ist, s.emaside_entry_hour, s.emaside_entry_minute,
                             s.emaside_exit_hour, s.emaside_exit_minute)
        if win is None:
            return
        start, end = win
        session = start.date().isoformat()
        if self._last_traded_session == session:
            return
        if start.weekday() in s.skip_weekday_ints:
            return
        if self._foreign_symbols:
            log.warning("EmaSide9pm: untracked position(s) on the account — entries blocked",
                        extra={"extra": {"symbols": sorted(self._foreign_symbols)}})
            return
        boundary = int(time.time()) // self.bar * self.bar
        if boundary < int(start.timestamp()) or boundary >= int(end.timestamp()):
            return
        candles = await self._closed_signal_candles(boundary)
        if not candles:
            return
        closes = [c.close for c in candles]
        direction, ef, es = stacked_signal(closes, s.emaside_fast, s.emaside_slow)
        log.info("EmaSide9pm: candle check", extra={"extra": {
            "session": session, "close": closes[-1],
            "ema_fast": round(ef, 1) if ef else None, "ema_slow": round(es, 1) if es else None,
            "signal": {SignalDir.LONG.value: "SELL PUT", SignalDir.SHORT.value: "SELL CALL"}.get(direction, "wait")}})
        if direction is None:
            return
        await self._enter(direction, closes[-1], start.date() + timedelta(days=1), session)

    async def _enter(self, direction: int, spot: float, expiry: date, session: str) -> None:
        s = self.settings
        self.entry_in_progress = True
        try:
            otype = OptionType.PUT if direction == SignalDir.LONG.value else OptionType.CALL
            pct = s.emaside_otm_pct / 100.0
            target = spot * (1 - pct) if otype == OptionType.PUT else spot * (1 + pct)
            try:
                fill, symbol = await self.executor.open_option_at_strike(otype, expiry, target)
            except OptionsMarginError as exc:
                log.error("EmaSide9pm: margin error", extra={"extra": {"error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"balance: {exc}")
                self._mark_session_traded(session)        # don't hammer a failing entry all night
                return
            if not self.executor.has_open_position:
                log.error("EmaSide9pm: entry order not placed — will retry next candle")
                return
            if fill is None:
                fill = await asyncio.to_thread(self.rest.get_mark_price, symbol) if symbol else None
                log.warning("EmaSide9pm: no fill price returned — using mark as entry reference",
                            extra={"extra": {"symbol": symbol, "mark": fill}})
            self.entry_premium = fill
            self.direction = direction
            self._mark_session_traded(session)
            if self.state_file:
                position_state.save(
                    self.state_file, symbol=symbol or "", product_id=self.executor.tracked_product_id,
                    size=self.executor.tracked_size, entry_premium=fill, direction=direction,
                    session=session)
            log.info("EmaSide9pm entry", extra={"extra": {
                "symbol": symbol, "fill": fill, "spot": spot, "target_strike": round(target),
                "expiry": expiry.isoformat(), "lots": self.executor.tracked_size}})
            await self.notifier.notify(
                NotifyEvent.ENTRY_LONG if direction == SignalDir.LONG.value else NotifyEvent.ENTRY_SHORT,
                direction=otype.value.upper(), contract=symbol or "?", premium=fill,
                btc_price=spot, side="sell")
        finally:
            self.entry_in_progress = False

    # ------------------------------------------------------------------ #
    # TARGET / SL poll (once a minute on the latest closed 1m option candle)
    # ------------------------------------------------------------------ #
    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                await self._check_exit()
                await self._verify_position()
            except Exception as exc:  # noqa: BLE001
                log.error("EmaSide9pm: poll failed", extra={"extra": {"error": str(exc)}})

    async def _closed_candle_premium(self, symbol: str, max_age_sec: int = 120) -> float | None:
        now = int(time.time())
        try:
            candles = await asyncio.to_thread(self.rest.get_candles, symbol, "1m", now - max_age_sec, now)
        except Exception as exc:  # noqa: BLE001
            log.warning("EmaSide9pm: option candle fetch failed — using mark",
                        extra={"extra": {"symbol": symbol, "error": str(exc)}})
            candles = None
        closed = [c for c in (candles or []) if c.start_time + 60 <= now]
        if closed and now - closed[-1].start_time <= max_age_sec:
            return closed[-1].close
        try:
            return await asyncio.to_thread(self.rest.get_mark_price, symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("EmaSide9pm: mark fetch failed", extra={"extra": {"symbol": symbol, "error": str(exc)}})
            return None

    async def _check_exit(self) -> None:
        if self.closing or not self.executor.has_open_position:
            return
        if self._pending_exit:
            await self._close(self._pending_exit)
            return
        symbol = self.executor.tracked_symbol
        if not symbol or not self.entry_premium:
            return
        px = await self._closed_candle_premium(symbol)
        if px is None:
            return
        if px <= self.entry_premium * (1 - self.settings.emaside_target_pct / 100.0):
            await self._close("TARGET")
        elif px >= self.entry_premium * (1 + self.settings.emaside_sl_pct / 100.0):
            await self._close("SL")

    async def _verify_position(self) -> None:
        """Self-heal: if the tracked leg is gone from the exchange (expired, closed by
        hand, liquidated) on two consecutive checks, forget it."""
        if not self.executor.has_open_position or self.closing:
            self._missing_checks = 0
            return
        positions = await asyncio.to_thread(self.rest.get_option_positions, self.executor.underlying)
        symbol = self.executor.tracked_symbol
        if any(p.get("symbol") == symbol for p in positions):
            self._missing_checks = 0
            return
        self._missing_checks += 1
        if self._missing_checks < 2:
            return
        log.warning("EmaSide9pm: tracked leg no longer on the exchange — clearing", extra={"extra": {"symbol": symbol}})
        await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{symbol} vanished from exchange — bot flat")
        self._reset_position()

    def _reset_position(self) -> None:
        self.executor.clear()
        self.entry_premium = None
        self.direction = None
        self._pending_exit = None
        self._missing_checks = 0
        if self.state_file:
            position_state.clear(self.state_file)

    # ------------------------------------------------------------------ #
    async def _fallback_scheduler(self) -> None:
        while True:
            target = _next_occurrence(self.settings.emaside_exit_hour, self.settings.emaside_exit_minute)
            log.info("EmaSide9pm: next square-off", extra={"extra": {"at": target.isoformat()}})
            await asyncio.sleep(max(1.0, (target - datetime.now(_IST)).total_seconds()))
            try:
                if self.executor.has_open_position:
                    await self._close("EOD")
            except Exception as exc:  # noqa: BLE001
                log.error("EmaSide9pm: square-off failed", extra={"extra": {"error": str(exc)}})
            await asyncio.sleep(60)

    async def _close(self, reason: str) -> None:
        if self.closing:
            return
        self.closing = True
        try:
            symbol = self.executor.tracked_symbol
            lots = self.executor.tracked_size
            try:
                fill = await self.executor.close_option()
            except Exception as exc:  # noqa: BLE001
                if not self.executor.has_open_position:
                    # the buy-back was accepted; only the post-order bookkeeping raised
                    log.error("EmaSide9pm: close accepted but errored afterwards — treating as closed",
                              extra={"extra": {"reason": reason, "error": str(exc)}})
                    await self.notifier.notify(NotifyEvent.API_ERROR,
                                               detail=f"{reason} close of {symbol}: {exc} (order was sent)")
                    self._reset_position()
                    return
                self._pending_exit = reason
                log.error("EmaSide9pm: close failed — retrying next minute",
                          extra={"extra": {"reason": reason, "error": str(exc)}})
                await self.notifier.notify(NotifyEvent.API_ERROR, detail=f"{reason} close failed: {exc}")
                return
            entry = self.entry_premium
            pnl = round((entry - fill) * lots * 0.001, 2) if (entry is not None and fill is not None) else None
            log.info("EmaSide9pm exit", extra={"extra": {
                "reason": reason, "symbol": symbol, "entry": entry, "fill": fill, "pnl": pnl}})
            await self.notifier.notify(
                NotifyEvent.EXIT, reason=reason, contract=symbol or "?", entry_premium=entry,
                exit_premium=fill, pnl=pnl, size=lots, side="sell")
            self._reset_position()
        finally:
            self.closing = False

    # ------------------------------------------------------------------ #
    async def _reconcile(self) -> None:
        """Adopt only the state-file leg. Any other open option on this account is
        flagged (Telegram) and blocks entries until it is gone."""
        saved = position_state.load(self.state_file) if self.state_file else None
        owned = saved.get("symbol") if saved else None
        try:
            positions = await asyncio.to_thread(self.rest.get_option_positions, self.executor.underlying)
        except Exception as exc:  # noqa: BLE001
            log.error("EmaSide9pm reconcile: fetch failed", extra={"extra": {"error": str(exc)}})
            if saved and saved.get("product_id"):
                self._adopt(saved, int(saved["product_id"]), int(saved.get("size") or 0), owned)
                log.warning("EmaSide9pm reconcile: exchange unreachable — kept state-file leg",
                            extra={"extra": {"symbol": owned}})
            return
        mine = next((p for p in positions if owned and p.get("symbol") == owned), None)
        self._foreign_symbols = {p["symbol"] for p in positions if p is not mine}
        if mine:
            self._adopt(saved, mine["product_id"], mine["size"], owned)
            log.info("EmaSide9pm reconcile: adopted open leg", extra={"extra": {
                "symbol": owned, "entry_premium": self.entry_premium}})
        else:
            if owned:
                log.info("EmaSide9pm reconcile: state-file leg is closed on the exchange — FLAT",
                         extra={"extra": {"symbol": owned}})
            self._reset_position()
        if self._foreign_symbols:
            log.warning("EmaSide9pm reconcile: untracked option position(s) — entries blocked",
                        extra={"extra": {"symbols": sorted(self._foreign_symbols)}})
            await self.notifier.notify(
                NotifyEvent.API_ERROR,
                detail=f"untracked position(s) {sorted(self._foreign_symbols)} — close them and restart")

    def _adopt(self, saved: dict, product_id: int, size: int, symbol: str | None) -> None:
        otype = OptionType.CALL if (symbol or "").startswith("C-") else OptionType.PUT
        self.executor.adopt(product_id, size, otype, symbol)
        self.entry_premium = saved.get("entry_premium")
        self.direction = saved.get("direction")
