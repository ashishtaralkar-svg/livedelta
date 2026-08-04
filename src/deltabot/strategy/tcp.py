"""TCP (Three Candle Pattern) strategy (option SELL only).

All levels/shapes are computed on synthetic Heikin Ashi (matching DCv2's
formula); "touch" and trigger/SL crossings use REAL candle high/low, same
convention as DCv2. No EMA direction gate -- the sell pattern and buy pattern
are independently hunted at all times.

SELL pattern (three EXACT consecutive bars; any shape mismatch resets the hunt):
  A: HA open == HA low, AND HA high touches/exceeds the Donchian(20) upper
     band (highest high of the prior 20 closed bars).
  B (the immediate next bar): ONLY requirement is a wick on BOTH sides (not
     open==high, not open==low) -- no constraint relative to A's high/low.
     RE-ANCHOR: if this bar fails the B shape but is ITSELF a valid fresh A
     (open==low + touches the upper band -- e.g. this always happens when the
     "candidate B" bar is itself open==low, since that shape can never have a
     lower wick), it becomes the new candle 1 immediately instead of being
     discarded while the hunt waits for a separate future bar.
  C (the immediate next bar after B): HA open == HA high; C.low < B.low AND
     C.low < A.low (the pattern must trend lower than where it started, not
     just lower than B -- B has no relation constraint to A, so C<B alone
     doesn't guarantee that).
  Pattern complete on C -> ARM a pending SELL:
     trigger = min(A.low, B.low, C.low)   (= C.low by construction)
     sl      = max(A.high, B.high, C.high)
  ENTRY: REAL price breaks below the trigger -> SELL signal (sell a CALL).
     Pre-entry invalidation (conservative, same as DCv2): if real price
     touches the SL side FIRST, the armed pattern is discarded untraded.
  EXIT: SL only (real price back up to the pattern high). The 70%
     premium-decay TP is handled externally by the option-backtest/engine,
     same as DCv2 -- this strategy only tracks the underlying-price SL.

BUY pattern is the exact mirror (Donchian LOWER band, open==high touch,
open==low confirm; trigger = pattern high, sl = pattern low; sell a PUT).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..enums import PositionState
from ..models import Candle

__all__ = ["TCPStrategy", "TCPDecision"]


def _close_enough(a: float, b: float, scale: float) -> bool:
    return abs(a - b) <= max(1e-9 * scale, 1e-9)


class _Donchian:
    """Highest-high / lowest-low over the PRIOR ``period`` closed bars --
    the current bar is excluded until pushed at the end of ``update()``."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._highs: deque[float] = deque(maxlen=period)
        self._lows: deque[float] = deque(maxlen=period)

    @property
    def ready(self) -> bool:
        return len(self._highs) >= self.period

    @property
    def upper(self) -> float | None:
        return max(self._highs) if self._highs else None

    @property
    def lower(self) -> float | None:
        return min(self._lows) if self._lows else None

    def push(self, high: float, low: float) -> None:
        self._highs.append(high)
        self._lows.append(low)


@dataclass(frozen=True)
class TCPDecision:
    candle: Candle
    long_exit: bool
    short_exit: bool
    long_exit_price: float
    short_exit_price: float
    buy_signal: bool
    sell_signal: bool
    entry_price: float
    exit_reason: str | None   # "SL" | None
    sl_level: float | None    # the SL active right after a just-opened position

    @property
    def has_exit(self) -> bool:
        return self.long_exit or self.short_exit

    @property
    def has_entry(self) -> bool:
        return self.buy_signal or self.sell_signal


class TCPStrategy:
    def __init__(self, *, dc_period: int = 20, use_heikin_ashi: bool = True,
                skip_weekdays: frozenset[int] = frozenset(),
                day_tz: str = "Asia/Kolkata", target_rr: float = 0.0,
                daily_filter_days: int = 0) -> None:
        self.dc_period = dc_period
        self.use_heikin_ashi = use_heikin_ashi
        # skip_weekdays gates only the TRADE itself (a trigger hit on a
        # blocked day consumes the pending setup untraded) -- pattern
        # formation/arming continues regardless, same convention as DCv2.
        self.skip_weekdays = skip_weekdays
        self._tz = ZoneInfo(day_tz)
        # target_rr > 0: on entry, set a fixed BTC-price profit target at
        # entry +/- risk*target_rr, where risk = |entry - sl_level| (the
        # pattern's own SL distance). E.g. target_rr=2.0 -> a 1:2 risk:reward
        # target. Checked on every bar alongside the fixed SL (SL takes
        # priority if both would fire the same bar). 0 = off (SL-only exit).
        self.target_rr = target_rr
        # daily_filter_days > 0: an additional DAILY-timeframe bias gate on
        # the trade itself (pattern formation/arming continues regardless,
        # same convention as skip_weekdays). A SELL only fires if AT LEAST
        # ONE of the last daily_filter_days CLOSED daily candles (IST
        # calendar day) was open==high (a bearish, no-upper-wick day). A BUY
        # only fires if any of them was open==low (mirror). 0 = off.
        self.daily_filter_days = daily_filter_days
        self.reset()

    def reset(self) -> None:
        self._dc = _Donchian(self.dc_period)
        self._warmup_bars = 0
        self._ha_open: float | None = None
        self._ha_close: float | None = None

        # Pattern-match state machines, independent per direction:
        # "idle" -> (found A) "got_a" -> (found B) "got_ab" -> (C checked, always back to idle)
        self._sell_state = "idle"
        self._sell_a: dict | None = None
        self._sell_b: dict | None = None
        self._buy_state = "idle"
        self._buy_a: dict | None = None
        self._buy_b: dict | None = None

        # Daily-timeframe aggregation for the daily_filter_days bias gate.
        self._cur_day = None
        self._day_open: float | None = None
        self._day_high: float | None = None
        self._day_low: float | None = None
        n = max(self.daily_filter_days, 1)
        self._daily_open_high: deque[bool] = deque(maxlen=n)
        self._daily_open_low: deque[bool] = deque(maxlen=n)

        # Armed pending setup (post pattern-complete); only one at a time.
        self._pending_short = False
        self._pending_long = False
        self._pending_trigger: float | None = None
        self._pending_sl: float | None = None

        # Open position.
        self._in_long = self._in_short = False
        self._sl_level: float | None = None
        self._target_level: float | None = None

    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        return self._dc.ready

    @property
    def position_state(self) -> PositionState:
        if self._in_long:
            return PositionState.LONG
        if self._in_short:
            return PositionState.SHORT
        return PositionState.FLAT

    @property
    def sl_level(self) -> float | None:
        return self._sl_level

    @property
    def has_pending(self) -> bool:
        return self._pending_long or self._pending_short

    def enter_tp_cooldown(self) -> None:
        """No-op: this strategy has no TP-cooldown concept (called
        unconditionally by the shared option-backtest runner after a TP)."""

    def debug_state(self) -> dict:
        """Per-candle diagnostic snapshot for the live engine's optional
        DELTA_DCV2_DEBUG_STATE logging."""
        def r(x):
            return round(x, 2) if isinstance(x, (int, float)) else None
        return {
            "sell_state": self._sell_state, "buy_state": self._buy_state,
            "pending_long": self._pending_long, "pending_short": self._pending_short,
            "pending_trig": r(self._pending_trigger), "pending_sl": r(self._pending_sl),
            "pos": self.position_state.name, "sl_level": r(self._sl_level),
            "target_level": r(self._target_level),
        }

    # ------------------------------------------------------------------ #
    # Intracandle (live/ASAP) helpers -- same conventions as DCv2Strategy.
    # ------------------------------------------------------------------ #
    def check_intracandle_sl(self, price: float) -> tuple[bool, bool, float | None]:
        long_sl = self._in_long and self._sl_level is not None and price <= self._sl_level
        short_sl = self._in_short and self._sl_level is not None and price >= self._sl_level
        return bool(long_sl), bool(short_sl), self._sl_level

    def apply_intracandle_pending(self, candle: Candle) -> tuple[bool, bool, float]:
        """ASAP entry/invalidation against REAL price -- SL side checked BEFORE
        the trigger (conservative), same convention as DCv2. Returns
        ``(confirmed, invalidated, entry_price)``. Day-of-week blocking is the
        live engine's job (wall-clock _entries_blocked()); only the daily
        open==extreme bias filter is checked here, same as the closed-bar path."""
        if self._pending_long and self._pending_trigger is not None and self._pending_sl is not None:
            trig, sl = self._pending_trigger, self._pending_sl
            if candle.open <= sl or candle.low <= sl:
                self._pending_long = False
                self._pending_trigger = self._pending_sl = None
                return False, True, 0.0
            if candle.open >= trig or candle.high >= trig:
                if not self._daily_filter_ok(is_long=True):
                    self._pending_long = False
                    self._pending_trigger = self._pending_sl = None
                    return False, False, 0.0
                self._in_long, self._in_short = True, False
                self._sl_level = sl
                self._target_level = self._target_for(trig, sl, is_long=True)
                self._pending_long = False
                self._pending_trigger = self._pending_sl = None
                return True, False, trig
        elif self._pending_short and self._pending_trigger is not None and self._pending_sl is not None:
            trig, sl = self._pending_trigger, self._pending_sl
            if candle.open >= sl or candle.high >= sl:
                self._pending_short = False
                self._pending_trigger = self._pending_sl = None
                return False, True, 0.0
            if candle.open <= trig or candle.low <= trig:
                if not self._daily_filter_ok(is_long=False):
                    self._pending_short = False
                    self._pending_trigger = self._pending_sl = None
                    return False, False, 0.0
                self._in_short, self._in_long = True, False
                self._sl_level = sl
                self._target_level = self._target_for(trig, sl, is_long=False)
                self._pending_short = False
                self._pending_trigger = self._pending_sl = None
                return True, False, trig
        return False, False, 0.0

    def _update_daily(self, candle: Candle) -> None:
        """Roll the incoming 5m candle into the running DAILY (IST calendar
        day) bar. On a day boundary, finalize the just-completed day's
        open==high / open==low shape into the rolling window."""
        if self.daily_filter_days <= 0:
            return
        day = datetime.fromtimestamp(candle.start_time, tz=self._tz).date()
        if self._cur_day is None:
            self._cur_day = day
            self._day_open, self._day_high, self._day_low = candle.open, candle.high, candle.low
            return
        if day != self._cur_day:
            self._daily_open_high.append(_close_enough(self._day_open, self._day_high, self._day_open))
            self._daily_open_low.append(_close_enough(self._day_open, self._day_low, self._day_open))
            self._cur_day = day
            self._day_open, self._day_high, self._day_low = candle.open, candle.high, candle.low
        else:
            self._day_high = max(self._day_high, candle.high)
            self._day_low = min(self._day_low, candle.low)

    def _daily_filter_ok(self, is_long: bool) -> bool:
        """Off -> always True. On: needs at least one open==low (buy) /
        open==high (sell) day among the last daily_filter_days CLOSED days."""
        if self.daily_filter_days <= 0:
            return True
        return any(self._daily_open_low if is_long else self._daily_open_high)

    def _target_for(self, entry_price: float, sl_level: float | None, is_long: bool) -> float | None:
        if self.target_rr <= 0 or sl_level is None:
            return None
        risk = abs(entry_price - sl_level)
        return entry_price + risk * self.target_rr if is_long else entry_price - risk * self.target_rr

    def force_flat(self) -> None:
        self._in_long = self._in_short = False
        self._sl_level = None
        self._target_level = None
        self._pending_long = self._pending_short = False
        self._pending_trigger = self._pending_sl = None
        self._sell_state = self._buy_state = "idle"
        self._sell_a = self._sell_b = self._buy_a = self._buy_b = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _wick_both_sides(o: float, h: float, l: float, c: float) -> bool:
        body_top, body_bot = max(o, c), min(o, c)
        return h > body_top and l < body_bot

    def _progress_sell(self, bo: float, bh: float, bl: float, bc: float,
                       dc_upper: float | None, close_scale: float) -> None:
        if self._sell_state == "idle":
            if (dc_upper is not None and _close_enough(bo, bl, close_scale)
                    and bh >= dc_upper):
                self._sell_a = {"high": bh, "low": bl}
                self._sell_state = "got_a"
            return
        if self._sell_state == "got_a":
            if self._wick_both_sides(bo, bh, bl, bc):
                self._sell_b = {"high": bh, "low": bl}
                self._sell_state = "got_ab"
            elif (dc_upper is not None and _close_enough(bo, bl, close_scale)
                    and bh >= dc_upper):
                # This bar failed as B, but it independently qualifies as a
                # FRESH candle A (e.g. an open==low bar can never have a lower
                # wick, so it always fails B) -- re-anchor to THIS bar as the
                # new candle 1 instead of discarding it and waiting for a
                # separate future bar.
                self._sell_a = {"high": bh, "low": bl}
            else:
                self._sell_state, self._sell_a = "idle", None
            return
        if self._sell_state == "got_ab":
            a, b = self._sell_a, self._sell_b
            if _close_enough(bo, bh, close_scale) and bl < b["low"] and bl < a["low"]:
                pattern_low = min(a["low"], b["low"], bl)
                pattern_high = max(a["high"], b["high"], bh)
                self._pending_short = True
                self._pending_trigger, self._pending_sl = pattern_low, pattern_high
            self._sell_state, self._sell_a, self._sell_b = "idle", None, None

    def _progress_buy(self, bo: float, bh: float, bl: float, bc: float,
                      dc_lower: float | None, close_scale: float) -> None:
        if self._buy_state == "idle":
            if (dc_lower is not None and _close_enough(bo, bh, close_scale)
                    and bl <= dc_lower):
                self._buy_a = {"high": bh, "low": bl}
                self._buy_state = "got_a"
            return
        if self._buy_state == "got_a":
            if self._wick_both_sides(bo, bh, bl, bc):
                self._buy_b = {"high": bh, "low": bl}
                self._buy_state = "got_ab"
            elif (dc_lower is not None and _close_enough(bo, bh, close_scale)
                    and bl <= dc_lower):
                # Same re-anchor as the sell side: this bar failed as B but is
                # itself a fresh valid A -- re-anchor to it immediately.
                self._buy_a = {"high": bh, "low": bl}
            else:
                self._buy_state, self._buy_a = "idle", None
            return
        if self._buy_state == "got_ab":
            a, b = self._buy_a, self._buy_b
            if _close_enough(bo, bl, close_scale) and bh > b["high"] and bh > a["high"]:
                pattern_high = max(a["high"], b["high"], bh)
                pattern_low = min(a["low"], b["low"], bl)
                self._pending_long = True
                self._pending_trigger, self._pending_sl = pattern_high, pattern_low
            self._buy_state, self._buy_a, self._buy_b = "idle", None, None

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> TCPDecision | None:
        if self._ha_open is None:
            ha_open = (candle.open + candle.close) / 2.0
        else:
            ha_open = (self._ha_open + self._ha_close) / 2.0
        ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0
        ha_high = max(candle.high, ha_open, ha_close)
        ha_low = min(candle.low, ha_open, ha_close)
        self._ha_open, self._ha_close = ha_open, ha_close

        if self.use_heikin_ashi:
            bo, bh, bl, bc = ha_open, ha_high, ha_low, ha_close
        else:
            bo, bh, bl, bc = candle.open, candle.high, candle.low, candle.close

        dc_upper, dc_lower = self._dc.upper, self._dc.lower
        self._warmup_bars += 1
        self._update_daily(candle)
        day_blocked = datetime.fromtimestamp(candle.start_time, tz=self._tz).weekday() in self.skip_weekdays

        long_exit = short_exit = False
        long_exit_price = short_exit_price = candle.close
        exit_reason: str | None = None
        buy_signal = sell_signal = False
        entry_price = candle.close
        new_sl: float | None = None

        # --- 1/2. Exit: fixed SL only (pattern high/low from the entry). ---
        if self._in_long:
            if self._sl_level is not None and candle.low <= self._sl_level:
                long_exit, long_exit_price, exit_reason = True, self._sl_level, "SL"
            elif self._target_level is not None and candle.high >= self._target_level:
                long_exit, long_exit_price, exit_reason = True, self._target_level, "TARGET"
        elif self._in_short:
            if self._sl_level is not None and candle.high >= self._sl_level:
                short_exit, short_exit_price, exit_reason = True, self._sl_level, "SL"
            elif self._target_level is not None and candle.low <= self._target_level:
                short_exit, short_exit_price, exit_reason = True, self._target_level, "TARGET"
        if long_exit or short_exit:
            self._in_long = self._in_short = False
            self._sl_level = None
            self._target_level = None

        # --- 3. Breakout trigger for an armed pending setup (REAL price).
        #     SL side hit first invalidates the setup untraded (conservative,
        #     same convention as DCv2). ---
        flat = not self._in_long and not self._in_short
        if flat and self._pending_short:
            trig, sl = self._pending_trigger, self._pending_sl
            if candle.high >= sl:
                self._pending_short = False
                self._pending_trigger = self._pending_sl = None
            elif candle.low <= trig:
                if not day_blocked and self._daily_filter_ok(is_long=False):
                    sell_signal, entry_price, new_sl = True, trig, sl
                    self._in_short, self._sl_level = True, sl
                    self._target_level = self._target_for(trig, sl, is_long=False)
                self._pending_short = False   # consumed (traded or blocked)
                self._pending_trigger = self._pending_sl = None
        elif flat and self._pending_long:
            trig, sl = self._pending_trigger, self._pending_sl
            if candle.low <= sl:
                self._pending_long = False
                self._pending_trigger = self._pending_sl = None
            elif candle.high >= trig:
                if not day_blocked and self._daily_filter_ok(is_long=True):
                    buy_signal, entry_price, new_sl = True, trig, sl
                    self._in_long, self._sl_level = True, sl
                    self._target_level = self._target_for(trig, sl, is_long=True)
                self._pending_long = False   # consumed (traded or blocked)
                self._pending_trigger = self._pending_sl = None

        # --- 4. Pattern progression (HA-based), only while flat/no pending. ---
        flat_now = not self._in_long and not self._in_short
        if flat_now and not self.has_pending and self.ready:
            self._progress_sell(bo, bh, bl, bc, dc_upper, candle.close)
            self._progress_buy(bo, bh, bl, bc, dc_lower, candle.close)

        self._dc.push(bh, bl)

        return TCPDecision(
            candle=candle, long_exit=long_exit, short_exit=short_exit,
            long_exit_price=long_exit_price, short_exit_price=short_exit_price,
            buy_signal=buy_signal, sell_signal=sell_signal,
            entry_price=entry_price, exit_reason=exit_reason, sl_level=new_sl,
        )
