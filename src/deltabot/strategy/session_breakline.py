"""Session Breakline Strategy -- Python port of session_breakline_strategy.pine.

Heikin Ashi open==extreme reversal pattern, gated by a daily 05:30 PM IST
session line. See session_breakline_strategy.pine for the full rule set and
the assumptions made where the original spec was silent (pattern-candle
line check, no re-anchoring, hunting paused while a position is open, etc.)
-- this port mirrors that script's closed-bar logic exactly, bar for bar.

Rules (LONG side; SHORT is the exact mirror):
  * SESSION LINE: the CLOSE of the first candle at/after 17:30 IST, held
    constant until the next day's 17:30 reset.
  * ANCHOR: a Heikin Ashi candle with open == high (no upper wick), entirely
    above the line (HA low > line).
  * CONFIRM: ANY later HA candle (any gap; candles in between are ignored)
    with open == low (no lower wick), entirely above the line. Sets the
    breakout trigger at THIS candle's HA high. First confirm wins -- no
    re-anchoring.
  * ENTRY: real BTC price breaks ABOVE the trigger (checked against this
    same closed candle's high, matching the Pine script's closed-bar style).
  * SL: NOT a pattern-based level -- exit the instant a REAL candle CLOSES
    back through the session line.
  * NO PROFIT TARGET. Only other exit: forced square-off at 05:25 PM IST
    (5 minutes before the line resets).
  * TRADE CAP: at most one long entry and one short entry per session
    (17:30 -> next 17:25).
  * Pattern hunting for a side only progresses while FLAT (no open position,
    either side) -- avoids a stale pending trigger from ever colliding with
    an open position.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..enums import PositionState
from ..models import Candle

__all__ = ["SessionBreaklineStrategy", "SessionBreaklineDecision"]

_EPS_REL = 1e-9  # tolerant equality for HA-derived float levels


def _close_enough(a: float, b: float, scale: float) -> bool:
    return abs(a - b) <= max(_EPS_REL * scale, 1e-9)


@dataclass(frozen=True)
class SessionBreaklineDecision:
    candle: Candle
    long_exit: bool
    short_exit: bool
    long_exit_price: float
    short_exit_price: float
    buy_signal: bool
    sell_signal: bool
    entry_price: float
    exit_reason: str | None  # "SL" | "EOD" | None

    @property
    def has_exit(self) -> bool:
        return self.long_exit or self.short_exit

    @property
    def has_entry(self) -> bool:
        return self.buy_signal or self.sell_signal


class SessionBreaklineStrategy:
    def __init__(
        self,
        *,
        day_tz: str = "Asia/Kolkata",
        sess_hour: int = 17,
        sess_minute: int = 30,
        sq_off_hour: int = 17,
        sq_off_minute: int = 25,
    ) -> None:
        self.sess_hour = sess_hour
        self.sess_minute = sess_minute
        self.sq_off_hour = sq_off_hour
        self.sq_off_minute = sq_off_minute
        self._tz = ZoneInfo(day_tz)
        self._sess_mins = sess_hour * 60 + sess_minute
        self._sq_mins = sq_off_hour * 60 + sq_off_minute
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._ha_open: float | None = None
        self._ha_close: float | None = None
        self._prev_now_mins: int | None = None

        self._sess_line: float | None = None

        self._long_armed = False
        self._short_armed = False
        self._pending_long_trigger: float | None = None
        self._pending_short_trigger: float | None = None
        self._long_taken = False
        self._short_taken = False

        self._in_long = self._in_short = False

    @property
    def position_state(self) -> PositionState:
        if self._in_long:
            return PositionState.LONG
        if self._in_short:
            return PositionState.SHORT
        return PositionState.FLAT

    @property
    def ready(self) -> bool:
        return self._sess_line is not None

    @property
    def has_pending(self) -> bool:
        return self._pending_long_trigger is not None or self._pending_short_trigger is not None

    @property
    def session_line(self) -> float | None:
        return self._sess_line

    def force_flat(self) -> None:
        self._in_long = self._in_short = False

    def debug_state(self) -> dict:
        def r(x):
            return round(x, 2) if isinstance(x, (int, float)) else None
        return {
            "sess_line": r(self._sess_line),
            "long_armed": self._long_armed, "short_armed": self._short_armed,
            "pending_long_trig": r(self._pending_long_trigger),
            "pending_short_trig": r(self._pending_short_trigger),
            "long_taken": self._long_taken, "short_taken": self._short_taken,
            "pos": self.position_state.name,
        }

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> SessionBreaklineDecision | None:
        # --- Advance the running Heikin Ashi candle ---
        if self._ha_open is None:
            ha_open = (candle.open + candle.close) / 2.0
        else:
            ha_open = (self._ha_open + self._ha_close) / 2.0
        ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0
        ha_high = max(candle.high, ha_open, ha_close)
        ha_low = min(candle.low, ha_open, ha_close)
        self._ha_open, self._ha_close = ha_open, ha_close

        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute
        session_start = (self._prev_now_mins is not None
                          and now_mins >= self._sess_mins and self._prev_now_mins < self._sess_mins)
        square_off = (self._prev_now_mins is not None
                      and now_mins >= self._sq_mins and self._prev_now_mins < self._sq_mins)

        long_exit = short_exit = False
        long_exit_price = short_exit_price = candle.close
        exit_reason: str | None = None
        buy_signal = sell_signal = False
        entry_price = candle.close

        # --- 0. Session reset: new line value always updates; hunting state
        #     only clears while flat (an open position should never still be
        #     alive here -- the 17:25 square-off always precedes it). ---
        if session_start:
            self._sess_line = candle.close
            if not self._in_long and not self._in_short:
                self._long_armed = self._short_armed = False
                self._pending_long_trigger = self._pending_short_trigger = None
                self._long_taken = self._short_taken = False

        # --- 1. Exits: session-line close-through SL takes priority over the
        #     time exit on the same bar (matches the Pine script's ordering). ---
        if self._in_long:
            if self._sess_line is not None and candle.close < self._sess_line:
                long_exit, long_exit_price, exit_reason = True, candle.close, "SL"
            elif square_off:
                long_exit, long_exit_price, exit_reason = True, candle.close, "EOD"
        elif self._in_short:
            if self._sess_line is not None and candle.close > self._sess_line:
                short_exit, short_exit_price, exit_reason = True, candle.close, "SL"
            elif square_off:
                short_exit, short_exit_price, exit_reason = True, candle.close, "EOD"
        if long_exit or short_exit:
            self._in_long = self._in_short = False

        # --- 2. Entry: real-price breakout of a PRIOR-armed pending trigger
        #     (set on an earlier bar -- runs BEFORE pattern hunting below, so
        #     a confirm that arms a trigger THIS bar can only fire starting
        #     the NEXT bar, matching "wait for real price to break" and the
        #     same PRIOR-armed convention used elsewhere in this codebase).
        #     if/elif tie-break mirrors the Pine script. ---
        flat_now = not self._in_long and not self._in_short
        if flat_now and self._pending_long_trigger is not None and candle.high >= self._pending_long_trigger:
            buy_signal, entry_price = True, self._pending_long_trigger
            self._in_long = True
            self._long_taken = True
            self._pending_long_trigger = None
        elif flat_now and self._pending_short_trigger is not None and candle.low <= self._pending_short_trigger:
            sell_signal, entry_price = True, self._pending_short_trigger
            self._in_short = True
            self._short_taken = True
            self._pending_short_trigger = None

        # --- 3. Pattern hunting (anchor -> confirm), only while FLAT. ---
        flat_now = not self._in_long and not self._in_short
        if flat_now and self._sess_line is not None:
            if not self._long_taken and self._pending_long_trigger is None:
                if not self._long_armed:
                    if _close_enough(ha_open, ha_high, candle.close) and ha_low > self._sess_line:
                        self._long_armed = True
                else:
                    if _close_enough(ha_open, ha_low, candle.close) and ha_low > self._sess_line:
                        self._pending_long_trigger = ha_high
                        self._long_armed = False
            if not self._short_taken and self._pending_short_trigger is None:
                if not self._short_armed:
                    if _close_enough(ha_open, ha_low, candle.close) and ha_high < self._sess_line:
                        self._short_armed = True
                else:
                    if _close_enough(ha_open, ha_high, candle.close) and ha_high < self._sess_line:
                        self._pending_short_trigger = ha_low
                        self._short_armed = False

        self._prev_now_mins = now_mins
        if not (long_exit or short_exit or buy_signal or sell_signal):
            return None
        return SessionBreaklineDecision(
            candle=candle, long_exit=long_exit, short_exit=short_exit,
            long_exit_price=long_exit_price, short_exit_price=short_exit_price,
            buy_signal=buy_signal, sell_signal=sell_signal, entry_price=entry_price,
            exit_reason=exit_reason,
        )
