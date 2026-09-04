"""SupertrendSarStrategy v5 -- user-specified strategy: from a fixed daily
start time, the CANDLE that closes at that moment (05:30-05:35 IST by
default) arms a single position -- green candle -> BUY/PE, red -> SELL/CE --
then Stop-And-Reverse (SAR) every time the frozen SL is hit, continuing
until the daily square-off, whereupon it immediately resumes (see EVENING
RESTART below). No EMA filter, no profit target -- sell-mode only.

Version history (each superseded by/layered onto the next per direct
follow-up requests -- kept here so the "why" of the current rules is
traceable):
  * v1: reversal SL = a FRESH Supertrend value read at the reversal moment;
    session reset = calendar-day (IST midnight).
  * v2: reversal SL = the running day-low/day-high instead (see the SAR
    rule below); session reset = a custom 05:30 IST boundary instead of
    midnight (see the SESSION RESET rule below). Both still current.
  * v3: the SESSION'S FIRST entry's DIRECTION no longer reads Supertrend at
    all -- it reads the closing candle's own color instead (see DAY'S
    FIRST ENTRY below). Supertrend is still computed continuously (needed
    for its own ATR warmup) and its value is still used as THIS entry's
    SL -- only the direction choice stopped reading it. Still current.
  * v4: a MINIMUM SL DISTANCE guard on reversals (see min_sl_atr_mult
    below), added after real backtest trade logs showed repeated
    same-session whipsaw -- multiple reversals firing minutes apart right
    after the session reset, because day-high/day-low is still only a few
    bars wide that early and sits too close to price to be a meaningful
    stop. Still current.
  * v5 (current): EVENING RESTART (see below) -- previously the strategy
    sat flat from square-off (17:25 IST) until next session's first entry
    (05:35 IST); now it resumes, in the same direction, 5 minutes after
    square-off (restart_hour:restart_minute, default 17:30 IST) instead of
    waiting for the next session.

Rules:
  * Supertrend(10,3) on real (non-HA) OHLC -- computed continuously across
    the whole backtest, same as every other Supertrend-based strategy here
    (it needs uninterrupted history for its ATR; only the TRADING logic
    below is what resets/restarts each session).
  * SESSION RESET: everything session-scoped -- the running day-high/
    day-low, and the "day's first entry has already fired" flag -- rolls
    over together the instant the clock crosses reset_hour:reset_minute
    (default 05:30 IST), NOT at calendar midnight. This is five minutes
    before the strategy starts hunting for a signal at start_hour:
    start_minute (default 05:35 IST). Detected the same way as the daily
    square-off elsewhere in this repo: a same-bar-vs-previous-bar minute
    crossing, robust to data gaps as long as some earlier bar in the
    session was still below the boundary.
  * DAY'S FIRST ENTRY (v3): the first closed candle each session at or
    after start_hour:start_minute whose evaluation finds the strategy FLAT
    arms the session's first position -- i.e. by default, the candle
    spanning 05:30-05:35 IST itself. Direction ignores Supertrend
    entirely: close > open (a green candle) -> BUY a PUT (PE); close <=
    open (red or a doji) -> SELL a CALL (CE).
  * SL = Supertrend's OWN VALUE at the moment of THIS FIRST entry only,
    frozen from that point on (same "frozen, not trailing" discipline as
    supertrend_fixed_sl.py) -- Supertrend still sets the STOP even though
    it no longer picks the direction.
  * STOP-AND-REVERSE (v2 SL rule, v4 minimum-distance guard): the instant
    real price crosses the frozen SL, that leg closes AND a NEW leg
    immediately opens on the OPPOSITE side. Its fresh frozen SL is NOT
    Supertrend's value anymore -- a reversal INTO A LONG freezes SL at
    that moment's running LOW OF SESSION; a reversal INTO A SHORT freezes
    SL at that moment's running HIGH OF SESSION (the natural support/
    resistance level for that direction). Day-high/day-low are running
    extremes from the session's 05:30 reset through the current bar,
    simply frozen (not trailed further) into activeSL at the instant a
    reversal uses them -- UNLESS that level is closer than
    min_sl_atr_mult * ATR to price (default 1.0x ATR), in which case the
    SL is pushed out to that minimum distance instead (v4 -- prevents the
    day-extreme SL from being so tight, right after a session reset, that
    it triggers rapid same-session whipsaw reversals). Unlike
    supertrend_fixed_sl.py, this strategy is STRICT SINGLE POSITION -- a CE
    and a PE are never open at the same time; closing one and opening the
    other happens atomically within the same closed-bar decision.
  * Continues reversing all session, with no cap on how many times, until
    the daily square-off (a backtest/live-engine concept, NOT modeled
    inside this class -- see scripts/backtest_supertrend_sar.py's own
    --square-off-hour/minute, matching every other strategy here).
  * EVENING RESTART (v5): square-off is external (a force_flat() call the
    strategy doesn't initiate), so this class can't tell WHY it went
    flat -- it just remembers whichever side (self._last_closed_was_short)
    was open at the moment it did, updated on every close (a normal SL-hit
    reversal, or force_flat() itself). At restart_hour:restart_minute
    (default 17:30 IST, 5 minutes after the usual 17:25 square-off), if
    still flat, it immediately resumes in that SAME direction -- SL =
    Supertrend's current value (the same convention as the session's first
    entry, since this is a fresh position, not a reversal). Fires at most
    once per session, the same way the first entry does (an internal
    _active_restart_session marker; force_flat() deliberately does not
    clear it, nor _active_session, nor the running day-high/day-low). From
    here it's an entirely ordinary position -- normal SL-hit reversals
    (day-low/day-high, v2) apply as usual, and it carries straight through
    the next session's 05:30 reset without the day's-first-entry branch
    ever re-firing (it only checks while flat). The backtest/live engine
    is expected to resolve this leg against the FOLLOWING day's option
    expiry like any other post-cutoff entry -- nothing SAR-specific is
    needed for that; it falls out of the shared expiry-selection logic.
  * NO in-strategy profit target at all -- matches supertrend_fixed_sl.py's
    own "pure premium-collection stop-and-hold" design; the frozen SL
    (immediately followed by a reversal) and the daily square-off are the
    only exits.
  * NO EMA filter of any kind (explicitly out of scope per the request).

Execution mapping (same convention as every other strategy here): bearish
(the day's first candle red/close<=open) -> sell a CALL; bullish (green)
-> sell a PUT. On a later reversal within the same session, direction is
simply whichever side is opposite the leg that just stopped out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..models import Candle

__all__ = ["SupertrendSarStrategy", "SupertrendSarDecision"]


class _Rma:
    """Wilder's smoothed moving average (alpha=1/length), seeded with the
    first value -- same convention as this codebase's other _Rma classes."""

    def __init__(self, length: int) -> None:
        self._alpha = 1.0 / length
        self._value: float | None = None

    def update(self, x: float) -> float:
        self._value = x if self._value is None else self._alpha * x + (1 - self._alpha) * self._value
        return self._value


class _Supertrend:
    """Standard ATR-based Supertrend. direction: -1 = uptrend (supertrend
    below price, "positive"), 1 = downtrend (supertrend above price,
    "negative") -- same sign convention as Pine's ta.supertrend() and this
    repo's supertrend_fixed_sl.py."""

    def __init__(self, atr_period: int, factor: float) -> None:
        self.factor = factor
        self._atr = _Rma(atr_period)
        self._prev_close: float | None = None
        self._final_upper: float | None = None
        self._final_lower: float | None = None
        self._value: float | None = None
        self._direction = 0   # 0 = undefined (first bar)

    @property
    def atr(self) -> float | None:
        return self._atr._value

    def update(self, high: float, low: float, close: float) -> tuple[float, int]:
        tr = (high - low) if self._prev_close is None else max(
            high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        atr = self._atr.update(tr)
        hl2 = (high + low) / 2.0
        basic_upper = hl2 + self.factor * atr
        basic_lower = hl2 - self.factor * atr

        if self._final_upper is None:
            final_upper, final_lower = basic_upper, basic_lower
        else:
            final_upper = (basic_upper if (basic_upper < self._final_upper
                                            or self._prev_close > self._final_upper)
                            else self._final_upper)
            final_lower = (basic_lower if (basic_lower > self._final_lower
                                            or self._prev_close < self._final_lower)
                            else self._final_lower)

        if self._direction == 0:
            direction = 1 if close <= final_upper else -1
        elif self._value == self._final_upper:   # previous bar was a downtrend
            direction = 1 if close <= final_upper else -1
        else:                                     # previous bar was an uptrend
            direction = -1 if close >= final_lower else 1

        value = final_upper if direction == 1 else final_lower

        self._prev_close = close
        self._final_upper, self._final_lower = final_upper, final_lower
        self._value, self._direction = value, direction
        return value, direction


@dataclass(frozen=True)
class SupertrendSarDecision:
    candle: Candle
    exit: bool               # a leg just closed (its frozen SL was hit)
    exit_price: float
    exit_was_short: bool     # which side just closed: True=CE/short, False=PE/long
    entry_signal: bool       # a leg is opening -- either the session's first, or an immediate reversal
    entry_is_short: bool     # which side is opening: True=CE/short, False=PE/long
    entry_price: float
    sl_level: float | None   # the frozen SL just set for the newly-opened leg

    @property
    def has_exit(self) -> bool:
        return self.exit

    @property
    def has_entry(self) -> bool:
        return self.entry_signal


class SupertrendSarStrategy:
    def __init__(
        self,
        *,
        atr_period: int = 10,
        factor: float = 3.0,
        day_tz: str = "Asia/Kolkata",
        start_hour: int = 5,
        start_minute: int = 35,
        reset_hour: int = 5,
        reset_minute: int = 30,
        min_sl_atr_mult: float = 1.0,
        restart_hour: int = 17,
        restart_minute: int = 30,
    ) -> None:
        self.atr_period = atr_period
        self.factor = factor
        self._tz = ZoneInfo(day_tz)
        self._start_mins = start_hour * 60 + start_minute
        self._reset_mins = reset_hour * 60 + reset_minute
        self.min_sl_atr_mult = min_sl_atr_mult
        self._restart_mins = restart_hour * 60 + restart_minute
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._st = _Supertrend(self.atr_period, self.factor)
        self._warmup_bars = 0
        self._in_position = False
        self._is_short: bool | None = None   # True=CE/short, False=PE/long, None=flat
        self._active_sl: float | None = None
        # v2 session bookkeeping (05:30 IST reset, not calendar midnight):
        self._prev_now_mins: int | None = None
        self._session_id = 0            # increments each time the reset boundary is crossed
        self._active_session: int | None = None   # session_id whose first-entry already fired
        self._day_high: float | None = None
        self._day_low: float | None = None
        # v5 evening-restart bookkeeping (see EVENING RESTART in the module
        # docstring):
        self._last_closed_was_short: bool | None = None   # side of whichever leg most recently closed
        self._active_restart_session: int | None = None   # session_id whose restart already fired

    @property
    def ready(self) -> bool:
        return self._warmup_bars >= self.atr_period

    @property
    def in_position(self) -> bool:
        return self._in_position

    @property
    def is_short(self) -> bool | None:
        return self._is_short

    def force_flat(self) -> None:
        if self._is_short is not None:
            # Remember which side was open going into this force-flat (e.g.
            # the daily square-off) -- the v5 evening-restart uses this to
            # resume the SAME direction later, without needing to know WHY
            # force_flat() was called.
            self._last_closed_was_short = self._is_short
        self._in_position = False
        self._is_short = None
        self._active_sl = None
        # Deliberately does NOT reset _active_session, _active_restart_
        # session, _day_high, or _day_low -- once a session's first-entry
        # (or evening-restart) signal has fired (whether the fill actually
        # succeeded or not), that trading window is considered used, and
        # the running day-high/low keep accumulating through the rest of
        # the session regardless of position state. Without preserving
        # these markers, a square-off's own force_flat() call would leave
        # the strategy flat with now_mins still >= start_mins/restart_mins
        # for the rest of that session, incorrectly re-arming a fresh
        # entry minutes after square-off instead of waiting for its
        # proper next window.

    def debug_state(self) -> dict:
        def r(x):
            return round(x, 2) if isinstance(x, (int, float)) else None
        return {
            "st_value": r(self._st._value), "st_direction": self._st._direction,
            "atr": r(self._st.atr),
            "in_position": self._in_position, "is_short": self._is_short,
            "active_sl": r(self._active_sl), "session_id": self._session_id,
            "active_session": self._active_session,
            "day_high": r(self._day_high), "day_low": r(self._day_low),
            "last_closed_was_short": self._last_closed_was_short,
            "active_restart_session": self._active_restart_session,
        }

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> SupertrendSarDecision | None:
        st_value, direction = self._st.update(candle.high, candle.low, candle.close)
        self._warmup_bars += 1

        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute

        # v2 session rollover: a minute-crossing of reset_mins (default
        # 05:30 IST), same idiom as the daily square-off elsewhere in this
        # repo -- kept in lockstep with the Pine port's sessionRollover.
        session_rollover = (
            self._prev_now_mins is not None
            and now_mins >= self._reset_mins
            and self._prev_now_mins < self._reset_mins
        )
        self._prev_now_mins = now_mins

        if session_rollover:
            self._session_id += 1
            self._day_high, self._day_low = candle.high, candle.low
        elif self._day_high is None:   # very first bar ever seen this run
            self._day_high, self._day_low = candle.high, candle.low
        else:
            self._day_high = max(self._day_high, candle.high)
            self._day_low = min(self._day_low, candle.low)

        exit_ = False
        exit_price = candle.close
        exit_was_short = False
        entry_signal = False
        entry_is_short = False
        entry_price = candle.close
        new_sl: float | None = None

        if self.ready:
            if self._in_position and self._active_sl is not None:
                hit = (candle.high >= self._active_sl) if self._is_short else (candle.low <= self._active_sl)
                if hit:
                    exit_, exit_price, exit_was_short = True, self._active_sl, self._is_short
                    self._last_closed_was_short = exit_was_short   # v5: remember for a later evening-restart
                    # v2 stop-and-reverse: the OPPOSITE side opens
                    # immediately, with a FRESH frozen SL = the running
                    # day-low (reversal into a long) / day-high (reversal
                    # into a short) -- NOT Supertrend's value, and NOT the
                    # old SL.
                    self._is_short = not self._is_short
                    self._active_sl = self._day_high if self._is_short else self._day_low
                    # v4 minimum-SL-distance guard: right after a session's
                    # reset, day-high/day-low can be only a few bars wide --
                    # too close to price to be a real stop, which was
                    # causing rapid-fire whipsaw reversals (repeated
                    # same-session flips minutes apart, seen in real
                    # backtest trade logs). If the day-extreme SL is closer
                    # than min_sl_atr_mult * ATR to the current price, push
                    # it further out to that minimum distance instead.
                    atr_val = self._st.atr
                    if atr_val is not None and self.min_sl_atr_mult > 0:
                        min_dist = atr_val * self.min_sl_atr_mult
                        if self._is_short:   # SL above price -- widen upward
                            self._active_sl = max(self._active_sl, candle.close + min_dist)
                        else:                 # SL below price -- widen downward
                            self._active_sl = min(self._active_sl, candle.close - min_dist)
                    entry_signal, entry_is_short = True, self._is_short
                    entry_price, new_sl = candle.close, self._active_sl
            elif not self._in_position and self._session_id != self._active_session and now_mins >= self._start_mins:
                # The session's first position (v3): direction ignores
                # Supertrend entirely and instead reads the CANDLE that
                # closes at start_mins itself (the 05:30-05:35 candle by
                # default) -- green (close > open) -> BUY/PE, red
                # (close <= open, a doji included) -> SELL/CE. Supertrend is
                # still computed continuously (needed for its own ATR
                # warmup) and its value is still used for THIS entry's SL --
                # only the DIRECTION choice stopped reading it.
                # self._session_id != self._active_session ensures this can
                # only fire ONCE per session -- see force_flat()'s own
                # comment for why this matters (without it, the square-
                # off's force_flat() would leave the strategy eligible to
                # re-arm minutes later, the same session).
                self._active_session = self._session_id
                self._is_short = candle.close < candle.open
                self._active_sl = st_value
                self._in_position = True
                entry_signal, entry_is_short = True, self._is_short
                entry_price, new_sl = candle.close, st_value
            elif (not self._in_position and self._last_closed_was_short is not None
                  and self._session_id != self._active_restart_session
                  and now_mins >= self._restart_mins):
                # v5 EVENING RESTART: once the daily square-off (an
                # external force_flat() call, not modeled inside this
                # class) has left the strategy flat, don't just sit out
                # until next session's 05:35 first-entry -- at
                # restart_mins (default 17:30 IST, 5 minutes after the
                # usual 17:25 square-off) resume trading immediately, in
                # the SAME direction as whichever leg was open going into
                # that square-off (self._last_closed_was_short, updated on
                # every close: a normal SL-hit reversal above, or
                # force_flat() itself). SL = Supertrend's current value,
                # same convention as the session's first entry (this is a
                # fresh, non-reversal position, so it doesn't borrow the
                # day-low/day-high reversal-SL rule). Gated to fire at most
                # once per session the same way the first entry is; the
                # backtest/live engine is expected to resolve this leg
                # against the FOLLOWING day's option expiry, same as any
                # other post-cutoff entry (see resolve_by_premium's own
                # expiry-cutoff handling -- nothing SAR-specific needed
                # here). Once open, this position is a completely ordinary
                # position from here on -- normal SL-hit reversals (using
                # day-low/day-high, v2 rule) apply exactly as usual, and it
                # naturally carries through the next session's 05:30 reset
                # without the day's-first-entry branch ever re-firing
                # (guarded by "not self._in_position" like everything
                # else here).
                self._active_restart_session = self._session_id
                self._is_short = self._last_closed_was_short
                self._active_sl = st_value
                self._in_position = True
                entry_signal, entry_is_short = True, self._is_short
                entry_price, new_sl = candle.close, st_value

        if not (exit_ or entry_signal):
            return None
        return SupertrendSarDecision(
            candle=candle, exit=exit_, exit_price=exit_price, exit_was_short=exit_was_short,
            entry_signal=entry_signal, entry_is_short=entry_is_short,
            entry_price=entry_price, sl_level=new_sl,
        )
