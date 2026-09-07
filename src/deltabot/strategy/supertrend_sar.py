"""SupertrendSarStrategy v9 -- user-specified strategy: the candle that
closes at start_mins (05:35 PM IST by default) arms a single position by
its own color -- green -> BUY/PE, red -> SELL/CE -- then Stop-And-Reverse
(SAR) every time the frozen SL is hit, continuing until the daily
square-off, whereupon it resumes shortly after (see EVENING RESTART
below). No EMA filter, no profit target -- sell-mode only. An Opening
Range Breakout (ORB) alternative for the first-entry direction (v7) is
still available, opt-in via orb_enabled=True -- see v9's own history note
below for why it's off by default.

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
  * v5: EVENING RESTART (see below) -- previously the strategy sat flat
    from square-off (17:25 IST) until next session's first entry (05:35
    IST); now it resumes, in the same direction, 5 minutes after
    square-off (restart_hour:restart_minute, default 17:30 IST) instead of
    waiting for the next session. Still current.
  * v6: ASAP (intracandle) SL/reversal -- check_intracandle_sl()/
    apply_intracandle_reversal() let the ENGINE react to the frozen SL
    being touched by REAL price the instant it happens, instead of waiting
    for the bar to close (this class's own update() still does the
    closed-bar version, as a fallback/backtest path -- the two are never
    both applied for the same touch, that's the engine's job to guard).
    The entry side (day's-first-entry, evening restart) is NOT ASAP --
    at the time this was written both needed the closing candle's own
    color/close-vs-open, which can't be known before the bar actually
    closes. Still current (ASAP SL/reversal), though v7 below changes
    what "the entry side" even means for the day's first entry.
  * v7: the SESSION'S FIRST entry no longer reads Supertrend OR the
    candle's own color at all -- it's now an Opening Range Breakout (ORB):
    orb_start_mins-orb_end_mins (default 17:30-18:00 IST) is watched as a
    range, and the first candle at/after orb_end_mins whose high/low
    breaks it arms the position in that direction (see DAY'S FIRST ENTRY
    below). When first shipped, orb_start_mins defaulted to the SAME
    moment as restart_mins (17:30 both) -- on any day the strategy was
    already flat right at 17:30, the v5 EVENING RESTART's own elif branch
    fired FIRST and claimed that slot before the ORB range even finished
    building, since restart only needed now_mins >= restart_mins (no
    range-break condition to wait for), so ORB almost never actually
    fired. Fixed in v8 below by moving restart_mins later. Made OPT-IN in
    v9 (default off) -- see that entry.
  * v8: TWO fixes based on real backtest trade-log review after v7
    shipped:
    (a) restart_hour:restart_minute moved from 17:30 to 18:05 (5 minutes
        after orb_end_mins) so ORB gets first crack at the day's first
        entry every session, with restart as the fallback for whatever
        ORB doesn't catch (was previously starving ORB out entirely, see
        the v7 note above). Reverted back to 17:30 in v9 alongside ORB
        being turned off by default -- see that entry.
    (b) reset_hour:reset_minute (the SESSION RESET boundary that drives
        day-high/day-low) moved from 05:30 to 17:30 -- a real backtest
        trade was found where a reversal's SL (the day-high) had been set
        by a price spike from EARLY THAT MORNING, hours before the
        evening session the position actually traded in even started, and
        that same stale level then carried through a TP-roll into a much
        later SL exit. Aligning the session reset to 17:30 means
        day-high/day-low -- and therefore every reversal SL for the rest
        of the evening -- only ever reflects price action from the
        CURRENT session onward. Still current regardless of orb_enabled
        (this fix is unrelated to ORB specifically). See the SESSION RESET
        rule below.
  * v9 (current): ORB (v7) made OPT-IN via orb_enabled (default False) per
    direct follow-up request -- with it off, the session's first entry
    goes back to the v3 candle-color rule (start_hour:start_minute,
    default 17:35 IST -- moved here from the original 05:35 to match the
    config validated as the winning baseline across this session's own
    1wk/1mo/3mo comparisons, back when v7/v8 were being evaluated against
    it). restart_hour:restart_minute also reverted to 17:30 (was 18:05) --
    that move only existed to give ORB a chance to fire; without ORB
    active there is no longer a range that needs time to finish building,
    so restart can resume immediately after square-off again, same as v5
    originally intended.

Rules:
  * Supertrend(10,3) on real (non-HA) OHLC -- computed continuously across
    the whole backtest, same as every other Supertrend-based strategy here
    (it needs uninterrupted history for its ATR; only the TRADING logic
    below is what resets/restarts each session).
  * SESSION RESET (v8: moved to 17:30, was 05:30): everything
    session-scoped -- the running day-high/day-low, and the "day's first
    entry has already fired" flag -- rolls over together the instant the
    clock crosses reset_hour:reset_minute (default 17:30 IST), NOT at
    calendar midnight. This is now the SAME moment orb_start_mins begins
    (deliberately -- see the v8 history note below): a reversal's SL
    (day-low/day-high) should only ever reflect price action from the
    CURRENT evening session onward, not carry over stale extremes from
    hours earlier in the same calendar day. Detected the same way as the
    daily square-off elsewhere in this repo: a same-bar-vs-previous-bar
    minute crossing, robust to data gaps as long as some earlier bar in
    the session was still below the boundary.
  * DAY'S FIRST ENTRY -- orb_enabled=False (v9, the default): the first
    closed candle each session at or after start_mins (default 17:35 IST)
    arms the session's first position by its own color: green (close >
    open) -> BUY a PUT (PE); red (close <= open, a doji included) -> SELL
    a CALL (CE). orb_enabled=True (v7, opt-in): orb_start_mins-orb_end_mins
    (default 17:30-18:00 IST) is watched each session as an Opening Range
    instead -- every bar in that window (while the strategy is FLAT or
    not, tracking doesn't care) extends orb_high/orb_low, and the first
    closed candle AT OR AFTER orb_end_mins whose own high/low breaks that
    range arms the position: broke below orb_low -> SELL a CALL (CE);
    broke above orb_high -> BUY a PUT (PE). (A bar that breaks both sides
    at once -- a big range or a gap -- follows the CLOSE instead: above
    the range -> PE, below -> CE, still inside -> close vs the range
    midpoint.) Either way, Supertrend itself is never consulted for
    direction.
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
    extremes from the session's 17:30 reset through the current bar,
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
  * EVENING RESTART (v5; timing moved to 18:05 in v8 for ORB's sake, back
    to 17:30 in v9 now that ORB is off by default): square-off is external
    (a force_flat() call the strategy doesn't initiate), so this class
    can't tell WHY it went flat -- it just remembers whichever side
    (self._last_closed_was_short) was open at the moment it did, updated
    on every close (a normal SL-hit reversal, or force_flat() itself). At
    restart_hour:restart_minute (default 17:30 IST, 5 minutes after the
    usual 17:25 square-off), if still flat, it immediately resumes in
    that SAME direction -- SL = Supertrend's current value (the same
    convention as the session's first entry, since this is a fresh
    position, not a reversal). Fires at most once per session, the same
    way the first entry does (an internal _active_restart_session marker;
    force_flat() deliberately does not
    clear it, nor _active_session, nor the running day-high/day-low). From
    here it's an entirely ordinary position -- normal SL-hit reversals
    (day-low/day-high, v2) apply as usual, and it carries straight through
    the next session's 17:30 reset without the day's-first-entry branch
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
        start_hour: int = 17,
        start_minute: int = 35,
        reset_hour: int = 17,
        reset_minute: int = 30,
        min_sl_atr_mult: float = 1.0,
        restart_hour: int = 17,
        restart_minute: int = 30,
        orb_enabled: bool = False,
        orb_start_hour: int = 17,
        orb_start_minute: int = 30,
        orb_end_hour: int = 18,
        orb_end_minute: int = 0,
    ) -> None:
        self.atr_period = atr_period
        self.factor = factor
        self._tz = ZoneInfo(day_tz)
        # v9: ORB (v7) is now OPT-IN via orb_enabled (default False, i.e.
        # REMOVED from the default behavior per direct follow-up request --
        # see the v9 history note below). With it off, the session's first
        # entry goes back to the v3 candle-color rule using start_hour/
        # start_minute; with it on, start_hour/start_minute are unused (see
        # the old v7 comment, still true in that case).
        self._start_mins = start_hour * 60 + start_minute
        self._reset_mins = reset_hour * 60 + reset_minute
        self.min_sl_atr_mult = min_sl_atr_mult
        self._restart_mins = restart_hour * 60 + restart_minute
        self.orb_enabled = orb_enabled
        self._orb_start_mins = orb_start_hour * 60 + orb_start_minute
        self._orb_end_mins = orb_end_hour * 60 + orb_end_minute
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._st = _Supertrend(self.atr_period, self.factor)
        self._warmup_bars = 0
        self._in_position = False
        self._is_short: bool | None = None   # True=CE/short, False=PE/long, None=flat
        self._active_sl: float | None = None
        # v2 session bookkeeping (v8: reset moved to 17:30 IST, not calendar midnight):
        self._prev_now_mins: int | None = None
        self._session_id = 0            # increments each time the reset boundary is crossed
        self._active_session: int | None = None   # session_id whose first-entry already fired
        self._day_high: float | None = None
        self._day_low: float | None = None
        # v5 evening-restart bookkeeping (see EVENING RESTART in the module
        # docstring):
        self._last_closed_was_short: bool | None = None   # side of whichever leg most recently closed
        self._active_restart_session: int | None = None   # session_id whose restart already fired
        # v7 Opening-Range-Breakout (ORB) bookkeeping (see DAY'S FIRST ENTRY
        # in the module docstring):
        self._orb_high: float | None = None
        self._orb_low: float | None = None
        self._orb_range_session: int | None = None   # session_id the current orb_high/low belongs to

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

    # ------------------------------------------------------------------ #
    # v6 ASAP (intracandle) execution -- mirrors this repo's established
    # HeikinAshiStrategy pattern (check_intracandle_sl/notify_exit): the SL
    # is checked against REAL price on every forming-candle tick instead of
    # waiting for the bar to close, so a stop fires the instant price
    # crosses it rather than up to one full bar late. Split the same way
    # HeikinAshi splits it: check_intracandle_sl() is a PURE check (no
    # mutation, safe to call every tick); apply_intracandle_reversal() is
    # called by the ENGINE only AFTER it has confirmed the exit fired (and,
    # for SAR, after the reversal leg's order actually went through) --
    # deferring the mutation until the trade is real, same safety
    # discipline as HeikinAshiEngine's notify_exit() call ordering.
    #
    # Unlike HeikinAshi (which just goes flat, no forced re-entry),
    # SupertrendSarStrategy ALWAYS reverses -- so apply_intracandle_reversal
    # both closes the old leg and opens the new one in a single mutation,
    # exactly mirroring update()'s own closed-bar reversal branch, just
    # driven by a real-time price instead of a closed candle's high/low.
    # The entry side of this strategy (day's-first-entry, evening restart)
    # is NOT made ASAP here -- both depend on the closing candle's own
    # color/close-vs-open, which is fundamentally unknowable before the bar
    # actually closes, so they stay on the closed-bar path.
    def check_intracandle_sl(self, price: float) -> tuple[bool, float | None]:
        """Has an intracandle REAL price crossed the open leg's frozen SL?
        -> (hit, level). Pure -- never mutates state, safe to call on every
        forming-candle tick."""
        if not self._in_position or self._active_sl is None:
            return False, None
        hit = (price >= self._active_sl) if self._is_short else (price <= self._active_sl)
        return bool(hit), self._active_sl

    def apply_intracandle_reversal(self, tick_price: float) -> tuple[bool, float]:
        """Call ONLY after check_intracandle_sl() returned a hit AND the
        engine has confirmed the reversal leg's order actually went
        through. Mutates state exactly like update()'s own closed-bar
        reversal branch (day-low/day-high SL, widened by min_sl_atr_mult *
        ATR if that would be tighter) -- the one difference is day-high/
        day-low are extended one step further using `tick_price` itself
        (the real price at this exact moment, ahead of whatever the still-
        forming candle's eventual close will be) so the fresh SL reflects
        the most current information available; the closed-bar update()
        later this bar naturally continues extending day-high/day-low from
        there, so this is a forward-looking peek, not a double-count.
        Returns (new_is_short, new_sl) for the engine's own notify/state.
        """
        if self._day_high is None or tick_price > self._day_high:
            self._day_high = tick_price
        if self._day_low is None or tick_price < self._day_low:
            self._day_low = tick_price
        self._is_short = not self._is_short
        self._active_sl = self._day_high if self._is_short else self._day_low
        atr_val = self._st.atr
        if atr_val is not None and self.min_sl_atr_mult > 0:
            min_dist = atr_val * self.min_sl_atr_mult
            if self._is_short:
                self._active_sl = max(self._active_sl, tick_price + min_dist)
            else:
                self._active_sl = min(self._active_sl, tick_price - min_dist)
        self._last_closed_was_short = not self._is_short   # the side that just closed
        return self._is_short, self._active_sl

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
            "orb_high": r(self._orb_high), "orb_low": r(self._orb_low),
        }

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> SupertrendSarDecision | None:
        st_value, direction = self._st.update(candle.high, candle.low, candle.close)
        self._warmup_bars += 1

        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute

        # v2 session rollover: a minute-crossing of reset_mins (default
        # 17:30 IST as of v8, was 05:30), same idiom as the daily
        # square-off elsewhere in this repo.
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

        # v7 ORB (Opening Range Breakout) range tracking: build orb_high/
        # orb_low fresh each session from every bar inside
        # orb_start_mins-orb_end_mins (default 17:30-18:00 IST); the day's
        # first entry (below) fires on a breakout of that finalized range,
        # not the candle's own color. Runs unconditionally (like day-high/
        # day-low above), not gated on self.ready, so the range is already
        # correctly built by the time warmup completes mid-window.
        if self._orb_range_session != self._session_id:
            self._orb_range_session = self._session_id
            self._orb_high = None
            self._orb_low = None
        if self._orb_start_mins <= now_mins < self._orb_end_mins:
            self._orb_high = candle.high if self._orb_high is None else max(self._orb_high, candle.high)
            self._orb_low = candle.low if self._orb_low is None else min(self._orb_low, candle.low)

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
            elif not self._in_position and self._session_id != self._active_session and (
                (self.orb_enabled and now_mins >= self._orb_end_mins
                 and self._orb_high is not None and self._orb_low is not None
                 and (candle.high > self._orb_high or candle.low < self._orb_low))
                or (not self.orb_enabled and now_mins >= self._start_mins)
            ):
                # The session's first position. TWO mutually exclusive
                # modes, chosen by orb_enabled (v9 -- default False, i.e.
                # ORB removed from the default behavior per direct
                # follow-up request; still available opt-in):
                #   orb_enabled=True (v7 ORB): direction ignores both
                #     Supertrend AND the candle's own color -- the
                #     orb_start_mins-orb_end_mins window (default
                #     17:30-18:00 IST) is watched as an Opening Range, and
                #     the first candle AT OR AFTER orb_end_mins whose
                #     high/low breaks that range arms the position: broke
                #     below orb_low -> bearish -> SELL/CE; broke above
                #     orb_high -> bullish -> BUY/PE. (On the rare bar that
                #     breaks BOTH sides at once -- a big range or a gap --
                #     direction follows whichever side the CLOSE ended up
                #     beyond; if the close is back inside the range too,
                #     fall back to close vs the range midpoint.)
                #   orb_enabled=False (v3, the current default): the
                #     candle that closes at start_mins itself (default
                #     05:35 IST) decides direction by its own color --
                #     green (close > open) -> BUY/PE, red (close <= open,
                #     a doji included) -> SELL/CE.
                # Either way Supertrend is still computed continuously
                # (needed for its own ATR warmup) and its value is still
                # used for THIS entry's SL. self._session_id !=
                # self._active_session ensures this can only fire ONCE per
                # session -- see force_flat()'s own comment for why this
                # matters (without it, the square-off's force_flat() would
                # leave the strategy eligible to re-arm minutes later, the
                # same session).
                self._active_session = self._session_id
                if self.orb_enabled:
                    broke_up = candle.high > self._orb_high
                    broke_down = candle.low < self._orb_low
                    if broke_down and not broke_up:
                        self._is_short = True
                    elif broke_up and not broke_down:
                        self._is_short = False
                    elif candle.close > self._orb_high:
                        self._is_short = False
                    elif candle.close < self._orb_low:
                        self._is_short = True
                    else:
                        self._is_short = candle.close < (self._orb_high + self._orb_low) / 2
                else:
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
                # until next session's first entry -- at restart_mins
                # (default 17:30 IST, 5 minutes after the usual 17:25
                # square-off) resume trading immediately, in the SAME
                # direction as
                # whichever leg was open going into that square-off
                # (self._last_closed_was_short, updated on every close: a
                # normal SL-hit reversal above, or force_flat() itself).
                # SL = Supertrend's current value, same convention as the
                # session's first entry (this is a fresh, non-reversal
                # position, so it doesn't borrow the day-low/day-high
                # reversal-SL rule). Gated to fire at most once per session
                # the same way the first entry is; the backtest/live engine
                # is expected to resolve this leg against the FOLLOWING
                # day's option expiry, same as any other post-cutoff entry
                # (see resolve_by_premium's own expiry-cutoff handling --
                # nothing SAR-specific needed here). Once open, this
                # position is a completely ordinary position from here on
                # -- normal SL-hit reversals (using day-low/day-high, v2
                # rule) apply exactly as usual, and it naturally carries
                # through the next session's 17:30 reset without the
                # day's-first-entry branch ever re-firing (guarded by "not
                # self._in_position" like everything else here).
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
