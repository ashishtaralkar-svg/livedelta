"""SupertrendSarStrategy v8: Opening-Range-Breakout (ORB) first entry
(17:30-18:00 IST by default), frozen SL, immediate stop-and-reverse on
every SL hit (reversal SL = running day-low/day-high, widened by a min-ATR
guard), ASAP intracandle SL/reversal, and an evening restart (18:05 by
default) after square-off -- strict single position, sell-mode only, no
EMA filter. Session (not calendar-day) reset at 17:30 IST by default (the
same moment the ORB window starts)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.models import Candle
from deltabot.strategy.supertrend_sar import SupertrendSarStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ts(hour: int, minute: int, day: int = 8) -> int:
    return int(datetime(2026, 7, day, hour, minute, tzinfo=_IST).timestamp())


def _c(ts: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=ts, open=o, high=h, low=low, close=cl, volume=1.0)


def _strategy(**kw) -> SupertrendSarStrategy:
    base = dict(atr_period=3, factor=3.0)
    base.update(kw)
    return SupertrendSarStrategy(**base)


def _ready(s: SupertrendSarStrategy) -> None:
    s._warmup_bars = 10


def _arm(s: SupertrendSarStrategy, is_short: bool, sl: float, session: int = 0) -> None:
    """Directly puts the strategy into an established 'already holding a
    position' state, bypassing however it was opened (ORB, evening
    restart, etc.) -- for tests that exercise something ELSE (reversal,
    intracandle checks, min-SL guard...)."""
    s._active_session = session
    s._in_position = True
    s._is_short = is_short
    s._active_sl = sl


def test_ready_requires_atr_period_warmup() -> None:
    s = _strategy()
    assert not s.ready
    s._warmup_bars = 3
    assert s.ready


# ---------------------------------------------------------------------- #
# v9 candle-color first entry (the DEFAULT, orb_enabled=False).
# ---------------------------------------------------------------------- #
def test_no_entry_before_start_time() -> None:
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (150.0, 1)
    d = s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))   # 17:30, before default start_mins (17:35)
    assert d is None
    assert not s.in_position


def test_first_entry_at_start_time_sells_ce_on_a_red_candle() -> None:
    """Default mode (orb_enabled=False): direction ignores Supertrend
    entirely -- close<=open (red or a doji) -> sell CE. SL still comes
    from Supertrend's own value, though."""
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    d = s.update(_c(_ts(17, 35), 100.0, 101.0, 97.0, 98.0))   # red: close(98) < open(100)
    assert d is not None and d.entry_signal
    assert d.entry_is_short is True
    assert d.sl_level == 105.0
    assert s.in_position and s.is_short is True


def test_first_entry_at_start_time_buys_pe_on_a_green_candle() -> None:
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (95.0, -1)
    d = s.update(_c(_ts(17, 35), 100.0, 103.0, 99.0, 102.0))   # green: close(102) > open(100)
    assert d is not None and d.entry_signal
    assert d.entry_is_short is False
    assert d.sl_level == 95.0
    assert s.in_position and s.is_short is False


def test_first_entry_direction_ignores_supertrends_own_bias() -> None:
    """Even when Supertrend's value would suggest the opposite direction,
    the day's first entry's direction is governed purely by candle color."""
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (95.0, -1)   # Supertrend says "uptrend"
    d = s.update(_c(_ts(17, 35), 100.0, 101.0, 97.0, 98.0))    # but the candle itself is RED
    assert d is not None and d.entry_is_short is True   # red wins -> SELL, not Supertrend's bias


def test_entry_fires_on_first_qualifying_bar_after_start_time_too() -> None:
    """No entry exactly AT 17:35 (e.g. a data gap) -- the next bar after it
    still fires, matching this repo's robust-retry convention elsewhere."""
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    d = s.update(_c(_ts(18, 0), 100.0, 101.0, 99.0, 100.0))   # 18:00, well after 17:35
    assert d is not None and d.entry_signal


# ---------------------------------------------------------------------- #
# v7 ORB (Opening Range Breakout) first entry -- opt-in (orb_enabled=True).
# ---------------------------------------------------------------------- #
def test_no_entry_during_the_orb_range_building_window() -> None:
    s = _strategy(orb_enabled=True)
    _ready(s)
    s._st.update = lambda h, l, c: (150.0, 1)
    d = s.update(_c(_ts(17, 45), 100.0, 101.0, 99.0, 100.0))   # inside 17:30-18:00, still building
    assert d is None
    assert not s.in_position


def test_orb_breakout_upward_buys_pe() -> None:
    s = _strategy(orb_enabled=True)
    _ready(s)
    s._st.update = lambda h, l, c: (95.0, -1)
    s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))   # building the range: 99-101
    s.update(_c(_ts(17, 45), 100.0, 102.0, 98.0, 100.0))   # still building: 98-102
    d = s.update(_c(_ts(18, 0), 100.0, 103.0, 99.0, 102.0))   # breaks ABOVE 102
    assert d is not None and d.entry_signal
    assert d.entry_is_short is False   # broke up -> BUY PE
    assert d.sl_level == 95.0            # SL still comes from Supertrend
    assert s.in_position and s.is_short is False


def test_orb_breakout_downward_sells_ce() -> None:
    s = _strategy(orb_enabled=True)
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))
    s.update(_c(_ts(17, 45), 100.0, 102.0, 98.0, 100.0))
    d = s.update(_c(_ts(18, 0), 100.0, 99.0, 97.0, 98.0))   # breaks BELOW 98
    assert d is not None and d.entry_signal
    assert d.entry_is_short is True   # broke down -> SELL CE
    assert d.sl_level == 105.0
    assert s.in_position and s.is_short is True


def test_orb_direction_ignores_supertrends_own_bias() -> None:
    """Even when Supertrend's value would suggest the opposite direction,
    the day's first entry's direction is governed purely by which side of
    the opening range price broke."""
    s = _strategy(orb_enabled=True)
    _ready(s)
    s._st.update = lambda h, l, c: (95.0, -1)   # Supertrend says "uptrend"
    s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))
    s.update(_c(_ts(17, 45), 100.0, 102.0, 98.0, 100.0))
    d = s.update(_c(_ts(18, 0), 100.0, 99.0, 97.0, 98.0))   # but price breaks DOWN
    assert d is not None and d.entry_is_short is True   # down-break wins -> SELL, not Supertrend's bias


def test_orb_breakout_can_fire_on_a_later_bar_too() -> None:
    """No breakout right at orb_end_mins -- a later bar that eventually
    breaks the (already-finalized) range still fires."""
    s = _strategy(orb_enabled=True)
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))
    d0 = s.update(_c(_ts(18, 0), 100.0, 101.0, 99.0, 100.0))   # right at orb_end, no breakout yet
    assert d0 is None
    d = s.update(_c(_ts(18, 15), 100.0, 105.0, 99.0, 103.0))   # later bar breaks the range
    assert d is not None and d.entry_signal


def test_orb_breakout_both_sides_at_once_follows_the_close() -> None:
    """A bar that breaks both range extremes at once (a big range/gap)
    follows the CLOSE instead of picking arbitrarily."""
    s = _strategy(orb_enabled=True)
    _ready(s)
    s._st.update = lambda h, l, c: (95.0, -1)
    s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))   # range: 99-101
    d = s.update(_c(_ts(18, 0), 100.0, 110.0, 90.0, 105.0))   # breaks BOTH sides; close(105) above range
    assert d is not None and d.entry_is_short is False   # close above the range -> BUY PE


def test_orb_range_resets_each_session() -> None:
    """A new session's ORB range starts fresh, unaffected by the prior
    day's range."""
    s = _strategy(orb_enabled=True)
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    s.update(_c(_ts(17, 30, day=8), 100.0, 200.0, 50.0, 100.0))   # huge range day 8: 50-200
    s.update(_c(_ts(12, 0, day=9), 100.0, 101.0, 99.0, 100.0))    # still before day 9's 17:30 reset
    s.update(_c(_ts(17, 30, day=9), 300.0, 301.0, 299.0, 300.0))   # session reset crosses 17:30 day 9; day 9's OWN, much narrower range
    d = s.update(_c(_ts(18, 0, day=9), 300.0, 305.0, 299.0, 303.0))   # breaks day 9's range (301), not day 8's (200)
    assert d is not None and d.entry_is_short is False


# ---------------------------------------------------------------------- #
# v2 stop-and-reverse (day-low/day-high SL) -- unaffected by v7's ORB
# change, so these arm a position directly via _arm() rather than going
# through an entry mechanism.
# ---------------------------------------------------------------------- #
def test_sl_hit_reverses_into_a_long_with_the_tracked_day_low_as_sl() -> None:
    """v2 rule: a reversal INTO A LONG freezes SL at the running day-low,
    NOT a fresh Supertrend value. Proven distinct from the SL-hit bar's own
    low by seeding a lower day-low on an earlier bar first."""
    s = _strategy()
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    s._st.update = lambda h, l, c: (108.0, 1)
    s.update(_c(_ts(6, 0), 100.0, 102.0, 90.0, 101.0))   # establishes day-low = 90.0
    # SL-hit bar: high(106) crosses the frozen SL(105.0). Its OWN low
    # (99.0) is higher than the already-tracked day-low (90.0) -- if the
    # reversal SL came from this bar alone it would be 99.0.
    d = s.update(_c(_ts(6, 30), 100.0, 106.0, 99.0, 104.0))
    assert d is not None
    assert d.exit and d.exit_price == 105.0 and d.exit_was_short is True
    assert d.entry_signal and d.entry_is_short is False   # reversed to long/PE
    assert d.sl_level == 90.0                              # v2: tracked day LOW
    assert s.in_position and s.is_short is False
    assert s.debug_state()["active_sl"] == 90.0


def test_sl_hit_reverses_into_a_short_with_the_tracked_day_high_as_sl() -> None:
    """Mirror of the above: a reversal INTO A SHORT freezes SL at the
    running day-high."""
    s = _strategy()
    _ready(s)
    _arm(s, is_short=False, sl=95.0)
    s._st.update = lambda h, l, c: (92.0, -1)
    s.update(_c(_ts(6, 0), 100.0, 115.0, 98.0, 101.0))   # establishes day-high = 115.0
    # SL-hit bar: low(89) crosses the frozen SL(95.0). Its OWN high (96.0)
    # is lower than the already-tracked day-high (115.0).
    d = s.update(_c(_ts(6, 30), 100.0, 96.0, 89.0, 91.0))
    assert d is not None
    assert d.exit and d.exit_price == 95.0 and d.exit_was_short is False
    assert d.entry_signal and d.entry_is_short is True   # reversed back to short/CE
    assert d.sl_level == 115.0                             # v2: tracked day HIGH
    assert s.in_position and s.is_short is True
    assert s.debug_state()["active_sl"] == 115.0


def test_reversal_sl_widens_to_the_atr_minimum_when_day_extreme_is_too_close() -> None:
    """v4: if the raw day-low would put the reversal SL closer than
    min_sl_atr_mult * ATR to price, it's pushed out to that minimum
    distance instead -- fixes the same-session whipsaw seen in real
    backtest trade logs right after a session reset (day-high/low is still
    only a few bars wide that early)."""
    s = _strategy(min_sl_atr_mult=1.0)
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    s._st.update = lambda h, l, c: (108.0, 1)
    s._st._atr._value = 20.0   # ATR=20 -> minimum reversal SL distance = 20.0
    s.update(_c(_ts(6, 0), 100.0, 101.0, 99.0, 100.0))   # day-low = 99.0 (tight)
    # SL-hit bar: high(106) crosses SL(105.0); close(104.0). Raw day-low
    # (99.0) is only 5.0 away from this close -- inside the 20.0 minimum.
    d = s.update(_c(_ts(6, 30), 100.0, 106.0, 99.5, 104.0))
    assert d is not None and d.entry_is_short is False   # reversed to long
    assert d.sl_level == 84.0                    # but SL was widened: 104.0 - 20.0
    assert s.debug_state()["active_sl"] == 84.0


def test_reversal_sl_uses_the_day_extreme_unchanged_when_already_wide_enough() -> None:
    """No widening needed when the tracked day-low is already farther from
    price than the ATR-based minimum."""
    s = _strategy(min_sl_atr_mult=1.0)
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    s._st.update = lambda h, l, c: (108.0, 1)
    s._st._atr._value = 2.0   # tiny ATR -> minimum distance = 2.0, easily beaten
    s.update(_c(_ts(6, 0), 100.0, 102.0, 90.0, 101.0))   # day-low = 90.0
    d = s.update(_c(_ts(6, 30), 100.0, 106.0, 99.0, 104.0))
    assert d is not None and d.sl_level == 90.0   # the real day-low, untouched


def test_min_sl_atr_mult_zero_disables_the_widening_guard() -> None:
    s = _strategy(min_sl_atr_mult=0.0)
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    s._st.update = lambda h, l, c: (108.0, 1)
    s._st._atr._value = 999.0   # huge ATR -- would normally force massive widening
    s.update(_c(_ts(6, 0), 100.0, 102.0, 90.0, 101.0))
    d = s.update(_c(_ts(6, 30), 100.0, 106.0, 99.0, 104.0))
    assert d is not None and d.sl_level == 90.0   # untouched -- guard disabled


# ---------------------------------------------------------------------- #
# v6 ASAP (intracandle) SL/reversal.
# ---------------------------------------------------------------------- #
def test_check_intracandle_sl_false_when_flat() -> None:
    s = _strategy()
    assert s.check_intracandle_sl(999.0) == (False, None)


def test_check_intracandle_sl_detects_a_short_leg_touch() -> None:
    s = _strategy()
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    assert s.check_intracandle_sl(104.9) == (False, 105.0)   # not touched yet
    assert s.check_intracandle_sl(105.0) == (True, 105.0)    # touched exactly
    assert s.check_intracandle_sl(110.0) == (True, 105.0)    # blown through


def test_check_intracandle_sl_detects_a_long_leg_touch() -> None:
    s = _strategy()
    _ready(s)
    _arm(s, is_short=False, sl=95.0)
    assert s.check_intracandle_sl(95.1) == (False, 95.0)
    assert s.check_intracandle_sl(95.0) == (True, 95.0)
    assert s.check_intracandle_sl(90.0) == (True, 95.0)


def test_check_intracandle_sl_never_mutates_state() -> None:
    """Pure check -- calling it repeatedly must not change anything, unlike
    apply_intracandle_reversal()."""
    s = _strategy()
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    for _ in range(5):
        s.check_intracandle_sl(200.0)
    assert s.is_short is True
    assert s.debug_state()["active_sl"] == 105.0


def test_apply_intracandle_reversal_flips_side_and_sets_day_extreme_sl() -> None:
    s = _strategy()
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    s.update(_c(_ts(6, 0), 100.0, 102.0, 90.0, 101.0))   # establishes day-low=90.0
    hit, level = s.check_intracandle_sl(106.0)
    assert hit and level == 105.0
    new_is_short, new_sl = s.apply_intracandle_reversal(106.0)
    assert new_is_short is False           # reversed to long/PE
    assert new_sl == 90.0                   # v2 rule: tracked day-low (106 doesn't beat it)
    assert s.is_short is False
    assert s.debug_state()["active_sl"] == 90.0
    assert s.debug_state()["last_closed_was_short"] is True   # the side that just closed


def test_apply_intracandle_reversal_extends_day_extreme_with_the_tick_price() -> None:
    """The reversal's day-low/day-high is extended one step further using
    the REAL tick price itself, ahead of whatever the still-forming
    candle's eventual close will be."""
    s = _strategy()
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    s.update(_c(_ts(6, 0), 100.0, 101.0, 97.0, 98.0))   # day-low so far = 97.0
    new_is_short, new_sl = s.apply_intracandle_reversal(106.0)
    assert new_is_short is False
    assert new_sl == 97.0   # the tick price (106.0) doesn't beat the existing day-low (97.0)


def test_apply_intracandle_reversal_respects_the_min_sl_atr_guard() -> None:
    s = _strategy(min_sl_atr_mult=1.0)
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    s._st._atr._value = 20.0   # minimum reversal distance = 20.0
    new_is_short, new_sl = s.apply_intracandle_reversal(104.0)   # raw day-low would be na/104
    assert new_is_short is False
    assert new_sl == 84.0   # widened: 104.0 (tick price) - 20.0


# ---------------------------------------------------------------------- #
# v5 evening restart -- all armed via _arm()/force_flat(), never touches
# the ORB mechanism, so unaffected by the v7 change.
# ---------------------------------------------------------------------- #
def test_evening_restart_fires_5min_after_squareoff_in_the_same_direction() -> None:
    """v5: 5 minutes after the daily square-off (an external force_flat()
    call), the strategy resumes trading immediately instead of waiting for
    next session's ORB breakout -- same direction as whatever was open
    going into square-off, SL from a fresh Supertrend read (like a first
    entry, not a reversal)."""
    s = _strategy()
    _ready(s)
    _arm(s, is_short=True, sl=105.0)   # pretend a leg is already open, ORB-armed earlier this session
    s.force_flat()          # simulates the daily square-off
    assert s.debug_state()["last_closed_was_short"] is True
    s._st.update = lambda h, l, c: (108.0, 1)   # Supertrend's value at 18:05
    d = s.update(_c(_ts(18, 5), 100.0, 101.0, 99.0, 100.0))
    assert d is not None and d.entry_signal
    assert d.entry_is_short is True    # same direction as before square-off
    assert d.sl_level == 108.0          # fresh Supertrend value, not day-high/low
    assert s.in_position and s.is_short is True


def test_evening_restart_resumes_the_long_direction_too() -> None:
    s = _strategy()
    _ready(s)
    _arm(s, is_short=False, sl=95.0)
    s.force_flat()
    s._st.update = lambda h, l, c: (92.0, -1)
    d = s.update(_c(_ts(18, 5), 100.0, 101.0, 99.0, 100.0))
    assert d is not None and d.entry_is_short is False
    assert d.sl_level == 92.0


def test_evening_restart_only_fires_once_per_session() -> None:
    s = _strategy()
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    s.force_flat()
    s._st.update = lambda h, l, c: (108.0, 1)
    d1 = s.update(_c(_ts(18, 5), 100.0, 101.0, 99.0, 100.0))
    assert d1 is not None and d1.entry_signal
    s.force_flat()   # e.g. stopped straight back out again
    d2 = s.update(_c(_ts(18, 30), 100.0, 101.0, 99.0, 100.0))
    assert d2 is None
    assert not s.in_position


def test_evening_restart_does_not_fire_without_a_prior_close() -> None:
    """A fresh strategy that's never held any position has no remembered
    direction to resume -- the restart branch must not fire even well
    past restart_mins."""
    s = _strategy()
    _ready(s)
    s._active_session = 0   # morning's first entry considered done, but nothing ever closed
    d = s.update(_c(_ts(19, 0), 100.0, 101.0, 99.0, 100.0))   # well past restart_mins (18:05)
    assert d is None
    assert not s.in_position


def test_evening_restart_position_then_reverses_normally_via_day_extremes() -> None:
    """Once opened, the evening-restart leg is an entirely ordinary
    position -- a later SL hit reverses using the v2 day-low/day-high
    rule, not another fresh Supertrend read."""
    s = _strategy()
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    s.force_flat()
    s._st.update = lambda h, l, c: (108.0, 1)
    d0 = s.update(_c(_ts(18, 5), 100.0, 101.0, 99.0, 100.0))   # evening restart: short, SL=108.0
    assert d0 is not None and d0.entry_is_short is True and d0.sl_level == 108.0
    s.update(_c(_ts(18, 30), 100.0, 102.0, 85.0, 101.0))   # establishes a wide day-low = 85.0
    d = s.update(_c(_ts(19, 0), 100.0, 109.0, 99.0, 104.0))   # high(109) crosses SL(108.0)
    assert d is not None and d.exit and d.entry_is_short is False
    assert d.sl_level == 85.0   # v2 day-low rule, NOT another Supertrend read


def test_no_pyramiding_no_reentry_while_sl_not_hit() -> None:
    s = _strategy()
    _ready(s)
    _arm(s, is_short=True, sl=200.0)   # far away -- not hit
    d = s.update(_c(_ts(10, 10), 100.0, 101.0, 99.0, 100.0))
    assert d is None
    assert s.in_position and s.is_short is True
    assert s.debug_state()["active_sl"] == 200.0   # untouched


def test_force_flat_clears_state() -> None:
    s = _strategy()
    _arm(s, is_short=True, sl=105.0)
    s.force_flat()
    assert not s.in_position
    assert s.is_short is None
    assert s.debug_state()["active_sl"] is None


def test_square_off_does_not_immediately_rearm_the_candle_color_first_entry() -> None:
    """Regression: the session's FIRST-ENTRY signal (default candle-color
    mode) must fire at most ONCE per session. Without _active_session
    tracking, square-off's own force_flat() call would incorrectly re-arm
    a fresh position minutes later, the SAME session -- this was a real
    bug caught while reviewing a real backtest run's trade log (entries
    kept firing right after 17:25 square-off instead of waiting for the
    next session). Evening restart (v5) is pushed to a time that can't
    fire in this test (23:59) so it doesn't also become eligible and mask
    what's being tested here."""
    s = _strategy(restart_hour=23, restart_minute=59)
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    d1 = s.update(_c(_ts(17, 35), 100.0, 101.0, 99.0, 100.0))   # session's first entry
    assert d1 is not None and d1.entry_signal
    # Square-off (external, via force_flat()) happens at 17:25 the next day
    # in reality -- simulated here directly.
    s.force_flat()
    assert not s.in_position
    # A later bar the SAME session must NOT re-arm a fresh entry.
    d2 = s.update(_c(_ts(18, 10), 100.0, 101.0, 99.0, 100.0))
    assert d2 is None
    assert not s.in_position


def test_square_off_does_not_immediately_rearm_the_orb_first_entry() -> None:
    """Same regression, opt-in ORB mode (orb_enabled=True): ORB's own
    default window (17:30-18:00) sits right on top of restart's own
    default (17:30), so the two are NOT cleanly separable at default
    settings -- restart is pushed to 23:59 here so it doesn't also become
    eligible and mask what's being tested; see the v7 history note in the
    module docstring."""
    s = _strategy(orb_enabled=True, restart_hour=23, restart_minute=59)
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))   # build the range
    d1 = s.update(_c(_ts(18, 0), 100.0, 99.0, 97.0, 98.0))   # breaks down -> session's first entry
    assert d1 is not None and d1.entry_signal
    # Square-off (external, via force_flat()) happens at 17:25 the next day
    # in reality -- simulated here directly.
    s.force_flat()
    assert not s.in_position
    # A later bar the SAME session, even one that would ALSO break the
    # (already-consumed) range, must NOT re-arm a fresh ORB entry.
    d2 = s.update(_c(_ts(18, 10), 100.0, 200.0, 50.0, 100.0))
    assert d2 is None
    assert not s.in_position


def test_after_force_flat_the_next_session_can_start_again() -> None:
    """The session AFTER square-off, the ORB mechanism arms again from
    scratch -- once an intervening bar has actually crossed the 17:30
    reset boundary."""
    s = _strategy(orb_enabled=True)
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    s.update(_c(_ts(17, 30, day=8), 100.0, 101.0, 99.0, 100.0))
    s.update(_c(_ts(18, 0, day=8), 100.0, 99.0, 97.0, 98.0))   # day 8's ORB entry: short/CE
    s.force_flat()
    s.update(_c(_ts(12, 0, day=9), 100.0, 101.0, 99.0, 100.0))   # still before day 9's 17:30 reset
    s._st.update = lambda h, l, c: (95.0, -1)
    s.update(_c(_ts(17, 30, day=9), 100.0, 101.0, 99.0, 100.0))   # session reset crosses 17:30 day 9; day 9's OWN range
    d = s.update(_c(_ts(18, 0, day=9), 100.0, 103.0, 99.0, 102.0))   # breaks UP -- NEXT session
    assert d is not None and d.entry_signal and d.entry_is_short is False


def test_day_high_low_keep_accumulating_across_a_force_flat() -> None:
    """force_flat() (square-off) must NOT reset the running day-high/low --
    they keep accumulating through the rest of the session regardless of
    position state, same as the Pine port."""
    s = _strategy()
    _ready(s)
    _arm(s, is_short=True, sl=105.0)
    s.force_flat()
    s.update(_c(_ts(17, 25), 100.0, 130.0, 70.0, 100.0))   # a post-square-off bar
    assert s.debug_state()["day_high"] == 130.0
    assert s.debug_state()["day_low"] == 70.0
