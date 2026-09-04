"""SupertrendSarStrategy v2: fixed daily start time, close-vs-Supertrend
direction, frozen SL, immediate stop-and-reverse on every SL hit (reversal
SL = running day-low/day-high, not a fresh Supertrend value) -- strict
single position, sell-mode only, no EMA filter. Session (not calendar-day)
reset at 05:30 IST by default."""

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
    base = dict(atr_period=3, factor=3.0, start_hour=5, start_minute=35)
    base.update(kw)
    return SupertrendSarStrategy(**base)


def _ready(s: SupertrendSarStrategy) -> None:
    s._warmup_bars = 10


def test_ready_requires_atr_period_warmup() -> None:
    s = _strategy()
    assert not s.ready
    s._warmup_bars = 3
    assert s.ready


def test_no_entry_before_start_time() -> None:
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (150.0, 1)
    d = s.update(_c(_ts(5, 30), 100.0, 101.0, 99.0, 100.0))   # 5:30, before 5:35
    assert d is None
    assert not s.in_position


def test_first_entry_at_start_time_sells_ce_on_a_red_candle() -> None:
    """v3: direction ignores Supertrend entirely -- close<=open (red or a
    doji) -> sell CE. SL still comes from Supertrend's own value, though."""
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    d = s.update(_c(_ts(5, 35), 100.0, 101.0, 97.0, 98.0))   # red: close(98) < open(100)
    assert d is not None and d.entry_signal
    assert d.entry_is_short is True
    assert d.sl_level == 105.0
    assert s.in_position and s.is_short is True


def test_first_entry_at_start_time_buys_pe_on_a_green_candle() -> None:
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (95.0, -1)
    d = s.update(_c(_ts(5, 35), 100.0, 103.0, 99.0, 102.0))   # green: close(102) > open(100)
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
    d = s.update(_c(_ts(5, 35), 100.0, 101.0, 97.0, 98.0))    # but the candle itself is RED
    assert d is not None and d.entry_is_short is True   # red wins -> SELL, not Supertrend's bias


def test_entry_fires_on_first_qualifying_bar_after_start_time_too() -> None:
    """No entry exactly AT 5:35 (e.g. a data gap) -- the next bar after it
    still fires, matching this repo's robust-retry convention elsewhere."""
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    d = s.update(_c(_ts(6, 0), 100.0, 101.0, 99.0, 100.0))   # 6:00, well after 5:35
    assert d is not None and d.entry_signal


def test_sl_hit_reverses_into_a_long_with_the_tracked_day_low_as_sl() -> None:
    """v2 rule: a reversal INTO A LONG freezes SL at the running day-low,
    NOT a fresh Supertrend value. Proven distinct from the SL-hit bar's own
    low by seeding a lower day-low on an earlier bar first."""
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    d0 = s.update(_c(_ts(5, 35), 100.0, 101.0, 97.0, 98.0))   # red candle -> short/CE, SL=105.0
    assert d0 is not None and d0.entry_signal and d0.entry_is_short is True
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
    s._st.update = lambda h, l, c: (95.0, -1)
    d0 = s.update(_c(_ts(5, 35), 100.0, 101.0, 99.0, 100.0))   # long/PE, SL=95.0
    assert d0 is not None and d0.entry_signal and d0.entry_is_short is False
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
    s._st.update = lambda h, l, c: (105.0, 1)
    d0 = s.update(_c(_ts(5, 35), 100.0, 101.0, 97.0, 98.0))   # red -> short/CE, SL=105.0
    assert d0 is not None and d0.entry_is_short is True
    s._st._atr._value = 20.0   # ATR=20 -> minimum reversal SL distance = 20.0
    s.update(_c(_ts(6, 0), 100.0, 101.0, 99.0, 100.0))   # day-low = 99.0 (tight)
    # SL-hit bar: high(106) crosses SL(105.0); close(104.0). Raw day-low
    # (99.0) is only 5.0 away from this close -- inside the 20.0 minimum.
    d = s.update(_c(_ts(6, 30), 100.0, 106.0, 99.5, 104.0))
    assert d is not None and d.entry_is_short is False   # reversed to long
    assert s.debug_state()["day_low"] == 97.0   # the RAW day-low (from the entry bar's own low)
    assert d.sl_level == 84.0                    # but SL was widened: 104.0 - 20.0
    assert s.debug_state()["active_sl"] == 84.0


def test_reversal_sl_uses_the_day_extreme_unchanged_when_already_wide_enough() -> None:
    """No widening needed when the tracked day-low is already farther from
    price than the ATR-based minimum."""
    s = _strategy(min_sl_atr_mult=1.0)
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    s.update(_c(_ts(5, 35), 100.0, 101.0, 97.0, 98.0))
    s._st._atr._value = 2.0   # tiny ATR -> minimum distance = 2.0, easily beaten
    s.update(_c(_ts(6, 0), 100.0, 102.0, 90.0, 101.0))   # day-low = 90.0
    d = s.update(_c(_ts(6, 30), 100.0, 106.0, 99.0, 104.0))
    assert d is not None and d.sl_level == 90.0   # the real day-low, untouched


def test_min_sl_atr_mult_zero_disables_the_widening_guard() -> None:
    s = _strategy(min_sl_atr_mult=0.0)
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    s.update(_c(_ts(5, 35), 100.0, 101.0, 97.0, 98.0))
    s._st._atr._value = 999.0   # huge ATR -- would normally force massive widening
    s.update(_c(_ts(6, 0), 100.0, 102.0, 90.0, 101.0))
    d = s.update(_c(_ts(6, 30), 100.0, 106.0, 99.0, 104.0))
    assert d is not None and d.sl_level == 90.0   # untouched -- guard disabled


def test_evening_restart_fires_5min_after_squareoff_in_the_same_direction() -> None:
    """v5: 5 minutes after the daily square-off (an external force_flat()
    call), the strategy resumes trading immediately instead of waiting for
    next session's 05:35 first entry -- same direction as whatever was
    open going into square-off, SL from a fresh Supertrend read (like a
    first entry, not a reversal)."""
    s = _strategy()
    _ready(s)
    s._active_session = 0   # pretend the morning's first entry already used this session
    s._in_position = True
    s._is_short = True      # was short/CE going into square-off
    s._active_sl = 105.0
    s.force_flat()          # simulates the daily square-off
    assert s.debug_state()["last_closed_was_short"] is True
    s._st.update = lambda h, l, c: (108.0, 1)   # Supertrend's value at 17:30
    d = s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))
    assert d is not None and d.entry_signal
    assert d.entry_is_short is True    # same direction as before square-off
    assert d.sl_level == 108.0          # fresh Supertrend value, not day-high/low
    assert s.in_position and s.is_short is True


def test_evening_restart_resumes_the_long_direction_too() -> None:
    s = _strategy()
    _ready(s)
    s._active_session = 0
    s._in_position = True
    s._is_short = False   # was long/PE going into square-off
    s._active_sl = 95.0
    s.force_flat()
    s._st.update = lambda h, l, c: (92.0, -1)
    d = s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))
    assert d is not None and d.entry_is_short is False
    assert d.sl_level == 92.0


def test_evening_restart_only_fires_once_per_session() -> None:
    s = _strategy()
    _ready(s)
    s._active_session = 0
    s._in_position = True
    s._is_short = True
    s._active_sl = 105.0
    s.force_flat()
    s._st.update = lambda h, l, c: (108.0, 1)
    d1 = s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))
    assert d1 is not None and d1.entry_signal
    s.force_flat()   # e.g. stopped straight back out again
    d2 = s.update(_c(_ts(18, 0), 100.0, 101.0, 99.0, 100.0))
    assert d2 is None
    assert not s.in_position


def test_evening_restart_does_not_fire_without_a_prior_close() -> None:
    """A fresh strategy that's never held any position has no remembered
    direction to resume -- the restart branch must not fire even well
    past restart_mins."""
    s = _strategy()
    _ready(s)
    s._active_session = 0   # morning's first entry considered done, but nothing ever closed
    d = s.update(_c(_ts(18, 0), 100.0, 101.0, 99.0, 100.0))
    assert d is None
    assert not s.in_position


def test_evening_restart_position_then_reverses_normally_via_day_extremes() -> None:
    """Once opened, the evening-restart leg is an entirely ordinary
    position -- a later SL hit reverses using the v2 day-low/day-high
    rule, not another fresh Supertrend read."""
    s = _strategy()
    _ready(s)
    s._active_session = 0
    s._in_position = True
    s._is_short = True
    s._active_sl = 105.0
    s.force_flat()
    s._st.update = lambda h, l, c: (108.0, 1)
    d0 = s.update(_c(_ts(17, 30), 100.0, 101.0, 99.0, 100.0))   # evening restart: short, SL=108.0
    assert d0 is not None and d0.entry_is_short is True and d0.sl_level == 108.0
    s.update(_c(_ts(18, 0), 100.0, 102.0, 85.0, 101.0))   # establishes a wide day-low = 85.0
    d = s.update(_c(_ts(18, 30), 100.0, 109.0, 99.0, 104.0))   # high(109) crosses SL(108.0)
    assert d is not None and d.exit and d.entry_is_short is False
    assert d.sl_level == 85.0   # v2 day-low rule, NOT another Supertrend read


def test_no_pyramiding_no_reentry_while_sl_not_hit() -> None:
    s = _strategy()
    s._in_position = True
    s._is_short = True
    s._active_sl = 200.0   # far away -- not hit
    _ready(s)
    d = s.update(_c(_ts(10, 10), 100.0, 101.0, 99.0, 100.0))
    assert d is None
    assert s.in_position and s.is_short is True
    assert s.debug_state()["active_sl"] == 200.0   # untouched


def test_force_flat_clears_state() -> None:
    s = _strategy()
    s._in_position = True
    s._is_short = True
    s._active_sl = 105.0
    s.force_flat()
    assert not s.in_position
    assert s.is_short is None
    assert s.debug_state()["active_sl"] is None


def test_square_off_does_not_immediately_rearm_the_day_first_entry() -> None:
    """Regression: the session's FIRST-ENTRY signal (candle-color-based,
    not the v5 evening restart) must fire at most ONCE per session. Without
    _active_session tracking, square-off's own force_flat() call (leaving
    the strategy flat with now_mins already well past start_mins) would
    incorrectly re-arm a fresh position minutes later, the SAME session --
    this was a real bug caught while reviewing a real backtest run's trade
    log (entries kept firing right after 17:25 square-off instead of
    waiting for the next session). Checked in the narrow 17:25-17:30 gap,
    BEFORE the v5 evening restart's own trigger at 17:30 -- see the
    test_evening_restart_* tests for what SHOULD happen at/after 17:30."""
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    d1 = s.update(_c(_ts(5, 35), 100.0, 101.0, 99.0, 100.0))   # session's first entry
    assert d1 is not None and d1.entry_signal
    # Square-off (external, via force_flat()) happens at 17:25.
    s.force_flat()
    assert not s.in_position
    # A bar shortly after, but still BEFORE the 17:30 evening-restart
    # trigger, must NOT re-arm anything.
    d2 = s.update(_c(_ts(17, 27), 100.0, 101.0, 99.0, 100.0))
    assert d2 is None
    assert not s.in_position


def test_after_force_flat_the_next_session_can_start_again() -> None:
    """The session AFTER square-off, past start_mins, the session's first
    entry fires normally again -- once an intervening bar has actually
    crossed the 05:30 reset boundary. (The crossing-detection idiom needs
    to observe that transition; jumping straight from one day's 05:35 bar
    to the next day's 05:35 bar would never see it -- exactly like a real
    continuous candle stream always would, and like this repo's other
    minute-crossing detectors, e.g. the daily square-off.)"""
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    s.update(_c(_ts(5, 35, day=8), 100.0, 101.0, 99.0, 100.0))
    s.force_flat()
    s.update(_c(_ts(2, 0, day=9), 100.0, 101.0, 99.0, 100.0))   # still before 05:30
    s._st.update = lambda h, l, c: (95.0, -1)
    d = s.update(_c(_ts(5, 35, day=9), 100.0, 101.0, 99.0, 100.0))   # NEXT session
    assert d is not None and d.entry_signal and d.entry_is_short is False


def test_day_high_low_keep_accumulating_across_a_force_flat() -> None:
    """force_flat() (square-off) must NOT reset the running day-high/low --
    they keep accumulating through the rest of the session regardless of
    position state, same as the Pine port."""
    s = _strategy()
    _ready(s)
    s._st.update = lambda h, l, c: (105.0, 1)
    s.update(_c(_ts(5, 35), 100.0, 101.0, 99.0, 100.0))
    s.force_flat()
    s.update(_c(_ts(17, 25), 100.0, 130.0, 70.0, 100.0))   # a post-square-off bar
    assert s.debug_state()["day_high"] == 130.0
    assert s.debug_state()["day_low"] == 70.0
