"""SessionBreaklineStrategy: session line, anchor->confirm->trigger pattern,
session-line SL, daily square-off, and the one-trade-per-side-per-session cap."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.enums import PositionState
from deltabot.models import Candle
from deltabot.strategy.session_breakline import SessionBreaklineStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ts(hour: int, minute: int, day: int = 8) -> int:
    # 2026-07-08 is a Wednesday.
    return int(datetime(2026, 7, day, hour, minute, tzinfo=_IST).timestamp())


def _c(ts: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=ts, open=o, high=h, low=low, close=cl, volume=1.0)


def _seed_session(s: SessionBreaklineStrategy, day: int = 8, line_close: float = 100.0):
    """Seed bar (establishes prev_now_mins) + the 17:30 session-start bar
    (sets the session line to ``line_close``). Returns the next free timestamp."""
    s.update(_c(_ts(17, 25, day), 100.0, 100.0, 100.0, 100.0))
    s.update(_c(_ts(17, 30, day), 100.0, line_close + 1.0, line_close - 1.0, line_close))
    return _ts(17, 35, day)


def _bull_setup(s: SessionBreaklineStrategy, day: int = 8) -> int:
    """Drift above the line, then a verified anchor (open==high) -> confirm
    (open==low) sequence. Leaves pending_long_trigger=120.0, unfired. Returns
    the next free timestamp."""
    t = _seed_session(s, day, line_close=100.0)
    for o, h, low, cl in ((105, 112, 104, 110), (110, 118, 109, 116), (116, 122, 115, 120)):
        s.update(_c(t, o, h, low, cl))
        t += 300
    s.update(_c(t, 112, 113, 105, 106))   # anchor: open==high
    t += 300
    s.update(_c(t, 112, 120, 112, 118))   # confirm: open==low -> pending_long_trigger = 120
    t += 300
    return t


def test_session_line_set_at_1730() -> None:
    s = SessionBreaklineStrategy()
    assert s.session_line is None
    s.update(_c(_ts(17, 25), 100.0, 100.0, 100.0, 100.0))
    s.update(_c(_ts(17, 30), 100.0, 106.0, 99.0, 103.5))
    assert s.session_line == 103.5


def test_long_anchor_confirm_then_next_bar_breakout_enters() -> None:
    s = SessionBreaklineStrategy()
    t = _bull_setup(s)
    assert s.position_state == PositionState.FLAT
    assert s._pending_long_trigger == 120.0

    # Confirm bar itself must NOT have fired (waits for a later bar).
    d = s.update(_c(t, 118.0, 119.0, 117.0, 118.0))   # high < 120, no break yet
    assert d is None
    assert s.position_state == PositionState.FLAT
    t += 300

    d = s.update(_c(t, 118.0, 122.0, 117.0, 121.0))   # high >= 120 -> breaks trigger
    assert d is not None and d.buy_signal and d.entry_price == 120.0
    assert s.position_state == PositionState.LONG


def test_sl_exit_on_real_close_below_session_line() -> None:
    s = SessionBreaklineStrategy()
    _seed_session(s, line_close=100.0)
    s._in_long, s._long_taken = True, True   # directly open a long, as if just triggered
    t = _ts(18, 0)
    d = s.update(_c(t, 101.0, 102.0, 95.0, 99.0))   # real close (99) < session line (100)
    assert d is not None and d.long_exit and d.exit_reason == "SL"
    assert s.position_state == PositionState.FLAT


def test_no_sl_while_close_stays_on_the_right_side_of_the_line() -> None:
    s = SessionBreaklineStrategy()
    _seed_session(s, line_close=100.0)
    s._in_long, s._long_taken = True, True
    t = _ts(18, 0)
    d = s.update(_c(t, 101.0, 102.0, 95.0, 100.5))   # close (100.5) stays >= line (100)
    assert d is None
    assert s.position_state == PositionState.LONG


def test_square_off_next_day_1725_closes_open_trade() -> None:
    s = SessionBreaklineStrategy()
    _seed_session(s, line_close=100.0)
    s._in_short, s._short_taken = True, True
    s._prev_now_mins = 17 * 60 + 24   # so the 17:25 crossing is detected on the next update
    d = s.update(_c(_ts(17, 25, day=9), 99.0, 100.0, 97.0, 98.0))   # close stays <= the line, no SL
    assert d is not None and d.short_exit and d.exit_reason == "EOD"
    assert s.position_state == PositionState.FLAT


def test_max_one_long_per_session_blocks_a_second_anchor() -> None:
    s = SessionBreaklineStrategy()
    t = _bull_setup(s)
    s.update(_c(t, 118.0, 122.0, 117.0, 121.0))   # fires the long
    assert s.position_state == PositionState.LONG
    s.force_flat()
    t += 300
    # A fresh, otherwise-valid anchor candle must NOT arm a second long setup
    # this session (long_taken is already True).
    s.update(_c(t, 130.0, 131.0, 120.0, 121.0))   # open==high shaped, well above the line
    assert not s._long_armed
    assert s._pending_long_trigger is None


def test_session_reset_clears_untaken_pending_and_trade_cap() -> None:
    s = SessionBreaklineStrategy()
    _bull_setup(s)
    assert s._pending_long_trigger == 120.0

    # Next day's 17:30 reset: new session line, pending/armed state cleared,
    # and (since the side never actually traded) the cap is lifted too.
    s._prev_now_mins = 17 * 60 + 29
    s.update(_c(_ts(17, 30, day=9), 90.0, 92.0, 88.0, 90.0))
    assert s.session_line == 90.0
    assert s._pending_long_trigger is None
    assert not s._long_armed
    assert not s._long_taken


def test_pattern_below_the_line_does_not_arm_long_anchor() -> None:
    s = SessionBreaklineStrategy()
    t = _seed_session(s, line_close=100.0)
    # An open==high candle whose HA low sits AT/BELOW the line must not arm.
    s.update(_c(t, 95.0, 96.0, 90.0, 91.0))
    assert not s._long_armed
