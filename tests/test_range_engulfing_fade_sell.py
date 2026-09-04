"""RangeEngulfingFadeSellStrategy: red->green range-engulfing candle fades
the breakout of its own high (sell CE), entering the INSTANT a later
candle's HIGH reaches the trigger (not waiting for its close). Sell-only,
1:1 R:R, strict single trade."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.models import Candle
from deltabot.strategy.range_engulfing_fade_sell import RangeEngulfingFadeSellStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ts(hour: int, minute: int, day: int = 8) -> int:
    return int(datetime(2026, 7, day, hour, minute, tzinfo=_IST).timestamp())


def _c(ts: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=ts, open=o, high=h, low=low, close=cl, volume=1.0)


def _strategy() -> RangeEngulfingFadeSellStrategy:
    return RangeEngulfingFadeSellStrategy()


def test_bull_engulf_arms_the_trigger_with_1to1_rr() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))      # red: close(99)<open(100)
    d = s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))  # green, engulfs, close(103)>red open(100)
    assert d is None
    st = s.debug_state()
    assert st["pending_trigger"] == 105.0   # green high
    assert s._pending_target == 96.0        # green low
    assert s._pending_sl == 114.0           # 105 + (105-96) = 114 -- 1:1 R:R


def test_range_must_fully_engulf_not_just_body() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))     # red
    s.update(_c(_ts(10, 5), 97.0, 100.5, 96.0, 100.2))      # high(100.5) <= prev high(101)
    assert s.debug_state()["pending_trigger"] is None


def test_close_confirmation_required_even_if_range_engulfs() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))     # red, open=100
    s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 99.5))       # engulfs, but close(99.5) < red open(100)
    assert s.debug_state()["pending_trigger"] is None


def test_entry_fires_on_intrabar_high_touch_even_if_close_does_not_clear_trigger() -> None:
    """The key difference from the closed-bar port: a candle that pokes
    above the trigger and closes back below it STILL fires the entry --
    matching the .pine chart's resting stop order, which fills the moment
    price crosses, not at the candle's close."""
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))
    s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))          # armed: trigger=105, sl=114, target=96
    d = s.update(_c(_ts(10, 10), 104.0, 106.0, 103.5, 104.5))   # high(106)>=105, but close(104.5)<105
    assert d is not None and d.sell_signal
    assert d.entry_price == 105.0
    assert d.sl_level == 114.0
    assert s.in_short
    assert s.debug_state()["pending_trigger"] is None


def test_pending_trigger_expires_after_one_miss() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))
    s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))          # armed: trigger=105
    d = s.update(_c(_ts(10, 10), 103.0, 104.0, 102.0, 103.5))   # high(104) never reaches 105
    assert d is None
    assert s.debug_state()["pending_trigger"] is None


def test_short_sl_exit() -> None:
    s = _strategy()
    s._in_short = True
    s._active_sl = 114.0
    s._active_target = 96.0
    d = s.update(_c(_ts(12, 0), 110.0, 115.0, 108.0, 112.0))   # high touches 114
    assert d is not None and d.exit and d.exit_price == 114.0
    assert d.exit_reason == "SL"
    assert not s.in_short


def test_short_target_exit() -> None:
    s = _strategy()
    s._in_short = True
    s._active_sl = 114.0
    s._active_target = 96.0
    d = s.update(_c(_ts(12, 0), 100.0, 101.0, 95.0, 97.0))   # low touches 96
    assert d is not None and d.exit and d.exit_price == 96.0
    assert d.exit_reason == "TARGET"
    assert not s.in_short


def test_sl_takes_priority_over_target_on_same_bar() -> None:
    s = _strategy()
    s._in_short = True
    s._active_sl = 105.0
    s._active_target = 96.0
    d = s.update(_c(_ts(12, 0), 100.0, 106.0, 95.0, 100.0))   # both SL(105) and target(96) touched
    assert d is not None and d.exit_reason == "SL"


def test_no_pyramiding_while_already_short() -> None:
    s = _strategy()
    s._in_short = True
    s._active_sl = 200.0
    s._pending_trigger = 105.0
    s._pending_sl = 114.0
    s._pending_target = 96.0
    d = s.update(_c(_ts(12, 0), 104.0, 106.0, 103.0, 105.5))   # would otherwise trigger
    assert d is None
    assert s._pending_trigger == 105.0   # untouched -- entry gate blocked by "flat"


def test_fresh_pattern_replaces_a_still_pending_one() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))
    s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))    # combo A armed: trigger=105
    first_trigger = s._pending_trigger
    s.update(_c(_ts(10, 10), 98.0, 99.0, 90.0, 91.0))     # red, high(99)<105 -- A doesn't fire
    d = s.update(_c(_ts(10, 15), 92.0, 104.0, 89.0, 100.0))   # green, engulfs, close>prev open(98)
    assert d is None
    assert s._pending_trigger != first_trigger   # replaced by the fresher combo
    assert s._pending_trigger == 104.0


def test_force_flat_clears_state() -> None:
    s = _strategy()
    s._in_short = True
    s._active_sl = 114.0
    s.force_flat()
    assert not s.in_short


# ---------------------------------------------------------------------- #
# Live/intracandle API -- exercised directly by the live engine (see
# range_engulfing_fade_sell_trader.py), not through update().
# ---------------------------------------------------------------------- #

def test_arm_from_closed_candle_arms_a_fresh_pattern() -> None:
    s = _strategy()
    s.arm_from_closed_candle(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))     # red
    s.arm_from_closed_candle(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))     # green, engulfs
    st = s.debug_state()
    assert st["pending_trigger"] == 105.0
    assert s._pending_sl == 114.0
    assert s._pending_target == 96.0


def test_check_intracandle_entry_fires_on_high_touch() -> None:
    s = _strategy()
    s._pending_trigger, s._pending_sl, s._pending_target = 105.0, 114.0, 96.0
    result = s.check_intracandle_entry(_c(_ts(10, 10), 104.0, 106.0, 103.0, 104.5))
    assert result == (105.0, 114.0, 96.0)
    assert s.in_short
    assert s._active_sl == 114.0
    assert s._active_target == 96.0
    assert s.debug_state()["pending_trigger"] is None


def test_check_intracandle_entry_is_idempotent_once_fired() -> None:
    """A second tick after the first fire must be a safe no-op (in_short
    is already True), matching how forming ticks arrive repeatedly."""
    s = _strategy()
    s._pending_trigger, s._pending_sl, s._pending_target = 105.0, 114.0, 96.0
    first = s.check_intracandle_entry(_c(_ts(10, 10), 104.0, 106.0, 103.0, 106.0))
    second = s.check_intracandle_entry(_c(_ts(10, 10), 106.0, 108.0, 105.0, 107.0))
    assert first is not None
    assert second is None


def test_check_intracandle_entry_none_while_no_pending_trigger() -> None:
    s = _strategy()
    assert s.check_intracandle_entry(_c(_ts(10, 10), 100.0, 200.0, 90.0, 150.0)) is None


def test_check_intracandle_exit_sl_fires_first() -> None:
    s = _strategy()
    s._in_short = True
    s._active_sl, s._active_target = 114.0, 96.0
    result = s.check_intracandle_exit(_c(_ts(12, 0), 110.0, 120.0, 90.0, 115.0))   # both touched
    assert result == ("SL", 114.0)
    assert not s.in_short


def test_check_intracandle_exit_target() -> None:
    s = _strategy()
    s._in_short = True
    s._active_sl, s._active_target = 114.0, 96.0
    result = s.check_intracandle_exit(_c(_ts(12, 0), 100.0, 101.0, 95.0, 97.0))
    assert result == ("TARGET", 96.0)
    assert not s.in_short


def test_check_intracandle_exit_none_while_flat() -> None:
    s = _strategy()
    assert s.check_intracandle_exit(_c(_ts(12, 0), 100.0, 200.0, 50.0, 150.0)) is None


def test_arm_from_closed_candle_expires_a_trigger_never_consumed_intracandle() -> None:
    """The engine calls check_intracandle_entry on every forming tick of
    the next candle; if none of them fired, arm_from_closed_candle (called
    once that candle closes) must expire the stale trigger."""
    s = _strategy()
    s.arm_from_closed_candle(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))
    s.arm_from_closed_candle(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))     # armed: trigger=105
    next_candle = _c(_ts(10, 10), 103.0, 104.0, 102.0, 103.5)               # high never reached 105
    assert s.check_intracandle_entry(next_candle) is None   # simulates every forming tick missing
    s.arm_from_closed_candle(next_candle)                    # candle now closed
    assert s.debug_state()["pending_trigger"] is None
