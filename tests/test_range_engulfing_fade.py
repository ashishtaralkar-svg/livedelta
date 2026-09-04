"""RangeEngulfingFadeStrategy: red->green range-engulfing candle fades the
breakout of its own high (sell CE); green->red is the exact mirror (sell PE).
Strict single trade, closed-bar only."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.models import Candle
from deltabot.strategy.range_engulfing_fade import RangeEngulfingFadeStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ts(hour: int, minute: int, day: int = 8) -> int:
    return int(datetime(2026, 7, day, hour, minute, tzinfo=_IST).timestamp())


def _c(ts: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=ts, open=o, high=h, low=low, close=cl, volume=1.0)


def _strategy(**kw) -> RangeEngulfingFadeStrategy:
    return RangeEngulfingFadeStrategy(**kw)


def test_bull_engulf_arms_the_short_trigger() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))    # red: close(99)<open(100)
    d = s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))  # green, engulfs, close(103)>red open(100)
    assert d is None   # arming isn't itself a Decision-worthy event
    st = s.debug_state()
    assert st["pending_short_trigger"] == 105.0   # green high
    # target = green low (96.0), sl = green high + (green high - green low)/2 = 105 + 4.5 = 109.5 (1:2 R:R)
    assert s._pending_short_target == 96.0
    assert s._pending_short_sl == 109.5


def test_bear_engulf_arms_the_long_trigger() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 102.0, 99.0, 101.0))    # green: close(101)>open(100)
    d = s.update(_c(_ts(10, 5), 103.0, 104.0, 95.0, 96.0))  # red, engulfs, close(96)<green open(100)
    assert d is None
    st = s.debug_state()
    assert st["pending_long_trigger"] == 95.0   # red low
    assert s._pending_long_target == 104.0       # red high
    assert s._pending_long_sl == 90.5             # 95 - (104-95)/2 = 90.5 (1:2 R:R)


def test_range_must_fully_engulf_not_just_body() -> None:
    """Body confirmation holds (close > prev open) but the range does NOT
    fully engulf (high does not exceed prev high) -- must not arm."""
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))     # red
    s.update(_c(_ts(10, 5), 97.0, 100.5, 96.0, 100.2))      # green, low<prev low, but high(100.5)<=prev high(101)
    assert s.debug_state()["pending_short_trigger"] is None


def test_close_confirmation_required_even_if_range_engulfs() -> None:
    """Range fully engulfs but green's close does NOT clear red's open --
    must not arm."""
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))     # red, open=100
    s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 99.5))       # green, engulfs range, but close(99.5) < red open(100)
    assert s.debug_state()["pending_short_trigger"] is None


def test_breakout_of_armed_short_trigger_sells_ce() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))
    s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))   # armed: trigger=105, sl=109.5, target=96
    d = s.update(_c(_ts(10, 10), 104.0, 106.0, 103.0, 105.5))   # close(105.5) > trigger(105)
    assert d is not None and d.sell_signal
    assert d.entry_price == 105.0
    assert d.sl_level == 109.5
    assert s.in_short
    assert s._active_short_target == 96.0
    assert s.debug_state()["pending_short_trigger"] is None


def test_breakdown_of_armed_long_trigger_sells_pe() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 102.0, 99.0, 101.0))
    s.update(_c(_ts(10, 5), 103.0, 104.0, 95.0, 96.0))   # armed: trigger=95, sl=90.5, target=104
    d = s.update(_c(_ts(10, 10), 94.5, 95.0, 90.0, 92.0))   # close(92) < trigger(95)
    assert d is not None and d.buy_signal
    assert d.entry_price == 95.0
    assert d.sl_level == 90.5
    assert s.in_long
    assert s._active_long_target == 104.0


def test_short_sl_exit() -> None:
    s = _strategy()
    s._in_short = True
    s._active_short_sl = 114.0
    s._active_short_target = 96.0
    d = s.update(_c(_ts(12, 0), 110.0, 115.0, 108.0, 112.0))   # high touches 114
    assert d is not None and d.short_exit and d.short_exit_price == 114.0
    assert d.short_exit_reason == "SL"
    assert not s.in_short


def test_short_target_exit() -> None:
    s = _strategy()
    s._in_short = True
    s._active_short_sl = 114.0
    s._active_short_target = 96.0
    d = s.update(_c(_ts(12, 0), 100.0, 101.0, 95.0, 97.0))   # low touches 96
    assert d is not None and d.short_exit and d.short_exit_price == 96.0
    assert d.short_exit_reason == "TARGET"
    assert not s.in_short


def test_long_sl_exit() -> None:
    s = _strategy()
    s._in_long = True
    s._active_long_sl = 86.0
    s._active_long_target = 104.0
    d = s.update(_c(_ts(12, 0), 90.0, 92.0, 85.0, 87.0))   # low touches 86
    assert d is not None and d.long_exit and d.long_exit_price == 86.0
    assert d.long_exit_reason == "SL"
    assert not s.in_long


def test_long_target_exit() -> None:
    s = _strategy()
    s._in_long = True
    s._active_long_sl = 86.0
    s._active_long_target = 104.0
    d = s.update(_c(_ts(12, 0), 100.0, 105.0, 99.0, 103.0))   # high touches 104
    assert d is not None and d.long_exit and d.long_exit_price == 104.0
    assert d.long_exit_reason == "TARGET"
    assert not s.in_long


def test_sl_takes_priority_over_target_on_same_bar() -> None:
    s = _strategy()
    s._in_short = True
    s._active_short_sl = 105.0
    s._active_short_target = 96.0
    d = s.update(_c(_ts(12, 0), 100.0, 106.0, 95.0, 100.0))   # both SL(105) and target(96) touched
    assert d is not None and d.short_exit_reason == "SL"


def test_fresh_pattern_replaces_a_still_pending_one() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))
    s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))   # combo A armed: trigger=105
    first_trigger = s._pending_short_trigger
    # A later valid combo forms too, without A ever breaking (all closes stay < 105).
    s.update(_c(_ts(10, 10), 98.0, 99.0, 90.0, 91.0))     # red
    d = s.update(_c(_ts(10, 15), 92.0, 104.0, 89.0, 100.0))   # green, engulfs, close>prev open(98)
    assert d is None
    assert s._pending_short_trigger != first_trigger   # replaced by the fresher combo
    assert s._pending_short_trigger == 104.0


def test_pending_short_trigger_expires_after_one_miss() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))     # red
    s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))     # green, armed: trigger=105
    # close(104) doesn't clear 105, and is a flat-body candle so it can't
    # itself arm a fresh pattern either.
    d = s.update(_c(_ts(10, 10), 104.0, 104.5, 103.0, 104.0))
    assert d is None
    assert s._pending_short_trigger is None   # only had the one candle's chance


def test_pending_long_trigger_expires_after_one_miss() -> None:
    s = _strategy()
    s.update(_c(_ts(10, 0), 100.0, 102.0, 99.0, 101.0))    # green
    s.update(_c(_ts(10, 5), 103.0, 104.0, 95.0, 96.0))     # red, armed: trigger=95
    d = s.update(_c(_ts(10, 10), 96.0, 97.0, 95.5, 96.0))  # close(96) doesn't clear below 95
    assert d is None
    assert s._pending_long_trigger is None


def test_no_pyramiding_while_already_short() -> None:
    s = _strategy()
    s._in_short = True
    s._active_short_sl = 200.0
    s._pending_short_trigger = 105.0
    s._pending_short_sl = 114.0
    s._pending_short_target = 96.0
    d = s.update(_c(_ts(12, 0), 104.0, 106.0, 103.0, 105.5))   # would otherwise break out
    assert d is None
    assert s._pending_short_trigger == 105.0   # untouched -- entry gate blocked by "flat"


def test_trade_ce_false_disables_the_bearish_side() -> None:
    s = _strategy(trade_ce=False)
    s.update(_c(_ts(10, 0), 100.0, 101.0, 98.0, 99.0))
    s.update(_c(_ts(10, 5), 97.0, 105.0, 96.0, 103.0))
    assert s._pending_short_trigger is None


def test_trade_pe_false_disables_the_bullish_side() -> None:
    s = _strategy(trade_pe=False)
    s.update(_c(_ts(10, 0), 100.0, 102.0, 99.0, 101.0))
    s.update(_c(_ts(10, 5), 103.0, 104.0, 95.0, 96.0))
    assert s._pending_long_trigger is None


def test_force_flat_clears_state() -> None:
    s = _strategy()
    s._in_short = True
    s._active_short_sl = 114.0
    s._in_long = True
    s._active_long_sl = 86.0
    s.force_flat()
    assert not s.in_short and not s.in_long
