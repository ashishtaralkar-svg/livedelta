"""Ema21BreakdownStrategy: signal1 (open>ema, close<ema) -> up to 3-bar wait
for a green signal2 -> next-bar-only breakout entry -> fixed SL (signal1
high) / 2R target -- the faithfully-ported SELL side -- plus the mirrored
BUY side added on request (signal1: open<ema, close>ema; wait for RED;
SL = signal1 low)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.enums import PositionState
from deltabot.models import Candle
from deltabot.strategy.ema21_breakdown import Ema21BreakdownStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ts(hour: int, minute: int, day: int = 8) -> int:
    return int(datetime(2026, 7, day, hour, minute, tzinfo=_IST).timestamp())


def _c(ts: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=ts, open=o, high=h, low=low, close=cl, volume=1.0)


def _strategy(**kw) -> Ema21BreakdownStrategy:
    base = dict(ema_len=3, max_wait=3, target_rr=2.0)
    base.update(kw)
    return Ema21BreakdownStrategy(**base)


def test_signal1_arms_tracking_and_pins_anchor_high() -> None:
    s = _strategy()
    for cl in (100.0, 100.0, 100.0):   # warm EMA(3) flat at 100
        s.update(_c(_ts(10, 0), cl - 1, cl + 1, cl - 1, cl))
    d = s.update(_c(_ts(10, 5), 101.0, 105.0, 95.0, 96.0))   # open(101)>ema, close(96)<ema
    assert d is None
    assert s._tracking and s._anchor_high == 105.0


def test_close_above_ema_invalidates_tracking() -> None:
    s = _strategy()
    for cl in (100.0, 100.0, 100.0):
        s.update(_c(_ts(10, 0), cl - 1, cl + 1, cl - 1, cl))
    s.update(_c(_ts(10, 5), 101.0, 105.0, 95.0, 96.0))   # signal1
    assert s._tracking
    s.update(_c(_ts(10, 10), 96.0, 102.0, 95.0, 101.0))   # closes back above ema -> invalidate
    assert not s._tracking


def test_max_wait_exceeded_invalidates() -> None:
    # ema_len=50 -> EMA barely moves per bar, so "waiting" candles can sit
    # comfortably below it (open AND close) without accidentally re-firing
    # signal1_raw (which needs open > ema) on every bar.
    s = _strategy(ema_len=50, max_wait=2)
    for _ in range(50):
        s.update(_c(_ts(10, 0), 99.0, 101.0, 99.0, 100.0))
    s.update(_c(_ts(10, 5), 101.0, 105.0, 95.0, 96.0))    # signal1: open>ema, close<ema
    assert s._tracking
    # two red waiting bars, opens well below the (slow-moving, ~99) EMA.
    s.update(_c(_ts(10, 10), 95.0, 96.0, 90.0, 91.0))
    s.update(_c(_ts(10, 15), 91.0, 92.0, 87.0, 88.0))
    assert s._tracking   # still within max_wait after bar 2
    s.update(_c(_ts(10, 20), 88.0, 89.0, 83.0, 84.0))   # bar 3 -> exceeds max_wait(2)
    assert not s._tracking
    assert s._pending_trigger is None


def test_green_signal2_arms_trigger_and_next_bar_breakout_enters() -> None:
    s = _strategy(ema_len=50)   # slow-moving EMA -- see max_wait test for why
    for _ in range(50):
        s.update(_c(_ts(10, 0), 99.0, 101.0, 99.0, 100.0))
    s.update(_c(_ts(10, 5), 101.0, 105.0, 95.0, 96.0))      # signal1: anchor_high=105
    d = s.update(_c(_ts(10, 10), 90.0, 94.0, 88.0, 93.0))   # green (93>90), open well below ema
    assert d is None
    assert s._pending_trigger == 88.0   # signal2 candle's low
    assert s._pending_sl == 105.0       # signal1 (anchor) candle's high, NOT signal2's high

    d2 = s.update(_c(_ts(10, 15), 93.0, 95.0, 89.0, 94.0))   # low(89) stays above trigger(88) -- no break yet
    assert d2 is None
    assert s.position_state == PositionState.FLAT

    d3 = s.update(_c(_ts(10, 20), 89.0, 90.0, 80.0, 81.0))   # breaks below 88
    assert d3 is not None and d3.sell_signal and d3.entry_price == 88.0
    assert s.position_state == PositionState.SHORT


def test_sl_and_target_from_signal1_high() -> None:
    s = _strategy(target_rr=2.0)
    s._in_short = True
    s._active_sl = 105.0
    s._active_target = 90.0
    d = s.update(_c(_ts(12, 0), 100.0, 106.0, 99.0, 104.0))   # touches SL 105
    assert d is not None and d.short_exit and d.exit_reason == "SL"
    assert s.position_state == PositionState.FLAT


def test_fresh_signal1_reanchors_while_still_waiting_for_signal2() -> None:
    """A later, fresher signal1 candle (still before signal2 fires) resets
    the anchor -- matches the source Pine script's lack of a 'not already
    tracking' guard."""
    s = _strategy()
    for cl in (100.0, 100.0, 100.0):
        s.update(_c(_ts(10, 0), cl - 1, cl + 1, cl - 1, cl))
    s.update(_c(_ts(10, 5), 101.0, 105.0, 95.0, 96.0))    # signal1 #1: anchor_high=105
    assert s._anchor_high == 105.0
    s.update(_c(_ts(10, 10), 97.0, 103.0, 90.0, 92.0))    # ALSO a valid signal1 (open>ema, close<ema)
    assert s._anchor_high == 103.0    # re-anchored to the fresher one
    assert s._bars_since_signal1 == 0


# --------------------------------------------------------------------- #
# Mirrored BUY side (added on request; not in the source Pine script).
# --------------------------------------------------------------------- #

def test_bull_signal1_arms_tracking_and_pins_anchor_low() -> None:
    s = _strategy(ema_len=50)
    for _ in range(50):
        s.update(_c(_ts(10, 0), 99.0, 101.0, 99.0, 100.0))
    d = s.update(_c(_ts(10, 5), 99.0, 105.0, 95.0, 101.0))   # open(99)<ema, close(101)>ema
    assert d is None
    assert s._tracking_bull and s._anchor_low == 95.0


def test_bull_red_signal2_arms_trigger_and_next_bar_breakout_enters() -> None:
    s = _strategy(ema_len=50)
    for _ in range(50):
        s.update(_c(_ts(10, 0), 99.0, 101.0, 99.0, 100.0))
    s.update(_c(_ts(10, 5), 99.0, 105.0, 95.0, 101.0))       # signal1: anchor_low=95
    d = s.update(_c(_ts(10, 10), 104.0, 106.0, 99.0, 101.0))  # red (101<104), open well above ema
    assert d is None
    assert s._pending_trigger_bull == 106.0   # signal2 candle's high
    assert s._pending_sl_bull == 95.0         # signal1 (anchor) candle's low, NOT signal2's low

    d2 = s.update(_c(_ts(10, 15), 101.0, 104.0, 98.0, 102.0))   # high(104) stays below trigger(106)
    assert d2 is None
    assert s.position_state == PositionState.FLAT

    d3 = s.update(_c(_ts(10, 20), 102.0, 110.0, 100.0, 108.0))   # breaks above 106
    assert d3 is not None and d3.buy_signal and d3.entry_price == 106.0
    assert s.position_state == PositionState.LONG


def test_bull_sl_and_target() -> None:
    s = _strategy(target_rr=2.0)
    s._in_long = True
    s._active_sl_bull = 90.0
    s._active_target_bull = 120.0
    d = s.update(_c(_ts(12, 0), 95.0, 96.0, 89.0, 94.0))   # touches SL 90
    assert d is not None and d.long_exit and d.exit_reason == "SL"
    assert s.position_state == PositionState.FLAT


def test_trade_ce_false_disables_short_side_only() -> None:
    s = _strategy(ema_len=50, trade_ce=False)
    for _ in range(50):
        s.update(_c(_ts(10, 0), 99.0, 101.0, 99.0, 100.0))
    s.update(_c(_ts(10, 5), 101.0, 105.0, 95.0, 96.0))   # bear signal1 -- should be ignored
    assert not s._tracking
    s.update(_c(_ts(10, 10), 99.0, 105.0, 95.0, 101.0))  # bull signal1 -- should still arm
    assert s._tracking_bull


# --------------------------------------------------------------------- #
# trend_filter: price vs EMA50/EMA200 stack, checked AT THE TRIGGER.
# --------------------------------------------------------------------- #

def test_trend_filter_blocks_sell_when_price_above_ema50() -> None:
    s = _strategy(trend_filter=True)
    s._pending_trigger = 88.0
    s._pending_sl = 105.0
    s._ema50._value = 50.0    # close(86) will land ABOVE ema50 -- disagrees with the sell filter
    s._ema200._value = 40.0
    d = s.update(_c(_ts(10, 0), 90.0, 91.0, 85.0, 86.0))   # breaks the trigger (low<=88)
    assert d is None                       # consumed untraded, no signal
    assert s.position_state == PositionState.FLAT
    assert s._pending_trigger is None      # setup discarded, not left resting


def test_trend_filter_allows_sell_when_stack_agrees() -> None:
    s = _strategy(trend_filter=True, target_rr=2.0)
    s._pending_trigger = 88.0
    s._pending_sl = 105.0
    s._ema50._value = 100.0
    s._ema200._value = 110.0   # close(86) < ema50(~93 after blending) < ema200(~109.6) -- agrees
    d = s.update(_c(_ts(10, 0), 90.0, 91.0, 85.0, 86.0))
    assert d is not None and d.sell_signal
    assert s.position_state == PositionState.SHORT


def test_ema200_filter_blocks_sell_when_price_above_ema200() -> None:
    s = _strategy(ema200_filter=True, target_rr=2.0)
    s._pending_trigger = 88.0
    s._pending_sl = 105.0
    s._ema200._value = 50.0   # close(86) will land ABOVE ema200 -- disagrees
    d = s.update(_c(_ts(10, 0), 90.0, 91.0, 85.0, 86.0))
    assert d is None
    assert s.position_state == PositionState.FLAT
    assert s._pending_trigger is None


def test_ema200_filter_allows_sell_when_price_below_ema200() -> None:
    s = _strategy(ema200_filter=True, target_rr=2.0)
    s._pending_trigger = 88.0
    s._pending_sl = 105.0
    s._ema200._value = 200.0   # close(86) stays well below ema200 -- agrees
    d = s.update(_c(_ts(10, 0), 90.0, 91.0, 85.0, 86.0))
    assert d is not None and d.sell_signal
    assert s.position_state == PositionState.SHORT


def test_trend_filter_blocks_buy_when_price_below_ema50() -> None:
    s = _strategy(trend_filter=True)
    s._pending_trigger_bull = 112.0
    s._pending_sl_bull = 95.0
    s._ema50._value = 150.0   # close(114) will land BELOW ema50 -- disagrees with the buy filter
    s._ema200._value = 160.0
    d = s.update(_c(_ts(10, 0), 110.0, 115.0, 109.0, 114.0))   # breaks the trigger (high>=112)
    assert d is None
    assert s.position_state == PositionState.FLAT
    assert s._pending_trigger_bull is None
