"""SupertrendFixedSlStrategy: flip-triggered entry, frozen (non-trailing) SL,
independent CE/PE legs that can be open simultaneously, no pyramiding into
an already-open leg."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.models import Candle
from deltabot.strategy.supertrend_fixed_sl import SupertrendFixedSlStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ts(hour: int, minute: int, day: int = 8) -> int:
    return int(datetime(2026, 7, day, hour, minute, tzinfo=_IST).timestamp())


def _c(ts: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=ts, open=o, high=h, low=low, close=cl, volume=1.0)


def _strategy(**kw) -> SupertrendFixedSlStrategy:
    base = dict(atr_period=3, factor=3.0)
    base.update(kw)
    return SupertrendFixedSlStrategy(**base)


def _feed_uptrend_then_reversal(s: SupertrendFixedSlStrategy) -> list:
    """A steady uptrend (flips Supertrend bullish -> sells PE) followed by a
    sharp reversal down (flips bearish -> closes the PE via SL, sells CE)."""
    out = []
    t = _ts(10, 0)
    seq = [(100, 101, 99, 100), (101, 103, 100, 102), (102, 104, 101, 103), (103, 106, 102, 105),
           (105, 108, 104, 107), (107, 109, 106, 108),
           (108, 108, 95, 96), (96, 97, 90, 91), (91, 92, 85, 86)]
    for o, h, low, cl in seq:
        out.append(s.update(_c(t, o, h, low, cl)))
        t += 300
    return out


def test_bull_flip_sells_pe_with_frozen_sl() -> None:
    s = _strategy()
    decisions = _feed_uptrend_then_reversal(s)
    bull = next(d for d in decisions if d is not None and d.buy_signal)
    assert bull.long_sl is not None
    assert bull.long_sl < bull.entry_price   # frozen SL sits BELOW price in an uptrend


def test_reversal_closes_pe_via_sl_and_opens_ce() -> None:
    s = _strategy()
    decisions = _feed_uptrend_then_reversal(s)
    rev = next(d for d in decisions if d is not None and d.sell_signal)
    assert rev.long_exit and rev.short_sl is not None
    assert s.in_short and not s.in_long


def test_frozen_sl_does_not_move_on_later_bars() -> None:
    """Supertrend's own value keeps changing every bar while a trend holds;
    the leg's SL must stay pinned to whatever it was AT THE FLIP, not track
    that ongoing movement (the whole point of "frozen, not trailing")."""
    s = _strategy()
    t = _ts(10, 0)
    seq = [(100, 101, 99, 100), (101, 103, 100, 102), (102, 104, 101, 103), (103, 106, 102, 105),
           (105, 108, 104, 107), (107, 109, 106, 108)]
    for o, h, low, cl in seq:
        s.update(_c(t, o, h, low, cl))
        t += 300
    assert s.in_long
    frozen_sl = s._active_long_sl

    live_st_value_before = s._st._value
    s.update(_c(t, 108, 110, 107, 109))   # one more up bar -- Supertrend itself moves
    assert s._st._value != live_st_value_before   # confirms Supertrend DID update
    assert s._active_long_sl == frozen_sl          # but the leg's SL did NOT


def test_sl_exit_short() -> None:
    s = _strategy()
    s._in_short = True
    s._active_short_sl = 105.0
    d = s.update(_c(_ts(12, 0), 100, 106, 99, 104))   # high touches 105
    assert d is not None and d.short_exit and d.short_exit_price == 105.0
    assert not s.in_short


def test_sl_exit_long() -> None:
    s = _strategy()
    s._in_long = True
    s._active_long_sl = 90.0
    d = s.update(_c(_ts(12, 0), 95, 96, 89, 94))   # low touches 90
    assert d is not None and d.long_exit and d.long_exit_price == 90.0
    assert not s.in_long


def test_ce_and_pe_can_be_open_simultaneously() -> None:
    s = _strategy()
    s._in_short = True
    s._active_short_sl = 200.0
    s._in_long = True
    s._active_long_sl = 50.0
    d = s.update(_c(_ts(12, 0), 100, 101, 99, 100))   # neither SL touched
    assert d is None
    assert s.in_short and s.in_long


def test_no_pyramiding_while_already_short() -> None:
    """A fresh bear flip must not re-sell the CE while one is already open."""
    s = _strategy()
    s._in_short = True
    s._active_short_sl = 200.0
    s._st._direction = 1
    s._prev_direction = -1   # pretend a bear flip is about to be detected
    d = s.update(_c(_ts(12, 0), 100, 101, 99, 100))
    assert d is None or not d.sell_signal
    assert s._active_short_sl == 200.0   # untouched, not overwritten
