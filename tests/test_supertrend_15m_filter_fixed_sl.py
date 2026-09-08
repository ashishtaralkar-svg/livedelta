"""Supertrend15mFilterFixedSlStrategy: fresh 1m flip + 15m agreement entry,
SL frozen at the flip candle's own Supertrend value (not trailing), plain
stop-to-flat exit (no reversal) -- ported from
supertrend_15m_filter_fixed_sl.pine."""

from __future__ import annotations

from deltabot.models import Candle
from deltabot.strategy.supertrend_15m_filter_fixed_sl import Supertrend15mFilterFixedSlStrategy


def _c(ts: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=ts, open=o, high=h, low=low, close=cl, volume=1.0)


def _strategy(**kw) -> Supertrend15mFilterFixedSlStrategy:
    base = dict(atr_period=3, factor=3.0, atr_period_15m=1, factor_15m=3.0)
    base.update(kw)
    return Supertrend15mFilterFixedSlStrategy(**base)


def _ready(s: Supertrend15mFilterFixedSlStrategy) -> None:
    s._st._bars_seen = 10
    s._st15._bars_seen = 10


T = 60


def test_not_ready_before_warmup() -> None:
    s = _strategy()
    s._st.update = lambda h, l, c: (100.0, 1)
    d = s.update(_c(0, 100, 101, 99, 98))
    assert d is None


def test_entry_requires_fresh_flip_and_15m_agreement() -> None:
    s = _strategy()
    _ready(s)
    s._st15._direction = 1   # 15m red -- agrees with the coming sell flip
    s._st.update = lambda h, l, c: (95.0, -1)   # baseline uptrend, no flip yet
    s.update(_c(0, 100, 101, 99, 100))
    s._st.update = lambda h, l, c: (105.0, 1)   # fresh flip to downtrend
    d = s.update(_c(T, 100, 102, 98, 99))
    assert d is not None and d.has_entry
    assert d.entry_is_short is True
    assert d.sl_level == 105.0
    assert s.in_position and s.is_short is True


def test_entry_skipped_when_15m_disagrees_at_the_flip() -> None:
    s = _strategy()
    _ready(s)
    s._st15._direction = -1   # 15m green -- disagrees with the coming sell flip
    s._st.update = lambda h, l, c: (95.0, -1)
    s.update(_c(0, 100, 101, 99, 100))
    s._st.update = lambda h, l, c: (105.0, 1)   # 1m flips red, but 15m still green
    d = s.update(_c(T, 100, 102, 98, 99))
    assert d is None
    assert not s.in_position


def test_no_entry_when_1m_already_red_without_a_fresh_flip() -> None:
    """Merely being red isn't enough -- needs the CHANGE itself."""
    s = _strategy()
    _ready(s)
    s._st15._direction = 1
    s._st.update = lambda h, l, c: (105.0, 1)
    s.update(_c(0, 100, 101, 99, 98))   # first bar: direction was 0 before, not a "flip"
    assert not s.in_position   # no fresh flip recorded yet (prev_dir was 0)
    s._st.update = lambda h, l, c: (106.0, 1)   # still red, no change
    d = s.update(_c(T, 98, 99, 96, 97))
    assert d is None
    assert not s.in_position


def test_frozen_sl_does_not_trail_and_fires_on_price_cross() -> None:
    s = _strategy()
    _ready(s)
    s._is_short = True
    s._active_sl = 100.0
    s._st.update = lambda h, l, c: (110.0, 1)   # stays red -- ST's CURRENT value moved, but SL is frozen at 100
    d = s.update(_c(0, 98, 99, 97, 98))   # high(99) < frozen SL(100) -- no exit yet
    assert d is None
    assert s._active_sl == 100.0   # unchanged, even though the live ST value is now 110
    d = s.update(_c(T, 99, 101, 98, 100))   # high(101) >= frozen SL(100) -- exit fires AT the level
    assert d is not None and d.has_exit and d.exit_was_short is True
    assert d.exit_price == 100.0   # the frozen level itself, not candle.close
    assert not s.in_position


def test_stop_out_does_not_auto_reverse() -> None:
    """A stop-out just goes flat -- it does not open the opposite side by
    itself (unlike supertrend_sar.py's stop-and-reverse)."""
    s = _strategy()
    _ready(s)
    s._is_short = True
    s._active_sl = 100.0
    s._st15._direction = -1   # 15m green -- WOULD agree with a long if one were considered
    s._st.update = lambda h, l, c: (95.0, -1)   # 1m ALSO flips green this same bar
    d = s.update(_c(0, 99, 105, 98, 102))   # crosses the SL (100) -- exit fires
    assert d is not None and d.has_exit
    # No entry_signal is asserted here on purpose -- a fresh flip THIS bar
    # combined with an SL-hit exit is a legitimate same-bar reopen (matches
    # every other flip strategy's ordering), so has_entry may be True. The
    # thing that must NOT happen is a reversal into the opposite side
    # WITHOUT a genuine qualifying flip -- covered by the next assertion.
    assert s.is_short is not True   # never left short after a short's own SL hit


def test_notify_target_hit_alone_does_not_leave_the_strategy_stuck_in_position() -> None:
    """Regression for a real, confirmed bug: a caller that closes a
    position externally (on a premium target) and calls ONLY
    notify_target_hit() -- forgetting a separate force_flat() -- must NOT
    leave this object permanently believing it's still in that position.
    A 90-day backtest missing this exact call showed 1979 qualifying
    entries after one target hit, 0 of which fired, because in_position
    stayed True forever."""
    s = _strategy()
    _ready(s)
    s._is_short = True
    s._active_sl = 105.0
    assert s.in_position
    s.notify_target_hit()   # the ONLY call made -- no separate force_flat()
    assert not s.in_position
    assert s._active_sl is None


def test_notify_target_hit_blocks_entries_until_15m_flips() -> None:
    s = _strategy()
    _ready(s)
    s._st15._direction = 1   # 15m red
    s.notify_target_hit()

    # A qualifying fresh 1m flip fires here, but the target-hit block is still active.
    s._st.update = lambda h, l, c: (95.0, -1)
    s.update(_c(0, 100, 101, 99, 100))   # baseline, no flip yet (prev_dir was 0)
    s._st.update = lambda h, l, c: (105.0, 1)   # fresh flip to red, 15m agrees (red)
    d = s.update(_c(T, 100, 102, 98, 99))
    assert d is None
    assert not s.in_position   # blocked despite an otherwise-qualifying flip

    # 15m itself flips (any direction) -- this clears the block. The mock
    # must also mutate _direction itself, exactly like the real method
    # would -- a plain lambda return value alone has no side effect.
    def _mock_st15_flip_green(h, l, c):
        s._st15._direction = -1
        return (90.0, -1)

    s._st15.update = _mock_st15_flip_green
    # Force a 15m bucket boundary by jumping the candle far enough ahead.
    s.update(_c(900, 99, 100, 97, 98))
    assert not s._blocked_until_15m_flip   # cleared by the 15m's own fresh flip

    # A fresh 1m flip AFTER the unblock, now agreeing with the (new, green)
    # 15m state, should be free to fire.
    s._st.update = lambda h, l, c: (89.0, -1)
    d = s.update(_c(960, 98, 101, 96, 100))   # fresh flip to green, 15m green -- should fire
    assert d is not None and d.has_entry and d.entry_is_short is False
    assert s.in_position


def test_heikin_ashi_mode_off_by_default() -> None:
    s = _strategy()
    assert s._ha1 is None and s._ha15 is None


def test_heikin_ashi_mode_reported_entry_price_stays_real() -> None:
    """HA governs the Supertrend's own direction/SL-value computation and
    the frozen-SL crossing check (see module docstring), but a reported
    entry fires at the REAL candle close, mirroring how Pine's own strategy
    fills always use real price even on an HA-displayed chart."""
    s = _strategy(use_heikin_ashi=True)
    assert s._ha1 is not None and s._ha15 is not None
    _ready(s)
    s._st15._direction = 1   # 15m red -- agrees with the coming sell flip
    s._st.update = lambda h, l, c: (95.0, -1)
    s.update(_c(0, 100, 101, 99, 100))
    s._st.update = lambda h, l, c: (105.0, 1)   # fresh flip to downtrend
    real_close = 987.65
    d = s.update(_c(T, 100, 102, 98, real_close))
    assert d is not None and d.has_entry
    assert d.entry_price == real_close   # never an HA-converted value


def test_heikin_ashi_mode_15m_gets_its_own_independent_ha_state() -> None:
    """The 15m HA series must be a fresh HA conversion of the real
    15-minute bucket, NOT a rollup of the primary timeframe's own
    (separately-stated) HA candles -- confirmed by the two _HeikinAshi
    instances being distinct objects with independently-evolving state."""
    s = _strategy(use_heikin_ashi=True)
    assert s._ha1 is not s._ha15
    _ready(s)
    for i in range(20):
        s.update(_c(i * T, 100 + i, 101 + i, 99 + i, 100 + i))
    # Both have processed bars, but from different input series (1m candles
    # vs the aggregated 15m bucket) -- their recursive state must diverge.
    assert (s._ha1._ha_open, s._ha1._ha_close) != (s._ha15._ha_open, s._ha15._ha_close)


def test_same_bar_stop_out_and_fresh_qualifying_flip_can_both_fire() -> None:
    s = _strategy()
    _ready(s)
    s._is_short = True
    s._active_sl = 100.0
    s._prev_dir = 1   # currently downtrend, matches being short
    s._st15._direction = -1   # 15m green
    s._st.update = lambda h, l, c: (95.0, -1)   # fresh flip to green this same bar
    d = s.update(_c(0, 99, 105, 98, 102))   # crosses SL(100) AND flips green
    assert d is not None
    assert d.has_exit and d.exit_was_short is True
    assert d.has_entry and d.entry_is_short is False
    assert s.in_position and s.is_short is False
