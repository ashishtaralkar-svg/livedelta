"""DCv2Strategy: Pine-port behaviors -- state-based hunt arming, touch/confirm/
breakout flow, fixed range SL, EMA-relationship exit, no EOD close, weekend
entry block. Uses tiny EMA/DC lengths so sequences stay short."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from deltabot.enums import PositionState
from deltabot.models import Candle
from deltabot.strategy.dcv2 import DCv2Strategy

_IST = ZoneInfo("Asia/Kolkata")


def _ts(hour: int, minute: int, day: int = 8) -> int:
    # 2026-07-08 is a Wednesday; 2026-07-11 is a Saturday.
    return int(datetime(2026, 7, day, hour, minute, tzinfo=_IST).timestamp())


def _c(ts: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=ts, open=o, high=h, low=low, close=cl, volume=1.0)


def _strategy(**kw) -> DCv2Strategy:
    base = dict(dc_period=2, ema_trend_length=2, ema_long_length=4,
                skip_weekdays=frozenset())
    base.update(kw)
    return DCv2Strategy(**base)


def _bull_setup(s: DCv2Strategy, day: int = 8) -> list:
    """Warmup (rising, EMA2>EMA4) -> DC-lower touch -> open==low confirm.
    Returns the decisions; leaves the strategy with a pending long."""
    out = []
    t = _ts(10, 0, day)
    for i, cl in enumerate((100.0, 101.0, 102.0, 103.0)):        # warmup, rising
        out.append(s.update(_c(t + i * 300, cl - 0.5, cl + 0.5, cl - 1.0, cl)))
    out.append(s.update(_c(t + 4 * 300, 103.0, 104.0, 95.0, 103.0)))   # deep dip: touch
    out.append(s.update(_c(t + 5 * 300, 104.0, 110.0, 104.0, 110.0)))  # open==low confirm
    return out


def test_bull_flow_touch_confirm_breakout_entry() -> None:
    s = _strategy()
    _bull_setup(s)
    assert s.has_pending
    d = s.update(_c(_ts(10, 30), 110.0, 112.0, 109.0, 111.0))    # breaks range high
    assert d is not None and d.buy_signal
    assert s.position_state == PositionState.LONG
    assert d.sl_level is not None and d.sl_level < d.entry_price
    assert d.entry_price >= 110.0    # range high includes the confirm candle's HA high


def test_no_sell_signals_while_ema_state_bullish() -> None:
    s = _strategy()
    decisions = _bull_setup(s)
    assert all(d is None or not d.sell_signal for d in decisions)
    assert not s._hunt_bear and s._pending_long


def test_sl_exit_at_range_low() -> None:
    s = _strategy()
    _bull_setup(s)
    d_entry = s.update(_c(_ts(10, 30), 110.0, 112.0, 109.0, 111.0))
    sl = d_entry.sl_level
    d = s.update(_c(_ts(10, 35), 111.0, 111.5, sl - 1.0, sl - 0.5))
    assert d is not None and d.long_exit and d.exit_reason == "SL"
    assert d.long_exit_price == pytest.approx(sl)
    assert s.position_state == PositionState.FLAT


def test_ema_cross_exit_closes_long() -> None:
    s = _strategy()
    _bull_setup(s)
    d_entry = s.update(_c(_ts(10, 30), 110.0, 112.0, 109.0, 111.0))
    sl = d_entry.sl_level
    # Falling closes flip EMA2 below EMA4 without ever touching the SL.
    d = None
    for i, cl in enumerate((106.0, 102.0, 99.0)):
        assert cl > sl
        d = s.update(_c(_ts(10, 35 + 5 * i), cl + 1.0, cl + 2.0, max(cl - 1.0, sl + 1.0), cl))
        if d is not None:
            break
    assert d is not None and d.long_exit and d.exit_reason == "EMA_CROSS"
    assert s.position_state == PositionState.FLAT


def test_invalidation_before_trigger_discards_setup() -> None:
    s = _strategy()
    _bull_setup(s)
    sl = s._pending_sl
    d = s.update(_c(_ts(10, 30), 104.0, 105.0, sl - 1.0, 104.0))  # hits SL side first
    assert d is None
    assert not s.has_pending and s.position_state == PositionState.FLAT


def test_no_eod_close_position_survives_1725() -> None:
    s = _strategy()
    _bull_setup(s, day=8)
    s.update(_c(_ts(10, 30), 110.0, 112.0, 109.0, 111.0))
    assert s.position_state == PositionState.LONG
    # Cross 17:25 (square-off edge) and sit inside the gap: still long.
    for hm in ((17, 20), (17, 25), (17, 27)):
        s.update(_c(_ts(*hm), 111.0, 112.0, 110.5, 111.5))
    assert s.position_state == PositionState.LONG


def test_weekend_block_consumes_trigger_without_trade() -> None:
    s = _strategy(skip_weekdays=frozenset({5, 6}))
    _bull_setup(s, day=11)   # Saturday IST
    assert s.has_pending
    d = s.update(_c(_ts(10, 30, day=11), 110.0, 112.0, 109.0, 111.0))
    assert d is None
    assert not s.has_pending and s.position_state == PositionState.FLAT


def test_gap_cancels_pending_and_rearms_after() -> None:
    s = _strategy()
    _bull_setup(s)
    assert s.has_pending
    s.update(_c(_ts(17, 26), 110.0, 110.5, 109.5, 110.0))   # inside 17:25-17:30 gap
    assert not s.has_pending
    s.update(_c(_ts(17, 30), 110.0, 110.5, 109.5, 110.0))   # gap over
    assert s._hunt_bull or s._hunt_bear   # state-based arming resumed immediately
