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


def test_trail_exit_skips_its_own_bar_like_sl() -> None:
    """After a TRAIL exit (like an SL exit), the exit bar must NOT start a new
    range even if it touches a Donchian band -- hunting resumes NEXT bar."""
    s = _strategy()
    # Put the strategy directly into an armed trail-mode long with a far SL.
    s._in_long, s._exit_mode, s._trail_armed, s._sl_level = True, "trail", True, 50.0
    s._warmup_bars = 100
    s._ema_trend._value = s._ema_long._value = 200.0   # EMAs high -> a low close is below both
    for _ in range(s.dc_period):
        s._dc.push(100.0, 100.0)                        # dc_upper = dc_lower = 100

    # Exit candle closes at 100 (below both post-update EMAs -> TRAIL) and its
    # real high/low (101/99) touch BOTH Donchian bands.
    d = s.update(_c(_ts(10, 30), 100.0, 101.0, 99.0, 100.0))
    assert d is not None and d.long_exit and d.exit_reason == "TRAIL"
    assert not s._touched_bull and not s._touched_bear   # exit bar skipped, no touch registered


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


def test_buy_below_both_emas_uses_trail_mode_sl_until_armed() -> None:
    """A BUY triggering BELOW both EMAs enters "trail" mode: only the fixed
    range-low SL is live until the trail arms, and the EMA relationship
    flipping does NOT close it."""
    s = _strategy()
    for i, cl in enumerate((100.0, 101.0, 102.0, 103.0)):        # bull warmup, EMAs ~101-103
        s.update(_c(_ts(10, 0) + i * 300, cl - 0.5, cl + 0.5, cl - 1.0, cl))
    s._pending_long, s._pending_trigger, s._pending_sl = True, 95.0, 90.0

    d = s.update(_c(_ts(10, 20), 94.0, 96.0, 93.0, 95.0))        # triggers at 95, below both EMAs
    assert d is not None and d.buy_signal and d.entry_price == 95.0
    assert s._exit_mode == "trail" and s._trail_armed is False

    # Close below both EMAs while NOT armed: neither EMA_CROSS nor TRAIL fires.
    d = s.update(_c(_ts(10, 25), 95.0, 96.0, 94.0, 95.0))
    assert d is None and s.position_state == PositionState.LONG

    d = s.update(_c(_ts(10, 30), 95.0, 95.5, 89.0, 89.5))        # SL still live at 90
    assert d is not None and d.long_exit and d.exit_reason == "SL"
    assert d.long_exit_price == pytest.approx(90.0)
    assert s._exit_mode == "cross"                                # reset for the next trade


def test_buy_trail_arms_above_both_emas_then_close_below_exits() -> None:
    s = _strategy()
    for i, cl in enumerate((100.0, 101.0, 102.0, 103.0)):
        s.update(_c(_ts(10, 0) + i * 300, cl - 0.5, cl + 0.5, cl - 1.0, cl))
    s._pending_long, s._pending_trigger, s._pending_sl = True, 95.0, 90.0
    s.update(_c(_ts(10, 20), 94.0, 96.0, 93.0, 95.0))            # entry at 95 (trail mode)

    d = s.update(_c(_ts(10, 25), 110.0, 122.0, 109.0, 120.0))    # closes far above both EMAs
    assert d is None and s._trail_armed is True                   # armed, trade stays open

    d = s.update(_c(_ts(10, 30), 96.0, 97.0, 94.0, 95.0))        # closes below both, low > SL 90
    assert d is not None and d.long_exit and d.exit_reason == "TRAIL"
    assert d.long_exit_price == pytest.approx(95.0)               # exits at the close
    assert s.position_state == PositionState.FLAT


def test_sell_above_both_emas_uses_trail_mode() -> None:
    s = _strategy()
    for i, cl in enumerate((103.0, 102.0, 101.0, 100.0)):        # bear warmup, EMAs ~100-102
        s.update(_c(_ts(10, 0) + i * 300, cl + 0.5, cl + 1.0, cl - 0.5, cl))
    s._pending_short, s._pending_trigger, s._pending_sl = True, 110.0, 115.0

    d = s.update(_c(_ts(10, 20), 111.0, 112.0, 109.0, 110.5))    # triggers at 110, above both EMAs
    assert d is not None and d.sell_signal and d.entry_price == 110.0
    assert s._exit_mode == "trail" and s._trail_armed is False

    # Closes above both EMAs (against the short) while NOT armed: no exit.
    d = None
    for i, cl in enumerate((111.0, 112.0, 113.0)):
        d = s.update(_c(_ts(10, 25 + 5 * i), cl - 0.5, cl + 0.5, cl - 1.0, cl))
        assert d is None
    assert s.position_state == PositionState.SHORT

    d = s.update(_c(_ts(10, 45), 114.0, 116.0, 113.5, 115.5))    # SL still live at 115
    assert d is not None and d.short_exit and d.exit_reason == "SL"
    assert d.short_exit_price == pytest.approx(115.0)


def test_sell_trail_arms_below_both_emas_then_close_above_exits() -> None:
    s = _strategy()
    for i, cl in enumerate((103.0, 102.0, 101.0, 100.0)):
        s.update(_c(_ts(10, 0) + i * 300, cl + 0.5, cl + 1.0, cl - 0.5, cl))
    s._pending_short, s._pending_trigger, s._pending_sl = True, 110.0, 115.0
    s.update(_c(_ts(10, 20), 111.0, 112.0, 109.0, 110.5))        # entry at 110 (trail mode)

    d = s.update(_c(_ts(10, 25), 90.0, 91.0, 78.0, 80.0))        # closes far below both EMAs
    assert d is None and s._trail_armed is True

    d = s.update(_c(_ts(10, 30), 104.0, 106.0, 103.0, 105.0))    # closes above both, high < SL 115
    assert d is not None and d.short_exit and d.exit_reason == "TRAIL"
    assert d.short_exit_price == pytest.approx(105.0)
    assert s.position_state == PositionState.FLAT


def test_next_green_raw_forms_two_candle_range() -> None:
    """Research variant: raw candles + next_green confirm. A DC-lower touch
    followed by an immediate GREEN candle arms a 2-candle-range long."""
    from deltabot.strategy.dcv2 import DCv2Strategy
    s = DCv2Strategy(dc_period=2, ema_trend_length=2, ema_long_length=4,
                     use_heikin_ashi=False, confirm_mode="next_green",
                     skip_weekdays=frozenset())
    t = _ts(10, 0)
    for i, cl in enumerate((100.0, 101.0, 102.0, 103.0)):   # bull warmup (raw closes rising)
        s.update(_c(t + i * 300, cl - 0.5, cl + 0.5, cl - 1.0, cl))
    # candle A: long lower wick to DC low, closes back up (keeps EMA bullish).
    s.update(_c(t + 4 * 300, 103.0, 104.0, 95.0, 103.0))
    assert s._touched_bull is True
    d = s.update(_c(t + 5 * 300, 100.0, 110.0, 100.0, 109.0))  # candle B: GREEN -> arm
    assert s.has_pending and s._pending_long
    assert s._pending_trigger == 110.0 and s._pending_sl == 95.0   # spans both candles


def test_next_green_discards_when_next_candle_not_green() -> None:
    from deltabot.strategy.dcv2 import DCv2Strategy
    s = DCv2Strategy(dc_period=2, ema_trend_length=2, ema_long_length=4,
                     use_heikin_ashi=False, confirm_mode="next_green",
                     skip_weekdays=frozenset())
    t = _ts(10, 0)
    for i, cl in enumerate((100.0, 101.0, 102.0, 103.0)):
        s.update(_c(t + i * 300, cl - 0.5, cl + 0.5, cl - 1.0, cl))
    s.update(_c(t + 4 * 300, 103.0, 104.0, 95.0, 103.0))    # touch (long lower wick, closes up)
    assert s._touched_bull is True
    s.update(_c(t + 5 * 300, 105.0, 106.0, 101.0, 104.0))   # RED, no re-touch -> discard
    assert not s.has_pending and s._touched_bull is False


def test_gap_cancels_pending_and_rearms_after() -> None:
    s = _strategy()
    _bull_setup(s)
    assert s.has_pending
    s.update(_c(_ts(17, 26), 110.0, 110.5, 109.5, 110.0))   # inside 17:25-17:30 gap
    assert not s.has_pending
    s.update(_c(_ts(17, 30), 110.0, 110.5, 109.5, 110.0))   # gap over
    assert s._hunt_bull or s._hunt_bear   # state-based arming resumed immediately
