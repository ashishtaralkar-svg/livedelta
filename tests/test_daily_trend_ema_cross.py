"""DailyTrendEmaCrossStrategy: daily bear-trigger/reclaim trend flips, the
EMA/MA-cross-armed HA green->red setup scan, next-bar-only entry, fixed SL,
and the trend-flip-must-not-wipe-an-open-position's-stop regression."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.enums import PositionState
from deltabot.models import Candle
from deltabot.strategy.daily_trend_ema_cross import DailyTrendEmaCrossStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ts(y: int, m: int, d: int, hour: int, minute: int) -> int:
    return int(datetime(y, m, d, hour, minute, tzinfo=_IST).timestamp())


def _c(ts: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=ts, open=o, high=h, low=low, close=cl, volume=1.0)


def _strategy(**kw) -> DailyTrendEmaCrossStrategy:
    base = dict(ema_len=3, ma_len=3)
    base.update(kw)
    return DailyTrendEmaCrossStrategy(**base)


def test_bear_trigger_flips_trend_down_with_correct_pivot() -> None:
    s = _strategy()
    s.update(_c(_ts(2026, 7, 1, 10, 0), 100, 111, 99, 110))   # day1: green
    s.update(_c(_ts(2026, 7, 2, 10, 0), 105, 106, 90, 95))    # day2: red, closes < day1 open(100)
    s.update(_c(_ts(2026, 7, 3, 10, 0), 96, 97, 94, 96))      # day3 starts -> boundary evaluated
    assert s.trend == -1
    assert s._key_level == 105.0   # day2's OWN open, not day1's


def test_reclaim_flips_trend_back_up() -> None:
    s = _strategy()
    s.update(_c(_ts(2026, 7, 1, 10, 0), 100, 111, 99, 110))
    s.update(_c(_ts(2026, 7, 2, 10, 0), 105, 106, 90, 95))
    s.update(_c(_ts(2026, 7, 3, 10, 0), 96, 97, 94, 96))
    assert s.trend == -1
    s.update(_c(_ts(2026, 7, 3, 15, 0), 96, 112, 95, 110))    # day3 closes back above 105
    s.update(_c(_ts(2026, 7, 4, 10, 0), 111, 112, 109, 111))  # day4 starts -> boundary
    assert s.trend == 1
    assert s._key_level is None


def test_setup_scan_arms_pending_trigger_and_entry_waits_a_bar() -> None:
    s = _strategy()
    s._trend = -1
    s._warmup_bars = 100
    s._hunting_bear = True
    s._prev_ha_high, s._prev_ha_low = 105.0, 95.0   # stand-in for the green anchor candle
    s._prev_ha_green_bear = True
    s._ha_open, s._ha_close = 100.0, 100.0

    t = _ts(2026, 7, 10, 10, 0)
    d = s.update(_c(t, 100, 101, 90, 92))   # red-shaped HA candle -> completes the setup
    assert d is None
    assert s._pending_short_trigger == 90.0   # setup low
    assert s._pending_short_sl == 105.0       # setup high

    t += 300
    d = s.update(_c(t, 92, 93, 91, 92))   # no break yet
    assert d is None
    assert s.position_state == PositionState.FLAT

    t += 300
    d = s.update(_c(t, 89, 90, 85, 87))   # breaks below 90 -> fires
    assert d is not None and d.sell_signal and d.entry_price == 90.0
    assert s.position_state == PositionState.SHORT
    assert s._active_short_sl == 105.0
    assert s._pending_short_trigger is None


def test_sl_exit_short_at_setup_high() -> None:
    s = _strategy()
    s._in_short = True
    s._active_short_sl = 105.0
    d = s.update(_c(_ts(2026, 7, 10, 12, 0), 100, 106, 99, 104))   # high touches 105
    assert d is not None and d.short_exit and d.exit_reason == "SL"
    assert s.position_state == PositionState.FLAT


def test_sl_exit_long_at_setup_low() -> None:
    s = _strategy()
    s._in_long = True
    s._active_long_sl = 90.0
    d = s.update(_c(_ts(2026, 7, 10, 12, 0), 95, 96, 89, 94))   # low touches 90
    assert d is not None and d.long_exit and d.exit_reason == "SL"
    assert s.position_state == PositionState.FLAT


def test_trend_flip_mid_trade_does_not_wipe_the_open_positions_sl() -> None:
    """Regression: pending-setup SL and an OPEN position's SL must be
    tracked separately -- a trend flip used to clear both, which would have
    left a live position with no stop at all."""
    s = _strategy()
    s._in_short = True
    s._active_short_sl = 105.0
    s._trend = -1
    # day1 RED so day2 can never itself satisfy "prev day green" -> day2's own
    # bear_trigger is guaranteed false, leaving the manually-injected
    # trend/key_level below untouched by the day3 boundary evaluation. Highs
    # kept well under the injected active_short_sl (105) so the SL doesn't
    # fire prematurely on one of these setup bars.
    s.update(_c(_ts(2026, 7, 1, 10, 0), 104, 104.5, 99, 100))   # day1: red
    s.update(_c(_ts(2026, 7, 2, 10, 0), 101, 102, 90, 95))      # day2: red too
    # Manually set up a reclaim so the NEXT boundary flips trend while short is open.
    s._trend, s._key_level = -1, 50.0   # an artificially low pivot that gets reclaimed immediately
    d = s.update(_c(_ts(2026, 7, 3, 10, 0), 96, 97, 94, 96))   # day3 starts, close(96) > key_level(50) -> flips UP
    assert s.trend == 1
    assert s.position_state == PositionState.SHORT   # still open
    assert s._active_short_sl == 105.0                # SL survived the flip
    # And the SL still actually works afterwards:
    d2 = s.update(_c(_ts(2026, 7, 3, 12, 0), 100, 106, 99, 104))
    assert d2 is not None and d2.short_exit and d2.exit_reason == "SL"


def test_pending_setup_cleared_on_trend_flip() -> None:
    s = _strategy()
    s._trend = -1
    s._pending_short_trigger = 90.0
    s._pending_short_sl = 105.0
    s._hunting_bear = True
    s._key_level = 50.0
    d = s.update(_c(_ts(2026, 7, 1, 10, 0), 60, 61, 59, 60))   # some bar
    s.update(_c(_ts(2026, 7, 2, 10, 0), 105, 106, 90, 95))
    d = s.update(_c(_ts(2026, 7, 3, 10, 0), 96, 97, 94, 96))   # boundary; close(96)>key_level(50) -> flips UP
    assert s.trend == 1
    assert s._pending_short_trigger is None
    assert not s._hunting_bear
