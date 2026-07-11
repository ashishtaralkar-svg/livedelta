"""DchannelStrategy (2026-07-10 rewrite): Williams %R arming, Donchian-touch
+ open=low/open=high confirmation on synthetic Heikin Ashi candles, EMA1000
trend filter, ASAP breakout entry with pre-entry invalidation, and REAL-price
SL/EOD exits. Option BUYING (unlike the pre-rewrite version).
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.enums import PositionState
from deltabot.models import Candle
from deltabot.strategy.dchannel import DchannelStrategy

_IST = ZoneInfo("Asia/Kolkata")
_BASE_TS = 1_700_000_000


def _c(start: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _new_strategy() -> DchannelStrategy:
    return DchannelStrategy(dc_period=5, wr_period=5, ema_length=5)


def _ts_ist(h: int, m: int) -> int:
    return int(datetime(2026, 7, 10, h, m, tzinfo=_IST).timestamp())


# --- Session-open-line filter -------------------------------------------- #
def test_session_line_captured_at_1730_from_real_open() -> None:
    s = _new_strategy()
    s.update(_c(_ts_ist(17, 25), 100.0, 101.0, 99.0, 100.0))            # prev_now_mins = 1045
    s.update(_c(_ts_ist(17, 30), 62500.0, 62600.0, 62400.0, 62550.0))  # crosses 17:30
    assert s._session_line == 62500.0  # the REAL open, not HA


def test_session_filter_off_is_noop() -> None:
    s = DchannelStrategy(session_line_filter=False)
    assert s._session_ok(100.0, bull=True) is True
    assert s._session_ok(100.0, bull=False) is True


def test_session_filter_blocks_when_line_unset() -> None:
    s = DchannelStrategy(session_line_filter=True)
    assert s._session_line is None
    assert s._session_ok(100.0, bull=True) is False


def test_session_filter_bull_needs_close_above_line() -> None:
    s = DchannelStrategy(session_line_filter=True)
    s._session_line = 60000.0
    assert s._session_ok(60001.0, bull=True) is True
    assert s._session_ok(59999.0, bull=True) is False


def test_session_filter_bear_needs_close_below_line() -> None:
    s = DchannelStrategy(session_line_filter=True)
    s._session_line = 60000.0
    assert s._session_ok(59999.0, bull=False) is True
    assert s._session_ok(60001.0, bull=False) is False


def _make_ready(s: DchannelStrategy, dc_upper: float, dc_lower: float,
                wr_hh: float, wr_ll: float, ema: float) -> None:
    """Poke the strategy into a fully warmed-up state with fixed indicator
    values, bypassing real multi-bar warmup (matches this codebase's
    established test convention)."""
    s._dc._highs = deque([dc_upper] * s.dc_period, maxlen=s.dc_period)
    s._dc._lows = deque([dc_lower] * s.dc_period, maxlen=s.dc_period)
    s._wr._highs = deque([wr_hh] * s.wr_period, maxlen=s.wr_period)
    s._wr._lows = deque([wr_ll] * s.wr_period, maxlen=s.wr_period)
    s._warmup_bars = s.ema_length
    s._ema._value = ema
    # _ha_open/_ha_close deliberately left None -- update() seeds them from
    # the first REAL candle's own (open+close)/2, which keeps HA values in
    # the actual price range instead of an arbitrary seed distorting them.


# ---------------------------------------------------------------------- #
# Williams %R arming
# ---------------------------------------------------------------------- #
def test_wr_below_minus80_arms_bull_hunt() -> None:
    s = _new_strategy()
    # wr_hh/wr_ll fixed at 110/90; feeding a low close pushes %R deep negative.
    _make_ready(s, dc_upper=200.0, dc_lower=0.0, wr_hh=110.0, wr_ll=90.0, ema=100.0)
    dec = s.update(_c(_BASE_TS, 92.0, 93.0, 91.0, 91.5))  # %R = (110-91.5)/20*-100 = -92.5
    assert dec is None
    assert s._hunt_bull is True
    assert s._hunt_bear is False


def test_wr_near_zero_arms_bear_hunt() -> None:
    s = _new_strategy()
    # wr_hh/wr_ll fixed at 100/80; a close right at the window high pushes
    # %R to ~0, deep inside the "overbought" zone (wr > -20).
    _make_ready(s, dc_upper=200.0, dc_lower=0.0, wr_hh=100.0, wr_ll=80.0, ema=100.0)
    dec = s.update(_c(_BASE_TS, 99.0, 100.0, 98.5, 99.9))  # hh becomes 100, %R = (100-99.9)/20*-100 = -0.5
    assert dec is None
    assert s._hunt_bear is True
    assert s._hunt_bull is False


def test_wr_midrange_arms_neither() -> None:
    s = _new_strategy()
    _make_ready(s, dc_upper=200.0, dc_lower=0.0, wr_hh=110.0, wr_ll=90.0, ema=100.0)
    dec = s.update(_c(_BASE_TS, 99.0, 100.0, 98.0, 99.5))  # %R = (110-99.5)/20*-100 = -52.5, neither zone
    assert dec is None
    assert not s._hunt_bull and not s._hunt_bear


def test_opposite_wr_cross_switches_bull_hunt_to_bear() -> None:
    s = _new_strategy()
    _make_ready(s, dc_upper=200.0, dc_lower=0.0, wr_hh=100.0, wr_ll=80.0, ema=100.0)
    s._hunt_bull = True
    s._touched_bull = True
    s._range_hi_bull, s._range_lo_bull = 105.0, 95.0
    # A close pinning %R near 0 (overbought) should switch bull -> bear.
    dec = s.update(_c(_BASE_TS, 99.0, 100.0, 98.5, 99.9))
    assert dec is None
    assert s._hunt_bear is True
    assert s._hunt_bull is False
    assert s._touched_bull is False  # the abandoned bull hunt's progress is cleared
    assert s._range_hi_bull is None and s._range_lo_bull is None


# ---------------------------------------------------------------------- #
# Donchian touch starts range accumulation; open==low + EMA finalizes it
# ---------------------------------------------------------------------- #
def test_dc_touch_then_open_eq_low_above_ema_finalizes_pending_long() -> None:
    s = _new_strategy()
    _make_ready(s, dc_upper=200.0, dc_lower=50.0, wr_hh=110.0, wr_ll=90.0, ema=40.0)
    s._hunt_bull = True
    # Bar 1: low touches dc_lower (50) -> starts range accumulation.
    dec1 = s.update(_c(_BASE_TS, 52.0, 53.0, 49.0, 51.0))
    assert dec1 is None
    assert s._touched_bull is True
    # Bar 2: open == low (HA), close above EMA(40) -> finalize.
    # HA open is a smoothed running value, NOT the real candle's own open --
    # after bar 1 (o=52,h=53,l=49,c=51 -> ha_open=51.5, ha_close=51.25), bar
    # 2's ha_open = (51.5+51.25)/2 = 51.375 (fixed regardless of bar 2's real
    # OHLC). For ha_low to equal that carried-over ha_open, bar 2's REAL low
    # must sit ABOVE 51.375 (so ha_open, not the real low, becomes the min).
    dec2 = s.update(_c(_BASE_TS + 60, 52.0, 60.0, 52.0, 59.0))
    assert dec2 is None  # arming only, no entry yet
    assert s.has_pending
    assert s._pending_long is True
    assert s._pending_sl is not None and s._pending_trigger is not None
    assert s._pending_trigger >= s._pending_sl


def test_signal_range_accumulates_across_multiple_bars() -> None:
    s = _new_strategy()
    _make_ready(s, dc_upper=200.0, dc_lower=50.0, wr_hh=110.0, wr_ll=90.0, ema=10.0)
    s._hunt_bull = True
    s.update(_c(_BASE_TS, 52.0, 53.0, 49.0, 51.0))  # touch -> range starts ~ (53,49) HA-adjusted
    assert s._touched_bull is True
    range_hi_after_touch = s._range_hi_bull
    # A non-confirming bar in between (open != low) should still widen the range.
    dec = s.update(_c(_BASE_TS + 60, 51.0, 58.0, 50.0, 55.0))
    assert dec is None
    assert not s.has_pending  # not yet confirmed (open != low)
    assert s._range_hi_bull >= range_hi_after_touch


def test_ratchet_resets_range_to_more_oversold_lower_high_candle() -> None:
    """Replicates the 2026-07-10 18:14->18:16 real scenario: a touch candle
    (18:14) is superseded by a LATER candle (18:16) that is both more
    oversold AND has a lower high -- the range should RESET to start fresh
    from 18:16, not keep accumulating from 18:14."""
    s = _new_strategy()
    s._dc._highs = deque([200.0] * s.dc_period, maxlen=s.dc_period)
    s._dc._lows = deque([50.0] * s.dc_period, maxlen=s.dc_period)
    s._warmup_bars = s.ema_length
    s._ema._value = 10.0
    s._hunt_bull = True
    s._touched_bull = True
    s._range_hi_bull, s._range_lo_bull = 100.0, 90.0
    s._ref_wr_bull = -88.7  # the 18:14-equivalent reference
    # Directly poke a MORE oversold wr window (wr will compute deep negative)
    # and feed a candle with a LOWER high than the current reference (100).
    s._wr._highs = deque([95.0] * s.wr_period, maxlen=s.wr_period)
    s._wr._lows = deque([80.0] * s.wr_period, maxlen=s.wr_period)
    # This candle's real low/high are chosen so its own HA high ends up < 100.
    dec = s.update(_c(_BASE_TS, 82.0, 83.0, 80.5, 81.0))
    assert dec is None
    # Range should have RESET (not accumulated) -- new hi should be this
    # candle's own HA high, well below the old reference's 100.
    assert s._range_hi_bull < 100.0
    assert s._ref_wr_bull is not None and s._ref_wr_bull < -88.7


def test_non_extreme_candle_does_not_ratchet_just_widens() -> None:
    """A candle that is NOT more oversold than the current reference should
    only widen the accumulated range (existing max/min behavior), not reset it."""
    s = _new_strategy()
    s._dc._highs = deque([200.0] * s.dc_period, maxlen=s.dc_period)
    s._dc._lows = deque([50.0] * s.dc_period, maxlen=s.dc_period)
    s._warmup_bars = s.ema_length
    s._ema._value = 10.0
    s._hunt_bull = True
    s._touched_bull = True
    s._range_hi_bull, s._range_lo_bull = 100.0, 90.0
    s._ref_wr_bull = -88.7
    # wr window fixed so this bar's wr computes LESS oversold than -88.7.
    s._wr._highs = deque([110.0] * s.wr_period, maxlen=s.wr_period)
    s._wr._lows = deque([90.0] * s.wr_period, maxlen=s.wr_period)
    dec = s.update(_c(_BASE_TS, 99.0, 105.0, 98.0, 99.5))  # high(105) > old range_hi(100) -> widens
    assert dec is None
    assert s._range_hi_bull == 105.0  # widened via max(), not reset
    assert s._ref_wr_bull == -88.7  # reference unchanged -- no ratchet


# ---------------------------------------------------------------------- #
# Entry: ASAP breakout of the finalized signal range, real price
# ---------------------------------------------------------------------- #
def test_entry_fires_on_range_high_break() -> None:
    s = _new_strategy()
    s._pending_long = True
    s._pending_trigger = 60.0
    s._pending_sl = 48.0
    confirmed, invalidated, entry = s.apply_intracandle_pending(_c(1, 59.0, 61.0, 58.5, 60.5))
    assert confirmed and not invalidated
    assert entry == 60.0
    assert s.position_state == PositionState.LONG
    assert s.sl_level == 48.0
    assert not s.has_pending
    # TP = entry + rr_multiple(2.0) * (entry - sl) = 60 + 2*(60-48) = 84
    assert s._tp_level == 84.0


def test_short_entry_tp_is_1to2_below_entry() -> None:
    s = _new_strategy()
    s._pending_short = True
    s._pending_trigger = 40.0
    s._pending_sl = 46.0
    confirmed, invalidated, entry = s.apply_intracandle_pending(_c(1, 41.0, 41.5, 39.0, 39.5))
    assert confirmed and not invalidated
    assert entry == 40.0
    # TP = entry - rr_multiple(2.0) * (sl - entry) = 40 - 2*(46-40) = 28
    assert s._tp_level == 28.0


def test_tp_exit_fires_before_eod_on_closed_bar() -> None:
    s = _new_strategy()
    _make_ready(s, dc_upper=200.0, dc_lower=50.0, wr_hh=110.0, wr_ll=90.0, ema=10.0)
    s._in_long = True
    s._sl_level = 48.0
    s._tp_level = 84.0
    dec = s.update(_c(_BASE_TS, 80.0, 85.0, 79.0, 84.5))  # high crosses TP, low stays above SL
    assert dec is not None
    assert dec.exit_reason == "TP"
    assert dec.long_exit_price == 84.0
    assert s.position_state == PositionState.FLAT
    assert s._tp_level is None  # cleared on exit


def test_sl_wins_over_tp_on_same_bar() -> None:
    s = _new_strategy()
    _make_ready(s, dc_upper=200.0, dc_lower=50.0, wr_hh=110.0, wr_ll=90.0, ema=10.0)
    s._in_long = True
    s._sl_level = 48.0
    s._tp_level = 84.0
    # A wide bar sweeping BOTH levels -- SL must win (conservative), matching
    # every other strategy's same-bar priority.
    dec = s.update(_c(_BASE_TS, 60.0, 90.0, 47.0, 85.0))
    assert dec is not None
    assert dec.exit_reason == "SL"


def test_pre_entry_invalidation_when_low_breaks_first() -> None:
    s = _new_strategy()
    s._pending_long = True
    s._pending_trigger = 60.0
    s._pending_sl = 48.0
    confirmed, invalidated, _ = s.apply_intracandle_pending(_c(1, 50.0, 50.5, 47.5, 48.0))
    assert invalidated and not confirmed
    assert s.position_state == PositionState.FLAT
    assert not s.has_pending


# ---------------------------------------------------------------------- #
# Exits: SL / EOD
# ---------------------------------------------------------------------- #
def test_closed_bar_sl_exit() -> None:
    s = _new_strategy()
    s._in_long = True
    s._sl_level = 48.0
    _make_ready(s, dc_upper=200.0, dc_lower=50.0, wr_hh=110.0, wr_ll=90.0, ema=10.0)
    dec = s.update(_c(_BASE_TS, 50.0, 51.0, 47.0, 49.0))  # low breaches SL
    assert dec is not None
    assert dec.exit_reason == "SL"
    assert dec.long_exit_price == 48.0
    assert s.position_state == PositionState.FLAT


def test_eod_square_off_closes_short_and_clears_hunt() -> None:
    s = DchannelStrategy(dc_period=5, wr_period=5, ema_length=5,
                         square_off_hour=17, square_off_minute=25)
    t1 = int(datetime(2026, 7, 6, 17, 20, tzinfo=_IST).timestamp())
    _make_ready(s, dc_upper=200.0, dc_lower=50.0, wr_hh=110.0, wr_ll=90.0, ema=10.0)
    s.update(_c(t1, 100.0, 101.0, 99.0, 100.5))  # establishes _prev_now_mins

    s._in_short = True
    s._sl_level = 500.0  # far away -- SL must not be the cause
    s._hunt_bull = True  # a stray hunt too, should also be cancelled

    t2 = int(datetime(2026, 7, 6, 17, 25, tzinfo=_IST).timestamp())
    dec = s.update(_c(t2, 100.0, 101.0, 99.0, 100.0))
    assert dec is not None
    assert dec.exit_reason == "EOD"
    assert s.position_state == PositionState.FLAT
    assert not s._hunt_bull


def test_notify_exit_flattens_position() -> None:
    s = _new_strategy()
    s._in_long = True
    s._sl_level = 90.0
    s.notify_exit("TP")
    assert s.position_state == PositionState.FLAT


def test_force_flat_clears_everything() -> None:
    s = _new_strategy()
    s._in_short = True
    s._pending_long = True
    s._hunt_bear = True
    s._touched_bear = True
    s.force_flat()
    assert s.position_state == PositionState.FLAT
    assert not s.has_pending
    assert not s._hunt_bear and not s._touched_bear
