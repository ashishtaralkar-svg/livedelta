"""HeikinAshiStrategy: HA conversion, pattern arming, ASAP intracandle entry/SL,
and invalidate-before-trigger ordering.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.enums import PositionState, SignalDir
from deltabot.models import Candle
from deltabot.strategy.heikin_ashi import HeikinAshiStrategy

_IST = ZoneInfo("Asia/Kolkata")
# Real-epoch anchor for tests that use small relative offsets (0, 300, ...) as
# `start` -- update() now subtracts the day_start offset (17:30 IST) before
# calling datetime.fromtimestamp for the session-open gate, which goes negative
# (and raises on Windows) for tiny synthetic timestamps near epoch 0.
_BASE_TS = 1_700_000_000


def _c(start: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _arm_short(strat: HeikinAshiStrategy) -> None:
    strat._pending_short = True
    strat._pending_long = False
    strat._pending_trigger = 60_000.0   # enter when price breaks BELOW this
    strat._pending_sl = 60_300.0        # invalid if price breaks ABOVE this first


def _arm_long(strat: HeikinAshiStrategy) -> None:
    strat._pending_long = True
    strat._pending_short = False
    strat._pending_trigger = 60_300.0   # enter when price breaks ABOVE this
    strat._pending_sl = 60_000.0        # invalid if price breaks BELOW this first


# ---------------------------------------------------------------------- #
# Heikin Ashi conversion
# ---------------------------------------------------------------------- #
def test_heikin_ashi_conversion_matches_hand_computed_values() -> None:
    s = HeikinAshiStrategy()
    # Bar 1: seeded ha_open = (open+close)/2.
    s.update(_c(_BASE_TS, 100.0, 110.0, 90.0, 105.0))
    assert s._ha_open == (100.0 + 105.0) / 2.0        # 102.5
    assert s._ha_close == (100.0 + 110.0 + 90.0 + 105.0) / 4.0  # 101.25

    prev_open, prev_close = s._ha_open, s._ha_close
    s.update(_c(_BASE_TS + 300, 106.0, 112.0, 104.0, 108.0))
    expected_open = (prev_open + prev_close) / 2.0
    expected_close = (106.0 + 112.0 + 104.0 + 108.0) / 4.0
    assert s._ha_open == expected_open
    assert s._ha_close == expected_close


# ---------------------------------------------------------------------- #
# Pattern arming (state poked directly to bypass the 50-bar warmup, mirroring
# how test_revbreak_intracandle.py isolates the intracandle logic)
# ---------------------------------------------------------------------- #
def test_buy_pattern_arms_pending_long() -> None:
    s = HeikinAshiStrategy()
    s._warmup_bars = 199          # ready needs warmup_bars >= ema200_length(200)
    s._ha_open, s._ha_close = 100.0, 90.0          # running HA state from the prior bar
    s._prev_ha = (100.0, 100.0, 85.0, 90.0)        # (o,h,l,c): only p_h/p_l matter now (SL/trigger widening)
    s._st._ready = True
    s._st._direction = SignalDir.LONG.value        # bullish Supertrend gates the BUY pattern
    s._cur_open = 90.0                              # session open below this bar's close -> bull_gate
    s._ema._value = 90.0                            # ema50, well above ema200 -- satisfies trend_up
    s._ema200._value = 50.0                         # ema200, well below ha_close and ema50

    # ha_open this bar = (100+90)/2 = 95. Real candle -> ha_close=(96+105+95+100)/4=99,
    # ha_high=max(105,95,99)=105, ha_low=min(95,95,99)=95 == ha_open (single-candle pattern).
    dec = s.update(_c(_BASE_TS, 96.0, 105.0, 95.0, 100.0))
    assert dec is None  # arming only, no entry/exit this bar
    assert s.has_pending
    assert s._pending_long and not s._pending_short
    assert s._pending_trigger == 105.0   # max(ha_high=105, prev_high=100)
    assert s._pending_sl == 85.0         # min(ha_low=95, prev_low=85)


def test_ema200_filter_blocks_buy_pattern() -> None:
    """Same bullish-Supertrend + valid pattern + session-open gate as the
    arming test, but ha_close is BELOW ema200 -- the trend filter alone must
    block the BUY."""
    s = HeikinAshiStrategy()
    s._warmup_bars = 199
    s._ha_open, s._ha_close = 100.0, 90.0
    s._prev_ha = (100.0, 100.0, 85.0, 90.0)
    s._st._ready = True
    s._st._direction = SignalDir.LONG.value
    s._cur_open = 90.0
    s._ema._value = 90.0
    s._ema200._value = 150.0   # ABOVE this bar's ha_close (99) -> trend_up False

    dec = s.update(_c(_BASE_TS, 96.0, 105.0, 95.0, 100.0))
    assert dec is None
    assert not s.has_pending


def test_buy_needs_close_above_ema50_not_just_above_ema200() -> None:
    """The trend filter is a CHAINED stack (ha_close > ema50 > ema200), not
    two independent checks against ema200. Craft a case where the OLD filter
    (ha_close > ema200 AND ema50 > ema200) would have passed -- ema50 is well
    above ema200, and ha_close is above ema200 too -- but ha_close is BELOW
    ema50. The new chained filter must block this."""
    s = HeikinAshiStrategy()
    s._warmup_bars = 199
    s._ha_open, s._ha_close = 100.0, 90.0
    s._prev_ha = (100.0, 100.0, 85.0, 90.0)
    s._st._ready = True
    s._st._direction = SignalDir.LONG.value
    s._cur_open = 90.0
    s._ema._value = 400.0     # ema50 far above ha_close (99) -- old filter didn't check this
    s._ema200._value = 10.0   # ema200 well below ha_close -- old filter's two checks both pass

    dec = s.update(_c(_BASE_TS, 96.0, 105.0, 95.0, 100.0))
    assert dec is None
    assert not s.has_pending


def test_bearish_supertrend_blocks_buy_pattern() -> None:
    """The exact same single-candle open==low pattern as above must NOT arm a
    BUY when Supertrend is bearish -- proves the gate actually blocks the
    mismatched side, not just permits the matched one."""
    s = HeikinAshiStrategy()
    s._warmup_bars = 199
    s._ha_open, s._ha_close = 100.0, 90.0
    s._prev_ha = (100.0, 100.0, 85.0, 90.0)
    s._st._ready = True
    s._st._direction = SignalDir.SHORT.value       # bearish -- wrong side for BUY
    s._cur_open = 90.0                              # session-open gate satisfied -- isolates the ST gate
    s._ema._value = 90.0
    s._ema200._value = 50.0                         # EMA200 filter satisfied -- isolates the ST gate

    dec = s.update(_c(_BASE_TS, 96.0, 105.0, 95.0, 100.0))
    assert dec is None
    assert not s.has_pending


def test_buy_blocked_when_price_below_session_open() -> None:
    """Same bullish-Supertrend + valid pattern as the arming test, but REAL
    price (this bar's close, 100) is below the session's opening price -- the
    session-open gate alone must block the BUY."""
    s = HeikinAshiStrategy()
    s._warmup_bars = 199
    s._ha_open, s._ha_close = 100.0, 90.0
    s._prev_ha = (100.0, 100.0, 85.0, 90.0)
    s._st._ready = True
    s._st._direction = SignalDir.LONG.value
    s._cur_open = 110.0                             # ABOVE this bar's close (100) -> bull_gate False
    s._ema._value = 90.0
    s._ema200._value = 50.0

    dec = s.update(_c(_BASE_TS, 96.0, 105.0, 95.0, 100.0))
    assert dec is None
    assert not s.has_pending


def test_sell_pattern_arms_pending_short() -> None:
    s = HeikinAshiStrategy()
    s._warmup_bars = 199
    s._ha_open, s._ha_close = 100.0, 110.0          # prior bar
    s._prev_ha = (100.0, 115.0, 100.0, 110.0)       # (o,h,l,c): only p_h/p_l matter now
    s._st._ready = True
    s._st._direction = SignalDir.SHORT.value       # bearish Supertrend gates the SELL pattern
    s._cur_open = 110.0                             # session open above this bar's close -> bear_gate
    s._ema._value = 110.0                           # ema50, well below ema200 -- satisfies trend_down
    s._ema200._value = 150.0                        # ema200, well above ha_close and ema50

    # ha_open this bar = (100+110)/2 = 105. Real candle -> ha_close=(104+105+95+100)/4=101,
    # ha_high=max(105,105,101)=105 == ha_open (single-candle pattern), ha_low=min(95,105,101)=95.
    dec = s.update(_c(_BASE_TS, 104.0, 105.0, 95.0, 100.0))
    assert dec is None
    assert s.has_pending
    assert s._pending_short and not s._pending_long
    assert s._pending_trigger == 95.0    # min(ha_low=95, prev_low=100)
    assert s._pending_sl == 115.0        # max(ha_high=105, prev_high=115)


def test_ema200_filter_blocks_sell_pattern() -> None:
    """Same bearish-Supertrend + valid pattern + session-open gate as the
    arming test, but ha_close is ABOVE ema200 -- the trend filter alone must
    block the SELL."""
    s = HeikinAshiStrategy()
    s._warmup_bars = 199
    s._ha_open, s._ha_close = 100.0, 110.0
    s._prev_ha = (100.0, 115.0, 100.0, 110.0)
    s._st._ready = True
    s._st._direction = SignalDir.SHORT.value
    s._cur_open = 110.0
    s._ema._value = 110.0
    s._ema200._value = 50.0   # BELOW this bar's ha_close (101) -> trend_down False

    dec = s.update(_c(_BASE_TS, 104.0, 105.0, 95.0, 100.0))
    assert dec is None
    assert not s.has_pending


def test_sell_needs_close_below_ema50_not_just_below_ema200() -> None:
    """Mirror of the buy-side chained-filter test: ema50 far below ha_close
    and ema200 far above ha_close would have passed the OLD two-independent-
    checks filter, but the new chained filter (ha_close < ema50 < ema200)
    must block it since ha_close is ABOVE ema50."""
    s = HeikinAshiStrategy()
    s._warmup_bars = 199
    s._ha_open, s._ha_close = 100.0, 110.0
    s._prev_ha = (100.0, 115.0, 100.0, 110.0)
    s._st._ready = True
    s._st._direction = SignalDir.SHORT.value
    s._cur_open = 110.0
    s._ema._value = 10.0      # ema50 far below ha_close (101) -- old filter didn't check this
    s._ema200._value = 400.0  # ema200 well above ha_close -- old filter's two checks both pass

    dec = s.update(_c(_BASE_TS, 104.0, 105.0, 95.0, 100.0))
    assert dec is None
    assert not s.has_pending


def test_bullish_supertrend_blocks_sell_pattern() -> None:
    """Mirror of the bearish-blocks-buy test: a bullish Supertrend must block
    an otherwise-valid single-candle open==high SELL pattern."""
    s = HeikinAshiStrategy()
    s._warmup_bars = 199
    s._ha_open, s._ha_close = 100.0, 110.0
    s._prev_ha = (100.0, 115.0, 100.0, 110.0)
    s._st._ready = True
    s._st._direction = SignalDir.LONG.value        # bullish -- wrong side for SELL
    s._cur_open = 110.0                             # session-open gate satisfied -- isolates the ST gate
    s._ema._value = 110.0
    s._ema200._value = 150.0                        # EMA200 filter satisfied -- isolates the ST gate

    dec = s.update(_c(_BASE_TS, 104.0, 105.0, 95.0, 100.0))
    assert dec is None
    assert not s.has_pending


def test_sell_blocked_when_price_above_session_open() -> None:
    """Mirror of the buy-blocked-below-open test: bearish Supertrend + valid
    pattern, but REAL price (close 100) is above the session open -- the
    session-open gate alone must block the SELL."""
    s = HeikinAshiStrategy()
    s._warmup_bars = 199
    s._ha_open, s._ha_close = 100.0, 110.0
    s._prev_ha = (100.0, 115.0, 100.0, 110.0)
    s._st._ready = True
    s._st._direction = SignalDir.SHORT.value
    s._cur_open = 90.0                              # BELOW this bar's close (100) -> bear_gate False
    s._ema._value = 110.0
    s._ema200._value = 150.0

    dec = s.update(_c(_BASE_TS, 104.0, 105.0, 95.0, 100.0))
    assert dec is None
    assert not s.has_pending


def test_session_open_updates_on_day_start_rollover() -> None:
    """_cur_open tracks REAL price at the most recent day_start (17:30 IST
    default) and only updates again at the NEXT rollover."""
    s = HeikinAshiStrategy()
    t1 = int(datetime(2026, 7, 6, 17, 30, tzinfo=_IST).timestamp())
    s.update(_c(t1, 100.0, 101.0, 99.0, 100.5))
    assert s._cur_open == 100.0

    t2 = int(datetime(2026, 7, 6, 20, 0, tzinfo=_IST).timestamp())  # same session
    s.update(_c(t2, 105.0, 106.0, 104.0, 105.5))
    assert s._cur_open == 100.0                     # unchanged within the same session

    t3 = int(datetime(2026, 7, 7, 17, 30, tzinfo=_IST).timestamp())  # next session
    s.update(_c(t3, 110.0, 111.0, 109.0, 110.5))
    assert s._cur_open == 110.0                     # rolled over to the new session's open


# ---------------------------------------------------------------------- #
# Trailing SL exit: ASAP, real price crossing the 50 EMA level fixed as of the
# last CLOSED bar (mirrors the fixed-SL convention) -- fires immediately, not
# waiting for this bar's HA close to settle.
# ---------------------------------------------------------------------- #
def test_trail_closes_long_immediately_when_low_drops_below_last_closed_ema() -> None:
    s = HeikinAshiStrategy()
    s._in_long, s._in_short = True, False
    s._sl_level = 50.0        # far below this candle's low -- SL must not be the cause
    s._ema._value = 95.0      # EMA level fixed as of the last closed bar
    s._ha_open, s._ha_close = 95.0, 95.0
    s._prev_ha = (95.0, 96.0, 94.0, 95.0)

    # This bar's real low (86) drops below the fixed trail level (95) -- TRAIL
    # fires ASAP at that level, using real price, regardless of this bar's own
    # HA close.
    candle = _c(_BASE_TS, 88.0, 94.0, 86.0, 92.0)
    dec = s.update(candle)

    assert dec is not None
    assert dec.long_exit and not dec.short_exit
    assert dec.exit_reason == "TRAIL"
    assert dec.long_exit_price == 95.0   # closes AT the trail level, not this bar's close
    assert s.position_state == PositionState.FLAT
    assert s.sl_level is None


def test_trail_closes_short_immediately_when_high_rises_above_last_closed_ema() -> None:
    """Mirror of the long case: short closes the instant real price rises above
    the fixed trail level."""
    s = HeikinAshiStrategy()
    s._in_long, s._in_short = False, True
    s._sl_level = 200.0        # far above this candle's high -- SL must not be the cause
    s._ema._value = 95.0       # EMA level fixed as of the last closed bar
    s._ha_open, s._ha_close = 95.0, 95.0
    s._prev_ha = (95.0, 96.0, 94.0, 95.0)

    # This bar's real high (106) rises above the fixed trail level (95).
    candle = _c(_BASE_TS, 102.0, 106.0, 98.0, 104.0)
    dec = s.update(candle)

    assert dec is not None
    assert dec.short_exit and not dec.long_exit
    assert dec.exit_reason == "TRAIL"
    assert dec.short_exit_price == 95.0
    assert s.position_state == PositionState.FLAT
    assert s.sl_level is None


def test_check_intracandle_trail_after_long_entry() -> None:
    s = HeikinAshiStrategy()
    _arm_long(s)
    s.apply_intracandle_pending(_c(1, 60_050, 60_310, 60_020, 60_290))  # now LONG, sl=60_000
    s._ema._value = 60_100.0   # trail level fixed as of the last closed bar
    long_trail, short_trail, level = s.check_intracandle_trail(60_050.0)
    assert long_trail and not short_trail
    assert level == 60_100.0
    long_trail, short_trail, _ = s.check_intracandle_trail(60_200.0)
    assert not long_trail and not short_trail


def test_fixed_sl_takes_priority_over_trail_on_same_bar() -> None:
    """If the fixed SL and the trailing-EMA exit would both fire on the same
    closed bar, the fixed SL wins (matches the SL-before-EOD/trail precedence)."""
    s = HeikinAshiStrategy()
    s._in_long, s._in_short = True, False
    s._sl_level = 91.0         # THIS bar's low (86) breaches it
    s._ema._value = 95.0
    s._ha_open, s._ha_close = 95.0, 95.0
    s._prev_ha = (95.0, 96.0, 94.0, 95.0)

    candle = _c(_BASE_TS, 88.0, 94.0, 86.0, 92.0)
    dec = s.update(candle)

    assert dec is not None
    assert dec.exit_reason == "SL"
    assert dec.long_exit_price == 91.0


# ---------------------------------------------------------------------- #
# ASAP intracandle entry/SL + invalidate-before-trigger (mirrors
# test_revbreak_intracandle.py)
# ---------------------------------------------------------------------- #
def test_short_enters_when_price_breaks_below_low() -> None:
    s = HeikinAshiStrategy()
    _arm_short(s)
    confirmed, invalidated, entry = s.apply_intracandle_pending(_c(1, 60_050, 60_100, 59_990, 60_010))
    assert confirmed and not invalidated
    assert entry == 60_000.0
    assert s.position_state == PositionState.SHORT
    assert s.sl_level == 60_300.0
    assert not s.has_pending


def test_short_invalidated_when_high_hits_sl_first() -> None:
    s = HeikinAshiStrategy()
    _arm_short(s)
    confirmed, invalidated, _ = s.apply_intracandle_pending(_c(1, 60_100, 60_320, 60_050, 60_280))
    assert invalidated and not confirmed
    assert s.position_state == PositionState.FLAT
    assert not s.has_pending


def test_long_enters_when_price_breaks_above_high() -> None:
    s = HeikinAshiStrategy()
    _arm_long(s)
    confirmed, invalidated, entry = s.apply_intracandle_pending(_c(1, 60_050, 60_310, 60_020, 60_290))
    assert confirmed and not invalidated
    assert entry == 60_300.0
    assert s.position_state == PositionState.LONG
    assert s.sl_level == 60_000.0


def test_check_intracandle_sl_after_short_entry() -> None:
    s = HeikinAshiStrategy()
    _arm_short(s)
    s.apply_intracandle_pending(_c(1, 60_050, 60_100, 59_990, 60_010))  # now SHORT, sl=60_300
    long_sl, short_sl, level = s.check_intracandle_sl(60_320.0)
    assert short_sl and not long_sl
    assert level == 60_300.0
    long_sl, short_sl, _ = s.check_intracandle_sl(60_100.0)
    assert not short_sl and not long_sl


def test_force_flat_clears_position_and_pending() -> None:
    s = HeikinAshiStrategy()
    _arm_short(s)
    s.apply_intracandle_pending(_c(1, 60_050, 60_100, 59_990, 60_010))
    assert s.position_state == PositionState.SHORT
    s.force_flat()
    assert s.position_state == PositionState.FLAT
    assert not s.has_pending
    assert s.sl_level is None


# ---------------------------------------------------------------------- #
# Settlement gap (17:25-17:30 IST): pending setups are CANCELLED, not merely
# blocked -- matches the Pine script's cancel-all in the gap.
# ---------------------------------------------------------------------- #
def test_intracandle_pending_cancelled_in_settlement_gap() -> None:
    s = HeikinAshiStrategy()
    _arm_long(s)
    ts = int(datetime(2026, 7, 6, 17, 26, tzinfo=_IST).timestamp())
    # This bar crosses the 60_300 trigger -- but it is inside the gap, so the
    # setup must be cancelled untraded, not filled.
    confirmed, invalidated, _ = s.apply_intracandle_pending(_c(ts, 60_310, 60_400, 60_305, 60_350))
    assert invalidated and not confirmed
    assert not s.has_pending
    assert s.position_state == PositionState.FLAT


def test_update_cancels_pending_at_settlement() -> None:
    s = HeikinAshiStrategy()
    _arm_long(s)
    ts = int(datetime(2026, 7, 6, 17, 25, tzinfo=_IST).timestamp())
    # Closed bar starting 17:25 IST whose high crosses the trigger: the pending
    # must be cancelled before the trigger check, so no entry fires.
    dec = s.update(_c(ts, 60_100, 60_400, 60_050, 60_350))
    assert dec is None
    assert not s.has_pending
    assert s.position_state == PositionState.FLAT


def test_fixed_sl_takes_priority_over_eod_on_same_bar() -> None:
    """If the fixed SL and the 17:25 EOD square-off would both fire on the same
    closed bar, the fixed SL wins (matches the SL-before-EOD precedence order)."""
    s = HeikinAshiStrategy()
    t1 = int(datetime(2026, 7, 6, 17, 20, tzinfo=_IST).timestamp())
    s.update(_c(t1, 100.0, 101.0, 99.0, 100.5))  # establishes _prev_now_mins before square-off

    s._in_long, s._in_short = True, False
    s._sl_level = 91.0   # this next bar's low (86) breaches it

    t2 = int(datetime(2026, 7, 6, 17, 25, tzinfo=_IST).timestamp())  # crosses square-off
    candle = _c(t2, 88.0, 94.0, 86.0, 92.0)
    dec = s.update(candle)

    assert dec is not None
    assert dec.exit_reason == "SL"
    assert dec.long_exit_price == 91.0
