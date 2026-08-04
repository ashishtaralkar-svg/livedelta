"""ThreeCandleStrategy: pattern match (A touch+shape, B wick-both-sides, C
confirm) -> armed pending -> breakout entry -> fixed SL exit. Uses
use_heikin_ashi=False throughout so tests control raw OHLC directly."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.enums import PositionState
from deltabot.models import Candle
from deltabot.strategy.three_candle import ThreeCandleStrategy

_T0 = 1_700_000_000
_IST = ZoneInfo("Asia/Kolkata")


def _ist_ts(y: int, m: int, d: int, h: int, mi: int) -> int:
    return int(datetime(y, m, d, h, mi, tzinfo=_IST).timestamp())


def _c(i: int, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(start_time=_T0 + i * 300, open=o, high=h, low=l, close=c, volume=1.0)


def _warm(strat: ThreeCandleStrategy, n: int = 20, hi: float = 100.0, lo: float = 90.0) -> int:
    """Push n flat warmup bars (dc_upper=hi, dc_lower=lo after this)."""
    i = 0
    for i in range(n):
        strat.update(_c(i, 95.0, hi, lo, 95.0))
    return i + 1


def test_sell_pattern_arms_pending_with_correct_trigger_and_sl() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)   # dc_upper=100, dc_lower=90

    # A: open==low, high touches/exceeds dc_upper(100).
    dec = s.update(_c(i, 95, 101, 95, 99)); i += 1
    assert not dec.has_entry and not s.has_pending

    # B: wick both sides, high<=A.high(101), low<A.low(95).
    dec = s.update(_c(i, 97, 100, 93, 96)); i += 1
    assert not dec.has_entry and not s.has_pending

    # C: open==high, low<B.low(93).
    dec = s.update(_c(i, 98, 98, 92, 94)); i += 1
    assert s._pending_short is True
    assert s._pending_trigger == 92   # min(95,93,92)
    assert s._pending_sl == 101       # max(101,100,98)
    assert not dec.has_entry


def test_sell_trigger_breaks_and_sl_exits() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1        # A
    s.update(_c(i, 97, 100, 93, 96)); i += 1        # B
    s.update(_c(i, 98, 98, 92, 94)); i += 1         # C -> armed

    dec = s.update(_c(i, 95, 96, 90, 91)); i += 1   # breaks below trigger(92)
    assert dec.sell_signal is True
    assert dec.entry_price == 92
    assert s.position_state == PositionState.SHORT
    assert s.sl_level == 101

    dec = s.update(_c(i, 100, 102, 99, 101)); i += 1   # rallies back to SL(101)
    assert dec.short_exit is True
    assert dec.exit_reason == "SL"
    assert dec.short_exit_price == 101
    assert s.position_state == PositionState.FLAT


def test_sell_setup_invalidated_when_sl_side_hit_before_trigger() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1
    s.update(_c(i, 97, 100, 93, 96)); i += 1
    s.update(_c(i, 98, 98, 92, 94)); i += 1   # armed: trigger=92, sl=101

    dec = s.update(_c(i, 100, 101, 99, 100)); i += 1   # touches SL(101) first
    assert not dec.has_entry
    assert s.has_pending is False   # discarded untraded


def test_candle_b_shape_mismatch_resets_the_hunt() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1   # A ok
    # B candidate with NO lower wick (open==low) -> fails "wick both sides".
    s.update(_c(i, 93, 99, 93, 96)); i += 1
    assert s._sell_state == "idle"
    assert s._sell_a is None

    # Even though this next bar looks exactly like a valid "C" shape, the
    # pattern was reset -- nothing should arm.
    dec = s.update(_c(i, 98, 98, 85, 90)); i += 1
    assert s.has_pending is False
    assert not dec.has_entry


def test_candle_b_only_needs_wick_both_sides_no_high_low_constraint() -> None:
    """B making a NEW HIGH above A, and NOT undercutting A's low, still
    qualifies as long as it has a wick on both sides (the old high<=A.high /
    low<A.low constraints were dropped)."""
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1        # A: high=101, low=95

    # B: new high (105 > A.high 101), low (96) does NOT undercut A.low (95) --
    # would have FAILED the old rule -- but has a wick on both sides.
    s.update(_c(i, 99, 105, 96, 100)); i += 1
    assert s._sell_state == "got_ab"

    dec = s.update(_c(i, 98, 98, 92, 94)); i += 1   # C: open==high, low<B.low(96)
    assert s._pending_short is True
    assert s._pending_trigger == 92    # min(95, 96, 92)
    assert s._pending_sl == 105        # max(101, 105, 98)


def test_candle_b_that_is_itself_a_fresh_a_reanchors_instead_of_resetting() -> None:
    """A candle that fails the B shape check (wick both sides) because it is
    ITSELF a valid open==low touch candle must re-anchor as the new candle 1,
    not be discarded while the hunt resets to idle."""
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1        # bar1 = A (open==low, touches 100)

    # bar2: ALSO open==low + touches dc_upper -- fails B (no lower wick since
    # open==low), but should re-anchor as a FRESH A with bar2's own high/low.
    dec = s.update(_c(i, 93, 102, 93, 98)); i += 1
    assert s._sell_state == "got_a"
    assert s._sell_a == {"high": 102, "low": 93}    # re-anchored to bar2, not bar1
    assert not dec.has_entry

    # bar3: valid B relative to the NEW anchor (bar2): wick both sides,
    # doesn't need to relate to bar1 at all (B has no high/low constraint).
    s.update(_c(i, 97, 100, 90, 96)); i += 1        # low=90 < bar2's low(93)

    # bar4: C -- open==high, low < B.low(90).
    dec = s.update(_c(i, 98, 98, 85, 92)); i += 1
    assert s._pending_short is True
    # Pattern uses bar2/bar3/bar4, NOT bar1 -- trigger/sl reflect the re-anchor.
    assert s._pending_trigger == 85    # min(93, 90, 85)
    assert s._pending_sl == 102        # max(102, 100, 98)


def test_candle_c_must_undercut_a_low_not_just_b_low() -> None:
    """C below B is not enough on its own -- since B has no relation
    constraint to A, B can sit ABOVE A's low. C must break below A's low too,
    or the pattern must NOT arm (the whole sequence has to trend lower than
    where it started, not just lower than the middle bar)."""
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1        # A: low=95, high=101
    s.update(_c(i, 98, 102, 96, 99)); i += 1        # B: low=96 (does NOT undercut A)

    # C: low=95.5 -- below B.low(96) but NOT below A.low(95).
    dec = s.update(_c(i, 97, 97, 95.5, 96)); i += 1
    assert s._pending_short is False   # must NOT arm (would have under the old C<B-only rule)
    assert not dec.has_entry


def test_candle_c_shape_mismatch_resets_the_hunt() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1   # A
    s.update(_c(i, 97, 100, 93, 96)); i += 1   # B ok
    # C candidate that is NOT open==high.
    s.update(_c(i, 95, 98, 90, 94)); i += 1
    assert s._sell_state == "idle"
    assert s.has_pending is False


def test_buy_pattern_mirrors_sell() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)   # dc_upper=100, dc_lower=90

    # A: open==high, low touches/breaches dc_lower(90).
    s.update(_c(i, 96, 96, 89, 92)); i += 1
    # B: wick both sides, low>=A.low(89), high>A.high(96).
    s.update(_c(i, 92, 98, 90, 95)); i += 1
    # C: open==low, high>B.high(98).
    dec = s.update(_c(i, 96, 100, 96, 98)); i += 1
    assert s._pending_long is True
    assert s._pending_trigger == 100   # max(96,98,100)
    assert s._pending_sl == 89         # min(89,90,96)

    dec = s.update(_c(i, 99, 101, 98, 100)); i += 1   # breaks above trigger(100)
    assert dec.buy_signal is True
    assert dec.entry_price == 100
    assert s.position_state == PositionState.LONG
    assert s.sl_level == 89


def test_ready_reflects_donchian_warmup() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    assert s.ready is False
    _warm(s, n=19)
    assert s.ready is False
    s.update(_c(19, 95, 100, 90, 95))
    assert s.ready is True


def test_target_rr_1to2_fires_before_and_takes_priority_over_sl_tie() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False, target_rr=2.0)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1        # A
    s.update(_c(i, 97, 100, 93, 96)); i += 1        # B
    s.update(_c(i, 98, 98, 92, 94)); i += 1         # C -> armed trigger=92 sl=101

    dec = s.update(_c(i, 95, 96, 90, 91)); i += 1   # breaks trigger(92) -> SHORT
    assert dec.sell_signal is True
    # risk = |92-101| = 9; short target = 92 - 9*2 = 74
    assert s._target_level == 74

    # Real price dips to the target before ever threatening the SL(101).
    dec = s.update(_c(i, 80, 82, 73, 75)); i += 1
    assert dec.short_exit is True
    assert dec.exit_reason == "TARGET"
    assert dec.short_exit_price == 74
    assert s.position_state == PositionState.FLAT


def test_target_rr_off_by_default_only_sl_exits() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)   # target_rr=0
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1
    s.update(_c(i, 97, 100, 93, 96)); i += 1
    s.update(_c(i, 98, 98, 92, 94)); i += 1
    s.update(_c(i, 95, 96, 90, 91)); i += 1
    assert s._target_level is None


def test_daily_filter_blocks_sell_when_no_prior_open_high_day() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False, daily_filter_days=3)
    # Day 1 & 2: NOT open==high (high exceeds open both days) -> filter unsatisfied.
    s.update(Candle(start_time=_ist_ts(2026, 1, 1, 10, 0), open=100, high=105, low=98, close=102, volume=1.0))
    s.update(Candle(start_time=_ist_ts(2026, 1, 2, 10, 0), open=100, high=106, low=97, close=101, volume=1.0))

    day3 = _ist_ts(2026, 1, 3, 0, 0)
    for i in range(20):
        s.update(Candle(start_time=day3 + i * 300, open=95, high=100, low=90, close=95, volume=1.0))
    t = day3 + 20 * 300
    s.update(Candle(start_time=t, open=95, high=101, low=95, close=99, volume=1.0)); t += 300   # A
    s.update(Candle(start_time=t, open=97, high=100, low=93, close=96, volume=1.0)); t += 300   # B
    s.update(Candle(start_time=t, open=98, high=98, low=92, close=94, volume=1.0)); t += 300    # C -> armed
    assert s._pending_short is True

    dec = s.update(Candle(start_time=t, open=95, high=96, low=90, close=91, volume=1.0))   # breaks trigger(92)
    assert dec.sell_signal is False          # blocked: no prior open==high day
    assert s.has_pending is False            # consumed regardless


def test_daily_filter_allows_sell_when_a_prior_day_was_open_high() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False, daily_filter_days=3)
    # Day 1: open==high (bearish daily shape) -> satisfies the filter.
    s.update(Candle(start_time=_ist_ts(2026, 1, 1, 10, 0), open=100, high=100, low=95, close=97, volume=1.0))
    s.update(Candle(start_time=_ist_ts(2026, 1, 2, 10, 0), open=100, high=106, low=97, close=101, volume=1.0))

    day3 = _ist_ts(2026, 1, 3, 0, 0)
    for i in range(20):
        s.update(Candle(start_time=day3 + i * 300, open=95, high=100, low=90, close=95, volume=1.0))
    t = day3 + 20 * 300
    s.update(Candle(start_time=t, open=95, high=101, low=95, close=99, volume=1.0)); t += 300
    s.update(Candle(start_time=t, open=97, high=100, low=93, close=96, volume=1.0)); t += 300
    s.update(Candle(start_time=t, open=98, high=98, low=92, close=94, volume=1.0)); t += 300

    dec = s.update(Candle(start_time=t, open=95, high=96, low=90, close=91, volume=1.0))
    assert dec.sell_signal is True
    assert dec.entry_price == 92


def test_daily_filter_blocks_buy_when_no_prior_open_low_day() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False, daily_filter_days=3)
    s.update(Candle(start_time=_ist_ts(2026, 1, 1, 10, 0), open=100, high=105, low=98, close=102, volume=1.0))
    s.update(Candle(start_time=_ist_ts(2026, 1, 2, 10, 0), open=100, high=106, low=97, close=101, volume=1.0))

    day3 = _ist_ts(2026, 1, 3, 0, 0)
    for i in range(20):
        s.update(Candle(start_time=day3 + i * 300, open=95, high=100, low=90, close=95, volume=1.0))
    t = day3 + 20 * 300
    s.update(Candle(start_time=t, open=96, high=96, low=89, close=92, volume=1.0)); t += 300   # A
    s.update(Candle(start_time=t, open=92, high=98, low=90, close=95, volume=1.0)); t += 300   # B
    s.update(Candle(start_time=t, open=96, high=100, low=96, close=98, volume=1.0)); t += 300  # C -> armed
    assert s._pending_long is True

    dec = s.update(Candle(start_time=t, open=99, high=101, low=98, close=100, volume=1.0))  # breaks trigger(100)
    assert dec.buy_signal is False
    assert s.has_pending is False


def test_intracandle_pending_confirms_on_real_price_break() -> None:
    """apply_intracandle_pending fires the instant the FORMING candle's real
    price breaks the trigger, without waiting for the bar to close."""
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1        # A
    s.update(_c(i, 97, 100, 93, 96)); i += 1        # B
    s.update(_c(i, 98, 98, 92, 94)); i += 1         # C -> armed: trigger=92, sl=101
    assert s.has_pending is True

    forming = _c(i, 95, 96, 90, 91)   # a forming (not-yet-closed) bar breaking trigger(92)
    confirmed, invalidated, entry_price = s.apply_intracandle_pending(forming)
    assert confirmed is True and invalidated is False
    assert entry_price == 92
    assert s.position_state == PositionState.SHORT
    assert s.sl_level == 101
    assert s.has_pending is False


def test_intracandle_pending_invalidates_on_sl_side_first() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1
    s.update(_c(i, 97, 100, 93, 96)); i += 1
    s.update(_c(i, 98, 98, 92, 94)); i += 1          # armed: trigger=92, sl=101

    forming = _c(i, 100, 101, 99, 100)   # touches SL(101) before the trigger
    confirmed, invalidated, entry_price = s.apply_intracandle_pending(forming)
    assert confirmed is False and invalidated is True
    assert s.has_pending is False
    assert s.position_state == PositionState.FLAT


def test_check_intracandle_sl_detects_real_price_touch() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1
    s.update(_c(i, 97, 100, 93, 96)); i += 1
    s.update(_c(i, 98, 98, 92, 94)); i += 1
    s.apply_intracandle_pending(_c(i, 95, 96, 90, 91))   # enters SHORT, sl=101
    assert s.position_state == PositionState.SHORT

    long_sl, short_sl, sl = s.check_intracandle_sl(101.0)
    assert short_sl is True and long_sl is False and sl == 101
    long_sl, short_sl, sl = s.check_intracandle_sl(99.0)
    assert short_sl is False and long_sl is False


def test_debug_state_reports_current_hunt_and_position() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1
    d = s.debug_state()
    assert d["sell_state"] == "got_a"
    assert d["pos"] == "FLAT"


def test_force_flat_clears_position_and_hunt_state() -> None:
    s = ThreeCandleStrategy(dc_period=20, use_heikin_ashi=False)
    i = _warm(s)
    s.update(_c(i, 95, 101, 95, 99)); i += 1
    s.update(_c(i, 97, 100, 93, 96)); i += 1
    s.update(_c(i, 98, 98, 92, 94)); i += 1
    assert s.has_pending is True
    s.force_flat()
    assert s.has_pending is False
    assert s.position_state == PositionState.FLAT
    assert s._sell_state == "idle" and s._buy_state == "idle"
