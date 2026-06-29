"""Intracandle (ASAP) entry/SL for RevBreakStrategy.

Verifies apply_intracandle_pending() enters the moment price crosses the pattern
trigger and invalidates the moment the stop is touched first, mirroring the rules
the live engine relies on for sub-5m entries.
"""

from __future__ import annotations

from deltabot.enums import PositionState
from deltabot.models import Candle
from deltabot.strategy.revbreak import RevBreakStrategy


def _c(start: int, o: float, h: float, l: float, cl: float) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=l, close=cl, volume=1.0)


def _arm_short(strat: RevBreakStrategy) -> None:
    """Force an armed short setup: trigger = pattern low, sl = pattern high."""
    strat._pending_short = True
    strat._pending_long = False
    strat._pending_trigger = 60_000.0   # enter when price breaks BELOW this
    strat._pending_sl = 60_300.0        # invalid if price breaks ABOVE this first


def _arm_long(strat: RevBreakStrategy) -> None:
    strat._pending_long = True
    strat._pending_short = False
    strat._pending_trigger = 60_300.0   # enter when price breaks ABOVE this
    strat._pending_sl = 60_000.0        # invalid if price breaks BELOW this first


def test_short_enters_when_price_breaks_below_low() -> None:
    s = RevBreakStrategy(gate="open", st_entry_filter=False, reentry_block=False)
    _arm_short(s)
    # Forming candle dips to 59,990 (below the 60,000 trigger), high stays under SL.
    confirmed, invalidated, entry = s.apply_intracandle_pending(_c(1, 60_050, 60_100, 59_990, 60_010))
    assert confirmed and not invalidated
    assert entry == 60_000.0
    assert s.position_state == PositionState.SHORT
    assert s.sl_level == 60_300.0
    assert not s.has_pending


def test_short_invalidated_when_high_hits_sl_first() -> None:
    s = RevBreakStrategy(gate="open", st_entry_filter=False, reentry_block=False)
    _arm_short(s)
    # Price rallies to the SL (60,300) before ever reaching the 60,000 trigger.
    confirmed, invalidated, _ = s.apply_intracandle_pending(_c(1, 60_100, 60_320, 60_050, 60_280))
    assert invalidated and not confirmed
    assert s.position_state == PositionState.FLAT
    assert not s.has_pending


def test_long_enters_when_price_breaks_above_high() -> None:
    s = RevBreakStrategy(gate="open", st_entry_filter=False, reentry_block=False)
    _arm_long(s)
    confirmed, invalidated, entry = s.apply_intracandle_pending(_c(1, 60_050, 60_310, 60_020, 60_290))
    assert confirmed and not invalidated
    assert entry == 60_300.0
    assert s.position_state == PositionState.LONG
    assert s.sl_level == 60_000.0


def test_no_action_when_price_between_levels() -> None:
    s = RevBreakStrategy(gate="open", st_entry_filter=False, reentry_block=False)
    _arm_short(s)
    confirmed, invalidated, _ = s.apply_intracandle_pending(_c(1, 60_150, 60_200, 60_050, 60_120))
    assert not confirmed and not invalidated
    assert s.has_pending  # still armed, still watching


def test_check_intracandle_sl_after_short_entry() -> None:
    s = RevBreakStrategy(gate="open", st_entry_filter=False, reentry_block=False)
    _arm_short(s)
    s.apply_intracandle_pending(_c(1, 60_050, 60_100, 59_990, 60_010))  # now SHORT, sl=60_300
    long_sl, short_sl, level = s.check_intracandle_sl(60_320.0)         # price rose to the stop
    assert short_sl and not long_sl
    assert level == 60_300.0
    # below the stop → no exit
    long_sl, short_sl, _ = s.check_intracandle_sl(60_100.0)
    assert not short_sl and not long_sl
