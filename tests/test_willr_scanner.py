"""WilliamsRScanner: per-CONTRACT oversold-bounce breakout, computed on the
option's OWN premium candles (never BTC). Uses small periods (5/5/5) instead
of the live default (50/50/14) purely so a handful of candles can drive the
indicators through a real state -- the arithmetic is identical either way.

The warmup sequence below (a plateau, a deep trough, a partial recovery,
then a smaller re-dip) is numerically chosen, not arbitrary: it's the
shortest path found that produces close < ma < ema (EMA reacts faster than
SMA in both directions, so simple monotonic moves can't produce this
ordering -- it takes a genuine second dip after a recovery) at the SAME
candle Williams %R re-crosses below -80, matching the strategy's real arm
condition."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.models import Candle
from deltabot.strategy.willr_scanner import WilliamsRScanner

_IST = ZoneInfo("Asia/Kolkata")

# (close, high, low) -- high/low = close +/- 2. Index 11 is the arm candle:
# close=98, ema=105.63, ma=105.6 (close < ma < ema), wr=-81.48 (crossed < -80).
_WARMUP = [120, 120, 120, 120, 120, 90, 88, 95, 105, 112, 118, 98]


def _ts(hour: int, minute: int, day: int = 8) -> int:
    return int(datetime(2026, 7, day, hour, minute, tzinfo=_IST).timestamp())


def _c(ts: int, close: float, high: float | None = None, low: float | None = None) -> Candle:
    return Candle(start_time=ts, open=close, high=high if high is not None else close + 2,
                  low=low if low is not None else close - 2, close=close, volume=1.0)


def _scanner(**kw) -> WilliamsRScanner:
    base = dict(ema_len=5, ma_len=5, wr_period=5, breakout_wait_bars=3,
                entry_start_hour=14, entry_end_hour=17)
    base.update(kw)
    return WilliamsRScanner(**base)


def _feed_warmup(scanner: WilliamsRScanner, symbol: str, start_hour: int = 10) -> int:
    """Feeds _WARMUP (5-minute bars from start_hour:00) and returns the arm
    candle's own high (100 -- the breakout level to watch for next)."""
    ts = _ts(start_hour, 0)
    last_high = None
    for close in _WARMUP:
        c = _c(ts, close)
        scanner.update_contract(symbol, c)
        last_high = c.high
        ts += 300
    return last_high


def test_warmup_sequence_arms_but_does_not_enter_immediately() -> None:
    s = _scanner()
    _feed_warmup(s, "C-BTC-1")
    assert s.is_armed("C-BTC-1")
    assert not s.has_entered("C-BTC-1")


def test_breakout_above_armed_high_enters_within_window() -> None:
    s = _scanner()
    armed_high = _feed_warmup(s, "C-BTC-1")
    ts = _ts(15, 0)   # inside the 14-17 entry window
    entered = s.update_contract("C-BTC-1", _c(ts, close=99, high=armed_high + 1, low=97))
    assert entered is True
    assert s.has_entered("C-BTC-1")


def test_no_entry_outside_the_entry_window_even_on_a_real_breakout() -> None:
    s = _scanner()
    armed_high = _feed_warmup(s, "C-BTC-1", start_hour=10)
    ts = _ts(12, 0)   # before the 14:00 window opens
    entered = s.update_contract("C-BTC-1", _c(ts, close=99, high=armed_high + 1, low=97))
    assert entered is False
    assert not s.has_entered("C-BTC-1")


def test_armed_state_expires_after_breakout_wait_bars_without_a_breakout() -> None:
    s = _scanner(breakout_wait_bars=2)
    armed_high = _feed_warmup(s, "C-BTC-1")
    assert s.is_armed("C-BTC-1")
    ts = _ts(15, 0)
    s.update_contract("C-BTC-1", _c(ts, close=98, high=armed_high - 5, low=95))         # bar 1: no breakout
    assert s.is_armed("C-BTC-1")
    s.update_contract("C-BTC-1", _c(ts + 300, close=98, high=armed_high - 5, low=95))   # bar 2: still none -> expires
    assert not s.is_armed("C-BTC-1")
    entered = s.update_contract("C-BTC-1", _c(ts + 600, close=99, high=armed_high + 5, low=97))
    assert entered is False   # too late, already un-armed


def test_a_contract_never_re_arms_after_it_has_entered() -> None:
    s = _scanner()
    armed_high = _feed_warmup(s, "C-BTC-1")
    ts = _ts(15, 0)
    assert s.update_contract("C-BTC-1", _c(ts, close=99, high=armed_high + 1, low=97)) is True
    # Feeding the exact same arming warmup sequence again must not re-enter.
    _feed_warmup(s, "C-BTC-1", start_hour=10)
    assert s.update_contract("C-BTC-1", _c(ts + 3000, close=99, high=armed_high + 1, low=97)) is False


def test_contracts_are_tracked_independently() -> None:
    s = _scanner()
    _feed_warmup(s, "C-BTC-1")
    assert s.is_armed("C-BTC-1")
    assert not s.is_armed("P-BTC-1")   # a second, never-fed symbol has its own untouched state


def test_reset_all_clears_every_tracked_contract() -> None:
    s = _scanner()
    _feed_warmup(s, "C-BTC-1")
    assert s.is_armed("C-BTC-1")
    s.reset_all()
    assert not s.is_armed("C-BTC-1")
