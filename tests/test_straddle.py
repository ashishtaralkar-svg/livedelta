"""StraddleStrategy: a fixed-TIME daily entry trigger, no BTC-price signal
at all -- fires at most once per calendar day, on the first closed candle
whose local time falls in [entry_hour:minute, +entry_grace_minutes)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.models import Candle
from deltabot.strategy.straddle import StraddleStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ts(hour: int, minute: int, day: int = 8) -> int:
    return int(datetime(2026, 7, day, hour, minute, tzinfo=_IST).timestamp())


def _c(ts: int) -> Candle:
    return Candle(start_time=ts, open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0)


def _strategy(**kw) -> StraddleStrategy:
    base = dict(entry_hour=16, entry_minute=0, entry_grace_minutes=15)
    base.update(kw)
    return StraddleStrategy(**base)


def test_fires_on_the_exact_entry_bar() -> None:
    s = _strategy()
    assert s.update(_c(_ts(15, 55))) is False
    assert s.update(_c(_ts(16, 0))) is True


def test_does_not_fire_twice_the_same_day() -> None:
    s = _strategy()
    assert s.update(_c(_ts(16, 0))) is True
    assert s.update(_c(_ts(16, 5))) is False
    assert s.update(_c(_ts(16, 10))) is False


def test_fires_once_more_the_next_day() -> None:
    s = _strategy()
    assert s.update(_c(_ts(16, 0, day=8))) is True
    assert s.update(_c(_ts(16, 0, day=9))) is True


def test_fires_within_grace_window_if_the_exact_bar_was_missed() -> None:
    s = _strategy(entry_grace_minutes=15)
    assert s.update(_c(_ts(16, 10))) is True   # 10 min late, still inside the 15-min grace window


def test_never_fires_past_the_grace_window() -> None:
    s = _strategy(entry_grace_minutes=15)
    assert s.update(_c(_ts(16, 20))) is False   # 20 min late, past grace
    assert s.update(_c(_ts(17, 0))) is False    # and stays false for the rest of the day


def test_unfire_today_allows_a_retry_within_the_same_grace_window() -> None:
    s = _strategy(entry_grace_minutes=15)
    assert s.update(_c(_ts(16, 0))) is True
    s.unfire_today()   # e.g. the caller's attempt to open both legs failed
    assert s.update(_c(_ts(16, 5))) is True    # retries, still within grace


def test_unfire_today_is_a_noop_once_grace_window_has_passed() -> None:
    s = _strategy(entry_grace_minutes=15)
    assert s.update(_c(_ts(16, 0))) is True
    s.unfire_today()
    assert s.update(_c(_ts(16, 30))) is False   # too late now, grace window already closed
