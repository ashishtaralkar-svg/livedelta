"""Tests for the Pine-equivalent strategy port."""

from __future__ import annotations

from datetime import UTC, datetime

from deltabot.enums import PositionState
from deltabot.models import Candle
from deltabot.strategy.pine_strategy import PineStrategy

HOUR = 3600


def _ts(day: int, hour: int) -> int:
    return int(datetime(2023, 1, day, hour, tzinfo=UTC).timestamp())


def _candle(ts: int, o: float, h: float, low: float, c: float) -> Candle:
    return Candle(start_time=ts, open=o, high=h, low=low, close=c, volume=1.0)


def _strategy(**overrides) -> PineStrategy:
    # UTC midnight day boundary keeps the previous-day arithmetic easy to reason about.
    params = dict(
        atr_period=2,
        st_multiplier=3.0,
        ema_length=2,
        day_tz="UTC",
        day_start_hour=0,
        day_start_minute=0,
        square_off_hour=12,
        square_off_minute=0,
    )
    params.update(overrides)
    return PineStrategy(**params)


def _uptrend(strategy: PineStrategy, *, hours: int = 60) -> list:
    """Feed a strong monotonic uptrend of hourly candles; return the decisions."""
    decisions = []
    price = 100.0
    base = _ts(1, 0)
    prev_close = price
    for i in range(hours):
        close = 100.0 + 5.0 * i
        candle = _candle(base + i * HOUR, o=prev_close, h=close, low=close - 2.0, c=close)
        decisions.append(strategy.update(candle))
        prev_close = close
    return decisions


# --------------------------------------------------------------------------- #
# Previous-day open/close
# --------------------------------------------------------------------------- #
def test_prev_day_levels_track_custom_day_boundary():
    s = _strategy(square_off_hour=23)
    # Day 1: open 100, last close 130.
    for hour, close in zip(range(4), [100.0, 110.0, 120.0, 130.0], strict=True):
        s.update(_candle(_ts(1, hour * 6), close - 1, close + 1, close - 1, close))
    # No previous day yet -> levels still unset.
    assert s.pd_open is None

    # Day 2 first bar opens at 200; previous-day (day 1) values must NOT appear yet
    # because cur_open was na before the first boundary (matches Pine's na seed).
    s.update(_candle(_ts(2, 0), 200.0, 201.0, 199.0, 200.0))
    assert s.pd_open is None
    assert s.pd_close == 130.0  # day-1 last close froze as prev-day close

    for hour, close in zip(range(1, 4), [210.0, 220.0, 230.0], strict=True):
        s.update(_candle(_ts(2, hour * 6), close - 1, close + 1, close - 1, close))

    # Day 3 first bar: now the previous day (day 2) is fully formed.
    s.update(_candle(_ts(3, 0), 300.0, 301.0, 299.0, 300.0))
    assert s.pd_open == 200.0  # day-2 first open
    assert s.pd_close == 230.0  # day-2 last close


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def test_uptrend_fires_a_single_buy_and_goes_long():
    s = _strategy(square_off_hour=23)
    decisions = [d for d in _uptrend(s) if d is not None]
    buys = [d for d in decisions if d.buy_signal]
    assert buys, "expected at least one BUY signal in a strong uptrend"
    # Fresh-crossing + one-trade-at-a-time: the condition stays true but only the
    # first crossing opens a position.
    assert len(buys) == 1
    assert buys[0].target_state == PositionState.LONG
    assert s.position_state == PositionState.LONG


def test_no_signal_before_levels_ready():
    s = _strategy()
    # A couple of early bars cannot produce a decision (no prev-day levels yet).
    assert s.update(_candle(_ts(1, 0), 100, 101, 99, 100)) is None
    assert s.update(_candle(_ts(1, 1), 100, 101, 99, 100)) is None


# --------------------------------------------------------------------------- #
# Exits: stop-loss and square-off
# --------------------------------------------------------------------------- #
def _seed_levels(s: PineStrategy) -> int:
    """Warm the strategy so levels are ready and it is flat; return next hour ts."""
    candles = []
    for day in (1, 2):
        for hour in range(0, 24, 6):
            c = 100.0 + day * 10 + hour
            candles.append(_candle(_ts(day, hour), c - 1, c + 1, c - 1, c))
    # A few day-3 bars (before square-off at 12:00) to establish prev_now_mins.
    for hour in (0, 6):
        c = 130.0 + hour
        candles.append(_candle(_ts(3, hour), c - 1, c + 1, c - 1, c))
    s.seed(candles)
    assert s.ready
    assert s.position_state == PositionState.FLAT
    return _ts(3, 9)


def test_stop_loss_exit_fills_at_stop_level():
    s = _strategy()
    _seed_levels(s)
    # Inject a long with a known stop level, then a bar whose low pierces it.
    s.sync_position(PositionState.LONG, entry_price=140.0, stop_level=135.0)
    dec = s.update(_candle(_ts(3, 9), 138, 139, 130, 134))  # low 130 <= 135
    assert dec is not None
    assert dec.long_exit and dec.long_exit_sl
    assert dec.long_exit_price == 135.0  # filled at the stop, not the close
    assert dec.target_state == PositionState.FLAT
    assert s.position_state == PositionState.FLAT


def test_square_off_exit_at_cutoff_time():
    s = _strategy()
    _seed_levels(s)  # last seeded bar is day-3 06:00 (before 12:00 cutoff)
    s.sync_position(PositionState.LONG, entry_price=140.0, stop_level=100.0)
    # 12:00 crosses the square-off threshold; stop not hit -> exit at close.
    dec = s.update(_candle(_ts(3, 12), 145, 146, 142, 144))
    assert dec is not None
    assert dec.long_exit and dec.long_sq_off and not dec.long_exit_sl
    assert dec.long_exit_price == 144.0  # bar close
    assert dec.target_state == PositionState.FLAT


def test_no_entry_on_square_off_bar():
    s = _strategy()
    _seed_levels(s)
    # Even if a buy condition were true, no fresh entry may fire on the square-off bar.
    dec = s.update(_candle(_ts(3, 12), 200, 200, 198, 200))
    assert dec is not None
    assert not dec.buy_signal and not dec.sell_signal
