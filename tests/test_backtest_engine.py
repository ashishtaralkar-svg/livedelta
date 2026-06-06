"""Tests for the backtest engine and metrics."""

from __future__ import annotations

from datetime import UTC, datetime

from deltabot.backtest import metrics
from deltabot.backtest.engine import BacktestEngine
from deltabot.config import Settings
from deltabot.models import Candle
from deltabot.pnl import RoundTrip

HOUR = 3600


def _multiday_uptrend(days: int = 5) -> list[Candle]:
    """A strong monotonic climb of hourly candles spanning several UTC days.

    The Pine strategy only enters once the previous-day open/close levels exist
    (i.e. from the third day on), so the data must span multiple days for trades
    to fire — unlike the old always-in-market reversal strategy.
    """
    candles: list[Candle] = []
    base = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp())
    prev_close = 100.0
    for i in range(days * 24):
        close = 100.0 + 5.0 * i
        candles.append(
            Candle(
                start_time=base + i * HOUR, open=prev_close, high=close,
                low=close - 2.0, close=close, volume=1.0,
            )
        )
        prev_close = close
    return candles


def _utc_settings() -> Settings:
    # UTC midnight day boundary + late square-off makes the scenario deterministic.
    return Settings(
        day_tz="UTC", day_start_hour=0, day_start_minute=0,
        square_off_hour=23, square_off_minute=0, ema_length=2, atr_period=2,
    )


def test_backtest_runs_and_produces_trips():
    candles = _multiday_uptrend(days=5)
    engine = BacktestEngine(
        period=2, multiplier=3.0, contracts=1, contract_value=0.001, settings=_utc_settings()
    )
    result = engine.run(candles)
    assert result.candles_processed == len(candles)
    assert len(result.trips) >= 1
    # Every trip should have a valid direction and prices.
    for t in result.trips:
        assert t.direction in (1, -1)
        assert t.entry_price > 0 and t.exit_price > 0


def test_metrics_consistency():
    trips = [
        RoundTrip(0, 60, 1, 0.001, 100.0, 110.0),  # +0.01
        RoundTrip(60, 120, -1, 0.001, 110.0, 115.0),  # short, price up -> loss -0.005
        RoundTrip(120, 180, 1, 0.001, 115.0, 120.0),  # +0.005
    ]
    m = metrics.compute(trips)
    assert m.total_trades == 3
    assert m.winning_trades == 2
    assert m.losing_trades == 1
    assert abs(m.win_rate - 2 / 3) < 1e-9
    assert abs(m.net_profit - (0.01 - 0.005 + 0.005)) < 1e-9
    assert m.profit_factor > 1.0
    # Max drawdown is the largest peak-to-trough drop along the equity curve.
    assert m.max_drawdown >= 0.0


def test_metrics_empty():
    m = metrics.compute([])
    assert m.total_trades == 0
    assert m.win_rate == 0.0
    assert m.net_profit == 0.0
