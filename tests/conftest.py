"""Shared test fixtures and helpers."""

from __future__ import annotations

import math

from deltabot.models import Candle


def synthetic_candles(n: int = 300, seed: int = 1234) -> list[Candle]:
    """Deterministic pseudo-random OHLC series (no external RNG dependency).

    Uses a simple LCG so the series is identical across runs/platforms without
    importing numpy or relying on Python's random seeding internals.
    """
    candles: list[Candle] = []
    state = seed
    price = 30000.0
    start = 1_700_000_000
    for i in range(n):
        # Linear congruential generator -> uniform in [0, 1).
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        r = state / 0x7FFFFFFF
        # Trending sine wave + noise so the Supertrend flips multiple times.
        drift = 150.0 * math.sin(i / 18.0)
        noise = (r - 0.5) * 120.0
        close = price + drift + noise
        high = max(price, close) + 40.0 * r
        low = min(price, close) - 40.0 * (1.0 - r)
        candles.append(
            Candle(
                start_time=start + i * 60,
                open=price,
                high=high,
                low=low,
                close=close,
                volume=1.0 + r,
            )
        )
        price = close
    return candles
