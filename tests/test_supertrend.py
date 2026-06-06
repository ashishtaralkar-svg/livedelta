"""Validate the incremental Supertrend against an independent vectorized reference.

The incremental ``SupertrendCalculator`` is the code path used live and in
backtests. Here we re-implement Supertrend the "obvious" vectorized way (looping
over a full series with carried bands) and assert the incremental version
produces identical direction and band values bar-for-bar. This is the golden test.
"""

from __future__ import annotations

from conftest import synthetic_candles

from deltabot.models import Candle
from deltabot.strategy.supertrend import SupertrendCalculator


def reference_supertrend(
    candles: list[Candle], period: int = 10, multiplier: float = 3.0
) -> list[tuple[int, float] | None]:
    """Reference implementation returning (direction, value) per bar, or None pre-warmup."""
    n = len(candles)
    out: list[tuple[int, float] | None] = [None] * n

    tr = [0.0] * n
    for i, c in enumerate(candles):
        if i == 0:
            tr[i] = c.high - c.low
        else:
            pc = candles[i - 1].close
            tr[i] = max(c.high - c.low, abs(c.high - pc), abs(c.low - pc))

    atr: list[float | None] = [None] * n
    for i in range(n):
        if i + 1 == period:
            atr[i] = sum(tr[: i + 1]) / period
        elif i + 1 > period:
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period  # type: ignore[operator]

    final_upper: list[float | None] = [None] * n
    final_lower: list[float | None] = [None] * n
    direction: list[int | None] = [None] * n

    started = False
    for i, c in enumerate(candles):
        if atr[i] is None:
            continue
        hl2 = (c.high + c.low) / 2.0
        bu = hl2 + multiplier * atr[i]  # type: ignore[operator]
        bl = hl2 - multiplier * atr[i]  # type: ignore[operator]
        if not started:
            final_upper[i] = bu
            final_lower[i] = bl
            direction[i] = 1 if c.close > hl2 else -1
            started = True
        else:
            pc = candles[i - 1].close
            fu_prev = final_upper[i - 1]
            fl_prev = final_lower[i - 1]
            final_upper[i] = bu if (bu < fu_prev or pc > fu_prev) else fu_prev  # type: ignore[operator]
            final_lower[i] = bl if (bl > fl_prev or pc < fl_prev) else fl_prev  # type: ignore[operator]
            prev_dir = direction[i - 1]
            if prev_dir == 1:
                direction[i] = -1 if c.close < final_lower[i] else 1  # type: ignore[operator]
            else:
                direction[i] = 1 if c.close > final_upper[i] else -1  # type: ignore[operator]
        d = direction[i]
        out[i] = (d, final_lower[i] if d == 1 else final_upper[i])  # type: ignore[index]
    return out


def test_incremental_matches_reference():
    candles = synthetic_candles(300)
    expected = reference_supertrend(candles, period=10, multiplier=3.0)

    calc = SupertrendCalculator(period=10, multiplier=3.0)
    for i, candle in enumerate(candles):
        sig = calc.update(candle)
        exp = expected[i]
        if exp is None:
            assert sig is None or not calc.ready
            continue
        assert sig is not None, f"bar {i}: expected a signal"
        assert sig.direction.value == exp[0], f"bar {i}: direction mismatch"
        assert abs(sig.value - exp[1]) < 1e-6, f"bar {i}: band value mismatch"


def test_flip_detection():
    """``flipped`` must be True exactly when direction changes between bars."""
    candles = synthetic_candles(300)
    calc = SupertrendCalculator(period=10, multiplier=3.0)
    prev_dir: int | None = None
    flips = 0
    for candle in candles:
        sig = calc.update(candle)
        if sig is None:
            continue
        if prev_dir is not None:
            changed = sig.direction.value != prev_dir
            assert sig.flipped == changed
            if changed:
                flips += 1
        prev_dir = sig.direction.value
    # The synthetic series is built to oscillate, so we expect multiple flips.
    assert flips >= 3, "expected the oscillating series to flip several times"


def test_seed_matches_streaming():
    """Seeding N candles then streaming must equal streaming all N+M candles."""
    candles = synthetic_candles(250)
    split = 200

    streamed = SupertrendCalculator(period=10, multiplier=3.0)
    last_stream = None
    for c in candles:
        s = streamed.update(c)
        if s is not None:
            last_stream = s

    seeded = SupertrendCalculator(period=10, multiplier=3.0)
    seeded.seed(candles[:split])
    last_seed = None
    for c in candles[split:]:
        s = seeded.update(c)
        if s is not None:
            last_seed = s

    assert last_stream is not None and last_seed is not None
    assert last_stream.direction == last_seed.direction
    assert abs(last_stream.value - last_seed.value) < 1e-6


def test_only_closed_candles_no_repaint():
    """Feeding the same candle object repeatedly must be deterministic per call.

    (The aggregator guarantees one call per closed bar; here we assert the
    calculator's output for a given state+candle is stable.)
    """
    candles = synthetic_candles(60)
    calc = SupertrendCalculator(period=10, multiplier=3.0)
    for c in candles[:-1]:
        calc.update(c)
    last = candles[-1]
    a = calc.update(last)
    # Direction is now committed; a fresh calc seeded identically agrees.
    other = SupertrendCalculator(period=10, multiplier=3.0)
    other.seed(candles)
    assert a is not None
    assert a.direction.value == other.direction()
