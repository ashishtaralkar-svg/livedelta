"""Tests for the small incremental indicators."""

from __future__ import annotations

from deltabot.strategy.indicators import EmaCalculator


def _pine_ema(values: list[float], length: int) -> list[float]:
    """Reference EMA matching Pine's ``ta.ema`` (seed = first source value)."""
    alpha = 2.0 / (length + 1.0)
    out: list[float] = []
    ema: float | None = None
    for v in values:
        ema = v if ema is None else alpha * v + (1.0 - alpha) * ema
        out.append(ema)
    return out


def test_ema_seeds_with_first_value():
    ema = EmaCalculator(50)
    assert ema.value is None
    first = ema.update(123.0)
    assert first == 123.0
    assert ema.value == 123.0


def test_ema_matches_pine_reference():
    values = [float(v) for v in [10, 11, 13, 12, 15, 18, 17, 20, 19, 25]]
    ref = _pine_ema(values, length=5)
    ema = EmaCalculator(5)
    got = [ema.update(v) for v in values]
    for g, r in zip(got, ref, strict=True):
        assert abs(g - r) < 1e-9


def test_ema_rejects_bad_length():
    import pytest

    with pytest.raises(ValueError):
        EmaCalculator(0)
