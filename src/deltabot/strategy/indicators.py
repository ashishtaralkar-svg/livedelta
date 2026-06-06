"""Small incremental indicators shared by the live engine and the backtester.

These mirror their TradingView (Pine Script v5) counterparts exactly so that the
Python strategy and the Pine indicator produce identical signals on the same
candles.
"""

from __future__ import annotations


class EmaCalculator:
    """Incremental exponential moving average matching Pine's ``ta.ema``.

    Pine seeds the EMA with the first source value (``na(ema[1]) ? src : ...``),
    then applies the recursive update ``ema = alpha*src + (1-alpha)*ema[1]`` with
    ``alpha = 2 / (length + 1)``.
    """

    def __init__(self, length: int) -> None:
        if length < 1:
            raise ValueError("length must be >= 1")
        self.length = length
        self.alpha = 2.0 / (length + 1.0)
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        """Current EMA, or ``None`` before the first sample has been seen."""
        return self._value

    def update(self, source: float) -> float:
        """Consume one source sample and return the updated EMA."""
        if self._value is None:
            self._value = source
        else:
            self._value = self.alpha * source + (1.0 - self.alpha) * self._value
        return self._value
