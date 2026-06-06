"""Repaint-safe, incremental Supertrend(ATR period, multiplier).

This module is the single source of truth for the indicator and is shared
verbatim by the live engine and the backtester so their results cannot diverge.

It operates strictly on *closed* candles. Each call to :meth:`update` consumes
one closed candle and returns a :class:`Signal` describing the current direction
and whether the trend flipped on this bar.

ATR uses Wilder smoothing (RMA), seeded with a simple average of True Range over
the first ``period`` bars — matching the conventional TradingView Supertrend.
"""

from __future__ import annotations

from ..enums import SignalDir
from ..models import Candle, Signal


class SupertrendCalculator:
    """Incremental Supertrend computed one closed candle at a time."""

    def __init__(self, period: int = 10, multiplier: float = 3.0) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.multiplier = multiplier
        self.reset()

    def reset(self) -> None:
        """Clear all state (used before re-seeding)."""
        self._count = 0
        self._tr_sum = 0.0  # accumulates TR during the warmup window
        self._atr: float | None = None
        self._prev_close: float | None = None
        self._final_upper: float | None = None
        self._final_lower: float | None = None
        self._direction: int = SignalDir.LONG.value  # +1 long, -1 short
        self._ready = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        """True once enough candles have been consumed to emit valid signals."""
        return self._ready

    def direction(self) -> int:
        """Current Supertrend direction: +1 (long) or -1 (short)."""
        return self._direction

    def seed(self, candles: list[Candle]) -> Signal | None:
        """Warm up from a list of historical *closed* candles in time order.

        Returns the Signal for the final seeded candle (``flipped`` is always
        False here — seeding establishes state, it does not generate trades).
        """
        self.reset()
        last: Signal | None = None
        for candle in candles:
            last = self._step(candle, emit_flip=False)
        return last

    def update(self, candle: Candle) -> Signal | None:
        """Consume one closed candle and return its Signal (or None pre-warmup)."""
        return self._step(candle, emit_flip=True)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _true_range(self, candle: Candle) -> float:
        if self._prev_close is None:
            return candle.high - candle.low
        return max(
            candle.high - candle.low,
            abs(candle.high - self._prev_close),
            abs(candle.low - self._prev_close),
        )

    def _step(self, candle: Candle, emit_flip: bool) -> Signal | None:
        tr = self._true_range(candle)
        self._count += 1

        # --- ATR (Wilder) ---
        if self._atr is None:
            # Still building the seed average of TR.
            self._tr_sum += tr
            if self._count >= self.period:
                self._atr = self._tr_sum / self.period
        else:
            self._atr = (self._atr * (self.period - 1) + tr) / self.period

        if self._atr is None:
            # Not enough bars yet to have an ATR — establish prev_close and bail.
            self._prev_close = candle.close
            return None

        hl2 = (candle.high + candle.low) / 2.0
        basic_upper = hl2 + self.multiplier * self._atr
        basic_lower = hl2 - self.multiplier * self._atr

        prev_close = self._prev_close if self._prev_close is not None else candle.close

        # --- Final bands (carried forward) ---
        if self._final_upper is None or self._final_lower is None:
            # First bar with a valid ATR initialises the bands and direction.
            final_upper = basic_upper
            final_lower = basic_lower
            self._direction = SignalDir.LONG.value if candle.close > hl2 else SignalDir.SHORT.value
            self._final_upper = final_upper
            self._final_lower = final_lower
            self._prev_close = candle.close
            self._ready = True
            return Signal(
                direction=SignalDir(self._direction),
                flipped=False,
                value=final_lower if self._direction == 1 else final_upper,
                candle=candle,
            )

        final_upper = (
            basic_upper
            if (basic_upper < self._final_upper or prev_close > self._final_upper)
            else self._final_upper
        )
        final_lower = (
            basic_lower
            if (basic_lower > self._final_lower or prev_close < self._final_lower)
            else self._final_lower
        )

        prev_direction = self._direction

        # --- Direction transition based on close vs the carried band ---
        if prev_direction == 1:
            # We were long (price riding the lower band). Flip short if we break it.
            direction = SignalDir.SHORT.value if candle.close < final_lower else SignalDir.LONG.value
        else:
            # We were short (price riding the upper band). Flip long if we break it.
            direction = SignalDir.LONG.value if candle.close > final_upper else SignalDir.SHORT.value

        flipped = direction != prev_direction

        self._direction = direction
        self._final_upper = final_upper
        self._final_lower = final_lower
        self._prev_close = candle.close
        self._ready = True

        return Signal(
            direction=SignalDir(direction),
            flipped=flipped and emit_flip,
            value=final_lower if direction == 1 else final_upper,
            candle=candle,
        )
