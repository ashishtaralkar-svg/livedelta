"""Closed-candle detection from the Delta WebSocket candlestick stream.

The ``candlestick_1m`` channel emits repeated updates for the *forming* candle.
A candle is considered CLOSED only when a message arrives carrying a newer
``candle_start_time`` than the one we are currently tracking: at that moment the
previously tracked candle is final and is emitted exactly once. The forming
candle is never passed downstream, which is what keeps the strategy repaint-free.
"""

from __future__ import annotations

from collections.abc import Callable

from ..logging_setup import get_logger
from ..models import Candle

log = get_logger(__name__)


class CandleAggregator:
    """Detect closed 1-minute candles and invoke a callback once per closed bar."""

    def __init__(self, on_closed: Callable[[Candle], object]) -> None:
        self.on_closed = on_closed
        self._current: Candle | None = None
        self._last_emitted_start: int | None = None

    @property
    def current_start_time(self) -> int | None:
        return self._current.start_time if self._current else None

    def ingest(self, msg: dict) -> None:
        """Process one raw WebSocket candlestick message."""
        if msg.get("type") not in (None, "candlestick_1m") and "candle_start_time" not in msg:
            return
        if "candle_start_time" not in msg:
            return
        candle = Candle.from_ws(msg)
        self._ingest_candle(candle)

    def _ingest_candle(self, candle: Candle) -> None:
        if self._current is None:
            self._current = candle
            return

        if candle.start_time == self._current.start_time:
            # Update to the still-forming candle: keep the latest snapshot.
            self._current = candle
            return

        if candle.start_time > self._current.start_time:
            # Roll-over: the tracked candle is now closed and final.
            closed = self._current
            self._current = candle
            self._emit(closed)
            return

        # Out-of-order / stale message for an older candle: ignore.
        log.warning(
            "Ignoring out-of-order candle",
            extra={"extra": {"got": candle.start_time, "current": self._current.start_time}},
        )

    def _emit(self, candle: Candle) -> None:
        if self._last_emitted_start == candle.start_time:
            return  # dedupe: never emit the same closed bar twice
        self._last_emitted_start = candle.start_time
        self.on_closed(candle)
