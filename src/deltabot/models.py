"""Plain data models passed between components.

These are deliberately framework-light dataclasses so the strategy and backtest
code can construct them without any exchange/networking dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import ActionKind, PositionState, Side, SignalDir


@dataclass(frozen=True)
class Candle:
    """A single OHLCV candle. ``start_time`` is epoch seconds of the candle open."""

    start_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_ws(cls, msg: dict) -> Candle:
        """Build from a Delta WebSocket ``candlestick_*`` message."""
        return cls(
            start_time=int(msg["candle_start_time"]) // 1_000_000
            if int(msg["candle_start_time"]) > 10_000_000_000
            else int(msg["candle_start_time"]),
            open=float(msg["open"]),
            high=float(msg["high"]),
            low=float(msg["low"]),
            close=float(msg["close"]),
            volume=float(msg.get("volume", 0.0)),
        )

    @classmethod
    def from_rest(cls, row: dict) -> Candle:
        """Build from a Delta REST ``/v2/history/candles`` row."""
        return cls(
            start_time=int(row["time"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
        )


@dataclass(frozen=True)
class Signal:
    """Result of feeding one closed candle to the Supertrend calculator."""

    direction: SignalDir
    flipped: bool
    value: float
    candle: Candle


@dataclass(frozen=True)
class Position:
    """Net position on the exchange for our product."""

    product_id: int
    size: int  # signed integer contracts: >0 long, <0 short, 0 flat
    entry_price: float | None = None

    @property
    def state(self) -> PositionState:
        if self.size > 0:
            return PositionState.LONG
        if self.size < 0:
            return PositionState.SHORT
        return PositionState.FLAT

    @property
    def abs_size(self) -> int:
        return abs(self.size)


@dataclass(frozen=True)
class OrderResult:
    """Outcome of an order placement."""

    order_id: int | None
    product_id: int
    side: Side
    size: int
    state: str | None = None
    average_fill_price: float | None = None
    raw: dict | None = None


@dataclass(frozen=True)
class Action:
    """A primitive instruction emitted by the position state machine."""

    kind: ActionKind
    side: Side
    reduce_only: bool = False


@dataclass(frozen=True)
class Trade:
    """A completed round-trip leg recorded in the ledger."""

    timestamp: int
    side: Side
    size: int
    price: float
    kind: str  # "ENTRY" | "EXIT"
    realized_pnl: float = 0.0
