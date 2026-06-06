"""Trade ledger and realized-PnL accounting, shared by live and backtest.

PnL model: positions are sized in *contracts*, where ``contract_value`` BTC is
the notional per contract (0.001 BTC for BTCUSD/BTCUSDT). Realized PnL is
computed in quote terms (USD/USDT) using the linear convention:

    pnl = direction * qty_btc * (exit_price - entry_price)
    qty_btc = contracts * contract_value

This is the standard way to evaluate the strategy's directional edge. For the
inverse BTCUSD contract the exchange's exact settlement differs slightly, but
the sign and ranking of trades — and therefore win rate / profit factor / draw-
down — are governed by the directional move captured here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .enums import SignalDir


@dataclass(frozen=True)
class RoundTrip:
    """A completed position: opened at entry, closed at exit."""

    entry_time: int
    exit_time: int
    direction: int  # +1 long, -1 short
    qty_btc: float
    entry_price: float
    exit_price: float

    @property
    def pnl(self) -> float:
        return self.direction * self.qty_btc * (self.exit_price - self.entry_price)

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class TradeLedger:
    """Tracks the open position and the history of completed round trips."""

    contract_value: float = 0.001
    contracts: int = 1
    trips: list[RoundTrip] = field(default_factory=list)

    # Open position state
    _open_dir: int | None = None
    _open_price: float | None = None
    _open_time: int | None = None
    _open_qty: float = 0.0

    # For daily summaries: index into ``trips`` of the last reported trip.
    _last_summary_index: int = 0

    @property
    def qty_btc(self) -> float:
        return self.contracts * self.contract_value

    @property
    def has_open(self) -> bool:
        return self._open_dir is not None

    def open(
        self, direction: int, price: float, timestamp: int | None = None, qty_btc: float | None = None
    ) -> None:
        """Record opening a new position."""
        self._open_dir = direction
        self._open_price = price
        self._open_time = timestamp if timestamp is not None else int(time.time())
        self._open_qty = qty_btc if qty_btc is not None else self.qty_btc

    def close(self, price: float, timestamp: int | None = None) -> RoundTrip | None:
        """Close the open position, recording a RoundTrip. Returns it (or None)."""
        if self._open_dir is None or self._open_price is None or self._open_time is None:
            return None
        trip = RoundTrip(
            entry_time=self._open_time,
            exit_time=timestamp if timestamp is not None else int(time.time()),
            direction=self._open_dir,
            qty_btc=self._open_qty,
            entry_price=self._open_price,
            exit_price=price,
        )
        self.trips.append(trip)
        self._open_dir = None
        self._open_price = None
        self._open_time = None
        self._open_qty = 0.0
        return trip

    def reverse(self, new_direction: int, price: float, timestamp: int | None = None) -> RoundTrip | None:
        """Close any open position and open the opposite. Returns the closed trip."""
        closed = self.close(price, timestamp)
        self.open(new_direction, price, timestamp)
        return closed

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.trips)

    def daily_summary(self) -> dict:
        """Return a summary of trips recorded since the last call."""
        new_trips = self.trips[self._last_summary_index :]
        self._last_summary_index = len(self.trips)
        wins = [t for t in new_trips if t.is_win]
        pnl = sum(t.pnl for t in new_trips)
        return {
            "trades": len(new_trips),
            "wins": len(wins),
            "losses": len(new_trips) - len(wins),
            "pnl": pnl,
            "cumulative_pnl": self.realized_pnl,
        }


def direction_from_signal(signal_dir: int) -> int:
    """Normalise a SignalDir value to +1 / -1."""
    return SignalDir.LONG.value if signal_dir == SignalDir.LONG.value else SignalDir.SHORT.value
