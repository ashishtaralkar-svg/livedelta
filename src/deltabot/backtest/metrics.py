"""Performance metrics computed from a list of completed round trips."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..pnl import RoundTrip


@dataclass
class Metrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    net_profit: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    equity_curve: list[tuple[int, float]] = field(default_factory=list)  # (exit_time, equity)
    monthly_returns: dict[str, float] = field(default_factory=dict)

    def render(self, starting_equity: float = 0.0) -> str:
        lines = [
            "==================== Backtest Results ====================",
            f"Total Trades     : {self.total_trades}",
            f"Winning Trades   : {self.winning_trades}",
            f"Losing Trades    : {self.losing_trades}",
            f"Win Rate         : {self.win_rate * 100:.2f}%",
            f"Net Profit       : {self.net_profit:,.2f}",
            f"Gross Profit     : {self.gross_profit:,.2f}",
            f"Gross Loss       : {self.gross_loss:,.2f}",
            f"Profit Factor    : {self.profit_factor:.2f}",
            f"Max Drawdown     : {self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)",
            "---------------------- Monthly Returns -------------------",
        ]
        for month, ret in sorted(self.monthly_returns.items()):
            lines.append(f"  {month}: {ret:,.2f}")
        lines.append("==========================================================")
        return "\n".join(lines)


def compute(trips: list[RoundTrip], starting_equity: float = 0.0) -> Metrics:
    """Compute the full metric set from completed round trips."""
    total = len(trips)
    wins = [t for t in trips if t.is_win]
    losses = [t for t in trips if not t.is_win]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = sum(t.pnl for t in losses)  # <= 0
    net = gross_profit + gross_loss
    win_rate = (len(wins) / total) if total else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")

    # Equity curve and max drawdown (in absolute PnL terms).
    equity = starting_equity
    peak = starting_equity
    max_dd = 0.0
    curve: list[tuple[int, float]] = []
    for t in trips:
        equity += t.pnl
        curve.append((t.exit_time, equity))
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
    max_dd_pct = (max_dd / peak * 100.0) if peak > 0 else 0.0

    # Monthly returns keyed by YYYY-MM of the exit time.
    monthly: dict[str, float] = {}
    for t in trips:
        month = datetime.fromtimestamp(t.exit_time, tz=UTC).strftime("%Y-%m")
        monthly[month] = monthly.get(month, 0.0) + t.pnl

    return Metrics(
        total_trades=total,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=win_rate,
        net_profit=net,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        equity_curve=curve,
        monthly_returns=monthly,
    )
