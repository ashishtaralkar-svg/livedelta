"""Export per-trade backtest details to an Excel (.xlsx) file.

Runs the same strategy/backtest engine used by ``deltabot.cli backtest`` over a
date window, then writes one row per completed round-trip trade with entry/exit
prices, buy/sell prices, PnL and timestamps (both UTC and the strategy's day
timezone, default IST).

Usage:
    python scripts/export_trades.py --start <epoch|YYYY-MM-DD> --end <epoch|YYYY-MM-DD> \
        --symbol BTCUSD --resolution 1m --out btc_trades.xlsx

Set ``DELTA_TESTNET=false`` in the environment for real (mainnet) prices.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from deltabot.backtest import metrics
from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.backtest.engine import BacktestEngine
from deltabot.config import load_settings
from deltabot.logging_setup import setup_logging


def _parse_date(s: str) -> int:
    if s.isdigit():
        return int(s)
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())


def main() -> None:
    p = argparse.ArgumentParser(description="Export per-trade backtest details to Excel")
    p.add_argument("--start", required=True, help="epoch seconds or YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="epoch seconds or YYYY-MM-DD (UTC)")
    p.add_argument("--symbol", default=None)
    p.add_argument("--resolution", default=None)
    p.add_argument("--cache-dir", default=".cache/candles")
    p.add_argument("--out", default="trades.xlsx", help="output .xlsx path")
    args = p.parse_args()

    settings = load_settings()
    setup_logging(settings.log_level)
    symbol = args.symbol or settings.symbol
    resolution = args.resolution or settings.resolution
    tz = ZoneInfo(settings.day_tz)

    df = download(
        symbol=symbol,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        resolution=resolution,
        base_url=settings.rest_base_url,
        cache_dir=Path(args.cache_dir),
    )
    candles = df_to_candles(df)

    engine = BacktestEngine(
        period=settings.atr_period,
        multiplier=settings.st_multiplier,
        contracts=settings.contracts,
        contract_value=0.001,
        settings=settings,
    )
    result = engine.run(candles)

    def _fmt(ts: int, zone) -> str:
        return datetime.fromtimestamp(ts, tz=zone).strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    cum = 0.0
    for i, t in enumerate(result.trips, start=1):
        cum += t.pnl
        is_long = t.direction == 1
        rows.append(
            {
                "Trade #": i,
                "Side": "LONG" if is_long else "SHORT",
                f"Entry Time ({settings.day_tz})": _fmt(t.entry_time, tz),
                f"Exit Time ({settings.day_tz})": _fmt(t.exit_time, tz),
                "Entry Time (UTC)": _fmt(t.entry_time, UTC),
                "Exit Time (UTC)": _fmt(t.exit_time, UTC),
                "Duration (min)": round((t.exit_time - t.entry_time) / 60.0, 1),
                "Entry Price": round(t.entry_price, 2),
                "Exit Price": round(t.exit_price, 2),
                # Buy is the entry for a long / the exit for a short; sell is the mirror.
                "Buy Price": round(t.entry_price if is_long else t.exit_price, 2),
                "Sell Price": round(t.exit_price if is_long else t.entry_price, 2),
                "Qty (BTC)": t.qty_btc,
                "Price Move": round((t.exit_price - t.entry_price) * (1 if is_long else -1), 2),
                "PnL (USD)": round(t.pnl, 4),
                "Cumulative PnL (USD)": round(cum, 4),
                "Result": "WIN" if t.is_win else "LOSS",
            }
        )

    trades_df = pd.DataFrame(rows)

    m = metrics.compute(result.trips)
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Symbol", "Resolution", "Candles processed", "Total trades",
                "Winning trades", "Losing trades", "Win rate %", "Net profit (USD)",
                "Gross profit (USD)", "Gross loss (USD)", "Profit factor",
                "Max drawdown (USD)", "Max drawdown %", "Contracts (0.001 BTC each)",
            ],
            "Value": [
                symbol, resolution, result.candles_processed, m.total_trades,
                m.winning_trades, m.losing_trades, round(m.win_rate * 100, 2),
                round(m.net_profit, 4), round(m.gross_profit, 4), round(m.gross_loss, 4),
                round(m.profit_factor, 2), round(m.max_drawdown, 4),
                round(m.max_drawdown_pct, 2), settings.contracts,
            ],
        }
    )

    out_path = Path(args.out)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        trades_df.to_excel(xl, sheet_name="Trades", index=False)
        summary_df.to_excel(xl, sheet_name="Summary", index=False)
        # Auto-size columns for readability.
        for sheet, frame in (("Trades", trades_df), ("Summary", summary_df)):
            ws = xl.sheets[sheet]
            for idx, col in enumerate(frame.columns, start=1):
                width = max(len(str(col)), *(len(str(v)) for v in frame[col])) + 2
                ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = min(width, 40)

    print(f"Wrote {len(trades_df)} trades to {out_path.resolve()}")


if __name__ == "__main__":
    main()
