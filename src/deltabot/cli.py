"""Command-line interface: ``live``, ``backtest`` and ``download`` subcommands."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .backtest import metrics
from .backtest.data_loader import df_to_candles, download
from .backtest.engine import BacktestEngine
from .config import load_settings
from .logging_setup import setup_logging


def _parse_date(s: str) -> int:
    """Parse YYYY-MM-DD (UTC) or an epoch-seconds integer into epoch seconds."""
    if s.isdigit():
        return int(s)
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp())


def _cmd_live(args: argparse.Namespace) -> None:
    from .main import main as live_main

    live_main()


def _cmd_download(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    df = download(
        symbol=args.symbol or settings.symbol,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        resolution=args.resolution or settings.resolution,
        base_url=settings.rest_base_url,
        cache_dir=Path(args.cache_dir),
    )
    print(f"Downloaded {len(df)} candles -> cache dir {args.cache_dir}")


def _cmd_backtest(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    symbol = args.symbol or settings.symbol
    df = download(
        symbol=symbol,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        resolution=args.resolution or settings.resolution,
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
    m = metrics.compute(result.trips, starting_equity=args.starting_equity)
    print(f"\nSymbol: {symbol}  Candles: {result.candles_processed}  "
          f"Supertrend({settings.atr_period},{settings.st_multiplier}) + "
          f"EMA{settings.ema_length} H/L + PrevDay OHLC")
    print(m.render())
    if args.equity_csv:
        _write_equity_csv(m, args.equity_csv)
        print(f"Equity curve written to {args.equity_csv}")


def _write_equity_csv(m: metrics.Metrics, path: str) -> None:
    lines = ["exit_time,equity"]
    lines += [f"{t},{e}" for t, e in m.equity_curve]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="deltabot", description="Delta Exchange Supertrend bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_live = sub.add_parser("live", help="Run the live trading engine")
    p_live.set_defaults(func=_cmd_live)

    common = {"symbol": "--symbol", "resolution": "--resolution"}
    for name in ("download", "backtest"):
        p = sub.add_parser(name, help=f"{name} historical candles")
        p.add_argument("--start", required=True, help="YYYY-MM-DD (UTC) or epoch seconds")
        p.add_argument("--end", required=True, help="YYYY-MM-DD (UTC) or epoch seconds")
        p.add_argument(common["symbol"], default=None)
        p.add_argument(common["resolution"], default=None)
        p.add_argument("--cache-dir", default=".cache/candles")
        if name == "backtest":
            p.add_argument("--equity-csv", default=None, help="Optional path to write equity curve CSV")
            p.add_argument(
                "--starting-equity",
                type=float,
                default=1000.0,
                help="Notional starting equity for drawdown %% and equity curve",
            )
            p.set_defaults(func=_cmd_backtest)
        else:
            p.set_defaults(func=_cmd_download)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
