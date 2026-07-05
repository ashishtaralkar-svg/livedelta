"""Command-line interface: ``live`` and ``download`` subcommands.

The strategy backtest lives in ``scripts/backtest_revbreak.py`` (RevBreak-Sell,
option-repriced). This CLI only wires the live engine and a candle downloader.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .backtest.data_loader import download
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="deltabot", description="Delta Exchange RevBreak-Sell bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_live = sub.add_parser("live", help="Run the live trading engine")
    p_live.set_defaults(func=_cmd_live)

    p_dl = sub.add_parser("download", help="Download historical candles into the cache")
    p_dl.add_argument("--start", required=True, help="YYYY-MM-DD (UTC) or epoch seconds")
    p_dl.add_argument("--end", required=True, help="YYYY-MM-DD (UTC) or epoch seconds")
    p_dl.add_argument("--symbol", default=None)
    p_dl.add_argument("--resolution", default=None)
    p_dl.add_argument("--cache-dir", default=".cache/candles")
    p_dl.set_defaults(func=_cmd_download)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
