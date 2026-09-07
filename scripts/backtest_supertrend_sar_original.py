"""Backtest: Supertrend SAR -- the ORIGINAL config, frozen as its own entry
point (candle-color first entry, restart 17:30, reset 05:30).

Reuses run()/report()/export() from backtest_supertrend_sar.py unchanged --
this script only fixes the argument DEFAULTS to the specific combination
that was this session's own comparison baseline before the v7 ORB
experiment and the v8 session-reset fix:

  * orb_enabled=False (candle-color first entry at start_hour:start_minute,
    default 17:35 IST) -- the CURRENT backtest_supertrend_sar.py default
    too, so this line isn't strictly needed to pin, but is spelled out
    here for clarity since this script exists specifically to freeze a
    known reference point.
  * reset_hour:reset_minute = 05:30 -- also now backtest_supertrend_sar.
    py's own default again (v8 moved it to 17:30 to fix a real stale-SL
    bug, more theoretically correct, but it backtested WORSE on every
    1wk/1mo/3mo window re-tested and was reverted). Both scripts agree
    here, but this one pins it explicitly since "Original" is specifically
    defined as 05:30 regardless of what the generic script's default
    happens to be at any given time.
  * restart_hour:restart_minute = 17:30 -- same as backtest_supertrend_sar.
    py's current default (this was only ever moved to 18:05 while ORB was
    the default; both scripts agree here).

Every other setting (TP-roll, min-SL guard, weekend blackout, --side,
etc.) is unchanged from backtest_supertrend_sar.py -- this script only
narrows the THREE knobs above, all still overridable via the same flags
if you want to deviate from "Original" without editing the file. Pass
--weekend-blackout for the FINAL/deployed config (src/deltabot/config.py's
sar_weekend_blackout default, live since the EC2 push) -- it isn't this
script's own default since "Original" predates that addition.

Run:  python scripts/backtest_supertrend_sar_original.py --days 7
"""

from __future__ import annotations

import argparse
import time

from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.config import load_settings
from deltabot.logging_setup import setup_logging

from backtest_supertrend_sar import _ist, export, report, run


def main() -> None:
    p = argparse.ArgumentParser(
        description="Supertrend SAR backtest -- the ORIGINAL config (candle-color entry, "
                     "restart 17:30, reset 05:30)"
    )
    p.add_argument("--days", type=float, default=7, help="look-back window in days")
    p.add_argument("--warmup-days", type=int, default=3,
                   help="extra leading days so Supertrend's ATR is warm before the report window")
    p.add_argument("--resolution", default="1m")
    p.add_argument("--atr-period", type=int, default=10)
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--min-sl-atr-mult", type=float, default=1.0)
    p.add_argument("--start-hour", type=int, default=17, help="candle-color first-entry hour (IST)")
    p.add_argument("--start-minute", type=int, default=35)
    p.add_argument("--reset-hour", type=int, default=5,
                   help="ORIGINAL session-reset hour (IST) -- 05:30, not the v8-fixed 17:30")
    p.add_argument("--reset-minute", type=int, default=30)
    p.add_argument("--square-off-hour", type=int, default=17)
    p.add_argument("--square-off-minute", type=int, default=25)
    p.add_argument("--restart-hour", type=int, default=17, help="evening restart hour (IST)")
    p.add_argument("--restart-minute", type=int, default=30)
    p.add_argument("--weekend-blackout", action="store_true")
    p.add_argument("--side", choices=["sell", "buy"], default="sell")
    p.add_argument("--target-premium", type=float, default=1400.0)
    p.add_argument("--tp-pct", type=float, default=70.0)
    p.add_argument("--lots", type=int, default=10)
    p.add_argument("--lot-size", type=float, default=0.0)
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true")
    p.add_argument("--out", default="", help="also write every leg to this .xlsx file")
    args = p.parse_args()
    args.orb_enabled = False   # ORIGINAL never used ORB -- not exposed as a flag here on purpose
    args.orb_start_hour, args.orb_start_minute = 17, 30
    args.orb_end_hour, args.orb_end_minute = 18, 0

    setup_logging("WARNING")
    settings = load_settings()

    now = int(time.time())
    sim_start = now - int(args.days * 86400)
    dl_start = sim_start - args.warmup_days * 86400

    df = download(symbol=settings.symbol, start=dl_start, end=now, resolution=args.resolution)
    candles = df_to_candles(df)
    print(f"Candles: {len(candles)} ({args.resolution}) {_ist(candles[0].start_time)} .. {_ist(candles[-1].start_time)}")

    trades = run(candles, settings, args, sim_start)
    report(trades, args)
    if args.out:
        export(trades, args, args.out)


if __name__ == "__main__":
    main()
