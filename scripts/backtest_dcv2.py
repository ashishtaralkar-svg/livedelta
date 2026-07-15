"""Backtest: DCv2 -- Python port of dchannel_strategy.pine (2026-07-15 state).

Directional BTC backtest (plain long/short, 1x qty -- NO options), exactly like
the Pine strategy tester: EMA(50)/EMA(200) state-armed Donchian-touch +
open==low/high confirm on synthetic Heikin Ashi; entry on the signal-range
break; exits ONLY on the fixed range SL or the EMA relationship flipping
against the position. No TP, no EOD close (positions can hold across days);
Sat/Sun entries blocked by default.

See src/deltabot/strategy/dcv2.py for the full rule set.

Run:  python scripts/backtest_dcv2.py --days 30
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.config import load_settings
from deltabot.logging_setup import setup_logging
from deltabot.models import Candle
from deltabot.strategy.dcv2 import DCv2Strategy

_IST = ZoneInfo("Asia/Kolkata")

_NATIVE_RES = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}


def _res_seconds(res: str) -> int | None:
    if res in _NATIVE_RES:
        return _NATIVE_RES[res]
    if res.endswith("m"):
        try:
            return int(res[:-1]) * 60
        except ValueError:
            return None
    return None


def _base_for(target_sec: int) -> tuple[str, int]:
    for res, sec in sorted(_NATIVE_RES.items(), key=lambda kv: -kv[1]):
        if sec <= target_sec and target_sec % sec == 0:
            return res, sec
    return "1m", 60


def _resample_candles(candles: list[Candle], target_sec: int) -> list[Candle]:
    buckets: dict[int, list] = {}
    order: list[int] = []
    for c in candles:
        key = c.start_time - (c.start_time % target_sec)
        b = buckets.get(key)
        if b is None:
            buckets[key] = [c.open, c.high, c.low, c.close, c.volume]
            order.append(key)
        else:
            b[1] = max(b[1], c.high)
            b[2] = min(b[2], c.low)
            b[3] = c.close
            b[4] += c.volume
    return [Candle(start_time=k, open=b[0], high=b[1], low=b[2], close=b[3], volume=b[4])
            for k in order for b in (buckets[k],)]


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M IST")


def run(candles: list[Candle], args, sim_start: int) -> list[dict]:
    strategy = DCv2Strategy(
        dc_period=args.dc_period,
        ema_trend_length=args.ema_trend_length,
        ema_long_length=args.ema_long_length,
        skip_weekdays=frozenset({5, 6}) if not args.no_skip_weekends else frozenset(),
        day_start_hour=args.day_start_hour, day_start_minute=args.day_start_minute,
        square_off_hour=args.square_off_hour, square_off_minute=args.square_off_minute,
    )

    trades: list[dict] = []
    open_trade: dict | None = None

    for candle in candles:
        decision = strategy.update(candle)
        if decision is None:
            continue

        if decision.has_exit and open_trade is not None:
            exit_price = decision.long_exit_price if decision.long_exit else decision.short_exit_price
            side = open_trade["side"]
            points = (exit_price - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - exit_price)
            open_trade.update(
                exit_time=candle.start_time, exit=exit_price,
                reason=decision.exit_reason, points=points,
                pnl=points * args.qty,
            )
            trades.append(open_trade)
            open_trade = None

        if decision.has_entry and candle.start_time >= sim_start:
            open_trade = {
                "entry_time": candle.start_time,
                "side": "LONG" if decision.buy_signal else "SHORT",
                "entry": decision.entry_price,
                "sl": decision.sl_level,
            }
        elif decision.has_entry:
            # Warmup-window entry: track it so its exit doesn't corrupt state,
            # but don't report it.
            open_trade = None
            strategy.force_flat()

    if open_trade is not None:
        last = candles[-1]
        side = open_trade["side"]
        points = (last.close - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - last.close)
        open_trade.update(exit_time=last.start_time, exit=last.close,
                          reason="OPEN_AT_END", points=points, pnl=points * args.qty)
        trades.append(open_trade)

    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 100}")
    print(f"DCv2 backtest -- {args.days}d, {args.resolution}, DC({args.dc_period}), "
          f"EMA({args.ema_trend_length}/{args.ema_long_length}), qty {args.qty} BTC, "
          f"weekends {'TRADED' if args.no_skip_weekends else 'blocked'}")
    print(f"{'=' * 100}")
    if not trades:
        print("No trades.")
        return

    print(f"{'entry (IST)':<22}{'side':<7}{'entry':>12}{'exit':>12}{'sl':>12}"
          f"{'reason':<14}{'points':>10}{'pnl $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<22}{t['side']:<7}{t['entry']:>12.2f}{t['exit']:>12.2f}"
              f"{(t['sl'] if t['sl'] is not None else float('nan')):>12.2f}"
              f"{t['reason']:<14}{t['points']:>10.2f}{t['pnl']:>10.2f}")

    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["points"] > 0]
    total_pnl = sum(t["pnl"] for t in trades)
    print(f"{'-' * 100}")
    print(f"Trades: {len(closed)} closed"
          + (f" (+1 still open at data end)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "EMA_CROSS", "TRAIL", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<12} n={len(rs):<4} pnl ${sum(t['pnl'] for t in rs):>10.2f}")
    print(f"TOTAL P&L: ${total_pnl:.2f}  ({sum(t['points'] for t in trades):+.2f} BTC points x {args.qty})")


def main() -> None:
    p = argparse.ArgumentParser(description="DCv2 (Pine port) directional BTC backtest")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--warmup-days", type=int, default=7,
                   help="extra leading days so EMA200/Donchian are warm before the report window")
    p.add_argument("--resolution", default="5m")
    p.add_argument("--dc-period", type=int, default=20)
    p.add_argument("--ema-trend-length", type=int, default=50)
    p.add_argument("--ema-long-length", type=int, default=200)
    p.add_argument("--qty", type=float, default=1.0, help="BTC position size (P&L = points x qty)")
    p.add_argument("--no-skip-weekends", action="store_true",
                   help="also take entries on Sat/Sun (blocked by default, like the Pine script)")
    p.add_argument("--day-start-hour", type=int, default=17)
    p.add_argument("--day-start-minute", type=int, default=30)
    p.add_argument("--square-off-hour", type=int, default=17)
    p.add_argument("--square-off-minute", type=int, default=25)
    args = p.parse_args()

    setup_logging("WARNING")
    settings = load_settings()

    now = int(time.time())
    sim_start = now - args.days * 86400
    dl_start = sim_start - args.warmup_days * 86400

    target_sec = _res_seconds(args.resolution)
    if target_sec is None:
        raise SystemExit(f"Unsupported resolution: {args.resolution}")
    if args.resolution in _NATIVE_RES:
        df = download(symbol=settings.symbol, start=dl_start, end=now, resolution=args.resolution)
        candles = df_to_candles(df)
    else:
        base_res, _ = _base_for(target_sec)
        df = download(symbol=settings.symbol, start=dl_start, end=now, resolution=base_res)
        candles = _resample_candles(df_to_candles(df), target_sec)

    print(f"Candles: {len(candles)} ({args.resolution}) "
          f"{_ist(candles[0].start_time)} .. {_ist(candles[-1].start_time)}")
    trades = run(candles, args, sim_start)
    report(trades, args)


if __name__ == "__main__":
    main()
