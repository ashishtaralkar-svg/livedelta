"""Backtest: DCv2 -- Python port of dchannel_strategy.pine (2026-07-15 state).

Two execution modes:

  --mode option (DEFAULT): the signal SELLS options, matching the live bots'
    convention -- BUY signal -> SELL a PUT near --target-premium (default 900),
    SELL signal -> SELL a CALL. Exits when the SYSTEM says so (BTC touching the
    signal-range SL, or the EMA reversal; no TP, no EOD close): the option is
    bought back at its market premium at that moment. Intrinsic-value flooring
    is ON by default (illiquid ITM option candles print below intrinsic and
    inflate results -- see backtest-intrinsic-floor).

  --mode btc: plain directional long/short on BTC (Pine-strategy-tester style),
    P&L = points x --qty.

See src/deltabot/strategy/dcv2.py for the full rule set.

Run:  python scripts/backtest_dcv2.py --days 30
      python scripts/backtest_dcv2.py --days 30 --mode btc
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from deltabot.backtest import option_pricing as op
from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.config import load_settings
from deltabot.enums import OptionType
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


def _make_strategy(args) -> DCv2Strategy:
    return DCv2Strategy(
        dc_period=args.dc_period,
        ema_trend_length=args.ema_trend_length,
        ema_long_length=args.ema_long_length,
        skip_weekdays=frozenset({5, 6}) if not args.no_skip_weekends else frozenset(),
        day_start_hour=args.day_start_hour, day_start_minute=args.day_start_minute,
        square_off_hour=args.square_off_hour, square_off_minute=args.square_off_minute,
    )


# --------------------------------------------------------------------- #
# Mode: btc (directional, Pine-strategy-tester style)
# --------------------------------------------------------------------- #
def run_btc(candles: list[Candle], args, sim_start: int) -> list[dict]:
    strategy = _make_strategy(args)
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
            open_trade.update(exit_time=candle.start_time, exit=exit_price,
                              reason=decision.exit_reason, points=points, pnl=points * args.qty)
            trades.append(open_trade)
            open_trade = None

        if decision.has_entry and candle.start_time >= sim_start:
            open_trade = {"entry_time": candle.start_time,
                          "side": "LONG" if decision.buy_signal else "SHORT",
                          "entry": decision.entry_price, "sl": decision.sl_level}
        elif decision.has_entry:
            open_trade = None
            strategy.force_flat()   # warmup-window entry: don't report it

    if open_trade is not None:
        last = candles[-1]
        side = open_trade["side"]
        points = (last.close - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - last.close)
        open_trade.update(exit_time=last.start_time, exit=last.close,
                          reason="OPEN_AT_END", points=points, pnl=points * args.qty)
        trades.append(open_trade)
    return trades


def report_btc(trades: list[dict], args) -> None:
    print(f"\n{'=' * 100}")
    print(f"DCv2 backtest [BTC directional] -- {args.days}d, {args.resolution}, "
          f"DC({args.dc_period}), EMA({args.ema_trend_length}/{args.ema_long_length}), "
          f"qty {args.qty} BTC, weekends {'TRADED' if args.no_skip_weekends else 'blocked'}")
    print(f"{'=' * 100}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<22}{'side':<7}{'entry':>12}{'exit':>12}{'sl':>12} "
          f"{'reason':<13}{'points':>10}{'pnl $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<22}{t['side']:<7}{t['entry']:>12.2f}{t['exit']:>12.2f}"
              f"{(t['sl'] if t['sl'] is not None else float('nan')):>12.2f} "
              f"{t['reason']:<13}{t['points']:>10.2f}{t['pnl']:>10.2f}")
    _summary(trades, "pnl")
    print(f"TOTAL P&L: ${sum(t['pnl'] for t in trades):.2f} "
          f"({sum(t['points'] for t in trades):+.2f} BTC points x {args.qty})")


# --------------------------------------------------------------------- #
# Mode: option (SELL a PUT on a buy signal / SELL a CALL on a sell signal)
# --------------------------------------------------------------------- #
def _settle_ts(expiry_dt: datetime) -> int:
    """Daily options settle at 17:30 IST on the expiry date."""
    return int(datetime(expiry_dt.year, expiry_dt.month, expiry_dt.day,
                        17, 30, tzinfo=_IST).timestamp())


def run_option(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = _make_strategy(args)
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor
    cache: dict = {}
    trades: list[dict] = []
    pos: dict | None = None

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float) -> None:
        nonlocal pos
        assert pos is not None
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        entry_fill = pos["entry_prem"] * (1 - es)   # selling: receive less on entry
        exit_fill = exit_prem * (1 + xs)            # buying back: pay more
        gross = (entry_fill - exit_fill) * lots * op.LOT_BTC
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots)
               + op.side_fee(exit_btc, exit_fill, lots))
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time,
            "signal": "BUY" if pos["is_buy"] else "SELL", "contract": pos["sym"],
            "btc_entry": pos["entry_btc"], "btc_exit": exit_btc,
            "sl": pos["sl_level"], "opt_in": entry_fill, "opt_out": exit_fill,
            "reason": reason, "gross": gross, "fee": fee, "net": gross - fee,
        })
        pos = None

    def open_leg(client: httpx.Client, ts: int, btc_px: float, is_buy: bool,
                 sl_level: float | None) -> bool:
        """Sell the option nearest --target-premium (PUT for a buy signal,
        CALL for a sell signal). Returns False if it couldn't be priced."""
        nonlocal pos
        otype = OptionType.PUT if is_buy else OptionType.CALL
        expiry = op.select_expiry_date(ts, cutoff)
        resolved = op.resolve_by_premium(
            client, underlying, otype, btc_px, expiry, interval,
            args.target_premium, ts, ts, ts - 86400, ts + 2 * 86400,
            args.opt_resolution, step, cache,
        )
        if resolved is None:
            return False
        sym, _, ocandles = resolved
        entry_prem = op.premium_at(ocandles, ts, step)
        if entry_prem is None:
            return False
        if floor:
            entry_prem = max(entry_prem, op.intrinsic_value(sym, btc_px))
        pos = {
            "is_buy": is_buy, "sym": sym, "candles": ocandles,
            "entry_time": ts, "entry_btc": btc_px, "entry_prem": entry_prem,
            "sl_level": sl_level, "settle_ts": _settle_ts(expiry),
        }
        return True

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            dec = strategy.update(c)

            # 0. Expiry roll: a sold daily option cannot be held across its
            #    17:30 IST settlement. Settle the leg at intrinsic; if the
            #    SYSTEM trade is still open (no exit this bar), immediately
            #    sell a fresh option in the same direction (daily roll).
            if pos is not None and c.start_time >= pos["settle_ts"]:
                intr = op.intrinsic_value(pos["sym"], c.close)
                was_buy, was_sl = pos["is_buy"], pos["sl_level"]
                close("EXPIRY_ROLL", intr, c.start_time, c.close)
                still_open = strategy.position_state.name != "FLAT" and (dec is None or not dec.has_exit)
                if still_open and not open_leg(client, c.start_time, c.close, was_buy, was_sl):
                    strategy.force_flat()   # can't roll -> the trade ends here

            if dec is None:
                continue

            # 1. System exit (SL at the range extreme, or EMA reversal): buy the
            #    option back at its market premium at that moment.
            if pos is not None and dec.has_exit:
                eprice = dec.long_exit_price if dec.long_exit else dec.short_exit_price
                exit_prem = op.premium_at(pos["candles"], c.start_time, step)
                if exit_prem is not None:
                    close(dec.exit_reason, exit_prem, c.start_time, eprice)
                else:
                    pos = None   # cannot price the exit; strategy is already flat

            # 2. New entry: BUY signal -> SELL PUT, SELL signal -> SELL CALL,
            #    nearest --target-premium.
            if pos is None and dec.has_entry:
                if c.start_time < sim_start:
                    strategy.force_flat()   # warmup-window entry: don't take it
                    continue
                if not open_leg(client, c.start_time, c.close, dec.buy_signal, dec.sl_level):
                    strategy.force_flat()   # couldn't price the contract; stay flat
                    continue

        if pos is not None:
            last = candles[-1]
            exit_prem = op.premium_at(pos["candles"], last.start_time, step)
            if exit_prem is not None:
                close("OPEN_AT_END", exit_prem, last.start_time, last.close)
            else:
                print(f"NOTE: open position {pos['sym']} could not be priced at data end; dropped.")
                pos = None

    return trades


def report_option(trades: list[dict], args) -> None:
    print(f"\n{'=' * 118}")
    print(f"DCv2 backtest [OPTION SELL] -- {args.days}d, {args.resolution}, "
          f"DC({args.dc_period}), EMA({args.ema_trend_length}/{args.ema_long_length}), "
          f"premium ~{args.target_premium:.0f}, {args.lots} lots, "
          f"intrinsic floor {'OFF' if args.no_intrinsic_floor else 'ON'}, "
          f"weekends {'TRADED' if args.no_skip_weekends else 'blocked'}")
    print(f"{'=' * 118}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<22}{'sig':<5}{'contract':<22}{'btc in':>9}{'btc out':>9}"
          f"{'opt in':>8}{'opt out':>8} {'reason':<13}{'net $':>9}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<22}{t['signal']:<5}{t['contract']:<22}"
              f"{t['btc_entry']:>9.1f}{t['btc_exit']:>9.1f}"
              f"{t['opt_in']:>8.1f}{t['opt_out']:>8.1f} {t['reason']:<13}{t['net']:>9.2f}")
    _summary(trades, "net")
    total_fee = sum(t["fee"] for t in trades)
    print(f"TOTAL NET: ${sum(t['net'] for t in trades):.2f} "
          f"(gross ${sum(t['gross'] for t in trades):.2f}, fees ${total_fee:.2f})")


def _summary(trades: list[dict], pnl_key: str) -> None:
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t[pnl_key] > 0]
    print(f"{'-' * 118}")
    print(f"Trades: {len(closed)} closed"
          + (" (+1 still open at data end)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "EMA_CROSS", "EXPIRY_ROLL", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<12} n={len(rs):<4} pnl ${sum(t[pnl_key] for t in rs):>10.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="DCv2 (Pine port) backtest -- option-sell (default) or directional BTC")
    p.add_argument("--mode", choices=("option", "btc"), default="option")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--warmup-days", type=int, default=7,
                   help="extra leading days so EMA200/Donchian are warm before the report window")
    p.add_argument("--resolution", default="5m")
    p.add_argument("--dc-period", type=int, default=20)
    p.add_argument("--ema-trend-length", type=int, default=50)
    p.add_argument("--ema-long-length", type=int, default=200)
    p.add_argument("--no-skip-weekends", action="store_true",
                   help="also take entries on Sat/Sun (blocked by default, like the Pine script)")
    p.add_argument("--day-start-hour", type=int, default=17)
    p.add_argument("--day-start-minute", type=int, default=30)
    p.add_argument("--square-off-hour", type=int, default=17)
    p.add_argument("--square-off-minute", type=int, default=25)
    # btc mode
    p.add_argument("--qty", type=float, default=1.0, help="[btc mode] BTC position size")
    # option mode
    p.add_argument("--target-premium", type=float, default=900.0,
                   help="[option mode] sell the strike whose premium is nearest this (default 900)")
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--opt-resolution", default="1m", help="[option mode] option candle resolution")
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true",
                   help="[option mode] disable intrinsic-value flooring (NOT recommended)")
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
    if args.mode == "btc":
        report_btc(run_btc(candles, args, sim_start), args)
    else:
        report_option(run_option(candles, settings, args, sim_start), args)


if __name__ == "__main__":
    main()
