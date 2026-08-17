"""Backtest: Daily Trend + EMA/MA Cross + HA Setup -- option-SELL execution of
src/deltabot/strategy/daily_trend_ema_cross.py (Python port of
daily_trend_ema_cross_strategy.pine).

SELL signal (bearish setup breaks) -> SELL a CALL (CE) near --target-premium
(default 1400). BUY signal (bullish setup breaks) -> SELL a PUT (PE). 70%
decay target (buy back at 30% of entry premium), else the fixed setup-range
SL. Daily --square-off-hour/minute (default 17:25 IST, matching every other
backtest in this repo) force-closes any still-open position -- added because
each leg's premium history is only fetched for a ~3-day window around entry
(see resolve_by_premium's win_start/win_end), so a trade that never hits its
SL/TP within that window would otherwise sit "open" against stale, frozen
premium data for however long the backtest runs. Intrinsic-value flooring is
ON by default.

Run:  python scripts/backtest_daily_trend_ema_cross.py --days 7
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
from deltabot.strategy.daily_trend_ema_cross import DailyTrendEmaCrossStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _ist_mins(ts: int) -> int:
    d = datetime.fromtimestamp(ts, tz=_IST)
    return d.hour * 60 + d.minute


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = DailyTrendEmaCrossStrategy(
        ema_len=args.ema_len, ma_len=args.ma_len,
        gap_start_hour=args.gap_start_hour, gap_start_minute=args.gap_start_minute,
        gap_end_hour=args.gap_end_hour, gap_end_minute=args.gap_end_minute,
    )
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots
    lot_size = args.lot_size if args.lot_size > 0 else op.LOT_SIZE.get(underlying, op.LOT_BTC)
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor
    tp_decay = args.tp_decay_pct / 100.0

    cache: dict = {}
    trades: list[dict] = []
    pos: dict | None = None

    def open_leg(client, ts: int, btc_px: float, is_buy: bool, tag: str) -> bool:
        nonlocal pos
        # bullish signal (buy_signal) -> sell PUT; bearish -> sell CALL.
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
        tp_price = entry_prem * (1.0 - tp_decay) if tp_decay > 0 else None
        pos = {"is_buy": is_buy, "sym": sym, "candles": ocandles, "entry_time": ts,
               "entry_btc": btc_px, "entry_prem": entry_prem, "tag": tag,
               "tp_price": tp_price, "last_check": ts}
        return True

    def buyback_prem(ts: int, exit_btc: float) -> float:
        p = op.premium_at(pos["candles"], ts, step)
        return p if p is not None else op.intrinsic_value(pos["sym"], exit_btc)

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float) -> None:
        nonlocal pos
        assert pos is not None
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        entry_fill = pos["entry_prem"] * (1 - es)   # selling: receive less on entry
        exit_fill = exit_prem * (1 + xs)            # buying back: pay more
        gross = (entry_fill - exit_fill) * lots * lot_size   # profit when premium DECAYS
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots, lot_size)
               + op.side_fee(exit_btc, exit_fill, lots, lot_size))
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time,
            "signal": "BUY" if pos["is_buy"] else "SELL", "contract": pos["sym"],
            "tag": pos["tag"], "btc_entry": pos["entry_btc"], "btc_exit": exit_btc,
            "opt_in": entry_fill, "opt_out": exit_fill, "reason": reason,
            "gross": gross, "fee": fee, "net": gross - fee,
        })
        pos = None

    sq_mins = args.square_off_hour * 60 + args.square_off_minute
    prev_mins: int | None = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            mins = _ist_mins(c.start_time)
            square_off = prev_mins is not None and mins >= sq_mins and prev_mins < sq_mins
            prev_mins = mins

            dec = strategy.update(c)

            if pos is not None and dec is not None and dec.has_exit:
                eprice = dec.long_exit_price if dec.long_exit else dec.short_exit_price
                close(dec.exit_reason, buyback_prem(c.start_time, eprice), c.start_time, eprice)

            # 70% decay TP (mark check on the closed bar).
            if pos is not None and pos["tp_price"] is not None:
                can_decay = (not floor) or op.intrinsic_value(pos["sym"], c.close) <= pos["tp_price"]
                if can_decay:
                    t_tp = op.premium_at(pos["candles"], c.start_time, step)
                    if t_tp is not None and t_tp <= pos["tp_price"]:
                        close("TP", pos["tp_price"], c.start_time, c.close)
                        strategy.force_flat()

            # Daily square-off: force-closes any still-open position so it
            # never rides on stale premium data past resolve_by_premium's
            # ~3-day fetch window (see module docstring).
            if pos is not None and square_off:
                close("EOD", buyback_prem(c.start_time, c.close), c.start_time, c.close)
                strategy.force_flat()

            if pos is None and dec is not None and dec.has_entry:
                if c.start_time < sim_start:
                    strategy.force_flat()   # warmup-window entry: don't take it
                elif not open_leg(client, c.start_time, c.close, dec.buy_signal, "ENTRY"):
                    strategy.force_flat()   # couldn't price the contract; stay flat

        if pos is not None:
            last = candles[-1]
            close("OPEN_AT_END", buyback_prem(last.start_time, last.close), last.start_time, last.close)

    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 118}")
    print(f"Daily Trend + EMA/MA Cross + HA Setup [OPTION SELL] -- {args.days}d, "
          f"premium ~{args.target_premium:.0f}, {args.lots} lots, {args.tp_decay_pct:.0f}%-decay TP, "
          f"floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 118}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'sig':<5}{'contract':<22}{'btc in':>9}{'btc out':>9}"
          f"{'opt in':>8}{'opt out':>8} {'reason':<10}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{t['signal']:<5}{t['contract']:<22}"
              f"{t['btc_entry']:>9.1f}{t['btc_exit']:>9.1f}{t['opt_in']:>8.1f}{t['opt_out']:>8.1f} "
              f"{t['reason']:<10}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 118}")
    print(f"Legs: {len(closed)} closed" + (" (+1 still open at data end)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "TP", "EOD", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<12} n={len(rs):<4} net ${sum(t['net'] for t in rs):>11.2f}")
    print(f"TOTAL NET: ${sum(t['net'] for t in trades):.2f} "
          f"(gross ${sum(t['gross'] for t in trades):.2f}, fees ${sum(t['fee'] for t in trades):.2f})")


def main() -> None:
    p = argparse.ArgumentParser(description="Daily Trend + EMA/MA Cross + HA Setup backtest -- option-sell execution")
    p.add_argument("--days", type=float, default=7, help="look-back window in days")
    p.add_argument("--warmup-days", type=int, default=30,
                   help="extra leading days so EMA50/MA50 and the daily trend are warm before the report window")
    p.add_argument("--resolution", default="5m")
    p.add_argument("--ema-len", type=int, default=50)
    p.add_argument("--ma-len", type=int, default=50)
    p.add_argument("--gap-start-hour", type=int, default=17)
    p.add_argument("--gap-start-minute", type=int, default=25)
    p.add_argument("--gap-end-hour", type=int, default=17)
    p.add_argument("--gap-end-minute", type=int, default=30)
    p.add_argument("--square-off-hour", type=int, default=17)
    p.add_argument("--square-off-minute", type=int, default=25)
    p.add_argument("--target-premium", type=float, default=1400.0)
    p.add_argument("--tp-decay-pct", type=float, default=70.0)
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lot-size", type=float, default=0.0, help="0 = auto-derive from the underlying symbol")
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true",
                   help="disable intrinsic-value flooring (NOT recommended)")
    args = p.parse_args()

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


if __name__ == "__main__":
    main()
