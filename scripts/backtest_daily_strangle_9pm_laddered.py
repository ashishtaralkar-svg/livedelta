"""Backtest: Daily 9 PM IST LADDERED Short Strangle -- multiple OTM%% strikes
per side, combined target/SL (an update to backtest_daily_strangle_9pm.py).

Same daily entry timing and combined target/SL rules as the base 9pm
strangle -- the only difference is HOW MANY strikes are sold per side.
Instead of one CALL + one PUT at --otm-pct%%, this sells ONE CALL and ONE
PUT at EACH of --otm-pcts (default "1,1.5,2" -- i.e. 3 calls + 3 puts = 6
legs total), all opened together and all closed together as ONE combined
position, exactly like the base script's 2-leg version:

  * TARGET: close ALL legs together the instant the COMBINED buyback cost
    (sum of every leg) decays to (100 - --target-pct)%% of the COMBINED
    entry premium (sum of every leg's entry) -- same REDUCTION convention,
    same default 70%%, matching the live 9pm bot exactly.
  * SL: close ALL legs together the instant the COMBINED buyback cost
    RISES to (100 + --sl-pct)%% of the COMBINED entry -- same RISE
    convention, same default 105%% (matches the live bot's current config).
  * FALLBACK: force-close whatever's open at --exit-hour:--exit-minute
    (default 17:00) IST THE NEXT DAY, same as the base script.

Run: python scripts/backtest_daily_strangle_9pm_laddered.py --days 30
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from deltabot.backtest import option_pricing as op
from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.config import load_settings
from deltabot.enums import OptionType
from deltabot.logging_setup import setup_logging
from deltabot.models import Candle

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _ist_mins(ts: int) -> int:
    d = datetime.fromtimestamp(ts, tz=_IST)
    return d.hour * 60 + d.minute


def _select_expiry(ts: int, cutoff_hour: int, cutoff_minute: int) -> datetime:
    ist = datetime.fromtimestamp(ts, tz=_IST)
    if ist.hour * 60 + ist.minute >= cutoff_hour * 60 + cutoff_minute:
        ist = ist + timedelta(days=1)
    return ist


def run(candles: list[Candle], settings, args, sim_start: int, otm_pcts: list[float]) -> list[dict]:
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    lot_size = args.lot_size if args.lot_size > 0 else op.LOT_SIZE.get(underlying, op.LOT_BTC)
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    bar_seconds = op.RES_SECONDS.get(args.resolution, 60)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor
    entry_mins = args.entry_hour * 60 + args.entry_minute
    exit_mins = args.exit_hour * 60 + args.exit_minute
    target_frac = 1 - args.target_pct / 100.0
    sl_frac = 1 + args.sl_pct / 100.0

    trades: list[dict] = []
    pos: list[dict] | None = None   # list of legs: {"sym","candles","otm_pct","is_call","entry_prem"}
    entry_time = entry_btc = combined_entry = None
    prev_mins: int | None = None

    def combined_prem(ts: int) -> list[float] | None:
        out = []
        for leg in pos:
            p = op.premium_at(leg["candles"], ts, step)
            if p is None:
                return None
            if floor:
                p = max(p, op.intrinsic_value(leg["sym"], entry_btc))
            out.append(p)
        return out

    def close(reason: str, ts: int, exit_prems: list[float], exit_btc: float) -> None:
        nonlocal pos
        gross = 0.0
        fee = 0.0
        for leg, exit_p in zip(pos, exit_prems):
            entry_fill = leg["entry_prem"] * (1 - es)
            exit_fill = exit_p * (1 + xs)
            gross += (entry_fill - exit_fill) * args.lots * lot_size
            fee += (op.side_fee(entry_btc, entry_fill, args.lots, lot_size)
                    + op.side_fee(exit_btc, exit_fill, args.lots, lot_size))
        trades.append({
            "entry_time": entry_time, "exit_time": ts,
            "legs": ", ".join(leg["sym"] for leg in pos),
            "combined_entry": combined_entry, "combined_exit": sum(exit_prems),
            "reason": reason, "gross": gross, "fee": fee, "net": gross - fee,
        })
        pos = None

    def open_ladder(client, ts: int, btc_price: float) -> bool:
        nonlocal pos, entry_time, entry_btc, combined_entry
        expiry = _select_expiry(ts, args.expiry_cutoff_hour, args.expiry_cutoff_minute)
        win_end = ts + 30 * 3600
        legs: list[dict] = []
        for pct in otm_pcts:
            raw_offset = round((btc_price * pct / 100.0) / interval) * interval
            for otype, strike in (
                (OptionType.CALL, int(round((btc_price + raw_offset) / interval) * interval)),
                (OptionType.PUT, int(round((btc_price - raw_offset) / interval) * interval)),
            ):
                r = op.resolve_contract(client, underlying, otype, strike, expiry, interval,
                                        ts, ts, ts - 86400, win_end, args.opt_resolution, step)
                if r is None:
                    return False
                sym, _, cands = r
                prem = op.premium_at(cands, ts, step)
                if prem is None:
                    return False
                if floor:
                    prem = max(prem, op.intrinsic_value(sym, btc_price))
                legs.append({"sym": sym, "candles": cands, "otm_pct": pct,
                            "is_call": otype == OptionType.CALL, "entry_prem": prem})
        pos = legs
        entry_time, entry_btc = ts, btc_price
        combined_entry = sum(leg["entry_prem"] for leg in legs)
        return True

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            mins = _ist_mins(c.start_time)
            entry_crossed = prev_mins is not None and mins >= entry_mins and prev_mins < entry_mins
            exit_crossed = prev_mins is not None and mins >= exit_mins and prev_mins < exit_mins
            prev_mins = mins
            decision_ts = c.start_time + bar_seconds
            dt_ist = datetime.fromtimestamp(decision_ts, tz=_IST)
            is_weekend = dt_ist.weekday() >= 5 and not args.include_weekends

            if exit_crossed and pos is not None:
                res = combined_prem(decision_ts)
                if res is not None:
                    close("EOD", decision_ts, res, c.close)

            if pos is not None:
                res = combined_prem(decision_ts)
                if res is not None:
                    combined = sum(res)
                    if combined <= combined_entry * target_frac:
                        close("TARGET", decision_ts, res, c.close)
                    elif combined >= combined_entry * sl_frac:
                        close("SL", decision_ts, res, c.close)

            if (entry_crossed and pos is None and not is_weekend
                    and c.start_time >= sim_start):
                if not open_ladder(client, decision_ts, c.close):
                    pos = None

    if pos is not None:
        last = candles[-1]
        last_ts = last.start_time + bar_seconds
        res = combined_prem(last_ts)
        if res is not None:
            close("OPEN_AT_END", last_ts, res, last.close)

    return trades


def report(trades: list[dict], args, otm_pcts: list[float]) -> None:
    pcts_str = "/".join(f"{p:g}%" for p in otm_pcts)
    print(f"\n{'=' * 110}")
    print(f"Daily {args.entry_hour:02d}:{args.entry_minute:02d} IST LADDERED Short Strangle -- {args.days}d, "
          f"OTM strikes [{pcts_str}] per side ({len(otm_pcts) * 2} legs), target {args.target_pct:.0f}% combined "
          f"decay, SL {args.sl_pct:.0f}% combined rise, {args.lots} lots")
    print(f"{'=' * 110}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'exit (IST)':<18}{'comb entry':>11}{'comb exit':>11}  {'reason':<10}{'net $':>9}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{_ist(t['exit_time']):<18}{t['combined_entry']:>11.1f}"
              f"{t['combined_exit']:>11.1f}  {t['reason']:<10}{t['net']:>9.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 110}")
    print(f"Strangles: {len(closed)} closed" + (" (+1 still open)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("TARGET", "SL", "EOD", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<10} n={len(rs):<4} net ${sum(t['net'] for t in rs):>11.2f}")
    print(f"TOTAL NET: ${sum(t['net'] for t in trades):.2f} "
          f"(gross ${sum(t['gross'] for t in trades):.2f}, fees ${sum(t['fee'] for t in trades):.2f})")


def main() -> None:
    p = argparse.ArgumentParser(description="Daily 9PM LADDERED Short Strangle backtest")
    p.add_argument("--days", type=float, default=30)
    p.add_argument("--warmup-days", type=int, default=1)
    p.add_argument("--resolution", default="1m")
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--entry-hour", type=int, default=21)
    p.add_argument("--entry-minute", type=int, default=0)
    p.add_argument("--exit-hour", type=int, default=17)
    p.add_argument("--exit-minute", type=int, default=0)
    p.add_argument("--otm-pcts", default="1,1.5,2",
                   help="Comma-separated OTM%% distances -- one CALL+PUT pair sold at EACH.")
    p.add_argument("--target-pct", type=float, default=70.0)
    p.add_argument("--sl-pct", type=float, default=105.0)
    p.add_argument("--expiry-cutoff-hour", type=int, default=17)
    p.add_argument("--expiry-cutoff-minute", type=int, default=26)
    p.add_argument("--lots", type=int, default=10)
    p.add_argument("--include-weekends", action="store_true")
    p.add_argument("--lot-size", type=float, default=0.0)
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true")
    args = p.parse_args()
    otm_pcts = [float(x.strip()) for x in args.otm_pcts.split(",") if x.strip()]

    setup_logging("WARNING")
    settings = load_settings()

    now = int(time.time())
    sim_start = now - int(args.days * 86400)
    dl_start = sim_start - args.warmup_days * 86400

    df = download(symbol=settings.symbol, start=dl_start, end=now, resolution=args.resolution)
    candles = df_to_candles(df)
    print(f"Candles: {len(candles)} ({args.resolution}) {_ist(candles[0].start_time)} .. {_ist(candles[-1].start_time)}")

    trades = run(candles, settings, args, sim_start, otm_pcts)
    report(trades, args, otm_pcts)


if __name__ == "__main__":
    main()
