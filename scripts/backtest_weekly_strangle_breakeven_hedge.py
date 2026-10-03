"""Backtest: Weekly short strangle with breakeven-triggered long hedge.

Every Friday at --entry-hour:--entry-minute IST (default 21:00):
  * SELL a CALL ~--otm-pct% (default 2%) above spot and a PUT ~--otm-pct% below spot,
    both on NEXT Friday's weekly expiry (17:30 IST).
  * Combined premium P = CE premium + PE premium. Breakevens:
        upper = CE strike + P,  lower = PE strike - P
  * If a 1m BTC candle CLOSES above `upper`, BUY a CALL at strike ~`upper` (same weekly
    expiry, same lots). If it closes below `lower`, BUY a PUT at strike ~`lower`.
    Each side hedges at most once per week.
  * Everything is closed at --exit-hour:--exit-minute IST on expiry Friday (default 17:25).
No target / SL. The report also shows the same week WITHOUT the hedge for comparison.

Run: python scripts/backtest_weekly_strangle_breakeven_hedge.py --days 90 --lots 15
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
_FRIDAY = 4


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%m-%d %H:%M")


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    lot_size = op.LOT_SIZE.get(underlying, op.LOT_BTC)
    step = op.RES_SECONDS.get(args.opt_resolution, 900)
    bar = op.RES_SECONDS.get(args.resolution, 60)
    floor = not args.no_intrinsic_floor
    weeks: list[dict] = []
    pos: dict | None = None

    def snap(x: float) -> int:
        return int(round(x / interval) * interval)

    def price(leg: dict, ts: int, btc: float) -> float:
        p = op.premium_at(leg["candles"], ts, step)
        iv = op.intrinsic_value(leg["sym"], btc)
        if p is None:
            return iv
        return max(p, iv) if floor else p

    def resolve(client, otype, strike, ts) -> dict | None:
        r = op.resolve_contract(client, underlying, otype, strike, pos_expiry_dt(ts), interval,
                                ts, ts, ts - 86400, pos["expiry_ts"] + 3600, args.opt_resolution, step) \
            if pos else None
        if r is None:
            return None
        sym, k, cs = r
        p0 = op.premium_at(cs, ts, step)
        return None if p0 is None else {"sym": sym, "strike": k, "candles": cs, "entry": p0}

    def pos_expiry_dt(_ts: int) -> datetime:
        return pos["expiry_dt"]

    def leg_pnl(leg: dict, exit_p: float, exit_btc: float, entry_btc: float) -> float:
        sign = -1 if leg["sell"] else 1
        g = sign * (exit_p - leg["entry"]) * leg["lots"] * lot_size
        f = (op.side_fee(entry_btc, leg["entry"], leg["lots"], lot_size)
             + op.side_fee(exit_btc, exit_p, leg["lots"], lot_size))
        return g - f

    def close_leg(leg: dict, ts: int, btc: float) -> None:
        leg["exit"] = price(leg, ts, btc)
        leg["exit_btc"] = btc
        leg["exit_time"] = ts

    def close(ts: int, btc: float, reason: str) -> None:
        nonlocal pos
        sold = hedge = 0.0
        for leg in pos["legs"]:
            if "exit" not in leg:
                close_leg(leg, ts, btc)
            pnl = leg_pnl(leg, leg["exit"], leg["exit_btc"], leg["entry_btc"])
            if leg["sell"]:
                sold += pnl
            else:
                hedge += pnl
        pos.update({"exit_ts": ts, "exit_btc": btc, "reason": reason,
                    "sold_net": sold, "hedge_net": hedge, "net": sold + hedge})
        weeks.append(pos)
        pos = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        prev_key = None
        for c in candles:
            ts = c.start_time + bar
            d = datetime.fromtimestamp(ts, tz=_IST)
            key = (d.date(), d.hour * 60 + d.minute)

            if pos is not None:
                sold_now = sum(price(leg, ts, c.close) for leg in pos["legs"][:2])
                if ts >= pos["close_ts"]:
                    close(ts, c.close, "EXPIRY")
                elif args.target_pct > 0 and sold_now <= pos["combined"] * (1 - args.target_pct / 100.0):
                    close(ts, c.close, "TARGET")
                elif args.sl_pct > 0 and sold_now >= pos["combined"] * (1 + args.sl_pct / 100.0):
                    close(ts, c.close, "SL")
                elif not args.no_hedge:
                    buf = c.close * args.exit_buffer_pct / 100.0
                    for side, otype, hit, back_inside, lvl in (
                        ("CE", OptionType.CALL, c.close > pos["upper"], c.close < pos["upper"] - buf, pos["upper"]),
                        ("PE", OptionType.PUT, c.close < pos["lower"], c.close > pos["lower"] + buf, pos["lower"]),
                    ):
                        active = pos["hedged"].get(side)
                        if args.exit_hedge_inside and active is not None and back_inside:
                            close_leg(active, ts, c.close)
                            del pos["hedged"][side]
                            continue
                        if hit and side not in pos["hedged"] and (args.exit_hedge_inside or side not in pos["ever"]):
                            off = (pos["entry_btc"] * args.hedge_offset_pct / 100.0
                                   if args.hedge_offset_pct > 0 else args.hedge_offset)
                            strike = lvl + off if side == "CE" else lvl - off
                            h = resolve(client, otype, snap(strike), ts)
                            if h is None:
                                continue
                            if floor:
                                h["entry"] = max(h["entry"], op.intrinsic_value(h["sym"], c.close))
                            h.update({"side": side, "sell": False, "lots": args.lots,
                                      "entry_btc": c.close, "time": ts})
                            pos["legs"].append(h)
                            pos["hedged"][side] = h
                            pos["ever"].add(side)

            entry_min = args.entry_hour * 60 + args.entry_minute
            is_cycle_day = args.daily_entry or args.daily_expiry or (d.weekday() == _FRIDAY and (
                not args.monthly or (d + timedelta(days=7)).month != d.month))
            if (pos is None and is_cycle_day and key[1] >= entry_min
                    and (prev_key is None or prev_key[0] != key[0] or prev_key[1] < entry_min)
                    and c.start_time >= sim_start):
                btc = c.close
                exp = (d + timedelta(days=1) if args.daily_expiry
                       else d + timedelta(days=(_FRIDAY - d.weekday()) % 7 or 7))
                if args.monthly:
                    while (exp + timedelta(days=7)).month == exp.month:
                        exp += timedelta(days=7)
                expiry_dt = exp.replace(hour=17, minute=30, second=0, microsecond=0)
                pos = {"entry_ts": ts, "entry_btc": btc, "expiry_dt": expiry_dt,
                       "expiry_ts": int(expiry_dt.timestamp()),
                       "close_ts": int(expiry_dt.replace(hour=args.exit_hour, minute=args.exit_minute).timestamp()),
                       "legs": [], "hedged": {}, "ever": set()}
                off = btc * args.otm_pct / 100.0
                ce = resolve(client, OptionType.CALL, snap(btc + off), ts)
                pe = resolve(client, OptionType.PUT, snap(btc - off), ts)
                if ce is None or pe is None:
                    print(f"NOTE: skipping week of {_ist(ts)} -- weekly contract data unavailable")
                    pos = None
                else:
                    for leg, side in ((ce, "CE"), (pe, "PE")):
                        leg.update({"side": side, "sell": True, "lots": args.lots, "entry_btc": btc})
                    pos["legs"] = [ce, pe]
                    combined = ce["entry"] + pe["entry"]
                    pos.update({"combined": combined, "upper": ce["strike"] + combined,
                                "lower": pe["strike"] - combined})
            prev_key = key

        if pos is not None:
            last = candles[-1]
            close(last.start_time + bar, last.close, "OPEN_AT_END")

    return weeks


def report(weeks: list[dict], args) -> None:
    w = 150
    print(f"\n{'=' * w}")
    cycle = "MONTHLY (last Fri -> next month's last-Fri expiry)" if args.monthly else "Weekly (Fri -> next-Fri expiry)"
    print(f"{cycle} short strangle (entry {args.entry_hour:02d}:{args.entry_minute:02d} IST, "
          f"~{args.otm_pct:g}% OTM) + breakeven-triggered long hedge -- {args.days}d, {args.lots} lots, "
          f"close expiry Fri {args.exit_hour:02d}:{args.exit_minute:02d}")
    print(f"{'=' * w}")
    if not weeks:
        print("No weeks traded.")
        return
    print(f"{'entry':<12}{'BTC in->out':<17}{'sold CE / PE (prem)':<36}{'comb':>6}{'  breakevens':<18}"
          f"{'hedge bought':<34}{'no-hedge $':>11}{'hedge $':>9}{'net $':>9}")
    eq = pk = dd = 0.0
    for wk in weeks:
        ce, pe = wk["legs"][0], wk["legs"][1]
        sold_txt = f"{ce['strike']}C {ce['entry']:.0f}->{ce['exit']:.0f} / {pe['strike']}P {pe['entry']:.0f}->{pe['exit']:.0f}"
        hl = [leg for leg in wk["legs"] if not leg["sell"]]
        if len(hl) == 1:
            h = hl[0]
            hedges = [f"{h['strike']}{'C' if h['side'] == 'CE' else 'P'} @{h['entry']:.0f}->{h['exit']:.0f} {_ist(h['time'])}"]
        else:
            hedges = [f"{n}x {s}" for s in ("CE", "PE") if (n := sum(1 for h in hl if h["side"] == s))]
        print(f"{_ist(wk['entry_ts']):<12}{wk['entry_btc']:>7.0f}->{wk['exit_btc']:<8.0f}{sold_txt:<36}"
              f"{wk['combined']:>6.0f}  {wk['lower']:.0f}/{wk['upper']:<10.0f}{(', '.join(hedges) or '-'):<34}"
              f"{wk['sold_net']:>11.2f}{wk['hedge_net']:>9.2f}{wk['net']:>9.2f}"
              + f"  {wk['reason']} {_ist(wk['exit_ts'])}")
        eq += wk["net"]
        pk = max(pk, eq)
        dd = max(dd, pk - eq)
    n = len(weeks)
    wins = sum(1 for wk in weeks if wk["net"] > 0)
    hedged = sum(1 for wk in weeks if wk["ever"])
    n_hedges = sum(1 for wk in weeks for leg in wk["legs"] if not leg["sell"])
    print(f"{'-' * w}")
    print(f"Weeks: {n}   Win rate: {wins}/{n} = {100.0 * wins / n:.0f}%   Weeks with a hedge: {hedged}   "
          f"Hedge buys: {n_hedges}")
    for reason in ("TARGET", "SL", "EXPIRY", "OPEN_AT_END"):
        rs = [wk for wk in weeks if wk["reason"] == reason]
        if rs:
            print(f"  {reason:<12} n={len(rs):<4} net ${sum(wk['net'] for wk in rs):>10.2f}")
    print(f"WITHOUT hedge (plain weekly strangle): ${sum(wk['sold_net'] for wk in weeks):.2f}")
    print(f"Hedge contribution:                    ${sum(wk['hedge_net'] for wk in weeks):.2f}")
    print(f"Worst week ${min(wk['net'] for wk in weeks):.2f}   Best week ${max(wk['net'] for wk in weeks):.2f}   "
          f"Max drawdown ${dd:.2f}")
    print(f"TOTAL NET (with hedge, after fees): ${sum(wk['net'] for wk in weeks):.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Weekly strangle + breakeven hedge backtest")
    p.add_argument("--days", type=float, default=90)
    p.add_argument("--resolution", default="1m")
    p.add_argument("--opt-resolution", default="15m")
    p.add_argument("--entry-hour", type=int, default=21)
    p.add_argument("--entry-minute", type=int, default=0)
    p.add_argument("--exit-hour", type=int, default=17)
    p.add_argument("--exit-minute", type=int, default=25)
    p.add_argument("--otm-pct", type=float, default=2.0)
    p.add_argument("--lots", type=int, default=15)
    p.add_argument("--no-intrinsic-floor", action="store_true")
    p.add_argument("--target-pct", type=float, default=0.0,
                   help="Close everything once the sold CE+PE premium has decayed this %% (0 = off).")
    p.add_argument("--sl-pct", type=float, default=0.0,
                   help="Close everything once the sold CE+PE premium has risen this %% (0 = off).")
    p.add_argument("--no-hedge", action="store_true")
    p.add_argument("--daily-entry", action="store_true",
                   help="Enter at the entry time on ANY day when flat, selling the coming Friday's weekly expiry.")
    p.add_argument("--monthly", action="store_true",
                   help="Enter on the last Friday of each month and sell next month's last-Friday expiry.")
    p.add_argument("--daily-expiry", action="store_true",
                   help="Enter EVERY day and sell the NEXT-DAY expiry (closes 17:25 next day).")
    p.add_argument("--hedge-offset-pct", type=float, default=0.0,
                   help="Hedge strike this %% of entry BTC beyond the breakeven (overrides --hedge-offset).")
    p.add_argument("--hedge-offset", type=float, default=0.0,
                   help="Buy the hedge this many points FURTHER out than the breakeven (trigger unchanged).")
    p.add_argument("--exit-hedge-inside", action="store_true",
                   help="Sell the hedge once BTC closes back inside the breakeven; re-buy on a new breakout.")
    p.add_argument("--exit-buffer-pct", type=float, default=0.0,
                   help="With --exit-hedge-inside: BTC must be this %% back inside the breakeven to sell the hedge.")
    args = p.parse_args()

    setup_logging("WARNING")
    settings = load_settings()
    now = int(time.time())
    sim_start = now - int(args.days * 86400)
    df = download(symbol=settings.symbol, start=sim_start - 86400, end=now, resolution=args.resolution)
    candles = df_to_candles(df)
    print(f"Candles: {len(candles)} ({args.resolution}) {_ist(candles[0].start_time)} .. {_ist(candles[-1].start_time)}")
    report(run(candles, settings, args, sim_start), args)


if __name__ == "__main__":
    main()
