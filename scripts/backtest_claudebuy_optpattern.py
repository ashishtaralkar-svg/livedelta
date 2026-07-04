"""claudebuy (option-pattern variant).

Pattern is detected on the OPTION's own candles; BTC vs the 17:30 open only chooses
which side (CE/PE) to hunt.

RULES
  * Session starts 17:30 IST (record BTC open then); ends / force-exit 17:25 next day.
  * Direction each bar while flat:
      BTC close > 17:30 open  -> hunt a CALL (CE)
      BTC close < 17:30 open  -> hunt a PUT  (PE)
  * Pick the CE/PE strike whose premium is ~ --target-premium (default 325, i.e. 300-350).
  * On THAT option's candles look for red->green (prev close<open, cur close>open).
  * Arm: entry = max(opt.high, opt.high[-1]); SL = min(opt.low, opt.low[-1]).
  * BUY when the option trades above the pattern high; TARGET = 2x buy price;
    STOP = pattern lower-low. Invalidate if the low breaks the stop before entry.
  * On SL or TP, resume scanning. One position at a time.

P&L (long option) = (exit_prem - buy_prem) * lots * 0.001 - brokerage(2 sides).

Run: python scripts/backtest_claudebuy_optpattern.py [--days 4] [--target-premium 325]
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from deltabot.backtest import option_pricing as op
from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.config import load_settings
from deltabot.enums import OptionType
from deltabot.models import Candle

_IST = ZoneInfo("Asia/Kolkata")
SESSION_START = (17, 30)
SESSION_END = (17, 25)


def _ist(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=_IST)


def in_session(ts: int) -> bool:
    t = _ist(ts)
    after = (t.hour > SESSION_START[0]) or (t.hour == SESSION_START[0] and t.minute >= SESSION_START[1])
    before = (t.hour < SESSION_END[0]) or (t.hour == SESSION_END[0] and t.minute <= SESSION_END[1])
    return after or before


def candle_at(candles: dict[int, Candle], ts: int, step: int) -> Candle | None:
    """Option candle at/just-before ts (search back a few bars over gaps)."""
    for k in range(0, 12):
        c = candles.get(ts - k * step)
        if c is not None:
            return c
    earlier = [c for t, c in candles.items() if t <= ts]
    return max(earlier, key=lambda c: c.start_time) if earlier else None


def resolve_near_premium(client, underlying, otype, spot, expiry, interval, target,
                         ts, step, full_start, full_end, resolution, cache):
    """Strike whose premium at ts is closest to target. Fetches each candidate's
    FULL-window candles once (cached by symbol). Returns (symbol, candles, prem)."""
    ddmmyy = expiry.strftime("%d%m%y")
    atm = int(round(spot / interval) * interval)
    otm = interval if otype == OptionType.CALL else -interval  # toward OTM = cheaper
    best = None

    def ev(strike):
        sym = f"{otype.value}-{underlying}-{strike}-{ddmmyy}"
        if sym not in cache:
            cache[sym] = op.fetch_option_candles(client, sym, full_start, full_end, resolution)
        cands = cache[sym]
        if not cands:
            return None
        p = op.premium_at(cands, ts, step)
        return (sym, cands, p) if p is not None else None

    strike = atm
    for _ in range(30):
        r = ev(strike)
        if r is not None:
            diff = abs(r[2] - target)
            if best is None or diff < best[0]:
                best = (diff, r[0], r[1], r[2])
            if r[2] <= target:
                break
        strike += otm
    strike = atm - otm
    for _ in range(6):
        r = ev(strike)
        if r is not None:
            diff = abs(r[2] - target)
            if best is None or diff < best[0]:
                best = (diff, r[0], r[1], r[2])
        strike -= otm
    return (best[1], best[2], best[3]) if best else None


def run(candles, settings, args):
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots if args.lots is not None else settings.option_contracts
    tp_mult = 1.0 + args.take_profit / 100.0  # default 100 -> 2x
    step = op.RES_SECONDS.get(args.opt_resolution, 300)
    win_start = int(time.time()) - int(args.days * 86400)
    full_start = candles[0].start_time
    full_end = candles[-1].start_time + step
    cache: dict = {}
    trips: list[dict] = []

    session_open = None       # BTC open at 17:30
    armed = in_trade = False
    lock_sym = ""
    lock_candles: dict = {}
    entry_level = sl_level = buy_prem = tp = 0.0
    otype = None
    entry_ts = pattern_ts = 0

    def close_trade(reason, exit_prem, exit_ts, exit_btc):
        nonlocal armed, in_trade
        gross = (exit_prem - buy_prem) * lots * op.LOT_BTC
        fee = op.side_fee(exit_btc, buy_prem, lots) + op.side_fee(exit_btc, exit_prem, lots)
        trips.append({
            "side": "CE" if otype == OptionType.CALL else "PE",
            "contract": lock_sym,
            "entry_ist": _ist(entry_ts).strftime("%Y-%m-%d %H:%M IST"),
            "exit_ist": _ist(exit_ts).strftime("%Y-%m-%d %H:%M IST"),
            "reason": reason, "buy_prem": round(buy_prem, 1),
            "sl_prem": round(sl_level, 1), "tp_prem": round(tp, 1),
            "exit_prem": round(exit_prem, 1), "lots": lots,
            "gross_usd": round(gross, 2), "fee_usd": round(fee, 2),
            "net_usd": round(gross - fee, 2),
        })
        armed = in_trade = False

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for i in range(1, len(candles)):
            ts = candles[i].start_time
            it = in_session(ts)
            t = _ist(ts)

            # Session start = the 17:30 candle (the 17:26-17:29 gap has no 5m bar,
            # so an in-session transition never fires). Record BTC's 17:30 open.
            if t.hour == SESSION_START[0] and t.minute == SESSION_START[1]:
                session_open = candles[i].open
                if not in_trade:
                    armed = False  # a fresh session cancels a dangling armed setup

            if not it or session_open is None:
                continue

            # 4. Auto-exit at 17:25.
            if t.hour == SESSION_END[0] and t.minute == SESSION_END[1]:
                if in_trade:
                    c = candle_at(lock_candles, ts, step)
                    close_trade("EOD", c.close if c else buy_prem, ts, candles[i].close)
                armed = in_trade = False
                continue

            # 3. Manage an open trade on the locked option.
            if in_trade:
                c = candle_at(lock_candles, ts, step)
                if c is not None:
                    if c.high >= tp:
                        close_trade("TP", tp, ts, candles[i].close)
                    elif c.low <= sl_level:
                        close_trade("SL", sl_level, ts, candles[i].close)
                continue

            # 2. Armed: wait for the option to break the pattern high (or invalidate).
            if armed:
                c = candle_at(lock_candles, ts, step)
                if c is not None:
                    if c.low <= sl_level:            # stop broke before entry
                        armed = False
                    elif c.high > entry_level:        # breakout -> BUY
                        in_trade, armed = True, False
                        buy_prem = entry_level
                        tp = buy_prem * tp_mult
                        entry_ts = ts
                # if still armed, keep waiting (do not scan for a new pattern)
                continue

            # 1. Flat: choose side from BTC vs 17:30 open, find ~target-prem strike,
            #    look for red->green on that option.
            direction_call = candles[i].close > session_open
            ot = OptionType.CALL if direction_call else OptionType.PUT
            expiry = op.select_expiry_date(ts, cutoff)
            resolved = resolve_near_premium(client, underlying, ot, candles[i].close, expiry,
                                            interval, args.target_premium, ts, step,
                                            full_start, full_end, args.opt_resolution, cache)
            if resolved is None:
                continue
            sym, cands, prem = resolved
            if not (args.prem_lo <= prem <= args.prem_hi):
                continue  # option not in the 300-350 zone right now
            cur = candle_at(cands, ts, step)
            prev = candle_at(cands, ts - step, step)
            if cur is None or prev is None:
                continue
            red_then_green = prev.close < prev.open and cur.close > cur.open
            if red_then_green:
                armed = True
                otype = ot
                lock_sym, lock_candles = sym, cands
                entry_level = max(cur.high, prev.high)
                sl_level = min(cur.low, prev.low)
                pattern_ts = ts

    return [t for t in trips if int(datetime.strptime(
        t["entry_ist"].replace(" IST", ""), "%Y-%m-%d %H:%M")
        .replace(tzinfo=_IST).timestamp()) >= win_start]


def main() -> None:
    ap = argparse.ArgumentParser(description="claudebuy option-pattern backtest")
    ap.add_argument("--days", type=float, default=4)
    ap.add_argument("--resolution", default="5m")
    ap.add_argument("--warmup-days", type=int, default=1)
    ap.add_argument("--target-premium", type=float, default=325.0)
    ap.add_argument("--prem-lo", type=float, default=300.0)
    ap.add_argument("--prem-hi", type=float, default=350.0)
    ap.add_argument("--take-profit", type=float, default=100.0, help="%% gain target (100 = 2x)")
    ap.add_argument("--opt-resolution", default="5m")
    ap.add_argument("--lots", type=int, default=None)
    ap.add_argument("--excel", default="claudebuy_optpattern.xlsx")
    args = ap.parse_args()

    s = load_settings()
    now = int(time.time())
    win_start = now - int(args.days * 86400)
    df = download(symbol=s.symbol, start=win_start - args.warmup_days * 86400,
                  end=now, resolution=args.resolution, base_url=s.rest_base_url)
    candles = df_to_candles(df)
    trips = run(candles, s, args)
    lots = args.lots if args.lots is not None else s.option_contracts

    print(f"\n{s.symbol}  {args.resolution}  claudebuy OPT-PATTERN  prem {args.prem_lo:.0f}-{args.prem_hi:.0f}, "
          f"+{args.take_profit:.0f}% target, SL=pattern low, lots {lots}")
    print(f"Window {_ist(win_start):%Y-%m-%d %H:%M} -> {_ist(now):%Y-%m-%d %H:%M} IST")
    print("=" * 104)
    print(f"{'Entry(IST)':<18}{'Exit(IST)':<18}{'Side':<5}{'Contract':<22}{'Why':<6}"
          f"{'Buy':>7}{'Exit':>8}{'Net$':>9}")
    print("-" * 104)
    net = fees = gross = 0.0
    wins = 0
    for tr in trips:
        print(f"{tr['entry_ist'][5:16]:<18}{tr['exit_ist'][5:16]:<18}{tr['side']:<5}"
              f"{tr['contract']:<22}{tr['reason']:<6}{tr['buy_prem']:>7.1f}{tr['exit_prem']:>8.1f}{tr['net_usd']:>9.2f}")
        net += tr["net_usd"]; fees += tr["fee_usd"]; gross += tr["gross_usd"]
        wins += 1 if tr["net_usd"] > 0 else 0
    print("=" * 104)
    n = len(trips)
    wr = (wins / n * 100) if n else 0
    reasons = {r: sum(1 for t in trips if t["reason"] == r) for r in ("TP", "SL", "EOD")}
    print(f"Trades {n}  Wins/Losses {wins}/{n - wins}  Win rate {wr:.1f}%  "
          f"exits: TP={reasons['TP']} SL={reasons['SL']} EOD={reasons['EOD']}")
    print(f"Gross {gross:,.2f}  -  brokerage {fees:,.2f}  =  NET {net:,.2f} USD")

    with pd.ExcelWriter(Path(args.excel), engine="openpyxl") as xl:
        pd.DataFrame(trips).to_excel(xl, sheet_name="Trades", index=False)
        pd.DataFrame([
            {"metric": "window_IST", "value": f"{_ist(win_start):%Y-%m-%d %H:%M} -> {_ist(now):%Y-%m-%d %H:%M}"},
            {"metric": "premium_zone", "value": f"{args.prem_lo:.0f}-{args.prem_hi:.0f}"},
            {"metric": "target", "value": f"+{args.take_profit:.0f}%"}, {"metric": "lots", "value": lots},
            {"metric": "trades", "value": n}, {"metric": "win_rate_pct", "value": round(wr, 1)},
            {"metric": "exits_TP/SL/EOD", "value": f"{reasons['TP']}/{reasons['SL']}/{reasons['EOD']}"},
            {"metric": "gross_usd", "value": round(gross, 2)},
            {"metric": "brokerage_usd", "value": round(fees, 2)},
            {"metric": "net_usd", "value": round(net, 2)},
        ]).to_excel(xl, sheet_name="Summary", index=False)
    print(f"\nExcel written to {Path(args.excel).resolve()}")


if __name__ == "__main__":
    main()
