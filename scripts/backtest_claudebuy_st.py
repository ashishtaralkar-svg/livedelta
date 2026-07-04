"""claudebuy (Supertrend-gated option-pattern, CALL-only, buying only).

BTC REGIME GATE (all must hold to hunt):
  * Supertrend(10,3) is GREEN (uptrend), and
  * EMA50 > SMA50 > EMA200 (bullish stack).

TRADE (option-pattern, calls only):
  * Pick a CALL whose premium is in [--prem-lo, --prem-hi] (300-500).
  * On THAT option's candles find red->green (prev close<open, cur close>open).
  * Arm: entry = max(opt.high, opt.high[-1]); SL = min(opt.low, opt.low[-1]).
  * BUY when the option trades above the pattern high; TARGET = 2x buy price;
    STOP = pattern lower-low. Invalidate if the low breaks the stop before entry.
  * One position at a time; force-exit 17:25 IST (daily expiry protection).

P&L (long option) = (exit_prem - buy_prem) * lots * 0.001 - brokerage(2 sides).

Run: python scripts/backtest_claudebuy_st.py [--days 4] [--prem-lo 300 --prem-hi 500]
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

from backtest_claudebuy_optpattern import candle_at, resolve_near_premium

_IST = ZoneInfo("Asia/Kolkata")
SESSION_END = (17, 25)  # daily option settlement / force-exit


def _ist(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=_IST)


def ema(values, length):
    a = 2.0 / (length + 1.0)
    out, prev = [], values[0]
    for i, v in enumerate(values):
        prev = v if i == 0 else a * v + (1 - a) * prev
        out.append(prev)
    return out


def sma(values, length):
    out, s = [], 0.0
    from collections import deque
    q = deque()
    for v in values:
        q.append(v); s += v
        if len(q) > length:
            s -= q.popleft()
        out.append(s / len(q))
    return out


def supertrend(high, low, close, period=10, mult=3.0):
    """Return a list of booleans: True = GREEN (uptrend)."""
    n = len(close)
    tr = [0.0] * n
    for i in range(n):
        tr[i] = high[i] - low[i] if i == 0 else max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = [0.0] * n
    for i in range(n):
        if i < period:
            atr[i] = sum(tr[:i + 1]) / (i + 1)
        else:
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period  # Wilder RMA
    hl2 = [(high[i] + low[i]) / 2 for i in range(n)]
    upper = [hl2[i] + mult * atr[i] for i in range(n)]
    lower = [hl2[i] - mult * atr[i] for i in range(n)]
    fu = [0.0] * n
    fl = [0.0] * n
    up = [True] * n
    for i in range(n):
        if i == 0:
            fu[i], fl[i], up[i] = upper[i], lower[i], True
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if close[i] > fu[i - 1]:
            up[i] = True
        elif close[i] < fl[i - 1]:
            up[i] = False
        else:
            up[i] = up[i - 1]
    return up


def run(candles, settings, args):
    o = [c.open for c in candles]
    h = [c.high for c in candles]
    lo = [c.low for c in candles]
    cl = [c.close for c in candles]
    ema50, sma50, ema200 = ema(cl, 50), sma(cl, 50), ema(cl, 200)
    st_green = supertrend(h, lo, cl, args.atr_period, args.st_mult)

    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots if args.lots is not None else settings.option_contracts
    tp_mult = 1.0 + args.take_profit / 100.0
    step = op.RES_SECONDS.get(args.opt_resolution, 300)
    win_start = int(time.time()) - int(args.days * 86400)
    full_start = candles[0].start_time
    full_end = candles[-1].start_time + step
    cache: dict = {}
    trips: list[dict] = []

    armed = in_trade = False
    lock_sym = ""
    lock_candles: dict = {}
    entry_level = sl_level = buy_prem = tp = 0.0
    entry_ts = 0

    def close_trade(reason, exit_prem, exit_ts, exit_btc):
        nonlocal armed, in_trade
        gross = (exit_prem - buy_prem) * lots * op.LOT_BTC
        fee = op.side_fee(exit_btc, buy_prem, lots) + op.side_fee(exit_btc, exit_prem, lots)
        trips.append({
            "side": "CE", "contract": lock_sym,
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
            t = _ist(ts)

            # Force-exit at 17:25 (option expiry).
            if t.hour == SESSION_END[0] and t.minute == SESSION_END[1]:
                if in_trade:
                    c = candle_at(lock_candles, ts, step)
                    close_trade("EOD", c.close if c else buy_prem, ts, cl[i])
                armed = in_trade = False
                continue

            # Manage open trade.
            if in_trade:
                c = candle_at(lock_candles, ts, step)
                if c is not None:
                    if c.high >= tp:
                        close_trade("TP", tp, ts, cl[i])
                    elif c.low <= sl_level:
                        close_trade("SL", sl_level, ts, cl[i])
                continue

            # Armed: wait for the option breakout (or invalidate).
            if armed:
                c = candle_at(lock_candles, ts, step)
                if c is not None:
                    if c.low <= sl_level:
                        armed = False
                    elif c.high > entry_level:
                        in_trade, armed = True, False
                        buy_prem = entry_level
                        tp = buy_prem * tp_mult
                        entry_ts = ts
                continue

            # Flat: BTC regime gate must hold (supertrend green + EMA stack).
            if not (st_green[i] and ema50[i] > sma50[i] and sma50[i] > ema200[i]):
                continue

            # Find a CALL priced in the zone, look for red->green on it.
            expiry = op.select_expiry_date(ts, cutoff)
            resolved = resolve_near_premium(client, underlying, OptionType.CALL, cl[i], expiry,
                                            interval, args.target_premium, ts, step,
                                            full_start, full_end, args.opt_resolution, cache)
            if resolved is None:
                continue
            sym, cands, prem = resolved
            if not (args.prem_lo <= prem <= args.prem_hi):
                continue
            cur = candle_at(cands, ts, step)
            prev = candle_at(cands, ts - step, step)
            if cur is None or prev is None:
                continue
            if prev.close < prev.open and cur.close > cur.open:  # red -> green
                armed = True
                lock_sym, lock_candles = sym, cands
                entry_level = max(cur.high, prev.high)
                sl_level = min(cur.low, prev.low)

    return [t for t in trips if int(datetime.strptime(
        t["entry_ist"].replace(" IST", ""), "%Y-%m-%d %H:%M")
        .replace(tzinfo=_IST).timestamp()) >= win_start]


def main() -> None:
    ap = argparse.ArgumentParser(description="claudebuy supertrend-gated CALL-buy backtest")
    ap.add_argument("--days", type=float, default=4)
    ap.add_argument("--resolution", default="5m")
    ap.add_argument("--warmup-days", type=int, default=3)
    ap.add_argument("--atr-period", type=int, default=10)
    ap.add_argument("--st-mult", type=float, default=3.0)
    ap.add_argument("--target-premium", type=float, default=400.0)
    ap.add_argument("--prem-lo", type=float, default=300.0)
    ap.add_argument("--prem-hi", type=float, default=500.0)
    ap.add_argument("--take-profit", type=float, default=100.0, help="%% gain target (100 = 2x)")
    ap.add_argument("--opt-resolution", default="5m")
    ap.add_argument("--lots", type=int, default=None)
    ap.add_argument("--excel", default="claudebuy_st.xlsx")
    args = ap.parse_args()

    s = load_settings()
    now = int(time.time())
    win_start = now - int(args.days * 86400)
    df = download(symbol=s.symbol, start=win_start - args.warmup_days * 86400,
                  end=now, resolution=args.resolution, base_url=s.rest_base_url)
    candles = df_to_candles(df)
    trips = run(candles, s, args)
    lots = args.lots if args.lots is not None else s.option_contracts

    print(f"\n{s.symbol}  {args.resolution}  claudebuy ST-GATE CALL-buy  prem {args.prem_lo:.0f}-{args.prem_hi:.0f}, "
          f"+{args.take_profit:.0f}% target, lots {lots}")
    print(f"Gate: Supertrend({args.atr_period},{args.st_mult:g}) GREEN & EMA50>SMA50>EMA200")
    print(f"Window {_ist(win_start):%Y-%m-%d %H:%M} -> {_ist(now):%Y-%m-%d %H:%M} IST")
    print("=" * 104)
    print(f"{'Entry(IST)':<18}{'Exit(IST)':<18}{'Contract':<24}{'Why':<6}{'Buy':>7}{'Exit':>8}{'Net$':>9}")
    print("-" * 104)
    net = fees = gross = 0.0
    wins = 0
    for tr in trips:
        print(f"{tr['entry_ist'][5:16]:<18}{tr['exit_ist'][5:16]:<18}{tr['contract']:<24}"
              f"{tr['reason']:<6}{tr['buy_prem']:>7.1f}{tr['exit_prem']:>8.1f}{tr['net_usd']:>9.2f}")
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
            {"metric": "gate", "value": f"ST({args.atr_period},{args.st_mult:g}) green & EMA50>SMA50>EMA200"},
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
