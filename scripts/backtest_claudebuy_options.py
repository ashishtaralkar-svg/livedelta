"""claudebuy (option-buying variant).

Same directional signal as claudebuy (EMA10/20/200 breakout, session 17:30->17:25),
but each signal is executed by BUYING an option instead of trading BTC directly:

  * LONG signal  -> buy a CALL with premium ~ --target-premium (default 500)
  * SHORT signal -> buy a PUT  with premium ~ --target-premium
  * TAKE-PROFIT  : option premium reaches 2x the buy price (i.e. +100%)
  * STOP-LOSS    : the SYSTEM stop on BTC (pattern stop, trailed to EMA200,
                   or a close beyond EMA200) -> sell the option at its premium then
  * EOD          : force-close at 17:25 at the option's premium

P&L per trade (long option, 2 fills) =
  (exit_premium - entry_premium) * lots * 0.001  -  brokerage(entry)+brokerage(exit)

Run: python scripts/backtest_claudebuy_options.py [--days 4] [--target-premium 500] [--lots 50]
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

_IST = ZoneInfo("Asia/Kolkata")
SESSION_START = (17, 30)
SESSION_END = (17, 25)


def _ist(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=_IST)


def ema(values: list[float], length: int) -> list[float]:
    a = 2.0 / (length + 1.0)
    out: list[float] = []
    prev = values[0]
    for i, v in enumerate(values):
        prev = v if i == 0 else a * v + (1 - a) * prev
        out.append(prev)
    return out


def in_session(ts: int) -> bool:
    t = _ist(ts)
    after = (t.hour > SESSION_START[0]) or (t.hour == SESSION_START[0] and t.minute >= SESSION_START[1])
    before = (t.hour < SESSION_END[0]) or (t.hour == SESSION_END[0] and t.minute <= SESSION_END[1])
    return after or before


def opt_high_in(ocandles: dict, after: int, upto: int, step: int) -> float | None:
    """Max option high across option candles in (after, upto]."""
    highs = [c.high for t, c in ocandles.items() if after < t <= upto]
    return max(highs) if highs else None


def run(candles, settings, args):
    o = [c.open for c in candles]
    h = [c.high for c in candles]
    lo = [c.low for c in candles]
    cl = [c.close for c in candles]
    ema10, ema20, ema200 = ema(cl, 10), ema(cl, 20), ema(cl, 200)

    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots if args.lots is not None else settings.option_contracts
    tp_mult = 1.0 + args.take_profit / 100.0  # e.g. 50 -> exit at 1.5x the buy premium
    step = op.RES_SECONDS.get(args.opt_resolution, 300)
    win_start = int(time.time()) - int(args.days * 86400)
    cache: dict = {}
    trips: list[dict] = []

    active = triggered = False
    entry_level = sl_level = 0.0
    direction = ""
    entry_ts = 0
    pos: dict | None = None  # {sym, candles, entry_prem, tp, last_check}

    def close_opt(reason: str, exit_prem: float, exit_ts: int, exit_btc: float):
        nonlocal active, triggered, pos
        assert pos is not None
        gross = (exit_prem - pos["entry_prem"]) * lots * op.LOT_BTC
        fee = op.side_fee(pos["entry_btc"], pos["entry_prem"], lots) + op.side_fee(exit_btc, exit_prem, lots)
        trips.append({
            "direction": direction,
            "action": "BUY " + ("CALL" if pos["sym"].startswith("C-") else "PUT"),
            "contract": pos["sym"],
            "entry_ist": _ist(entry_ts).strftime("%Y-%m-%d %H:%M IST"),
            "exit_ist": _ist(exit_ts).strftime("%Y-%m-%d %H:%M IST"),
            "reason": reason,
            "btc_entry": round(pos["entry_btc"], 1), "btc_sl": round(sl_level, 1),
            "opt_in": round(pos["entry_prem"], 1), "opt_out": round(exit_prem, 1),
            "lots": lots, "gross_usd": round(gross, 2), "fee_usd": round(fee, 2),
            "net_usd": round(gross - fee, 2),
        })
        active = triggered = False
        pos = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for i in range(1, len(candles)):
            ts = candles[i].start_time
            it, it_prev = in_session(ts), in_session(candles[i - 1].start_time)
            new_session = it and not it_prev
            t = _ist(ts)

            # 1. Arm a setup (never on the new-session bar).
            if not active and it and not new_session:
                bull = (cl[i - 1] < o[i - 1] and cl[i] > o[i] and cl[i] > ema10[i]
                        and ema10[i] > ema20[i] and cl[i] > ema200[i])
                bear = (cl[i - 1] > o[i - 1] and cl[i] < o[i] and cl[i] < ema10[i]
                        and ema10[i] < ema20[i] and cl[i] < ema200[i])
                if bull:
                    active, triggered, direction = True, False, "LONG"
                    entry_level, sl_level = max(h[i], h[i - 1]), min(lo[i], lo[i - 1])
                elif bear:
                    active, triggered, direction = True, False, "SHORT"
                    entry_level, sl_level = min(lo[i], lo[i - 1]), max(h[i], h[i - 1])

            # 2. Entry trigger / pre-entry invalidation -> BUY the option.
            if active and not triggered and it:
                hit_entry = (direction == "LONG" and h[i] > entry_level) or \
                            (direction == "SHORT" and lo[i] < entry_level)
                hit_sl_first = (direction == "LONG" and lo[i] < sl_level) or \
                               (direction == "SHORT" and h[i] > sl_level)
                if hit_entry:
                    otype = OptionType.CALL if direction == "LONG" else OptionType.PUT
                    expiry = op.select_expiry_date(ts, cutoff)
                    resolved = op.resolve_by_premium(
                        client, underlying, otype, cl[i], expiry, interval,
                        args.target_premium, ts, ts, ts - 86400, ts + 2 * 86400,
                        args.opt_resolution, step, cache)
                    if resolved is None:
                        active = False  # no option data -> drop setup
                    else:
                        sym, _, ocandles = resolved
                        eprem = op.premium_at(ocandles, ts, step)
                        if eprem is None:
                            active = False
                        else:
                            triggered, entry_ts = True, ts
                            pos = {"sym": sym, "candles": ocandles, "entry_btc": cl[i],
                                   "entry_prem": eprem, "tp": eprem * tp_mult, "last_check": ts}
                elif hit_sl_first:
                    active = False  # invalidated before entry

            # 3. Manage the open option position.
            if active and triggered and pos is not None:
                exited = False
                # 3a. System stop (BTC pattern stop, trailed to EMA200, close-beyond exit).
                if direction == "LONG":
                    if lo[i] <= sl_level:
                        p = op.premium_at(pos["candles"], ts, step)
                        if p is not None:
                            close_opt("SL", p, ts, cl[i]); exited = True
                    elif sl_level < ema200[i] and cl[i] > ema200[i]:
                        sl_level = max(sl_level, ema200[i])
                    elif cl[i] < ema200[i]:
                        p = op.premium_at(pos["candles"], ts, step)
                        if p is not None:
                            close_opt("EMA200", p, ts, cl[i]); exited = True
                else:
                    if h[i] >= sl_level:
                        p = op.premium_at(pos["candles"], ts, step)
                        if p is not None:
                            close_opt("SL", p, ts, cl[i]); exited = True
                    elif sl_level > ema200[i] and cl[i] < ema200[i]:
                        sl_level = min(sl_level, ema200[i])
                    elif cl[i] > ema200[i]:
                        p = op.premium_at(pos["candles"], ts, step)
                        if p is not None:
                            close_opt("EMA200", p, ts, cl[i]); exited = True

                # 3b. Option +100% take-profit (premium doubled).
                if not exited and pos is not None:
                    hi = opt_high_in(pos["candles"], pos["last_check"], ts, step)
                    if hi is not None and hi >= pos["tp"]:
                        close_opt("TP", pos["tp"], ts, cl[i]); exited = True
                    elif pos is not None:
                        pos["last_check"] = ts

            # 4. Auto-exit at 17:25.
            if t.hour == SESSION_END[0] and t.minute == SESSION_END[1] and active:
                if triggered and pos is not None:
                    p = op.premium_at(pos["candles"], ts, step)
                    close_opt("EOD", p if p is not None else pos["entry_prem"], ts, cl[i])
                else:
                    active = triggered = False
                    pos = None

    return [t for t in trips if int(datetime.strptime(
        t["entry_ist"].replace(" IST", ""), "%Y-%m-%d %H:%M")
        .replace(tzinfo=_IST).timestamp()) >= win_start]


def main() -> None:
    ap = argparse.ArgumentParser(description="claudebuy option-buying backtest")
    ap.add_argument("--days", type=float, default=4)
    ap.add_argument("--resolution", default="5m")
    ap.add_argument("--warmup-days", type=int, default=3)
    ap.add_argument("--target-premium", type=float, default=500.0)
    ap.add_argument("--take-profit", type=float, default=50.0,
                    help="%% premium gain to book profit (50 = exit at 1.5x buy price)")
    ap.add_argument("--opt-resolution", default="5m")
    ap.add_argument("--lots", type=int, default=None)
    ap.add_argument("--excel", default="claudebuy_options.xlsx")
    args = ap.parse_args()

    s = load_settings()
    now = int(time.time())
    win_start = now - int(args.days * 86400)
    df = download(symbol=s.symbol, start=win_start - args.warmup_days * 86400,
                  end=now, resolution=args.resolution, base_url=s.rest_base_url)
    candles = df_to_candles(df)
    trips = run(candles, s, args)
    lots = args.lots if args.lots is not None else s.option_contracts

    print(f"\n{s.symbol}  {args.resolution}  claudebuy OPTION-BUY ~{args.target_premium:.0f} prem, "
          f"+{args.take_profit:.0f}% target, SL=system, lots {lots}")
    print(f"Window {_ist(win_start):%Y-%m-%d %H:%M} -> {_ist(now):%Y-%m-%d %H:%M} IST")
    print("=" * 104)
    print(f"{'Entry(IST)':<18}{'Exit(IST)':<18}{'Action':<10}{'Contract':<22}{'Why':<8}"
          f"{'In':>7}{'Out':>8}{'Net$':>9}")
    print("-" * 104)
    net = fees = gross = 0.0
    wins = 0
    for t in trips:
        print(f"{t['entry_ist'][5:16]:<18}{t['exit_ist'][5:16]:<18}{t['action']:<10}"
              f"{t['contract']:<22}{t['reason']:<8}{t['opt_in']:>7.1f}{t['opt_out']:>8.1f}{t['net_usd']:>9.2f}")
        net += t["net_usd"]; fees += t["fee_usd"]; gross += t["gross_usd"]
        wins += 1 if t["net_usd"] > 0 else 0
    print("=" * 104)
    n = len(trips)
    wr = (wins / n * 100) if n else 0
    reasons = {r: sum(1 for t in trips if t["reason"] == r) for r in ("TP", "SL", "EMA200", "EOD")}
    print(f"Trades {n}  Wins/Losses {wins}/{n - wins}  Win rate {wr:.1f}%  "
          f"exits: TP={reasons['TP']} SL={reasons['SL']} EMA200={reasons['EMA200']} EOD={reasons['EOD']}")
    print(f"Gross {gross:,.2f}  -  brokerage {fees:,.2f}  =  NET {net:,.2f} USD")

    with pd.ExcelWriter(Path(args.excel), engine="openpyxl") as xl:
        pd.DataFrame(trips).to_excel(xl, sheet_name="Trades", index=False)
        pd.DataFrame([
            {"metric": "window_IST", "value": f"{_ist(win_start):%Y-%m-%d %H:%M} -> {_ist(now):%Y-%m-%d %H:%M}"},
            {"metric": "target_premium", "value": args.target_premium},
            {"metric": "target", "value": "2x (+100%)"}, {"metric": "lots", "value": lots},
            {"metric": "trades", "value": n}, {"metric": "win_rate_pct", "value": round(wr, 1)},
            {"metric": "exits_TP/SL/EMA200/EOD", "value": f"{reasons['TP']}/{reasons['SL']}/{reasons['EMA200']}/{reasons['EOD']}"},
            {"metric": "gross_usd", "value": round(gross, 2)},
            {"metric": "brokerage_usd", "value": round(fees, 2)},
            {"metric": "net_usd", "value": round(net, 2)},
        ]).to_excel(xl, sheet_name="Summary", index=False)
    print(f"\nExcel written to {Path(args.excel).resolve()}")


if __name__ == "__main__":
    main()
