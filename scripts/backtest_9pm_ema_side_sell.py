"""Backtest: 9PM EMA50/EMA200 one-side option SELL (directional version of the 9PM strangle).

Instead of selling both legs at 21:00 IST, sell only ONE leg chosen by trend on the 5m chart:

* At --entry-hour:minute (21:00) look at the last CLOSED 5m BTC candle:
    close ABOVE both EMA50 and EMA200 -> SELL a PUT  ~--otm-pct (2%) below spot
    close BELOW both EMA50 and EMA200 -> SELL a CALL ~--otm-pct (2%) above spot
    close BETWEEN the two EMAs        -> wait; enter on the first later 5m candle that
                                         CLOSES above both (PUT) or below both (CALL).
  No entry if no such close happens before --entry-deadline-hour:minute (default: the
  next-day exit time, i.e. keep waiting the whole session).
* Contract: next-day 17:30 expiry (same as the strangle), strike nearest the %-OTM target.
* Exits on the single leg (checked every 1m): TARGET when premium decays --target-pct (70%),
  SL when premium rises --sl-pct (100%), otherwise bought back at --exit-hour:minute
  (17:00) the next day. One trade per evening.

Run: python scripts/backtest_9pm_ema_side_sell.py --days 30 --lots 200 --skip-weekdays Sat,Sun
"""

from __future__ import annotations

import argparse
import collections
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from deltabot.backtest import option_pricing as op
from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.config import load_settings
from deltabot.enums import OptionType
from deltabot.logging_setup import setup_logging

_IST = ZoneInfo("Asia/Kolkata")
BAR = 300                               # signal-chart bar length (seconds), set from --resolution
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _t(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%m-%d %H:%M")


def ema(values, length):
    """Pine ta.ema: seeded with the SMA of the first ``length`` values."""
    out = [None] * len(values)
    if len(values) < length:
        return out
    a = 2.0 / (length + 1)
    v = sum(values[:length]) / length
    out[length - 1] = v
    for i in range(length, len(values)):
        v = a * values[i] + (1 - a) * v
        out[i] = v
    return out


def find_signal(m5, ef, es, i0, deadline, stacked=False):
    """From 5m bar index i0 on, return (bar_index, up) of the first qualifying close.
    both mode:    close above both EMAs (up=True) / below both (up=False).
    stacked mode: close > EMA_fast > EMA_slow (up=True) / close < EMA_fast < EMA_slow (up=False)."""
    for i in range(i0, len(m5)):
        b = m5[i]
        if b.start_time + BAR > deadline:
            return None
        if ef[i] is None or es[i] is None:
            continue
        if stacked:
            if b.close > ef[i] > es[i]:
                return i, True
            if b.close < ef[i] < es[i]:
                return i, False
        else:
            if b.close > ef[i] and b.close > es[i]:
                return i, True
            if b.close < ef[i] and b.close < es[i]:
                return i, False
    return None


def trade(client, m5, m1_close, ef, es, idx, i, waited, up, start, wd, exp, exit_ts, last_ts,
          step, tp_frac, sl_frac, args):
    ent = m5[i].start_time + BAR                    # decision at the signal candle close
    spot = m5[i].close
    otype = OptionType.PUT if up else OptionType.CALL
    off = spot * args.otm_pct / 100
    target = int(round((spot - off if up else spot + off) / 200) * 200)
    end = min(exit_ts, last_ts)
    if ent >= end:
        return None
    r = op.resolve_contract(client, "BTC", otype, target, exp, 200, ent, ent,
                            ent - 86400, exit_ts + 600, args.opt_resolution, step)
    if r is None:
        print(f"NOTE: {_t(ent)} no option data near {target} -- skipped")
        return None
    sym, strike, oc = r
    p_in = op.premium_at(oc, ent, step)
    if p_in is None:
        return None
    p_in = max(p_in, op.intrinsic_value(sym, spot))
    reason, out_ts, p_out, btc_out = None, end, None, spot
    ts = ent
    while ts < end:
        px = m1_close.get(ts)
        v = op.premium_at(oc, ts, step) if px is not None else None
        ts += 60
        if v is not None:
            v = max(v, op.intrinsic_value(sym, px))
            if v <= p_in * tp_frac:
                reason, out_ts, p_out, btc_out = "TARGET", ts, v, px
                break
            if args.sl_pct > 0 and v >= p_in * sl_frac:
                reason, out_ts, p_out, btc_out = "SL", ts, v, px
                break
        if args.ema_sl and ts % BAR == 0:              # a 5m candle just closed
            j = idx.get(ts - BAR)
            if j is not None and ef[j] is not None:
                b = m5[j]
                hit = (b.close < ef[j] and b.close < es[j]) if up else (b.close > ef[j] and b.close > es[j])
                if hit:
                    v = op.premium_at(oc, ts, step)
                    if v is None:
                        v = op.premium_at(oc, ts - 60, step)
                    reason, out_ts, btc_out = "EMA_SL", ts, b.close
                    p_out = max(v if v is not None else 0.0, op.intrinsic_value(sym, b.close))
                    break
    if reason is None:
        reason = "EOD" if end == exit_ts else "OPEN"
        btc_out = m1_close.get(end - 60, spot)
        v = op.premium_at(oc, end - 60, step)
        p_out = max(v if v is not None else 0.0, op.intrinsic_value(sym, btc_out))
    fees = op.side_fee(spot, p_in, args.lots) + op.side_fee(btc_out, p_out, args.lots)
    net = (p_in - p_out) * args.lots * op.LOT_BTC - fees
    return {"ts": start, "ent": ent, "wd": wd, "up": up, "spot": spot, "ef": ef[i], "es": es[i],
            "wait": waited, "sym": sym, "in": p_in, "out": p_out, "out_ts": out_ts,
            "btc_out": btc_out, "reason": reason, "net": net, "fees": fees, "skip": False}


def run(m5, m1, settings, args, sim_start):
    step = op.RES_SECONDS[args.opt_resolution]
    closes = [c.close for c in m5]
    ef, es = ema(closes, args.fast), ema(closes, args.slow)
    idx = {c.start_time: i for i, c in enumerate(m5)}
    m1_close = {c.start_time: c.close for c in m1}
    skip = {s.strip().capitalize()[:3] for s in args.skip_weekdays.split(",") if s.strip()}
    last_ts = m5[-1].start_time + BAR
    tp_frac = 1 - args.target_pct / 100
    sl_frac = 1 + args.sl_pct / 100
    day = datetime.fromtimestamp(sim_start, tz=_IST).replace(
        hour=args.entry_hour, minute=args.entry_minute, second=0, microsecond=0)
    if day.timestamp() < sim_start:
        day += timedelta(days=1)
    trades = []
    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        while day.timestamp() < last_ts:
            start = int(day.timestamp())
            wd = _DAYS[day.weekday()]
            exp = day + timedelta(days=1)
            exit_ts = int(exp.replace(hour=args.exit_hour, minute=args.exit_minute).timestamp())
            if args.entry_deadline_hour >= 0:
                dl = day.replace(hour=args.entry_deadline_hour, minute=args.entry_deadline_minute)
                if dl <= day:
                    dl += timedelta(days=1)
                deadline = min(int(dl.timestamp()), exit_ts)
            else:
                deadline = exit_ts
            day += timedelta(days=1)
            if wd in skip:
                continue
            i0 = idx.get(start - BAR)                       # 5m candle that closed at the session start
            if i0 is None:
                continue
            i, n_sess = i0, 0
            while n_sess < args.max_trades:
                sig = find_signal(m5, ef, es, i, deadline, args.stacked)
                if sig is None:
                    if n_sess == 0 and deadline <= last_ts:
                        trades.append({"ts": start, "wd": wd, "skip": True})
                    break
                i, up = sig
                t = trade(client, m5, m1_close, ef, es, idx, i, i > i0, up, start, wd, exp, exit_ts,
                          last_ts, step, tp_frac, sl_frac, args)
                if t is None:
                    break
                trades.append(t)
                n_sess += 1
                j = idx.get(t["out_ts"] - t["out_ts"] % BAR)  # bar in progress at the exit
                if j is None or t["reason"] in ("EOD", "OPEN"):
                    break
                i = j                                   # first 5m bar that closes after the exit
    return trades


def report(trades, args):
    rule = (f"close>EMA{args.fast}>EMA{args.slow}: sell PUT / close<EMA{args.fast}<EMA{args.slow}: sell CALL"
            if args.stacked else "above both: sell PUT / below both: sell CALL")
    sl = " + ".join(x for x in ((f"premium +{args.sl_pct:g}%" if args.sl_pct > 0 else ""),
                               ("5m close beyond both EMAs" if args.ema_sl else "")) if x)
    print(f"\n{args.entry_hour:02d}:{args.entry_minute:02d} EMA{args.fast}/EMA{args.slow} ONE-SIDE SELL -- {rule} "
          f"(else wait for a later {args.resolution} close); {args.otm_pct:g}% OTM, next-day expiry, max {args.max_trades}/session, "
          f"TP {args.target_pct:g}% decay, SL {sl}, exit "
          f"{args.exit_hour:02d}:{args.exit_minute:02d}, {args.lots} lots")
    print(f"{'#':<4}{'session':<13}{'day':<5}{'entry':<13}{'side':<6}{'BTC':>8}{'EMA50':>8}{'EMA200':>8}  "
          f"{'contract':<21}{'prem in->out':<17}{'exit':<7}{'at':<12}{'net $':>9}{'cum $':>9}")
    cum, n = 0.0, 0
    for t in trades:
        if t["skip"]:
            print(f"{'-':<4}{_t(t['ts']):<13}{t['wd']:<5}no close outside both EMAs -- no trade")
            continue
        n += 1
        cum += t["net"]
        side = ("PE" if t["up"] else "CE") + ("*" if t["wait"] else "")
        print(f"{n:<4}{_t(t['ts']):<13}{t['wd']:<5}{_t(t['ent']):<13}{side:<6}{t['spot']:>8.0f}{t['ef']:>8.0f}"
              f"{t['es']:>8.0f}  {t['sym']:<21}{t['in']:>6.1f}->{t['out']:<9.1f}{t['reason']:<7}"
              f"{_t(t['out_ts']):<12}{t['net']:>9.2f}{cum:>9.2f}")
    tr = [t for t in trades if not t["skip"]]
    if not tr:
        print("No trades.")
        return
    print("  (* = no signal on the session-start candle; entered on a later close)")
    nets = [t["net"] for t in tr]
    cum = pk = dd = 0.0
    for v in nets:
        cum += v
        pk = max(pk, cum)
        dd = min(dd, cum - pk)
    by = collections.OrderedDict()
    months = collections.OrderedDict()
    for t in tr:
        c, v = by.get(t["reason"], (0, 0.0))
        by[t["reason"]] = (c + 1, v + t["net"])
        k = datetime.fromtimestamp(t["ts"], tz=_IST).strftime("%Y-%m")
        months[k] = months.get(k, 0.0) + t["net"]
    print("-" * 140)
    print(f"Trades: {len(tr)}  wins {sum(v > 0 for v in nets)} ({100 * sum(v > 0 for v in nets) / len(nets):.0f}%)  "
          f"PUT sells ${sum(t['net'] for t in tr if t['up']):+.2f}  "
          f"CALL sells ${sum(t['net'] for t in tr if not t['up']):+.2f}  "
          f"waited-entries ${sum(t['net'] for t in tr if t['wait']):+.2f} (n={sum(t['wait'] for t in tr)})")
    for r, (c, v) in by.items():
        print(f"  {r:<7} n={c:<4} net $ {v:>9.2f}")
    print("Months $: " + "  ".join(f"{k}:{v:+.2f}" for k, v in months.items()))
    print(f"TOTAL NET: ${sum(nets):+.2f} at {args.lots} lots (fees ${sum(t['fees'] for t in tr):.2f})  "
          f"max drawdown ${dd:.2f}  best ${max(nets):.2f}  worst ${min(nets):.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="9PM EMA50/EMA200 one-side option selling")
    p.add_argument("--days", type=float, default=30)
    p.add_argument("--end-days-ago", type=float, default=0.0)
    p.add_argument("--fast", type=int, default=50)
    p.add_argument("--slow", type=int, default=200)
    p.add_argument("--otm-pct", type=float, default=2.0)
    p.add_argument("--target-pct", type=float, default=70.0, help="premium decay %% to take profit")
    p.add_argument("--sl-pct", type=float, default=100.0, help="premium rise %% to stop out")
    p.add_argument("--entry-hour", type=int, default=21)
    p.add_argument("--entry-minute", type=int, default=0)
    p.add_argument("--entry-deadline-hour", type=int, default=-1,
                   help="stop waiting for a close outside both EMAs at this IST hour (-1 = until exit)")
    p.add_argument("--entry-deadline-minute", type=int, default=0)
    p.add_argument("--exit-hour", type=int, default=17)
    p.add_argument("--exit-minute", type=int, default=0)
    p.add_argument("--lots", type=int, default=200)
    p.add_argument("--skip-weekdays", default="", help="e.g. Sat,Sun (session start day)")
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--resolution", default="5m", help="BTC signal chart for the EMAs / entry closes (1m, 5m, 15m)")
    p.add_argument("--stacked", action="store_true",
                   help="entry needs close < EMA_fast < EMA_slow (CE) / close > EMA_fast > EMA_slow (PE)")
    p.add_argument("--ema-sl", action="store_true",
                   help="SL when a 5m candle closes beyond both EMAs against the trade "
                        "(CE: close above both, PE: close below both); use --sl-pct 0 to drop the premium SL")
    p.add_argument("--max-trades", type=int, default=1, help="trades per session (re-entry after exit)")
    args = p.parse_args()

    setup_logging("WARNING")
    settings = load_settings()
    end = int(time.time() - args.end_days_ago * 86400)
    sim_start = end - int(args.days * 86400)
    global BAR
    BAR = op.RES_SECONDS[args.resolution]
    m1 = df_to_candles(download(symbol="BTCUSD", start=sim_start - 4 * 86400, end=end, resolution="1m"))
    m5 = m1 if args.resolution == "1m" else df_to_candles(
        download(symbol="BTCUSD", start=sim_start - 4 * 86400, end=end, resolution=args.resolution))
    print(f"Candles: {len(m5)} ({args.resolution}) {_t(m5[0].start_time)} .. {_t(m5[-1].start_time)} IST")
    report(run(m5, m1, settings, args, sim_start), args)


if __name__ == "__main__":
    main()
