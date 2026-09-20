"""Backtest: Daily 9 PM IST Short Strangle.

A genuinely different kind of strategy from everything else in this repo:
PURE TIME-BASED, no technical indicator, no Supertrend, nothing chart-
driven at all. Every trading day (Mon-Fri IST) at exactly --entry-hour:
--entry-minute (default 21:00):

  1. Resolve the NEXT-DAY expiry (entry is fixed well past any same-day
     settlement window, so it's always tomorrow's contract).
  2. Sell one CALL strike ~--otm-pct% (default 2%) ABOVE spot and one PUT
     strike ~--otm-pct% BELOW spot (each rounded to the nearest listed
     strike via op.resolve_contract's nearest-listed-strike search) -- a short
     strangle, tracked as ONE COMBINED position: both legs open together
     and close together, never independently.
  3. TARGET: close BOTH legs together the instant the COMBINED buyback
     cost (CE + PE) decays to (100 - --target-pct)% of the COMBINED entry
     premium -- default 70% reduction, e.g. sold for 100+100=200
     combined, target is 200*0.30=60. Same REDUCTION convention as
     supertrend_15m_filter_fixed_sl.py's own --target-pct.
  4. SL: close BOTH legs together the instant the COMBINED buyback cost
     RISES to (100 + --sl-pct)% of the COMBINED entry premium -- default
     50% rise, e.g. sold for 200 combined, SL at 300. Checked every bar
     alongside the target; target is evaluated first if somehow both
     would fire the same bar (arbitrary tie-break, essentially never hit
     in practice since decay and rise are opposite directions).
  5. Regardless of target/SL, force-close both legs (whichever reason
     hasn't already closed them) at --exit-hour:--exit-minute (default
     17:00) IST THE NEXT DAY.
  6. No NEW entry on Saturday/Sunday (IST) -- an in-flight position that
     happens to span into the weekend (e.g. a Friday 9 PM entry) still
     exits normally via its own target/SL/next-day-5PM rules.

NOT IMPLEMENTABLE AS LITERALLY DESCRIBED IN THE SOURCE SPEC: alongside the
"~2% OTM" strike rule, the source also mentions "~0.18 delta" -- this
codebase has no options-Greeks (delta) data available anywhere (every
other strategy here selects strikes by premium-target or, now, this
script's %-OTM-from-spot rule). The 2% OTM rule is therefore the actual
operative strike-selection mechanism; the delta figure is descriptive
context only, not a separately-enforced constraint.

Run:  python scripts/backtest_daily_strangle_9pm.py --days 7
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


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    lot_size = args.lot_size if args.lot_size > 0 else op.LOT_SIZE.get(underlying, op.LOT_BTC)
    opt_step = op.RES_SECONDS.get(args.opt_resolution, 60)
    bar_seconds = op.RES_SECONDS.get(args.resolution, 60)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor

    entry_mins = args.entry_hour * 60 + args.entry_minute
    exit_mins = args.exit_hour * 60 + args.exit_minute
    # SELL, combined: target_pct is a REDUCTION (same convention as
    # supertrend_15m_filter_fixed_sl.py's own --target-pct), sl_pct is a
    # RISE -- these are two independent, opposite-direction thresholds on
    # the SAME combined-premium number, not the SAR-style single dial.
    target_frac = 1 - args.target_pct / 100.0
    sl_frac = 1 + args.sl_pct / 100.0

    trades: list[dict] = []
    cache: dict = {}
    pos: dict | None = None
    prev_mins: int | None = None

    def combined_prem(ts: int, btc_px: float) -> tuple[float, float] | None:
        ce_p = op.premium_at(pos["ce_candles"], ts, opt_step)
        pe_p = op.premium_at(pos["pe_candles"], ts, opt_step)
        if ce_p is None or pe_p is None:
            return None
        if floor:
            ce_p = max(ce_p, op.intrinsic_value(pos["ce_sym"], btc_px))
            pe_p = max(pe_p, op.intrinsic_value(pos["pe_sym"], btc_px))
        return ce_p, pe_p

    def close(reason: str, exit_time: int, ce_exit: float, pe_exit: float, btc_px: float) -> None:
        nonlocal pos
        ce_entry_fill = pos["ce_entry_prem"] * (1 - es)
        pe_entry_fill = pos["pe_entry_prem"] * (1 - es)
        ce_exit_fill = ce_exit * (1 + xs)
        pe_exit_fill = pe_exit * (1 + xs)
        gross = ((ce_entry_fill - ce_exit_fill) + (pe_entry_fill - pe_exit_fill)) * args.lots * lot_size
        fee = (op.side_fee(pos["entry_btc"], ce_entry_fill, args.lots, lot_size)
               + op.side_fee(pos["entry_btc"], pe_entry_fill, args.lots, lot_size)
               + op.side_fee(btc_px, ce_exit_fill, args.lots, lot_size)
               + op.side_fee(btc_px, pe_exit_fill, args.lots, lot_size))
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time,
            "ce_sym": pos["ce_sym"], "pe_sym": pos["pe_sym"],
            "combined_entry": pos["combined_entry"], "combined_exit": ce_exit + pe_exit,
            "reason": reason, "gross": gross, "fee": fee, "net": gross - fee,
        })
        pos = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            mins = _ist_mins(c.start_time)
            entry_crossed = prev_mins is not None and mins >= entry_mins and prev_mins < entry_mins
            exit_crossed = prev_mins is not None and mins >= exit_mins and prev_mins < exit_mins
            prev_mins = mins
            decision_ts = c.start_time + bar_seconds
            dt_ist = datetime.fromtimestamp(decision_ts, tz=_IST)
            is_weekend = dt_ist.weekday() >= 5 and not args.include_weekends   # Sat=5, Sun=6 IST

            # 1. Hard square-off at next-day exit time, regardless of target/SL.
            if exit_crossed and pos is not None:
                res = combined_prem(decision_ts, c.close)
                if res is not None:
                    close("EOD", decision_ts, res[0], res[1], c.close)
                else:
                    close("EOD", decision_ts, op.intrinsic_value(pos["ce_sym"], c.close),
                          op.intrinsic_value(pos["pe_sym"], c.close), c.close)

            # 2. Combined target/SL, checked every bar while in position.
            if pos is not None:
                res = combined_prem(decision_ts, c.close)
                if res is not None:
                    ce_p, pe_p = res
                    combined = ce_p + pe_p
                    if combined <= pos["combined_entry"] * target_frac:
                        close("TARGET", decision_ts, ce_p, pe_p, c.close)
                    elif combined >= pos["combined_entry"] * sl_frac:
                        close("SL", decision_ts, ce_p, pe_p, c.close)

            # 3. Entry: exactly entry_hour:entry_minute IST, weekdays only.
            if (entry_crossed and pos is None and not is_weekend
                    and c.start_time >= sim_start):
                btc_price = c.close
                expiry = dt_ist + timedelta(days=1)   # ALWAYS next day
                win_end = decision_ts + 30 * 3600      # full hold window + margin
                # Target strikes, snapped to the interval grid. Deliberately
                # using resolve_contract() (nearest-LISTED-strike search),
                # not resolve_atm() (exact-strike-only, no fallback) --
                # Delta's actual listed BTC option strikes have real gaps in
                # the interval grid (confirmed during development: e.g.
                # P-BTC-75400 had zero historical data while the strikes on
                # either side of it did), so an exact-grid-point request can
                # legitimately miss even when a very close, tradeable
                # contract exists right next to it.
                raw_offset = round((btc_price * args.otm_pct / 100.0) / interval) * interval
                call_target = int(round((btc_price + raw_offset) / interval) * interval)
                put_target = int(round((btc_price - raw_offset) / interval) * interval)
                ce_r = op.resolve_contract(client, underlying, OptionType.CALL, call_target, expiry,
                                            interval, decision_ts, decision_ts, decision_ts - 86400,
                                            win_end, args.opt_resolution, opt_step)
                pe_r = op.resolve_contract(client, underlying, OptionType.PUT, put_target, expiry,
                                            interval, decision_ts, decision_ts, decision_ts - 86400,
                                            win_end, args.opt_resolution, opt_step)
                if ce_r is not None and pe_r is not None:
                    ce_sym, _, ce_candles = ce_r
                    pe_sym, _, pe_candles = pe_r
                    ce_entry_prem = op.premium_at(ce_candles, decision_ts, opt_step)
                    pe_entry_prem = op.premium_at(pe_candles, decision_ts, opt_step)
                    if ce_entry_prem is not None and pe_entry_prem is not None:
                        if floor:
                            ce_entry_prem = max(ce_entry_prem, op.intrinsic_value(ce_sym, btc_price))
                            pe_entry_prem = max(pe_entry_prem, op.intrinsic_value(pe_sym, btc_price))
                        pos = {
                            "ce_sym": ce_sym, "pe_sym": pe_sym,
                            "ce_candles": ce_candles, "pe_candles": pe_candles,
                            "entry_time": decision_ts, "entry_btc": btc_price,
                            "ce_entry_prem": ce_entry_prem, "pe_entry_prem": pe_entry_prem,
                            "combined_entry": ce_entry_prem + pe_entry_prem,
                        }

        if pos is not None:
            last = candles[-1]
            last_ts = last.start_time + bar_seconds
            res = combined_prem(last_ts, last.close)
            if res is not None:
                close("OPEN_AT_END", last_ts, res[0], res[1], last.close)
            else:
                close("OPEN_AT_END", last_ts, pos["ce_entry_prem"], pos["pe_entry_prem"], last.close)

    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 118}")
    print(f"Daily {args.entry_hour:02d}:{args.entry_minute:02d} IST Short Strangle -- {args.days}d, "
          f"exit next-day {args.exit_hour:02d}:{args.exit_minute:02d} IST, ~{args.otm_pct:.0f}% OTM strikes, "
          f"target {args.target_pct:.0f}% combined decay, SL {args.sl_pct:.0f}% combined rise, "
          f"{args.lots} lots, floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 118}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'exit (IST)':<18}{'CE':<22}{'PE':<22}{'reason':<10}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{_ist(t['exit_time']):<18}{t['ce_sym']:<22}{t['pe_sym']:<22}"
              f"{t['reason']:<10}{t['net']:>10.2f}")
    wins = [t for t in trades if t["net"] > 0]
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    print(f"{'-' * 118}")
    print(f"Strangles: {len(trades)}" + (" (last one still open at data end)"
                                          if trades and trades[-1]["reason"] == "OPEN_AT_END" else ""))
    if closed:
        wins_closed = [t for t in closed if t["net"] > 0]
        print(f"Win rate: {len(wins_closed)}/{len(closed)} = {100.0 * len(wins_closed) / len(closed):.1f}%")
    for reason in ("TARGET", "SL", "EOD", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<12} n={len(rs):<4} net ${sum(t['net'] for t in rs):>11.2f}")
    print(f"TOTAL NET: ${sum(t['net'] for t in trades):.2f} "
          f"(gross ${sum(t['gross'] for t in trades):.2f}, fees ${sum(t['fee'] for t in trades):.2f})")


def export(trades: list[dict], args, path: str) -> None:
    import pandas as pd

    rows, cum = [], 0.0
    for i, t in enumerate(trades, 1):
        cum += t["net"]
        rows.append({
            "#": i, "entry_IST": _ist(t["entry_time"]), "exit_IST": _ist(t["exit_time"]),
            "ce_contract": t["ce_sym"], "pe_contract": t["pe_sym"],
            "combined_entry_prem": round(t["combined_entry"], 1),
            "combined_exit_prem": round(t["combined_exit"], 1),
            "exit_reason": t["reason"], "gross_usd": round(t["gross"], 2),
            "fee_usd": round(t["fee"], 2), "net_usd": round(t["net"], 2),
            "cumulative_net_usd": round(cum, 2),
        })
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Trades", index=False)
    print(f"\nExcel written: {path}  ({len(df)} strangles)")


def main() -> None:
    p = argparse.ArgumentParser(description="Daily 9 PM IST Short Strangle backtest")
    p.add_argument("--days", type=float, default=7)
    p.add_argument("--warmup-days", type=int, default=1)
    p.add_argument("--resolution", default="1m")
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--entry-hour", type=int, default=21)
    p.add_argument("--entry-minute", type=int, default=0)
    p.add_argument("--exit-hour", type=int, default=17)
    p.add_argument("--exit-minute", type=int, default=0)
    p.add_argument("--otm-pct", type=float, default=2.0)
    p.add_argument("--target-pct", type=float, default=70.0,
                    help="Combined-premium profit target as a REDUCTION from combined entry "
                         "(e.g. 70 -> target = 30%% of combined entry).")
    p.add_argument("--sl-pct", type=float, default=50.0,
                    help="Combined-premium stop-loss as a RISE from combined entry "
                         "(e.g. 50 -> SL = 150%% of combined entry).")
    p.add_argument("--lots", type=int, default=10)
    p.add_argument("--include-weekends", action="store_true",
                    help="Trade Saturday/Sunday entries too, instead of the default weekday-only "
                         "gate. Real weekend option data exists (verified) -- the default skip is "
                         "a deliberate design choice, not a data limitation.")
    p.add_argument("--lot-size", type=float, default=0.0)
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true")
    p.add_argument("--out", default="")
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
    if args.out:
        export(trades, args, args.out)


if __name__ == "__main__":
    main()
