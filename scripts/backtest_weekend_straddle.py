"""Backtest: Weekend Long Straddle (Fri 5:30pm IST -> Sun 5:30pm IST).

Another PURE TIME-BASED strategy (same family as backtest_daily_strangle_9pm.py),
but weekly instead of daily, and BUYING instead of selling:

  1. The window is [Friday 17:30 IST, Sunday 17:30 IST) -- exactly 48 hours,
     once per week. No entries happen outside this window.
  2. At the instant the window opens (Friday 17:30 IST), BUY one CALL and one
     PUT -- a long straddle -- each strike picked independently as the
     LISTED strike whose entry premium is closest to --target-premium
     (default $50), via ``op.resolve_by_premium`` (mirrors
     OptionsExecutor.select_by_premium's chain-wide nearest-premium search).
     Expiry is fixed at the Sunday two days out, so it comfortably covers the
     whole 48h hold (Delta lists a daily-expiry contract for every calendar
     date, including weekends -- confirmed real data exists, see
     backtest_daily_strangle_9pm.py's own note on this).
  3. TARGET: the instant EITHER leg's premium first trades at or above
     --target-price (default $150 -- a 3x on a $50 entry), close BOTH legs
     together: take the winning leg's profit, cut the other leg's decayed
     remainder. No stop-loss -- a bought straddle's max loss is already
     capped at the premium paid.
  4. If neither leg has hit target by the time the window closes (Sunday
     17:30 IST), force-close BOTH legs at market -- one straddle per
     weekend, never carried into the next window.

Run:  python scripts/backtest_weekend_straddle.py --days 60
"""

from __future__ import annotations

import argparse
import bisect
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
_OPEN_WD, _OPEN_H, _OPEN_M = 4, 17, 30   # Friday 17:30 IST
_CLOSE_WD, _CLOSE_H, _CLOSE_M = 6, 17, 30  # Sunday 17:30 IST


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _in_window(dt: datetime) -> bool:
    """True from Friday 17:30 IST (inclusive) to Sunday 17:30 IST (exclusive)."""
    wd, mins = dt.weekday(), dt.hour * 60 + dt.minute
    if wd == _OPEN_WD:
        return mins >= _OPEN_H * 60 + _OPEN_M
    if wd == _CLOSE_WD:
        return mins < _CLOSE_H * 60 + _CLOSE_M
    return _OPEN_WD < wd < _CLOSE_WD  # Saturday only


def _first_touch(candles: dict[int, Candle], after: int, upto: int, target: float) -> int | None:
    """Earliest candle start_time in ``(after, upto]`` whose high reaches ``target``."""
    hits = [t for t, c in candles.items() if after < t <= upto and c.high >= target]
    return min(hits) if hits else None


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    lot_size = args.lot_size if args.lot_size > 0 else op.LOT_SIZE.get(underlying, op.LOT_BTC)
    opt_step = op.RES_SECONDS.get(args.opt_resolution, 60)
    bar_seconds = op.RES_SECONDS.get(args.resolution, 60)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor

    trades: list[dict] = []
    cache: dict = {}
    pos: dict | None = None
    prev_in_window: bool | None = None

    _spot_ts = sorted(c.start_time for c in candles)
    _spot = {c.start_time: c.close for c in candles}

    def btc_at(t: int) -> float | None:
        i = bisect.bisect_right(_spot_ts, t) - 1
        return _spot[_spot_ts[i]] if i >= 0 else None

    def close(reason: str, exit_time: int, ce_exit: float, pe_exit: float, btc_px: float) -> None:
        nonlocal pos
        ce_entry_fill = pos["ce_entry_prem"] * (1 + es)
        pe_entry_fill = pos["pe_entry_prem"] * (1 + es)
        ce_exit_fill = ce_exit * (1 - xs)
        pe_exit_fill = pe_exit * (1 - xs)
        gross = ((ce_exit_fill - ce_entry_fill) + (pe_exit_fill - pe_entry_fill)) * args.lots * lot_size
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
            decision_ts = c.start_time + bar_seconds
            dt_ist = datetime.fromtimestamp(decision_ts, tz=_IST)
            now_in_window = _in_window(dt_ist)
            entered = prev_in_window is False and now_in_window
            window_closed = prev_in_window is True and not now_in_window
            prev_in_window = now_in_window

            # 1. Hard square-off the instant the weekend window closes, whatever
            #    the target status -- one straddle per weekend, no carry-over.
            if window_closed and pos is not None:
                ce_p = op.premium_at(pos["ce_candles"], decision_ts, opt_step)
                pe_p = op.premium_at(pos["pe_candles"], decision_ts, opt_step)
                if ce_p is None:
                    ce_p = op.intrinsic_value(pos["ce_sym"], c.close)
                if pe_p is None:
                    pe_p = op.intrinsic_value(pos["pe_sym"], c.close)
                close("WINDOW_CLOSE", decision_ts, ce_p, pe_p, c.close)

            # 2. Target: either leg touching --target-price closes BOTH legs.
            if pos is not None:
                t_ce = _first_touch(pos["ce_candles"], pos["last_check"], decision_ts, args.target_price)
                t_pe = _first_touch(pos["pe_candles"], pos["last_check"], decision_ts, args.target_price)
                candidates = [t for t in (t_ce, t_pe) if t is not None]
                if candidates:
                    t_hit = min(candidates)
                    ce_p = (args.target_price if t_ce == t_hit
                            else op.premium_at(pos["ce_candles"], t_hit, opt_step) or pos["ce_entry_prem"])
                    pe_p = (args.target_price if t_pe == t_hit
                            else op.premium_at(pos["pe_candles"], t_hit, opt_step) or pos["pe_entry_prem"])
                    close("TARGET", t_hit, ce_p, pe_p, btc_at(t_hit) or c.close)
                else:
                    pos["last_check"] = decision_ts

            # 3. Entry: exactly once per week, at the Friday 17:30 IST window open.
            if entered and pos is None and c.start_time >= sim_start:
                btc_price = c.close
                expiry = dt_ist + timedelta(days=2)  # window opens Fri, closes Sun -> +2 days
                win_start = decision_ts - 86400
                win_end = decision_ts + 50 * 3600  # full 48h window + margin
                # A ~2-day-to-expiry option carries much more time value than the
                # same/next-day contracts option_pricing's default search range is
                # tuned for, so a given target premium can sit much further OTM --
                # widen the OTM walk accordingly (still bounded by whatever strikes
                # Delta actually lists/traded that week).
                ce_r = op.resolve_by_premium(client, underlying, OptionType.CALL, btc_price, expiry,
                                              interval, args.target_premium, decision_ts, decision_ts,
                                              win_start, win_end, args.opt_resolution, opt_step, cache,
                                              otm_steps=60)
                pe_r = op.resolve_by_premium(client, underlying, OptionType.PUT, btc_price, expiry,
                                              interval, args.target_premium, decision_ts, decision_ts,
                                              win_start, win_end, args.opt_resolution, opt_step, cache,
                                              otm_steps=60)
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
                            "last_check": decision_ts,
                        }

        if pos is not None:
            last = candles[-1]
            last_ts = last.start_time + bar_seconds
            ce_p = op.premium_at(pos["ce_candles"], last_ts, opt_step)
            pe_p = op.premium_at(pos["pe_candles"], last_ts, opt_step)
            close("OPEN_AT_END", last_ts, ce_p if ce_p is not None else pos["ce_entry_prem"],
                  pe_p if pe_p is not None else pos["pe_entry_prem"], last.close)

    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 118}")
    print(f"Weekend Long Straddle (Fri 17:30 IST -> Sun 17:30 IST) -- {args.days}d, "
          f"~${args.target_premium:.0f} entry/leg, target ${args.target_price:.0f}/leg, "
          f"{args.lots} lots, floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 118}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'exit (IST)':<18}{'CE':<22}{'PE':<22}{'reason':<14}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{_ist(t['exit_time']):<18}{t['ce_sym']:<22}{t['pe_sym']:<22}"
              f"{t['reason']:<14}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    print(f"{'-' * 118}")
    print(f"Straddles: {len(trades)}" + (" (last one still open at data end)"
                                          if trades and trades[-1]["reason"] == "OPEN_AT_END" else ""))
    if closed:
        wins_closed = [t for t in closed if t["net"] > 0]
        print(f"Win rate: {len(wins_closed)}/{len(closed)} = {100.0 * len(wins_closed) / len(closed):.1f}%")
    for reason in ("TARGET", "WINDOW_CLOSE", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<14} n={len(rs):<4} net ${sum(t['net'] for t in rs):>11.2f}")
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
    print(f"\nExcel written: {path}  ({len(df)} straddles)")


def main() -> None:
    p = argparse.ArgumentParser(description="Weekend Long Straddle backtest (Fri 5:30pm - Sun 5:30pm IST)")
    p.add_argument("--days", type=float, default=60)
    p.add_argument("--warmup-days", type=int, default=2)
    p.add_argument("--resolution", default="1m")
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--target-premium", type=float, default=50.0,
                    help="Entry premium target per leg -- the strike whose premium is closest "
                         "to this is picked independently for the CALL and the PUT.")
    p.add_argument("--target-price", type=float, default=150.0,
                    help="Exit trigger: close BOTH legs the instant EITHER leg's premium "
                         "first reaches this price.")
    p.add_argument("--lots", type=int, default=10)
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
