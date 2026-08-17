"""Backtest: EMA(21) Breakdown run DIRECTLY on the CE option's own premium
series, instead of on BTC price. Same architecture as backtest_dcv2.py's
--mode ce_price: each IST day, splice the CALL nearest --target-premium into
one continuous series, and run Ema21BreakdownStrategy.update() on THAT
series -- EMA(21), the open/close-vs-EMA pattern, entry, SL, and target are
all computed on option premium, not BTC. Since the strategy is already
CE-only (sell_signal only, no buy side), every completed setup trades.

CAVEAT (same as dcv2's ce_price mode): each day's splice point is a jump
from one contract's last premium to a brand-new, unrelated contract's
premium. The EMA/candle-pattern logic has no notion of "new instrument" and
will treat that jump as ordinary price action -- can read as a fake
breakout/gap right at the daily 17:30 IST boundary.

No rollover -- EOD-closes at 17:25 IST every day (the next day is already a
different spliced-in contract, so "continuing" the trade has no natural
meaning here). Intrinsic-value flooring is ON by default.

Run:  python scripts/backtest_ema21_breakdown_ce_price.py --days 7
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
from deltabot.strategy.ema21_breakdown import Ema21BreakdownStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _ist_mins(ts: int) -> int:
    d = datetime.fromtimestamp(ts, tz=_IST)
    return d.hour * 60 + d.minute


def _day_boundaries(start_ts: int, end_ts: int, hour: int, minute: int) -> list[int]:
    d = datetime.fromtimestamp(start_ts, tz=_IST).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if d.timestamp() > start_ts:
        d -= timedelta(days=1)
    out = []
    while d.timestamp() < end_ts:
        out.append(int(d.timestamp()))
        d += timedelta(days=1)
    return out


def _build_ce_stitched_series(
    client: httpx.Client, btc_candles: list[Candle], settings, args, cache: dict,
) -> tuple[list[Candle], dict[int, tuple[str, int]], list[int]]:
    """Splice each day's near-target-premium CALL into one continuous series.
    Returns (stitched_candles, day_info, boundaries) where
    day_info[boundary_ts] = (symbol, strike) for the contract active over
    [boundary_ts, next_boundary_ts)."""
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    step = op.RES_SECONDS.get(args.opt_resolution, 60)

    times = [c.start_time for c in btc_candles]

    def btc_price_at(ts: int) -> float:
        i = max(0, min(bisect.bisect_right(times, ts) - 1, len(btc_candles) - 1))
        return btc_candles[i].close

    start_ts, end_ts = btc_candles[0].start_time, btc_candles[-1].start_time
    boundaries = _day_boundaries(start_ts, end_ts, args.day_start_hour, args.day_start_minute)

    stitched: list[Candle] = []
    day_info: dict[int, tuple[str, int]] = {}
    for i, b in enumerate(boundaries):
        nb = boundaries[i + 1] if i + 1 < len(boundaries) else end_ts + 1
        btc_px = btc_price_at(b)
        expiry = op.select_expiry_date(b, cutoff)
        resolved = op.resolve_by_premium(
            client, underlying, OptionType.CALL, btc_px, expiry, interval,
            args.target_premium, b, b, b - 86400, b + 2 * 86400,
            args.opt_resolution, step, cache,
        )
        if resolved is None:
            continue
        sym, strike, ocandles = resolved
        day_info[b] = (sym, strike)
        stitched.extend(ocandles[t] for t in sorted(ocandles) if b <= t < nb)
    stitched.sort(key=lambda c: c.start_time)
    return stitched, day_info, boundaries


def run(btc_candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = Ema21BreakdownStrategy(
        ema_len=args.ema_len, max_wait=args.max_wait, target_rr=args.target_rr,
    )
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    lots = args.lots
    lot_size = args.lot_size if args.lot_size > 0 else op.LOT_SIZE.get(underlying, op.LOT_BTC)
    floor = not args.no_intrinsic_floor
    sq_mins = args.square_off_hour * 60 + args.square_off_minute
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0

    cache: dict = {}
    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        stitched, day_info, boundaries = _build_ce_stitched_series(client, btc_candles, settings, args, cache)

    if not stitched:
        return []

    times_btc = [c.start_time for c in btc_candles]

    def btc_ref(ts: int) -> float:
        i = max(0, min(bisect.bisect_right(times_btc, ts) - 1, len(btc_candles) - 1))
        return btc_candles[i].close

    def symbol_for(ts: int) -> str:
        i = max(0, bisect.bisect_right(boundaries, ts) - 1)
        return day_info.get(boundaries[i], ("?", 0))[0]

    trades: list[dict] = []
    pos: dict | None = None
    prev_mins: int | None = None

    def open_leg(ts: int, prem: float) -> None:
        nonlocal pos
        sym = symbol_for(ts)
        btc_px = btc_ref(ts)
        if floor:
            prem = max(prem, op.intrinsic_value(sym, btc_px))
        pos = {"entry_time": ts, "entry_prem": prem, "sym": sym, "btc_ref": btc_px}

    def close(reason: str, exit_prem: float, exit_time: int) -> None:
        nonlocal pos
        assert pos is not None
        exit_btc = btc_ref(exit_time)
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        entry_fill = pos["entry_prem"] * (1 - es)
        exit_fill = exit_prem * (1 + xs)
        gross = (entry_fill - exit_fill) * lots * lot_size
        fee = (op.side_fee(pos["btc_ref"], entry_fill, lots, lot_size)
               + op.side_fee(exit_btc, exit_fill, lots, lot_size))
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time, "contract": pos["sym"],
            "btc_entry": pos["btc_ref"], "btc_exit": exit_btc,
            "opt_in": entry_fill, "opt_out": exit_fill, "reason": reason,
            "gross": gross, "fee": fee, "net": gross - fee,
        })
        pos = None

    for c in stitched:
        mins = _ist_mins(c.start_time)
        square_off = prev_mins is not None and mins >= sq_mins and prev_mins < sq_mins
        prev_mins = mins
        dec = strategy.update(c)

        # 1. Strategy exit (SL / TARGET), priced directly off the premium series.
        if pos is not None and dec is not None and dec.has_exit:
            close(dec.exit_reason, dec.short_exit_price, c.start_time)

        # 2. New entry, priced directly off the premium series -- entry_price
        #    IS a premium value here (the strategy operated on the premium
        #    series directly), no BTC->premium conversion needed.
        if pos is None and dec is not None and dec.has_entry and c.start_time >= sim_start:
            open_leg(c.start_time, dec.entry_price)

        # 3. 17:25 EOD square-off -- no rollover, tomorrow is a different
        #    spliced-in contract by construction.
        if pos is not None and square_off:
            close("EOD", c.close, c.start_time)
            strategy.force_flat()

    if pos is not None:
        last = stitched[-1]
        close("OPEN_AT_END", last.close, last.start_time)

    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 112}")
    print(f"EMA(21) Breakdown [CE-PRICE MODE, sell_signal only, 17:25 EOD, no rollover] -- {args.days}d, "
          f"premium ~{args.target_premium:.0f}, {args.lots} lots, target {args.target_rr:.1f}R, "
          f"floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 112}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'contract':<22}{'btc in':>9}{'btc out':>9}"
          f"{'opt in':>8}{'opt out':>8} {'reason':<12}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{t['contract']:<22}"
              f"{t['btc_entry']:>9.1f}{t['btc_exit']:>9.1f}{t['opt_in']:>8.1f}{t['opt_out']:>8.1f} "
              f"{t['reason']:<12}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 112}")
    print(f"Legs: {len(closed)} closed" + (" (+1 still open at data end)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "TARGET", "EOD", "OPEN_AT_END"):
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
            "contract": t["contract"], "btc_entry": round(t["btc_entry"], 1), "btc_exit": round(t["btc_exit"], 1),
            "opt_sold": round(t["opt_in"], 1), "opt_bought_back": round(t["opt_out"], 1),
            "exit_reason": t["reason"], "gross_usd": round(t["gross"], 2),
            "fee_usd": round(t["fee"], 2), "net_usd": round(t["net"], 2),
            "cumulative_net_usd": round(cum, 2),
        })
    df = pd.DataFrame(rows)
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    srows = [
        ("days", args.days), ("mode", "ce_price"), ("ema_len", args.ema_len), ("max_wait", args.max_wait),
        ("target_rr", args.target_rr), ("premium_target", args.target_premium), ("lots", args.lots),
        ("legs_closed", len(closed)),
        ("win_rate_pct", round(100.0 * len(wins) / len(closed), 1) if closed else 0.0),
        ("gross_usd", round(sum(t["gross"] for t in trades), 2)),
        ("fees_usd", round(sum(t["fee"] for t in trades), 2)),
        ("TOTAL_NET_usd", round(sum(t["net"] for t in trades), 2)),
    ]
    summary = pd.DataFrame(srows, columns=["metric", "value"])
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        df.to_excel(xl, sheet_name="Trades", index=False)
    print(f"\nExcel written: {path}  ({len(df)} legs)")


def main() -> None:
    p = argparse.ArgumentParser(description="EMA(21) Breakdown backtest -- CE-PRICE mode (pattern runs on premium, not BTC)")
    p.add_argument("--days", type=float, default=7, help="look-back window in days")
    p.add_argument("--warmup-days", type=int, default=2)
    p.add_argument("--btc-resolution", default="5m", help="resolution for the BTC reference series (strike selection / intrinsic floor)")
    p.add_argument("--ema-len", type=int, default=21)
    p.add_argument("--max-wait", type=int, default=3)
    p.add_argument("--target-rr", type=float, default=2.0)
    p.add_argument("--target-premium", type=float, default=1000.0)
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lot-size", type=float, default=0.0)
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--day-start-hour", type=int, default=17)
    p.add_argument("--day-start-minute", type=int, default=30)
    p.add_argument("--square-off-hour", type=int, default=17)
    p.add_argument("--square-off-minute", type=int, default=25)
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true")
    p.add_argument("--out", default="", help="also write every leg to this .xlsx file")
    args = p.parse_args()

    setup_logging("WARNING")
    settings = load_settings()

    now = int(time.time())
    sim_start = now - int(args.days * 86400)
    dl_start = sim_start - args.warmup_days * 86400

    df = download(symbol=settings.symbol, start=dl_start, end=now, resolution=args.btc_resolution)
    btc_candles = df_to_candles(df)
    print(f"BTC reference candles: {len(btc_candles)} ({args.btc_resolution}) "
          f"{_ist(btc_candles[0].start_time)} .. {_ist(btc_candles[-1].start_time)}")

    trades = run(btc_candles, settings, args, sim_start)
    report(trades, args)
    if args.out:
        export(trades, args, args.out)


if __name__ == "__main__":
    main()
