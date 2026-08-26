"""Backtest: Straddle4pm -- option-BUY execution of
src/deltabot/strategy/straddle.py (StraddleStrategy).

NO BTC-price signal at all -- once per day, at --entry-hour:--entry-minute
IST (default 16:00, same-day expiry), BUYS a CALL and a PUT together, each
near --target-premium (default 100). Whichever leg's premium first reaches
--exit-target (default 250 -- a FIXED ABSOLUTE value, NOT a % of entry,
unlike every other TP in this repo) is closed at that mark, and the OTHER
leg is closed alongside it immediately at whatever it's currently worth.
Neither leg has its own SL -- max loss per leg is capped at the premium
paid (buying, not selling). If NEITHER leg reaches target, the daily
--square-off-hour/minute (default 17:25 IST) closes both. This class is
shared with the live engine (core/straddle_trader.py) so the day-boundary
entry-trigger logic can't drift between backtest and live -- the exact
mismatch this repo's other backtests have been burned by before.

If either leg fails to resolve a contract (no premium data for that day),
the day's entry is abandoned via strategy.unfire_today() -- same recovery
path the live engine uses after a failed leg.

Run:  python scripts/backtest_straddle.py --days 7
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
from deltabot.strategy.straddle import StraddleStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _ist_mins(ts: int) -> int:
    d = datetime.fromtimestamp(ts, tz=_IST)
    return d.hour * 60 + d.minute


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = StraddleStrategy(
        entry_hour=args.entry_hour, entry_minute=args.entry_minute,
        entry_grace_minutes=args.entry_grace_minutes,
    )
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots
    lot_size = args.lot_size if args.lot_size > 0 else op.LOT_SIZE.get(underlying, op.LOT_BTC)
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    bar_seconds = op.RES_SECONDS.get(args.resolution, 300)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor

    cache: dict = {}
    trades: list[dict] = []
    pos_call: dict | None = None
    pos_put: dict | None = None

    def open_leg(client, ts: int, btc_px: float, otype: OptionType) -> dict | None:
        expiry = op.select_expiry_date(ts, cutoff)
        resolved = op.resolve_by_premium(
            client, underlying, otype, btc_px, expiry, interval,
            args.target_premium, ts, ts, ts - 86400, ts + 2 * 86400,
            args.opt_resolution, step, cache,
        )
        if resolved is None:
            return None
        sym, _, ocandles = resolved
        entry_prem = op.premium_at(ocandles, ts, step)
        if entry_prem is None:
            return None
        if floor:
            entry_prem = max(entry_prem, op.intrinsic_value(sym, btc_px))
        return {"otype": otype, "sym": sym, "candles": ocandles, "entry_time": ts,
                "entry_btc": btc_px, "entry_prem": entry_prem}

    def mark_at(pos: dict, ts: int, exit_btc: float) -> float:
        p = op.premium_at(pos["candles"], ts, step)
        return p if p is not None else op.intrinsic_value(pos["sym"], exit_btc)

    def close(pos: dict, reason: str, exit_prem: float, exit_time: int, exit_btc: float) -> None:
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        # BUY side: paid entry_fill, receive exit_fill; profit when premium ROSE.
        entry_fill = pos["entry_prem"] * (1 + es)
        exit_fill = exit_prem * (1 - xs)
        gross = (exit_fill - entry_fill) * lots * lot_size
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots, lot_size)
               + op.side_fee(exit_btc, exit_fill, lots, lot_size))
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time,
            "signal": "BUY CE" if pos["otype"] == OptionType.CALL else "BUY PE",
            "contract": pos["sym"],
            "btc_entry": pos["entry_btc"], "btc_exit": exit_btc,
            "opt_in": entry_fill, "opt_out": exit_fill, "reason": reason,
            "gross": gross, "fee": fee, "net": gross - fee,
        })

    sq_mins = args.square_off_hour * 60 + args.square_off_minute
    prev_mins: int | None = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            mins = _ist_mins(c.start_time)
            square_off = prev_mins is not None and mins >= sq_mins and prev_mins < sq_mins
            prev_mins = mins

            # Signal is a wall-clock trigger, genuinely known the instant the
            # bar closes -- decision_ts matches the OTHER backtests' "acted on
            # at the just-closed bar's close" convention for pricing lookups.
            should_fire = strategy.update(c)
            decision_ts = c.start_time + bar_seconds

            # TP check: EITHER leg reaching --exit-target closes BOTH.
            if pos_call is not None or pos_put is not None:
                target = args.exit_target
                hit = None
                if pos_call is not None and mark_at(pos_call, decision_ts, c.close) >= target:
                    hit = "call"
                elif pos_put is not None and mark_at(pos_put, decision_ts, c.close) >= target:
                    hit = "put"
                if hit is not None:
                    winner, other = (pos_call, pos_put) if hit == "call" else (pos_put, pos_call)
                    close(winner, "TARGET", mark_at(winner, decision_ts, c.close), decision_ts, c.close)
                    if other is not None:
                        close(other, "PAIR", mark_at(other, decision_ts, c.close), decision_ts, c.close)
                    pos_call = pos_put = None

            # Daily square-off: closes any still-open leg(s) so they never ride
            # on stale premium data past resolve_by_premium's ~3-day window.
            if square_off:
                if pos_call is not None:
                    close(pos_call, "EOD", mark_at(pos_call, c.start_time, c.close), c.start_time, c.close)
                    pos_call = None
                if pos_put is not None:
                    close(pos_put, "EOD", mark_at(pos_put, c.start_time, c.close), c.start_time, c.close)
                    pos_put = None

            if should_fire and c.start_time >= sim_start and pos_call is None and pos_put is None:
                new_call = open_leg(client, decision_ts, c.close, OptionType.CALL)
                new_put = open_leg(client, decision_ts, c.close, OptionType.PUT)
                if new_call is None or new_put is None:
                    strategy.unfire_today()   # let a later candle this grace window retry both
                else:
                    pos_call, pos_put = new_call, new_put

        last = candles[-1]
        last_ts = last.start_time + bar_seconds
        if pos_call is not None:
            close(pos_call, "OPEN_AT_END", mark_at(pos_call, last_ts, last.close), last_ts, last.close)
        if pos_put is not None:
            close(pos_put, "OPEN_AT_END", mark_at(pos_put, last_ts, last.close), last_ts, last.close)

    trades.sort(key=lambda t: t["entry_time"])
    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 112}")
    print(f"Straddle4pm [OPTION BUY] -- {args.days}d, entry {args.entry_hour:02d}:{args.entry_minute:02d} IST, "
          f"premium ~{args.target_premium:.0f}, exit target {args.exit_target:.0f}, {args.lots} lots, "
          f"floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 112}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'signal':<8}{'contract':<22}{'btc in':>9}{'btc out':>9}"
          f"{'opt in':>8}{'opt out':>8} {'reason':<12}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{t['signal']:<8}{t['contract']:<22}"
              f"{t['btc_entry']:>9.1f}{t['btc_exit']:>9.1f}{t['opt_in']:>8.1f}{t['opt_out']:>8.1f} "
              f"{t['reason']:<12}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 112}")
    print(f"Legs: {len(closed)} closed" + (f" (+{len(trades) - len(closed)} still open at data end)"
                                            if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("TARGET", "PAIR", "EOD", "OPEN_AT_END"):
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
            "signal": t["signal"], "contract": t["contract"],
            "btc_entry": round(t["btc_entry"], 1), "btc_exit": round(t["btc_exit"], 1),
            "opt_bought": round(t["opt_in"], 1), "opt_sold": round(t["opt_out"], 1),
            "exit_reason": t["reason"], "gross_usd": round(t["gross"], 2),
            "fee_usd": round(t["fee"], 2), "net_usd": round(t["net"], 2),
            "cumulative_net_usd": round(cum, 2),
        })
    df = pd.DataFrame(rows)
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    srows = [
        ("days", args.days), ("entry_hour", args.entry_hour), ("entry_minute", args.entry_minute),
        ("target_premium", args.target_premium), ("exit_target", args.exit_target),
        ("lots", args.lots), ("legs_closed", len(closed)),
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
    p = argparse.ArgumentParser(description="Straddle4pm backtest -- option-buy execution")
    p.add_argument("--days", type=float, default=7, help="look-back window in days")
    p.add_argument("--warmup-days", type=int, default=1,
                   help="extra leading days (this strategy has no indicator to warm up -- kept "
                        "small only so the very first sim day isn't the very first candle fetched)")
    p.add_argument("--resolution", default="5m")
    p.add_argument("--entry-hour", type=int, default=16)
    p.add_argument("--entry-minute", type=int, default=0)
    p.add_argument("--entry-grace-minutes", type=int, default=15,
                   help="if the entry bar itself is unpriceable, retry on later bars within "
                        "this many minutes of entry-hour:minute before giving up for the day")
    p.add_argument("--square-off-hour", type=int, default=17)
    p.add_argument("--square-off-minute", type=int, default=25)
    p.add_argument("--exit-target", type=float, default=250.0,
                   help="FIXED ABSOLUTE premium (not a %% of entry) -- either leg reaching this "
                        "closes BOTH legs")
    p.add_argument("--target-premium", type=float, default=100.0)
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lot-size", type=float, default=0.0, help="0 = auto-derive from the underlying symbol")
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true",
                   help="disable intrinsic-value flooring (NOT recommended)")
    p.add_argument("--out", default="", help="also write every leg to this .xlsx file")
    p.add_argument("--json-out", default="", help="write trades to this .json file")
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
    if args.json_out:
        import json
        with open(args.json_out, "w") as f:
            json.dump(trades, f)
        print(f"\nJSON written: {args.json_out}  ({len(trades)} legs)")


if __name__ == "__main__":
    main()
