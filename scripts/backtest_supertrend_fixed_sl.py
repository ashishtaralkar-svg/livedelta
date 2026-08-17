"""Backtest: Supertrend Fixed-SL -- option-SELL execution of
src/deltabot/strategy/supertrend_fixed_sl.py (Python port of
supertrend_fixed_sl_strategy.pine).

A bearish Supertrend(10,3) flip sells a CALL (CE) near --target-premium
(default 1000); a bullish flip sells a PUT (PE). SL = Supertrend's own value
AT THE FLIP, frozen for the life of that leg -- never re-derived from
Supertrend's still-updating value. No profit target. Unlike the Pine chart
(which can only hold one net BTC position), this backtest tracks the CE and
PE legs as the independent contracts they are -- both can be open at once.
Daily --square-off-hour/minute (default 17:25 IST, matching every other
backtest in this repo) force-closes any still-open leg(s) -- added because
each leg's premium history is only fetched for a ~3-day window around entry
(see resolve_by_premium's win_start/win_end), so with no profit target and
only a Supertrend flip to exit, a leg that never gets an opposing flip would
otherwise sit "open" against stale, frozen premium data for however long the
backtest runs. Intrinsic-value flooring is ON by default.

Run:  python scripts/backtest_supertrend_fixed_sl.py --days 7
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
from deltabot.strategy.supertrend_fixed_sl import SupertrendFixedSlStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _ist_mins(ts: int) -> int:
    d = datetime.fromtimestamp(ts, tz=_IST)
    return d.hour * 60 + d.minute


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = SupertrendFixedSlStrategy(
        atr_period=args.atr_period, factor=args.factor,
        gap_start_hour=args.gap_start_hour, gap_start_minute=args.gap_start_minute,
        gap_end_hour=args.gap_end_hour, gap_end_minute=args.gap_end_minute,
        trade_ce=not args.pe_only, trade_pe=not args.ce_only,
    )
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots
    lot_size = args.lot_size if args.lot_size > 0 else op.LOT_SIZE.get(underlying, op.LOT_BTC)
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor

    cache: dict = {}
    trades: list[dict] = []
    pos_short: dict | None = None   # CE leg
    pos_long: dict | None = None    # PE leg

    def open_leg(client, ts: int, btc_px: float, is_short: bool) -> dict | None:
        # is_short (CE/bearish) -> sell CALL; else (PE/bullish) -> sell PUT.
        otype = OptionType.CALL if is_short else OptionType.PUT
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
        return {"is_short": is_short, "sym": sym, "candles": ocandles, "entry_time": ts,
                "entry_btc": btc_px, "entry_prem": entry_prem}

    def buyback_prem(pos: dict, ts: int, exit_btc: float) -> float:
        p = op.premium_at(pos["candles"], ts, step)
        return p if p is not None else op.intrinsic_value(pos["sym"], exit_btc)

    def close(pos: dict, reason: str, exit_prem: float, exit_time: int, exit_btc: float) -> None:
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        entry_fill = pos["entry_prem"] * (1 - es)
        exit_fill = exit_prem * (1 + xs)
        gross = (entry_fill - exit_fill) * lots * lot_size
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots, lot_size)
               + op.side_fee(exit_btc, exit_fill, lots, lot_size))
        trade = {
            "entry_time": pos["entry_time"], "exit_time": exit_time,
            "signal": "SELL" if pos["is_short"] else "BUY", "contract": pos["sym"],
            "btc_entry": pos["entry_btc"], "btc_exit": exit_btc,
            "opt_in": entry_fill, "opt_out": exit_fill, "reason": reason,
            "gross": gross, "fee": fee, "net": gross - fee,
        }
        if args.capture_charts:
            win_start, win_end = pos["entry_time"] - 1800, exit_time + 1800
            trade["chart"] = [
                {"t": t, "o": cd.open, "h": cd.high, "l": cd.low, "c": cd.close}
                for t, cd in sorted(pos["candles"].items()) if win_start <= t <= win_end
            ]
        trades.append(trade)

    sq_mins = args.square_off_hour * 60 + args.square_off_minute
    prev_mins: int | None = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            mins = _ist_mins(c.start_time)
            square_off = prev_mins is not None and mins >= sq_mins and prev_mins < sq_mins
            prev_mins = mins

            dec = strategy.update(c)

            if pos_short is not None and dec is not None and dec.short_exit:
                close(pos_short, "SL", buyback_prem(pos_short, c.start_time, dec.short_exit_price),
                      c.start_time, dec.short_exit_price)
                pos_short = None
            if pos_long is not None and dec is not None and dec.long_exit:
                close(pos_long, "SL", buyback_prem(pos_long, c.start_time, dec.long_exit_price),
                      c.start_time, dec.long_exit_price)
                pos_long = None

            # Daily square-off: force-closes any still-open leg(s) so they
            # never ride on stale premium data past resolve_by_premium's
            # ~3-day fetch window (see module docstring).
            if square_off:
                if pos_short is not None:
                    close(pos_short, "EOD", buyback_prem(pos_short, c.start_time, c.close), c.start_time, c.close)
                    pos_short = None
                    strategy.force_flat_short()
                if pos_long is not None:
                    close(pos_long, "EOD", buyback_prem(pos_long, c.start_time, c.close), c.start_time, c.close)
                    pos_long = None
                    strategy.force_flat_long()

            if dec is not None and dec.sell_signal and pos_short is None:
                if c.start_time < sim_start:
                    strategy.force_flat_short()
                else:
                    pos_short = open_leg(client, c.start_time, c.close, is_short=True)
                    if pos_short is None:
                        strategy.force_flat_short()
            if dec is not None and dec.buy_signal and pos_long is None:
                if c.start_time < sim_start:
                    strategy.force_flat_long()
                else:
                    pos_long = open_leg(client, c.start_time, c.close, is_short=False)
                    if pos_long is None:
                        strategy.force_flat_long()

        last = candles[-1]
        if pos_short is not None:
            close(pos_short, "OPEN_AT_END", buyback_prem(pos_short, last.start_time, last.close),
                  last.start_time, last.close)
        if pos_long is not None:
            close(pos_long, "OPEN_AT_END", buyback_prem(pos_long, last.start_time, last.close),
                  last.start_time, last.close)

    trades.sort(key=lambda t: t["entry_time"])
    return trades


def report(trades: list[dict], args) -> None:
    side_txt = " [CE ONLY]" if args.ce_only else " [PE ONLY]" if args.pe_only else ""
    print(f"\n{'=' * 118}")
    print(f"Supertrend({args.atr_period},{args.factor:.0f}) Fixed-SL [OPTION SELL]{side_txt} -- {args.days}d, "
          f"premium ~{args.target_premium:.0f}, {args.lots} lots, floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 118}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'sig':<5}{'contract':<22}{'btc in':>9}{'btc out':>9}"
          f"{'opt in':>8}{'opt out':>8} {'reason':<12}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{t['signal']:<5}{t['contract']:<22}"
              f"{t['btc_entry']:>9.1f}{t['btc_exit']:>9.1f}{t['opt_in']:>8.1f}{t['opt_out']:>8.1f} "
              f"{t['reason']:<12}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 118}")
    print(f"Legs: {len(closed)} closed" + (f" (+{len(trades) - len(closed)} still open at data end)"
                                            if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "EOD", "OPEN_AT_END"):
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
            "opt_sold": round(t["opt_in"], 1), "opt_bought_back": round(t["opt_out"], 1),
            "exit_reason": t["reason"], "gross_usd": round(t["gross"], 2),
            "fee_usd": round(t["fee"], 2), "net_usd": round(t["net"], 2),
            "cumulative_net_usd": round(cum, 2),
        })
    df = pd.DataFrame(rows)
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    srows = [
        ("days", args.days), ("atr_period", args.atr_period), ("factor", args.factor),
        ("premium_target", args.target_premium), ("lots", args.lots),
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
    p = argparse.ArgumentParser(description="Supertrend Fixed-SL backtest -- option-sell execution")
    p.add_argument("--days", type=float, default=7, help="look-back window in days")
    p.add_argument("--warmup-days", type=int, default=3,
                   help="extra leading days so ATR/Supertrend are warm before the report window")
    p.add_argument("--resolution", default="5m")
    p.add_argument("--atr-period", type=int, default=10)
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--gap-start-hour", type=int, default=17)
    p.add_argument("--gap-start-minute", type=int, default=25)
    p.add_argument("--gap-end-hour", type=int, default=17)
    p.add_argument("--gap-end-minute", type=int, default=30)
    p.add_argument("--square-off-hour", type=int, default=17)
    p.add_argument("--square-off-minute", type=int, default=25)
    p.add_argument("--target-premium", type=float, default=1000.0)
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lot-size", type=float, default=0.0, help="0 = auto-derive from the underlying symbol")
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true",
                   help="disable intrinsic-value flooring (NOT recommended)")
    p.add_argument("--out", default="", help="also write every leg to this .xlsx file")
    p.add_argument("--capture-charts", action="store_true",
                   help="attach each leg's real 1m option-premium candles (entry-1800s to exit+1800s) "
                        "for charting -- fetched anyway via resolve_by_premium, just retained instead of discarded")
    p.add_argument("--json-out", default="", help="write trades (with --capture-charts data if set) to this .json file")
    p.add_argument("--ce-only", action="store_true", help="only take bearish flips (sell CE) -- bullish flips are ignored")
    p.add_argument("--pe-only", action="store_true", help="only take bullish flips (sell PE) -- bearish flips are ignored")
    args = p.parse_args()
    if args.ce_only and args.pe_only:
        raise SystemExit("--ce-only and --pe-only are mutually exclusive")

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
