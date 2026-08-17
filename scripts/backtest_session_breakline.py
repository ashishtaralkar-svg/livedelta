"""Backtest: Session Breakline -- option-SELL execution of
src/deltabot/strategy/session_breakline.py (Python port of
session_breakline_strategy.pine).

BUY signal -> SELL a PUT near --target-premium (default 600). SELL signal ->
SELL a CALL. No target, no rollover -- the strategy's own daily square-off
(05:25 IST, 5 minutes before the next session line) always closes the trade
before the sold option would ever need re-selling (the option bought at
entry is priced off whichever daily expiry is next, which always falls AFTER
that square-off). Intrinsic-value flooring is ON by default (see
backtest-intrinsic-floor: illiquid ITM candles print below intrinsic and
inflate results ~5x otherwise).

Run:  python scripts/backtest_session_breakline.py --days 7
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
from deltabot.strategy.session_breakline import SessionBreaklineStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = SessionBreaklineStrategy(
        sess_hour=args.sess_hour, sess_minute=args.sess_minute,
        sq_off_hour=args.sq_off_hour, sq_off_minute=args.sq_off_minute,
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
    pos: dict | None = None

    def open_leg(client, ts: int, btc_px: float, is_buy: bool, tag: str) -> bool:
        nonlocal pos
        # bullish signal (buy_signal) -> sell PUT; bearish -> sell CALL.
        otype = OptionType.PUT if is_buy else OptionType.CALL
        expiry = op.select_expiry_date(ts, cutoff)
        resolved = op.resolve_by_premium(
            client, underlying, otype, btc_px, expiry, interval,
            args.target_premium, ts, ts, ts - 86400, ts + 2 * 86400,
            args.opt_resolution, step, cache,
        )
        if resolved is None:
            return False
        sym, _, ocandles = resolved
        entry_prem = op.premium_at(ocandles, ts, step)
        if entry_prem is None:
            return False
        if floor:
            entry_prem = max(entry_prem, op.intrinsic_value(sym, btc_px))
        pos = {"is_buy": is_buy, "sym": sym, "candles": ocandles, "entry_time": ts,
               "entry_btc": btc_px, "entry_prem": entry_prem, "tag": tag}
        return True

    def buyback_prem(ts: int, exit_btc: float) -> float:
        p = op.premium_at(pos["candles"], ts, step)
        return p if p is not None else op.intrinsic_value(pos["sym"], exit_btc)

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float) -> None:
        nonlocal pos
        assert pos is not None
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        entry_fill = pos["entry_prem"] * (1 - es)   # selling: receive less on entry
        exit_fill = exit_prem * (1 + xs)            # buying back: pay more
        gross = (entry_fill - exit_fill) * lots * lot_size   # profit when premium DECAYS
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots, lot_size)
               + op.side_fee(exit_btc, exit_fill, lots, lot_size))
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time,
            "signal": "BUY" if pos["is_buy"] else "SELL", "contract": pos["sym"],
            "tag": pos["tag"], "btc_entry": pos["entry_btc"], "btc_exit": exit_btc,
            "opt_in": entry_fill, "opt_out": exit_fill, "reason": reason,
            "gross": gross, "fee": fee, "net": gross - fee,
        })
        pos = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            dec = strategy.update(c)

            if pos is not None and dec is not None and dec.has_exit:
                eprice = dec.long_exit_price if dec.long_exit else dec.short_exit_price
                close(dec.exit_reason, buyback_prem(c.start_time, eprice), c.start_time, eprice)

            if pos is None and dec is not None and dec.has_entry:
                if c.start_time < sim_start:
                    strategy.force_flat()   # warmup-window entry: don't take it
                elif not open_leg(client, c.start_time, c.close, dec.buy_signal, "ENTRY"):
                    strategy.force_flat()   # couldn't price the contract; stay flat

        if pos is not None:
            last = candles[-1]
            close("OPEN_AT_END", buyback_prem(last.start_time, last.close), last.start_time, last.close)

    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 118}")
    print(f"Session Breakline [OPTION SELL, no target, no rollover] -- {args.days}d, {args.resolution}, "
          f"premium ~{args.target_premium:.0f}, {args.lots} lots, "
          f"floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 118}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'sig':<5}{'contract':<22}{'btc in':>9}{'btc out':>9}"
          f"{'opt in':>8}{'opt out':>8} {'reason':<10}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{t['signal']:<5}{t['contract']:<22}"
              f"{t['btc_entry']:>9.1f}{t['btc_exit']:>9.1f}{t['opt_in']:>8.1f}{t['opt_out']:>8.1f} "
              f"{t['reason']:<10}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 118}")
    print(f"Legs: {len(closed)} closed" + (" (+1 still open at data end)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "EOD", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<12} n={len(rs):<4} net ${sum(t['net'] for t in rs):>11.2f}")
    print(f"TOTAL NET: ${sum(t['net'] for t in trades):.2f} "
          f"(gross ${sum(t['gross'] for t in trades):.2f}, fees ${sum(t['fee'] for t in trades):.2f})")


def export(trades: list[dict], args, path: str) -> None:
    """Write every leg to an .xlsx (Trades sheet + a Summary sheet), same
    layout convention as backtest_dcv2.py's export_option()."""
    import pandas as pd

    rows, cum = [], 0.0
    for i, t in enumerate(trades, 1):
        cum += t["net"]
        rows.append({
            "#": i, "entry_IST": _ist(t["entry_time"]), "exit_IST": _ist(t["exit_time"]),
            "signal": t["signal"], "contract": t["contract"],
            "btc_entry": round(t["btc_entry"], 1), "btc_exit": round(t["btc_exit"], 1),
            "opt_sold_or_bought": round(t["opt_in"], 1), "opt_closed": round(t["opt_out"], 1),
            "exit_reason": t["reason"], "gross_usd": round(t["gross"], 2),
            "fee_usd": round(t["fee"], 2), "net_usd": round(t["net"], 2),
            "cumulative_net_usd": round(cum, 2),
        })
    df = pd.DataFrame(rows)

    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    srows = [
        ("days", args.days), ("resolution", args.resolution),
        ("premium_target", args.target_premium), ("lots", args.lots),
        ("legs_closed", len(closed)),
        ("win_rate_pct", round(100.0 * len(wins) / len(closed), 1) if closed else 0.0),
        ("gross_usd", round(sum(t["gross"] for t in trades), 2)),
        ("fees_usd", round(sum(t["fee"] for t in trades), 2)),
        ("TOTAL_NET_usd", round(sum(t["net"] for t in trades), 2)),
    ]
    for reason in ("SL", "EOD", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            srows.append((f"{reason}_n", len(rs)))
            srows.append((f"{reason}_net_usd", round(sum(t["net"] for t in rs), 2)))
    summary = pd.DataFrame(srows, columns=["metric", "value"])

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        df.to_excel(xl, sheet_name="Trades", index=False)
    print(f"\nExcel written: {path}  ({len(df)} legs)")


def main() -> None:
    p = argparse.ArgumentParser(description="Session Breakline backtest -- option-sell execution")
    p.add_argument("--days", type=float, default=7, help="look-back window in days")
    p.add_argument("--warmup-days", type=int, default=1,
                   help="extra leading days so the first session line is established before the report window")
    p.add_argument("--resolution", default="5m")
    p.add_argument("--sess-hour", type=int, default=17)
    p.add_argument("--sess-minute", type=int, default=30)
    p.add_argument("--sq-off-hour", type=int, default=17)
    p.add_argument("--sq-off-minute", type=int, default=25)
    p.add_argument("--target-premium", type=float, default=600.0)
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lot-size", type=float, default=0.0, help="0 = auto-derive from the underlying symbol")
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true",
                   help="disable intrinsic-value flooring (NOT recommended)")
    p.add_argument("--out", default="", help="also write every leg to this .xlsx file")
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
