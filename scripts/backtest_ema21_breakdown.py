"""Backtest: EMA(21) Breakdown -- option-SELL execution of
src/deltabot/strategy/ema21_breakdown.py (Python port of the user-supplied
"21 EMA Breakdown Strategy (Strict Single Trade)" Pine script).

Ported mirrored on both sides now (see the strategy module) -- SELL
(bearish) signals sell a CALL (CE), BUY (bullish) signals sell a PUT (PE),
--ce-only/--pe-only to run either side alone. Buys the leg back on the
strategy's own SL/TARGET exit, an optional --tp-decay-pct premium-decay TP,
or the daily --square-off-hour/minute (default 17:25 IST, matching every
other backtest in this repo) -- ADDED beyond the source script, which has no
time exit at all. That square-off matters for correctness, not just realism:
each option leg's premium history is only fetched for a ~3-day window around
entry, so a trade that never hits SL/TP within that window would otherwise
sit "open" against stale, frozen premium data for however long the backtest
runs -- confirmed this actually happened in a 3-month test before the
square-off was added. Intrinsic-value flooring is ON by default.

Run:  python scripts/backtest_ema21_breakdown.py --days 7
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
from deltabot.strategy.ema21_breakdown import Ema21BreakdownStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _ist_mins(ts: int) -> int:
    d = datetime.fromtimestamp(ts, tz=_IST)
    return d.hour * 60 + d.minute


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = Ema21BreakdownStrategy(
        ema_len=args.ema_len, max_wait=args.max_wait, target_rr=args.target_rr,
        trade_ce=not args.pe_only, trade_pe=not args.ce_only,
        trend_filter=args.trend_filter, ema200_filter=args.ema200_filter,
        ema50_len=args.ema50_len, ema200_len=args.ema200_len,
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
    tp_decay = args.tp_decay_pct / 100.0

    cache: dict = {}
    trades: list[dict] = []
    pos: dict | None = None

    def open_leg(client, ts: int, btc_px: float, is_buy: bool) -> bool:
        nonlocal pos
        # bearish signal (sell CE) -> CALL; bullish signal (sell PE) -> PUT.
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
        # tp_price: buy back when the SOLD premium decays this %% (e.g. 70 ->
        # 30% of entry). Independent of the strategy's own BTC-price target
        # (which --target-rr 0 disables) -- see the module CLI help.
        tp_price = entry_prem * (1.0 - tp_decay) if tp_decay > 0 else None
        pos = {"is_buy": is_buy, "sym": sym, "candles": ocandles, "entry_time": ts,
               "entry_btc": btc_px, "entry_prem": entry_prem, "tp_price": tp_price}
        return True

    def buyback_prem(ts: int, exit_btc: float) -> float:
        p = op.premium_at(pos["candles"], ts, step)
        return p if p is not None else op.intrinsic_value(pos["sym"], exit_btc)

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float) -> None:
        nonlocal pos
        assert pos is not None
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        entry_fill = pos["entry_prem"] * (1 - es)
        exit_fill = exit_prem * (1 + xs)
        gross = (entry_fill - exit_fill) * lots * lot_size
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots, lot_size)
               + op.side_fee(exit_btc, exit_fill, lots, lot_size))
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time, "contract": pos["sym"],
            "signal": "BUY" if pos["is_buy"] else "SELL",
            "btc_entry": pos["entry_btc"], "btc_exit": exit_btc,
            "opt_in": entry_fill, "opt_out": exit_fill, "reason": reason,
            "gross": gross, "fee": fee, "net": gross - fee,
        })
        pos = None

    sq_mins = args.square_off_hour * 60 + args.square_off_minute
    prev_mins: int | None = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            mins = _ist_mins(c.start_time)
            square_off = prev_mins is not None and mins >= sq_mins and prev_mins < sq_mins
            prev_mins = mins

            dec = strategy.update(c)

            if pos is not None and dec is not None and dec.has_exit:
                exit_price = dec.long_exit_price if dec.long_exit else dec.short_exit_price
                close(dec.exit_reason, buyback_prem(c.start_time, exit_price), c.start_time, exit_price)

            # 70%-decay TP on the SOLD option premium (only if --tp-decay-pct
            # > 0; independent of/in addition to the strategy's own SL, which
            # always stays live off the BTC-price anchor high/low).
            if pos is not None and pos["tp_price"] is not None:
                can_decay = (not floor) or op.intrinsic_value(pos["sym"], c.close) <= pos["tp_price"]
                if can_decay:
                    t_tp = op.premium_at(pos["candles"], c.start_time, step)
                    if t_tp is not None and t_tp <= pos["tp_price"]:
                        close("TP", pos["tp_price"], c.start_time, c.close)
                        strategy.force_flat()

            # Daily square-off (default 17:25 IST, matching every other
            # backtest in this repo): force-closes any still-open position.
            # ADDED beyond the source script (which has no time exit at all)
            # -- without this, a trade that never hits its SL or TP can sit
            # open for weeks/months, well past its actual option contract's
            # real-world expiry, and the backtest's ~3-day premium fetch
            # window around entry runs out of real data long before that --
            # see conversation for the corrupted 3-month run this fixes.
            if pos is not None and square_off:
                close("EOD", buyback_prem(c.start_time, c.close), c.start_time, c.close)
                strategy.force_flat()

            if pos is None and dec is not None and dec.has_entry:
                if c.start_time < sim_start:
                    strategy.force_flat()   # warmup-window entry: don't take it
                elif not open_leg(client, c.start_time, c.close, dec.buy_signal):
                    strategy.force_flat()   # couldn't price the contract; stay flat

        if pos is not None:
            last = candles[-1]
            close("OPEN_AT_END", buyback_prem(last.start_time, last.close), last.start_time, last.close)

    return trades


def report(trades: list[dict], args) -> None:
    tp_txt = f"{args.tp_decay_pct:.0f}%-decay TP" if args.tp_decay_pct > 0 else "no premium TP"
    rr_txt = f"{args.target_rr:.1f}R BTC-price target" if args.target_rr > 0 else "no BTC-price target"
    side_txt = " [CE ONLY]" if args.ce_only else " [PE ONLY]" if args.pe_only else ""
    print(f"\n{'=' * 112}")
    print(f"EMA(21) Breakdown [OPTION SELL]{side_txt} -- {args.days}d, {args.resolution}, "
          f"premium ~{args.target_premium:.0f}, {args.lots} lots, {rr_txt}, {tp_txt}, "
          f"floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 112}")
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
    print(f"{'-' * 112}")
    print(f"Legs: {len(closed)} closed" + (" (+1 still open at data end)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "TARGET", "TP", "EOD", "OPEN_AT_END"):
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
            "signal": t["signal"],
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
        ("days", args.days), ("ema_len", args.ema_len), ("max_wait", args.max_wait),
        ("target_rr", args.target_rr), ("tp_decay_pct", args.tp_decay_pct),
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
    p = argparse.ArgumentParser(description="EMA(21) Breakdown backtest -- option-sell execution")
    p.add_argument("--days", type=float, default=7, help="look-back window in days")
    p.add_argument("--warmup-days", type=int, default=4,
                   help="extra leading days so EMA(21) (or EMA(200) with --trend-filter) is warm "
                        "before the report window")
    p.add_argument("--resolution", default="15m")
    p.add_argument("--ema-len", type=int, default=21)
    p.add_argument("--max-wait", type=int, default=3)
    p.add_argument("--square-off-hour", type=int, default=17,
                   help="daily force-close hour (IST) for any still-open position")
    p.add_argument("--square-off-minute", type=int, default=25)
    p.add_argument("--trend-filter", action="store_true",
                   help="additional gate checked AT THE TRIGGER: BUY only fires if "
                        "close > EMA50 > EMA200 (bullish stack); SELL only if "
                        "close < EMA50 < EMA200 (mirror). Disagreeing consumes the setup untraded.")
    p.add_argument("--ema200-filter", action="store_true",
                   help="lighter single-EMA gate checked AT THE TRIGGER: SELL only fires if "
                        "close < EMA200 (EMA50 irrelevant); BUY only if close > EMA200. "
                        "Independent of --trend-filter -- combine both if you want both to agree.")
    p.add_argument("--ema50-len", type=int, default=50)
    p.add_argument("--ema200-len", type=int, default=200)
    p.add_argument("--target-rr", type=float, default=2.0,
                   help="strategy's own BTC-price target, multiple of risk (entry vs SL). "
                        "0 disables it -- use with --tp-decay-pct for a premium-decay TP instead.")
    p.add_argument("--tp-decay-pct", type=float, default=0.0,
                   help="book profit when the SOLD premium decays this %% (e.g. 70 -> buy back at "
                        "30%% of entry). 0 (default) = off. Independent of --target-rr -- combine "
                        "with --target-rr 0 to make this the ONLY profit exit, or leave both on so "
                        "whichever hits first closes the trade.")
    p.add_argument("--target-premium", type=float, default=400.0)
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lot-size", type=float, default=0.0, help="0 = auto-derive from the underlying symbol")
    p.add_argument("--opt-resolution", default="1m")
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true",
                   help="disable intrinsic-value flooring (NOT recommended)")
    p.add_argument("--out", default="", help="also write every leg to this .xlsx file")
    p.add_argument("--ce-only", action="store_true", help="only take SELL (bearish) signals -- sell CE")
    p.add_argument("--pe-only", action="store_true", help="only take BUY (bullish, mirrored) signals -- sell PE")
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


if __name__ == "__main__":
    main()
