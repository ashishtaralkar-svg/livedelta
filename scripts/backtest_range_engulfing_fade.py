"""Backtest: Range Engulfing Fade -- option-SELL, ATM execution of
src/deltabot/strategy/range_engulfing_fade.py.

A bearish-fade breakout (see the strategy's own docstring) sells an
AT-THE-MONEY CALL; a bullish-fade breakdown sells an AT-THE-MONEY PUT --
strike = nearest --interval to the underlying at entry (via
op.resolve_atm(), offset=0), NOT a premium target/search like every other
backtest script in this repo. Exits via the strategy's own SL/TARGET
(both real BTC-price levels) or the daily --square-off-hour/minute
(default 17:25 IST, matching every other backtest here). Intrinsic-value
flooring is ON by default.

Per the request: BTC candle resolution defaults to 15m (pattern
detection/signals); option premium data defaults to 1m CLOSE candles
(--opt-resolution, matching every other script's own convention of using
op.premium_at()'s .close read).

Timing: strategy.update(c) evaluates the JUST-CLOSED candle `c` -- a
breakout is only genuinely known once `c` closes, i.e. at c.start_time +
bar_seconds, matching when live would actually act (same convention
applied to every other backtest fixed this session). Both the recorded
time AND the option-premium lookup use this moment for signal-driven
entries/exits; the daily square-off is wall-clock-scheduled and
deliberately keeps using c.start_time.

Run:  python scripts/backtest_range_engulfing_fade.py --days 7
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
from deltabot.strategy.range_engulfing_fade import RangeEngulfingFadeStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _ist_mins(ts: int) -> int:
    d = datetime.fromtimestamp(ts, tz=_IST)
    return d.hour * 60 + d.minute


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = RangeEngulfingFadeStrategy(trade_ce=not args.pe_only, trade_pe=not args.ce_only)
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots
    lot_size = args.lot_size if args.lot_size > 0 else op.LOT_SIZE.get(underlying, op.LOT_BTC)
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    bar_seconds = op.RES_SECONDS.get(args.resolution, 900)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor

    cache: dict = {}
    trades: list[dict] = []
    pos: dict | None = None

    def open_leg(client, ts: int, btc_px: float, is_short: bool) -> bool:
        # bearish-fade breakout (is_short=True) -> sell CALL; bullish-fade
        # breakdown (is_short=False) -> sell PUT -- this repo's standard
        # sell-mode convention.
        nonlocal pos
        otype = OptionType.CALL if is_short else OptionType.PUT
        expiry = op.select_expiry_date(ts, cutoff)
        resolved = op.resolve_atm(
            client, underlying, otype, btc_px, expiry, interval,
            ts, ts, ts - 86400, ts + 2 * 86400, args.opt_resolution, step, cache,
        )
        if resolved is None:
            return False
        sym, _, ocandles = resolved
        entry_prem = op.premium_at(ocandles, ts, step)
        if entry_prem is None:
            return False
        if floor:
            entry_prem = max(entry_prem, op.intrinsic_value(sym, btc_px))
        pos = {"is_short": is_short, "sym": sym, "is_call": otype == OptionType.CALL,
               "candles": ocandles, "entry_time": ts,
               "entry_btc": btc_px, "entry_prem": entry_prem}
        return True

    def buyback_prem(ts: int, exit_btc: float) -> float:
        p = op.premium_at(pos["candles"], ts, step)
        return p if p is not None else op.intrinsic_value(pos["sym"], exit_btc)

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float) -> None:
        nonlocal pos
        assert pos is not None
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        entry_fill = pos["entry_prem"] * (1 - es)   # selling: receive slightly less
        exit_fill = exit_prem * (1 + xs)             # buying back: pay slightly more
        gross = (entry_fill - exit_fill) * lots * lot_size   # profit when premium DECAYS
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots, lot_size)
               + op.side_fee(exit_btc, exit_fill, lots, lot_size))
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time, "contract": pos["sym"],
            "action": f"SELL {'CE' if pos['is_call'] else 'PE'}",
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
            # dec reflects the JUST-CLOSED candle `c` -- a breakout is only
            # genuinely known at c.start_time + bar_seconds (c's CLOSE),
            # matching when live would actually act. EOD below is
            # different -- a wall-clock/scheduled check, deliberately kept
            # on c.start_time.
            decision_ts = c.start_time + bar_seconds

            if pos is not None and dec is not None and dec.has_exit:
                exit_price = dec.long_exit_price if dec.long_exit else dec.short_exit_price
                reason = dec.long_exit_reason if dec.long_exit else dec.short_exit_reason
                close(reason or "SL", buyback_prem(decision_ts, exit_price), decision_ts, exit_price)

            # Daily square-off: force-closes any still-open leg so it never
            # rides on stale premium data past resolve_atm's ~3-day fetch
            # window.
            if pos is not None and square_off:
                close("EOD", buyback_prem(c.start_time, c.close), c.start_time, c.close)
                strategy.force_flat()

            if pos is None and dec is not None and dec.has_entry:
                if c.start_time < sim_start:
                    strategy.force_flat()   # warmup-window entry: don't take it
                elif not open_leg(client, decision_ts, c.close, dec.sell_signal):
                    strategy.force_flat()   # couldn't price the contract; stay flat

        if pos is not None:
            last = candles[-1]
            last_ts = last.start_time + bar_seconds
            close("OPEN_AT_END", buyback_prem(last_ts, last.close), last_ts, last.close)

    return trades


def report(trades: list[dict], args) -> None:
    side_txt = " [CE ONLY]" if args.ce_only else " [PE ONLY]" if args.pe_only else ""
    print(f"\n{'=' * 108}")
    print(f"Range Engulfing Fade [OPTION SELL, ATM]{side_txt} -- {args.days}d, {args.resolution}, "
          f"opt {args.opt_resolution}, {args.lots} lots, "
          f"floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 108}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'action':<8}{'contract':<22}{'btc in':>9}{'btc out':>9}"
          f"{'opt in':>8}{'opt out':>8} {'reason':<12}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{t['action']:<8}{t['contract']:<22}"
              f"{t['btc_entry']:>9.1f}{t['btc_exit']:>9.1f}{t['opt_in']:>8.1f}{t['opt_out']:>8.1f} "
              f"{t['reason']:<12}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 108}")
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
            "action": t["action"], "contract": t["contract"],
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
        ("days", args.days), ("resolution", args.resolution), ("opt_resolution", args.opt_resolution),
        ("lots", args.lots),
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
    p = argparse.ArgumentParser(description="Range Engulfing Fade backtest -- option-sell, ATM execution")
    p.add_argument("--days", type=float, default=7, help="look-back window in days")
    p.add_argument("--warmup-days", type=int, default=1,
                   help="extra leading days (this strategy needs almost no warmup -- just 1 prior candle)")
    p.add_argument("--resolution", default="15m", help="BTC candle resolution for pattern detection")
    p.add_argument("--opt-resolution", default="1m", help="option premium candle resolution (uses .close)")
    p.add_argument("--square-off-hour", type=int, default=17,
                   help="daily force-close hour (IST) for any still-open position")
    p.add_argument("--square-off-minute", type=int, default=25)
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lot-size", type=float, default=0.0, help="0 = auto-derive from the underlying symbol")
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true",
                   help="disable intrinsic-value flooring (NOT recommended)")
    p.add_argument("--out", default="", help="also write every leg to this .xlsx file")
    p.add_argument("--ce-only", action="store_true", help="only take bearish-fade breakouts -- sell CE")
    p.add_argument("--pe-only", action="store_true", help="only take bullish-fade breakdowns -- sell PE")
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
