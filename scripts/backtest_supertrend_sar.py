"""Backtest: Supertrend Stop-And-Reverse (SAR) -- option-SELL (default) or
--side buy execution of src/deltabot/strategy/supertrend_sar.py.

Each session, the candle that closes at --start-hour:--start-minute
(default 17:35 IST) arms a single position by its own color -- green ->
BUY/PE, red -> SELL/CE (Supertrend is NOT consulted for direction, only
for this first leg's SL). Passing --orb-enabled switches this to an
Opening Range Breakout instead (v7, opt-in): the --orb-start-hour/minute
to --orb-end-hour/minute window (default 17:30-18:00 IST) is watched as a
range, and the first candle at/after --orb-end-hour/minute whose high/low
breaks it arms the position -- broke below -> SELL/CE, broke above ->
BUY/PE; --start-hour/minute are then unused. See
src/deltabot/strategy/supertrend_sar.py's v7/v9 history notes for why ORB
is opt-in rather than the default (real backtests came back mixed-to-worse
against the candle-color baseline on matched windows). Either way,
Supertrend's value is still used for this first leg's SL. The instant real
price crosses the frozen SL, the leg closes AND the OPPOSITE side opens
immediately with a fresh SL (the running day-low/day-high, not a new
Supertrend read) -- this repeats all session (unbounded) until the daily
--square-off-hour/minute (default 17:25 IST) force-closes whatever's open.
Strict single position (never CE+PE simultaneously, unlike
supertrend_fixed_sl.py). Intrinsic-value flooring is ON by default. The
STRATEGY's own SL/reversal logic is entirely BTC-price based and identical
regardless of --side -- only the OPTION EXECUTION below (which type gets
traded, P&L sign, slippage direction) changes.

NOTE: if you DO pass --orb-enabled, --orb-start-hour/minute defaults to
the SAME moment as --restart-hour/minute (17:30) -- on any day the
strategy is already flat right at 17:30 (not carrying a position over
from the prior session), evening restart will normally claim that slot
before the ORB range even finishes building, since restart only needs the
clock to reach restart_mins (no range-break condition to wait for). Push
--restart-hour/minute later (e.g. 18:05) if you want ORB to actually get
a chance to fire most sessions.

--SIDE {sell,buy} (default sell): sell = SELL the option, profit off
premium DECAY, bearish->sell CALL/bullish->sell PUT (unlimited-ish loss,
margin required, matches every other bot in this fleet except dcv3/ema21).
buy = BUY the option instead, profit off a premium RALLY, bearish->buy
PUT/bullish->buy CALL (max loss capped at the premium paid, no margin) --
mirrors OptionsExecutor._option_type_for's own buy-side mapping.

ROLL-ON-TARGET (--tp-pct, default 50.0, 0 disables): purely an OPTION-level
mechanic layered on top of the strategy above, which itself still has no
profit target of its own -- src/deltabot/strategy/supertrend_sar.py never
sees this. --side sell: every bar a leg is open, once its live buyback
premium decays to --tp-pct percent of what it was originally sold for,
book the profit ("TP") and IMMEDIATELY resell a fresh contract in the SAME
direction at --target-premium again. --side buy: the mirror -- once the
premium RISES by --tp-pct percent of entry (e.g. entry~400, --tp-pct 200
-> target ~1200), book the profit and immediately rebuy the same direction
at --target-premium again. Either way the underlying frozen SL (tracked
entirely inside `strategy`, in BTC terms) is untouched by a roll -- only
which OPTION contract is held changes, not the SL level or the strategy's
own state, so a roll can happen any number of times before the next
SL-hit/reversal or square-off.

Timing: strategy.update(c) evaluates the JUST-CLOSED candle `c` -- an SL
cross (and the reversal entry that follows it) is only genuinely known
once `c` closes, i.e. at c.start_time + bar_seconds, matching when live
would actually act. Both the recorded time AND the option-premium lookup
use this moment for signal-driven entries/exits (including TP rolls); the
daily square-off is wall-clock/independent of this and uses c.start_time
directly (same distinction established in backtest_supertrend_fixed_sl.py
and applied here from the start).

Run:  python scripts/backtest_supertrend_sar.py --days 7
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
from deltabot.strategy.supertrend_sar import SupertrendSarStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _ist_mins(ts: int) -> int:
    d = datetime.fromtimestamp(ts, tz=_IST)
    return d.hour * 60 + d.minute


# Weekend blackout window: Friday 17:35 IST through Sunday 17:35 IST (a
# rolling 48h block, not just "skip Sat/Sun") -- only blocks FRESH entries
# (the day's first entry, or the evening restart); a same-bar stop-and-
# reverse is NEVER blocked by this, since it's just closing out already-
# open risk, not opening new exposure (same principle already used for
# DELTA_SKIP_WEEKDAYS in the live engine's _entries_blocked()).
_BLACKOUT_START_MINS = 17 * 60 + 35   # Friday 17:35
_BLACKOUT_END_MINS = 17 * 60 + 35     # Sunday 17:35


def _in_weekend_blackout(ts: int) -> bool:
    d = datetime.fromtimestamp(ts, tz=_IST)
    wd = d.weekday()   # Mon=0 ... Sun=6
    mins = d.hour * 60 + d.minute
    if wd == 4:   # Friday
        return mins >= _BLACKOUT_START_MINS
    if wd == 5:   # Saturday -- blacked out all day
        return True
    if wd == 6:   # Sunday
        return mins < _BLACKOUT_END_MINS
    return False


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = SupertrendSarStrategy(
        atr_period=args.atr_period, factor=args.factor,
        start_hour=args.start_hour, start_minute=args.start_minute,
        reset_hour=args.reset_hour, reset_minute=args.reset_minute,
        min_sl_atr_mult=args.min_sl_atr_mult,
        restart_hour=args.restart_hour, restart_minute=args.restart_minute,
        orb_enabled=args.orb_enabled,
        orb_start_hour=args.orb_start_hour, orb_start_minute=args.orb_start_minute,
        orb_end_hour=args.orb_end_hour, orb_end_minute=args.orb_end_minute,
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
    is_buy = args.side == "buy"
    # SELL: tp_pct is a DECAY target (book profit once premium falls to this
    # % of entry). BUY: tp_pct is a GAIN target (book profit once premium
    # RISES to entry * (1 + tp_pct/100)) -- same field, same convention this
    # repo's live buy-mode bots (ema21bot) already use for take_profit_pct.
    tp_frac = args.tp_pct / 100.0 if args.tp_pct > 0 else None

    cache: dict = {}
    trades: list[dict] = []
    pos: dict | None = None

    def open_leg(client, ts: int, btc_px: float, is_short: bool) -> bool:
        # SELL: is_short (CE/bearish) -> sell CALL; not is_short (PE/bullish) -> sell PUT.
        # BUY:  is_short (bearish)    -> buy PUT;   not is_short (bullish)    -> buy CALL
        # (mirrors OptionsExecutor._option_type_for's own buy/sell mapping).
        nonlocal pos
        if is_buy:
            otype = OptionType.PUT if is_short else OptionType.CALL
        else:
            otype = OptionType.CALL if is_short else OptionType.PUT
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
        if is_buy:
            entry_fill = pos["entry_prem"] * (1 + es)   # buying: pay slightly more
            exit_fill = exit_prem * (1 - xs)             # selling back: receive slightly less
            gross = (exit_fill - entry_fill) * lots * lot_size   # profit when premium RISES
        else:
            entry_fill = pos["entry_prem"] * (1 - es)   # selling: receive slightly less
            exit_fill = exit_prem * (1 + xs)             # buying back: pay slightly more
            gross = (entry_fill - exit_fill) * lots * lot_size   # profit when premium DECAYS
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots, lot_size)
               + op.side_fee(exit_btc, exit_fill, lots, lot_size))
        action_word = "BUY" if is_buy else "SELL"
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time, "contract": pos["sym"],
            "action": f"{action_word} {'CE' if pos['is_call'] else 'PE'}",
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
            # dec reflects the JUST-CLOSED candle `c` -- an SL cross (and
            # the reversal that follows it) is only genuinely known at
            # c.start_time + bar_seconds (c's CLOSE), matching when live
            # would actually act. EOD below is different: a wall-clock
            # event that genuinely fires AT c.start_time.
            decision_ts = c.start_time + bar_seconds

            if pos is not None and dec is not None and dec.has_exit:
                close("SL", buyback_prem(decision_ts, dec.exit_price), decision_ts, dec.exit_price)

            # Stop-and-reverse: the SAME Decision that just closed the old
            # leg (above) also carries the reversal's entry signal -- pos
            # is None again by the time this runs, so it opens right away,
            # same bar, same decision_ts. A FRESH entry (day's-first or
            # evening-restart, i.e. not a same-bar reversal) is the only
            # kind --weekend-blackout can block -- a reversal is just
            # closing out already-open risk, never new exposure.
            if pos is None and dec is not None and dec.has_entry:
                is_fresh_entry = not dec.has_exit
                if c.start_time < sim_start:
                    strategy.force_flat()   # warmup-window entry: don't take it
                elif (args.weekend_blackout and is_fresh_entry
                      and _in_weekend_blackout(decision_ts)):
                    strategy.force_flat()   # blacked-out fresh entry: skip, retry next bar
                elif not open_leg(client, decision_ts, c.close, dec.entry_is_short):
                    strategy.force_flat()   # couldn't price the contract; stay flat, retry next bar

            # Daily square-off: force-closes any still-open leg so it never
            # rides on stale premium data past resolve_by_premium's ~3-day
            # fetch window, and so the SAR cycle genuinely stops for the day.
            if pos is not None and square_off:
                close("EOD", buyback_prem(c.start_time, c.close), c.start_time, c.close)
                strategy.force_flat()

            # Roll-on-target: purely an OPTION-level mechanic, invisible to
            # `strategy` -- its own frozen SL (in BTC terms) is never
            # touched here, so a roll can repeat any number of times before
            # the next real SL-hit/reversal or square-off closes the BTC-
            # level position for real.
            if pos is not None and tp_frac is not None:
                cur_prem = buyback_prem(decision_ts, c.close)
                tp_hit = (cur_prem >= pos["entry_prem"] * (1 + tp_frac)) if is_buy \
                    else (cur_prem <= pos["entry_prem"] * tp_frac)
                if tp_hit:
                    was_short = pos["is_short"]
                    close("TP", cur_prem, decision_ts, c.close)
                    if not open_leg(client, decision_ts, c.close, was_short):
                        strategy.force_flat()   # couldn't re-price; give up until the next real signal

        if pos is not None:
            last = candles[-1]
            last_ts = last.start_time + bar_seconds
            close("OPEN_AT_END", buyback_prem(last_ts, last.close), last_ts, last.close)

    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 108}")
    is_buy = args.side == "buy"
    if args.tp_pct > 0:
        tp_desc = (f"TP +{args.tp_pct:.0f}%->roll" if is_buy else f"TP {args.tp_pct:.0f}%->roll")
    else:
        tp_desc = "no TP (pure SAR)"
    sl_desc = f"minSL {args.min_sl_atr_mult:.1f}xATR" if args.min_sl_atr_mult > 0 else "raw day-extreme SL"
    wk_desc = "Fri17:35-Sun17:35 blackout" if args.weekend_blackout else "24/7 (no blackout)"
    entry_desc = (f"ORB {args.orb_start_hour:02d}:{args.orb_start_minute:02d}-"
                  f"{args.orb_end_hour:02d}:{args.orb_end_minute:02d} IST" if args.orb_enabled
                  else f"start {args.start_hour:02d}:{args.start_minute:02d} IST (candle-color)")
    print(f"Supertrend SAR [OPTION {'BUY' if is_buy else 'SELL'}] -- {args.days}d, {args.resolution}, "
          f"Supertrend({args.atr_period},{args.factor:.0f}), {entry_desc}, "
          f"restart {args.restart_hour:02d}:{args.restart_minute:02d} IST, "
          f"premium ~{args.target_premium:.0f}, {tp_desc}, {sl_desc}, {wk_desc}, {args.lots} lots, "
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
    for reason in ("SL", "TP", "EOD", "OPEN_AT_END"):
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
        ("days", args.days), ("resolution", args.resolution), ("side", args.side),
        ("atr_period", args.atr_period), ("factor", args.factor),
        ("start_hour", args.start_hour), ("start_minute", args.start_minute),
        ("premium_target", args.target_premium), ("tp_pct", args.tp_pct), ("lots", args.lots),
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
    p = argparse.ArgumentParser(description="Supertrend Stop-And-Reverse backtest -- option-sell execution")
    p.add_argument("--days", type=float, default=7, help="look-back window in days")
    p.add_argument("--warmup-days", type=int, default=3,
                   help="extra leading days so Supertrend's ATR is warm before the report window")
    p.add_argument("--resolution", default="5m")
    p.add_argument("--atr-period", type=int, default=10)
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--min-sl-atr-mult", type=float, default=1.0,
                   help="widen a reversal's day-low/day-high SL to at least this many ATRs from "
                        "price if it would otherwise be tighter; 0 disables (raw day-extreme SL)")
    p.add_argument("--start-hour", type=int, default=17,
                   help="v9 default (candle-color first entry): daily first-entry hour (IST). "
                        "Unused if --orb-enabled is passed (see --orb-start-hour/minute instead)")
    p.add_argument("--start-minute", type=int, default=35)
    p.add_argument("--orb-enabled", action="store_true",
                   help="v7 opt-in: switch the session's first entry to an Opening Range Breakout "
                        "of --orb-start-hour/minute to --orb-end-hour/minute instead of the default "
                        "candle-color rule at --start-hour/minute. Real backtests came back "
                        "mixed-to-worse vs the candle-color default on matched windows, and it also "
                        "needs --restart-hour/minute pushed later (e.g. 18:05) to avoid colliding "
                        "with its own range window -- off by default")
    p.add_argument("--orb-start-hour", type=int, default=17,
                   help="v7 Opening Range Breakout: watch this hour:minute (IST) as the start of "
                        "the day's opening range (only used with --orb-enabled)")
    p.add_argument("--orb-start-minute", type=int, default=30)
    p.add_argument("--orb-end-hour", type=int, default=18,
                   help="v7 ORB: the range closes at this hour:minute (IST); the first candle at/"
                        "after this that breaks the range arms the session's first position")
    p.add_argument("--orb-end-minute", type=int, default=0)
    p.add_argument("--reset-hour", type=int, default=5,
                   help="session reset hour (IST) -- drives day-high/day-low and the 'first entry "
                        "already fired' flag. Default 05:30 (the ORIGINAL/deployed value). v8 tried "
                        "moving this to 17:30 so a reversal's SL could never reflect a stale extreme "
                        "from earlier the same calendar day -- more theoretically correct, but it "
                        "backtested WORSE on every 1wk/1mo/3mo window re-tested (more legs, lower win "
                        "rate), so it was reverted; pass --reset-hour 17 to reproduce that variant. "
                        "Independent of --orb-enabled")
    p.add_argument("--reset-minute", type=int, default=30)
    p.add_argument("--square-off-hour", type=int, default=17,
                   help="daily force-close hour (IST) for any still-open position")
    p.add_argument("--square-off-minute", type=int, default=25)
    p.add_argument("--restart-hour", type=int, default=17,
                   help="evening restart hour (IST) -- resume trading same-direction this many "
                        "hours:minutes after square-off, instead of waiting for next session's first "
                        "entry. Default 17:30 (5 min after the usual 17:25 square-off). If "
                        "--orb-enabled is also passed, consider moving this later (e.g. 18:05, 5 min "
                        "after --orb-end-hour/minute) or restart will normally win the race and the "
                        "ORB range will rarely get a chance to finish building first")
    p.add_argument("--restart-minute", type=int, default=30)
    p.add_argument("--weekend-blackout", action="store_true",
                   help="block FRESH entries (day's-first-entry / evening-restart, never a "
                        "same-bar reversal) from Friday 17:35 IST through Sunday 17:35 IST")
    p.add_argument("--side", choices=["sell", "buy"], default="sell",
                   help="sell (default): sell the option, profit off premium DECAY. buy: buy the "
                        "option instead (bearish -> buy PUT, bullish -> buy CALL), profit off a "
                        "premium RALLY -- max loss capped at the premium paid, no margin concept")
    p.add_argument("--target-premium", type=float, default=600.0)
    p.add_argument("--tp-pct", type=float, default=50.0,
                   help="--side sell: book profit + immediately resell same-direction once the held "
                        "premium decays to this pct of entry. --side buy: book profit + immediately "
                        "rebuy same-direction once premium RISES by this pct of entry (e.g. 200 -> "
                        "entry~400 needs to reach ~1200). 0 disables (pure SAR, no roll)")
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
