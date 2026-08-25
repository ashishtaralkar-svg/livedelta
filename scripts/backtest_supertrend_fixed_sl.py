"""Backtest: Supertrend Fixed-SL -- option-SELL execution of
src/deltabot/strategy/supertrend_fixed_sl.py (Python port of
supertrend_fixed_sl_strategy.pine).

A bearish Supertrend(10,3) flip sells a CALL (CE) near --target-premium
(default 1000); a bullish flip sells a PUT (PE). SL = Supertrend's own value
AT THE FLIP, frozen for the life of that leg -- never re-derived from
Supertrend's still-updating value. Exits are that frozen SL, the daily
square-off, or the premium-decay TP (--tp-pct, default 70 -- sold at 100,
buy back at 30; 0 disables it). When EITHER leg's TP hits, BOTH legs are
blocked from new entries until --tp-block-hour/minute (default 17:30) that
same day. Unlike the Pine chart (which can only hold one net BTC position),
this backtest tracks the CE and PE legs as the independent contracts they
are -- both can be open at once. Daily --square-off-hour/minute (default
17:25 IST, matching every other backtest in this repo) force-closes any
still-open leg(s) -- added because each leg's premium history is only
fetched for a ~3-day window around entry (see resolve_by_premium's
win_start/win_end), so a leg that never gets an opposing flip, a TP, or an
SL would otherwise sit "open" against stale, frozen premium data for
however long the backtest runs. Intrinsic-value flooring is ON by default.

--ema200-filter: additional gate at the flip -- bearish (CE) only if
close < EMA200, bullish (PE) only if close > EMA200. Disagreeing consumes
the flip untraded.

--rollover: if a leg is closed ONLY by the daily square-off (its frozen SL
was never actually crossed), immediately sell a FRESH option for the SAME
side, keeping the SAME frozen SL (the strategy's own _active_short_sl/
_active_long_sl already survive untouched as long as force_flat_short()/
force_flat_long() is never called for that side -- see the strategy
module's own docstring). A leg that closed via a real SL cross this same
bar never rolls (there's nothing left to continue). Off (default) =
unchanged behavior -- every square-off just ends the leg.

--weekend-blackout: a backtest-level scheduling concept (like
--square-off-hour/minute), NOT modeled inside the strategy class -- no NEW
entry from Saturday 17:30 IST through Sunday 23:55 IST. The daily
square-off already force-closes any open leg by 17:25 every day including
Saturday, so this window never needs to force-close anything -- it only
ever suppresses a fresh flip signal (same force_flat_short/long treatment
already used for the pre-sim-start warmup window). Off (default) =
unchanged behavior.

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


_SAT_BLACKOUT_START_MINS = 17 * 60 + 30   # Saturday 17:30 IST
_SUN_BLACKOUT_END_MINS = 23 * 60 + 55     # Sunday 23:55 IST


def _in_weekend_blackout(ts: int) -> bool:
    d = datetime.fromtimestamp(ts, tz=_IST)
    wd = d.weekday()   # Mon=0 ... Sat=5, Sun=6
    mins = d.hour * 60 + d.minute
    if wd == 5:
        return mins >= _SAT_BLACKOUT_START_MINS
    if wd == 6:
        return mins <= _SUN_BLACKOUT_END_MINS
    return False


def _tp_block_until(ts: int, hour: int, minute: int) -> int:
    """Epoch seconds for ``ts``'s own IST calendar day at ``hour:minute``. If
    ``ts`` is already at/past that time (square-off already ran), returns
    ``ts`` itself -- a no-op block, mirroring the live engine's edge case."""
    d = datetime.fromtimestamp(ts, tz=_IST)
    target = d.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return ts if d >= target else int(target.timestamp())


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = SupertrendFixedSlStrategy(
        atr_period=args.atr_period, factor=args.factor,
        gap_start_hour=args.gap_start_hour, gap_start_minute=args.gap_start_minute,
        gap_end_hour=args.gap_end_hour, gap_end_minute=args.gap_end_minute,
        trade_ce=not args.pe_only, trade_pe=not args.ce_only,
        use_heikin_ashi=args.heikin_ashi,
        ema200_filter=args.ema200_filter, ema200_len=args.ema200_len,
        trend_filter=args.trend_filter, trend_fast_len=args.trend_fast_len,
        trend_slow_len=args.trend_slow_len, trend_flip_exit=args.trend_flip_exit,
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

    tp_mult = 1.0 - args.tp_pct / 100.0 if args.tp_pct > 0 else None

    cache: dict = {}
    trades: list[dict] = []
    pos_short: dict | None = None   # CE leg
    pos_long: dict | None = None    # PE leg
    block_until: int = 0            # epoch secs; entries blocked while ts < this

    def open_leg(client, ts: int, btc_px: float, is_short: bool, tag: str = "ENTRY") -> dict | None:
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
        tp_price = entry_prem * tp_mult if tp_mult is not None else None
        return {"is_short": is_short, "sym": sym, "candles": ocandles, "entry_time": ts,
                "entry_btc": btc_px, "entry_prem": entry_prem, "tp_price": tp_price, "tag": tag}

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
            "tag": pos.get("tag", "ENTRY"),
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
            # strategy.update(c) evaluates the JUST-CLOSED candle `c` -- the
            # signal (flip / SL cross) is only genuinely KNOWN once `c`
            # closes, i.e. at c.start_time + bar_seconds, matching when live
            # actually acts (confirmed against real Telegram timestamps).
            # Both the recorded time AND the option-premium lookup must use
            # this moment, not c.start_time (the bar's OPEN) -- pricing off
            # c.start_time was fetching the option's premium from 1 bar too
            # early. EOD/ROLL below are different: those are scheduled
            # wall-clock events that genuinely fire AT c.start_time (17:25),
            # so they deliberately do NOT get this adjustment.
            decision_ts = c.start_time + bar_seconds
            blackout = args.weekend_blackout and _in_weekend_blackout(decision_ts)

            if pos_short is not None and dec is not None and dec.short_exit:
                reason = "TREND" if dec.short_trend_exit else "SL"
                close(pos_short, reason, buyback_prem(pos_short, decision_ts, dec.short_exit_price),
                      decision_ts, dec.short_exit_price)
                pos_short = None
            if pos_long is not None and dec is not None and dec.long_exit:
                reason = "TREND" if dec.long_trend_exit else "SL"
                close(pos_long, reason, buyback_prem(pos_long, decision_ts, dec.long_exit_price),
                      decision_ts, dec.long_exit_price)
                pos_long = None

            # Premium-decay TP (mark check on the closed bar, using the SAME
            # premium history already fetched for that leg). When EITHER leg
            # hits, BOTH legs are blocked from new entries until
            # --tp-block-hour/minute that day (see run()'s tp_mult comment).
            if pos_short is not None and pos_short["tp_price"] is not None:
                p = op.premium_at(pos_short["candles"], decision_ts, step)
                if p is not None and p <= pos_short["tp_price"]:
                    close(pos_short, "TP", p, decision_ts, c.close)
                    pos_short = None
                    strategy.force_flat_short()
                    block_until = _tp_block_until(decision_ts, args.tp_block_hour, args.tp_block_minute)
            if pos_long is not None and pos_long["tp_price"] is not None:
                p = op.premium_at(pos_long["candles"], decision_ts, step)
                if p is not None and p <= pos_long["tp_price"]:
                    close(pos_long, "TP", p, decision_ts, c.close)
                    pos_long = None
                    strategy.force_flat_long()
                    block_until = _tp_block_until(decision_ts, args.tp_block_hour, args.tp_block_minute)

            # Daily square-off: closes any still-open leg(s) so they never
            # ride on stale premium data past resolve_by_premium's ~3-day
            # fetch window (see module docstring). Without --rollover this
            # always ends the leg (force_flat). With --rollover, a leg that
            # is STILL directionally committed (strategy.in_short/in_long --
            # i.e. its SL was NOT crossed this same bar, only squared off by
            # the clock) immediately sells a fresh option for the same side,
            # keeping the SAME frozen SL (never cleared, since force_flat is
            # skipped for that side).
            eod_blackout = args.weekend_blackout and _in_weekend_blackout(c.start_time)
            if square_off:
                if pos_short is not None:
                    close(pos_short, "EOD", buyback_prem(pos_short, c.start_time, c.close), c.start_time, c.close)
                    pos_short = None
                    if args.rollover and strategy.in_short and not eod_blackout:
                        pos_short = open_leg(client, c.start_time, c.close, is_short=True, tag="ROLL")
                        if pos_short is None:
                            strategy.force_flat_short()
                    else:
                        strategy.force_flat_short()
                if pos_long is not None:
                    close(pos_long, "EOD", buyback_prem(pos_long, c.start_time, c.close), c.start_time, c.close)
                    pos_long = None
                    if args.rollover and strategy.in_long and not eod_blackout:
                        pos_long = open_leg(client, c.start_time, c.close, is_short=False, tag="ROLL")
                        if pos_long is None:
                            strategy.force_flat_long()
                    else:
                        strategy.force_flat_long()

            tp_blocked = decision_ts < block_until
            if dec is not None and dec.sell_signal and pos_short is None:
                if c.start_time < sim_start or blackout or tp_blocked:
                    strategy.force_flat_short()
                else:
                    pos_short = open_leg(client, decision_ts, c.close, is_short=True)
                    if pos_short is None:
                        strategy.force_flat_short()
            if dec is not None and dec.buy_signal and pos_long is None:
                if c.start_time < sim_start or blackout or tp_blocked:
                    strategy.force_flat_long()
                else:
                    pos_long = open_leg(client, decision_ts, c.close, is_short=False)
                    if pos_long is None:
                        strategy.force_flat_long()

        last = candles[-1]
        last_ts = last.start_time + bar_seconds   # "as of" the last candle's CLOSE, the most
                                                    # recent genuinely-known moment.
        if pos_short is not None:
            close(pos_short, "OPEN_AT_END", buyback_prem(pos_short, last_ts, last.close),
                  last_ts, last.close)
        if pos_long is not None:
            close(pos_long, "OPEN_AT_END", buyback_prem(pos_long, last_ts, last.close),
                  last_ts, last.close)

    trades.sort(key=lambda t: t["entry_time"])
    return trades


def report(trades: list[dict], args) -> None:
    side_txt = " [CE ONLY]" if args.ce_only else " [PE ONLY]" if args.pe_only else ""
    ha_txt = ", Heikin Ashi" if args.heikin_ashi else ", real candles"
    ema_txt = f", EMA{args.ema200_len} filter" if args.ema200_filter else ""
    trend_txt = f", trend EMA{args.trend_fast_len}/{args.trend_slow_len} filter" if args.trend_filter else ""
    trend_txt += "+flip-exit" if (args.trend_filter and args.trend_flip_exit) else ""
    roll_txt = ", rollover ON" if args.rollover else ""
    wb_txt = ", weekend blackout (Sat 17:30-Sun 23:55)" if args.weekend_blackout else ""
    tp_txt = (f", TP{args.tp_pct:.0f}%->block till {args.tp_block_hour}:{args.tp_block_minute:02d}"
              if args.tp_pct > 0 else "")
    print(f"\n{'=' * 118}")
    print(f"Supertrend({args.atr_period},{args.factor:.0f}) Fixed-SL [OPTION SELL]{side_txt} -- {args.days}d, "
          f"premium ~{args.target_premium:.0f}, {args.lots} lots{ha_txt}{ema_txt}{trend_txt}{roll_txt}{wb_txt}"
          f"{tp_txt}, floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 118}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'sig':<5}{'tag':<6}{'contract':<22}{'btc in':>9}{'btc out':>9}"
          f"{'opt in':>8}{'opt out':>8} {'reason':<12}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{t['signal']:<5}{t['tag']:<6}{t['contract']:<22}"
              f"{t['btc_entry']:>9.1f}{t['btc_exit']:>9.1f}{t['opt_in']:>8.1f}{t['opt_out']:>8.1f} "
              f"{t['reason']:<12}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 118}")
    print(f"Legs: {len(closed)} closed" + (f" (+{len(trades) - len(closed)} still open at data end)"
                                            if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "TP", "TREND", "EOD", "OPEN_AT_END"):
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
            "signal": t["signal"], "tag": t["tag"], "contract": t["contract"],
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
        ("heikin_ashi", args.heikin_ashi), ("ema200_filter", args.ema200_filter),
        ("trend_filter", args.trend_filter), ("trend_fast_len", args.trend_fast_len),
        ("trend_slow_len", args.trend_slow_len), ("trend_flip_exit", args.trend_flip_exit),
        ("rollover", args.rollover), ("weekend_blackout", args.weekend_blackout),
        ("tp_pct", args.tp_pct), ("tp_block_hour", args.tp_block_hour),
        ("tp_block_minute", args.tp_block_minute),
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
    p.add_argument("--heikin-ashi", action="store_true",
                   help="run Supertrend on synthetic Heikin Ashi OHLC instead of the real candle's own "
                        "OHLC. Entry price and the frozen SL's crossing check still use the real candle "
                        "either way -- only the indicator's own feed changes.")
    p.add_argument("--ema200-filter", action="store_true",
                   help="additional gate at the flip: bearish (CE) only if close < EMA200, bullish (PE) "
                        "only if close > EMA200. Disagreeing consumes the flip untraded.")
    p.add_argument("--ema200-len", type=int, default=200)
    p.add_argument("--trend-filter", action="store_true",
                   help="separate gate at the flip, combinable with --ema200-filter (both must agree): "
                        "EMA(trend-fast-len) > EMA(trend-slow-len) is a POSITIVE trend -- only a bullish "
                        "flip (PE) is allowed; NEGATIVE trend -- only a bearish flip (CE) is allowed. "
                        "Disagreeing consumes the flip untraded.")
    p.add_argument("--trend-fast-len", type=int, default=150)
    p.add_argument("--trend-slow-len", type=int, default=600)
    p.add_argument("--trend-flip-exit", action="store_true",
                   help="SEPARATE toggle from --trend-filter (only takes effect when it's also on): "
                        "force-close an open leg the instant the trend EMA relationship itself "
                        "crosses, tagged TREND, instead of letting it ride to its frozen SL/EOD. "
                        "Off (default) = --trend-filter alone only gates NEW entries.")
    p.add_argument("--weekend-blackout", action="store_true",
                   help="no NEW entry from Saturday 17:30 IST through Sunday 23:55 IST. The daily "
                        "square-off already flattens any open leg by 17:25 every day, so this only "
                        "ever suppresses a fresh flip signal (and a --rollover reopen). Off (default) "
                        "= unchanged behavior.")
    p.add_argument("--rollover", action="store_true",
                   help="if a leg is closed ONLY by the daily square-off (its frozen SL was never "
                        "crossed), immediately sell a fresh option for the SAME side, keeping the SAME "
                        "frozen SL. A leg that closed via a real SL cross this same bar never rolls. "
                        "Off (default) = every square-off just ends the leg.")
    p.add_argument("--tp-pct", type=float, default=70.0,
                   help="premium-decay TP: exit a leg once its option premium decays by this %% "
                        "of the entry fill (sold at 100, 70%% decay -> buy back at 30). Checked "
                        "on every candle close using the same premium history already fetched "
                        "for that leg. 0 = disabled (ride to SL/EOD only, the original behavior). "
                        "When EITHER leg's TP hits, BOTH legs are blocked from new entries until "
                        "--tp-block-hour/--tp-block-minute that same day -- a blocked flip is "
                        "consumed untraded via force_flat, same treatment as --weekend-blackout.")
    p.add_argument("--tp-block-hour", type=int, default=17)
    p.add_argument("--tp-block-minute", type=int, default=30)
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
