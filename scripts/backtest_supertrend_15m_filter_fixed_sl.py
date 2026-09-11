"""Backtest: Supertrend15mFilterFixedSlStrategy -- OPTION SELL execution.

Pure BTC-price signal from
src/deltabot/strategy/supertrend_15m_filter_fixed_sl.py (ported from
supertrend_15m_filter_fixed_sl.pine): a fresh 1-minute Supertrend flip,
confirmed by the 15-minute Supertrend already agreeing at that exact
moment, arms a position whose SL is FROZEN at the 1-minute Supertrend's own
value on that flip candle (a real BTC price level, not a trend-change
event) -- the instant price crosses it, the leg closes and the strategy
waits flat for the next fresh, qualifying flip. No reversal, no profit
target, no daily square-off -- none of those were described, so this
backtest doesn't invent any square-off timer either; a leg only closes on
its own SL, or (see EXPIRY HANDLING below) when its option contract runs
out of data.

EXECUTION (sell-only): entry_is_short=True (bearish/SELL signal) -> sell a
CALL; entry_is_short=False (bullish/BUY signal) -> sell a PUT. Same mapping
OptionsExecutor._option_type_for already uses for every sell-side bot in
this fleet.

EXPIRY CUTOFF -- MINUTE PRECISE (this script's own rule, not
option_expiry_cutoff_hour): "if at 17:26 you should take trade in next day
option". Every other backtest in this repo rolls to the next day's expiry
at an HOUR boundary (option_expiry_cutoff_hour, default 17 -- i.e. anywhere
in the 17:00 hour already rolls). This strategy's entries needed
MINUTE precision instead (17:26 specifically), since entries can fire at
any minute a fresh flip happens, so a plain hour-cutoff would be both too
early (rolling entries at 17:01) and structurally the wrong granularity for
what was asked. _select_expiry() below is a dedicated, minute-precise
version local to this script -- it does not touch or reuse
option_pricing.select_expiry_date().

EXPIRY HANDLING (this script's own addition, not explicitly specified,
needed because there's no square-off): since a leg only exits on its own
SL and can otherwise run indefinitely, and each entry resolves to a SPECIFIC
next-day-or-later contract, a still-open leg whose contract's own historical
candles run out (i.e. it has expired) is force-closed at that point,
settled at intrinsic value -- the same way a real cash-settled option
that's never bought back gets closed out at expiry. This is flagged
explicitly rather than silently modeled, since it wasn't asked for but is
necessary for the backtest to terminate positions at all.

Timing: strategy.update(c) evaluates the JUST-CLOSED candle c -- entries/
exits are only genuinely known once c closes, i.e. at c.start_time +
bar_seconds, matching every other backtest in this repo.

Run:  python scripts/backtest_supertrend_15m_filter_fixed_sl.py --days 7
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
from deltabot.strategy.supertrend_15m_filter_fixed_sl import Supertrend15mFilterFixedSlStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _select_expiry(ts: int, cutoff_hour: int, cutoff_minute: int) -> datetime:
    """Minute-precise expiry cutoff, local to this script -- see module
    docstring's EXPIRY CUTOFF note for why option_pricing.select_expiry_date
    (hour-only) isn't reused here."""
    ist = datetime.fromtimestamp(ts, tz=_IST)
    if ist.hour * 60 + ist.minute >= cutoff_hour * 60 + cutoff_minute:
        ist = ist + timedelta(days=1)
    return ist


def run(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = Supertrend15mFilterFixedSlStrategy(
        atr_period=args.atr_period, factor=args.factor,
        atr_period_15m=args.atr_period_15m, factor_15m=args.factor_15m,
        use_heikin_ashi=args.heikin_ashi,
    )
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    lot_size = args.lot_size if args.lot_size > 0 else op.LOT_SIZE.get(underlying, op.LOT_BTC)
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    bar_seconds = op.RES_SECONDS.get(args.resolution, 60)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor

    cache: dict = {}
    trades: list[dict] = []
    pos: dict | None = None   # {"is_short", "sym", "is_call", "candles", "entry_time", "entry_btc", "entry_prem"}

    # Compounding lot-sizing (on-request variant, added 2026-09-11): start at
    # --start-capital, size every NEW entry off the CURRENT capital
    # (lots = floor(capital / --capital-per-lot)), but only RECOMPUTE that
    # lot count once per calendar day, at the --eod-square-off boundary --
    # "at every day close check deposit and update lots". Intraday SL/target
    # closes still realize into `capital` immediately (so it's always
    # accurate), they just don't change the lot size used for a same-day
    # re-entry; that only happens at the next day-close checkpoint. Disabled
    # (args.start_capital <= 0, the default) -- every lot uses the plain
    # static --lots value, unchanged from before this feature existed.
    compounding = args.start_capital > 0
    capital = args.start_capital
    current_lots = max(0, int(capital // args.capital_per_lot)) if compounding else args.lots
    capital_trace: list[tuple[int, float, int]] = []   # (ts, capital, lots_for_next_day) -- one row per day-close

    def open_leg(client, ts: int, btc_px: float, is_short: bool) -> bool:
        nonlocal pos
        lots_this_entry = current_lots if compounding else args.lots
        if lots_this_entry <= 0:
            return False   # compounding: capital too small for even 1 lot right now
        # SELL: is_short (bearish/SELL signal) -> sell CALL; not is_short (bullish/BUY signal) -> sell PUT.
        otype = OptionType.CALL if is_short else OptionType.PUT
        expiry = _select_expiry(ts, args.expiry_cutoff_hour, args.expiry_cutoff_minute)
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
        # Real expiry moment: 17:30 IST on `expiry`'s own date (this repo's
        # standard daily-settlement time) -- tracked explicitly rather than
        # inferred from premium_at() returning None, which it deliberately
        # avoids doing (falls back to the last known candle instead), so it
        # never actually fires as an expiry signal. See EXPIRY HANDLING.
        expiry_dt = expiry.replace(hour=17, minute=30, second=0, microsecond=0)
        expiry_ts = int(expiry_dt.timestamp())
        pos = {"is_short": is_short, "sym": sym, "is_call": otype == OptionType.CALL,
               "candles": ocandles, "entry_time": ts, "entry_btc": btc_px, "entry_prem": entry_prem,
               "expiry_ts": expiry_ts, "lots_remaining": lots_this_entry, "partial_taken": False}
        return True

    def buyback_prem(ts: int, exit_btc: float) -> float | None:
        return op.premium_at(pos["candles"], ts, step)

    def _record_trade(reason: str, exit_prem: float, exit_time: int, exit_btc: float, lots: int) -> None:
        nonlocal capital
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        entry_fill = pos["entry_prem"] * (1 - es)
        exit_fill = exit_prem * (1 + xs)
        gross = (entry_fill - exit_fill) * lots * lot_size
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots, lot_size)
               + op.side_fee(exit_btc, exit_fill, lots, lot_size))
        net = gross - fee
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time, "contract": pos["sym"],
            "action": f"SELL {'CE' if pos['is_call'] else 'PE'}",
            "btc_entry": pos["entry_btc"], "btc_exit": exit_btc,
            "opt_in": entry_fill, "opt_out": exit_fill, "reason": reason, "lots": lots,
            "gross": gross, "fee": fee, "net": net,
        })
        if compounding:
            # Realized immediately (SL/TARGET closes too, not just EOD) so
            # `capital` is always accurate -- only the LOT SIZE used for new
            # entries waits for the next day-close checkpoint, see the
            # `compounding` block comment above.
            capital += net

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float) -> None:
        """FULL close -- whatever lots remain (all of them, unless a PARTIAL
        close already booked some earlier in this same leg's life)."""
        nonlocal pos
        assert pos is not None
        _record_trade(reason, exit_prem, exit_time, exit_btc, pos["lots_remaining"])
        pos = None

    def close_partial(reason: str, exit_prem: float, exit_time: int, exit_btc: float, lots: int) -> None:
        """PARTIAL close -- books only `lots` of the leg's remaining size and
        keeps the REST open (see --partial-lots/--partial-target-pct). Does
        NOT clear `pos` and deliberately does NOT touch strategy state (no
        force_flat()/notify_target_hit()) -- the leg is still genuinely open,
        the strategy's own SL must keep watching the remainder."""
        assert pos is not None
        _record_trade(reason, exit_prem, exit_time, exit_btc, lots)
        pos["lots_remaining"] -= lots
        pos["partial_taken"] = True

    # SELL: target_pct is a REDUCTION -- "target 70% reduce price of option
    # means if sold at 100 then target is 30", i.e. target = entry * (1 -
    # target_pct/100). This is the OPPOSITE convention from
    # backtest_supertrend_sar.py's --tp-pct (a DECAY-TO target, entry *
    # tp_pct/100) -- do not port one script's target math to the other
    # without converting it.
    target_frac = (1 - args.target_pct / 100.0) if args.target_pct > 0 else None
    last_squareoff_date = None   # only used when args.eod_square_off -- fires once per calendar day

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            dec = strategy.update(c)
            decision_ts = c.start_time + bar_seconds

            # On-request comparison variant: hold to end of day instead of
            # exiting on the premium target (see --eod-square-off help).
            # Fires once per calendar day at the configured IST time,
            # force-closing whatever happens to be open at that moment --
            # NOT part of the validated base strategy.
            if args.eod_square_off:
                ist_dt = datetime.fromtimestamp(decision_ts, tz=_IST)
                if ((ist_dt.hour, ist_dt.minute) >= (args.square_off_hour, args.square_off_minute)
                        and last_squareoff_date != ist_dt.date()):
                    last_squareoff_date = ist_dt.date()
                    if pos is not None:
                        exit_prem = buyback_prem(decision_ts, c.close)
                        close("EOD", exit_prem if exit_prem is not None else pos["entry_prem"],
                              decision_ts, c.close)
                        strategy.force_flat()
                    if compounding:
                        # "at every day close check deposit and update lots" --
                        # `capital` already reflects EVERY trade realized so
                        # far today (SL/TARGET closes too, not just the EOD
                        # close just above), so this is the true day-end
                        # deposit. Recomputed once per day; stays fixed for
                        # every entry until the NEXT day-close checkpoint.
                        current_lots = max(0, int(capital // args.capital_per_lot))
                        capital_trace.append((decision_ts, capital, current_lots))

            # Expiry handling: a still-open leg whose contract has passed
            # its own real expiry moment (tracked at entry, see open_leg())
            # is force-closed at intrinsic value -- see module docstring's
            # EXPIRY HANDLING note. Checked every bar regardless of `dec`,
            # since expiry isn't a strategy signal.
            if pos is not None and decision_ts >= pos["expiry_ts"]:
                close("EXPIRED", op.intrinsic_value(pos["sym"], c.close), decision_ts, c.close)
                strategy.force_flat()

            if dec is not None and dec.has_exit and pos is not None:
                exit_prem = buyback_prem(decision_ts, dec.exit_price)
                close("SL", exit_prem if exit_prem is not None else dec.exit_price, decision_ts, dec.exit_price)

            # Partial profit-booking (on-request comparison variant, added
            # 2026-09-11): book --partial-lots of the leg at
            # --partial-target-pct (REDUCTION convention, same as
            # --target-pct), leaving the REST open for whatever the normal
            # exits (SL / --target-pct / --eod-square-off) do with it. Fires
            # at most once per leg (pos["partial_taken"]). Deliberately does
            # NOT call force_flat()/notify_target_hit() -- the leg is still
            # genuinely open (lots_remaining > 0), so the strategy's own SL
            # must keep watching it exactly as if nothing happened.
            if (pos is not None and args.partial_lots > 0 and not pos["partial_taken"]
                    and pos["lots_remaining"] > 0):
                ptgt_frac = 1 - args.partial_target_pct / 100.0
                cur_prem = buyback_prem(decision_ts, c.close)
                if cur_prem is not None and cur_prem <= pos["entry_prem"] * ptgt_frac:
                    lots_to_book = min(args.partial_lots, pos["lots_remaining"])
                    close_partial("PARTIAL", cur_prem, decision_ts, c.close, lots_to_book)

            # Profit target: purely an OPTION-PREMIUM mechanic, invisible to
            # the strategy's own BTC-price SL -- "once target is done then
            # no new trade until 15 min supertrend changes" needs the
            # strategy to know a target fired, so notify_target_hit() is
            # called immediately after closing. No roll/re-sell here (that
            # would contradict "no new trade" -- this just closes flat).
            if pos is not None and target_frac is not None:
                cur_prem = buyback_prem(decision_ts, c.close)
                if cur_prem is not None and cur_prem <= pos["entry_prem"] * target_frac:
                    close("TARGET", cur_prem, decision_ts, c.close)
                    # force_flat() clears the strategy's OWN _is_short/
                    # _active_sl (a target close is invisible to the
                    # strategy's own update() -- unlike an SL-hit, which
                    # clears this internally -- so without this call the
                    # strategy stays permanently "in position" and NEVER
                    # enters again; this was a real, confirmed bug: a
                    # 90-day backtest showed 1979 qualifying entries after
                    # one target hit, 0 of which fired). Then
                    # notify_target_hit() layers the 15m-flip block on top.
                    strategy.force_flat()
                    strategy.notify_target_hit()

            if dec is not None and dec.has_entry and pos is None:
                if c.start_time < sim_start:
                    strategy.force_flat()   # warmup-window entry: don't take it
                elif not open_leg(client, decision_ts, c.close, dec.entry_is_short):
                    strategy.force_flat()   # couldn't price the contract; stay flat, retry next bar

        if pos is not None:
            last = candles[-1]
            last_ts = last.start_time + bar_seconds
            exit_prem = buyback_prem(last_ts, last.close)
            close("OPEN_AT_END", exit_prem if exit_prem is not None else pos["entry_prem"], last_ts, last.close)

    return trades, capital_trace, capital


def report(trades: list[dict], args, capital_trace: list[tuple[int, float, int]] | None = None,
           final_capital: float | None = None) -> None:
    print(f"\n{'=' * 112}")
    tgt_frac = 100 - args.target_pct
    tgt_desc = f"target {args.target_pct:.0f}% reduction (->{tgt_frac:.0f}% of entry)" if args.target_pct > 0 else "no target"
    if args.eod_square_off:
        tgt_desc += f", hold to EOD {args.square_off_hour:02d}:{args.square_off_minute:02d} IST"
    if args.partial_lots > 0:
        tgt_desc += (f", book {args.partial_lots}/{args.lots} lots at "
                     f"{args.partial_target_pct:.0f}% reduction, rest as above")
    lots_desc = (f"start capital ${args.start_capital:.2f} @ ${args.capital_per_lot:.0f}/lot"
                 if args.start_capital > 0 else f"{args.lots} lots")
    print(f"Supertrend 15m-Filter Fixed-SL [OPTION SELL] -- {args.days}d, {args.resolution}, "
          f"{'HEIKIN ASHI' if args.heikin_ashi else 'real'} candles, "
          f"Supertrend({args.atr_period},{args.factor:.0f}) + 15m({args.atr_period_15m},{args.factor_15m:.0f}) filter, "
          f"expiry cutoff {args.expiry_cutoff_hour:02d}:{args.expiry_cutoff_minute:02d} IST, "
          f"premium ~{args.target_premium:.0f}, {tgt_desc}, {lots_desc}, floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 112}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'exit (IST)':<18}{'action':<8}{'contract':<22}{'reason':<10}{'lots':>5}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{_ist(t['exit_time']):<18}{t['action']:<8}{t['contract']:<22}"
              f"{t['reason']:<10}{t['lots']:>5}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 112}")
    print(f"Legs: {len(closed)} closed" + (" (+1 still open at data end)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "TARGET", "PARTIAL", "EOD", "EXPIRED", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<12} n={len(rs):<4} net ${sum(t['net'] for t in rs):>11.2f}")
    print(f"TOTAL NET: ${sum(t['net'] for t in trades):.2f} "
          f"(gross ${sum(t['gross'] for t in trades):.2f}, fees ${sum(t['fee'] for t in trades):.2f})")

    if args.start_capital > 0 and capital_trace:
        print(f"\n{'-' * 60}")
        print("Compounding: deposit + lot size at each day-close checkpoint")
        print(f"{'-' * 60}")
        print(f"{'date (IST)':<14}{'capital $':>14}{'lots (next day)':>18}")
        prev_lots = int(args.start_capital // args.capital_per_lot)
        print(f"{'(start)':<14}{args.start_capital:>14.2f}{prev_lots:>18}")
        for ts, cap, lots in capital_trace:
            print(f"{_ist(ts).split(' ')[0]:<14}{cap:>14.2f}{lots:>18}")
        print(f"{'-' * 60}")
        print(f"FINAL DEPOSIT: ${final_capital:.2f}  "
              f"({'+' if final_capital >= args.start_capital else ''}"
              f"{final_capital - args.start_capital:.2f} on ${args.start_capital:.2f} start, "
              f"{100.0 * (final_capital / args.start_capital - 1):.1f}%)")


def export(trades: list[dict], args, path: str) -> None:
    import pandas as pd

    rows, cum = [], 0.0
    for i, t in enumerate(trades, 1):
        cum += t["net"]
        rows.append({
            "#": i, "entry_IST": _ist(t["entry_time"]), "exit_IST": _ist(t["exit_time"]),
            "action": t["action"], "contract": t["contract"], "lots": t["lots"],
            "btc_entry": round(t["btc_entry"], 1), "btc_exit": round(t["btc_exit"], 1),
            "opt_sold": round(t["opt_in"], 1), "opt_bought_back": round(t["opt_out"], 1),
            "exit_reason": t["reason"], "gross_usd": round(t["gross"], 2),
            "fee_usd": round(t["fee"], 2), "net_usd": round(t["net"], 2),
            "cumulative_net_usd": round(cum, 2),
        })
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Trades", index=False)
    print(f"\nExcel written: {path}  ({len(df)} legs)")


def main() -> None:
    p = argparse.ArgumentParser(description="Supertrend 15m-Filter Fixed-SL backtest -- option SELL")
    p.add_argument("--days", type=float, default=7)
    p.add_argument("--warmup-days", type=int, default=3)
    p.add_argument("--resolution", default="1m")
    p.add_argument("--atr-period", type=int, default=10)
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--atr-period-15m", type=int, default=10)
    p.add_argument("--factor-15m", type=float, default=3.0)
    p.add_argument("--heikin-ashi", action="store_true",
                    help="Run both Supertrends (1m + 15m) on Heikin Ashi OHLC instead of real "
                         "candles -- see the strategy module docstring's OPT-IN HEIKIN ASHI MODE "
                         "note. Reported entry price stays REAL regardless.")
    p.add_argument("--expiry-cutoff-hour", type=int, default=17)
    p.add_argument("--expiry-cutoff-minute", type=int, default=26)
    p.add_argument("--target-premium", type=float, default=1400.0)
    p.add_argument("--eod-square-off", action="store_true",
                    help="Force-close whatever's open at --square-off-hour:--square-off-minute IST "
                         "each day, INSTEAD of exiting on the premium target -- pair with "
                         "--target-pct 0 to fully disable the target and hold to end of day. Not "
                         "part of the base strategy/backtest (neither was ever described or "
                         "validated) -- this is an on-request comparison variant only.")
    p.add_argument("--square-off-hour", type=int, default=17)
    p.add_argument("--square-off-minute", type=int, default=25)
    p.add_argument("--target-pct", type=float, default=70.0,
                    help="Profit target as a REDUCTION from entry (e.g. 70 -> target = 30%% of entry, "
                         "\"if sold at 100 then target is 30\"). 0 disables.")
    p.add_argument("--partial-lots", type=int, default=0,
                    help="On-request variant: book this many lots early at --partial-target-pct, "
                         "leaving (--lots minus this) open for the normal exits (SL / --target-pct / "
                         "--eod-square-off) to handle. 0 (default) disables -- every lot closes "
                         "together as one leg, exactly like the validated base strategy. Typical use: "
                         "--partial-lots 5 --partial-target-pct 50 --target-pct 0 --eod-square-off "
                         "(book half at 50%%, hold the rest to end of day).")
    p.add_argument("--partial-target-pct", type=float, default=50.0,
                    help="REDUCTION convention, same as --target-pct, but applies only to "
                         "--partial-lots.")
    p.add_argument("--start-capital", type=float, default=0.0,
                    help="On-request compounding variant: start with this much capital ($) and, "
                         "once per day at the --eod-square-off checkpoint, resize every NEW entry to "
                         "floor(capital / --capital-per-lot) lots -- \"at every $3 capital take 1 "
                         "lot\". Realized P&L (every SL/TARGET/EOD close) feeds capital immediately; "
                         "only the LOT SIZE waits for the next day-close to update. 0 (default) "
                         "disables -- --lots is used unchanged, exactly like before this existed. "
                         "Requires --eod-square-off (the day-close checkpoint IS the recompute point).")
    p.add_argument("--capital-per-lot", type=float, default=3.0,
                    help="Only used with --start-capital > 0. $ of capital per 1 lot of size.")
    p.add_argument("--lots", type=int, default=10)
    p.add_argument("--lot-size", type=float, default=0.0)
    p.add_argument("--opt-resolution", default="1m")
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

    trades, capital_trace, final_capital = run(candles, settings, args, sim_start)
    report(trades, args, capital_trace, final_capital)
    if args.out:
        export(trades, args, args.out)


if __name__ == "__main__":
    main()
