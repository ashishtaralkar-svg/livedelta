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

    def open_leg(client, ts: int, btc_px: float, is_short: bool) -> bool:
        nonlocal pos
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
               "expiry_ts": expiry_ts}
        return True

    def buyback_prem(ts: int, exit_btc: float) -> float | None:
        return op.premium_at(pos["candles"], ts, step)

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float) -> None:
        nonlocal pos
        assert pos is not None
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        entry_fill = pos["entry_prem"] * (1 - es)
        exit_fill = exit_prem * (1 + xs)
        gross = (entry_fill - exit_fill) * args.lots * lot_size
        fee = (op.side_fee(pos["entry_btc"], entry_fill, args.lots, lot_size)
               + op.side_fee(exit_btc, exit_fill, args.lots, lot_size))
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time, "contract": pos["sym"],
            "action": f"SELL {'CE' if pos['is_call'] else 'PE'}",
            "btc_entry": pos["entry_btc"], "btc_exit": exit_btc,
            "opt_in": entry_fill, "opt_out": exit_fill, "reason": reason,
            "gross": gross, "fee": fee, "net": gross - fee,
        })
        pos = None

    # SELL: target_pct is a REDUCTION -- "target 70% reduce price of option
    # means if sold at 100 then target is 30", i.e. target = entry * (1 -
    # target_pct/100). This is the OPPOSITE convention from
    # backtest_supertrend_sar.py's --tp-pct (a DECAY-TO target, entry *
    # tp_pct/100) -- do not port one script's target math to the other
    # without converting it.
    target_frac = (1 - args.target_pct / 100.0) if args.target_pct > 0 else None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            dec = strategy.update(c)
            decision_ts = c.start_time + bar_seconds

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

    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 112}")
    tgt_frac = 100 - args.target_pct
    tgt_desc = f"target {args.target_pct:.0f}% reduction (->{tgt_frac:.0f}% of entry)" if args.target_pct > 0 else "no target"
    print(f"Supertrend 15m-Filter Fixed-SL [OPTION SELL] -- {args.days}d, {args.resolution}, "
          f"Supertrend({args.atr_period},{args.factor:.0f}) + 15m({args.atr_period_15m},{args.factor_15m:.0f}) filter, "
          f"expiry cutoff {args.expiry_cutoff_hour:02d}:{args.expiry_cutoff_minute:02d} IST, "
          f"premium ~{args.target_premium:.0f}, {tgt_desc}, {args.lots} lots, floor {'OFF' if args.no_intrinsic_floor else 'ON'}")
    print(f"{'=' * 112}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'exit (IST)':<18}{'action':<8}{'contract':<22}{'reason':<10}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{_ist(t['exit_time']):<18}{t['action']:<8}{t['contract']:<22}"
              f"{t['reason']:<10}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 112}")
    print(f"Legs: {len(closed)} closed" + (" (+1 still open at data end)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "TARGET", "EXPIRED", "OPEN_AT_END"):
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
    p.add_argument("--expiry-cutoff-hour", type=int, default=17)
    p.add_argument("--expiry-cutoff-minute", type=int, default=26)
    p.add_argument("--target-premium", type=float, default=1400.0)
    p.add_argument("--target-pct", type=float, default=70.0,
                    help="Profit target as a REDUCTION from entry (e.g. 70 -> target = 30%% of entry, "
                         "\"if sold at 100 then target is 30\"). 0 disables.")
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

    trades = run(candles, settings, args, sim_start)
    report(trades, args)
    if args.out:
        export(trades, args, args.out)


if __name__ == "__main__":
    main()
