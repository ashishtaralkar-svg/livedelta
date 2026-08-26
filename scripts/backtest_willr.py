"""Backtest: WilliamsRScanner -- option-BUY execution of
src/deltabot/strategy/willr_scanner.py.

UNLIKE EVERY OTHER BACKTEST IN THIS REPO, there is no single BTC-driven
signal -- each day this scans a UNIVERSE of candidate CE/PE strikes around
that day's ATM (--strike-radius strikes each side of ATM, in
option_strike_interval steps) and feeds EACH candidate's own full-day 5m
premium candles into ONE shared WilliamsRScanner (reset daily -- same-day
expiry means yesterday's contracts don't exist today anyway). The scanner
arms/enters per-contract exactly as described in strategy/willr_scanner.py's
own docstring.

INTERPRETIVE CHOICES made building this (flag if any of these don't match
what you meant):
  * "select option price between 50-100" is checked ONLY at the moment a
    contract's entry actually fires (its own close at that candle), not
    continuously beforehand -- same convention as target_premium selection
    everywhere else in this repo (checked at the trade, not the whole path).
    A signal whose close has drifted outside [--premium-min,--premium-max]
    by the time it fires is discarded untraded.
  * At most ONE open CE-type position and ONE open PE-type position at a
    time (matching the no-pyramiding convention used everywhere else in
    this repo) -- a second contract's entry signal on the SAME side while
    one is already open is simply discarded, even though the scanner's own
    per-contract state still marks that OTHER contract as "entered" (so it
    can never re-signal later that day either).
  * ATM reference for the day's strike universe is the BTC close on the
    candle nearest 00:00 IST that day -- strikes are picked once per day
    from that single anchor, not re-centered as BTC moves through the day.
  * Target = entry premium x --target-mult (default 3.0, i.e. the premium
    must TRIPLE) -- checked the same way as every other buy-side TP in this
    repo (closed-bar candle check against that contract's own premium
    history). No SL at all -- max loss per leg is capped at the premium
    paid. Square-off (--square-off-hour/minute) closes anything still open.

COST WARNING: this is far more expensive than every other backtest here --
(2 x strike_radius + 1) candidates PER SIDE PER DAY each need their own
full-day premium-candle fetch. At the default --strike-radius 8 that's 34
option-history fetches per day (17 strikes x 2 sides). Start with a SHORT
--days window (a few days) to sanity-check before running anything long.

Run:  python scripts/backtest_willr.py --days 2
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
from deltabot.strategy.willr_scanner import WilliamsRScanner

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M")


def _day_start(ts: int) -> int:
    d = datetime.fromtimestamp(ts, tz=_IST).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(d.timestamp())


def run(client: httpx.Client, btc_candles: list, settings, args, sim_start: int) -> list[dict]:
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots
    lot_size = args.lot_size if args.lot_size > 0 else op.LOT_SIZE.get(underlying, op.LOT_BTC)
    step = 300  # 5m
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor
    sq_mins = args.square_off_hour * 60 + args.square_off_minute

    trades: list[dict] = []
    now = int(time.time())
    last_day_start = _day_start(now)
    first_day_start = _day_start(sim_start)

    # BTC close by 5m bar start, for accurate intrinsic-value flooring AT THE
    # MOMENT of each fill -- the day-start ATM anchor below is only used to
    # pick the day's candidate STRIKE universe, never for pricing.
    btc_by_ts: dict[int, float] = {c.start_time: c.close for c in btc_candles}

    def btc_near(ts: int) -> float | None:
        for k in range(0, 12):
            px = btc_by_ts.get(ts - k * step)
            if px is not None:
                return px
        earlier = [(t, px) for t, px in btc_by_ts.items() if t <= ts]
        return max(earlier, key=lambda tp: tp[0])[1] if earlier else None

    day_start = first_day_start
    while day_start <= last_day_start:
        day_end = day_start + 86400

        # ATM anchor: the BTC candle nearest this day's 00:00 IST.
        day_btc = [c for c in btc_candles if day_start <= c.start_time < day_start + 3600]
        if not day_btc:
            day_start += 86400
            continue
        atm_btc = day_btc[0].close
        atm = int(round(atm_btc / interval) * interval)
        # Midday of this day -> resolves to this day's own same-day expiry.
        expiry = op.select_expiry_date(day_start + 43200, cutoff)
        ddmmyy = expiry.strftime("%d%m%y")

        candidates: list[tuple[str, OptionType]] = []
        for k in range(-args.strike_radius, args.strike_radius + 1):
            strike = atm + k * interval
            candidates.append((f"C-{underlying}-{strike}-{ddmmyy}", OptionType.CALL))
            candidates.append((f"P-{underlying}-{strike}-{ddmmyy}", OptionType.PUT))

        per_symbol: dict[str, dict[int, object]] = {}
        for sym, _otype in candidates:
            per_symbol[sym] = op.fetch_option_candles(client, sym, day_start, day_end, "5m")

        timestamps = sorted({t for cd in per_symbol.values() for t in cd})
        if not timestamps:
            day_start += 86400
            continue

        scanner = WilliamsRScanner(
            ema_len=args.ema_len, ma_len=args.ma_len, wr_period=args.wr_period,
            breakout_wait_bars=args.breakout_wait_bars,
            entry_start_hour=args.entry_start_hour, entry_end_hour=args.entry_end_hour,
        )
        open_ce: dict | None = None
        open_pe: dict | None = None

        def mark_at(pos: dict, ts: int, _syms=per_symbol) -> float:
            p = op.premium_at(_syms[pos["sym"]], ts, step)
            if p is not None:
                return p
            btc_now = btc_near(ts)
            return op.intrinsic_value(pos["sym"], btc_now) if btc_now is not None else pos["entry_prem"]

        def close(pos: dict, reason: str, exit_prem: float, exit_time: int, _day_atm=atm_btc) -> None:
            btc_now = btc_near(exit_time)
            if floor and btc_now is not None:
                exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], btc_now))
            entry_fill = pos["entry_prem"] * (1 + es)
            exit_fill = exit_prem * (1 - xs)
            gross = (exit_fill - entry_fill) * lots * lot_size
            fee_btc = btc_now if btc_now is not None else _day_atm
            fee = (op.side_fee(fee_btc, entry_fill, lots, lot_size)
                   + op.side_fee(fee_btc, exit_fill, lots, lot_size))
            trades.append({
                "entry_time": pos["entry_time"], "exit_time": exit_time,
                "signal": "BUY CE" if pos["otype"] == OptionType.CALL else "BUY PE",
                "contract": pos["sym"], "opt_in": entry_fill, "opt_out": exit_fill,
                "reason": reason, "gross": gross, "fee": fee, "net": gross - fee,
            })

        prev_mins: int | None = None
        for t in timestamps:
            local = datetime.fromtimestamp(t, tz=_IST)
            mins = local.hour * 60 + local.minute
            square_off = prev_mins is not None and mins >= sq_mins and prev_mins < sq_mins
            prev_mins = mins
            decision_ts = t + step

            # 1. Feed every candidate with a candle at this timestamp; buy on a fresh entry.
            for sym, otype in candidates:
                c = per_symbol[sym].get(t)
                if c is None:
                    continue
                fired = scanner.update_contract(sym, c)
                if not fired or t < sim_start:
                    continue
                if otype == OptionType.CALL and open_ce is not None:
                    continue
                if otype == OptionType.PUT and open_pe is not None:
                    continue
                entry_prem = c.close
                if floor:
                    btc_now = btc_near(decision_ts)
                    if btc_now is not None:
                        entry_prem = max(entry_prem, op.intrinsic_value(sym, btc_now))
                if not (args.premium_min <= entry_prem <= args.premium_max):
                    continue   # the TRUE (floor-corrected) premium is outside the target band
                pos = {"sym": sym, "otype": otype, "entry_time": decision_ts,
                       "entry_prem": entry_prem, "target": entry_prem * args.target_mult}
                if otype == OptionType.CALL:
                    open_ce = pos
                else:
                    open_pe = pos

            # 2. TP check for whatever's open.
            if open_ce is not None:
                m = mark_at(open_ce, decision_ts)
                if m >= open_ce["target"]:
                    close(open_ce, "TARGET", m, decision_ts)
                    open_ce = None
            if open_pe is not None:
                m = mark_at(open_pe, decision_ts)
                if m >= open_pe["target"]:
                    close(open_pe, "TARGET", m, decision_ts)
                    open_pe = None

            # 3. Daily square-off.
            if square_off:
                if open_ce is not None:
                    close(open_ce, "EOD", mark_at(open_ce, t), t)
                    open_ce = None
                if open_pe is not None:
                    close(open_pe, "EOD", mark_at(open_pe, t), t)
                    open_pe = None

        last_t = timestamps[-1]
        if open_ce is not None:
            close(open_ce, "OPEN_AT_END", mark_at(open_ce, last_t + step), last_t + step)
        if open_pe is not None:
            close(open_pe, "OPEN_AT_END", mark_at(open_pe, last_t + step), last_t + step)

        day_start += 86400

    trades.sort(key=lambda tr: tr["entry_time"])
    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 106}")
    print(f"WilliamsRScanner [OPTION BUY] -- {args.days}d, "
          f"EMA{args.ema_len}/SMA{args.ma_len}/WmR{args.wr_period}, "
          f"entries {args.entry_start_hour}-{args.entry_end_hour}h IST, "
          f"premium {args.premium_min:.0f}-{args.premium_max:.0f}, "
          f"target {args.target_mult:.1f}x, {args.lots} lots, strike radius {args.strike_radius}")
    print(f"{'=' * 106}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<18}{'signal':<8}{'contract':<22}{'opt in':>8}{'opt out':>8} "
          f"{'reason':<12}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<18}{t['signal']:<8}{t['contract']:<22}"
              f"{t['opt_in']:>8.1f}{t['opt_out']:>8.1f} {t['reason']:<12}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 106}")
    print(f"Legs: {len(closed)} closed" + (f" (+{len(trades) - len(closed)} still open at data end)"
                                            if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("TARGET", "EOD", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<12} n={len(rs):<4} net ${sum(t['net'] for t in rs):>11.2f}")
    print(f"TOTAL NET: ${sum(t['net'] for t in trades):.2f} "
          f"(gross ${sum(t['gross'] for t in trades):.2f}, fees ${sum(t['fee'] for t in trades):.2f})")


def main() -> None:
    p = argparse.ArgumentParser(description="WilliamsRScanner backtest -- option-buy execution")
    p.add_argument("--days", type=float, default=2, help="look-back window in days (see COST WARNING above)")
    p.add_argument("--ema-len", type=int, default=50)
    p.add_argument("--ma-len", type=int, default=50)
    p.add_argument("--wr-period", type=int, default=14)
    p.add_argument("--breakout-wait-bars", type=int, default=3)
    p.add_argument("--entry-start-hour", type=int, default=14)
    p.add_argument("--entry-end-hour", type=int, default=17)
    p.add_argument("--premium-min", type=float, default=50.0)
    p.add_argument("--premium-max", type=float, default=100.0)
    p.add_argument("--target-mult", type=float, default=3.0, help="exit target = entry premium x this")
    p.add_argument("--strike-radius", type=int, default=8,
                   help="candidate strikes each side of ATM (2*radius+1 per side; see COST WARNING)")
    p.add_argument("--square-off-hour", type=int, default=17)
    p.add_argument("--square-off-minute", type=int, default=25)
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lot-size", type=float, default=0.0, help="0 = auto-derive from the underlying symbol")
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true")
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    setup_logging("WARNING")
    settings = load_settings()

    now = int(time.time())
    sim_start = now - int(args.days * 86400)
    # BTC candles only needed for each day's ATM anchor -- a small, cheap download.
    dl_start = _day_start(sim_start) - 3600
    df = download(symbol=settings.symbol, start=dl_start, end=now, resolution="5m")
    btc_candles = df_to_candles(df)
    print(f"BTC candles: {len(btc_candles)} (5m, ATM anchors only) {_ist(btc_candles[0].start_time)} .. "
          f"{_ist(btc_candles[-1].start_time)}")

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        trades = run(client, btc_candles, settings, args, sim_start)
    report(trades, args)
    if args.json_out:
        import json
        with open(args.json_out, "w") as f:
            json.dump(trades, f)
        print(f"\nJSON written: {args.json_out}  ({len(trades)} legs)")


if __name__ == "__main__":
    main()
