"""Backtest: prev-day-zone reversal-breakout, executed by BUYING an option.

Setup (zone + red/green breakout + BTC stop) is on the BTC underlying; the trade
is a bought option (~target premium). The strategy and the open option position
run together in ONE candle loop so the +50% option take-profit can drive the
Supertrend re-entry gate (see src/deltabot/strategy/revbreak.py).

Exits per trade, first to fire: option premium +50% (TP), BTC stop (SL), 17:25
IST EOD square-off. Brokerage charged on both fills.

Run:  python scripts/backtest_revbreak.py [--days 7] [--target-premium 300]
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from deltabot.backtest import option_pricing as op
from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.config import load_settings
from deltabot.enums import OptionType, SignalDir
from deltabot.logging_setup import setup_logging
from deltabot.strategy.revbreak import RevBreakStrategy

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M IST")


def _first_tp_time(candles: dict, after: int, upto: int, tp_price: float, step: int,
                   sell: bool) -> int | None:
    """Earliest option-candle start_time in ``(after, upto]`` that reaches the TP:
    a SELL profits when premium falls (low <= tp), a BUY when it rises (high >= tp)."""
    if sell:
        hit = [t for t, c in candles.items() if after < t <= upto and c.low <= tp_price]
    else:
        hit = [t for t, c in candles.items() if after < t <= upto and c.high >= tp_price]
    return min(hit) if hit else None


def run(candles, settings, args):
    strategy = RevBreakStrategy(
        atr_period=settings.atr_period, st_multiplier=settings.st_multiplier,
        gate=args.gate, st_entry_filter=args.st_filter,
        reentry_block=not args.no_reentry_block,
        day_tz=settings.day_tz,
        day_start_hour=args.day_start_hour if args.day_start_hour is not None else settings.day_start_hour,
        day_start_minute=args.day_start_minute if args.day_start_minute is not None else settings.day_start_minute,
        square_off_hour=settings.square_off_hour, square_off_minute=settings.square_off_minute,
    )
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots if args.lots is not None else settings.option_contracts
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    sell = args.side == "sell"
    # SELL: take-profit when premium decays to (1 - tp%) of entry; BUY: rises to (1 + tp%).
    tp_mult = (1.0 - args.take_profit / 100.0) if sell else (1.0 + args.take_profit / 100.0)
    win_start = int(time.time()) - args.days * 86400
    cache: dict = {}
    trips: list[dict] = []
    pos: dict | None = None

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float):
        nonlocal pos
        assert pos is not None
        # SELL: profit when premium falls (entry - exit); BUY: when it rises (exit - entry).
        gross = ((pos["entry_prem"] - exit_prem) if sell else (exit_prem - pos["entry_prem"])) * lots * op.LOT_BTC
        fee = (op.side_fee(pos["entry_btc"], pos["entry_prem"], lots)
               + op.side_fee(exit_btc, exit_prem, lots))
        action = ("SELL " if sell else "BUY ") + ("CALL" if pos["sym"].startswith("C-") else "PUT")
        # Max adverse excursion: worst unrealized premium move during the hold.
        # SELL is hurt when premium RISES (use highs); BUY when it FALLS (use lows).
        adverse = [(c.high if sell else c.low) for t, c in pos["candles"].items()
                   if pos["entry_time"] <= t <= exit_time]
        worst_prem = (max(adverse) if sell else min(adverse)) if adverse else exit_prem
        mae = ((pos["entry_prem"] - worst_prem) if sell else (worst_prem - pos["entry_prem"])) * lots * op.LOT_BTC
        sl_distance = abs(pos["sl_level"] - pos["entry_btc"]) if pos.get("sl_level") else None
        trips.append({
            "action": action,
            "signal": "BUY" if pos["dir"] == SignalDir.LONG.value else "SELL",
            "contract": pos["sym"], "entry_time_ist": _ist(pos["entry_time"]),
            "exit_time_ist": _ist(exit_time), "exit_reason": reason,
            "btc_entry": round(pos["entry_btc"], 1), "btc_sl": round(pos.get("sl_level") or 0, 1),
            "sl_distance": round(sl_distance, 1) if sl_distance else None,
            "btc_exit": round(exit_btc, 1),
            "opt_in": round(pos["entry_prem"], 1), "opt_out": round(exit_prem, 1),
            "worst_prem": round(worst_prem, 1), "mae_usd": round(mae, 2),
            "lots": lots, "gross_usd": round(gross, 2),
            "fee_usd": round(fee, 2), "net_usd": round(gross - fee, 2),
        })
        pos = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            dec = strategy.update(c)

            # 1. BTC exit (SL/EOD) — these win over a same-bar option TP (conservative).
            if pos is not None and dec is not None and dec.has_exit:
                eprice = dec.long_exit_price if dec.long_exit else dec.short_exit_price
                exit_prem = op.premium_at(pos["candles"], c.start_time, step)
                if exit_prem is not None:
                    close(dec.exit_reason, exit_prem, c.start_time, eprice)
                else:
                    pos = None  # cannot price exit; drop (strategy already flat)

            # 2. Option +50% take-profit (only if no BTC exit this bar).
            if pos is not None:
                t_tp = _first_tp_time(pos["candles"], pos["last_check"], c.start_time,
                                      pos["tp_price"], step, sell)
                if t_tp is not None:
                    strategy.notify_exit(pos["dir"], "TP")
                    close("TP", pos["tp_price"], t_tp, pos["entry_btc"])
                else:
                    pos["last_check"] = c.start_time

            # 3. New entry — trade the ~target-premium option.
            #   BUY side : buy CALL on a buy signal / PUT on a sell signal.
            #   SELL side: sell PUT on a buy signal / CALL on a sell signal (same bias, sold).
            if pos is None and dec is not None and dec.has_entry:
                # Real-time SL filter: skip trades with wide stops.
                if args.max_sl_distance and dec.sl_level:
                    sl_dist = abs(dec.sl_level - c.close)
                    if sl_dist > args.max_sl_distance:
                        continue  # Skip this entry, continue to next candle
                is_buy = dec.buy_signal
                if sell:
                    otype = OptionType.PUT if is_buy else OptionType.CALL
                else:
                    otype = OptionType.CALL if is_buy else OptionType.PUT
                expiry = op.select_expiry_date(c.start_time, cutoff)
                resolved = op.resolve_by_premium(
                    client, underlying, otype, c.close, expiry, interval,
                    args.target_premium, c.start_time, c.start_time,
                    c.start_time - 86400, c.start_time + 2 * 86400,
                    args.opt_resolution, step, cache,
                )
                if resolved is None:
                    strategy.notify_exit(SignalDir.LONG.value if is_buy else SignalDir.SHORT.value, "SL")
                    continue
                sym, _, ocandles = resolved
                entry_prem = op.premium_at(ocandles, c.start_time, step)
                pos = {
                    "dir": SignalDir.LONG.value if is_buy else SignalDir.SHORT.value,
                    "sym": sym, "candles": ocandles, "entry_time": c.start_time,
                    "entry_btc": c.close, "entry_prem": entry_prem,
                    "tp_price": entry_prem * tp_mult, "last_check": c.start_time,
                    "sl_level": dec.sl_level,  # BTC stop-loss level (pattern extreme)
                }

    return [t for t in trips if op_entry_ts(t) >= win_start]


def op_entry_ts(trip: dict) -> int:
    return int(datetime.strptime(trip["entry_time_ist"].replace(" IST", ""),
                                 "%Y-%m-%d %H:%M").replace(tzinfo=_IST).timestamp())


def main() -> None:
    ap = argparse.ArgumentParser(description="Zone reversal-breakout option-buy backtest")
    ap.add_argument("--days", type=float, default=7, help="look-back window in days (fractional ok, e.g. 0.333 = 8h)")
    ap.add_argument("--resolution", default="5m", help="BTC candle resolution (default 5m)")
    ap.add_argument("--warmup-days", type=int, default=2)
    ap.add_argument("--max-sl-distance", type=float, default=None, help="skip trades where BTC SL is > this many points from entry (e.g. 400)")
    ap.add_argument("--side", choices=["buy", "sell"], default="buy",
                    help="buy = long option (+TP%% target); sell = short option (-TP%% target)")
    ap.add_argument("--gate", choices=["zone", "open"], default="zone",
                    help="zone = prev-day O/C no-trade zone; open = vs today's 05:30 open")
    ap.add_argument("--st-filter", action="store_true",
                    help="require Supertrend aligned to enter (replaces the re-entry block)")
    ap.add_argument("--no-reentry-block", action="store_true",
                    help="never block same-dir re-entry after a TP (fully ignore Supertrend gating)")
    ap.add_argument("--day-start-hour", type=int, default=None,
                    help="custom-day boundary hour in day_tz (default from .env, =5). Use 17 for 5:30 PM.")
    ap.add_argument("--day-start-minute", type=int, default=None,
                    help="custom-day boundary minute in day_tz (default from .env, =30)")
    ap.add_argument("--target-premium", type=float, default=300.0)
    ap.add_argument("--take-profit", type=float, default=50.0,
                    help="%% premium target: BUY exits at +TP%%, SELL exits at -TP%% (default 50)")
    ap.add_argument("--opt-resolution", default="1m")
    ap.add_argument("--lots", type=int, default=None)
    ap.add_argument("--excel", default="revbreak_backtest_IST.xlsx")
    args = ap.parse_args()

    settings = load_settings()
    setup_logging("WARNING")
    now = int(time.time())
    win_start = now - args.days * 86400
    df = download(symbol=settings.symbol, start=win_start - args.warmup_days * 86400, end=now,
                  resolution=args.resolution, base_url=settings.rest_base_url)
    candles = df_to_candles(df)
    trips = run(candles, settings, args)

    lots = args.lots if args.lots is not None else settings.option_contracts
    tp_sign = "-" if args.side == "sell" else "+"
    print(f"\n{settings.symbol}  {args.resolution}  ZONE reversal-breakout -> {args.side.upper()} option "
          f"~{args.target_premium:.0f} prem, {tp_sign}{args.take_profit:.0f}% target, lots {lots}")
    print(f"Window {_ist(win_start)} -> {_ist(now)}")
    print("=" * 104)
    print(f"{'Entry(IST)':<15}{'Exit(IST)':<15}{'Action':<10}{'Contract':<22}{'Why':<5}"
          f"{'PremIn':>8}{'PremOut':>8}{'Net$':>9}")
    print("-" * 104)
    net = fees = gross = 0.0
    wins = 0
    for t in trips:
        print(f"{t['entry_time_ist'][5:14]:<15}{t['exit_time_ist'][5:14]:<15}{t['action']:<10}"
              f"{t['contract']:<22}{t['exit_reason']:<5}{t['opt_in']:>8.1f}{t['opt_out']:>8.1f}"
              f"{t['net_usd']:>9.2f}")
        net += t["net_usd"]; fees += t["fee_usd"]; gross += t["gross_usd"]
        wins += 1 if t["net_usd"] > 0 else 0
    print("=" * 104)
    n = len(trips)
    wr = (wins / n * 100.0) if n else 0.0
    reasons = {r: sum(1 for t in trips if t["exit_reason"] == r) for r in ("TP", "SL", "EOD")}
    print(f"Trades {n}  Wins/Losses {wins}/{n - wins}  Win rate {wr:.1f}%  "
          f"exits: TP={reasons['TP']} SL={reasons['SL']} EOD={reasons['EOD']}")
    print(f"Gross {gross:,.2f}  -  brokerage {fees:,.2f}  =  NET {net:,.2f} USD")

    summary = [
        {"metric": "window_IST", "value": f"{_ist(win_start)} -> {_ist(now)}"},
        {"metric": "target_premium", "value": args.target_premium},
        {"metric": "take_profit_pct", "value": args.take_profit},
        {"metric": "lots", "value": lots},
        {"metric": "trades", "value": n},
        {"metric": "win_rate_pct", "value": round(wr, 1)},
        {"metric": "exits_TP/SL/EOD", "value": f"{reasons['TP']}/{reasons['SL']}/{reasons['EOD']}"},
        {"metric": "gross_usd", "value": round(gross, 2)},
        {"metric": "brokerage_usd", "value": round(fees, 2)},
        {"metric": "net_usd", "value": round(net, 2)},
    ]
    with pd.ExcelWriter(Path(args.excel), engine="openpyxl") as xl:
        pd.DataFrame(summary).to_excel(xl, sheet_name="Summary", index=False)
        pd.DataFrame(trips).to_excel(xl, sheet_name="Trades", index=False)
    print(f"\nExcel written to {Path(args.excel).resolve()}")


if __name__ == "__main__":
    main()
