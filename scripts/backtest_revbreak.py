"""Backtest: prev-day-zone reversal-breakout, executed by BUYING an option.

Setup (zone + red/green breakout + BTC stop) is on the BTC underlying; the trade
is a bought option (~target premium). The strategy and the open option position
run together in ONE candle loop so the +50% option take-profit can drive the
Supertrend re-entry gate (see src/deltabot/strategy/revbreak.py).

Exits per trade, first to fire: option premium +50% (TP), BTC stop (SL), 17:25
IST EOD square-off. Brokerage charged on both fills.

SL-band filtering mirrors revbreak_trader.py's ``_sl_out_of_band`` exactly:
a percentage-of-price band (--min-sl-pct/--max-sl-pct) takes precedence when
either is set (>0); otherwise falls back to the legacy fixed --max-sl-distance.
Out-of-band trades are either skipped entirely, or -- with --paper-trade-wide-sl
-- tracked internally with NO P&L (matching live: a paper position has no real
exchange leg, so it also never gets the option-premium TP check, only BTC
SL/EOD). Paper trades are reported separately and excluded from Net$.

Run:  python scripts/backtest_revbreak.py [--days 7] [--target-premium 300]
      python scripts/backtest_revbreak.py --min-sl-pct 0.25 --max-sl-pct 0.75 \\
          --paper-trade-wide-sl   # reproduces the live pct-band + paper config
"""

from __future__ import annotations

import argparse
import bisect
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from deltabot.backtest import option_pricing as op
from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.config import load_settings
from deltabot.enums import OptionType, SignalDir
from deltabot.logging_setup import setup_logging
from deltabot.strategy.revbreak import RevBreakSellStrategy

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


def _sl_out_of_band(sl_level: float | None, btc_price: float, min_pct: float, max_pct: float,
                    max_dist: float | None) -> tuple[bool, float, str]:
    """Classify a BTC stop distance -- copied 1:1 from revbreak_trader.py's
    ``_sl_out_of_band`` so the backtest reproduces the live filter exactly.
    A percentage-of-price band takes precedence when either bound is set (>0);
    otherwise falls back to the legacy fixed ``max_dist`` in points."""
    if sl_level is None or btc_price <= 0:
        return False, 0.0, ""
    dist = abs(sl_level - btc_price)
    if max_pct > 0 or min_pct > 0:
        pct = dist / btc_price * 100.0
        if min_pct > 0 and pct < min_pct:
            return True, dist, f"SL {pct:.2f}% < {min_pct:.2f}% (too tight)"
        if max_pct > 0 and pct > max_pct:
            return True, dist, f"SL {pct:.2f}% > {max_pct:.2f}% (too wide)"
        return False, dist, ""
    if max_dist and dist > max_dist:
        return True, dist, f"SL {dist:.0f}pts > {max_dist:.0f} (too wide)"
    return False, dist, ""


def run(candles, settings, args):
    strategy = RevBreakSellStrategy(
        atr_period=args.atr_period if args.atr_period is not None else settings.atr_period,
        st_multiplier=args.st_multiplier if args.st_multiplier is not None else settings.st_multiplier,
        gate=args.gate, st_entry_filter=args.st_filter,
        reentry_block=args.reentry_block,
        day_tz=settings.day_tz,
        day_start_hour=args.day_start_hour if args.day_start_hour is not None else settings.day_start_hour,
        day_start_minute=args.day_start_minute if args.day_start_minute is not None else settings.day_start_minute,
        square_off_hour=settings.square_off_hour, square_off_minute=settings.square_off_minute,
        morning_start_hour=args.morning_start_hour, morning_start_minute=args.morning_start_minute,
        morning_square_off_hour=args.morning_square_off_hour,
        morning_square_off_minute=args.morning_square_off_minute,
    )
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots if args.lots is not None else settings.option_contracts
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    sell = args.side == "sell"
    # Slippage on market-order fills (as a fraction of premium). A SELL receives
    # LESS on the sell-to-open and pays MORE on the buy-to-close; BUY is the mirror.
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    # SELL: take-profit when premium decays to (1 - tp%) of entry; BUY: rises to (1 + tp%).
    tp_mult = (1.0 - args.take_profit / 100.0) if sell else (1.0 + args.take_profit / 100.0)
    win_start = int(time.time()) - args.days * 86400
    cache: dict = {}
    trips: list[dict] = []
    # Trading "day" = the 17:30 IST session boundary. Shifting back by the day-start
    # offset makes every candle in one session share a date key.
    _ds_h = args.day_start_hour if args.day_start_hour is not None else settings.day_start_hour
    _ds_m = args.day_start_minute if args.day_start_minute is not None else settings.day_start_minute
    def day_key(ts: int):
        return (datetime.fromtimestamp(ts, tz=_IST) - timedelta(hours=_ds_h, minutes=_ds_m)).date()
    tp_day = None  # session in which a REAL take-profit already fired
    _DOW = {"mon":0,"tue":1,"wed":2,"thu":3,"fri":4,"sat":5,"sun":6}
    _skip_days = {_DOW[t.strip().lower()[:3]] for t in (args.skip_weekdays or "").split(",")
                  if t.strip().lower()[:3] in _DOW}
    paper_trips: list[dict] = []
    pos: dict | None = None

    _spot_ts = sorted(cc.start_time for cc in candles)
    _spot = {cc.start_time: cc.close for cc in candles}
    def btc_at(t: int) -> float | None:
        i = bisect.bisect_right(_spot_ts, t) - 1
        return _spot[_spot_ts[i]] if i >= 0 else None

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float):
        nonlocal pos
        assert pos is not None
        if args.intrinsic_floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        # Apply slippage to the actual fills. SELL: worse = receive less at entry,
        # pay more at exit. BUY: pay more at entry, receive less at exit.
        if sell:
            entry_fill = pos["entry_prem"] * (1 - es)
            exit_fill = exit_prem * (1 + xs)
            gross = (entry_fill - exit_fill) * lots * op.LOT_BTC
        else:
            entry_fill = pos["entry_prem"] * (1 + es)
            exit_fill = exit_prem * (1 - xs)
            gross = (exit_fill - entry_fill) * lots * op.LOT_BTC
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots)
               + op.side_fee(exit_btc, exit_fill, lots))
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
            "opt_in": round(entry_fill, 1), "opt_out": round(exit_fill, 1),
            "worst_prem": round(worst_prem, 1), "mae_usd": round(mae, 2),
            "lots": lots, "gross_usd": round(gross, 2),
            "fee_usd": round(fee, 2), "net_usd": round(gross - fee, 2),
        })
        pos = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            dec = strategy.update(c)

            # 1. BTC exit (SL/EOD) — these win over a same-bar option TP (conservative).
            #    Paper positions have no exchange leg, only internal tracking, so
            #    they close here too but with no P&L (matches revbreak_trader.py).
            if pos is not None and dec is not None and dec.has_exit:
                eprice = dec.long_exit_price if dec.long_exit else dec.short_exit_price
                if pos.get("is_paper"):
                    paper_trips.append({
                        "signal": "BUY" if pos["dir"] == SignalDir.LONG.value else "SELL",
                        "entry_time_ist": _ist(pos["entry_time"]), "exit_time_ist": _ist(c.start_time),
                        "exit_reason": dec.exit_reason,
                        "btc_entry": round(pos["entry_btc"], 1), "btc_exit": round(eprice, 1),
                        "sl_distance": pos.get("sl_distance"), "band_reason": pos.get("paper_reason"),
                    })
                    pos = None
                else:
                    exit_prem = op.premium_at(pos["candles"], c.start_time, step)
                    if exit_prem is not None:
                        close(dec.exit_reason, exit_prem, c.start_time, eprice)
                    else:
                        pos = None  # cannot price exit; drop (strategy already flat)

            # 2. Option take-profit check (only if no BTC exit this bar). Paper
            #    positions get this check too (2026-07-10 fix, matches live):
            #    they track a real (never-ordered) contract so they can unblock
            #    via TP instead of only ever via SL/EOD. A paper trade that
            #    failed to resolve a contract at entry has no candles/tp_price
            #    and simply falls through, unblocking via SL/EOD as before.
            if pos is not None and pos.get("candles") and pos.get("tp_price") is not None:
                # A deep-ITM sold option can't decay to tp_price while its
                # intrinsic exceeds it -- with the floor on, only look when OTM enough.
                can_tp = (not args.intrinsic_floor or not sell
                          or op.intrinsic_value(pos["sym"], c.close) <= pos["tp_price"])
                t_tp = (_first_tp_time(pos["candles"], pos["last_check"], c.start_time,
                                       pos["tp_price"], step, sell) if can_tp else None)
                if t_tp is not None:
                    strategy.notify_exit(pos["dir"], "TP")
                    if pos.get("is_paper"):
                        paper_trips.append({
                            "signal": "BUY" if pos["dir"] == SignalDir.LONG.value else "SELL",
                            "entry_time_ist": _ist(pos["entry_time"]), "exit_time_ist": _ist(t_tp),
                            "exit_reason": "TP",
                            "btc_entry": round(pos["entry_btc"], 1), "btc_exit": round(c.close, 1),
                            "sl_distance": pos.get("sl_distance"), "band_reason": pos.get("paper_reason"),
                        })
                        pos = None
                    else:
                        close("TP", pos["tp_price"], t_tp, btc_at(t_tp) or pos["entry_btc"])
                        tp_day = day_key(c.start_time)   # target hit for this session
                else:
                    pos["last_check"] = c.start_time

            # 3. New entry — trade the ~target-premium option.
            #   BUY side : buy CALL on a buy signal / PUT on a sell signal.
            #   SELL side: sell PUT on a buy signal / CALL on a sell signal (same bias, sold).
            if pos is None and dec is not None and dec.has_entry:
                if _skip_days and datetime.fromtimestamp(c.start_time, tz=_IST).weekday() in _skip_days:
                    strategy.notify_exit(
                        SignalDir.LONG.value if dec.buy_signal else SignalDir.SHORT.value, "SL")
                    continue
                if args.stop_after_daily_tp and tp_day == day_key(c.start_time):
                    # Target already completed this session -> no further trades until 17:30.
                    strategy.notify_exit(
                        SignalDir.LONG.value if dec.buy_signal else SignalDir.SHORT.value, "SL")
                    continue
                # SL-band filter: percentage band (if set) takes precedence over
                # the legacy fixed --max-sl-distance -- exactly like live.
                out_of_band, sl_dist, band_reason = _sl_out_of_band(
                    dec.sl_level, c.close, args.min_sl_pct, args.max_sl_pct, args.max_sl_distance)
                if out_of_band and not args.paper_trade_wide_sl:
                    continue  # skip entirely, no real or paper trade taken
                is_buy = dec.buy_signal
                if out_of_band:
                    # paper_trade_wide_sl is on: track internally, no real order --
                    # but DO resolve a real contract (mirrors live's
                    # select_by_premium) so its premium can be polled for a
                    # would-be TP, matching the 2026-07-10 live fix.
                    p_otype = (OptionType.PUT if is_buy else OptionType.CALL) if sell \
                        else (OptionType.CALL if is_buy else OptionType.PUT)
                    p_expiry = op.select_expiry_date(c.start_time, cutoff)
                    p_resolved = op.resolve_by_premium(
                        client, underlying, p_otype, c.close, p_expiry, interval,
                        args.target_premium, c.start_time, c.start_time,
                        c.start_time - 86400, c.start_time + 2 * 86400,
                        args.opt_resolution, step, cache,
                    )
                    if p_resolved is not None:
                        p_sym, _, p_candles = p_resolved
                        p_entry_prem = op.premium_at(p_candles, c.start_time, step)
                    else:
                        p_sym = p_candles = p_entry_prem = None
                    pos = {
                        "dir": SignalDir.LONG.value if is_buy else SignalDir.SHORT.value,
                        "is_paper": True, "entry_time": c.start_time, "entry_btc": c.close,
                        "sl_distance": round(sl_dist, 1), "paper_reason": band_reason,
                        "sym": p_sym, "candles": p_candles,
                        "tp_price": (p_entry_prem * tp_mult) if p_entry_prem is not None else None,
                        "last_check": c.start_time,
                    }
                    continue
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
                if args.intrinsic_floor and entry_prem is not None:
                    entry_prem = max(entry_prem, op.intrinsic_value(sym, c.close))
                pos = {
                    "dir": SignalDir.LONG.value if is_buy else SignalDir.SHORT.value,
                    "sym": sym, "candles": ocandles, "entry_time": c.start_time,
                    "entry_btc": c.close, "entry_prem": entry_prem,
                    "tp_price": entry_prem * tp_mult, "last_check": c.start_time,
                    "sl_level": dec.sl_level,  # BTC stop-loss level (pattern extreme)
                }

    return ([t for t in trips if op_entry_ts(t) >= win_start],
            [t for t in paper_trips if op_entry_ts(t) >= win_start])


def op_entry_ts(trip: dict) -> int:
    return int(datetime.strptime(trip["entry_time_ist"].replace(" IST", ""),
                                 "%Y-%m-%d %H:%M").replace(tzinfo=_IST).timestamp())


def main() -> None:
    ap = argparse.ArgumentParser(description="Zone reversal-breakout option-buy backtest")
    ap.add_argument("--days", type=float, default=7, help="look-back window in days (fractional ok, e.g. 0.333 = 8h)")
    ap.add_argument("--resolution", default="5m", help="BTC candle resolution (default 5m)")
    ap.add_argument("--warmup-days", type=int, default=2)
    # ---------------------------------------------------------------- #
    # Every default below is baked in to match the DEPLOYED live RevBreak-
    # Sell config as of 2026-07-09 (see .env.revbreak), NOT the shared/generic
    # .env that load_settings() otherwise falls back to for anything left as
    # None. This is deliberate: on 2026-07-09 four separate flags were each
    # individually forgotten across a full day of backtest runs (SL-band mode,
    # slippage, day boundary, re-entry block) because they silently fell back
    # to a stale generic default instead of erroring -- producing a string of
    # inflated/wrong NET$ numbers. Run bare (just --days) to reproduce live;
    # override any flag explicitly for a deliberate comparison study.
    # ---------------------------------------------------------------- #
    ap.add_argument("--skip-weekdays", default="",
                    help="comma-separated IST days whose NEW entries are blocked (exits still run), "
                         "e.g. 'Sat,Sun'")
    ap.add_argument("--stop-after-daily-tp", action="store_true",
                    help="once a take-profit fires, take NO further trades for the rest of that "
                         "session (resumes at the next 17:30 day boundary)")
    ap.add_argument("--intrinsic-floor", action="store_true",
                    help="floor every option premium to intrinsic value (fixes illiquid-ITM candles "
                         "that print BELOW intrinsic and inflate the backtest -- 2026-07-12)")
    ap.add_argument("--max-sl-distance", type=float, default=400.0,
                    help="legacy fixed cutoff: skip trades where BTC SL is > this many points from entry "
                         "(default 400, matches live). Ignored whenever --min-sl-pct or --max-sl-pct is set (>0).")
    ap.add_argument("--min-sl-pct", type=float, default=0.0,
                    help="pct-of-price band floor: out-of-band trades have SL closer than this %% (0=off, matches live)")
    ap.add_argument("--max-sl-pct", type=float, default=0.0,
                    help="pct-of-price band ceiling: out-of-band trades have SL wider than this %% (0=off, matches "
                         "live as of 2026-07-09). Setting either bound >0 switches to percentage-band mode, "
                         "matching DELTA_REVBREAK_MIN_SL_PCT/MAX_SL_PCT (the PRIOR live config, until 2026-07-09).")
    ap.add_argument("--paper-trade-wide-sl", action="store_true",
                    help="track out-of-band-SL trades internally with NO P&L instead of skipping them "
                         "-- matches DELTA_REVBREAK_PAPER_TRADE_WIDE_SL (off by default, matches live)")
    ap.add_argument("--side", choices=["buy", "sell"], default="sell",
                    help="buy = long option (+TP%% target); sell = short option (-TP%% target) (default sell, matches live)")
    ap.add_argument("--gate", choices=["zone", "open", "dual_session"], default="open",
                    help="zone = prev-day O/C no-trade zone; open = vs today's session open (default, matches live); "
                         "dual_session = two rolling lines (17:30 + 05:30 opens), no-trade between them, "
                         "bull/bear only above/below BOTH -- experimental refinement, not yet live")
    ap.add_argument("--morning-start-hour", type=int, default=5,
                    help="dual_session only: the SECOND session boundary hour (default 5 = 5:30 AM)")
    ap.add_argument("--morning-start-minute", type=int, default=30, help="dual_session only (default 30)")
    ap.add_argument("--morning-square-off-hour", type=int, default=5,
                    help="dual_session only: square-off for the 05:30 session boundary (default 5 = 5:25 AM)")
    ap.add_argument("--morning-square-off-minute", type=int, default=25, help="dual_session only (default 25)")
    ap.add_argument("--st-filter", action="store_true",
                    help="require Supertrend aligned to enter, replacing the re-entry block (off by default, matches live)")
    ap.add_argument("--atr-period", type=int, default=None,
                    help="Supertrend ATR period (default: settings.atr_period, currently 10)")
    ap.add_argument("--st-multiplier", type=float, default=None,
                    help="Supertrend ATR multiplier (default: settings.st_multiplier -- NOTE: local .env has this "
                         "overridden to 3, but .env.revbreak does not override it, so live actually runs the "
                         "config.py class default of 2.0. Pass explicitly to be unambiguous.)")
    ap.add_argument("--reentry-block", action="store_true",
                    help="block same-dir re-entry after a TP until a Supertrend flip (OFF by default, matches live "
                         "DELTA_REVBREAK_REENTRY_BLOCK=false -- pass this flag to turn blocking ON for comparison)")
    ap.add_argument("--day-start-hour", type=int, default=17,
                    help="custom-day boundary hour in day_tz (default 17 = 5:30 PM, matches live; "
                         "do NOT rely on .env's DELTA_DAY_START_HOUR=5, which is stale for RevBreak)")
    ap.add_argument("--day-start-minute", type=int, default=30,
                    help="custom-day boundary minute in day_tz (default 30, matches live)")
    ap.add_argument("--target-premium", type=float, default=900.0, help="default 900, matches live DELTA_TARGET_PREMIUM")
    ap.add_argument("--take-profit", type=float, default=70.0,
                    help="%% premium target: BUY exits at +TP%%, SELL exits at -TP%% (default 70, matches live)")
    ap.add_argument("--entry-slippage-pct", type=float, default=1.0,
                    help="premium %% given up on the entry market fill (default 1%%, calibrated against real "
                         "Delta order-history fills -- see commit b24921f)")
    ap.add_argument("--exit-slippage-pct", type=float, default=5.0,
                    help="premium %% paid through the spread on the exit market fill (default 5%%, calibrated "
                         "against real Delta order-history fills -- see commit b24921f)")
    ap.add_argument("--opt-resolution", default="1m")
    ap.add_argument("--lots", type=int, default=25,
                    help="default 25, matches live DELTA_OPTION_CONTRACTS -- do NOT rely on .env's "
                         "DELTA_OPTION_CONTRACTS=50, which is stale for RevBreak")
    ap.add_argument("--excel", default="revbreak_backtest_IST.xlsx")
    args = ap.parse_args()

    settings = load_settings()
    setup_logging("WARNING")
    now = int(time.time())
    win_start = now - args.days * 86400
    df = download(symbol=settings.symbol, start=win_start - args.warmup_days * 86400, end=now,
                  resolution=args.resolution, base_url=settings.rest_base_url)
    candles = df_to_candles(df)
    trips, paper_trips = run(candles, settings, args)

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

    if args.paper_trade_wide_sl:
        pn = len(paper_trips)
        preasons = {r: sum(1 for t in paper_trips if t["exit_reason"] == r) for r in ("TP", "SL", "EOD")}
        print(f"\nPaper trades (out-of-band SL, NOT included above -- no exchange leg, no P&L): {pn}  "
              f"exits: TP={preasons['TP']} SL={preasons['SL']} EOD={preasons['EOD']}")

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
        {"metric": "min_sl_pct", "value": args.min_sl_pct},
        {"metric": "max_sl_pct", "value": args.max_sl_pct},
        {"metric": "max_sl_distance", "value": args.max_sl_distance},
        {"metric": "paper_trade_wide_sl", "value": args.paper_trade_wide_sl},
        {"metric": "paper_trades", "value": len(paper_trips)},
    ]
    with pd.ExcelWriter(Path(args.excel), engine="openpyxl") as xl:
        pd.DataFrame(summary).to_excel(xl, sheet_name="Summary", index=False)
        pd.DataFrame(trips).to_excel(xl, sheet_name="Trades", index=False)
        if paper_trips:
            pd.DataFrame(paper_trips).to_excel(xl, sheet_name="PaperTrades", index=False)
    print(f"\nExcel written to {Path(args.excel).resolve()}")


if __name__ == "__main__":
    main()
