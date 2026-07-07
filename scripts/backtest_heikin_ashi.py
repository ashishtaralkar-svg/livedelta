"""Backtest: Heikin Ashi Supertrend(10,3)-gated reversal -> SELL option (Python port of
btc_heikinashi_strategy.pine).

Entries require ALL of: (1) Supertrend(10,3) computed on the HA candles -- bullish
arms the BUY pattern, bearish arms the SELL pattern; (2) REAL BTC price currently on
the correct side of this session's opening price (captured at day_start, default 17:30
IST): above it for BUY, below it for SELL; (3) a chained EMA trend filter on HA close --
BUY needs ha_close > ema50 > ema200, SELL needs the mirror (ha_close < ema50 < ema200).

Execution mirrors RevBreak-Sell: BUY signal -> sell a PUT; SELL signal -> sell a
CALL. Exit = buy back at that moment's premium. Exits are exactly: the fixed pattern
SL, the trailing SL (arms/tightens at a CLOSED bar whose HA close is beyond the 50
EMA against the position's bias -- level = that bar's HA low/high; once armed it
fires ASAP the instant REAL price crosses it, same convention as the fixed SL), or
the 17:25 IST EOD square-off -- whichever fires first. NO profit target anywhere in
this script (unlike RevBreak-Sell's -70% premium-decay TP).

PER-TRADE CONTRACT RESOLUTION: every entry signal resolves a fresh
nearest-to-target-premium contract at that moment (matching live's
OptionsExecutor.open_option_by_premium). A daily-fixed-pair variant (resolve
once at day_start and hold until next day) was tried and tested worse
(strikes drift far from ATM as BTC moves intraday/across the month), so it
was reverted in favor of this per-trade model.

INTRACANDLE SIMULATION (the key lesson from RevBreak-Sell this session): that live
bot's own ASAP intracandle entry logic had to be disabled because its backtest
never modeled it -- signals were only ever evaluated at 5m candle close, so live
behavior diverged sharply (a stream of whipsaw losses the backtest was blind to).
By default (``--mode asap``) this script avoids repeating that mistake: it
downloads 1-MINUTE BTC candles, resamples them into 5-minute bars for the
strategy's closed-candle path (pattern detection, EOD), and additionally runs the
entry-trigger and fixed-SL checks against EVERY 1-minute bar as an intrabar proxy
for real-time ticks. (1-minute bars are the finest granularity Delta's historical
API offers; this is a proxy for real-time, not true tick data.)

``--mode closed`` switches to a pure CLOSED-CANDLE-ONLY mode (no intracandle
simulation at all): candles are downloaded directly at ``--candle-resolution``
(any resolution Delta supports -- 1m, 3m, 5m, 15m, 30m, 1h) and fed straight to
the strategy's own closed-candle path with NO resampling, so entries/SL/exits
are only ever evaluated at that resolution's own close -- e.g. ``--mode closed
--candle-resolution 1m`` runs the strategy's pattern/EMA/Supertrend logic
directly on 1-minute closed candles (not resampled to 5m), useful as a direct
comparison against the ASAP-simulated and 5m-closed numbers.

Run:  python scripts/backtest_heikin_ashi.py --days 30 --target-premium 900 --lots 100
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from deltabot.backtest import option_pricing as op
from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.config import load_settings
from deltabot.enums import OptionType, PositionState, SignalDir
from deltabot.models import Candle
from deltabot.strategy.heikin_ashi import HeikinAshiStrategy

_IST = ZoneInfo("Asia/Kolkata")
_5M = 300


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M IST")


def run(candles: list[Candle], settings, args) -> list[dict]:
    intracandle = args.mode == "asap"
    strategy = HeikinAshiStrategy(
        st_period=args.st_period, st_multiplier=args.st_multiplier,
        ema_length=args.ema_length, ema200_length=args.ema200_length,
        session_gate=not args.no_session_gate,
        trail_on_ema200=args.trail_ema200,
        day_tz=settings.day_tz,
        day_start_hour=args.day_start_hour, day_start_minute=args.day_start_minute,
        square_off_hour=args.square_off_hour, square_off_minute=args.square_off_minute,
    )
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots if args.lots is not None else settings.option_contracts
    opt_step = op.RES_SECONDS.get(args.opt_resolution, 300)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    win_start = int(time.time()) - int(args.days * 86400)
    cache: dict = {}
    trips: list[dict] = []
    pos: dict | None = None  # {sym, candles, entry_prem, entry_ts, entry_btc, dir, sl_level}

    # Running 5-minute resample bucket built from the 1-minute stream.
    bucket: dict | None = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:

        buying = args.execution == "buy"

        def open_position(signal_dir: int, ts: int, btc_price: float, sl_level: float) -> None:
            nonlocal pos
            is_buy = signal_dir == SignalDir.LONG.value
            if buying:
                # BUY signal -> buy CE (CALL); SELL signal -> buy PE (PUT).
                opt_type = OptionType.CALL if is_buy else OptionType.PUT
            else:
                # BUY signal -> sell PUT; SELL signal -> sell CALL.
                opt_type = OptionType.PUT if is_buy else OptionType.CALL
            expiry = op.select_expiry_date(ts, cutoff)
            resolved = op.resolve_by_premium(
                client, underlying, opt_type, btc_price, expiry, interval,
                args.target_premium, ts, ts, ts - 86400, ts + 2 * 86400,
                args.opt_resolution, opt_step, cache,
            )
            if resolved is None:
                print(f"  WARN {_ist(ts)}: entry signal SKIPPED -- no contract resolved "
                      f"near premium {args.target_premium:.0f}")
                strategy.notify_exit(signal_dir, "SL")  # couldn't price it; stay flat
                return
            sym, ocandles = resolved[0], resolved[2]
            entry_prem = op.premium_at(ocandles, ts, opt_step)
            if entry_prem is None:
                print(f"  WARN {_ist(ts)}: entry signal SKIPPED -- {sym} has no premium data at entry time")
                strategy.notify_exit(signal_dir, "SL")
                return
            # BUY option: pay MORE on a worse fill. SELL option: receive LESS on a worse fill.
            fill_prem = entry_prem * (1 + es) if buying else entry_prem * (1 - es)
            pos = {
                "sym": sym, "candles": ocandles, "dir": signal_dir,
                "entry_ts": ts, "entry_btc": btc_price,
                "entry_prem": fill_prem,
                "sl_level": sl_level,
            }

        def close_position(reason: str, ts: int, btc_exit: float) -> None:
            nonlocal pos
            assert pos is not None
            exit_prem_raw = op.premium_at(pos["candles"], ts, opt_step)
            if exit_prem_raw is None:
                print(f"  WARN {_ist(ts)}: no premium data at exit for {pos['sym']} "
                      f"(entered {_ist(pos['entry_ts'])}) -- trade DROPPED from results")
                pos = None
                return
            if buying:
                exit_prem = exit_prem_raw * (1 - xs)  # BUY: receives LESS on a worse fill (sold to close)
                gross = (exit_prem - pos["entry_prem"]) * lots * op.LOT_BTC
                action = "BUY CE" if pos["dir"] == SignalDir.LONG.value else "BUY PE"
            else:
                exit_prem = exit_prem_raw * (1 + xs)  # SELL: pays MORE on a worse fill (bought back)
                gross = (pos["entry_prem"] - exit_prem) * lots * op.LOT_BTC
                action = "SELL PUT" if pos["dir"] == SignalDir.LONG.value else "SELL CALL"
            fee = op.side_fee(pos["entry_btc"], pos["entry_prem"], lots) + op.side_fee(btc_exit, exit_prem, lots)
            trips.append({
                "action": action, "contract": pos["sym"],
                "entry_ist": _ist(pos["entry_ts"]), "exit_ist": _ist(ts),
                "exit_reason": reason,
                "btc_entry": round(pos["entry_btc"], 1), "btc_exit": round(btc_exit, 1),
                "sl_level": round(pos["sl_level"], 1) if pos["sl_level"] else None,
                "opt_in": round(pos["entry_prem"], 1), "opt_out": round(exit_prem, 1),
                "lots": lots, "gross_usd": round(gross, 2),
                "fee_usd": round(fee, 2), "net_usd": round(gross - fee, 2),
            })
            pos = None

        def handle_decision(dec) -> None:
            nonlocal pos
            if dec is None:
                return
            if dec.has_exit and pos is not None:
                eprice = dec.long_exit_price if dec.long_exit else dec.short_exit_price
                close_position(dec.exit_reason or "SL", dec.candle.start_time, eprice)
            if dec.has_entry and pos is None:
                signal_dir = SignalDir.LONG.value if dec.buy_signal else SignalDir.SHORT.value
                open_position(signal_dir, dec.candle.start_time, dec.candle.close, dec.sl_level)

        def report_open_position_if_any() -> None:
            # A position still open when the data simply runs out (only possible
            # for the most recent, still-live trade) is UNRESOLVED -- not a real
            # exit -- so it is reported separately as "OPEN" and excluded from
            # win/loss/P&L stats by the caller (matches backtest_vwap.py).
            nonlocal pos
            if pos is None:
                return
            last_ts = candles[-1].start_time
            exit_prem_raw = op.premium_at(pos["candles"], last_ts, opt_step)
            exit_prem = exit_prem_raw if exit_prem_raw is not None else pos["entry_prem"]
            if buying:
                gross = (exit_prem - pos["entry_prem"]) * lots * op.LOT_BTC
                action = "BUY CE" if pos["dir"] == SignalDir.LONG.value else "BUY PE"
            else:
                gross = (pos["entry_prem"] - exit_prem) * lots * op.LOT_BTC
                action = "SELL PUT" if pos["dir"] == SignalDir.LONG.value else "SELL CALL"
            fee = op.side_fee(pos["entry_btc"], pos["entry_prem"], lots) + op.side_fee(candles[-1].close, exit_prem, lots)
            trips.append({
                "action": action, "contract": pos["sym"],
                "entry_ist": _ist(pos["entry_ts"]), "exit_ist": _ist(last_ts),
                "exit_reason": "OPEN",
                "btc_entry": round(pos["entry_btc"], 1), "btc_exit": round(candles[-1].close, 1),
                "sl_level": round(pos["sl_level"], 1) if pos["sl_level"] else None,
                "opt_in": round(pos["entry_prem"], 1), "opt_out": round(exit_prem, 1),
                "lots": lots, "gross_usd": round(gross, 2),
                "fee_usd": round(fee, 2), "net_usd": round(gross - fee, 2),
            })
            pos = None

        if not intracandle:
            # --- Pure closed-candle-only mode: candles are already at the
            #     strategy's own bar size (e.g. 5m) -- feed each straight to
            #     update(), no resampling, no intracandle calls at all. ---
            for c in candles:
                handle_decision(strategy.update(c))
            report_open_position_if_any()
            return [t for t in trips if int(datetime.strptime(
                t["entry_ist"].replace(" IST", ""), "%Y-%m-%d %H:%M")
                .replace(tzinfo=_IST).timestamp()) >= win_start]

        for c in candles:
            bucket_start = (c.start_time // _5M) * _5M

            # --- Roll the 5-minute resample bucket; flush + run the closed-candle
            #     path exactly when a new bucket starts (the previous one is final). ---
            if bucket is None:
                bucket = {"start_time": bucket_start, "open": c.open, "high": c.high,
                         "low": c.low, "close": c.close, "volume": c.volume}
            elif bucket_start != bucket["start_time"]:
                closed5m = Candle(bucket["start_time"], bucket["open"], bucket["high"],
                                  bucket["low"], bucket["close"], bucket["volume"])
                handle_decision(strategy.update(closed5m))
                bucket = {"start_time": bucket_start, "open": c.open, "high": c.high,
                         "low": c.low, "close": c.close, "volume": c.volume}
            else:
                bucket["high"] = max(bucket["high"], c.high)
                bucket["low"] = min(bucket["low"], c.low)
                bucket["close"] = c.close
                bucket["volume"] += c.volume

            # --- Intracandle (ASAP) checks on THIS 1-minute bar: the entry
            #     trigger, fixed SL, and TRAIL all check real price crossing a
            #     level, firing the instant it happens rather than waiting for
            #     the (5m) closed-candle path. ---
            if not strategy.ready:
                continue
            if pos is not None:
                for price in (c.low, c.high):
                    long_sl, short_sl, level = strategy.check_intracandle_sl(price)
                    if long_sl or short_sl:
                        close_position("SL", c.start_time, c.close)
                        strategy.notify_exit(
                            SignalDir.LONG.value if long_sl else SignalDir.SHORT.value, "SL")
                        break
                    long_trail, short_trail, trail_level = strategy.check_intracandle_trail(price)
                    if long_trail or short_trail:
                        close_position("TRAIL", c.start_time, c.close)
                        strategy.notify_exit(
                            SignalDir.LONG.value if long_trail else SignalDir.SHORT.value, "TRAIL")
                        break
            elif strategy.has_pending:
                confirmed, invalidated, entry_price = strategy.apply_intracandle_pending(c)
                if confirmed:
                    signal_dir = (SignalDir.LONG.value
                                  if strategy.position_state == PositionState.LONG else SignalDir.SHORT.value)
                    open_position(signal_dir, c.start_time, entry_price, strategy.sl_level)

        report_open_position_if_any()

    return [t for t in trips if int(datetime.strptime(
        t["entry_ist"].replace(" IST", ""), "%Y-%m-%d %H:%M")
        .replace(tzinfo=_IST).timestamp()) >= win_start]


def main() -> None:
    ap = argparse.ArgumentParser(description="Heikin Ashi Supertrend(10,3) reversal -> SELL option backtest")
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--mode", choices=["asap", "closed"], default="asap",
                    help="asap = ASAP intracandle simulation (default; always uses 1m data internally, "
                         "resampled to 5m for the closed-candle path); closed = pure closed-candle-only "
                         "directly on --candle-resolution, no resampling, no intracandle")
    ap.add_argument("--candle-resolution", default="1m",
                    help="download/closed-candle resolution used when --mode closed (e.g. 1m, 3m, 5m, 15m, 30m, 1h); "
                         "ignored in --mode asap, which always uses 1m internally")
    ap.add_argument("--warmup-days", type=int, default=10, help="extra history so EMA(200)/Supertrend warm up before --days starts")
    ap.add_argument("--st-period", type=int, default=10, help="Supertrend ATR period (entry gate)")
    ap.add_argument("--st-multiplier", type=float, default=2.0, help="Supertrend ATR multiplier (entry gate); 2.0 validated to beat 3.0 on 1mo/4mo")
    ap.add_argument("--ema-length", type=int, default=50, help="EMA length used by both the entry trend filter and the trailing exit threshold")
    ap.add_argument("--ema200-length", type=int, default=200, help="EMA length for the trend filter (entry gate)")
    ap.add_argument("--no-session-gate", action="store_true",
                    help="ignore the session-open directional gate (BUY/SELL arm regardless of price vs session open)")
    ap.add_argument("--trail-ema200", action="store_true",
                    help="arm/tighten the trailing SL on the 200 EMA instead of the 50 EMA (arms later/deeper)")
    ap.add_argument("--day-start-hour", type=int, default=17)
    ap.add_argument("--day-start-minute", type=int, default=30)
    ap.add_argument("--square-off-hour", type=int, default=17)
    ap.add_argument("--square-off-minute", type=int, default=25)
    ap.add_argument("--execution", choices=["sell", "buy"], default="sell",
                    help="sell = SELL PUT on BUY / SELL CALL on SELL (default); "
                         "buy = BUY CE on BUY / BUY PE on SELL")
    ap.add_argument("--target-premium", type=float, default=900.0,
                    help="strike-selection anchor ONLY (which option to trade) -- not an exit condition")
    ap.add_argument("--opt-resolution", default="5m")
    ap.add_argument("--lots", type=int, default=None)
    ap.add_argument("--entry-slippage-pct", type=float, default=0.0)
    ap.add_argument("--exit-slippage-pct", type=float, default=0.0)
    ap.add_argument("--excel", default="heikin_ashi_backtest.xlsx")
    args = ap.parse_args()

    settings = load_settings()
    now = int(time.time())
    win_start = now - int(args.days * 86400)
    dl_resolution = "1m" if args.mode == "asap" else args.candle_resolution
    df = download(symbol=settings.symbol, start=win_start - args.warmup_days * 86400,
                  end=now, resolution=dl_resolution, base_url=settings.rest_base_url)
    candles = df_to_candles(df)
    trips = run(candles, settings, args)
    lots = args.lots if args.lots is not None else settings.option_contracts

    mode = "1m->5m ASAP-intracandle" if args.mode == "asap" else f"{args.candle_resolution} closed-candle-only"
    exec_desc = ("BUY CE on BUY-signal / BUY PE on SELL-signal" if args.execution == "buy"
                 else "SELL PUT on BUY-signal / SELL CALL on SELL-signal")
    print(f"\n{settings.symbol}  {mode} Heikin Ashi Supertrend(10,3)+EMA200 -> {exec_desc}, "
          f"target premium ~{args.target_premium:.0f}, lots {lots}, NO profit target")
    print(f"Window {_ist(win_start)} -> {_ist(now)}")
    print("=" * 104)
    print(f"{'Entry(IST)':<18}{'Exit(IST)':<18}{'Action':<10}{'Contract':<22}{'Why':<6}"
          f"{'PremIn':>8}{'PremOut':>8}{'Net$':>9}")
    print("-" * 104)
    # OPEN = still short/long when the data window ran out (the in-progress
    # trade); not a real exit, so it's shown but excluded from win/loss/P&L.
    closed = [t for t in trips if t["exit_reason"] != "OPEN"]
    open_trips = [t for t in trips if t["exit_reason"] == "OPEN"]

    net = fees = gross = 0.0
    wins = 0
    for t in trips:
        print(f"{t['entry_ist'][5:16]:<18}{t['exit_ist'][5:16]:<18}{t['action']:<10}"
              f"{t['contract']:<22}{t['exit_reason']:<6}{t['opt_in']:>8.1f}{t['opt_out']:>8.1f}"
              f"{t['net_usd']:>9.2f}")
        if t["exit_reason"] == "OPEN":
            continue
        net += t["net_usd"]; fees += t["fee_usd"]; gross += t["gross_usd"]
        wins += 1 if t["net_usd"] > 0 else 0
    print("=" * 104)
    n = len(closed)
    wr = (wins / n * 100.0) if n else 0.0
    reasons = {r: sum(1 for t in closed if t["exit_reason"] == r) for r in ("SL", "TRAIL", "EOD")}
    print(f"Trades {n}  Wins/Losses {wins}/{n - wins}  Win rate {wr:.1f}%  "
          f"exits: SL={reasons['SL']} TRAIL={reasons['TRAIL']} EOD={reasons['EOD']}"
          + (f"  (+{len(open_trips)} still OPEN, excluded)" if open_trips else ""))
    print(f"Gross {gross:,.2f}  -  brokerage {fees:,.2f}  =  NET {net:,.2f} USD")

    summary = [
        {"metric": "window_IST", "value": f"{_ist(win_start)} -> {_ist(now)}"},
        {"metric": "target_premium_strike_anchor", "value": args.target_premium},
        {"metric": "profit_target", "value": "NONE (SL / TRAIL / EOD only)"},
        {"metric": "lots", "value": lots},
        {"metric": "trades_closed", "value": n},
        {"metric": "trades_open_excluded", "value": len(open_trips)},
        {"metric": "win_rate_pct", "value": round(wr, 1)},
        {"metric": "exits_SL/TRAIL/EOD", "value": f"{reasons['SL']}/{reasons['TRAIL']}/{reasons['EOD']}"},
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
