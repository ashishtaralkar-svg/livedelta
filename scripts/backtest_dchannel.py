"""Backtest: Dchannel Strategy (2026-07-10 rewrite) -- Williams %R + Donchian
touch + open=low/open=high + EMA(1000), on 1-minute synthetic Heikin Ashi
candles. Option BUYING (unlike the pre-rewrite version, which sold).

See src/deltabot/strategy/dchannel.py for the full rule set. Exits per trade,
first to fire: BTC price touching the signal-range SL, BTC price touching the
1:2 risk:reward TP (--rr-multiple, both fully internal/BTC-driven now), or
17:25 IST EOD square-off. --trade-window optionally restricts which ENTRY
timestamps get taken (e.g. --trade-window --window-start-hour 10 to only
trade 10:00-17:25 IST) without changing the underlying signal hunt.

Run:  python scripts/backtest_dchannel.py --days 7
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
from deltabot.enums import OptionType
from deltabot.logging_setup import setup_logging
from deltabot.models import Candle
from deltabot.strategy.dchannel import DchannelStrategy

_IST = ZoneInfo("Asia/Kolkata")

# Resolutions Delta's candle API serves natively; anything else (e.g. 10m) is
# synthesized by aggregating a native base that evenly divides it.
_NATIVE_RES = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}


def _res_seconds(res: str) -> int | None:
    if res in _NATIVE_RES:
        return _NATIVE_RES[res]
    if res.endswith("m"):
        try:
            return int(res[:-1]) * 60
        except ValueError:
            return None
    return None


def _base_for(target_sec: int) -> tuple[str, int]:
    """Largest native resolution that evenly divides the target period."""
    for res, sec in sorted(_NATIVE_RES.items(), key=lambda kv: -kv[1]):
        if sec <= target_sec and target_sec % sec == 0:
            return res, sec
    return "1m", 60


def _resample_candles(candles: list[Candle], target_sec: int) -> list[Candle]:
    """Aggregate consecutive base candles into buckets aligned to multiples of
    ``target_sec`` from epoch (open=first, high=max, low=min, close=last)."""
    buckets: dict[int, list] = {}
    order: list[int] = []
    for c in candles:
        key = c.start_time - (c.start_time % target_sec)
        b = buckets.get(key)
        if b is None:
            buckets[key] = [c.open, c.high, c.low, c.close, c.volume]
            order.append(key)
        else:
            b[1] = max(b[1], c.high)
            b[2] = min(b[2], c.low)
            b[3] = c.close
            b[4] += c.volume
    return [Candle(start_time=k, open=b[0], high=b[1], low=b[2], close=b[3], volume=b[4])
            for k in order for b in (buckets[k],)]


def _ist(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M IST")


def _ist_mins(ts: int) -> int:
    d = datetime.fromtimestamp(ts, tz=_IST)
    return d.hour * 60 + d.minute


def _first_tp_time(candles: dict, after: int, upto: int, tp_price: float, step: int) -> int | None:
    """Earliest option-candle start_time in (after, upto] where premium rose to
    tp_price -- BUY-side rally TP (checks the bucket HIGH). Only used in
    --tp-mode premium."""
    hit = [t for t, c in candles.items() if after < t <= upto and c.high >= tp_price]
    return min(hit) if hit else None


def _first_decay_time(candles: dict, after: int, upto: int, tp_price: float, step: int) -> int | None:
    """Earliest option-candle start_time in (after, upto] where premium decayed
    DOWN to tp_price -- SELL-side decay TP (checks the bucket LOW). Only used
    with --side sell."""
    hit = [t for t, c in candles.items() if after < t <= upto and c.low <= tp_price]
    return min(hit) if hit else None


def run(candles, settings, args) -> list[dict]:
    # --tp-mode premium: the internal BTC-driven TP is effectively disabled
    # (rr_multiple huge -> unreachable in practice) and a premium-%% target is
    # checked externally instead, exactly like the strategy's pre-2026-07-10
    # behavior. --tp-mode rr (default) uses the strategy's own 1:2 BTC-price TP.
    # The strategy's OWN internal BTC-price TP is used whenever --tp-mode rr,
    # for BOTH buy and sell (for sell the underlying BTC bet is still
    # directional -- bullish=sell PUT profits as BTC rises to entry+risk*rr,
    # so the same long/short RR TP is exactly right). It is disabled
    # (unreachable) when an EXTERNAL target drives the exit instead:
    # tp-mode=premium (buy rally / sell decay) or --no-target (SL/EOD only).
    rr = args.rr_multiple if (args.tp_mode == "rr" and not args.no_target) else 10_000.0
    # tp-mode btcpct: internal BTC-price TP at a flat +/- pct of entry (both buy
    # and sell). Overrides the RR TP inside the strategy via tp_pct.
    tp_pct = (args.tp_btc_pct / 100.0) if (args.tp_mode == "btcpct" and not args.no_target) else None
    strategy = DchannelStrategy(
        dc_period=args.dc_period, wr_period=args.wr_period, wr_level=args.wr_level,
        ema_length=args.ema_length, ma_length=args.ma_length, wr_enabled=(args.wr == "on"),
        rr_multiple=rr, tp_pct=tp_pct, day_tz=settings.day_tz,
        day_start_hour=args.day_start_hour, day_start_minute=args.day_start_minute,
        square_off_hour=args.square_off_hour, square_off_minute=args.square_off_minute,
    )
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    win_start = int(time.time()) - int(args.days * 86400)
    cache: dict = {}
    trips: list[dict] = []
    pos: dict | None = None

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float):
        nonlocal pos
        assert pos is not None
        if args.side == "sell":
            entry_fill = pos["entry_prem"] * (1 - es)   # SELLING: receive less on entry
            exit_fill = exit_prem * (1 + xs)              # SELLING: pay more to buy back
            gross = (entry_fill - exit_fill) * lots * op.LOT_BTC
        else:
            entry_fill = pos["entry_prem"] * (1 + es)   # BUYING: pay more on entry
            exit_fill = exit_prem * (1 - xs)              # BUYING: receive less on exit
            gross = (exit_fill - entry_fill) * lots * op.LOT_BTC
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots)
               + op.side_fee(exit_btc, exit_fill, lots))
        action = ("SELL " if args.side == "sell" else "BUY ") + ("CALL" if pos["sym"].startswith("C-") else "PUT")
        # Extreme BTC price reached in the FAVORABLE direction while the trade
        # was open: highest for a bullish (buy) signal, lowest for a bearish
        # (sell) signal. For SL trades this shows how far it ran in your favour
        # before reversing to stop you out.
        btc_fav = pos["btc_high"] if pos["is_buy"] else pos["btc_low"]
        trips.append({
            "action": action, "signal": "BUY" if pos["is_buy"] else "SELL",
            "contract": pos["sym"], "entry_time_ist": _ist(pos["entry_time"]),
            "exit_time_ist": _ist(exit_time), "exit_reason": reason,
            "btc_entry": round(pos["entry_btc"], 1), "btc_exit": round(exit_btc, 1),
            "sl_level": round(pos["sl_level"], 1) if pos.get("sl_level") else None,
            "btc_favorable": round(btc_fav, 1),
            "btc_high": round(pos["btc_high"], 1), "btc_low": round(pos["btc_low"], 1),
            "opt_in": round(entry_fill, 1), "opt_out": round(exit_fill, 1),
            "lots": lots, "gross_usd": round(gross, 2),
            "fee_usd": round(fee, 2), "net_usd": round(gross - fee, 2),
        })
        pos = None

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            dec = strategy.update(c)

            # Track the max favourable BTC excursion while a trade is open
            # (includes the exit bar's range). Entry bar is seeded at entry.
            if pos is not None:
                pos["btc_high"] = max(pos["btc_high"], c.high)
                pos["btc_low"] = min(pos["btc_low"], c.low)

            # 1. BTC exit — SL and EOD are internal to the strategy (dec.exit_
            #    reason tells us which; in --tp-mode rr, TP is internal too and
            #    arrives here as "TP"). The option is priced at that moment.
            if pos is not None and dec is not None and dec.has_exit:
                eprice = dec.long_exit_price if dec.long_exit else dec.short_exit_price
                exit_prem = op.premium_at(pos["candles"], c.start_time, step)
                if exit_prem is not None:
                    close(dec.exit_reason, exit_prem, c.start_time, eprice)
                else:
                    pos = None  # cannot price exit; drop (strategy already flat)

            # 1b. External premium take-profit (only if no BTC exit this bar):
            #     --side buy + --tp-mode premium -> rally target (bucket HIGH).
            #     --side sell -> decay target (bucket LOW), always external
            #     since the strategy's internal TP is BTC-price/buy-shaped.
            if pos is not None and args.side == "sell" and args.tp_mode == "premium" and not args.no_target:
                t_tp = _first_decay_time(pos["candles"], pos["last_check"], c.start_time, pos["tp_price"], step)
                if t_tp is not None:
                    strategy.notify_exit("TP")
                    close("TP", pos["tp_price"], t_tp, pos["entry_btc"])
                else:
                    pos["last_check"] = c.start_time
            elif pos is not None and args.side == "buy" and args.tp_mode == "premium":
                t_tp = _first_tp_time(pos["candles"], pos["last_check"], c.start_time, pos["tp_price"], step)
                if t_tp is not None:
                    strategy.notify_exit("TP")
                    close("TP", pos["tp_price"], t_tp, pos["entry_btc"])
                else:
                    pos["last_check"] = c.start_time

            # 2. New entry — BUY mode: bullish -> buy CALL, bearish -> buy PUT.
            #    SELL mode (matches RevBreak's convention): bullish -> sell
            #    PUT, bearish -> sell CALL. --trade-window restricts which
            #    ENTRY timestamps are actually taken (the underlying signal-
            #    hunt still runs continuously; this only gates execution).
            if pos is None and dec is not None and dec.has_entry:
                entry_mins = _ist_mins(c.start_time)
                if args.trade_window and not (args.window_start_mins <= entry_mins < args.window_end_mins):
                    strategy.notify_exit("SL")  # signal fired outside the window -- skip it
                    continue
                is_buy = dec.buy_signal
                if args.side == "sell":
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
                    strategy.notify_exit("SL")  # couldn't price it -- stay flat
                    continue
                sym, _, ocandles = resolved
                entry_prem = op.premium_at(ocandles, c.start_time, step)
                if entry_prem is None:
                    strategy.notify_exit("SL")
                    continue
                pos = {
                    "is_buy": is_buy, "sym": sym, "candles": ocandles, "entry_time": c.start_time,
                    "entry_btc": c.close, "entry_prem": entry_prem,
                    "sl_level": dec.sl_level,
                    "btc_high": c.high, "btc_low": c.low,
                }
                if args.side == "sell" and args.tp_mode == "premium" and not args.no_target:
                    # Absolute decay target (e.g. entry ~300 -> target 200),
                    # matching the pre-2026-07-10 Dchannel's fixed-level convention.
                    pos["tp_price"] = args.tp_level if args.tp_level is not None else entry_prem * 0.7
                    pos["last_check"] = c.start_time
                elif args.side == "buy" and args.tp_mode == "premium":
                    pos["tp_price"] = entry_prem * (1.0 + args.take_profit / 100.0)
                    pos["last_check"] = c.start_time

    return [t for t in trips if op_entry_ts(t) >= win_start]


def op_entry_ts(trip: dict) -> int:
    return int(datetime.strptime(trip["entry_time_ist"].replace(" IST", ""),
                                 "%Y-%m-%d %H:%M").replace(tzinfo=_IST).timestamp())


def main() -> None:
    ap = argparse.ArgumentParser(description="Dchannel Strategy (WR + DC-touch + EMA1000) option-buy backtest")
    ap.add_argument("--days", type=float, default=7, help="look-back window in days")
    ap.add_argument("--resolution", default="1m", help="BTC candle resolution (default 1m, matches live)")
    ap.add_argument("--warmup-days", type=int, default=3, help="extra history so EMA(1000) is warm from day 1")
    ap.add_argument("--dc-period", type=int, default=20)
    ap.add_argument("--wr-period", type=int, default=14)
    ap.add_argument("--wr-level", type=float, default=80.0,
                    help="arm bull hunt when %%R < -this; arm bear hunt when %%R > -(100-this) (default 80)")
    ap.add_argument("--ema-length", type=int, default=1000)
    ap.add_argument("--ma-length", type=int, default=0,
                    help="0 (default) = trend filter is price-vs-EMA; >0 switches to an "
                         "EMA-vs-SMA(this length) cross filter (bull: EMA>SMA, bear: EMA<SMA)")
    ap.add_argument("--wr", choices=["on", "off"], default="on",
                    help="on (default) = Williams %%R oversold/overbought gate arms the hunt; "
                         "off = no %%R gate, both directions hunt continuously on DC touch")
    ap.add_argument("--day-start-hour", type=int, default=17)
    ap.add_argument("--day-start-minute", type=int, default=30)
    ap.add_argument("--square-off-hour", type=int, default=17)
    ap.add_argument("--square-off-minute", type=int, default=25)
    ap.add_argument("--target-premium", type=float, default=125.0, help="default 125, midpoint of the 100-150 range")
    ap.add_argument("--side", choices=["buy", "sell"], default="buy",
                    help="buy (default) = long option, matches the 2026-07-10 rewrite; "
                         "sell = short option matching RevBreak's convention (bullish -> sell PUT, "
                         "bearish -> sell CALL), SL stays BTC-price/internal, TP becomes an "
                         "absolute premium DECAY target via --tp-level")
    ap.add_argument("--tp-level", type=float, default=None,
                    help="side sell only: absolute premium decay target (e.g. entry ~300 -> "
                         "--tp-level 200). Defaults to 70%% of entry premium if not given.")
    ap.add_argument("--no-target", action="store_true",
                    help="side sell only: no take-profit at all -- ride to the (unchanged, BTC-price) "
                         "SL or the 17:25 EOD square-off, whichever fires first")
    ap.add_argument("--tp-mode", choices=["rr", "premium", "btcpct"], default="rr",
                    help="rr (default) = internal BTC-price RR target via --rr-multiple; "
                         "btcpct = internal BTC-price target at a flat +/- --tp-btc-pct%% of entry; "
                         "premium = external target (buy: +--take-profit%% rally; sell: --tp-level decay). "
                         "rr and btcpct both work for BOTH buy and sell (the bullish/bearish BTC bet "
                         "is the same regardless of option side).")
    ap.add_argument("--rr-multiple", type=float, default=2.0,
                    help="tp-mode rr only: TP = entry +/- this * (entry-to-SL risk), on BTC price (default 2.0)")
    ap.add_argument("--tp-btc-pct", type=float, default=0.5,
                    help="tp-mode btcpct only: TP = entry +/- this %% of the entry BTC price (default 0.5)")
    ap.add_argument("--take-profit", type=float, default=25.0,
                    help="tp-mode premium only: %% premium gain TP (default 25)")
    ap.add_argument("--entry-slippage-pct", type=float, default=1.0,
                    help="premium %% given up on entry (default 1%%, calibrated -- see commit b24921f)")
    ap.add_argument("--exit-slippage-pct", type=float, default=5.0,
                    help="premium %% paid through the spread on exit (default 5%%, calibrated)")
    ap.add_argument("--opt-resolution", default="1m")
    ap.add_argument("--lots", type=int, default=10, help="default 10, matches live sizing")
    ap.add_argument("--trade-window", action="store_true",
                    help="only TAKE entries whose trigger fires inside --window-start/--window-end "
                         "(the signal hunt itself still runs continuously; this only gates execution)")
    ap.add_argument("--window-start-hour", type=int, default=10)
    ap.add_argument("--window-start-minute", type=int, default=0)
    ap.add_argument("--window-end-hour", type=int, default=17)
    ap.add_argument("--window-end-minute", type=int, default=25)
    ap.add_argument("--excel", default="dchannel_backtest_IST.xlsx")
    args = ap.parse_args()
    args.window_start_mins = args.window_start_hour * 60 + args.window_start_minute
    args.window_end_mins = args.window_end_hour * 60 + args.window_end_minute

    settings = load_settings()
    setup_logging("WARNING")
    now = int(time.time())
    win_start = now - int(args.days * 86400)
    dl_start = win_start - args.warmup_days * 86400
    if args.resolution in _NATIVE_RES:
        df = download(symbol=settings.symbol, start=dl_start, end=now,
                      resolution=args.resolution, base_url=settings.rest_base_url)
        candles = df_to_candles(df)
    else:
        target_sec = _res_seconds(args.resolution)
        if target_sec is None:
            raise SystemExit(f"Unsupported resolution: {args.resolution}")
        base_res, _ = _base_for(target_sec)
        df = download(symbol=settings.symbol, start=dl_start, end=now,
                      resolution=base_res, base_url=settings.rest_base_url)
        candles = _resample_candles(df_to_candles(df), target_sec)
        print(f"(synthesized {args.resolution} from {base_res}: {len(candles)} candles)")
    trips = run(candles, settings, args)

    if args.no_target:
        tp_desc = "none -- SL/EOD only"
    elif args.tp_mode == "rr":
        tp_desc = f"1:{args.rr_multiple:g} RR (BTC)"
    elif args.tp_mode == "btcpct":
        tp_desc = f"{args.tp_btc_pct:g}% of BTC price"
    elif args.side == "sell":
        eff_tp_level = args.tp_level if args.tp_level is not None else args.target_premium * 0.7
        tp_desc = f"decay to {eff_tp_level:.0f}"
    else:
        tp_desc = f"+{args.take_profit:.0f}% premium"
    print(f"\n{settings.symbol}  {args.resolution}  Dchannel (WR{args.wr_period} + DC{args.dc_period} + "
          f"EMA{args.ema_length}) -> {args.side.upper()} option ~{args.target_premium:.0f} prem, "
          f"TP {tp_desc}, lots {args.lots}")
    print(f"Window {_ist(win_start)} -> {_ist(now)}")
    print("=" * 118)
    fav_hdr = "MFE(hi/lo)"
    print(f"{'Entry(IST)':<15}{'Exit(IST)':<15}{'Action':<10}{'Contract':<22}{'Why':<5}"
          f"{'SL':>9}{fav_hdr:>11}{'PremIn':>8}{'PremOut':>8}{'Net$':>9}")
    print("-" * 118)
    net = fees = gross = 0.0
    wins = 0
    for t in trips:
        sl_s = f"{t['sl_level']:.0f}" if t['sl_level'] is not None else "-"
        print(f"{t['entry_time_ist'][5:14]:<15}{t['exit_time_ist'][5:14]:<15}{t['action']:<10}"
              f"{t['contract']:<22}{t['exit_reason']:<5}{sl_s:>9}{t['btc_favorable']:>11.0f}"
              f"{t['opt_in']:>8.1f}{t['opt_out']:>8.1f}{t['net_usd']:>9.2f}")
        net += t["net_usd"]; fees += t["fee_usd"]; gross += t["gross_usd"]
        wins += 1 if t["net_usd"] > 0 else 0
    print("=" * 118)
    n = len(trips)
    wr = (wins / n * 100.0) if n else 0.0
    reasons = {r: sum(1 for t in trips if t["exit_reason"] == r) for r in ("TP", "SL", "EOD")}
    print(f"Trades {n}  Wins/Losses {wins}/{n - wins}  Win rate {wr:.1f}%  "
          f"exits: TP={reasons['TP']} SL={reasons['SL']} EOD={reasons['EOD']}")
    print(f"Gross {gross:,.2f}  -  brokerage {fees:,.2f}  =  NET {net:,.2f} USD")

    summary = [
        {"metric": "window_IST", "value": f"{_ist(win_start)} -> {_ist(now)}"},
        {"metric": "dc_period", "value": args.dc_period},
        {"metric": "wr_period", "value": args.wr_period},
        {"metric": "ema_length", "value": args.ema_length},
        {"metric": "target_premium", "value": args.target_premium},
        {"metric": "rr_multiple", "value": args.rr_multiple},
        {"metric": "lots", "value": args.lots},
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
