"""Backtest: DCv2 -- Python port of dchannel_strategy.pine (2026-07-15 state).

Two execution modes:

  --mode btc (default): plain directional long/short on BTC (Pine
    strategy-tester style), P&L = points x --qty. Positions hold across days;
    exits are the strategy's own (range SL / EMA reversal / two-stage trail).

  --mode option: the signal SELLS options with a DAILY square-off + rollover
    (matches how the live bots run a daily option against a multi-day
    directional trade). BUY signal -> SELL a PUT near --target-premium
    (default 900); SELL signal -> SELL a CALL. Intraday the leg is bought back
    when the STRATEGY exits (range SL / EMA reversal / trail). Additionally
    the option is CLOSED at 17:25 IST (EOD square-off) and, if the underlying
    directional trade is still open, a fresh ~premium option is SOLD after
    17:30 IST (rollover). Intrinsic-value flooring is ON by default.

See src/deltabot/strategy/dcv2.py for the full rule set.

Run:  python scripts/backtest_dcv2.py --days 30
      python scripts/backtest_dcv2.py --mode option --days 30 --lots 25
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
from deltabot.strategy.dcv2 import DCv2Strategy

_IST = ZoneInfo("Asia/Kolkata")

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
    for res, sec in sorted(_NATIVE_RES.items(), key=lambda kv: -kv[1]):
        if sec <= target_sec and target_sec % sec == 0:
            return res, sec
    return "1m", 60


def _resample_candles(candles: list[Candle], target_sec: int) -> list[Candle]:
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


def _first_decay_time(candles: dict, after: int, upto: int, tp_price: float) -> int | None:
    """Earliest option-candle start_time in (after, upto] whose LOW decayed
    DOWN to tp_price -- the sell-side decay take-profit."""
    hit = [t for t, c in candles.items() if after < t <= upto and c.low <= tp_price]
    return min(hit) if hit else None


def _make_strategy(args) -> DCv2Strategy:
    return DCv2Strategy(
        dc_period=args.dc_period,
        ema_trend_length=args.ema_trend_length,
        ema_long_length=args.ema_long_length,
        skip_weekdays=frozenset({5, 6}) if not args.no_skip_weekends else frozenset(),
        day_start_hour=args.day_start_hour, day_start_minute=args.day_start_minute,
        square_off_hour=args.square_off_hour, square_off_minute=args.square_off_minute,
    )


def run(candles: list[Candle], args, sim_start: int) -> list[dict]:
    strategy = _make_strategy(args)

    trades: list[dict] = []
    open_trade: dict | None = None

    for candle in candles:
        decision = strategy.update(candle)
        if decision is None:
            continue

        if decision.has_exit and open_trade is not None:
            exit_price = decision.long_exit_price if decision.long_exit else decision.short_exit_price
            side = open_trade["side"]
            points = (exit_price - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - exit_price)
            open_trade.update(
                exit_time=candle.start_time, exit=exit_price,
                reason=decision.exit_reason, points=points,
                pnl=points * args.qty,
            )
            trades.append(open_trade)
            open_trade = None

        if decision.has_entry and candle.start_time >= sim_start:
            open_trade = {
                "entry_time": candle.start_time,
                "side": "LONG" if decision.buy_signal else "SHORT",
                "entry": decision.entry_price,
                "sl": decision.sl_level,
            }
        elif decision.has_entry:
            # Warmup-window entry: track it so its exit doesn't corrupt state,
            # but don't report it.
            open_trade = None
            strategy.force_flat()

    if open_trade is not None:
        last = candles[-1]
        side = open_trade["side"]
        points = (last.close - open_trade["entry"]) if side == "LONG" else (open_trade["entry"] - last.close)
        open_trade.update(exit_time=last.start_time, exit=last.close,
                          reason="OPEN_AT_END", points=points, pnl=points * args.qty)
        trades.append(open_trade)

    return trades


def report(trades: list[dict], args) -> None:
    print(f"\n{'=' * 100}")
    print(f"DCv2 backtest -- {args.days}d, {args.resolution}, DC({args.dc_period}), "
          f"EMA({args.ema_trend_length}/{args.ema_long_length}), qty {args.qty} BTC, "
          f"weekends {'TRADED' if args.no_skip_weekends else 'blocked'}")
    print(f"{'=' * 100}")
    if not trades:
        print("No trades.")
        return

    print(f"{'entry (IST)':<22}{'side':<7}{'entry':>12}{'exit':>12}{'sl':>12}"
          f"{'reason':<14}{'points':>10}{'pnl $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<22}{t['side']:<7}{t['entry']:>12.2f}{t['exit']:>12.2f}"
              f"{(t['sl'] if t['sl'] is not None else float('nan')):>12.2f}"
              f"{t['reason']:<14}{t['points']:>10.2f}{t['pnl']:>10.2f}")

    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["points"] > 0]
    total_pnl = sum(t["pnl"] for t in trades)
    print(f"{'-' * 100}")
    print(f"Trades: {len(closed)} closed"
          + (f" (+1 still open at data end)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "EMA_CROSS", "TRAIL", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<12} n={len(rs):<4} pnl ${sum(t['pnl'] for t in rs):>10.2f}")
    print(f"TOTAL P&L: ${total_pnl:.2f}  ({sum(t['points'] for t in trades):+.2f} BTC points x {args.qty})")


# --------------------------------------------------------------------- #
# Mode: option (SELL a PUT on a buy signal / a CALL on a sell signal, with a
# 17:25 daily square-off and a 17:30 rollover while the directional trade holds)
# --------------------------------------------------------------------- #
def run_option(candles: list[Candle], settings, args, sim_start: int) -> list[dict]:
    strategy = _make_strategy(args)
    underlying = settings.symbol.replace("USDT", "").replace("USD", "")
    interval = settings.option_strike_interval
    cutoff = settings.option_expiry_cutoff_hour
    lots = args.lots
    step = op.RES_SECONDS.get(args.opt_resolution, 60)
    es = args.entry_slippage_pct / 100.0
    xs = args.exit_slippage_pct / 100.0
    floor = not args.no_intrinsic_floor
    tp_decay = args.tp_decay_pct / 100.0   # 0 = off; 0.70 -> buy back at 30% of entry
    sq_mins = args.square_off_hour * 60 + args.square_off_minute
    sess_mins = args.day_start_hour * 60 + args.day_start_minute
    # Weekend blackout: no position held or opened from Fri --weekend-fri-hour
    # to Mon --weekend-mon-hour:min IST. Disabled by --no-skip-weekends.
    fri_start = args.weekend_fri_hour * 60
    mon_end = args.weekend_mon_hour * 60 + args.weekend_mon_minute

    wmode = "none" if args.no_skip_weekends else args.weekend_mode

    def in_blackout(ts: int) -> bool:
        if wmode != "blackout":
            return False
        d = datetime.fromtimestamp(ts, tz=_IST)
        wd, mins = d.weekday(), d.hour * 60 + d.minute
        if wd == 4:            # Friday from the blackout hour
            return mins >= fri_start
        if wd in (5, 6):       # Sat/Sun all day
            return True
        if wd == 0:            # Monday before the resume time
            return mins < mon_end
        return False

    def flatten_before_weekend(ts: int) -> bool:
        """fri-flat mode: at a square-off whose day is a skip day OR whose NEXT
        day is (so Friday's 17:25 ends the trade before Saturday), close the
        whole trade -- no rollover into the weekend."""
        if wmode != "fri-flat":
            return False
        wd = datetime.fromtimestamp(ts, tz=_IST).weekday()
        return wd in (5, 6) or ((wd + 1) % 7) in (5, 6)

    cache: dict = {}
    trades: list[dict] = []
    pos: dict | None = None
    prev_mins: int | None = None

    def open_leg(client, ts: int, btc_px: float, is_buy: bool, sl_level, tag: str) -> bool:
        nonlocal pos
        otype = OptionType.PUT if is_buy else OptionType.CALL
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
        pos = {"is_buy": is_buy, "sym": sym, "candles": ocandles, "entry_time": ts,
               "entry_btc": btc_px, "entry_prem": entry_prem, "sl_level": sl_level, "tag": tag,
               "tp_price": entry_prem * (1.0 - tp_decay) if tp_decay > 0 else None,
               "last_check": ts}
        return True

    def close(reason: str, exit_prem: float, exit_time: int, exit_btc: float) -> None:
        nonlocal pos
        assert pos is not None
        if floor:
            exit_prem = max(exit_prem, op.intrinsic_value(pos["sym"], exit_btc))
        entry_fill = pos["entry_prem"] * (1 - es)   # selling: receive less on entry
        exit_fill = exit_prem * (1 + xs)            # buying back: pay more
        gross = (entry_fill - exit_fill) * lots * op.LOT_BTC
        fee = (op.side_fee(pos["entry_btc"], entry_fill, lots)
               + op.side_fee(exit_btc, exit_fill, lots))
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": exit_time,
            "signal": "BUY" if pos["is_buy"] else "SELL", "contract": pos["sym"],
            "tag": pos["tag"], "btc_entry": pos["entry_btc"], "btc_exit": exit_btc,
            "opt_in": entry_fill, "opt_out": exit_fill, "reason": reason,
            "gross": gross, "fee": fee, "net": gross - fee,
        })
        pos = None

    def buyback_prem(ts: int, exit_btc: float) -> float:
        p = op.premium_at(pos["candles"], ts, step)
        return p if p is not None else op.intrinsic_value(pos["sym"], exit_btc)

    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for c in candles:
            mins = _ist_mins(c.start_time)
            square_off = prev_mins is not None and mins >= sq_mins and prev_mins < sq_mins
            in_gap = sq_mins <= mins < sess_mins
            prev_mins = mins
            dec = strategy.update(c)

            # 0. Weekend blackout (Fri 20:00 -> Mon 05:30 IST): no trade may be
            #    held or opened. Close any live leg at the window's first bar and
            #    keep the strategy flat until it reopens Monday. update() still
            #    ran above, so EMAs/Donchian stay warm through the weekend.
            if in_blackout(c.start_time):
                if pos is not None:
                    close("WEEKEND", buyback_prem(c.start_time, c.close), c.start_time, c.close)
                strategy.force_flat()
                continue

            # 1. Strategy exit (range SL / EMA reversal / trail): buy the leg back.
            if pos is not None and dec is not None and dec.has_exit:
                eprice = dec.long_exit_price if dec.long_exit else dec.short_exit_price
                close(dec.exit_reason, buyback_prem(c.start_time, eprice), c.start_time, eprice)

            # 1b. Decay take-profit (only if no strategy exit this bar, matching
            #     the conservative "strategy exit first" convention): buy the
            #     option back once its premium has decayed to tp_price. Booking
            #     the profit ENDS the trade -- flatten the strategy so it hunts
            #     a fresh signal instead of rolling the position over. The
            #     intrinsic floor blocks an impossible decay below intrinsic.
            if pos is not None and tp_decay > 0 and pos["tp_price"] is not None:
                can_decay = (not floor) or op.intrinsic_value(pos["sym"], c.close) <= pos["tp_price"]
                t_tp = (_first_decay_time(pos["candles"], pos["last_check"], c.start_time, pos["tp_price"])
                        if can_decay else None)
                if t_tp is not None:
                    close("TP", pos["tp_price"], t_tp, c.close)
                    strategy.force_flat()
                else:
                    pos["last_check"] = c.start_time

            # 2. 17:25 EOD square-off: close the option. Normally the directional
            #    trade keeps running across the gap and rolls at 17:30. In
            #    fri-flat mode, the last square-off before the weekend (Friday)
            #    also FLATTENS the trade so nothing carries over (reason WEEKEND).
            if pos is not None and square_off:
                weekend = flatten_before_weekend(c.start_time)
                close("WEEKEND" if weekend else "EOD",
                      buyback_prem(c.start_time, c.close), c.start_time, c.close)
                if weekend:
                    strategy.force_flat()

            # 3. New entry from a fresh strategy signal.
            if pos is None and dec is not None and dec.has_entry:
                if c.start_time < sim_start:
                    strategy.force_flat()   # warmup-window entry: don't take it
                elif not open_leg(client, c.start_time, c.close, dec.buy_signal, dec.sl_level, "ENTRY"):
                    strategy.force_flat()   # couldn't price the contract; stay flat

            # 4. Rollover: past the 17:30 gap, if the directional trade is still
            #    open but the option is flat (squared off at 17:25), re-sell.
            #    Blackout mode blocks weekend rolls via section 0; fri-flat mode
            #    flattens at Friday's square-off, but guard Sat/Sun here too.
            elif (pos is None and not in_gap and not square_off
                  and c.start_time >= sim_start
                  and strategy.position_state.name != "FLAT"
                  and not (wmode == "fri-flat"
                           and datetime.fromtimestamp(c.start_time, tz=_IST).weekday() in (5, 6))):
                is_buy = strategy.position_state.name == "LONG"
                open_leg(client, c.start_time, c.close, is_buy, strategy.sl_level, "ROLL")

        if pos is not None:
            last = candles[-1]
            close("OPEN_AT_END", buyback_prem(last.start_time, last.close), last.start_time, last.close)

    return trades


def report_option(trades: list[dict], args) -> None:
    print(f"\n{'=' * 122}")
    tp_txt = f"{args.tp_decay_pct:.0f}%-decay TP" if args.tp_decay_pct > 0 else "no TP"
    if args.no_skip_weekends:
        wknd_txt = "weekends TRADED"
    elif args.weekend_mode == "fri-flat":
        wknd_txt = "flat at Fri 17:25"
    elif args.weekend_mode == "none":
        wknd_txt = "roll thru weekend (Sat/Sun entries blocked)"
    else:
        wknd_txt = f"blackout Fri {args.weekend_fri_hour:02d}:00->Mon {args.weekend_mon_hour:02d}:{args.weekend_mon_minute:02d}"
    print(f"DCv2 [OPTION SELL, 17:25 square-off + 17:30 rollover] -- {args.days}d, {args.resolution}, "
          f"premium ~{args.target_premium:.0f}, {args.lots} lots, {tp_txt}, "
          f"floor {'OFF' if args.no_intrinsic_floor else 'ON'}, {wknd_txt}")
    print(f"{'=' * 122}")
    if not trades:
        print("No trades.")
        return
    print(f"{'entry (IST)':<22}{'sig':<5}{'tag':<6}{'contract':<22}{'btc in':>9}{'btc out':>9}"
          f"{'opt in':>8}{'opt out':>8} {'reason':<12}{'net $':>10}")
    for t in trades:
        print(f"{_ist(t['entry_time']):<22}{t['signal']:<5}{t['tag']:<6}{t['contract']:<22}"
              f"{t['btc_entry']:>9.1f}{t['btc_exit']:>9.1f}{t['opt_in']:>8.1f}{t['opt_out']:>8.1f} "
              f"{t['reason']:<12}{t['net']:>10.2f}")
    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    print(f"{'-' * 122}")
    print(f"Legs: {len(closed)} closed"
          + (" (+1 still open at data end)" if len(trades) != len(closed) else ""))
    if closed:
        print(f"Win rate: {len(wins)}/{len(closed)} = {100.0 * len(wins) / len(closed):.1f}%")
    for reason in ("SL", "EMA_CROSS", "TRAIL", "TP", "EOD", "WEEKEND", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            print(f"  {reason:<12} n={len(rs):<4} net ${sum(t['net'] for t in rs):>11.2f}")
    total_fee = sum(t["fee"] for t in trades)
    print(f"TOTAL NET: ${sum(t['net'] for t in trades):.2f} "
          f"(gross ${sum(t['gross'] for t in trades):.2f}, fees ${total_fee:.2f})")


def export_option(trades: list[dict], args, path: str) -> None:
    """Write every leg to an .xlsx (Trades sheet + a Summary sheet)."""
    import pandas as pd

    rows, cum = [], 0.0
    for i, t in enumerate(trades, 1):
        cum += t["net"]
        rows.append({
            "#": i,
            "entry_IST": _ist(t["entry_time"]),
            "exit_IST": _ist(t["exit_time"]),
            "signal": t["signal"],
            "leg": t["tag"],                 # ENTRY or ROLL
            "contract": t["contract"],
            "btc_entry": round(t["btc_entry"], 1),
            "btc_exit": round(t["btc_exit"], 1),
            "opt_sold": round(t["opt_in"], 1),
            "opt_bought_back": round(t["opt_out"], 1),
            "exit_reason": t["reason"],
            "gross_usd": round(t["gross"], 2),
            "fee_usd": round(t["fee"], 2),
            "net_usd": round(t["net"], 2),
            "cumulative_net_usd": round(cum, 2),
        })
    df = pd.DataFrame(rows)

    closed = [t for t in trades if t["reason"] != "OPEN_AT_END"]
    wins = [t for t in closed if t["net"] > 0]
    srows = [
        ("days", args.days), ("resolution", args.resolution),
        ("premium_target", args.target_premium), ("lots", args.lots),
        ("tp_decay_pct", args.tp_decay_pct), ("weekend_mode", args.weekend_mode),
        ("legs_closed", len(closed)),
        ("win_rate_pct", round(100.0 * len(wins) / len(closed), 1) if closed else 0.0),
        ("gross_usd", round(sum(t["gross"] for t in trades), 2)),
        ("fees_usd", round(sum(t["fee"] for t in trades), 2)),
        ("TOTAL_NET_usd", round(sum(t["net"] for t in trades), 2)),
    ]
    for reason in ("SL", "EMA_CROSS", "TRAIL", "TP", "EOD", "WEEKEND", "OPEN_AT_END"):
        rs = [t for t in trades if t["reason"] == reason]
        if rs:
            srows.append((f"{reason}_n", len(rs)))
            srows.append((f"{reason}_net_usd", round(sum(t["net"] for t in rs), 2)))
    summary = pd.DataFrame(srows, columns=["metric", "value"])

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        df.to_excel(xl, sheet_name="Trades", index=False)
    print(f"\nExcel written: {path}  ({len(df)} legs)")


def main() -> None:
    p = argparse.ArgumentParser(description="DCv2 (Pine port) backtest -- directional BTC or option-sell")
    p.add_argument("--mode", choices=("btc", "option"), default="btc")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--warmup-days", type=int, default=7,
                   help="extra leading days so EMA200/Donchian are warm before the report window")
    p.add_argument("--resolution", default="5m")
    p.add_argument("--dc-period", type=int, default=20)
    p.add_argument("--ema-trend-length", type=int, default=50)
    p.add_argument("--ema-long-length", type=int, default=200)
    p.add_argument("--qty", type=float, default=1.0, help="[btc mode] BTC position size (P&L = points x qty)")
    p.add_argument("--no-skip-weekends", action="store_true",
                   help="also take entries on Sat/Sun (blocked by default, like the Pine script)")
    p.add_argument("--day-start-hour", type=int, default=17)
    p.add_argument("--day-start-minute", type=int, default=30)
    p.add_argument("--square-off-hour", type=int, default=17)
    p.add_argument("--square-off-minute", type=int, default=25)
    # option mode
    p.add_argument("--target-premium", type=float, default=900.0,
                   help="[option mode] sell the strike whose premium is nearest this (default 900)")
    p.add_argument("--lots", type=int, default=25)
    p.add_argument("--opt-resolution", default="1m", help="[option mode] option candle resolution")
    p.add_argument("--entry-slippage-pct", type=float, default=0.0)
    p.add_argument("--exit-slippage-pct", type=float, default=0.0)
    p.add_argument("--no-intrinsic-floor", action="store_true",
                   help="[option mode] disable intrinsic-value flooring (NOT recommended)")
    p.add_argument("--tp-decay-pct", type=float, default=0.0,
                   help="[option mode] book profit when the sold option decays this %% "
                        "(e.g. 70 -> buy back at 30%% of entry premium; 0 = off)")
    p.add_argument("--out", default="",
                   help="[option mode] also write every leg to this .xlsx file")
    p.add_argument("--weekend-mode", choices=("blackout", "fri-flat", "none"), default="blackout",
                   help="[option mode] weekend handling: blackout (Fri 20:00->Mon 05:30 no trade), "
                        "fri-flat (flatten the whole trade at Friday 17:25, roll Mon-Thu), "
                        "none (roll through the weekend, only Sat/Sun ENTRIES blocked)")
    p.add_argument("--weekend-fri-hour", type=int, default=20,
                   help="[option mode] weekend blackout starts Friday at this IST hour (default 20:00)")
    p.add_argument("--weekend-mon-hour", type=int, default=5,
                   help="[option mode] weekend blackout ends Monday at this IST hour (default 05:xx)")
    p.add_argument("--weekend-mon-minute", type=int, default=30,
                   help="[option mode] weekend blackout end minute (default :30 -> 05:30)")
    args = p.parse_args()

    setup_logging("WARNING")
    settings = load_settings()

    now = int(time.time())
    sim_start = now - args.days * 86400
    dl_start = sim_start - args.warmup_days * 86400

    target_sec = _res_seconds(args.resolution)
    if target_sec is None:
        raise SystemExit(f"Unsupported resolution: {args.resolution}")
    if args.resolution in _NATIVE_RES:
        df = download(symbol=settings.symbol, start=dl_start, end=now, resolution=args.resolution)
        candles = df_to_candles(df)
    else:
        base_res, _ = _base_for(target_sec)
        df = download(symbol=settings.symbol, start=dl_start, end=now, resolution=base_res)
        candles = _resample_candles(df_to_candles(df), target_sec)

    print(f"Candles: {len(candles)} ({args.resolution}) "
          f"{_ist(candles[0].start_time)} .. {_ist(candles[-1].start_time)}")
    if args.mode == "option":
        trades = run_option(candles, settings, args, sim_start)
        report_option(trades, args)
        if args.out:
            export_option(trades, args, args.out)
    else:
        report(run(candles, args, sim_start), args)


if __name__ == "__main__":
    main()
