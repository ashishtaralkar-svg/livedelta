"""Option-price backtest (strategy 1, options mode) over a recent window.

Unlike ``deltabot.cli backtest`` — which prices trades from the BTC directional
move (linear convention) — this script reprices every trade with the **real
option premiums** that traded on Delta:

  * Signals come from the SAME ``PineStrategy`` used live and in the futures
    backtest, run over BTC candles (so signals cannot diverge).
  * Each signal is mapped to the option the live ``OptionsExecutor`` would sell:
        BUY  -> sell ITM PUT  (strike = BTC_entry + offset)
        SELL -> sell ITM CALL (strike = BTC_entry - offset)
    same-day daily expiry (per the IST cutoff-hour rule), snapped to the nearest
    strike that actually traded.
  * Entry premium  = the option's candle at the entry time (premium received).
  * Exit  premium  = the option's candle at the exit time (premium paid back).
  * Short-option PnL (USD) = (entry_premium - exit_premium) * lots * lot_btc.

Run:  python scripts/backtest_options.py [--days 3] [--resolution 5m]
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.backtest.engine import BacktestEngine
from deltabot.config import load_settings
from deltabot.enums import OptionType, SignalDir
from deltabot.logging_setup import setup_logging
from deltabot.models import Candle

_IST = ZoneInfo("Asia/Kolkata")
_LOT_BTC = 0.001  # 1 option lot = 0.001 BTC of underlying on Delta BTC options
_RES_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}

# Delta options taker fee: 0.03% of underlying notional, capped at 10% of premium,
# charged per side (entry AND exit). This is an estimate; it ignores bid/ask spread.
_TAKER_FEE_PCT = 0.0003
_PREMIUM_FEE_CAP_PCT = 0.10


def _side_fee(underlying_px: float, premium: float, lots: int) -> float:
    """Per-side option fee in USD for ``lots`` lots (min of notional% and premium%)."""
    per_lot = min(_TAKER_FEE_PCT * underlying_px, _PREMIUM_FEE_CAP_PCT * premium) * _LOT_BTC
    return per_lot * lots


def _ist(ts: int) -> str:
    """Format an epoch second as 'YYYY-MM-DD HH:MM IST' in Asia/Kolkata."""
    return datetime.fromtimestamp(ts, tz=_IST).strftime("%Y-%m-%d %H:%M IST")


def _select_expiry_date(entry_ts: int, cutoff_hour: int) -> datetime:
    """Same-day daily expiry in IST, rolling to next day past the cutoff hour.

    Mirrors ``OptionsExecutor._select_expiry`` but anchored at the trade's entry
    time instead of "now".
    """
    ist = datetime.fromtimestamp(entry_ts, tz=_IST)
    if ist.hour >= cutoff_hour:
        ist = ist + timedelta(days=1)
    return ist


def _fetch_option_candles(
    client: httpx.Client, symbol: str, start: int, end: int, resolution: str
) -> dict[int, Candle]:
    """Fetch one option's history as a {start_time: Candle} map (empty if none)."""
    resp = client.get(
        "/v2/history/candles",
        params={"symbol": symbol, "resolution": resolution, "start": start, "end": end},
    )
    resp.raise_for_status()
    rows = resp.json().get("result") or []
    out: dict[int, Candle] = {}
    for r in rows:
        c = Candle.from_rest(r)
        # Some expired-contract rows carry 0.0 O/C placeholders — skip them.
        if c.close > 0:
            out[c.start_time] = c
    return out


def _premium_at(candles: dict[int, Candle], ts: int, step: int) -> float | None:
    """Option close at/just-before ``ts`` (search back a few bars for a gap)."""
    for k in range(0, 8):
        c = candles.get(ts - k * step)
        if c is not None:
            return c.close
    # Fall back to the nearest earlier candle of any time.
    earlier = [c for t, c in candles.items() if t <= ts]
    return max(earlier, key=lambda c: c.start_time).close if earlier else None


def _resolve_contract(
    client: httpx.Client,
    underlying: str,
    option_type: OptionType,
    target_strike: int,
    expiry: datetime,
    interval: int,
    entry_ts: int,
    exit_ts: int,
    win_start: int,
    win_end: int,
    resolution: str,
    step: int,
) -> tuple[str, int, dict[int, Candle]] | None:
    """Snap to the nearest traded strike and return (symbol, strike, candles).

    Searches strikes outward from ``target_strike`` (rounded to ``interval``) and
    returns the first that has premiums available at both entry and exit times.
    """
    base = int(round(target_strike / interval) * interval)
    ddmmyy = expiry.strftime("%d%m%y")
    for n in range(0, 9):  # search +/- up to 8 intervals out
        for strike in dict.fromkeys([base + n * interval, base - n * interval]):
            symbol = f"{option_type.value}-{underlying}-{strike}-{ddmmyy}"
            candles = _fetch_option_candles(client, symbol, win_start, win_end, resolution)
            if not candles:
                continue
            if _premium_at(candles, entry_ts, step) is not None and \
               _premium_at(candles, exit_ts, step) is not None:
                return symbol, strike, candles
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Option-price backtest (strategy 1)")
    ap.add_argument("--days", type=int, default=3, help="Look-back window in days (default 3)")
    ap.add_argument("--resolution", default=None, help="Candle resolution (default from .env)")
    ap.add_argument("--warmup-days", type=int, default=4, help="Extra prior days for indicator/level warmup")
    ap.add_argument("--excel", default="option_backtest_IST.xlsx", help="Output .xlsx path")
    ap.add_argument(
        "--mode", choices=["short", "long", "both"], default="both",
        help="short = SELL ITM PUT(buy)/CALL(sell) [current bot]; "
             "long = BUY ITM CALL on BUY signal, BUY ITM PUT on SELL signal; "
             "both = both strategies in one file (default)",
    )
    ap.add_argument("--offset", type=int, default=None, help="ITM depth in points (default from .env option_offset)")
    ap.add_argument("--lots", type=int, default=None, help="option lots (default from .env option_contracts)")
    ap.add_argument("--target-premium", type=float, default=None,
                    help="pick the strike whose entry premium is closest to this (overrides --offset)")
    ap.add_argument("--take-profit", type=float, default=None,
                    help="take-profit %% of premium for long options (e.g. 100 = exit at 2x entry)")
    args = ap.parse_args()

    settings = load_settings()
    if args.offset is not None:
        settings.option_offset = args.offset
    if args.lots is not None:
        settings.option_contracts = args.lots
    setup_logging("WARNING")  # quiet the download/info logs; we print our own report
    resolution = args.resolution or settings.resolution
    step = _RES_SECONDS.get(resolution, 300)
    symbol = settings.symbol

    now = int(time.time())
    win_start = now - args.days * 86400
    dl_start = win_start - args.warmup_days * 86400

    # --- 1. BTC candles + strategy signals (warmup included) ---
    df = download(symbol=symbol, start=dl_start, end=now, resolution=resolution,
                  base_url=settings.rest_base_url)
    candles = df_to_candles(df)
    engine = BacktestEngine(
        period=settings.atr_period, multiplier=settings.st_multiplier,
        contracts=settings.contracts, contract_value=_LOT_BTC, settings=settings,
    )
    trips = engine.run(candles).trips
    window_trips = [t for t in trips if t.entry_time >= win_start]

    print(f"\nUnderlying {symbol}  resolution {resolution}  "
          f"window {_ist(win_start)} -> {_ist(now)}")
    print(f"Signals in window: {len(window_trips)}  "
          f"(offset {settings.option_offset}, lots {settings.option_contracts})")

    if not window_trips:
        print("No signals fired in this window — nothing to price.")
        return

    underlying = symbol.replace("USDT", "").replace("USD", "")
    lots = settings.option_contracts
    modes = ["short", "long"] if args.mode == "both" else [args.mode]

    results: dict[str, list[dict]] = {}
    stats: dict[str, dict] = {}
    with httpx.Client(base_url=settings.rest_base_url, timeout=30.0) as client:
        for mode in modes:
            recs = _price_trips(client, window_trips, mode, settings,
                                underlying, lots, resolution, step,
                                target_premium=args.target_premium, cache={},
                                take_profit_pct=args.take_profit)
            results[mode] = recs
            priced = sum(1 for r in recs if r["pnl_usd"] is not None)
            wins = sum(1 for r in recs if (r["pnl_usd"] or 0) > 0)
            net = sum(r["pnl_usd"] or 0 for r in recs)
            gross = sum(r["pnl_gross_usd"] or 0 for r in recs)
            fees = sum(r["fee_usd"] or 0 for r in recs)
            stats[mode] = {"priced": priced, "wins": wins, "losses": priced - wins,
                           "net": net, "gross": gross, "fees": fees,
                           "win_rate": (wins / priced * 100.0) if priced else 0.0}

    # --- 3. Console report (per mode) ---
    for mode in modes:
        _print_mode(mode, results[mode], stats[mode], lots)

    # --- 4. Single Excel file (IST times) ---
    sheet_name = {"short": "Sell_ITM", "long": "Buy_ITM"}
    out_path = Path(args.excel)
    summary_rows = [
        {"metric": "window_IST", "value": f"{_ist(win_start)} -> {_ist(now)}"},
        {"metric": "timezone", "value": "Asia/Kolkata (IST)"},
        {"metric": "underlying", "value": symbol},
        {"metric": "resolution", "value": resolution},
        {"metric": "option_offset", "value": settings.option_offset},
        {"metric": "lots", "value": lots},
    ]
    for mode in modes:
        label = "Sell_ITM" if mode == "short" else "Buy_ITM"
        s = stats[mode]
        summary_rows += [
            {"metric": f"[{label}] priced_trades", "value": f"{s['priced']}/{len(results[mode])}"},
            {"metric": f"[{label}] wins/losses", "value": f"{s['wins']}/{s['losses']}"},
            {"metric": f"[{label}] win_rate_pct", "value": round(s["win_rate"], 1)},
            {"metric": f"[{label}] gross_pnl_usd", "value": round(s["gross"], 3)},
            {"metric": f"[{label}] est_fees_usd", "value": round(s["fees"], 3)},
            {"metric": f"[{label}] net_pnl_usd", "value": round(s["net"], 3)},
        ]
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        pd.DataFrame(summary_rows).to_excel(xl, sheet_name="Summary", index=False)
        for mode in modes:
            pd.DataFrame(results[mode]).to_excel(xl, sheet_name=sheet_name[mode], index=False)
    print(f"\nSingle Excel written to {out_path.resolve()}")


def _resolve_by_premium(
    client, underlying, otype, entry_btc, expiry, interval, target_premium,
    entry_ts, exit_ts, win_start, win_end, resolution, step, cache,
):
    """Pick the strike whose ENTRY premium is closest to ``target_premium``.

    Scans from ATM outward toward OTM (premium falls) until it brackets the
    target, plus a few ITM steps in case the ATM premium is already below it.
    Returns ``(symbol, strike, candles)`` or None.
    """
    ddmmyy = expiry.strftime("%d%m%y")
    atm = int(round(entry_btc / interval) * interval)
    otm = interval if otype == OptionType.CALL else -interval  # toward OTM = cheaper

    def ev(strike):
        symbol = f"{otype.value}-{underlying}-{strike}-{ddmmyy}"
        if symbol not in cache:
            cache[symbol] = _fetch_option_candles(client, symbol, win_start, win_end, resolution)
        candles = cache[symbol]
        if not candles:
            return None
        ein = _premium_at(candles, entry_ts, step)
        eout = _premium_at(candles, exit_ts, step)
        return (symbol, strike, candles, ein) if (ein is not None and eout is not None) else None

    best = None  # (abs_diff, symbol, strike, candles)
    strike = atm
    for _ in range(30):                 # OTM scan, premium decreasing
        r = ev(strike)
        if r is not None:
            diff = abs(r[3] - target_premium)
            if best is None or diff < best[0]:
                best = (diff, r[0], r[1], r[2])
            if r[3] <= target_premium:  # bracketed the target
                break
        strike += otm
    strike = atm - otm
    for _ in range(6):                  # a few ITM steps (premium increasing)
        r = ev(strike)
        if r is not None:
            diff = abs(r[3] - target_premium)
            if best is None or diff < best[0]:
                best = (diff, r[0], r[1], r[2])
        strike -= otm
    return (best[1], best[2], best[3]) if best else None


def _price_trips(client, window_trips, mode, settings, underlying, lots, resolution, step,
                 target_premium=None, cache=None, take_profit_pct=None):
    """Price every trip for one option strategy ``mode`` and return record dicts."""
    off = settings.option_offset
    cache = cache if cache is not None else {}
    records: list[dict] = []
    for t in window_trips:
        is_buy = t.direction == SignalDir.LONG.value
        if mode == "short":
            # Current bot: sell ITM PUT on BUY, sell ITM CALL on SELL.
            otype = OptionType.PUT if is_buy else OptionType.CALL
            target = (t.entry_price + off) if is_buy else (t.entry_price - off)
            action = "SELL"
        else:
            # Long ITM: BUY signal -> buy ITM CALL; SELL signal -> buy ITM PUT.
            otype = OptionType.CALL if is_buy else OptionType.PUT
            target = (t.entry_price - off) if is_buy else (t.entry_price + off)
            action = "BUY"
        expiry = _select_expiry_date(t.entry_time, settings.option_expiry_cutoff_hour)
        o_start = t.entry_time - 2 * 86400
        o_end = t.exit_time + 2 * 86400
        if target_premium is not None:
            resolved = _resolve_by_premium(
                client, underlying, otype, t.entry_price, expiry,
                settings.option_strike_interval, target_premium,
                t.entry_time, t.exit_time, o_start, o_end, resolution, step, cache,
            )
        else:
            resolved = _resolve_contract(
                client, underlying, otype, int(target), expiry,
                settings.option_strike_interval, t.entry_time, t.exit_time,
                o_start, o_end, resolution, step,
            )
        sym = None
        strike = int(round(target / settings.option_strike_interval) * settings.option_strike_interval)
        entry_prem = exit_prem = pnl_gross = fee = pnl_net = None
        exit_time_eff = t.exit_time
        tp_hit = False
        if resolved is not None:
            sym, strike, ocandles = resolved
            entry_prem = _premium_at(ocandles, t.entry_time, step)
            exit_prem = _premium_at(ocandles, t.exit_time, step)
            # Take-profit (long only): exit when the option premium first reaches
            # entry*(1+tp/100) intrabar, before the strategy's SL/EOD exit.
            if take_profit_pct is not None and mode != "short" and entry_prem:
                tp_price = entry_prem * (1.0 + take_profit_pct / 100.0)
                for ts in sorted(k for k in ocandles if t.entry_time < k <= t.exit_time):
                    if ocandles[ts].high >= tp_price:
                        exit_prem, exit_time_eff, tp_hit = tp_price, ts, True
                        break
            # Short profits when premium falls; long profits when it rises.
            pnl_gross = ((entry_prem - exit_prem) if mode == "short" else (exit_prem - entry_prem)) * lots * _LOT_BTC
            fee = _side_fee(t.entry_price, entry_prem, lots) + _side_fee(t.exit_price, exit_prem, lots)
            pnl_net = pnl_gross - fee
        records.append({
            "signal": "BUY" if is_buy else "SELL",
            "action": action,
            "option_type": otype.value,
            "contract": sym or f"{otype.value}-{underlying}-{strike}-{expiry:%d%m%y}",
            "strike": strike,
            "entry_time_ist": _ist(t.entry_time),
            "exit_time_ist": _ist(exit_time_eff),
            "exit_reason": "TP" if tp_hit else "SL/EOD",
            "btc_entry_price": round(t.entry_price, 1),
            "btc_exit_price": round(t.exit_price, 1),
            "option_entry_price": round(entry_prem, 1) if entry_prem is not None else None,
            "option_exit_price": round(exit_prem, 1) if exit_prem is not None else None,
            "lots": lots,
            "pnl_gross_usd": round(pnl_gross, 3) if pnl_gross is not None else None,
            "fee_usd": round(fee, 3) if fee is not None else None,
            "pnl_usd": round(pnl_net, 3) if pnl_net is not None else None,
        })
    return records


def _print_mode(mode: str, records: list[dict], s: dict, lots: int) -> None:
    label = "SELL ITM (current bot)" if mode == "short" else "BUY ITM (long)"
    print("\n" + "=" * 104)
    print(f"  {label}")
    print(f"{'Entry(IST)':<12} {'Exit(IST)':<12} {'Sig':<4} {'Contract':<22} "
          f"{'BTCin':>8} {'PremIn':>8} {'PremOut':>8} {'PnL$':>9}")
    print("-" * 104)
    for r in records:
        ein, eout, pnl = r["option_entry_price"], r["option_exit_price"], r["pnl_usd"]
        ein_s = f"{ein:>8.1f}" if ein is not None else f"{'n/a':>8}"
        eout_s = f"{eout:>8.1f}" if eout is not None else f"{'n/a':>8}"
        pnl_s = f"{pnl:>9.3f}" if pnl is not None else f"{'NO DATA':>9}"
        print(f"{r['entry_time_ist'][5:16]:<12} {r['exit_time_ist'][5:16]:<12} "
              f"{r['signal']:<4} {r['contract']:<22} {r['btc_entry_price']:>8.0f} "
              f"{ein_s} {eout_s} {pnl_s}")
    print("-" * 104)
    print(f"Priced {s['priced']}/{len(records)}  Wins/Losses {s['wins']}/{s['losses']}  "
          f"Win rate {s['win_rate']:.1f}%  (PnL$ column = NET after fees)")
    print(f"Gross PnL {s['gross']:,.2f}  -  est. fees {s['fees']:,.2f}  =  "
          f"NET {s['net']:,.2f} USD")


if __name__ == "__main__":
    main()
