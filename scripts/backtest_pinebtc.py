"""pinebtc — Python port of the "1 ashish final buy sell" Pine v5 strategy.

Directional BTC (points P&L, not options). Logic mirrors the Pine exactly:
  * EMAs 10/20/200 on close.
  * Trading session 17:30 IST -> 17:25 IST next day (dead window 17:26-17:29).
  * Long setup:  prev red, cur green, close>ema10, ema10>ema20, close>ema200.
    Short setup: mirror.
  * Arm: entry = max(high,high[1]) long / min(low,low[1]) short; SL = opposite extreme.
  * Trigger on break of entry level; invalidate if SL touched before trigger.
  * Manage: hard SL, trail SL up to EMA200 once price clears it, exit on close<EMA200
    (long) / close>EMA200 (short). Auto-exit at 17:25.

Run: python scripts/backtest_pinebtc.py --days 4 [--resolution 5m]
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from deltabot.backtest.data_loader import df_to_candles, download
from deltabot.config import load_settings

_IST = ZoneInfo("Asia/Kolkata")


def _ist(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=_IST)


def ema(values: list[float], length: int) -> list[float]:
    """Pine ta.ema: seed with first value, alpha = 2/(len+1)."""
    a = 2.0 / (length + 1.0)
    out: list[float] = []
    prev = values[0]
    for i, v in enumerate(values):
        prev = v if i == 0 else a * v + (1 - a) * prev
        out.append(prev)
    return out


def run(candles, days: float):
    o = [c.open for c in candles]
    h = [c.high for c in candles]
    lo = [c.low for c in candles]
    cl = [c.close for c in candles]
    ema10 = ema(cl, 10)
    ema20 = ema(cl, 20)
    ema200 = ema(cl, 200)

    def in_time(ts: int) -> bool:
        t = _ist(ts)
        after = (t.hour > 17) or (t.hour == 17 and t.minute >= 30)
        before = (t.hour < 17) or (t.hour == 17 and t.minute <= 25)
        return after or before

    win_start = int(time.time()) - int(days * 86400)

    trade_active = False
    entry_trig = False
    entry_level = sl_level = 0.0
    direction = ""
    entry_ts = 0
    trips: list[dict] = []

    def record(exit_price: float, exit_ts: int, reason: str):
        nonlocal trade_active, entry_trig
        pnl = (exit_price - entry_level) if direction == "LONG" else (entry_level - exit_price)
        trips.append({
            "direction": direction,
            "entry_ist": _ist(entry_ts).strftime("%Y-%m-%d %H:%M IST"),
            "exit_ist": _ist(exit_ts).strftime("%Y-%m-%d %H:%M IST"),
            "entry": round(entry_level, 1),
            "exit": round(exit_price, 1),
            "reason": reason,
            "pnl_pts": round(pnl, 1),
        })
        trade_active = False
        entry_trig = False

    for i in range(1, len(candles)):
        ts = candles[i].start_time
        it = in_time(ts)
        it_prev = in_time(candles[i - 1].start_time)
        new_session = it and not it_prev
        t = _ist(ts)

        # --- Setup detection / arming (skip on the session's first bar) ---
        if not trade_active and it and not new_session:
            bull = (cl[i - 1] < o[i - 1] and cl[i] > o[i] and cl[i] > ema10[i]
                    and ema10[i] > ema20[i] and cl[i] > ema200[i])
            bear = (cl[i - 1] > o[i - 1] and cl[i] < o[i] and cl[i] < ema10[i]
                    and ema10[i] < ema20[i] and cl[i] < ema200[i])
            if bull:
                trade_active, entry_trig = True, False
                entry_level = max(h[i], h[i - 1])
                sl_level = min(lo[i], lo[i - 1])
                direction = "LONG"
            elif bear:
                trade_active, entry_trig = True, False
                entry_level = min(lo[i], lo[i - 1])
                sl_level = max(h[i], h[i - 1])
                direction = "SHORT"

        # --- Entry trigger / invalidation ---
        if trade_active and not entry_trig and it:
            if direction == "LONG" and h[i] > entry_level:
                entry_trig = True
                entry_ts = ts
            elif direction == "SHORT" and lo[i] < entry_level:
                entry_trig = True
                entry_ts = ts
            elif (direction == "LONG" and lo[i] < sl_level) or \
                 (direction == "SHORT" and h[i] > sl_level):
                trade_active = False  # invalidated before entry

        # --- Trade management (same-bar after trigger, matching Pine) ---
        if trade_active and entry_trig:
            if direction == "LONG":
                if lo[i] <= sl_level:
                    record(sl_level, ts, "SL")
                elif sl_level < ema200[i] and cl[i] > ema200[i]:
                    sl_level = max(sl_level, ema200[i])
                elif cl[i] < ema200[i]:
                    record(cl[i], ts, "EMA200")
            else:
                if h[i] >= sl_level:
                    record(sl_level, ts, "SL")
                elif sl_level > ema200[i] and cl[i] < ema200[i]:
                    sl_level = min(sl_level, ema200[i])
                elif cl[i] > ema200[i]:
                    record(cl[i], ts, "EMA200")

        # --- Auto exit at 17:25 ---
        if t.hour == 17 and t.minute == 25 and trade_active:
            if entry_trig:
                record(cl[i], ts, "EOD")
            else:
                trade_active = False
                entry_trig = False

    return [t for t in trips if int(datetime.strptime(
        t["entry_ist"].replace(" IST", ""), "%Y-%m-%d %H:%M")
        .replace(tzinfo=_IST).timestamp()) >= win_start]


def main() -> None:
    ap = argparse.ArgumentParser(description="pinebtc directional backtest")
    ap.add_argument("--days", type=float, default=4)
    ap.add_argument("--resolution", default="5m")
    ap.add_argument("--warmup-days", type=int, default=3)
    ap.add_argument("--excel", default="pinebtc_backtest.xlsx")
    args = ap.parse_args()

    s = load_settings()
    now = int(time.time())
    win_start = now - int(args.days * 86400)
    df = download(symbol=s.symbol, start=win_start - args.warmup_days * 86400,
                  end=now, resolution=args.resolution, base_url=s.rest_base_url)
    candles = df_to_candles(df)
    trips = run(candles, args.days)

    print(f"\n{s.symbol}  {args.resolution}  pinebtc (EMA10/20/200 breakout, EMA200 trail)")
    print(f"Window {_ist(win_start):%Y-%m-%d %H:%M} -> {_ist(now):%Y-%m-%d %H:%M} IST")
    print("=" * 92)
    print(f"{'Entry(IST)':<20}{'Exit(IST)':<20}{'Dir':<6}{'Entry':>9}{'Exit':>9}{'Why':>8}{'Pts':>9}")
    print("-" * 92)
    total = 0.0
    wins = 0
    for t in trips:
        print(f"{t['entry_ist'][:16]:<20}{t['exit_ist'][:16]:<20}{t['direction']:<6}"
              f"{t['entry']:>9.1f}{t['exit']:>9.1f}{t['reason']:>8}{t['pnl_pts']:>9.1f}")
        total += t["pnl_pts"]
        wins += 1 if t["pnl_pts"] > 0 else 0
    print("=" * 92)
    n = len(trips)
    wr = (wins / n * 100) if n else 0
    reasons = {r: sum(1 for t in trips if t["reason"] == r) for r in ("SL", "EMA200", "EOD")}
    print(f"Trades {n}  Wins/Losses {wins}/{n - wins}  Win rate {wr:.1f}%  "
          f"exits: SL={reasons['SL']} EMA200={reasons['EMA200']} EOD={reasons['EOD']}")
    print(f"NET {total:,.1f} BTC points")

    with pd.ExcelWriter(Path(args.excel), engine="openpyxl") as xl:
        pd.DataFrame(trips).to_excel(xl, sheet_name="Trades", index=False)
        pd.DataFrame([
            {"metric": "window_IST", "value": f"{_ist(win_start):%Y-%m-%d %H:%M} -> {_ist(now):%Y-%m-%d %H:%M}"},
            {"metric": "resolution", "value": args.resolution},
            {"metric": "trades", "value": n},
            {"metric": "win_rate_pct", "value": round(wr, 1)},
            {"metric": "exits_SL/EMA200/EOD", "value": f"{reasons['SL']}/{reasons['EMA200']}/{reasons['EOD']}"},
            {"metric": "net_points", "value": round(total, 1)},
        ]).to_excel(xl, sheet_name="Summary", index=False)
    print(f"\nExcel written to {Path(args.excel).resolve()}")


if __name__ == "__main__":
    main()
