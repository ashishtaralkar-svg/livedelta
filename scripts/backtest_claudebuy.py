"""claudebuy — directional BTC strategy (raw points P&L).

Rules (as specified):

SESSION / TIME
  * Overnight session: starts 17:30, ends 17:25 next day (window wraps midnight).
  * New session detected on the first bar at/after 17:30.
  * No setup on the new-session bar itself.
  * Force-close any open trade at 17:25.

INDICATORS
  * EMA 10, EMA 20, EMA 200 on close.

LONG SETUP (session only, all true):
  prev bearish (close[1]<open[1]), cur bullish (close>open),
  close>EMA10, EMA10>EMA20, close>EMA200.
SHORT SETUP: mirror.

ENTRY (pending stop after a setup):
  LONG  entry = max(high,high[1]); triggers when price trades above it; SL = min(low,low[1]).
  SHORT entry = min(low,low[1]);  triggers when price trades below it; SL = max(high,high[1]).
  If SL is hit BEFORE entry triggers -> cancel setup ("Invalid").
  Only one trade active at a time.

MANAGEMENT (after entry):
  LONG : exit at SL; trail SL up to EMA200 once slLevel<EMA200 and close>EMA200;
         exit if a candle closes below EMA200. SHORT mirrors.
  Auto-exit at 17:25.

P&L: longs = exit-entry, shorts = entry-exit (raw points).

Run: python scripts/backtest_claudebuy.py [--days 4] [--resolution 5m]
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

SESSION_START = (17, 30)   # hh, mm  -> session opens
SESSION_END = (17, 25)     # hh, mm  -> session closes / auto-exit


def _ist(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=_IST)


def ema(values: list[float], length: int) -> list[float]:
    """EMA seeded with the first value, alpha = 2/(len+1)."""
    a = 2.0 / (length + 1.0)
    out: list[float] = []
    prev = values[0]
    for i, v in enumerate(values):
        prev = v if i == 0 else a * v + (1 - a) * prev
        out.append(prev)
    return out


def in_session(ts: int) -> bool:
    """True during the overnight window 17:30 -> 17:25 next day."""
    t = _ist(ts)
    after_start = (t.hour > SESSION_START[0]) or (t.hour == SESSION_START[0] and t.minute >= SESSION_START[1])
    before_end = (t.hour < SESSION_END[0]) or (t.hour == SESSION_END[0] and t.minute <= SESSION_END[1])
    return after_start or before_end


def run(candles, days: float):
    o = [c.open for c in candles]
    h = [c.high for c in candles]
    lo = [c.low for c in candles]
    cl = [c.close for c in candles]
    ema10, ema20, ema200 = ema(cl, 10), ema(cl, 20), ema(cl, 200)

    win_start = int(time.time()) - int(days * 86400)

    active = False        # a setup is armed or a trade is live
    triggered = False     # entry has fired
    entry_level = sl_level = 0.0
    direction = ""
    entry_ts = 0
    trips: list[dict] = []

    def record(exit_price: float, exit_ts: int, reason: str):
        nonlocal active, triggered
        pnl = (exit_price - entry_level) if direction == "LONG" else (entry_level - exit_price)
        trips.append({
            "direction": direction,
            "entry_ist": _ist(entry_ts).strftime("%Y-%m-%d %H:%M IST"),
            "exit_ist": _ist(exit_ts).strftime("%Y-%m-%d %H:%M IST"),
            "entry": round(entry_level, 1), "exit": round(exit_price, 1),
            "reason": reason, "pnl_pts": round(pnl, 1),
        })
        active = triggered = False

    for i in range(1, len(candles)):
        ts = candles[i].start_time
        it, it_prev = in_session(ts), in_session(candles[i - 1].start_time)
        new_session = it and not it_prev
        t = _ist(ts)

        # 1. Setup detection / arming (never on the new-session bar).
        if not active and it and not new_session:
            bull = (cl[i - 1] < o[i - 1] and cl[i] > o[i] and cl[i] > ema10[i]
                    and ema10[i] > ema20[i] and cl[i] > ema200[i])
            bear = (cl[i - 1] > o[i - 1] and cl[i] < o[i] and cl[i] < ema10[i]
                    and ema10[i] < ema20[i] and cl[i] < ema200[i])
            if bull:
                active, triggered, direction = True, False, "LONG"
                entry_level, sl_level = max(h[i], h[i - 1]), min(lo[i], lo[i - 1])
            elif bear:
                active, triggered, direction = True, False, "SHORT"
                entry_level, sl_level = min(lo[i], lo[i - 1]), max(h[i], h[i - 1])

        # 2. Entry trigger / pre-entry invalidation.
        if active and not triggered and it:
            if direction == "LONG" and h[i] > entry_level:
                triggered, entry_ts = True, ts
            elif direction == "SHORT" and lo[i] < entry_level:
                triggered, entry_ts = True, ts
            elif (direction == "LONG" and lo[i] < sl_level) or \
                 (direction == "SHORT" and h[i] > sl_level):
                active = False  # SL hit before entry -> Invalid

        # 3. Trade management (same bar after trigger).
        if active and triggered:
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

        # 4. Auto-exit at 17:25.
        if t.hour == SESSION_END[0] and t.minute == SESSION_END[1] and active:
            if triggered:
                record(cl[i], ts, "EOD")
            else:
                active = triggered = False

    return [t for t in trips if int(datetime.strptime(
        t["entry_ist"].replace(" IST", ""), "%Y-%m-%d %H:%M")
        .replace(tzinfo=_IST).timestamp()) >= win_start]


def main() -> None:
    ap = argparse.ArgumentParser(description="claudebuy directional backtest")
    ap.add_argument("--days", type=float, default=4)
    ap.add_argument("--resolution", default="5m")
    ap.add_argument("--warmup-days", type=int, default=3)
    ap.add_argument("--excel", default="claudebuy_backtest.xlsx")
    args = ap.parse_args()

    s = load_settings()
    now = int(time.time())
    win_start = now - int(args.days * 86400)
    df = download(symbol=s.symbol, start=win_start - args.warmup_days * 86400,
                  end=now, resolution=args.resolution, base_url=s.rest_base_url)
    candles = df_to_candles(df)
    trips = run(candles, args.days)

    print(f"\n{s.symbol}  {args.resolution}  claudebuy (EMA10/20/200 breakout, EMA200 trail)")
    print(f"Window {_ist(win_start):%Y-%m-%d %H:%M} -> {_ist(now):%Y-%m-%d %H:%M} IST")
    print("=" * 92)
    print(f"{'Entry(IST)':<20}{'Exit(IST)':<20}{'Dir':<6}{'Entry':>9}{'Exit':>9}{'Why':>8}{'Pts':>9}")
    print("-" * 92)
    total = wins = 0.0
    for t in trips:
        print(f"{t['entry_ist'][:16]:<20}{t['exit_ist'][:16]:<20}{t['direction']:<6}"
              f"{t['entry']:>9.1f}{t['exit']:>9.1f}{t['reason']:>8}{t['pnl_pts']:>9.1f}")
        total += t["pnl_pts"]
        wins += 1 if t["pnl_pts"] > 0 else 0
    print("=" * 92)
    n = len(trips)
    wr = (wins / n * 100) if n else 0
    reasons = {r: sum(1 for t in trips if t["reason"] == r) for r in ("SL", "EMA200", "EOD")}
    print(f"Trades {n}  Wins/Losses {int(wins)}/{n - int(wins)}  Win rate {wr:.1f}%  "
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
