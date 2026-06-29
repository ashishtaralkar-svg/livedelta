"""Shared helpers to reprice strategy signals with real Delta option premiums.

Each strategy produces (entry_time, exit_time, direction, btc prices) trades;
these helpers map a trade to an option contract, fetch its real history, read the
premium at a time, and model the Delta taker fee (brokerage).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from ..enums import OptionType
from ..models import Candle

_IST = ZoneInfo("Asia/Kolkata")
LOT_BTC = 0.001  # 1 option lot = 0.001 BTC underlying on Delta BTC options
RES_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}

# Delta options taker fee (brokerage): 0.03% of underlying notional, capped at
# 10% of premium, charged per side. Excludes bid/ask spread.
TAKER_FEE_PCT = 0.0003
PREMIUM_FEE_CAP_PCT = 0.10


def side_fee(underlying_px: float, premium: float, lots: int) -> float:
    """Per-side option brokerage in USD for ``lots`` lots."""
    per_lot = min(TAKER_FEE_PCT * underlying_px, PREMIUM_FEE_CAP_PCT * premium) * LOT_BTC
    return per_lot * lots


def select_expiry_date(entry_ts: int, cutoff_hour: int) -> datetime:
    """Same-day daily expiry in IST, rolling to next day past the cutoff hour."""
    ist = datetime.fromtimestamp(entry_ts, tz=_IST)
    if ist.hour >= cutoff_hour:
        ist = ist + timedelta(days=1)
    return ist


def fetch_option_candles(
    client: httpx.Client, symbol: str, start: int, end: int, resolution: str
) -> dict[int, Candle]:
    """Fetch one option's history as a ``{start_time: Candle}`` map (empty if none).

    Retries transient network errors (Delta's history API occasionally read-times
    out) so a single blip doesn't abort a long backtest.
    """
    params = {"symbol": symbol, "resolution": resolution, "start": start, "end": end}
    for attempt in range(4):
        try:
            resp = client.get("/v2/history/candles", params=params)
            resp.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            # 4xx (except 429 rate-limit) is not transient — this contract/range is
            # not queryable, so skip it rather than aborting a long backtest.
            sc = exc.response.status_code
            if 400 <= sc < 500 and sc != 429:
                return {}
            if attempt == 3:
                raise
            time.sleep(1.0 + attempt)
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == 3:
                raise
            time.sleep(1.0 + attempt)
    rows = resp.json().get("result") or []
    out: dict[int, Candle] = {}
    for r in rows:
        c = Candle.from_rest(r)
        if c.close > 0:  # skip expired-contract 0.0 placeholders
            out[c.start_time] = c
    return out


def premium_at(candles: dict[int, Candle], ts: int, step: int) -> float | None:
    """Option close at/just-before ``ts`` (search back a few bars for a gap)."""
    for k in range(0, 12):
        c = candles.get(ts - k * step)
        if c is not None:
            return c.close
    earlier = [c for t, c in candles.items() if t <= ts]
    return max(earlier, key=lambda c: c.start_time).close if earlier else None


def resolve_contract(
    client: httpx.Client, underlying: str, option_type: OptionType, target_strike: int,
    expiry: datetime, interval: int, entry_ts: int, exit_ts: int,
    win_start: int, win_end: int, resolution: str, step: int,
) -> tuple[str, int, dict[int, Candle]] | None:
    """Snap to the nearest traded strike near ``target_strike`` with premiums at
    both entry and exit. Returns ``(symbol, strike, candles)`` or None."""
    base = int(round(target_strike / interval) * interval)
    ddmmyy = expiry.strftime("%d%m%y")
    for n in range(0, 9):
        for strike in dict.fromkeys([base + n * interval, base - n * interval]):
            symbol = f"{option_type.value}-{underlying}-{strike}-{ddmmyy}"
            candles = fetch_option_candles(client, symbol, win_start, win_end, resolution)
            if candles and premium_at(candles, entry_ts, step) is not None \
               and premium_at(candles, exit_ts, step) is not None:
                return symbol, strike, candles
    return None


def resolve_by_premium(
    client: httpx.Client, underlying: str, option_type: OptionType, entry_btc: float,
    expiry: datetime, interval: int, target_premium: float,
    entry_ts: int, exit_ts: int, win_start: int, win_end: int,
    resolution: str, step: int, cache: dict,
) -> tuple[str, int, dict[int, Candle]] | None:
    """Pick the strike whose ENTRY premium is closest to ``target_premium``.

    Scans from ATM toward OTM (premium falls) until it brackets the target, plus a
    few ITM steps. Returns ``(symbol, strike, candles)`` or None. ``cache`` keyed by
    symbol avoids refetching across trades.
    """
    ddmmyy = expiry.strftime("%d%m%y")
    atm = int(round(entry_btc / interval) * interval)
    otm = interval if option_type == OptionType.CALL else -interval  # toward OTM = cheaper

    def ev(strike: int):
        symbol = f"{option_type.value}-{underlying}-{strike}-{ddmmyy}"
        if symbol not in cache:
            cache[symbol] = fetch_option_candles(client, symbol, win_start, win_end, resolution)
        candles = cache[symbol]
        if not candles:
            return None
        ein = premium_at(candles, entry_ts, step)
        eout = premium_at(candles, exit_ts, step)
        return (symbol, strike, candles, ein) if (ein is not None and eout is not None) else None

    best = None  # (abs_diff, symbol, strike, candles)
    strike = atm
    for _ in range(30):
        r = ev(strike)
        if r is not None:
            diff = abs(r[3] - target_premium)
            if best is None or diff < best[0]:
                best = (diff, r[0], r[1], r[2])
            if r[3] <= target_premium:
                break
        strike += otm
    strike = atm - otm
    for _ in range(6):
        r = ev(strike)
        if r is not None:
            diff = abs(r[3] - target_premium)
            if best is None or diff < best[0]:
                best = (diff, r[0], r[1], r[2])
        strike -= otm
    return (best[1], best[2], best[3]) if best else None
