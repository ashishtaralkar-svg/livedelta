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
# Delta option contract sizes (underlying per lot). BTC = 0.001, ETH = 0.01.
LOT_SIZE = {"BTC": 0.001, "ETH": 0.01}
RES_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}

# Delta options taker fee (brokerage), calibrated to actual account fills
# (2026-07 order log: ~$0.37 per 50-lot leg ≈ 0.012% of underlying notional),
# capped at 10% of premium, charged per side. Excludes bid/ask spread.
TAKER_FEE_PCT = 0.00012
PREMIUM_FEE_CAP_PCT = 0.10


def side_fee(underlying_px: float, premium: float, lots: int, lot_size: float = LOT_BTC) -> float:
    """Per-side option brokerage in USD for ``lots`` lots (``lot_size`` = the
    underlying per lot: 0.001 for BTC, 0.01 for ETH)."""
    per_lot = min(TAKER_FEE_PCT * underlying_px, PREMIUM_FEE_CAP_PCT * premium) * lot_size
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
    # Transient network/DNS blips (getaddrinfo failed, read timeouts) can last
    # many seconds; retry generously with capped backoff so one hiccup never
    # aborts a multi-minute backtest. Total retry budget ~90s before giving up.
    attempts = 8
    backoff = [1, 2, 4, 8, 15, 20, 30]
    resp = None
    for attempt in range(attempts):
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
            if attempt == attempts - 1:
                raise
            time.sleep(backoff[min(attempt, len(backoff) - 1)])
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == attempts - 1:
                # Give up on THIS contract instead of killing the whole run --
                # the caller treats an empty result as "couldn't price, skip".
                print(f"NOTE: skipping {symbol} after {attempts} network failures ({exc!r})")
                return {}
            time.sleep(backoff[min(attempt, len(backoff) - 1)])
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


def option_low_over(candles: dict[int, Candle], ts: int, window_sec: int,
                    step: int) -> float | None:
    """Lowest option ``.low`` over the window ``[ts, ts + window_sec)`` -- the
    sold option's worst-case fill when BTC hits the trigger somewhere inside
    the BTC candle. Falls back to the nearest prior candle's ``.low`` (same
    back-search as ``premium_at``), then None."""
    lows = [c.low for t, c in candles.items() if ts <= t < ts + window_sec]
    if lows:
        return min(lows)
    for k in range(0, 12):
        c = candles.get(ts - k * step)
        if c is not None:
            return c.low
    earlier = [c for t, c in candles.items() if t <= ts]
    return max(earlier, key=lambda c: c.start_time).low if earlier else None


def intrinsic_value(symbol: str, underlying_px: float) -> float:
    """Intrinsic value of an option = its MINIMUM possible price (an option can
    never trade below what it's worth if exercised). ``symbol`` like
    ``P-BTC-65200-120726`` / ``C-BTC-63000-...``. CALL: max(0, px-strike);
    PUT: max(0, strike-px). Used to floor illiquid-ITM historical candle prices,
    which frequently print BELOW intrinsic (impossible) and inflate backtests."""
    try:
        parts = symbol.split("-")
        otype, strike = parts[0], float(parts[-2])
    except (IndexError, ValueError, TypeError, AttributeError):
        return 0.0
    if otype == "C":
        return max(0.0, underlying_px - strike)
    if otype == "P":
        return max(0.0, strike - underlying_px)
    return 0.0


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


def resolve_atm(
    client: httpx.Client, underlying: str, option_type: OptionType, entry_btc: float,
    expiry: datetime, interval: int,
    entry_ts: int, exit_ts: int, win_start: int, win_end: int,
    resolution: str, step: int, cache: dict,
    offset: float = 0.0,
) -> tuple[str, int, dict[int, Candle]] | None:
    """Pick the strike ``offset`` away from AT-THE-MONEY (nearest ``interval`` to
    ``entry_btc``, then ``+ offset``) directly -- no premium target/search, unlike
    ``resolve_by_premium``. ``offset`` is applied literally (same sign for calls and
    puts) -- e.g. offset=100 always selects atm+100, whichever side is traded.
    Returns ``(symbol, strike, candles)`` or None if that exact strike has no
    premium data at both entry and exit. ``cache`` keyed by symbol avoids refetching
    across trades.
    """
    ddmmyy = expiry.strftime("%d%m%y")
    strike = int(round(entry_btc / interval) * interval + offset)
    symbol = f"{option_type.value}-{underlying}-{strike}-{ddmmyy}"
    if symbol not in cache:
        cache[symbol] = fetch_option_candles(client, symbol, win_start, win_end, resolution)
    candles = cache[symbol]
    if not candles or premium_at(candles, entry_ts, step) is None \
       or premium_at(candles, exit_ts, step) is None:
        return None
    return symbol, strike, candles


def resolve_by_premium(
    client: httpx.Client, underlying: str, option_type: OptionType, entry_btc: float,
    expiry: datetime, interval: int, target_premium: float,
    entry_ts: int, exit_ts: int, win_start: int, win_end: int,
    resolution: str, step: int, cache: dict,
) -> tuple[str, int, dict[int, Candle]] | None:
    """Pick the strike whose ENTRY premium is closest to ``target_premium`` --
    mirrors ``OptionsExecutor.select_by_premium``'s GLOBAL minimum over the
    live chain (``min(chain, key=lambda c: abs(c.mark_price - target))``),
    not just the first strike a directional walk happens to cross.

    Walks from ATM toward OTM (premium falls) and separately toward ITM
    (premium rises), tracking the single best (closest) match seen in each
    pass. The OTM walk does NOT stop the instant it first crosses the
    target -- it continues ``OVERSHOOT`` more strikes past that crossing
    before stopping, since the far side of the crossing bracket is
    sometimes the truly closer strike (a plain "stop at first crossing"
    search can settle for a worse strike than live's exhaustive chain scan
    would actually pick -- this was a real, confirmed source of backtest
    vs. live strike-selection drift, independent of the separate mark-price
    vs. historical-trade-candle data-source gap). The ITM pass is likewise
    widened from a fixed 6 steps to be comparably thorough, since a high
    target_premium can genuinely sit many strikes ITM.

    Returns ``(symbol, strike, candles)`` or None. ``cache`` keyed by symbol
    avoids refetching across trades.
    """
    ddmmyy = expiry.strftime("%d%m%y")
    atm = int(round(entry_btc / interval) * interval)
    otm = interval if option_type == OptionType.CALL else -interval  # toward OTM = cheaper
    OVERSHOOT = 3

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

    def consider(r) -> None:
        nonlocal best
        if r is None:
            return
        diff = abs(r[3] - target_premium)
        if best is None or diff < best[0]:
            best = (diff, r[0], r[1], r[2])

    strike = atm
    crossed_at: int | None = None
    for i in range(30):
        r = ev(strike)
        consider(r)
        if r is not None and r[3] <= target_premium and crossed_at is None:
            crossed_at = i
        if crossed_at is not None and i - crossed_at >= OVERSHOOT:
            break
        strike += otm

    strike = atm - otm
    for _ in range(20):
        consider(ev(strike))
        strike -= otm

    return (best[1], best[2], best[3]) if best else None
