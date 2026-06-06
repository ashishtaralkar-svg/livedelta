"""Download and cache historical 1-minute candles from Delta's REST history API."""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from ..logging_setup import get_logger
from ..models import Candle

log = get_logger(__name__)

# Delta caps candles per request; page in chunks well under any server limit.
_MAX_CANDLES_PER_REQUEST = 2000
_RESOLUTION_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "1d": 86400,
}


def _cache_path(cache_dir: Path, symbol: str, resolution: str, start: int, end: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}_{resolution}_{start}_{end}.parquet"


def download(
    symbol: str,
    start: int,
    end: int,
    resolution: str = "1m",
    base_url: str = "https://api.india.delta.exchange",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Download closed candles in ``[start, end]`` (epoch seconds) as a DataFrame.

    Results are cached to parquet keyed by (symbol, resolution, start, end).
    Columns: time, open, high, low, close, volume — sorted ascending by time.
    """
    cache_dir = cache_dir or Path(".cache/candles")
    path = _cache_path(cache_dir, symbol, resolution, start, end)
    if path.exists():
        log.info("Loading candles from cache", extra={"extra": {"path": str(path)}})
        return pd.read_parquet(path)

    step = _RESOLUTION_SECONDS.get(resolution, 60)
    window = _MAX_CANDLES_PER_REQUEST * step
    rows: list[dict] = []
    cursor = start
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        while cursor < end:
            chunk_end = min(cursor + window, end)
            resp = client.get(
                "/v2/history/candles",
                params={"symbol": symbol, "resolution": resolution, "start": cursor, "end": chunk_end},
            )
            resp.raise_for_status()
            result = resp.json().get("result", []) or []
            rows.extend(result)
            log.info(
                "Fetched candle page",
                extra={"extra": {"start": cursor, "end": chunk_end, "count": len(result)}},
            )
            cursor = chunk_end

    if not rows:
        raise RuntimeError(f"No candles returned for {symbol} {resolution} {start}-{end}")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    df.to_parquet(path, index=False)
    log.info("Cached candles", extra={"extra": {"path": str(path), "rows": len(df)}})
    return df


def df_to_candles(df: pd.DataFrame) -> list[Candle]:
    """Convert a candle DataFrame into a list of :class:`Candle`."""
    return [
        Candle(
            start_time=int(row.time),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(getattr(row, "volume", 0.0) or 0.0),
        )
        for row in df.itertuples(index=False)
    ]
