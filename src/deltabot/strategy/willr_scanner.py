"""WilliamsRScanner Strategy -- a per-CONTRACT oversold-bounce breakout,
scanned across the live option chain. UNLIKE EVERY OTHER STRATEGY IN THIS
REPO, the indicators run on an OPTION'S OWN premium candles, never on BTC
price -- there is no BTC signal here at all.

Because same-day-expiry options (used throughout this repo) only trade for
a single day, a "50-period" indicator can only ever be intraday -- 50 bars
of that SAME contract's own 5-minute candles from today alone (~4.2 hours
of history), never a multi-day series on one option symbol. So this class
tracks MANY contracts simultaneously (one independent indicator state per
symbol), fed by the caller scanning the live option chain each closed bar
for whichever CE/PE currently sit inside the target premium band (e.g.
50-100) -- see core/willr_trader.py for live, scripts/backtest_willr.py for
backtest.

Per contract, independently (never cross-contract):
  1. ARM the instant Williams %R(wr_period) crosses from >= -80 to < -80 on
     THIS contract's own closed candle, AND that same candle's close is
     below its own SMA(ma_len), which is itself below its own EMA(ema_len)
     -- close < ma < ema. The armed level is THAT candle's own HIGH.
  2. ENTRY fires on a LATER candle (within breakout_wait_bars) whose HIGH
     first reaches/exceeds the armed level -- a classic "signal candle
     pins an anchor, wait for a breakout above it" pattern, same shape as
     ema21_breakdown.py's own signal1/breakout, just run per-contract on
     the contract's own OHLC instead of BTC's.
  3. Entries only fire inside [entry_start_hour, entry_end_hour) IST.
  4. A contract that has already produced one entry is never re-armed --
     see reset() for the caller's own daily reset.

The caller is responsible for: which contracts to feed each closed bar (the
"scan" -- see the trader/backtest module docstrings for the current premium
band and universe policy), buying the exact contract an ENTRY fires on
(never a different strike), the profit target (a %-gain on premium, this
class has no exit logic at all), and the daily square-off.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

from ..models import Candle


class _Ema:
    def __init__(self, length: int) -> None:
        self._alpha = 2.0 / (length + 1.0)
        self._value: float | None = None

    def update(self, x: float) -> float:
        self._value = x if self._value is None else self._alpha * x + (1 - self._alpha) * self._value
        return self._value


class _Sma:
    def __init__(self, length: int) -> None:
        self._length = length
        self._buf: deque[float] = deque(maxlen=length)

    def update(self, x: float) -> float | None:
        self._buf.append(x)
        if len(self._buf) < self._length:
            return None
        return sum(self._buf) / self._length


class _WilliamsR:
    """Standard Williams %R over `period` bars: range [-100, 0] -- -100 at
    the period's low, 0 at the period's high."""

    def __init__(self, period: int) -> None:
        self._period = period
        self._highs: deque[float] = deque(maxlen=period)
        self._lows: deque[float] = deque(maxlen=period)

    def update(self, high: float, low: float, close: float) -> float | None:
        self._highs.append(high)
        self._lows.append(low)
        if len(self._highs) < self._period:
            return None
        hh, ll = max(self._highs), min(self._lows)
        if hh == ll:
            return -50.0  # degenerate flat window -- neutral midpoint, avoids /0
        return -100.0 * (hh - close) / (hh - ll)


class _ContractState:
    def __init__(self, ema_len: int, ma_len: int, wr_period: int) -> None:
        self.ema = _Ema(ema_len)
        self.ma = _Sma(ma_len)
        self.wr = _WilliamsR(wr_period)
        self.prev_wr: float | None = None
        self.armed = False
        self.armed_high: float | None = None
        self.armed_bars_left = 0
        self.entered = False


class WilliamsRScanner:
    def __init__(
        self,
        ema_len: int = 50,
        ma_len: int = 50,
        wr_period: int = 14,
        breakout_wait_bars: int = 3,
        entry_start_hour: int = 14,
        entry_end_hour: int = 17,
        day_tz: str = "Asia/Kolkata",
    ) -> None:
        self.ema_len = ema_len
        self.ma_len = ma_len
        self.wr_period = wr_period
        self.breakout_wait_bars = breakout_wait_bars
        self._entry_start_mins = entry_start_hour * 60
        self._entry_end_mins = entry_end_hour * 60
        self._tz = ZoneInfo(day_tz)
        self._states: dict[str, _ContractState] = {}

    def _in_entry_window(self, ts: int) -> bool:
        local = datetime.fromtimestamp(ts, tz=self._tz)
        mins = local.hour * 60 + local.minute
        return self._entry_start_mins <= mins < self._entry_end_mins

    def update_contract(self, symbol: str, candle: Candle) -> bool:
        """Feed one contract's newly-closed 5m candle. Returns True if THIS
        candle is the entry trigger for THIS contract."""
        st = self._states.setdefault(symbol, _ContractState(self.ema_len, self.ma_len, self.wr_period))
        if st.entered:
            return False

        ema = st.ema.update(candle.close)
        ma = st.ma.update(candle.close)
        wr = st.wr.update(candle.high, candle.low, candle.close)
        ready = ma is not None and wr is not None

        entry = False
        if ready:
            crossed_oversold = st.prev_wr is not None and st.prev_wr >= -80.0 and wr < -80.0
            if crossed_oversold and candle.close < ma < ema:
                st.armed = True
                st.armed_high = candle.high
                st.armed_bars_left = self.breakout_wait_bars
            elif st.armed:
                if candle.high >= st.armed_high:
                    if self._in_entry_window(candle.start_time):
                        entry = True
                        st.entered = True
                    st.armed = False
                    st.armed_high = None
                else:
                    st.armed_bars_left -= 1
                    if st.armed_bars_left <= 0:
                        st.armed = False
                        st.armed_high = None
        if wr is not None:
            st.prev_wr = wr
        return entry

    def is_armed(self, symbol: str) -> bool:
        st = self._states.get(symbol)
        return bool(st and st.armed)

    def has_entered(self, symbol: str) -> bool:
        st = self._states.get(symbol)
        return bool(st and st.entered)

    def drop_contract(self, symbol: str) -> None:
        """Forget a contract entirely (e.g. it expired, or the caller's
        daily reset)."""
        self._states.pop(symbol, None)

    def reset_all(self) -> None:
        """Daily reset -- yesterday's contracts don't exist today anyway
        (same-day expiry), so every state is stale."""
        self._states.clear()
