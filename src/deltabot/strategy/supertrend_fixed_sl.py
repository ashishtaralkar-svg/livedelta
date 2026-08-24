"""SupertrendFixedSl Strategy -- Python port of supertrend_fixed_sl_strategy.pine.

Supertrend(10,3) flip timing, a FROZEN stop (read once at the flip, never
re-derived from Supertrend's still-updating value), sell-only. Unlike the
Pine chart (which can only simulate one net BTC position and so treats the
CE and PE legs as mutually exclusive -- see that file's LIMITATION note),
this port tracks the two legs as the genuinely INDEPENDENT contracts they
are live: a CE sale and a PE sale can be open AT THE SAME TIME.

Rules (CE/bearish side; PE/bullish is the exact mirror):
  * SUPERTREND(10,3) on real (non-HA) OHLC, closed 5-minute bars -- unless
    use_heikin_ashi=True, in which case Supertrend runs on synthetic
    Heikin Ashi OHLC instead (same conversion formula used throughout this
    repo, e.g. DCv2Strategy: ha_open = avg of the PRIOR ha_open/ha_close,
    ha_close = avg of this bar's O/H/L/C, ha_high/ha_low widened to also
    include ha_open/ha_close). Entry price, and the frozen SL's CROSSING
    check, always use the REAL candle regardless of this flag -- only the
    indicator's own OHLC feed changes; execution still happens against
    real price/premium, matching every other HA-based strategy here.
  * FLIP: the bar where Supertrend's direction changes from up to down ->
    SELL the CE (bearish signal -> sell CALL, same convention as every
    other strategy in this repo), entering at that bar's close. Blocked
    while a CE leg is ALREADY open (no pyramiding into the same contract).
  * SL = Supertrend's OWN VALUE on the flip bar, frozen from that point on
    -- never re-read from Supertrend's live value on later bars.
  * Real price crossing that frozen level closes the leg. NO profit target.
  * A CE leg and a PE leg can be open simultaneously (independent
    contracts); each has its own frozen SL and closes independently.

ema200_filter (added on request): an additional gate checked AT THE FLIP --
a bullish flip (PE) only fires if close > EMA(200); a bearish flip (CE)
only fires if close < EMA(200). Disagreeing simply consumes the flip
untraded (no leg opens) -- same convention as ema21_breakdown's
--ema200-filter. Off (default) = unchanged behavior.

Rollover (a backtest/live-engine concept, NOT modeled inside this class --
see scripts/backtest_supertrend_fixed_sl.py's --rollover / the module docs
there): this strategy's own state already supports it for free. _in_short/
_in_long and _active_short_sl/_active_long_sl represent the DIRECTIONAL
commitment + its frozen SL, entirely independent of whether an OPTION
CONTRACT is currently open -- the exit check (real price crossing the
frozen level) runs unconditionally every bar regardless. So a caller that
wants "the option got squared off at 17:25 but the frozen SL hasn't been
hit yet -- keep the SAME frozen SL and sell a fresh option for the next
day" simply must NOT call force_flat_short()/force_flat_long() at
square-off; the frozen SL survives automatically since nothing here ever
clears it except an actual SL cross or an explicit force_flat call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..models import Candle

__all__ = ["SupertrendFixedSlStrategy", "SupertrendFixedSlDecision"]


class _Rma:
    """Wilder's smoothed moving average (alpha=1/length), seeded with the
    first value -- same simplified-seed convention as this codebase's other
    _Ema classes (Pine's ta.atr() SMA-seeds the first `length` bars instead;
    the difference washes out after warmup)."""

    def __init__(self, length: int) -> None:
        self._alpha = 1.0 / length
        self._value: float | None = None

    def update(self, x: float) -> float:
        self._value = x if self._value is None else self._alpha * x + (1 - self._alpha) * self._value
        return self._value


class _Ema:
    def __init__(self, length: int) -> None:
        self._alpha = 2.0 / (length + 1.0)
        self._value: float | None = None

    def update(self, x: float) -> float:
        self._value = x if self._value is None else self._alpha * x + (1 - self._alpha) * self._value
        return self._value


class _Supertrend:
    """Standard ATR-based Supertrend. direction: -1 = uptrend (supertrend
    below price), 1 = downtrend (supertrend above price) -- same sign
    convention as Pine's built-in ta.supertrend()."""

    def __init__(self, atr_period: int, factor: float) -> None:
        self.factor = factor
        self._atr = _Rma(atr_period)
        self._prev_close: float | None = None
        self._final_upper: float | None = None
        self._final_lower: float | None = None
        self._value: float | None = None
        self._direction = 0   # 0 = undefined (first bar)

    def update(self, high: float, low: float, close: float) -> tuple[float, int]:
        tr = (high - low) if self._prev_close is None else max(
            high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        atr = self._atr.update(tr)
        hl2 = (high + low) / 2.0
        basic_upper = hl2 + self.factor * atr
        basic_lower = hl2 - self.factor * atr

        if self._final_upper is None:
            final_upper, final_lower = basic_upper, basic_lower
        else:
            final_upper = (basic_upper if (basic_upper < self._final_upper
                                            or self._prev_close > self._final_upper)
                            else self._final_upper)
            final_lower = (basic_lower if (basic_lower > self._final_lower
                                            or self._prev_close < self._final_lower)
                            else self._final_lower)

        if self._direction == 0:
            direction = 1 if close <= final_upper else -1
        elif self._value == self._final_upper:   # previous bar was a downtrend
            direction = 1 if close <= final_upper else -1
        else:                                     # previous bar was an uptrend
            direction = -1 if close >= final_lower else 1

        value = final_upper if direction == 1 else final_lower

        self._prev_close = close
        self._final_upper, self._final_lower = final_upper, final_lower
        self._value, self._direction = value, direction
        return value, direction


@dataclass(frozen=True)
class SupertrendFixedSlDecision:
    candle: Candle
    short_exit: bool   # CE leg closed (SL)
    long_exit: bool    # PE leg closed (SL)
    short_exit_price: float
    long_exit_price: float
    sell_signal: bool  # bearish flip -> new CE sale
    buy_signal: bool   # bullish flip -> new PE sale
    entry_price: float
    short_sl: float | None   # the frozen SL just set for a NEW CE leg
    long_sl: float | None    # the frozen SL just set for a NEW PE leg

    @property
    def has_exit(self) -> bool:
        return self.short_exit or self.long_exit

    @property
    def has_entry(self) -> bool:
        return self.sell_signal or self.buy_signal


class SupertrendFixedSlStrategy:
    def __init__(
        self,
        *,
        atr_period: int = 10,
        factor: float = 3.0,
        day_tz: str = "Asia/Kolkata",
        gap_start_hour: int = 17,
        gap_start_minute: int = 25,
        gap_end_hour: int = 17,
        gap_end_minute: int = 30,
        trade_ce: bool = True,
        trade_pe: bool = True,
        use_heikin_ashi: bool = False,
        ema200_filter: bool = False,
        ema200_len: int = 200,
    ) -> None:
        self.atr_period = atr_period
        self.factor = factor
        # trade_ce/trade_pe: which side(s) are allowed to fire. A bearish
        # flip while trade_ce=False (or a bullish flip while trade_pe=False)
        # is simply ignored -- no CE/PE sold, no state armed for that side.
        self.trade_ce = trade_ce
        self.trade_pe = trade_pe
        # use_heikin_ashi: feed Supertrend synthetic HA OHLC instead of the
        # real candle's own OHLC. See module docstring -- everything else
        # (entry price, SL-crossing check) still uses the real candle.
        self.use_heikin_ashi = use_heikin_ashi
        # ema200_filter: gate checked AT THE FLIP -- see module docstring.
        self.ema200_filter = ema200_filter
        self.ema200_len = ema200_len
        self._tz = ZoneInfo(day_tz)
        self._gap_start_mins = gap_start_hour * 60 + gap_start_minute
        self._gap_end_mins = gap_end_hour * 60 + gap_end_minute
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._st = _Supertrend(self.atr_period, self.factor)
        self._ema200 = _Ema(self.ema200_len)
        self._prev_direction = 0
        self._warmup_bars = 0
        self._ha_open: float | None = None
        self._ha_close: float | None = None

        self._in_short = False   # CE sold (open)
        self._in_long = False    # PE sold (open)
        self._active_short_sl: float | None = None
        self._active_long_sl: float | None = None

    @property
    def ready(self) -> bool:
        need = max(self.atr_period, self.ema200_len) if self.ema200_filter else self.atr_period
        return self._warmup_bars >= need

    @property
    def in_short(self) -> bool:
        return self._in_short

    @property
    def in_long(self) -> bool:
        return self._in_long

    def force_flat_short(self) -> None:
        self._in_short = False
        self._active_short_sl = None

    def force_flat_long(self) -> None:
        self._in_long = False
        self._active_long_sl = None

    def force_flat(self) -> None:
        self.force_flat_short()
        self.force_flat_long()

    def debug_state(self) -> dict:
        def r(x):
            return round(x, 2) if isinstance(x, (int, float)) else None
        return {
            "st_value": r(self._st._value), "direction": self._st._direction,
            "ema200": r(self._ema200._value),
            "in_short": self._in_short, "in_long": self._in_long,
            "active_short_sl": r(self._active_short_sl), "active_long_sl": r(self._active_long_sl),
        }

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> SupertrendFixedSlDecision | None:
        if self.use_heikin_ashi:
            ha_open = (candle.open + candle.close) / 2.0 if self._ha_open is None \
                else (self._ha_open + self._ha_close) / 2.0
            ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0
            ha_high = max(candle.high, ha_open, ha_close)
            ha_low = min(candle.low, ha_open, ha_close)
            self._ha_open, self._ha_close = ha_open, ha_close
            st_high, st_low, st_close = ha_high, ha_low, ha_close
        else:
            st_high, st_low, st_close = candle.high, candle.low, candle.close
        st_value, direction = self._st.update(st_high, st_low, st_close)
        # EMA200 always tracks the REAL close, regardless of use_heikin_ashi --
        # the trend filter reflects actual price, not a synthetic smoothed one.
        ema200 = self._ema200.update(candle.close)
        self._warmup_bars += 1

        bear_flip = self._prev_direction == -1 and direction == 1
        bull_flip = self._prev_direction == 1 and direction == -1
        self._prev_direction = direction

        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute
        in_gap = self._gap_start_mins <= now_mins < self._gap_end_mins

        short_exit = long_exit = False
        short_exit_price = long_exit_price = candle.close
        sell_signal = buy_signal = False
        entry_price = candle.close
        new_short_sl: float | None = None
        new_long_sl: float | None = None

        # --- Exits: real price crossing the FROZEN level (independent legs). ---
        if self._in_short and self._active_short_sl is not None and candle.high >= self._active_short_sl:
            short_exit, short_exit_price = True, self._active_short_sl
            self._in_short = False
            self._active_short_sl = None
        if self._in_long and self._active_long_sl is not None and candle.low <= self._active_long_sl:
            long_exit, long_exit_price = True, self._active_long_sl
            self._in_long = False
            self._active_long_sl = None

        # --- Entry: the flip bar itself is the signal. Blocked while that
        #     side's OWN leg is already open (no pyramiding); the OTHER
        #     side's state is irrelevant -- legs are independent.
        #     ema200_filter: bearish flip (CE) only if close < EMA200;
        #     bullish flip (PE) only if close > EMA200. Disagreeing just
        #     consumes the flip untraded -- no leg opens. ---
        bear_trend_ok = (not self.ema200_filter) or (candle.close < ema200)
        bull_trend_ok = (not self.ema200_filter) or (candle.close > ema200)
        if (not in_gap and bear_flip and not self._in_short and self.ready and self.trade_ce
                and bear_trend_ok):
            sell_signal, entry_price = True, candle.close
            self._in_short = True
            self._active_short_sl = new_short_sl = st_value
        if (not in_gap and bull_flip and not self._in_long and self.ready and self.trade_pe
                and bull_trend_ok):
            buy_signal, entry_price = True, candle.close
            self._in_long = True
            self._active_long_sl = new_long_sl = st_value

        if not (short_exit or long_exit or sell_signal or buy_signal):
            return None
        return SupertrendFixedSlDecision(
            candle=candle, short_exit=short_exit, long_exit=long_exit,
            short_exit_price=short_exit_price, long_exit_price=long_exit_price,
            sell_signal=sell_signal, buy_signal=buy_signal, entry_price=entry_price,
            short_sl=new_short_sl, long_sl=new_long_sl,
        )
