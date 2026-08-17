"""DailyTrendEmaCross Strategy -- Python port of daily_trend_ema_cross_strategy.pine.

Daily-candle trend gate -> EMA(50)/MA(50) crossover EVENT -> Heikin Ashi
green/red 2-candle setup -> fixed-range SL. See the .pine file for the full
rule set and the two bugs caught while building it (repainting daily lookup
in the original pasted logic; a pending-vs-active SL conflation that could
have wiped an open position's stop on a trend flip) -- this port mirrors the
FIXED version, bar for bar.

Rules (bearish/SHORT side; bullish/LONG is the exact mirror):
  * DAILY TREND (IST midnight calendar days): on each day boundary, using the
    day that just completed (D) and the one before it (D-1) --
      BEAR TRIGGER: D is red (close<open), D-1 was green (close>open), and
      D's close is below D-1's OPEN. -> trend = DOWN, key_level = D's own
      open. While DOWN, a later day's close reclaiming key_level flips
      trend = UP (the only path to UP -- there's no separate bull trigger,
      ported faithfully from the source spec).
  * EMA(50)/MA(50) on intraday HA close: a CROSS UNDER (discrete event, not
    just "is below") while trend=DOWN arms the bearish setup hunt. Each
    fresh cross while still in the same trend re-arms it. A trend flip
    clears any armed/in-progress hunt (but never an OPEN position's SL).
  * SETUP: once armed, a GREEN HA candle immediately followed by a RED HA
    candle. Setup range = the HA high/low spanning those two candles.
  * ENTRY: real price breaking BELOW the setup low, checked on bars AFTER
    the setup completed (never the setup's own completion bar).
  * SL: the setup's HIGH (opposite extreme) -- fixed for the life of the
    trade.
  * NO profit target modeled here (option-premium concept, no BTC
    equivalent -- see the live-execution note in the .pine file).
  * Settlement gap (17:25-17:30 IST by default): blocks NEW entries only,
    never closes an open position.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..enums import PositionState
from ..models import Candle

__all__ = ["DailyTrendEmaCrossStrategy", "DailyTrendEmaCrossDecision"]


class _Ema:
    def __init__(self, length: int) -> None:
        self._alpha = 2.0 / (length + 1.0)
        self._value: float | None = None

    def update(self, x: float) -> float:
        self._value = x if self._value is None else self._alpha * x + (1 - self._alpha) * self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


class _Sma:
    def __init__(self, length: int) -> None:
        from collections import deque
        self._window = deque(maxlen=length)

    def update(self, x: float) -> float:
        self._window.append(x)
        return sum(self._window) / len(self._window)

    @property
    def value(self) -> float | None:
        return sum(self._window) / len(self._window) if self._window else None


@dataclass(frozen=True)
class DailyTrendEmaCrossDecision:
    candle: Candle
    long_exit: bool
    short_exit: bool
    long_exit_price: float
    short_exit_price: float
    buy_signal: bool
    sell_signal: bool
    entry_price: float
    exit_reason: str | None  # "SL" | None

    @property
    def has_exit(self) -> bool:
        return self.long_exit or self.short_exit

    @property
    def has_entry(self) -> bool:
        return self.buy_signal or self.sell_signal


class DailyTrendEmaCrossStrategy:
    def __init__(
        self,
        *,
        ema_len: int = 50,
        ma_len: int = 50,
        day_tz: str = "Asia/Kolkata",
        gap_start_hour: int = 17,
        gap_start_minute: int = 25,
        gap_end_hour: int = 17,
        gap_end_minute: int = 30,
        use_longs: bool = True,
        use_shorts: bool = True,
    ) -> None:
        self.ema_len = ema_len
        self.ma_len = ma_len
        self.use_longs = use_longs
        self.use_shorts = use_shorts
        self._tz = ZoneInfo(day_tz)
        self._gap_start_mins = gap_start_hour * 60 + gap_start_minute
        self._gap_end_mins = gap_end_hour * 60 + gap_end_minute
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._ema = _Ema(self.ema_len)
        self._ma = _Sma(self.ma_len)
        self._warmup_bars = 0

        self._ha_open: float | None = None
        self._ha_close: float | None = None
        self._prev_ha_high: float | None = None
        self._prev_ha_low: float | None = None

        # Daily trend (IST midnight calendar days).
        self._cal_day = None
        self._today_open: float | None = None
        self._today_close: float | None = None
        self._prev_day_open: float | None = None
        self._prev_day_close: float | None = None
        self._trend = 0        # 1 = up, -1 = down, 0 = undefined
        self._key_level: float | None = None

        # EMA/MA cross-event tracking.
        self._prev_ema_sign = 0

        # Hunt / setup state.
        self._hunting_bear = False
        self._hunting_bull = False
        self._prev_ha_green_bear = False
        self._prev_ha_red_bull = False
        self._pending_short_trigger: float | None = None
        self._pending_short_sl: float | None = None
        self._pending_long_trigger: float | None = None
        self._pending_long_sl: float | None = None

        # Position + its OWN stop (kept separate from the pending setup's SL
        # so a trend flip mid-trade can never wipe it -- see module docstring).
        self._in_long = self._in_short = False
        self._active_short_sl: float | None = None
        self._active_long_sl: float | None = None

    @property
    def position_state(self) -> PositionState:
        if self._in_long:
            return PositionState.LONG
        if self._in_short:
            return PositionState.SHORT
        return PositionState.FLAT

    @property
    def trend(self) -> int:
        return self._trend

    @property
    def ready(self) -> bool:
        return self._warmup_bars >= max(self.ema_len, self.ma_len)

    def force_flat(self) -> None:
        self._in_long = self._in_short = False
        self._active_short_sl = self._active_long_sl = None

    def debug_state(self) -> dict:
        def r(x):
            return round(x, 2) if isinstance(x, (int, float)) else None
        return {
            "trend": self._trend, "key_level": r(self._key_level),
            "hunting_bear": self._hunting_bear, "hunting_bull": self._hunting_bull,
            "pending_short_trig": r(self._pending_short_trigger), "pending_short_sl": r(self._pending_short_sl),
            "pending_long_trig": r(self._pending_long_trigger), "pending_long_sl": r(self._pending_long_sl),
            "active_short_sl": r(self._active_short_sl), "active_long_sl": r(self._active_long_sl),
            "pos": self.position_state.name,
        }

    # ------------------------------------------------------------------ #
    def _update_daily_trend(self, candle: Candle) -> bool:
        """Roll RAW price into the running IST-midnight daily candle. On a day
        boundary, evaluate the bear trigger from the day that just completed
        vs the one before it, then check the reclaim-to-uptrend condition.
        Returns True iff the trend value changed on this call."""
        day = datetime.fromtimestamp(candle.start_time, tz=self._tz).date()
        if self._cal_day is None:
            self._cal_day = day
            self._today_open = candle.open
            self._today_close = candle.close
            return False
        if day == self._cal_day:
            self._today_close = candle.close
            return False

        completed_open, completed_close = self._today_open, self._today_close
        prev_trend = self._trend
        if self._prev_day_open is not None:
            is_red = completed_close < completed_open
            is_prev_green = self._prev_day_close > self._prev_day_open
            bear_trigger = is_red and is_prev_green and (completed_close < self._prev_day_open)
            if bear_trigger:
                self._trend = -1
                self._key_level = completed_open
            elif (self._trend == -1 and self._key_level is not None
                  and completed_close > self._key_level):
                self._trend = 1
                self._key_level = None
        self._prev_day_open, self._prev_day_close = completed_open, completed_close
        self._cal_day = day
        self._today_open = candle.open
        self._today_close = candle.close
        return self._trend != prev_trend

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> DailyTrendEmaCrossDecision | None:
        # --- Advance the running Heikin Ashi candle ---
        if self._ha_open is None:
            ha_open = (candle.open + candle.close) / 2.0
        else:
            ha_open = (self._ha_open + self._ha_close) / 2.0
        ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0
        ha_high = max(candle.high, ha_open, ha_close)
        ha_low = min(candle.low, ha_open, ha_close)
        prev_ha_high, prev_ha_low = self._prev_ha_high, self._prev_ha_low
        self._ha_open, self._ha_close = ha_open, ha_close
        ha_green = ha_close > ha_open
        ha_red = ha_close < ha_open

        ema_val = self._ema.update(ha_close)
        ma_val = self._ma.update(ha_close)
        self._warmup_bars += 1

        trend_changed = self._update_daily_trend(candle)

        sign = 1 if ema_val > ma_val else (-1 if ema_val < ma_val else 0)
        bear_cross = sign == -1 and self._prev_ema_sign == 1
        bull_cross = sign == 1 and self._prev_ema_sign == -1
        if sign != 0:
            self._prev_ema_sign = sign

        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute
        in_gap = self._gap_start_mins <= now_mins < self._gap_end_mins

        long_exit = short_exit = False
        long_exit_price = short_exit_price = candle.close
        exit_reason: str | None = None
        buy_signal = sell_signal = False
        entry_price = candle.close

        # --- 0. Trend flip: clear only PENDING/unfired hunting state -- an
        #     OPEN position's own stop is untouched. ---
        if trend_changed:
            self._hunting_bear = self._hunting_bull = False
            self._pending_short_trigger = self._pending_short_sl = None
            self._pending_long_trigger = self._pending_long_sl = None

        # --- 1. Exits: fixed setup-range SL only. ---
        if self._in_short:
            if self._active_short_sl is not None and candle.high >= self._active_short_sl:
                short_exit, short_exit_price, exit_reason = True, self._active_short_sl, "SL"
        elif self._in_long:
            if self._active_long_sl is not None and candle.low <= self._active_long_sl:
                long_exit, long_exit_price, exit_reason = True, self._active_long_sl, "SL"
        if long_exit or short_exit:
            self._in_long = self._in_short = False
            self._active_short_sl = self._active_long_sl = None

        # --- 2. Entry: real-price breakout of a PRIOR-armed pending trigger
        #     -- runs BEFORE setup hunting below, so a setup completing THIS
        #     bar can only fire starting the NEXT bar. Filling promotes the
        #     pending SL to the ACTIVE one. ---
        flat_now = not self._in_long and not self._in_short
        if (flat_now and not in_gap and self.use_shorts
                and self._pending_short_trigger is not None and candle.low <= self._pending_short_trigger):
            sell_signal, entry_price = True, self._pending_short_trigger
            self._in_short = True
            self._active_short_sl = self._pending_short_sl
            self._pending_short_trigger = self._pending_short_sl = None
        elif (flat_now and not in_gap and self.use_longs
                and self._pending_long_trigger is not None and candle.high >= self._pending_long_trigger):
            buy_signal, entry_price = True, self._pending_long_trigger
            self._in_long = True
            self._active_long_sl = self._pending_long_sl
            self._pending_long_trigger = self._pending_long_sl = None

        # --- 3. EMA/MA cross arms the hunt (only in the matching trend). ---
        flat_for_hunt = not self._in_long and not self._in_short and not buy_signal and not sell_signal
        if flat_for_hunt and self._trend == -1 and bear_cross and self._pending_short_trigger is None:
            self._hunting_bear = True
            self._prev_ha_green_bear = False
        if flat_for_hunt and self._trend == 1 and bull_cross and self._pending_long_trigger is None:
            self._hunting_bull = True
            self._prev_ha_red_bull = False

        # --- 4. HA green->red / red->green setup scan, only while armed+flat. ---
        if flat_for_hunt and self._hunting_bear and self._pending_short_trigger is None:
            if self._prev_ha_green_bear and ha_red and prev_ha_high is not None:
                setup_hi = max(ha_high, prev_ha_high)
                setup_lo = min(ha_low, prev_ha_low)
                self._pending_short_trigger = setup_lo
                self._pending_short_sl = setup_hi
                self._hunting_bear = False
            self._prev_ha_green_bear = ha_green
        if flat_for_hunt and self._hunting_bull and self._pending_long_trigger is None:
            if self._prev_ha_red_bull and ha_green and prev_ha_high is not None:
                setup_hi = max(ha_high, prev_ha_high)
                setup_lo = min(ha_low, prev_ha_low)
                self._pending_long_trigger = setup_hi
                self._pending_long_sl = setup_lo
                self._hunting_bull = False
            self._prev_ha_red_bull = ha_red

        self._prev_ha_high, self._prev_ha_low = ha_high, ha_low
        if not (long_exit or short_exit or buy_signal or sell_signal):
            return None
        return DailyTrendEmaCrossDecision(
            candle=candle, long_exit=long_exit, short_exit=short_exit,
            long_exit_price=long_exit_price, short_exit_price=short_exit_price,
            buy_signal=buy_signal, sell_signal=sell_signal, entry_price=entry_price,
            exit_reason=exit_reason,
        )
