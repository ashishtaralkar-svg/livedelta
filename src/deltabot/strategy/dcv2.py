"""DCv2 Strategy -- Python port of ``dchannel_strategy.pine`` (2026-07-15 state).

Directional BTC strategy (plain BUY/SELL, no options): Donchian(20) touch +
open==low/high confirm on synthetic Heikin Ashi, gated by the EMA(50)/EMA(200)
relationship, with a fixed signal-range SL and an EMA-reversal exit.

Rules (buy side described; sell side is the exact mirror):

  * **Hunting is STATE-based** (Pine UPDATE #9): while EMA(50) > EMA(200)
    (both on HA close) the BUY hunt is armed; while EMA(50) < EMA(200) the
    SELL hunt is armed. The transition resets both sides' touch/range state;
    staying in the same state never wipes an in-progress range.
  * **Touch**: a candle whose REAL low touches/breaches the Donchian(20)
    LOWER band (prior 20 closed HA candles, current excluded) anchors the
    signal range (range hi/lo start from that candle's HA high/low).
  * **highest_touch re-anchor**: a LATER band-touching candle with a lower
    HA low restarts the range from itself; any other candle just widens the
    range (max HA high / min HA low).
  * **Confirm**: a candle with HA open == HA low (no lower wick) completes
    the signal range -- pure price-action, no per-candle EMA re-check (the
    state gate already established direction). Range includes this candle.
  * **Entry**: ASAP when REAL price breaks ABOVE the range high. **Pre-entry
    invalidation**: REAL price touching the range low first discards the
    setup untraded -- and because hunting is state-armed, the invalidation
    candle itself can anchor the next range if it touches the band.
  * **Exits -- exactly two** (Pine UPDATE #5, no TP, no EOD close):
      (a) **SL**: REAL price touching the signal-range low, FIXED for the
          life of the trade.
      (b) **EMA reversal**: close at market the moment EMA(50) is on the
          unfavorable side of EMA(200) (below it, for a long). NOTE: the
          Pine script uses the discrete crossunder EVENT; this port checks
          the RELATIONSHIP each bar, which is identical whenever the trade
          opened with the relationship favorable and additionally closes the
          rare trade whose pending setup filled after the EMA had already
          flipped (in Pine that trade could only exit via SL).
      **Exception (2026-07-15)**: if at BUY trigger time BOTH EMA(50) and
      EMA(200) sit ABOVE the trigger price (price broke out from below the
      EMAs), the EMA-reversal exit is DISABLED for that trade -- it exits on
      the fixed signal-range SL only. Mirror for a SELL whose trigger is
      above both EMAs.
  * **After an SL exit**, the SL bar itself never counts as the re-touch --
    a FRESH band touch from the next bar onward starts the new range
    (Pine's justClosedSL).
  * **Settlement gap** (17:25-17:30 IST): cancels any pending setup and
    clears hunt state; blocks entries; does NOT close an open position.
    State-based arming re-arms immediately after the gap.
  * **Weekend block** (Sat/Sun IST by default): a trigger hit on a blocked
    day consumes the pending setup without placing a trade (mirrors the
    Pine dayBlocked gate on strategy.entry()).

The Williams %R gate that exists in the Pine inputs is OFF by default there
and unused live -- it is deliberately NOT ported.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..enums import PositionState
from ..models import Candle

__all__ = ["DCv2Strategy", "DCv2Decision"]

_EPS_REL = 1e-9  # tolerant equality for HA-derived float levels


def _close_enough(a: float, b: float, scale: float) -> bool:
    return abs(a - b) <= max(_EPS_REL * scale, 1e-9)


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


class _Donchian:
    """Highest-high / lowest-low over the PRIOR ``period`` closed bars --
    the current bar is excluded until pushed at the end of ``update()``."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._highs: deque[float] = deque(maxlen=period)
        self._lows: deque[float] = deque(maxlen=period)

    @property
    def ready(self) -> bool:
        return len(self._highs) >= self.period

    @property
    def upper(self) -> float | None:
        return max(self._highs) if self._highs else None

    @property
    def lower(self) -> float | None:
        return min(self._lows) if self._lows else None

    def push(self, high: float, low: float) -> None:
        self._highs.append(high)
        self._lows.append(low)


@dataclass(frozen=True)
class DCv2Decision:
    candle: Candle
    long_exit: bool
    short_exit: bool
    long_exit_price: float
    short_exit_price: float
    buy_signal: bool
    sell_signal: bool
    entry_price: float
    exit_reason: str | None  # "SL" | "EMA_CROSS" | None
    sl_level: float | None   # the SL active right after a just-opened position

    @property
    def has_exit(self) -> bool:
        return self.long_exit or self.short_exit

    @property
    def has_entry(self) -> bool:
        return self.buy_signal or self.sell_signal


class DCv2Strategy:
    def __init__(
        self,
        *,
        dc_period: int = 20,
        ema_trend_length: int = 50,
        ema_long_length: int = 200,
        skip_weekdays: frozenset[int] = frozenset({5, 6}),  # Mon=0 .. Sat=5, Sun=6 (IST)
        day_tz: str = "Asia/Kolkata",
        day_start_hour: int = 17,
        day_start_minute: int = 30,
        square_off_hour: int = 17,
        square_off_minute: int = 25,
    ) -> None:
        self.dc_period = dc_period
        self.ema_trend_length = ema_trend_length
        self.ema_long_length = ema_long_length
        self.skip_weekdays = skip_weekdays
        self._tz = ZoneInfo(day_tz)
        self._sess_mins = day_start_hour * 60 + day_start_minute
        self._sq_mins = square_off_hour * 60 + square_off_minute
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._dc = _Donchian(self.dc_period)
        self._ema_trend = _Ema(self.ema_trend_length)
        self._ema_long = _Ema(self.ema_long_length)
        self._warmup_bars = 0

        # Running Heikin Ashi state.
        self._ha_open: float | None = None
        self._ha_close: float | None = None

        self._prev_now_mins: int | None = None

        # Hunt state machine.
        self._hunt_bull = self._hunt_bear = False
        self._touched_bull = self._touched_bear = False
        self._range_hi_bull: float | None = None
        self._range_lo_bull: float | None = None
        self._range_hi_bear: float | None = None
        self._range_lo_bear: float | None = None
        self._anchor_lo_bull: float | None = None
        self._anchor_hi_bear: float | None = None

        # Pending breakout + open position.
        self._pending_long = self._pending_short = False
        self._pending_trigger: float | None = None
        self._pending_sl: float | None = None
        self._in_long = self._in_short = False
        self._sl_level: float | None = None
        # False when the trade triggered from the "wrong" side of both EMAs
        # (buy below both / sell above both) -> that trade exits on SL only.
        self._ema_exit_enabled = True

    @property
    def position_state(self) -> PositionState:
        if self._in_long:
            return PositionState.LONG
        if self._in_short:
            return PositionState.SHORT
        return PositionState.FLAT

    @property
    def ready(self) -> bool:
        return self._dc.ready and self._warmup_bars >= self.ema_long_length

    @property
    def has_pending(self) -> bool:
        return self._pending_long or self._pending_short

    @property
    def sl_level(self) -> float | None:
        return self._sl_level

    def force_flat(self) -> None:
        self._in_long = self._in_short = False
        self._sl_level = None
        self._ema_exit_enabled = True
        self._clear_pending()
        self._clear_hunts()

    def _ema_exit_for(self, trig: float, is_long: bool) -> bool:
        """EMA-reversal exit is DISABLED when the trade triggered from the
        wrong side of BOTH EMAs (buy below both, sell above both) -- that
        trade rides on its fixed signal-range SL alone."""
        et, el = self._ema_trend.value, self._ema_long.value
        if et is None or el is None:
            return True
        if is_long:
            return not (et > trig and el > trig)
        return not (et < trig and el < trig)

    # ------------------------------------------------------------------ #
    # Intracandle (live/ASAP) helpers -- same conventions as DchannelStrategy.
    # ------------------------------------------------------------------ #
    def check_intracandle_sl(self, price: float) -> tuple[bool, bool, float | None]:
        long_sl = self._in_long and self._sl_level is not None and price <= self._sl_level
        short_sl = self._in_short and self._sl_level is not None and price >= self._sl_level
        return bool(long_sl), bool(short_sl), self._sl_level

    def apply_intracandle_pending(self, candle: Candle) -> tuple[bool, bool, float]:
        """ASAP entry/invalidation against REAL price -- SL side checked BEFORE
        the trigger (conservative). Returns ``(confirmed, invalidated, entry_price)``."""
        if self._pending_long and self._pending_trigger is not None and self._pending_sl is not None:
            trig, sl = self._pending_trigger, self._pending_sl
            if candle.open <= sl or candle.low <= sl:
                self._clear_pending()
                return False, True, 0.0
            if candle.open >= trig or candle.high >= trig:
                self._in_long, self._in_short = True, False
                self._sl_level = sl
                self._ema_exit_enabled = self._ema_exit_for(trig, is_long=True)
                self._clear_pending()
                return True, False, trig
        elif self._pending_short and self._pending_trigger is not None and self._pending_sl is not None:
            trig, sl = self._pending_trigger, self._pending_sl
            if candle.open >= sl or candle.high >= sl:
                self._clear_pending()
                return False, True, 0.0
            if candle.open <= trig or candle.low <= trig:
                self._in_short, self._in_long = True, False
                self._sl_level = sl
                self._ema_exit_enabled = self._ema_exit_for(trig, is_long=False)
                self._clear_pending()
                return True, False, trig
        return False, False, 0.0

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> DCv2Decision | None:
        # --- Advance the running Heikin Ashi candle ---
        if self._ha_open is None:
            ha_open = (candle.open + candle.close) / 2.0
        else:
            ha_open = (self._ha_open + self._ha_close) / 2.0
        ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0
        ha_high = max(candle.high, ha_open, ha_close)
        ha_low = min(candle.low, ha_open, ha_close)
        self._ha_open, self._ha_close = ha_open, ha_close

        ema_trend = self._ema_trend.update(ha_close)
        ema_long = self._ema_long.update(ha_close)
        dc_upper, dc_lower = self._dc.upper, self._dc.lower
        self._warmup_bars += 1

        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute
        square_off = (self._prev_now_mins is not None
                      and now_mins >= self._sq_mins and self._prev_now_mins < self._sq_mins)
        in_gap = self._sq_mins <= now_mins < self._sess_mins
        day_blocked = local.weekday() in self.skip_weekdays

        # --- 0. Settlement gap: cancel pending + clear hunts (never closes a position) ---
        if square_off or in_gap:
            self._clear_pending()
            self._clear_hunts()

        # --- 0.5. Hunt arming: STATE-based on the EMA(50)/EMA(200) relationship.
        #     The "not already armed" guard makes the reset fire only on the
        #     transition, so an in-progress range is never wiped mid-hunt. ---
        if not square_off and not in_gap and self.ready:
            if ema_trend > ema_long and not self._hunt_bull:
                self._hunt_bull, self._hunt_bear = True, False
                self._reset_ranges()
            elif ema_trend < ema_long and not self._hunt_bear:
                self._hunt_bear, self._hunt_bull = True, False
                self._reset_ranges()

        long_exit = short_exit = False
        long_exit_price = short_exit_price = candle.close
        exit_reason: str | None = None
        buy_signal = sell_signal = False
        entry_price = candle.close
        new_sl: float | None = None
        just_closed_sl = False

        # --- 1/2. Exits: fixed SL (REAL price at the range extreme), or the EMA
        #     relationship flipping against the position. No TP, no EOD close. ---
        if self._in_long:
            if self._sl_level is not None and candle.low <= self._sl_level:
                long_exit, long_exit_price, exit_reason = True, self._sl_level, "SL"
                just_closed_sl = True
            elif self._ema_exit_enabled and ema_trend < ema_long:
                long_exit, long_exit_price, exit_reason = True, candle.close, "EMA_CROSS"
        elif self._in_short:
            if self._sl_level is not None and candle.high >= self._sl_level:
                short_exit, short_exit_price, exit_reason = True, self._sl_level, "SL"
                just_closed_sl = True
            elif self._ema_exit_enabled and ema_trend > ema_long:
                short_exit, short_exit_price, exit_reason = True, candle.close, "EMA_CROSS"
        if long_exit or short_exit:
            self._in_long = self._in_short = False
            self._sl_level = None
            self._ema_exit_enabled = True

        # --- 3. Breakout trigger for a PRIOR-armed pending setup (REAL price).
        #     Weekend block gates only the trade itself: a trigger hit on a
        #     blocked day consumes the setup untraded. Invalidation clears the
        #     pending; hunting is already state-armed, and the hunt block below
        #     runs on this SAME bar, so the invalidation candle can anchor the
        #     next range. ---
        flat = not self._in_long and not self._in_short
        if flat and not square_off and not in_gap and self._pending_long:
            trig, sl = self._pending_trigger, self._pending_sl
            if sl is not None and candle.low <= sl:
                self._clear_pending()   # invalidated before the trigger (SL side first, conservative)
            elif trig is not None and candle.high >= trig:
                if not day_blocked:
                    buy_signal, entry_price, new_sl = True, trig, sl
                    self._in_long, self._sl_level = True, sl
                    self._ema_exit_enabled = not (ema_trend > trig and ema_long > trig)
                self._clear_pending()
        elif flat and not square_off and not in_gap and self._pending_short:
            trig, sl = self._pending_trigger, self._pending_sl
            if sl is not None and candle.high >= sl:
                self._clear_pending()
            elif trig is not None and candle.low <= trig:
                if not day_blocked:
                    sell_signal, entry_price, new_sl = True, trig, sl
                    self._in_short, self._sl_level = True, sl
                    self._ema_exit_enabled = not (ema_trend < trig and ema_long < trig)
                self._clear_pending()

        # --- 4. Hunt progression (HA-based), only while flat/no pending/gates ok.
        #     just_closed_sl: the SL bar itself never counts as the re-touch. ---
        flat_now = not self._in_long and not self._in_short
        if (flat_now and not self.has_pending and not square_off and not in_gap
                and not just_closed_sl and self.ready):
            # Bullish hunt progression.
            if self._hunt_bull and dc_lower is not None:
                if not self._touched_bull:
                    if candle.low <= dc_lower:
                        self._touched_bull = True
                        self._range_hi_bull, self._range_lo_bull = ha_high, ha_low
                        self._anchor_lo_bull = ha_low
                else:
                    if (candle.low <= dc_lower and self._anchor_lo_bull is not None
                            and ha_low < self._anchor_lo_bull):
                        self._range_hi_bull, self._range_lo_bull = ha_high, ha_low
                        self._anchor_lo_bull = ha_low
                    else:
                        self._range_hi_bull = max(self._range_hi_bull, ha_high)
                        self._range_lo_bull = min(self._range_lo_bull, ha_low)
                    # CONFIRM: open==low shape only -- direction came from the state gate.
                    if _close_enough(ha_open, ha_low, candle.close):
                        self._pending_long = True
                        self._pending_trigger = self._range_hi_bull
                        self._pending_sl = self._range_lo_bull
                        self._hunt_bull = self._touched_bull = False
                        self._range_hi_bull = self._range_lo_bull = None
                        self._anchor_lo_bull = None

            # Bearish hunt progression (mirror).
            if self._hunt_bear and dc_upper is not None:
                if not self._touched_bear:
                    if candle.high >= dc_upper:
                        self._touched_bear = True
                        self._range_hi_bear, self._range_lo_bear = ha_high, ha_low
                        self._anchor_hi_bear = ha_high
                else:
                    if (candle.high >= dc_upper and self._anchor_hi_bear is not None
                            and ha_high > self._anchor_hi_bear):
                        self._range_hi_bear, self._range_lo_bear = ha_high, ha_low
                        self._anchor_hi_bear = ha_high
                    else:
                        self._range_hi_bear = max(self._range_hi_bear, ha_high)
                        self._range_lo_bear = min(self._range_lo_bear, ha_low)
                    if _close_enough(ha_open, ha_high, candle.close):
                        self._pending_short = True
                        self._pending_trigger = self._range_lo_bear
                        self._pending_sl = self._range_hi_bear
                        self._hunt_bear = self._touched_bear = False
                        self._range_hi_bear = self._range_lo_bear = None
                        self._anchor_hi_bear = None

        self._dc.push(ha_high, ha_low)
        self._prev_now_mins = now_mins

        if not (long_exit or short_exit or buy_signal or sell_signal):
            return None
        return DCv2Decision(
            candle=candle, long_exit=long_exit, short_exit=short_exit,
            long_exit_price=long_exit_price, short_exit_price=short_exit_price,
            buy_signal=buy_signal, sell_signal=sell_signal, entry_price=entry_price,
            exit_reason=exit_reason, sl_level=new_sl,
        )

    # ------------------------------------------------------------------ #
    def _clear_pending(self) -> None:
        self._pending_long = self._pending_short = False
        self._pending_trigger = self._pending_sl = None

    def _reset_ranges(self) -> None:
        self._touched_bull = self._touched_bear = False
        self._range_hi_bull = self._range_lo_bull = None
        self._range_hi_bear = self._range_lo_bear = None
        self._anchor_lo_bull = self._anchor_hi_bear = None

    def _clear_hunts(self) -> None:
        self._hunt_bull = self._hunt_bear = False
        self._reset_ranges()
