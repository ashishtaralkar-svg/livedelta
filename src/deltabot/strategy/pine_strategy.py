"""Pine-equivalent intraday strategy — the single source of truth shared by the
live engine and the backtester so their results cannot diverge.

This is a faithful Python port of ``ashish.pine`` ("Prev Day OHLC + Supertrend +
EMA H/L"). It operates strictly on *closed* candles; each call to :meth:`update`
consumes one closed candle and returns a :class:`StrategyDecision` describing the
exits/entries that fire on that bar.

Entry rules (one trade at a time, long *or* short, may be flat):

  BUY  when price is above prev-day open, prev-day close, EMA(high), EMA(low)
       AND the Supertrend is in an uptrend.
  SELL when price is below all of the same levels AND the Supertrend is in a
       downtrend.

A fresh signal requires the condition to have just become true (it was false on
the previous bar) — except on the first bar after the forced square-off, where a
still-true condition is allowed to re-enter.

Exit rules:
  * Stop-loss: long exits if the bar's low pierces the entry bar's previous-candle
    low; short exits if the bar's high pierces the entry bar's previous-candle high
    (filled at the stop level).
  * Forced square-off at the configured cut-off time (default 17:30 IST), filled at
    the bar close.

Day boundaries (for the previous-day open/close levels) use a custom day that
begins at ``day_start_hour:day_start_minute`` in ``day_tz`` (default 05:30 IST),
exactly as the Pine script does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..enums import PositionState
from ..models import Candle
from .indicators import EmaCalculator
from .supertrend import SupertrendCalculator


@dataclass(frozen=True)
class StrategyDecision:
    """The exits/entries produced by feeding one closed candle to the strategy."""

    candle: Candle

    # Exits evaluated on this bar.
    long_exit: bool
    short_exit: bool
    long_exit_sl: bool
    short_exit_sl: bool
    long_sq_off: bool
    short_sq_off: bool
    long_exit_price: float
    short_exit_price: float

    # Fresh entries fired on this bar (a new entry may flip an opposite position
    # that exits on the same bar).
    buy_signal: bool
    sell_signal: bool
    entry_price: float  # fill price for the entry (the bar close)

    # Resulting target position after applying this bar's exits and entries.
    target_state: PositionState

    @property
    def has_exit(self) -> bool:
        return self.long_exit or self.short_exit

    @property
    def has_entry(self) -> bool:
        return self.buy_signal or self.sell_signal

    @property
    def exit_reason(self) -> str | None:
        if self.long_exit_sl or self.short_exit_sl:
            return "SL"
        if self.long_sq_off or self.short_sq_off:
            return "EOD"
        return None


class PineStrategy:
    """Stateful, repaint-safe port of the Pine indicator's signal logic."""

    def __init__(
        self,
        *,
        atr_period: int = 10,
        st_multiplier: float = 3.0,
        ema_length: int = 50,
        day_tz: str = "Asia/Kolkata",
        day_start_hour: int = 5,
        day_start_minute: int = 30,
        square_off_hour: int = 17,
        square_off_minute: int = 30,
        use_close: bool = True,
    ) -> None:
        self.atr_period = atr_period
        self.st_multiplier = st_multiplier
        self.ema_length = ema_length
        self._tz = ZoneInfo(day_tz)
        self._day_offset_s = (day_start_hour * 60 + day_start_minute) * 60
        self._sq_mins = square_off_hour * 60 + square_off_minute
        self.use_close = use_close
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear all state (indicators, day tracking, and position)."""
        self.st = SupertrendCalculator(self.atr_period, self.st_multiplier)
        self.ema_high = EmaCalculator(self.ema_length)
        self.ema_low = EmaCalculator(self.ema_length)

        # Previous-day open/close tracking (custom day boundary).
        self._pd_open: float | None = None
        self._pd_close: float | None = None
        self._cur_open: float | None = None
        self._last_close: float | None = None
        self._prev_day_ord: int | None = None

        # Per-bar context carried to the next bar.
        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._prev_buy_cond = False
        self._prev_sell_cond = False
        self._prev_now_mins: int | None = None
        self._prev_square_off = False

        # Open position state.
        self._in_long = False
        self._in_short = False
        self._long_prev_low: float | None = None
        self._short_prev_high: float | None = None
        self._long_entry: float | None = None
        self._short_entry: float | None = None

        self._ready = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        """True once the indicators and prev-day levels can produce valid signals."""
        return self._ready

    @property
    def position_state(self) -> PositionState:
        if self._in_long:
            return PositionState.LONG
        if self._in_short:
            return PositionState.SHORT
        return PositionState.FLAT

    @property
    def pd_open(self) -> float | None:
        return self._pd_open

    @property
    def pd_close(self) -> float | None:
        return self._pd_close

    def seed(self, candles: list[Candle]) -> None:
        """Warm up indicators and day/level state from historical closed candles.

        Seeding establishes indicator and prev-day state and the ``cond[1]``
        history, but never opens a simulated position — the live position is taken
        from the exchange via reconciliation, not replayed from history.
        """
        self.reset()
        for candle in candles:
            self._step(candle, warmup=True)

    def update(self, candle: Candle) -> StrategyDecision | None:
        """Consume one closed candle and return its decision (or ``None``)."""
        return self._step(candle, warmup=False)

    def sync_position(
        self,
        state: PositionState,
        *,
        entry_price: float | None = None,
        stop_level: float | None = None,
    ) -> None:
        """Force the in-memory position to match the exchange (used on reconcile).

        When the exchange shows a position the strategy did not open, the stop
        level is unknown; ``stop_level`` (best-effort, e.g. the last candle's
        low/high) is used so a stop can still be tracked.
        """
        if state == PositionState.LONG:
            self._in_long, self._in_short = True, False
            self._long_entry = entry_price
            self._long_prev_low = stop_level if stop_level is not None else self._long_prev_low
            self._short_prev_high = self._short_entry = None
        elif state == PositionState.SHORT:
            self._in_short, self._in_long = True, False
            self._short_entry = entry_price
            self._short_prev_high = stop_level if stop_level is not None else self._short_prev_high
            self._long_prev_low = self._long_entry = None
        else:
            self._in_long = self._in_short = False
            self._long_prev_low = self._short_prev_high = None
            self._long_entry = self._short_entry = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _step(self, candle: Candle, warmup: bool) -> StrategyDecision | None:
        # --- Indicators ---
        self.st.update(candle)
        ema_high = self.ema_high.update(candle.high)
        ema_low = self.ema_low.update(candle.low)
        st_uptrend = self.st.ready and self.st.direction() == 1
        st_downtrend = self.st.ready and self.st.direction() == -1

        # --- Custom-day boundary -> previous-day open/close ---
        shifted = datetime.fromtimestamp(candle.start_time - self._day_offset_s, tz=self._tz)
        day_ord = shifted.toordinal()
        is_new_day = self._prev_day_ord is not None and day_ord != self._prev_day_ord
        if is_new_day:
            # Freeze the just-finished day as "previous day" (cur_open is None
            # before the very first boundary, exactly like the Pine na seed).
            self._pd_open = self._cur_open
            self._pd_close = self._last_close
            self._cur_open = candle.open
        self._last_close = candle.close

        # --- Square-off time (uses the unshifted bar time in the day timezone) ---
        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute
        square_off = (
            self._prev_now_mins is not None
            and now_mins >= self._sq_mins
            and self._prev_now_mins < self._sq_mins
        )
        after_sq_off = self._prev_square_off

        # --- Entry conditions ---
        buy_price = candle.close if self.use_close else candle.low
        sell_price = candle.close if self.use_close else candle.high
        pd_open, pd_close = self._pd_open, self._pd_close
        have_levels = pd_open is not None and pd_close is not None and self.st.ready
        buy_cond = False
        sell_cond = False
        if have_levels:
            assert pd_open is not None and pd_close is not None  # narrowed by have_levels
            levels_above = max(pd_open, pd_close, ema_high, ema_low)
            levels_below = min(pd_open, pd_close, ema_high, ema_low)
            buy_cond = buy_price > levels_above and st_uptrend
            sell_cond = sell_price < levels_below and st_downtrend

        # --- Exit evaluation (up-front, so a fresh signal can fire on the same
        #     candle an SL is hit) ---
        long_exit_sl = self._in_long and self._long_prev_low is not None and candle.low <= self._long_prev_low
        short_exit_sl = (
            self._in_short and self._short_prev_high is not None and candle.high >= self._short_prev_high
        )
        long_sq_off = self._in_long and square_off
        short_sq_off = self._in_short and square_off
        long_exit = bool(long_exit_sl or long_sq_off)
        short_exit = bool(short_exit_sl or short_sq_off)

        flat = (not self._in_long or long_exit) and (not self._in_short or short_exit)
        buy_signal = bool(
            buy_cond and flat and not square_off and (not self._prev_buy_cond or after_sq_off)
        )
        sell_signal = bool(
            sell_cond and flat and not square_off and (not self._prev_sell_cond or after_sq_off)
        )

        long_exit_price = (
            self._long_prev_low if long_exit_sl and self._long_prev_low is not None else candle.close
        )
        short_exit_price = (
            self._short_prev_high
            if short_exit_sl and self._short_prev_high is not None
            else candle.close
        )

        # --- Apply position changes (skipped during warmup) ---
        if not warmup:
            if long_exit:
                self._in_long = False
                self._long_prev_low = None
                self._long_entry = None
            if short_exit:
                self._in_short = False
                self._short_prev_high = None
                self._short_entry = None
            if buy_signal:
                self._in_long, self._in_short = True, False
                self._long_prev_low = self._prev_low  # entry bar's previous-candle low
                self._long_entry = candle.close
            if sell_signal:
                self._in_short, self._in_long = True, False
                self._short_prev_high = self._prev_high  # entry bar's previous-candle high
                self._short_entry = candle.close

        # --- Carry context to the next bar ---
        self._prev_buy_cond = buy_cond
        self._prev_sell_cond = sell_cond
        self._prev_now_mins = now_mins
        self._prev_square_off = square_off
        self._prev_high = candle.high
        self._prev_low = candle.low
        self._prev_day_ord = day_ord
        self._ready = have_levels

        if warmup or not have_levels:
            return None

        return StrategyDecision(
            candle=candle,
            long_exit=long_exit,
            short_exit=short_exit,
            long_exit_sl=bool(long_exit_sl),
            short_exit_sl=bool(short_exit_sl),
            long_sq_off=bool(long_sq_off),
            short_sq_off=bool(short_sq_off),
            long_exit_price=long_exit_price,
            short_exit_price=short_exit_price,
            buy_signal=buy_signal,
            sell_signal=sell_signal,
            entry_price=candle.close,
            target_state=self.position_state,
        )
