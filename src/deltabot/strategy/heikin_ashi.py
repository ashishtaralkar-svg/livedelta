"""Heikin Ashi Supertrend(10,3) reversal strategy (Python port of
btc_heikinashi_strategy.pine, backtest-only for now).

Setup detection runs on SYNTHETIC HEIKIN ASHI candles computed internally from the
regular candles fed to :meth:`update`/:meth:`apply_intracandle_pending` -- the caller
always passes real OHLC, exactly like every other strategy in this project. Trigger/SL
LEVELS come from the HA candles; whether price actually CROSSES those levels
(invalidation, entry fill, stop) is always checked against the REAL candle's
high/low/open/close, since that is what a live order actually fills against.

Rules (one position at a time):
  * **Heikin Ashi conversion** (standard formula): ``ha_close = (o+h+l+c)/4``,
    ``ha_open = (prev_ha_open + prev_ha_close)/2`` (seeded with ``(o[0]+c[0])/2``),
    ``ha_high = max(h, ha_open, ha_close)``, ``ha_low = min(l, ha_open, ha_close)``.
  * **Entry gate:** Supertrend(10,3) computed on the HA candles -- bullish direction
    allows a BUY pattern to arm, bearish allows a SELL pattern to arm (mirrors the
    `st_up`/`st_down` entry filter convention in `revbreak.py`).
  * **Session-open gate:** track REAL BTC price at the start of each session
    (``day_start_hour:minute``, default 17:30 IST). A BUY pattern only arms while
    REAL price is currently ABOVE that session's opening price; a SELL pattern only
    arms while price is BELOW it (mirrors `revbreak.py`'s ``gate="open"`` convention).
  * **EMA trend filter** on HA close: BUY additionally requires a chained bullish
    stack, ``ha_close > ema50 > ema200``; SELL requires the mirror (``ha_close <
    ema50 < ema200``). ALL gates (Supertrend direction, session-open, and the EMA
    trend filter) must agree to arm a pattern.
  * **BUY pattern (single HA candle):** Supertrend bullish, session-open gate
    satisfied, and this candle's ``ha_open == ha_low`` (tolerant equality within a
    small epsilon) -- structurally a green/flat candle with no lower wick, since
    ``ha_low = min(real_low, ha_open, ha_close)``. Trigger = ``max(this candle's HA
    high, previous candle's HA high)`` (enter on a break above). SL =
    ``min(this candle's HA low, previous candle's HA low)`` -- the previous candle
    still widens the stop for extra margin even though the pattern itself no longer
    requires a specific previous-candle color/shape.
  * **SELL pattern:** mirror -- Supertrend bearish, session-open gate satisfied,
    ``ha_open == ha_high`` (open==high, no upper wick). Trigger =
    ``min(this candle's HA low, previous candle's HA low)`` (enter on a break below).
    SL = ``max(this candle's HA high, previous candle's HA high)``.
  * **Invalidation:** while armed (not yet triggered), if the SL level is touched
    (on REAL price) before the entry trigger, the setup is cancelled untraded.
    Checked in that order every bar/tick, so a same-bar sweep of both levels is
    conservative (matches every other strategy in this project).
  * **Fixed SL:** protects the trade for its entire life -- it never moves.
  * **Trailing SL (arms/tightens on CLOSED bars, exits ASAP on real price):**
    while long, any CLOSED bar whose HA close is below the 50 EMA arms -- or,
    if already armed, tightens -- a stop at THAT bar's HA low (mirror for
    short: HA close above the EMA arms/tightens a stop at that bar's HA high).
    It only ever moves favorably (up for a long, down for a short), never
    loosens. Once armed, this trail-SL behaves exactly like the fixed SL:
    checked against REAL price, firing the instant it is touched -- not
    waiting for the next bar to close (``check_intracandle_trail`` for
    live/tick-level checks; the closed-candle path in :meth:`update` uses this
    bar's real high/low as a proxy).
  * **No profit target.** Exits are exactly: fixed SL, the trailing SL, or EOD
    -- whichever fires first.
  * Forced **EOD square-off** at ``square_off_hour:minute`` (default 17:25 IST); no
    new patterns arm during the ``square_off`` -> ``day_start`` settlement gap
    (default 17:25-17:30 IST), and any armed-but-untriggered setup is CANCELLED at
    square-off (matches the Pine script's cancel-all in the gap).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..enums import PositionState, SignalDir
from ..models import Candle
from .supertrend import SupertrendCalculator

_EPS_REL = 1e-9  # tolerant equality for HA-derived float levels


def _close_enough(a: float, b: float, scale: float) -> bool:
    return abs(a - b) <= max(_EPS_REL * scale, 1e-9)


@dataclass(frozen=True)
class HeikinAshiDecision:
    candle: Candle
    long_exit: bool
    short_exit: bool
    long_exit_price: float
    short_exit_price: float
    buy_signal: bool
    sell_signal: bool
    entry_price: float
    exit_reason: str | None  # "SL" | "TRAIL" | "EOD" | None
    sl_level: float | None    # the BTC stop level of a just-opened position

    @property
    def has_exit(self) -> bool:
        return self.long_exit or self.short_exit

    @property
    def has_entry(self) -> bool:
        return self.buy_signal or self.sell_signal


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


class HeikinAshiStrategy:
    def __init__(
        self,
        *,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        ema_length: int = 50,
        ema200_length: int = 200,
        day_tz: str = "Asia/Kolkata",
        day_start_hour: int = 17,
        day_start_minute: int = 30,
        square_off_hour: int = 17,
        square_off_minute: int = 25,
    ) -> None:
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.ema_length = ema_length
        self.ema200_length = ema200_length
        self._tz = ZoneInfo(day_tz)
        self._sess_mins = day_start_hour * 60 + day_start_minute
        self._sq_mins = square_off_hour * 60 + square_off_minute
        self._day_offset_s = self._sess_mins * 60
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._st = SupertrendCalculator(self.st_period, self.st_multiplier)
        self._ema = _Ema(self.ema_length)
        self._ema200 = _Ema(self.ema200_length)
        self._warmup_bars = 0

        # Running Heikin Ashi state.
        self._ha_open: float | None = None
        self._ha_close: float | None = None
        # Previous CLOSED bar's HA candle (for widening the trigger/SL).
        self._prev_ha: tuple[float, float, float, float] | None = None  # (o,h,l,c)

        self._prev_now_mins: int | None = None

        # Session-open gate (REAL price at this session's day_start).
        self._cur_open: float | None = None
        self._prev_day_ord: int | None = None

        # Pending breakout + open position.
        self._pending_long = self._pending_short = False
        self._pending_trigger: float | None = None
        self._pending_sl: float | None = None
        self._in_long = self._in_short = False
        self._sl_level: float | None = None
        # Trailing SL: None until a closed bar arms it (see update()); once
        # set, only ever tightens (up for long, down for short).
        self._trail_sl_level: float | None = None

    @property
    def position_state(self) -> PositionState:
        if self._in_long:
            return PositionState.LONG
        if self._in_short:
            return PositionState.SHORT
        return PositionState.FLAT

    @property
    def ready(self) -> bool:
        """True once the Supertrend AND the EMA(200) trend filter have both
        warmed up. NOTE: deliberate divergence from Pine, which has no
        explicit warmup gate -- avoids arming spurious signals against an
        immature EMA200 average (same rationale as Ema200SupertrendStrategy)."""
        return self._st.ready and self._warmup_bars >= self.ema200_length

    @property
    def has_pending(self) -> bool:
        return self._pending_long or self._pending_short

    @property
    def sl_level(self) -> float | None:
        return self._sl_level

    def force_flat(self) -> None:
        """Reset position + any pending setup to FLAT, leaving indicators/HA state
        intact. Used on reconcile when the exchange shows no owned position."""
        self._in_long = self._in_short = False
        self._sl_level = None
        self._trail_sl_level = None
        self._clear_pending()

    def check_intracandle_sl(self, price: float) -> tuple[bool, bool, float | None]:
        """Has an intracandle REAL price crossed the open position's fixed stop?
        -> (long_sl, short_sl, level). Never checks the TRAIL exit -- see
        :meth:`check_intracandle_trail` for that."""
        long_sl = self._in_long and self._sl_level is not None and price <= self._sl_level
        short_sl = self._in_short and self._sl_level is not None and price >= self._sl_level
        return bool(long_sl), bool(short_sl), self._sl_level

    def check_intracandle_trail(self, price: float) -> tuple[bool, bool, float | None]:
        """Has an intracandle REAL price crossed the ARMED trailing-SL level?
        -> (long_trail, short_trail, level). Returns all-False/None until a
        CLOSED bar has armed ``_trail_sl_level`` (see :meth:`update`); once
        armed it behaves exactly like the fixed SL -- checked against real
        price, firing the instant it is touched."""
        level = self._trail_sl_level
        long_trail = self._in_long and level is not None and price <= level
        short_trail = self._in_short and level is not None and price >= level
        return bool(long_trail), bool(short_trail), level

    def apply_intracandle_pending(self, candle: Candle) -> tuple[bool, bool, float]:
        """Check a forming (REAL price) candle against a pending breakout -- enter
        ASAP on a cross. SL is checked BEFORE the trigger (conservative: a touch of
        the stop before the breakout kills the setup untraded). Returns
        ``(confirmed, invalidated, entry_price)``."""
        if not (self._pending_long or self._pending_short):
            return False, False, 0.0
        # Settlement gap (square_off -> day_start, default 17:25-17:30 IST): the
        # Pine script cancels all pending stop orders here -- a setup armed before
        # square-off must never fill during the gap or survive into the next session.
        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        if self._sq_mins <= local.hour * 60 + local.minute < self._sess_mins:
            self._clear_pending()
            return False, True, 0.0
        if self._pending_long and self._pending_trigger is not None and self._pending_sl is not None:
            sl, trig = self._pending_sl, self._pending_trigger
            if candle.open <= sl or candle.low <= sl:
                self._clear_pending()
                return False, True, 0.0
            if candle.open >= trig or candle.high >= trig:
                self._in_long, self._in_short = True, False
                self._sl_level = sl
                self._trail_sl_level = None
                self._clear_pending()
                return True, False, trig
        elif self._pending_short and self._pending_trigger is not None and self._pending_sl is not None:
            sl, trig = self._pending_sl, self._pending_trigger
            if candle.open >= sl or candle.high >= sl:
                self._clear_pending()
                return False, True, 0.0
            if candle.open <= trig or candle.low <= trig:
                self._in_short, self._in_long = True, False
                self._sl_level = sl
                self._trail_sl_level = None
                self._clear_pending()
                return True, False, trig
        return False, False, 0.0

    # ------------------------------------------------------------------ #
    def notify_exit(self, direction: int, reason: str) -> None:
        """Backtest/engine tells us an open trade closed via the intracandle path
        (fixed SL / trailing SL). Just flattens the in-memory position; no
        re-entry gate exists for this strategy (unlike RevBreak-Sell's
        Supertrend-flip TP gate)."""
        self._in_long = self._in_short = False
        self._sl_level = None
        self._trail_sl_level = None

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> HeikinAshiDecision | None:
        # --- Advance the running Heikin Ashi candle ---
        if self._ha_open is None:
            ha_open = (candle.open + candle.close) / 2.0
        else:
            ha_open = (self._ha_open + self._ha_close) / 2.0
        ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0
        ha_high = max(candle.high, ha_open, ha_close)
        ha_low = min(candle.low, ha_open, ha_close)

        self._st.update(Candle(candle.start_time, ha_open, ha_high, ha_low, ha_close, candle.volume))
        ema_val = self._ema.update(ha_close)
        ema200_val = self._ema200.update(ha_close)
        self._warmup_bars += 1

        # --- Session-open (REAL price) tracking, custom day boundary = day_start ---
        shifted = datetime.fromtimestamp(candle.start_time - self._day_offset_s, tz=self._tz)
        day_ord = shifted.toordinal()
        if self._prev_day_ord is not None and day_ord != self._prev_day_ord:
            self._cur_open = candle.open
        elif self._cur_open is None:
            self._cur_open = candle.open
        self._prev_day_ord = day_ord

        # --- EOD square-off / settlement-gap crossing ---
        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute
        square_off = (self._prev_now_mins is not None
                      and now_mins >= self._sq_mins and self._prev_now_mins < self._sq_mins)
        in_settlement = self._sq_mins <= now_mins < self._sess_mins

        # Settlement-gap order cancellation (matches Pine's cancel-all in the
        # 17:25-17:30 gap): an armed-but-untriggered setup from before square-off
        # is killed outright, not merely blocked -- it must not survive into the
        # next session.
        if (square_off or in_settlement) and self.has_pending:
            self._clear_pending()

        long_exit = short_exit = False
        long_exit_price = short_exit_price = candle.close
        exit_reason: str | None = None
        buy_signal = sell_signal = False
        entry_price = candle.close
        new_sl: float | None = None

        # --- Exits first: fixed SL, then EOD, then the trailing SL -- SL-before-
        #     EOD matches the established precedent in revbreak.py for a same-bar
        #     tie; the trail is the lowest-priority (discretionary) exit. Both SL
        #     and TRAIL are ASAP/real-price checks; the live engine's intracandle
        #     path (check_intracandle_sl / check_intracandle_trail) normally fires
        #     first -- here (closed-candle backtest) this bar's real high/low is
        #     used as a proxy for the same real-price cross. TRAIL only applies
        #     once a prior CLOSED bar has armed self._trail_sl_level (below). ---
        if self._in_long:
            if self._sl_level is not None and candle.low <= self._sl_level:
                long_exit, long_exit_price, exit_reason = True, self._sl_level, "SL"
            elif square_off:
                long_exit, long_exit_price, exit_reason = True, candle.close, "EOD"
            elif self._trail_sl_level is not None and candle.low <= self._trail_sl_level:
                long_exit, long_exit_price, exit_reason = True, self._trail_sl_level, "TRAIL"
        elif self._in_short:
            if self._sl_level is not None and candle.high >= self._sl_level:
                short_exit, short_exit_price, exit_reason = True, self._sl_level, "SL"
            elif square_off:
                short_exit, short_exit_price, exit_reason = True, candle.close, "EOD"
            elif self._trail_sl_level is not None and candle.high >= self._trail_sl_level:
                short_exit, short_exit_price, exit_reason = True, self._trail_sl_level, "TRAIL"
        if long_exit or short_exit:
            self._in_long = self._in_short = False
            self._sl_level = None
            self._trail_sl_level = None

        # --- Arm/tighten the trailing SL: a CLOSED bar whose HA close is beyond
        #     the EMA (against the position's bias) arms -- or, if already armed,
        #     tightens -- a stop at THIS bar's HA low/high. Checked against real
        #     price from here on (see check_intracandle_trail above). Only ever
        #     moves favorably (up for a long, down for a short), never loosens. ---
        if self._in_long and ha_close < ema_val:
            self._trail_sl_level = ha_low if self._trail_sl_level is None else max(self._trail_sl_level, ha_low)
        elif self._in_short and ha_close > ema_val:
            self._trail_sl_level = ha_high if self._trail_sl_level is None else min(self._trail_sl_level, ha_high)

        # --- Breakout trigger for an armed setup (closed-bar fallback; the live
        #     engine's intracandle path normally fires first) -- only while flat,
        #     never during the settlement gap. ---
        flat = not self._in_long and not self._in_short
        if flat and not in_settlement and self._pending_long:
            trig, sl = self._pending_trigger, self._pending_sl
            if sl is not None and candle.low <= sl:
                self._clear_pending()
            elif trig is not None and candle.high >= trig:
                buy_signal, entry_price, new_sl = True, trig, sl
                self._in_long, self._sl_level = True, sl
                self._trail_sl_level = None
                self._clear_pending()
        elif flat and not in_settlement and self._pending_short:
            trig, sl = self._pending_trigger, self._pending_sl
            if sl is not None and candle.high >= sl:
                self._clear_pending()
            elif trig is not None and candle.low <= trig:
                sell_signal, entry_price, new_sl = True, trig, sl
                self._in_short, self._sl_level = True, sl
                self._trail_sl_level = None
                self._clear_pending()

        # --- Pattern detection -> arm a new setup (flat, nothing pending, outside
        #     the settlement gap, indicators warmed up) ---
        flat_now = not self._in_long and not self._in_short
        if (self._prev_ha is not None and flat_now and not self.has_pending
                and not in_settlement and self.ready):
            _, p_h, p_l, _ = self._prev_ha
            scale = candle.close
            st_up = self._st.ready and self._st.direction() == 1
            st_down = self._st.ready and self._st.direction() == -1
            bull_gate = self._cur_open is not None and candle.close > self._cur_open
            bear_gate = self._cur_open is not None and candle.close < self._cur_open
            # EMA trend filter: BUY needs a chained bullish stack, HA close >
            # ema50 > ema200; SELL needs the mirror (close < ema50 < ema200).
            trend_up = ha_close > ema_val and ema_val > ema200_val
            trend_down = ha_close < ema_val and ema_val < ema200_val
            # Single-candle pattern: this HA candle's open==low (buy) / open==high
            # (sell) -- structurally implies green/red respectively, since
            # ha_low/ha_high are already min/max(real, ha_open, ha_close).
            buy_ok = st_up and bull_gate and trend_up and _close_enough(ha_open, ha_low, scale)
            sell_ok = st_down and bear_gate and trend_down and _close_enough(ha_open, ha_high, scale)
            if buy_ok:
                self._arm(True, max(ha_high, p_h), min(ha_low, p_l))
            elif sell_ok:
                self._arm(False, min(ha_low, p_l), max(ha_high, p_h))

        self._prev_ha = (ha_open, ha_high, ha_low, ha_close)
        self._prev_now_mins = now_mins
        self._ha_open, self._ha_close = ha_open, ha_close

        if not (long_exit or short_exit or buy_signal or sell_signal):
            return None
        return HeikinAshiDecision(
            candle=candle, long_exit=long_exit, short_exit=short_exit,
            long_exit_price=long_exit_price, short_exit_price=short_exit_price,
            buy_signal=buy_signal, sell_signal=sell_signal, entry_price=entry_price,
            exit_reason=exit_reason, sl_level=new_sl,
        )

    # ------------------------------------------------------------------ #
    def _arm(self, is_long: bool, trigger: float, sl: float) -> None:
        self._pending_long, self._pending_short = is_long, not is_long
        self._pending_trigger, self._pending_sl = trigger, sl

    def _clear_pending(self) -> None:
        self._pending_long = self._pending_short = False
        self._pending_trigger = self._pending_sl = None
