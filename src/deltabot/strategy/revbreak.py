"""Previous-day-zone reversal-breakout strategy (regular candles, backtest-only).

Setup detection is on the BTC underlying; the trade itself is a bought option
(priced by the backtest). Rules (one position at a time):

  * **No-trade zone** = previous day's open & close on a custom day boundary
    (default 05:30 IST). With ``zone_hi = max(PDO,PDC)``, ``zone_lo = min(PDO,PDC)``:
    close > zone_hi -> buy-only; close < zone_lo -> sell-only; between -> no entry.
  * **Pattern (regular OHLC):** a RED candle then a GREEN candle (long), or GREEN
    then RED (short). ``trigger`` = the pair's far extreme (max high long / min low
    short); ``SL`` = the opposite extreme. Enter intrabar when BTC breaks the trigger.
  * **Stop-loss is on BTC price** (the pattern extreme). The +50% take-profit is on
    the OPTION premium and is applied by the backtest, which calls
    :meth:`notify_exit` so the strategy can manage the re-entry gate.
  * **Supertrend(10,3) re-entry gate:** after a trade closes by **TP** (the +50%
    option target), the same direction is blocked until the Supertrend makes a fresh
    flip to that direction (down->up re-enables buys; up->down re-enables sells). A
    close by **SL or EOD** imposes no block.
  * Forced **EOD square-off** at the cutoff time (default 17:25 IST).

Modeling notes: intrabar is approximated by the bar high/low; the strategy never
opens on the same bar that forms the pattern; Supertrend flips are read on closed
bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..enums import PositionState, SignalDir
from ..models import Candle
from .supertrend import SupertrendCalculator


@dataclass(frozen=True)
class RevDecision:
    candle: Candle
    long_exit: bool
    short_exit: bool
    long_exit_price: float
    short_exit_price: float
    buy_signal: bool
    sell_signal: bool
    entry_price: float
    exit_reason: str | None  # "SL" | "EOD" | None
    sl_level: float | None    # the BTC stop level of a just-opened position

    @property
    def has_exit(self) -> bool:
        return self.long_exit or self.short_exit

    @property
    def has_entry(self) -> bool:
        return self.buy_signal or self.sell_signal


class RevBreakSellStrategy:
    def __init__(
        self,
        *,
        atr_period: int = 10,
        st_multiplier: float = 3.0,
        gate: str = "zone",            # "zone" = prev-day O/C; "open" = today's 05:30 open
        st_entry_filter: bool = False,  # require Supertrend aligned to take the entry
        reentry_block: bool = True,     # block same-dir re-entry after a TP until an ST flip
        day_tz: str = "Asia/Kolkata",
        day_start_hour: int = 5,
        day_start_minute: int = 30,
        square_off_hour: int = 17,
        square_off_minute: int = 25,
    ) -> None:
        self.atr_period = atr_period
        self.st_multiplier = st_multiplier
        self.gate = gate
        self.st_entry_filter = st_entry_filter
        self.reentry_block = reentry_block
        self._tz = ZoneInfo(day_tz)
        self._day_offset_s = (day_start_hour * 60 + day_start_minute) * 60
        self._sq_mins = square_off_hour * 60 + square_off_minute
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.st = SupertrendCalculator(self.atr_period, self.st_multiplier)
        self._st_dir: int | None = None  # previous closed-bar ST direction

        # Prev-day O/C (custom day boundary).
        self._pd_open: float | None = None
        self._pd_close: float | None = None
        self._cur_open: float | None = None
        self._last_close: float | None = None
        self._prev_day_ord: int | None = None

        self._prev_candle: Candle | None = None
        self._prev_now_mins: int | None = None

        # Pending breakout + open position.
        self._pending_long = self._pending_short = False
        self._pending_trigger: float | None = None
        self._pending_sl: float | None = None
        self._in_long = self._in_short = False
        self._sl_level: float | None = None

        # Re-entry gate (set on a TP exit, cleared on a Supertrend flip).
        self._buy_blocked = False
        self._sell_blocked = False

    @property
    def position_state(self) -> PositionState:
        if self._in_long:
            return PositionState.LONG
        if self._in_short:
            return PositionState.SHORT
        return PositionState.FLAT

    @property
    def ready(self) -> bool:
        """True once the Supertrend has warmed up (enough history consumed)."""
        return self.st.ready

    @property
    def has_pending(self) -> bool:
        return self._pending_long or self._pending_short

    @property
    def sl_level(self) -> float | None:
        return self._sl_level

    def force_flat(self) -> None:
        """Reset position + any pending setup to FLAT, leaving indicators/day state
        intact. Used on reconcile when the exchange shows no owned position (e.g.
        after a manual close) so the strategy stops thinking it holds a trade."""
        self._in_long = self._in_short = False
        self._sl_level = None
        self._clear_pending()

    def check_intracandle_sl(self, price: float) -> tuple[bool, bool, float | None]:
        """Has an intracandle price crossed the open position's stop? -> (long_sl, short_sl, level)."""
        long_sl = self._in_long and self._sl_level is not None and price <= self._sl_level
        short_sl = self._in_short and self._sl_level is not None and price >= self._sl_level
        return bool(long_sl), bool(short_sl), self._sl_level

    def apply_intracandle_pending(self, candle: Candle) -> tuple[bool, bool, float]:
        """Check a forming candle against a pending breakout — enter ASAP on a cross.

        Called by the live engine on every intracandle update so entry fires the
        instant price crosses the pattern trigger (not at the 5m close), and the
        setup is invalidated the instant price hits the opposite extreme (SL) first.
        SL is checked BEFORE the trigger (conservative: a touch of the stop before
        the breakout kills the setup). Mutates state so the closed-candle path then
        sees no pending. Returns ``(confirmed, invalidated, entry_price)``.
        """
        if self._pending_long and self._pending_trigger is not None and self._pending_sl is not None:
            sl, trig = self._pending_sl, self._pending_trigger
            if candle.open <= sl or candle.low <= sl:        # SL hit before breakout -> invalid
                self._clear_pending()
                return False, True, 0.0
            if candle.open >= trig or candle.high >= trig:   # price broke above the pattern high
                self._in_long, self._in_short = True, False
                self._sl_level = sl
                self._clear_pending()
                return True, False, trig
        elif self._pending_short and self._pending_trigger is not None and self._pending_sl is not None:
            sl, trig = self._pending_sl, self._pending_trigger
            if candle.open >= sl or candle.high >= sl:        # SL hit before breakout -> invalid
                self._clear_pending()
                return False, True, 0.0
            if candle.open <= trig or candle.low <= trig:     # price broke below the pattern low
                self._in_short, self._in_long = True, False
                self._sl_level = sl
                self._clear_pending()
                return True, False, trig
        return False, False, 0.0

    # ------------------------------------------------------------------ #
    def notify_exit(self, direction: int, reason: str) -> None:
        """Backtest tells us how an open trade closed (esp. the option +50% TP).

        Flattens the in-memory position and, if the exit was a TP, blocks the same
        direction until the Supertrend flips to it.
        """
        self._in_long = self._in_short = False
        self._sl_level = None
        if reason == "TP":
            if direction == SignalDir.LONG.value:
                self._buy_blocked = True
            else:
                self._sell_blocked = True

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> RevDecision | None:
        # --- Supertrend on the real candle; flips clear the re-entry blocks ---
        self.st.update(candle)
        if self.st.ready:
            d = self.st.direction()
            if self._st_dir is not None and d != self._st_dir:
                if d == 1:
                    self._buy_blocked = False   # fresh down->up flip re-enables buys
                else:
                    self._sell_blocked = False  # fresh up->down flip re-enables sells
            self._st_dir = d

        # --- Custom-day previous-day open/close ---
        shifted = datetime.fromtimestamp(candle.start_time - self._day_offset_s, tz=self._tz)
        day_ord = shifted.toordinal()
        if self._prev_day_ord is not None and day_ord != self._prev_day_ord:
            self._pd_open, self._pd_close = self._cur_open, self._last_close
            self._cur_open = candle.open
        elif self._cur_open is None:
            self._cur_open = candle.open
        self._last_close = candle.close
        have_pd = self._pd_open is not None and self._pd_close is not None
        zone_hi = max(self._pd_open, self._pd_close) if have_pd else None
        zone_lo = min(self._pd_open, self._pd_close) if have_pd else None
        above = have_pd and candle.close > zone_hi
        below = have_pd and candle.close < zone_lo

        # --- EOD square-off crossing ---
        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute
        square_off = (self._prev_now_mins is not None
                      and now_mins >= self._sq_mins and self._prev_now_mins < self._sq_mins)

        long_exit = short_exit = False
        long_exit_price = short_exit_price = candle.close
        exit_reason: str | None = None
        buy_signal = sell_signal = False
        entry_price = candle.close
        new_sl: float | None = None

        # --- Exits first (BTC stop / EOD) ---
        if self._in_long:
            if self._sl_level is not None and candle.low <= self._sl_level:
                long_exit, long_exit_price, exit_reason = True, self._sl_level, "SL"
            elif square_off:
                long_exit, long_exit_price, exit_reason = True, candle.close, "EOD"
        elif self._in_short:
            if self._sl_level is not None and candle.high >= self._sl_level:
                short_exit, short_exit_price, exit_reason = True, self._sl_level, "SL"
            elif square_off:
                short_exit, short_exit_price, exit_reason = True, candle.close, "EOD"
        if long_exit or short_exit:
            self._in_long = self._in_short = False
            self._sl_level = None

        # --- Breakout trigger for an armed setup (only while flat) ---
        flat = not self._in_long and not self._in_short
        if flat and not square_off and self._pending_long:
            trig, sl = self._pending_trigger, self._pending_sl
            if trig is not None and candle.high >= trig:
                buy_signal, entry_price, new_sl = True, trig, sl
                self._in_long, self._sl_level = True, sl
                self._clear_pending()
            elif sl is not None and candle.low <= sl:
                self._clear_pending()
        elif flat and not square_off and self._pending_short:
            trig, sl = self._pending_trigger, self._pending_sl
            if trig is not None and candle.low <= trig:
                sell_signal, entry_price, new_sl = True, trig, sl
                self._in_short, self._sl_level = True, sl
                self._clear_pending()
            elif sl is not None and candle.high >= sl:
                self._clear_pending()

        # --- Directional gate (zone vs today's open) ---
        if self.gate == "open":
            bull_gate = self._cur_open is not None and candle.close > self._cur_open
            bear_gate = self._cur_open is not None and candle.close < self._cur_open
        else:
            bull_gate, bear_gate = above, below
        # --- Supertrend filter (hard entry filter when enabled) ---
        st_up = (not self.st_entry_filter) or (self.st.ready and self.st.direction() == 1)
        st_down = (not self.st_entry_filter) or (self.st.ready and self.st.direction() == -1)
        # --- Re-entry block (only in the zone/re-entry mode, and only if enabled) ---
        no_block = self.st_entry_filter or not self.reentry_block
        buy_ok = True if no_block else not self._buy_blocked
        sell_ok = True if no_block else not self._sell_blocked

        # --- Pattern detection -> arm a new setup (flat + gates allowed) ---
        if self._prev_candle is not None and not self._in_long and not self._in_short:
            prev, cur = self._prev_candle, candle
            prev_red = prev.close < prev.open
            prev_green = prev.close > prev.open
            cur_red = cur.close < cur.open
            cur_green = cur.close > cur.open
            if prev_red and cur_green and bull_gate and st_up and buy_ok:
                self._arm(True, max(prev.high, cur.high), min(prev.low, cur.low))
            elif prev_green and cur_red and bear_gate and st_down and sell_ok:
                self._arm(False, min(prev.low, cur.low), max(prev.high, cur.high))

        self._prev_candle = candle
        self._prev_now_mins = now_mins
        self._prev_day_ord = day_ord

        if not (long_exit or short_exit or buy_signal or sell_signal):
            return None
        return RevDecision(
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
