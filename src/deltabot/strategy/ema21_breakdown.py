"""Ema21Breakdown Strategy -- Python port of the user-supplied
"21 EMA Breakdown Strategy (Strict Single Trade)" Pine script, plus an
explicitly-requested mirrored BUY side (the source script itself has none).

SELL side (the original, faithfully ported):
  * SIGNAL 1 (anchor): a candle with OPEN > EMA(21) AND CLOSE < EMA(21).
    Arms tracking; SL is pinned to THIS candle's HIGH. Re-arms on every
    fresh matching candle WHILE still hunting for signal 2 (not yet
    waiting-for-breakout) -- a later, fresher anchor overwrites an
    in-progress one. Once signal 2 has fired (waiting for breakout), a new
    signal-1 candle no longer resets anything.
  * INVALIDATION: any candle CLOSING above the EMA cancels tracking,
    whether that's during the signal-1-to-signal-2 wait or (redundantly,
    since signal 2 itself required close < EMA) after.
  * MAX WAIT: if signal 2 hasn't appeared within 3 bars of signal 1,
    tracking cancels.
  * SIGNAL 2: the first GREEN candle (close > open) among bars 1-3 after
    signal 1. Sets the breakout trigger at THIS candle's LOW.
  * ENTRY: real price breaking below the signal-2 candle's low -- checked
    on bars AFTER it completed (this port's closed-bar update() always
    checks a PRIOR-armed trigger before running new pattern detection, so
    a signal 2 that completes THIS bar can't trigger on its own bar; the
    source Pine script sidesteps the same risk differently, via its default
    order-processing timing rather than statement ordering).
  * SL = the signal-1 (anchor) candle's high, fixed for the trade.
  * TARGET = entry - 2 x (SL - entry) -- a 2R target ("1:2" in the source
    comment). target_rr <= 0 disables this BTC-price target entirely (no
    "TARGET" exit reason will ever fire) -- same off-switch convention as
    dcv2's target_rr -- for callers that want to price a profit exit off
    the OPTION PREMIUM decaying instead (handled in the backtest/live layer).
  * No pre-entry SL-side invalidation in the source script (unlike this
    repo's other strategies) -- ported faithfully, i.e. NOT added here.

BUY side (mirror, added on request -- exact opposite of every rule above):
  signal 1 = open < EMA and close > EMA; wait for a RED candle (not closing
  below the EMA); trigger = that candle's HIGH; SL = signal-1 candle's LOW;
  target = entry + 2 x (entry - SL).

ema_sl mode (added on request -- an alternate, dynamic SL for BOTH sides):
  entry/pattern detection is unchanged. Once in a position, the fixed
  anchor-price SL is ignored; instead the trade exits the moment a candle
  CLOSES back through the EMA -- a LONG (PE sold) exits on close < EMA, a
  SHORT (CE sold) exits on close > EMA (mirror), exit price = that candle's
  close. This replaces "SL" as a concept (a BTC price level) with "close
  back through the trend line" (a BTC price ACTION) -- there's no fixed
  stop price at entry time in this mode, only a standing condition checked
  every bar. Meant to pair with target_rr=0 and a premium-decay TP (the
  backtest/live layer's own tp_decay_pct) so profit-taking happens off the
  OPTION premium while the stop stays anchored to BTC trend structure.

Execution mapping (same convention as every other strategy in this repo):
SELL signal -> sell a CALL (CE); BUY signal -> sell a PUT (PE). "Strict
single trade" is preserved for the COMBINED strategy, not per side: only one
of a short or a long position is ever open at once (if/elif between them),
matching the source script's own single-position design -- this is
deliberately NOT the independent-legs model used by supertrend_fixed_sl.
trade_ce/trade_pe let a caller run either side alone (e.g. to test the BUY
side in isolation).

entry_start_hour/entry_end_hour: gate the TRADE ITSELF (not pattern
detection -- signal1/signal2 still form/re-arm regardless) to a daily IST
time window, e.g. entry_start_hour=14, entry_end_hour=17 for "only enter
between 2pm and 5pm". Same convention as dcv2's --entry-start-hour/--entry-
end-hour: checked at the trigger moment, alongside trend_filter/
ema200_filter -- a breakout outside the window is consumed untraded rather
than held open waiting for the window to start. Default (0, 24) = no
restriction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..enums import PositionState
from ..models import Candle

__all__ = ["Ema21BreakdownStrategy", "Ema21BreakdownDecision"]


class _Ema:
    def __init__(self, length: int) -> None:
        self._alpha = 2.0 / (length + 1.0)
        self._value: float | None = None

    def update(self, x: float) -> float:
        self._value = x if self._value is None else self._alpha * x + (1 - self._alpha) * self._value
        return self._value


@dataclass(frozen=True)
class Ema21BreakdownDecision:
    candle: Candle
    short_exit: bool
    long_exit: bool
    short_exit_price: float
    long_exit_price: float
    exit_reason: str | None   # "SL" | "TARGET" | None -- whichever side exited this bar
    sell_signal: bool         # bearish -> sell CE
    buy_signal: bool          # bullish -> sell PE
    entry_price: float
    sl_level: float | None    # the SL active right after a just-opened position

    @property
    def has_exit(self) -> bool:
        return self.short_exit or self.long_exit

    @property
    def has_entry(self) -> bool:
        return self.sell_signal or self.buy_signal


class Ema21BreakdownStrategy:
    def __init__(
        self,
        *,
        ema_len: int = 21,
        max_wait: int = 3,
        target_rr: float = 2.0,
        trade_ce: bool = True,
        trade_pe: bool = True,
        trend_filter: bool = False,
        ema200_filter: bool = False,
        ema50_len: int = 50,
        ema200_len: int = 200,
        ema_sl: bool = False,
        entry_start_hour: int = 0,
        entry_end_hour: int = 24,
        day_tz: str = "Asia/Kolkata",
    ) -> None:
        self.ema_len = ema_len
        self.max_wait = max_wait
        self.target_rr = target_rr
        self.trade_ce = trade_ce
        self.trade_pe = trade_pe
        # ema_sl: replace the fixed anchor-price SL with a dynamic one --
        # exit the moment a candle CLOSES back through the EMA (long: close
        # < ema; short: close > ema), instead of price crossing a frozen
        # level. See the module docstring's "ema_sl mode" section.
        self.ema_sl = ema_sl
        # entry_start_hour/entry_end_hour: gate the trade itself to a daily
        # IST window -- see module docstring.
        self.entry_start_hour = entry_start_hour
        self.entry_end_hour = entry_end_hour
        self._tz = ZoneInfo(day_tz)
        # trend_filter: an ADDITIONAL gate on top of the signal1/signal2
        # pattern, checked AT THE TRIGGER (the moment real price would
        # otherwise fire the entry) -- BUY only fires if close > EMA(50) >
        # EMA(200) (price above a rising-stacked trend); SELL only if
        # close < EMA(50) < EMA(200) (mirror). Disagreeing consumes the
        # pending setup untraded, same convention as dcv2's
        # require_ema_direction_at_trigger. Off (default) = unchanged
        # behavior; the pattern alone decides.
        self.trend_filter = trend_filter
        # ema200_filter: a LIGHTER single-EMA version of the same idea --
        # SELL only fires if close < EMA(200) (EMA(50) is irrelevant here);
        # BUY only if close > EMA(200). Independent of trend_filter -- can
        # be used alone or combined with it (both must agree if both are on).
        self.ema200_filter = ema200_filter
        self.ema50_len = ema50_len
        self.ema200_len = ema200_len
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._ema = _Ema(self.ema_len)
        self._ema50 = _Ema(self.ema50_len)
        self._ema200 = _Ema(self.ema200_len)
        self._warmup_bars = 0

        # SELL/bear side.
        self._tracking = False
        self._bars_since_signal1 = 0
        self._anchor_high: float | None = None
        self._wait_for_breakout = False
        self._pending_trigger: float | None = None   # signal-2 candle's low
        self._pending_sl: float | None = None          # signal-1 candle's high
        self._in_short = False
        self._active_sl: float | None = None
        self._active_target: float | None = None

        # BUY/bull side (mirror).
        self._tracking_bull = False
        self._bars_since_signal1_bull = 0
        self._anchor_low: float | None = None
        self._wait_for_breakout_bull = False
        self._pending_trigger_bull: float | None = None   # signal-2 candle's high
        self._pending_sl_bull: float | None = None           # signal-1 candle's low
        self._in_long = False
        self._active_sl_bull: float | None = None
        self._active_target_bull: float | None = None

    @property
    def ready(self) -> bool:
        need = max(self.ema_len, self.ema200_len) if (self.trend_filter or self.ema200_filter) else self.ema_len
        return self._warmup_bars >= need

    @property
    def position_state(self) -> PositionState:
        if self._in_short:
            return PositionState.SHORT
        if self._in_long:
            return PositionState.LONG
        return PositionState.FLAT

    @property
    def has_pending(self) -> bool:
        return self._pending_trigger is not None or self._pending_trigger_bull is not None

    def force_flat(self) -> None:
        self._in_short = self._in_long = False
        self._active_sl = self._active_target = None
        self._active_sl_bull = self._active_target_bull = None

    def debug_state(self) -> dict:
        def r(x):
            return round(x, 2) if isinstance(x, (int, float)) else None
        return {
            "tracking": self._tracking, "bars_since_signal1": self._bars_since_signal1,
            "anchor_high": r(self._anchor_high), "wait_for_breakout": self._wait_for_breakout,
            "pending_trigger": r(self._pending_trigger), "pending_sl": r(self._pending_sl),
            "active_sl": r(self._active_sl), "active_target": r(self._active_target),
            "tracking_bull": self._tracking_bull, "bars_since_signal1_bull": self._bars_since_signal1_bull,
            "anchor_low": r(self._anchor_low), "wait_for_breakout_bull": self._wait_for_breakout_bull,
            "pending_trigger_bull": r(self._pending_trigger_bull), "pending_sl_bull": r(self._pending_sl_bull),
            "active_sl_bull": r(self._active_sl_bull), "active_target_bull": r(self._active_target_bull),
            "pos": self.position_state.name,
        }

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> Ema21BreakdownDecision | None:
        ema = self._ema.update(candle.close)
        ema50 = self._ema50.update(candle.close)
        ema200 = self._ema200.update(candle.close)
        self._warmup_bars += 1
        is_green = candle.close > candle.open
        is_red = candle.close < candle.open

        short_exit = long_exit = False
        short_exit_price = long_exit_price = candle.close
        exit_reason: str | None = None
        sell_signal = buy_signal = False
        entry_price = candle.close
        new_sl: float | None = None

        # --- 1. Exit: SL / TARGET bracket per side (SL checked first, same
        #     conservative convention as every other strategy here). SL is
        #     either the fixed anchor price (default) or, with ema_sl on, a
        #     dynamic "closed back through the EMA" condition -- see the
        #     module docstring's "ema_sl mode" section. ---
        if self._in_short:
            if self.ema_sl:
                if candle.close > ema:
                    short_exit, short_exit_price, exit_reason = True, candle.close, "SL"
            elif self._active_sl is not None and candle.high >= self._active_sl:
                short_exit, short_exit_price, exit_reason = True, self._active_sl, "SL"
            if not short_exit and self._active_target is not None and candle.low <= self._active_target:
                short_exit, short_exit_price, exit_reason = True, self._active_target, "TARGET"
            if short_exit:
                self._in_short = False
                self._active_sl = self._active_target = None
        elif self._in_long:
            if self.ema_sl:
                if candle.close < ema:
                    long_exit, long_exit_price, exit_reason = True, candle.close, "SL"
            elif self._active_sl_bull is not None and candle.low <= self._active_sl_bull:
                long_exit, long_exit_price, exit_reason = True, self._active_sl_bull, "SL"
            if not long_exit and self._active_target_bull is not None and candle.high >= self._active_target_bull:
                long_exit, long_exit_price, exit_reason = True, self._active_target_bull, "TARGET"
            if long_exit:
                self._in_long = False
                self._active_sl_bull = self._active_target_bull = None

        # --- 2. Entry: real-price breakout of a PRIOR-armed pending trigger
        #     -- runs BEFORE pattern detection below. if/elif between short
        #     and long preserves "strict single trade" (never both at once). ---
        flat = not self._in_short and not self._in_long
        local_hour = datetime.fromtimestamp(candle.start_time, tz=self._tz).hour
        time_ok = self.entry_start_hour <= local_hour < self.entry_end_hour
        if (flat and self.trade_ce and self._pending_trigger is not None
                and candle.low <= self._pending_trigger):
            # trend_filter: price < EMA50 < EMA200 must ALSO hold at the
            # trigger moment; ema200_filter: price < EMA200 alone.
            # entry_start_hour/entry_end_hour: the trigger bar's own IST hour
            # must also fall in the allowed window. All independent gates,
            # disagreeing consumes the setup untraded.
            trend_ok = ((not self.trend_filter) or (candle.close < ema50 < ema200)) \
                and ((not self.ema200_filter) or (candle.close < ema200)) \
                and time_ok
            if trend_ok:
                sell_signal, entry_price = True, self._pending_trigger
                self._in_short = True
                self._active_sl = self._pending_sl
                self._active_target = (entry_price - self.target_rr * (self._pending_sl - entry_price)
                                        if self.target_rr > 0 else None)
                new_sl = self._active_sl
            self._pending_trigger = self._pending_sl = None
            self._wait_for_breakout = False
            self._tracking = False
        elif (flat and self.trade_pe and self._pending_trigger_bull is not None
                and candle.high >= self._pending_trigger_bull):
            # trend_filter / ema200_filter / entry-window mirror.
            trend_ok = ((not self.trend_filter) or (candle.close > ema50 > ema200)) \
                and ((not self.ema200_filter) or (candle.close > ema200)) \
                and time_ok
            if trend_ok:
                buy_signal, entry_price = True, self._pending_trigger_bull
                self._in_long = True
                self._active_sl_bull = self._pending_sl_bull
                self._active_target_bull = (entry_price + self.target_rr * (entry_price - self._pending_sl_bull)
                                             if self.target_rr > 0 else None)
                new_sl = self._active_sl_bull
            self._pending_trigger_bull = self._pending_sl_bull = None
            self._wait_for_breakout_bull = False
            self._tracking_bull = False

        # --- 3. Pattern detection (signal 1 -> signal 2), only while flat. ---
        flat_now = not self._in_short and not self._in_long and not sell_signal and not buy_signal

        if flat_now and self.trade_ce and self.ready:
            signal1_raw = candle.open > ema and candle.close < ema
            if signal1_raw and not self._wait_for_breakout:
                self._tracking = True
                self._bars_since_signal1 = 0
                self._anchor_high = candle.high
            elif self._tracking:
                self._bars_since_signal1 += 1

            if candle.close > ema:
                self._tracking = False
                if self._wait_for_breakout:
                    self._pending_trigger = self._pending_sl = None
                self._wait_for_breakout = False

            if self._tracking and not self._wait_for_breakout and self._bars_since_signal1 > self.max_wait:
                self._tracking = False

            if (self._tracking and not self._wait_for_breakout
                    and 1 <= self._bars_since_signal1 <= self.max_wait and is_green):
                self._wait_for_breakout = True
                self._pending_trigger = candle.low
                self._pending_sl = self._anchor_high

        if flat_now and self.trade_pe and self.ready:
            signal1_raw_bull = candle.open < ema and candle.close > ema
            if signal1_raw_bull and not self._wait_for_breakout_bull:
                self._tracking_bull = True
                self._bars_since_signal1_bull = 0
                self._anchor_low = candle.low
            elif self._tracking_bull:
                self._bars_since_signal1_bull += 1

            if candle.close < ema:
                self._tracking_bull = False
                if self._wait_for_breakout_bull:
                    self._pending_trigger_bull = self._pending_sl_bull = None
                self._wait_for_breakout_bull = False

            if (self._tracking_bull and not self._wait_for_breakout_bull
                    and self._bars_since_signal1_bull > self.max_wait):
                self._tracking_bull = False

            if (self._tracking_bull and not self._wait_for_breakout_bull
                    and 1 <= self._bars_since_signal1_bull <= self.max_wait and is_red):
                self._wait_for_breakout_bull = True
                self._pending_trigger_bull = candle.high
                self._pending_sl_bull = self._anchor_low

        if not (short_exit or long_exit or sell_signal or buy_signal):
            return None
        return Ema21BreakdownDecision(
            candle=candle, short_exit=short_exit, long_exit=long_exit,
            short_exit_price=short_exit_price, long_exit_price=long_exit_price,
            exit_reason=exit_reason, sell_signal=sell_signal, buy_signal=buy_signal,
            entry_price=entry_price, sl_level=new_sl,
        )
