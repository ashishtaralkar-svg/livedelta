"""RangeEngulfingFadeSellStrategy -- Python port of the SELL-ONLY,
INTRACANDLE variant of range_engulfing_fade_strategy.pine.

Unlike RangeEngulfingFadeStrategy (this repo's earlier closed-bar port),
this class matches the .pine chart's actual ASAP/intracandle stop-order
mechanics exactly: once a pattern arms a trigger, entry fires the instant
a LATER candle's HIGH reaches that trigger (a resting stop order), not
only once that candle's CLOSE clears it. Sell-only (on request): only the
bearish range-engulfing fade side exists here -- a red candle followed by
a green candle whose range fully engulfs it -- there is no long/buy side.

PATTERN:
  * Two consecutive CLOSED candles: RED then GREEN (arming itself still
    needs a full candle close, since it reads that candle's own close/
    range -- same as the .pine chart, whose bullEngulf also only
    evaluates once per closed bar).
  * GREEN's range fully engulfs RED's: green.low < red.low and
    green.high > red.high.
  * CONFIRMATION: green.close > red.open (a real body move back above
    where the red candle started, not just a wick).
  * ARMED TRIGGER = green.high. ARMED TARGET = green.low (the pattern's
    own base). ARMED SL (not specified in the original request -- filled
    in using the pattern's own height as the risk unit, 1:1 risk:reward
    by construction, per request): trigger + (green.high - green.low),
    the SAME distance above trigger that the target sits below it.
  * ENTRY is INTRACANDLE: the trigger becomes live starting the very next
    candle. The instant that candle's HIGH reaches the trigger, entry
    fires -- fill price is the trigger level itself (a resting stop order
    fills at its own level, not the candle's actual high), matching how
    SL/TARGET exits below fill at their own level rather than the exact
    touch price.
  * ONE-CANDLE WINDOW: the armed trigger is only checked against the
    single candle immediately following the one that armed it. If that
    candle's high doesn't reach the trigger, the setup expires -- it
    never stays pending waiting for a later breakout. A fresh pattern on
    that same candle can still arm a brand-new setup of its own.
  * SL/TARGET exits use the candle's HIGH/LOW (a touch anywhere in the
    candle counts, same intrabar-touch convention as every strategy in
    this repo) -- SL checked first, then TARGET.
  * Strict single trade: no pyramiding while a short is already open.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Candle

__all__ = ["RangeEngulfingFadeSellStrategy", "RangeEngulfingFadeSellDecision"]


@dataclass(frozen=True)
class RangeEngulfingFadeSellDecision:
    candle: Candle
    exit: bool
    exit_price: float
    exit_reason: str | None   # "SL" | "TARGET" | None
    sell_signal: bool          # breakout -> sell CE
    entry_price: float
    sl_level: float | None     # the SL just set for a newly-opened leg

    @property
    def has_exit(self) -> bool:
        return self.exit

    @property
    def has_entry(self) -> bool:
        return self.sell_signal


class RangeEngulfingFadeSellStrategy:
    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._prev_candle: Candle | None = None

        self._pending_trigger: float | None = None   # green candle's high
        self._pending_sl: float | None = None
        self._pending_target: float | None = None

        self._in_short = False
        self._active_sl: float | None = None
        self._active_target: float | None = None

    @property
    def in_short(self) -> bool:
        return self._in_short

    def force_flat(self) -> None:
        self._in_short = False
        self._active_sl = self._active_target = None

    def debug_state(self) -> dict:
        def r(x):
            return round(x, 2) if isinstance(x, (int, float)) else None
        return {
            "pending_trigger": r(self._pending_trigger),
            "in_short": self._in_short,
            "active_sl": r(self._active_sl), "active_target": r(self._active_target),
        }

    # ------------------------------------------------------------------ #
    # Live/intracandle API -- used by the live engine, which calls these
    # directly on every real-time price tick (NOT through update(), which
    # is the backtest's bar-by-bar replay entry point and assumes exactly
    # one call per candle). Each of these is safe to call many times per
    # candle: they mutate state the INSTANT a condition is detected, so a
    # later call within the same tick-storm is a no-op by construction
    # (e.g. once in_short flips True, check_intracandle_entry's own guard
    # makes every subsequent call in the same forming candle inert).
    # ------------------------------------------------------------------ #
    def check_intracandle_entry(self, candle: Candle) -> tuple[float, float, float] | None:
        """Call on every forming-candle tick while flat. Fires the instant
        ``candle.high`` reaches a pending trigger armed by a PRIOR closed
        candle -- matching the .pine chart's resting stop order, which
        fills the moment price crosses, not at the candle's close. Returns
        ``(entry_price, sl, target)`` on fire, else ``None``."""
        if self._in_short or self._pending_trigger is None:
            return None
        if candle.high < self._pending_trigger:
            return None
        entry_price = self._pending_trigger
        sl, target = self._pending_sl, self._pending_target
        self._in_short = True
        self._active_sl = sl
        self._active_target = target
        self._pending_trigger = self._pending_sl = self._pending_target = None
        return entry_price, sl, target

    def check_intracandle_exit(self, candle: Candle) -> tuple[str, float] | None:
        """Call on every forming-candle tick while short. SL checked
        first, same conservative convention as everywhere else in this
        repo. Returns ``(reason, exit_price)`` on fire, else ``None``."""
        if not self._in_short:
            return None
        if self._active_sl is not None and candle.high >= self._active_sl:
            price = self._active_sl
            self._in_short = False
            self._active_sl = self._active_target = None
            return "SL", price
        if self._active_target is not None and candle.low <= self._active_target:
            price = self._active_target
            self._in_short = False
            self._active_sl = self._active_target = None
            return "TARGET", price
        return None

    def arm_from_closed_candle(self, candle: Candle) -> None:
        """Call exactly ONCE per CLOSED candle (the live counterpart of
        update()'s steps 2.5+3, with entry/exit split out to the two
        methods above since those fire continuously in real time, not
        just at candle close). Expires a still-pending trigger that
        wasn't consumed by check_intracandle_entry during this candle's
        life, then checks whether this candle forms a fresh red->green
        pattern to arm for the NEXT candle."""
        prev = self._prev_candle
        flat = not self._in_short

        if flat and self._pending_trigger is not None:
            self._pending_trigger = self._pending_sl = self._pending_target = None

        if flat and prev is not None:
            prev_red = prev.close < prev.open
            is_green = candle.close > candle.open
            if prev_red and is_green:
                bull_engulf = (candle.low < prev.low and candle.high > prev.high
                               and candle.close > prev.open)
                if bull_engulf:
                    pattern_height = candle.high - candle.low
                    self._pending_trigger = candle.high
                    self._pending_sl = candle.high + pattern_height   # 1:1 R:R
                    self._pending_target = candle.low

        self._prev_candle = candle

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> RangeEngulfingFadeSellDecision | None:
        prev = self._prev_candle

        exit_ = False
        exit_price = candle.close
        exit_reason: str | None = None
        sell_signal = False
        entry_price = candle.close
        new_sl: float | None = None

        # --- 1. Exit: intrabar touch of the frozen SL/TARGET. SL first,
        #     same conservative convention as every other strategy here. ---
        if self._in_short:
            if self._active_sl is not None and candle.high >= self._active_sl:
                exit_, exit_price, exit_reason = True, self._active_sl, "SL"
            elif self._active_target is not None and candle.low <= self._active_target:
                exit_, exit_price, exit_reason = True, self._active_target, "TARGET"
            if exit_:
                self._in_short = False
                self._active_sl = self._active_target = None

        # --- 2. Entry: this candle IS the one-candle window for a trigger
        #     armed on the PREVIOUS candle -- an intrabar HIGH touch fires
        #     it immediately (fill = the trigger level itself), matching
        #     the .pine chart's resting stop order. ---
        flat = not self._in_short
        if (flat and self._pending_trigger is not None
                and candle.high >= self._pending_trigger):
            sell_signal, entry_price = True, self._pending_trigger
            self._in_short = True
            self._active_sl = new_sl = self._pending_sl
            self._active_target = self._pending_target
            self._pending_trigger = self._pending_sl = self._pending_target = None

        # --- 2.5. Expire: this candle's one-candle window has now closed
        #     without a fill -- the setup is gone (only while `flat`, i.e.
        #     this candle was actually eligible to trade it). ---
        if flat and self._pending_trigger is not None:
            self._pending_trigger = self._pending_sl = self._pending_target = None

        # --- 3. Fresh pattern detection, only while flat and not just
        #     entered. By this point any older pending trigger has always
        #     already been cleared (fired or expired above), so this
        #     always arms into a clean slate. The new trigger becomes live
        #     starting the very next candle. ---
        flat_now = not self._in_short and not sell_signal

        if flat_now and prev is not None:
            prev_red = prev.close < prev.open
            is_green = candle.close > candle.open

            if prev_red and is_green:
                bull_engulf = (candle.low < prev.low and candle.high > prev.high
                               and candle.close > prev.open)
                if bull_engulf:
                    pattern_height = candle.high - candle.low
                    self._pending_trigger = candle.high
                    self._pending_sl = candle.high + pattern_height   # 1:1 R:R
                    self._pending_target = candle.low

        self._prev_candle = candle

        if not (exit_ or sell_signal):
            return None
        return RangeEngulfingFadeSellDecision(
            candle=candle, exit=exit_, exit_price=exit_price, exit_reason=exit_reason,
            sell_signal=sell_signal, entry_price=entry_price, sl_level=new_sl,
        )
