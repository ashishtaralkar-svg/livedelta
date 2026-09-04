"""RangeEngulfingFadeStrategy -- Python port of range_engulfing_fade_strategy.pine.

A MEAN-REVERSION/FADE strategy, not a continuation-breakout one: the
"confirmation" candle is itself a bullish/bearish signal, and this strategy
bets that a further push past its own extreme is exhaustion, not
follow-through. Option-SELL execution (per the request): the bearish/short
side sells a CALL, the bullish/long side sells a PUT (this repo's standard
sell-mode convention).

PATTERN (bearish/short-fade side; bullish/long-fade is the exact mirror):
  * Two consecutive CLOSED candles: RED then GREEN.
  * The GREEN candle's RANGE fully engulfs the RED candle's range:
    green low < red low AND green high > red high.
  * CONFIRMATION: green candle's close > red candle's open (a real body
    move back above where the red candle started, not just a wick).
  * ARMED TRIGGER = the green candle's HIGH. ARMED TARGET = the green
    candle's LOW (the pattern's own base) -- the reward leg is the full
    pattern height. ARMED SL (not specified in the original request --
    filled in using the pattern's own height as the risk unit): trigger +
    (green high - green low) / 2, i.e. HALF the distance the target sits
    below trigger, giving a 1:2 risk:reward by construction (updated from
    an earlier 1:1 default on request).
  * ENTRY (CLOSED-BAR, unlike the .pine chart's intracandle stop-order
    version -- this Python port follows this repo's own established
    convention for every other strategy built this session): once armed,
    on any later CLOSED candle where close > armed trigger, SELL the CALL.
  * ONE-CANDLE WINDOW: once armed, the setup is only checked against the
    single candle immediately following the one that armed it. If that
    candle's close doesn't clear the trigger, the setup expires -- it
    never stays pending waiting for some later breakout. A fresh pattern
    on that same candle can still arm a brand-new setup of its own
    (matching the .pine port's stop-order cancel-after-one-candle
    behavior).
  * Strict single trade: only one of a short or a long is ever open.

Execution mapping (same convention as every other strategy here): the
short/bearish-fade side sells a CALL; the long/bullish-fade side sells a
PUT.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Candle

__all__ = ["RangeEngulfingFadeStrategy", "RangeEngulfingFadeDecision"]


@dataclass(frozen=True)
class RangeEngulfingFadeDecision:
    candle: Candle
    short_exit: bool   # CE leg closed (SL or TARGET)
    long_exit: bool    # PE leg closed (SL or TARGET)
    short_exit_price: float
    long_exit_price: float
    short_exit_reason: str | None   # "SL" | "TARGET" | None
    long_exit_reason: str | None    # "SL" | "TARGET" | None
    sell_signal: bool  # bearish-fade breakout -> sell CE
    buy_signal: bool   # bullish-fade breakdown -> sell PE
    entry_price: float
    sl_level: float | None   # the SL just set for a newly-opened leg

    @property
    def has_exit(self) -> bool:
        return self.short_exit or self.long_exit

    @property
    def has_entry(self) -> bool:
        return self.sell_signal or self.buy_signal


class RangeEngulfingFadeStrategy:
    def __init__(self, *, trade_ce: bool = True, trade_pe: bool = True) -> None:
        # trade_ce/trade_pe: which side(s) are allowed to fire -- a
        # completed pattern on a disabled side is simply never armed.
        self.trade_ce = trade_ce
        self.trade_pe = trade_pe
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._prev_candle: Candle | None = None

        self._pending_short_trigger: float | None = None   # green candle's high
        self._pending_short_sl: float | None = None
        self._pending_short_target: float | None = None

        self._pending_long_trigger: float | None = None    # red candle's low
        self._pending_long_sl: float | None = None
        self._pending_long_target: float | None = None

        self._in_short = False
        self._in_long = False
        self._active_short_sl: float | None = None
        self._active_short_target: float | None = None
        self._active_long_sl: float | None = None
        self._active_long_target: float | None = None

    @property
    def in_short(self) -> bool:
        return self._in_short

    @property
    def in_long(self) -> bool:
        return self._in_long

    def force_flat(self) -> None:
        self._in_short = self._in_long = False
        self._active_short_sl = self._active_short_target = None
        self._active_long_sl = self._active_long_target = None

    def debug_state(self) -> dict:
        def r(x):
            return round(x, 2) if isinstance(x, (int, float)) else None
        return {
            "pending_short_trigger": r(self._pending_short_trigger),
            "pending_long_trigger": r(self._pending_long_trigger),
            "in_short": self._in_short, "in_long": self._in_long,
            "active_short_sl": r(self._active_short_sl), "active_short_target": r(self._active_short_target),
            "active_long_sl": r(self._active_long_sl), "active_long_target": r(self._active_long_target),
        }

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> RangeEngulfingFadeDecision | None:
        prev = self._prev_candle

        short_exit = long_exit = False
        short_exit_price = long_exit_price = candle.close
        short_exit_reason: str | None = None
        long_exit_reason: str | None = None
        sell_signal = buy_signal = False
        entry_price = candle.close
        new_sl: float | None = None

        # --- 1. Exits: real price crossing the FROZEN SL or TARGET
        #     (independent legs, but only one is ever open -- strict
        #     single trade). SL checked first, same conservative
        #     convention as every other strategy here. ---
        if self._in_short:
            if self._active_short_sl is not None and candle.high >= self._active_short_sl:
                short_exit, short_exit_price, short_exit_reason = True, self._active_short_sl, "SL"
            elif self._active_short_target is not None and candle.low <= self._active_short_target:
                short_exit, short_exit_price, short_exit_reason = True, self._active_short_target, "TARGET"
            if short_exit:
                self._in_short = False
                self._active_short_sl = self._active_short_target = None
        elif self._in_long:
            if self._active_long_sl is not None and candle.low <= self._active_long_sl:
                long_exit, long_exit_price, long_exit_reason = True, self._active_long_sl, "SL"
            elif self._active_long_target is not None and candle.high >= self._active_long_target:
                long_exit, long_exit_price, long_exit_reason = True, self._active_long_target, "TARGET"
            if long_exit:
                self._in_long = False
                self._active_long_sl = self._active_long_target = None

        # --- 2. Entry: real-price breakout of a PRIOR-armed level -- runs
        #     BEFORE fresh pattern detection below. Strict single trade
        #     (if/elif). ---
        flat = not self._in_short and not self._in_long
        if (flat and self.trade_ce and self._pending_short_trigger is not None
                and candle.close > self._pending_short_trigger):
            sell_signal, entry_price = True, self._pending_short_trigger
            self._in_short = True
            self._active_short_sl = new_sl = self._pending_short_sl
            self._active_short_target = self._pending_short_target
            self._pending_short_trigger = self._pending_short_sl = self._pending_short_target = None
        elif (flat and self.trade_pe and self._pending_long_trigger is not None
                and candle.close < self._pending_long_trigger):
            buy_signal, entry_price = True, self._pending_long_trigger
            self._in_long = True
            self._active_long_sl = new_sl = self._pending_long_sl
            self._active_long_target = self._pending_long_target
            self._pending_long_trigger = self._pending_long_sl = self._pending_long_target = None

        # --- 2.5. Expire a still-pending trigger that didn't fire on the
        #     one candle immediately following its arming -- a setup only
        #     ever gets that single candle's chance (only while `flat`,
        #     i.e. this candle was actually eligible to trade it; if
        #     blocked by an open position it hasn't had its chance yet). ---
        if flat and self._pending_short_trigger is not None:
            self._pending_short_trigger = self._pending_short_sl = self._pending_short_target = None
        if flat and self._pending_long_trigger is not None:
            self._pending_long_trigger = self._pending_long_sl = self._pending_long_target = None

        # --- 3. Fresh pattern detection, only while flat and not just
        #     entered. By this point any older pending trigger has always
        #     already been cleared (fired or expired above), so this
        #     always arms into a clean slate. ---
        flat_now = not self._in_short and not self._in_long and not sell_signal and not buy_signal

        if flat_now and prev is not None:
            prev_red = prev.close < prev.open
            prev_green = prev.close > prev.open
            is_red = candle.close < candle.open
            is_green = candle.close > candle.open

            if self.trade_ce and prev_red and is_green:
                bull_engulf = (candle.low < prev.low and candle.high > prev.high
                               and candle.close > prev.open)
                if bull_engulf:
                    pattern_height = candle.high - candle.low
                    self._pending_short_trigger = candle.high
                    self._pending_short_sl = candle.high + pattern_height / 2
                    self._pending_short_target = candle.low

            if self.trade_pe and prev_green and is_red:
                bear_engulf = (candle.low < prev.low and candle.high > prev.high
                               and candle.close < prev.open)
                if bear_engulf:
                    pattern_height = candle.high - candle.low
                    self._pending_long_trigger = candle.low
                    self._pending_long_sl = candle.low - pattern_height / 2
                    self._pending_long_target = candle.high

        self._prev_candle = candle

        if not (short_exit or long_exit or sell_signal or buy_signal):
            return None
        return RangeEngulfingFadeDecision(
            candle=candle, short_exit=short_exit, long_exit=long_exit,
            short_exit_price=short_exit_price, long_exit_price=long_exit_price,
            short_exit_reason=short_exit_reason, long_exit_reason=long_exit_reason,
            sell_signal=sell_signal, buy_signal=buy_signal, entry_price=entry_price,
            sl_level=new_sl,
        )
