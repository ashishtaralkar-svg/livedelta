"""Supertrend15mFilterFixedSlStrategy -- Python port of
supertrend_15m_filter_fixed_sl.pine.

Pure BTC-price signal logic: a fresh 1-minute Supertrend flip, confirmed by
the 15-minute Supertrend already agreeing at that same moment, arms a
position with its SL FROZEN at the 1-minute Supertrend's own value on that
flip candle -- it does not trail. The instant real price crosses that
frozen level, the position closes; the strategy then waits for the NEXT
fresh 1-minute flip (with 15-minute agreement) before re-entering. No
auto-reverse on a stop-out (unlike supertrend_sar.py's stop-and-reverse) --
this was described as a stop, not a reversal trigger. No profit target, no
end-of-day square-off -- neither was described, so neither is implemented
(the backtest/live engine is responsible for anything beyond this, same
separation of concerns as every other strategy in this package).

RULES:
  * PRIMARY Supertrend(10,3) on the engine's own candles (intended
    1-minute) decides direction: red (downtrend) -> SELL-eligible, green
    (uptrend) -> BUY-eligible.
  * 15-MINUTE Supertrend(10,3) confirmation filter, entry only -- must
    already agree at the exact moment the 1-minute flips (both red for a
    sell, both green for a buy). Same deliberate LOCKED (not live-
    repainting) divergence from the Pine script as supertrend_flip.py's own
    15m filter: this port locks the 15m Supertrend's value at each
    15-minute boundary and holds it constant until the next one completes,
    rather than Pine's request.security() repainting on the still-forming
    15m bar.
  * FRESH FLIP REQUIRED: entry only fires on the bar the PRIMARY Supertrend
    itself just changed direction -- if the 15m disagrees right at that
    moment, this flip is simply skipped (no "wait for 15m to catch up"
    mechanic like supertrend_flip.py's v5/v10).
  * SL: frozen at the PRIMARY Supertrend's own value on the flip/entry
    candle -- a genuine BTC PRICE LEVEL (unlike supertrend_flip.py, whose
    exit is a pure trend-change event with no price level at all). The
    instant real price crosses it, the position closes at that level.
  * No reversal on a stop-out -- goes flat and waits for the next fresh
    flip (which could be either direction).

Strict single position -- never a CE and PE open at once; a same-bar
close-then-reopen can still happen if a stop-out and a qualifying fresh
flip land on the exact same candle (checked in that order, matching every
other flip/reversal strategy in this package).

OPT-IN HEIKIN ASHI MODE (use_heikin_ashi=False by default -- the LIVE
st15fbot config, unchanged): when enabled, BOTH Supertrends (1-minute
primary and 15-minute confirmation) run on Heikin Ashi open/high/low/close
instead of real OHLC, mirroring supertrend_flip.py's own v11 HA conversion
and the same "each timeframe gets its OWN independent recursive HA state"
rule (the 15m HA series is a fresh HA conversion of the real 15-minute
aggregated bar, NOT a rollup of the primary's own HA candles). The frozen
SL crossing check ALSO uses HA high/low in this mode (consistent with
supertrend_flip.py's breakout-level checks also being HA-based) -- i.e. in
HA mode this strategy behaves exactly as if it were running on a genuine
Heikin Ashi chart end to end. The one thing that stays REAL regardless:
entry_price is always candle.close (never HA close), mirroring how Pine's
own strategy fills always use real market price even on an HA-displayed
chart -- the caller needs real price to resolve option strikes/premiums.
exit_price is unaffected either way: it was already the frozen SL LEVEL
itself (a literal stop price the caller uses to close the option), not a
candle close, so there's no separate "real vs HA" version of it to choose
between.

OPT-IN SECOND EXIT: 15m-FLIP-AGAINST (exit_on_15m_flip_against=False by
default -- the LIVE st15fbot config, unchanged): when enabled, a position
now closes on whichever of TWO independent triggers fires first --
(1) the existing frozen 1m-SL being crossed by real price, or (2) the
15-minute confirmation Supertrend itself having a FRESH flip AGAINST the
position (a short's 15m turning green, or a long's 15m turning red) --
checked EVERY bar, immediately, regardless of where price sits relative to
the frozen SL. This is a pure trend-change exit (no price level involved),
so exit_price is candle.close, not the frozen SL -- same distinction
supertrend_flip.py already draws between its own price-level and
trend-change exits. The Decision's exit_reason field ("SL" vs "15M_FLIP")
tells the caller which of the two fired. SL is checked FIRST -- if both
would fire on the same bar, the price-level stop wins (matches the
existing convention that a same-bar stop-out can still be followed by a
fresh qualifying re-entry on that same bar).

OPT-IN THIRD CONFIRMATION LAYER: 1-HOUR (use_1h_filter=False by default --
the LIVE st15fbot config, unchanged): when enabled, an entry requires the
1-minute flip, the 15-minute filter, AND a 1-hour Supertrend to ALL agree
at that exact moment -- a third AND-gate stacked on top of the existing
pair, entry-only (it plays no role in exits, blocking, or anything else).
Uses its own independent bucket-aggregation state, deliberately
DUPLICATED from the 15-minute filter's own _bucket_id/_bucket_open/etc.
machinery rather than sharing it, so the already-tested 15-minute path
(many existing tests poke its internals directly) is completely
untouched by this addition. Same LOCKED (not live-repainting) bucket-
boundary pattern as the 15-minute filter, just a 3600-second bucket
instead of 900. Disabled by default: ready, is_red1h/is_green1h all
degrade to a no-op (always True) so the entry condition is byte-for-byte
identical to before this existed.

OPT-IN CLOSE-ONLY SL (sl_on_close_only=False by default -- the LIVE
st15fbot config, unchanged): the SL check already only ever RUNS once per
CLOSED 1-minute candle (this strategy has no intracandle/tick method at
all -- the live engine's own on_closed handler is the only caller). What
this flag changes is what that check compares against the frozen level:
by default it's the closed candle's high/low (a same-candle wick that
pokes through the frozen SL and closes back on the safe side still stops
the position out, since the wick did genuinely trade through the level).
With this enabled, the check instead compares the frozen SL only to that
candle's CLOSE price -- a wick through the level with a close back on the
safe side no longer exits; the position only stops out once a 1-minute
candle actually CLOSES beyond the frozen level. In HA mode the HA close is
used instead of the real close, consistent with the HA mode's existing
"crossing check uses the same series as the chart" rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Candle

__all__ = ["Supertrend15mFilterFixedSlStrategy", "Supertrend15mFilterFixedSlDecision"]


class _Rma:
    """Wilder's smoothed moving average (alpha=1/length)."""

    def __init__(self, length: int) -> None:
        self._alpha = 1.0 / length
        self._value: float | None = None

    def update(self, x: float) -> float:
        self._value = x if self._value is None else self._alpha * x + (1 - self._alpha) * self._value
        return self._value


class _Supertrend:
    """Standard ATR-based Supertrend. direction: -1 = uptrend (green), 1 =
    downtrend (red) -- same convention as every other Supertrend in this
    package and in the Pine scripts."""

    def __init__(self, atr_period: int, factor: float) -> None:
        self.factor = factor
        self._atr = _Rma(atr_period)
        self._prev_close: float | None = None
        self._final_upper: float | None = None
        self._final_lower: float | None = None
        self._value: float | None = None
        self._direction = 0
        self._bars_seen = 0

    def update(self, high: float, low: float, close: float) -> tuple[float, int]:
        self._bars_seen += 1
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
        elif self._value == self._final_upper:
            direction = 1 if close <= final_upper else -1
        else:
            direction = -1 if close >= final_lower else 1

        value = final_upper if direction == 1 else final_lower

        self._prev_close = close
        self._final_upper, self._final_lower = final_upper, final_lower
        self._value, self._direction = value, direction
        return value, direction


class _HeikinAshi:
    """Standard recursive Heikin Ashi conversion: ha_close = (o+h+l+c)/4,
    ha_open = (prev_ha_open + prev_ha_close)/2 (seeded with (o+c)/2 on the
    first bar), ha_high = max(h, ha_open, ha_close), ha_low = min(l,
    ha_open, ha_close) -- same formula as supertrend_flip.py's own
    conversion. Each instance holds its OWN independent recursive state,
    since HA is timeframe-specific (see module docstring)."""

    def __init__(self) -> None:
        self._ha_open: float | None = None
        self._ha_close: float | None = None

    def convert(self, o: float, h: float, low: float, c: float) -> tuple[float, float, float, float]:
        ha_close = (o + h + low + c) / 4.0
        ha_open = (o + c) / 2.0 if self._ha_open is None else (self._ha_open + self._ha_close) / 2.0
        ha_high = max(h, ha_open, ha_close)
        ha_low = min(low, ha_open, ha_close)
        self._ha_open, self._ha_close = ha_open, ha_close
        return ha_open, ha_high, ha_low, ha_close


@dataclass(frozen=True)
class Supertrend15mFilterFixedSlDecision:
    candle: Candle
    exit: bool
    exit_price: float          # frozen SL level for "SL", candle.close for "15M_FLIP" (no price level)
    exit_was_short: bool
    exit_reason: str | None    # "SL" or "15M_FLIP" (None when exit is False)
    entry_signal: bool
    entry_is_short: bool
    entry_price: float
    sl_level: float | None     # the newly-frozen SL for the leg that just opened

    @property
    def has_exit(self) -> bool:
        return self.exit

    @property
    def has_entry(self) -> bool:
        return self.entry_signal


class Supertrend15mFilterFixedSlStrategy:
    def __init__(
        self,
        *,
        atr_period: int = 10,
        factor: float = 3.0,
        atr_period_15m: int = 10,
        factor_15m: float = 3.0,
        bucket_seconds: int = 900,   # 15 minutes -- the confirmation timeframe's own bar width
        use_heikin_ashi: bool = False,   # see module docstring's OPT-IN HEIKIN ASHI MODE note
        exit_on_15m_flip_against: bool = False,   # see module docstring's OPT-IN SECOND EXIT note
        use_1h_filter: bool = False,   # see module docstring's OPT-IN THIRD CONFIRMATION LAYER note
        atr_period_1h: int = 10,
        factor_1h: float = 3.0,
        bucket_seconds_1h: int = 3600,   # 1 hour
        sl_on_close_only: bool = False,   # see module docstring's OPT-IN CLOSE-ONLY SL note
    ) -> None:
        self.atr_period = atr_period
        self.factor = factor
        self.atr_period_15m = atr_period_15m
        self.factor_15m = factor_15m
        self.bucket_seconds = bucket_seconds
        self.use_heikin_ashi = use_heikin_ashi
        self.exit_on_15m_flip_against = exit_on_15m_flip_against
        self.use_1h_filter = use_1h_filter
        self.atr_period_1h = atr_period_1h
        self.factor_1h = factor_1h
        self.bucket_seconds_1h = bucket_seconds_1h
        self.sl_on_close_only = sl_on_close_only
        self.reset()

    def reset(self) -> None:
        self._st = _Supertrend(self.atr_period, self.factor)
        self._st15 = _Supertrend(self.atr_period_15m, self.factor_15m)
        # Independent recursive HA state per timeframe -- see module docstring.
        self._ha1 = _HeikinAshi() if self.use_heikin_ashi else None
        self._ha15 = _HeikinAshi() if self.use_heikin_ashi else None
        self._prev_dir = 0
        self._is_short: bool | None = None   # None=flat, True=short/CE, False=long/PE
        self._active_sl: float | None = None
        # 15-minute bucket aggregation (locked-at-close, see module docstring)
        self._bucket_id: int | None = None
        self._bucket_open: float | None = None
        self._bucket_high: float | None = None
        self._bucket_low: float | None = None
        self._bucket_close: float | None = None
        # 1-hour confirmation layer (opt-in) -- own independent Supertrend,
        # HA state, and bucket-aggregation state, deliberately DUPLICATED
        # from the 15-minute filter's own machinery rather than shared
        # (see module docstring's OPT-IN THIRD CONFIRMATION LAYER note).
        self._st1h = _Supertrend(self.atr_period_1h, self.factor_1h) if self.use_1h_filter else None
        self._ha1h = _HeikinAshi() if (self.use_1h_filter and self.use_heikin_ashi) else None
        self._bucket_id_1h: int | None = None
        self._bucket_open_1h: float | None = None
        self._bucket_high_1h: float | None = None
        self._bucket_low_1h: float | None = None
        self._bucket_close_1h: float | None = None
        # "once target is done then no new trade until 15 min supertrend
        # changes" -- set by notify_target_hit() (an OPTION-PREMIUM event
        # this BTC-price-only class can't see on its own), cleared the
        # instant the 15-minute Supertrend has its own next fresh flip (any
        # direction). See notify_target_hit()'s own docstring.
        self._blocked_until_15m_flip = False

    @property
    def ready(self) -> bool:
        base = self._st._bars_seen >= self.atr_period and self._st15._bars_seen >= self.atr_period_15m
        if self._st1h is not None:
            return base and self._st1h._bars_seen >= self.atr_period_1h
        return base

    @property
    def in_position(self) -> bool:
        return self._is_short is not None

    @property
    def is_short(self) -> bool | None:
        return self._is_short

    def force_flat(self) -> None:
        self._is_short = None
        self._active_sl = None

    def notify_target_hit(self) -> None:
        """Call this the instant the caller (backtest/live engine) closes a
        position on a PREMIUM-based profit target -- an option-level event
        this class has no visibility into by itself (it only ever sees BTC
        candles). Blocks new entries until the 15-MINUTE Supertrend has its
        own next fresh flip, in EITHER direction -- "once target is done
        then no new trade until 15 min supertrend changes".

        ALSO clears _is_short/_active_sl itself (deliberately overlapping
        with force_flat()) -- a caller that closes the position externally
        and calls ONLY notify_target_hit(), forgetting a separate
        force_flat(), would otherwise leave this object permanently
        believing it's still in that position, silently blocking EVERY
        future entry forever (confirmed for real: a 90-day backtest showed
        1979 qualifying entries after one missed force_flat() following a
        target hit, 0 of which fired). Making this method self-sufficient
        removes that whole class of caller mistake."""
        self._is_short = None
        self._active_sl = None
        self._blocked_until_15m_flip = True

    def debug_state(self) -> dict:
        def r(x):
            return round(x, 2) if isinstance(x, (int, float)) else None
        return {
            "st_value": r(self._st._value), "st_direction": self._st._direction,
            "st15_direction": self._st15._direction,
            "in_position": self.in_position, "is_short": self._is_short,
            "active_sl": r(self._active_sl),
            "blocked_until_15m_flip": self._blocked_until_15m_flip,
        }

    # ------------------------------------------------------------------ #
    def _feed_15m(self, candle: Candle) -> bool:
        """Aggregate the closed candle into the running 15-minute bucket;
        the instant a NEW bucket starts, the just-finished bucket's real
        O/H/L/C is fed into the 15m Supertrend as one recursive step
        (locked, not live -- see module docstring). Returns True the
        instant THAT step causes the 15m Supertrend to flip direction --
        used to clear notify_target_hit()'s entry block."""
        bucket_id = candle.start_time // self.bucket_seconds
        if self._bucket_id is None:
            self._bucket_id = bucket_id
            self._bucket_open = candle.open
            self._bucket_high = candle.high
            self._bucket_low = candle.low
            self._bucket_close = candle.close
            return False
        if bucket_id != self._bucket_id:
            prev_dir15 = self._st15._direction
            if self._ha15 is not None:
                # Fresh HA conversion of the REAL 15-minute bucket -- NOT a
                # rollup of the primary's own (separately-stated) HA candles.
                _, ha_h, ha_l, ha_c = self._ha15.convert(
                    self._bucket_open, self._bucket_high, self._bucket_low, self._bucket_close)
                self._st15.update(ha_h, ha_l, ha_c)
            else:
                self._st15.update(self._bucket_high, self._bucket_low, self._bucket_close)
            flipped15 = prev_dir15 != 0 and self._st15._direction != prev_dir15
            self._bucket_id = bucket_id
            self._bucket_open = candle.open
            self._bucket_high = candle.high
            self._bucket_low = candle.low
            self._bucket_close = candle.close
            return flipped15
        self._bucket_high = max(self._bucket_high, candle.high)
        self._bucket_low = min(self._bucket_low, candle.low)
        self._bucket_close = candle.close
        return False

    # ------------------------------------------------------------------ #
    def _feed_1h(self, candle: Candle) -> bool:
        """Opt-in 1-hour confirmation layer -- same LOCKED bucket-boundary
        pattern as _feed_15m() above, deliberately duplicated rather than
        shared (see module docstring's OPT-IN THIRD CONFIRMATION LAYER
        note). Only ever called when self._st1h is not None. Return value
        is unused today (no notify_target_hit()-style blocking is tied to
        the 1h layer), kept for symmetry with _feed_15m()."""
        bucket_id = candle.start_time // self.bucket_seconds_1h
        if self._bucket_id_1h is None:
            self._bucket_id_1h = bucket_id
            self._bucket_open_1h = candle.open
            self._bucket_high_1h = candle.high
            self._bucket_low_1h = candle.low
            self._bucket_close_1h = candle.close
            return False
        if bucket_id != self._bucket_id_1h:
            prev_dir1h = self._st1h._direction
            if self._ha1h is not None:
                _, ha_h, ha_l, ha_c = self._ha1h.convert(
                    self._bucket_open_1h, self._bucket_high_1h, self._bucket_low_1h, self._bucket_close_1h)
                self._st1h.update(ha_h, ha_l, ha_c)
            else:
                self._st1h.update(self._bucket_high_1h, self._bucket_low_1h, self._bucket_close_1h)
            flipped1h = prev_dir1h != 0 and self._st1h._direction != prev_dir1h
            self._bucket_id_1h = bucket_id
            self._bucket_open_1h = candle.open
            self._bucket_high_1h = candle.high
            self._bucket_low_1h = candle.low
            self._bucket_close_1h = candle.close
            return flipped1h
        self._bucket_high_1h = max(self._bucket_high_1h, candle.high)
        self._bucket_low_1h = min(self._bucket_low_1h, candle.low)
        self._bucket_close_1h = candle.close
        return False

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> Supertrend15mFilterFixedSlDecision | None:
        if self._ha1 is not None:
            _, ha_h, ha_l, ha_c = self._ha1.convert(candle.open, candle.high, candle.low, candle.close)
            st_value, direction = self._st.update(ha_h, ha_l, ha_c)
            # The frozen SL's crossing check also uses HA high/low in this
            # mode -- see module docstring (mirrors supertrend_flip.py's own
            # breakout-level checks, which are HA-based too).
            check_high, check_low = ha_h, ha_l
            if self.sl_on_close_only:
                check_high = check_low = ha_c
        else:
            st_value, direction = self._st.update(candle.high, candle.low, candle.close)
            check_high, check_low = candle.high, candle.low
            if self.sl_on_close_only:
                # OPT-IN CLOSE-ONLY SL: ignore the wick, only react to where
                # the 1-minute candle actually closed -- a same-candle wick
                # through the frozen SL that closes back on the safe side no
                # longer triggers a stop-out.
                check_high = check_low = candle.close
        dir15_flipped = self._feed_15m(candle)
        if dir15_flipped and self._blocked_until_15m_flip:
            self._blocked_until_15m_flip = False
        if self._st1h is not None:
            self._feed_1h(candle)

        dir_flipped = self._prev_dir != 0 and direction != self._prev_dir
        self._prev_dir = direction

        exit_ = False
        exit_price = candle.close
        exit_was_short = False
        exit_reason: str | None = None
        entry_signal = False
        entry_is_short = False
        entry_price = candle.close
        sl_level: float | None = None

        if not self.ready:
            return None

        is_red, is_green = direction > 0, direction < 0
        is_red15, is_green15 = self._st15._direction > 0, self._st15._direction < 0
        # No-op (always True) when the 1h filter is disabled -- see module
        # docstring's OPT-IN THIRD CONFIRMATION LAYER note.
        is_red1h = self._st1h._direction > 0 if self._st1h is not None else True
        is_green1h = self._st1h._direction < 0 if self._st1h is not None else True

        # ---- 1. Exit A: real price crosses the FROZEN (non-trailing) SL.
        #         Checked before entry so a same-bar stop-out + fresh
        #         qualifying flip can still open a new leg this bar. ----
        if self._is_short is True and self._active_sl is not None and check_high >= self._active_sl:
            exit_, exit_price, exit_was_short, exit_reason = True, self._active_sl, True, "SL"
            self._is_short = None
            self._active_sl = None
        elif self._is_short is False and self._active_sl is not None and check_low <= self._active_sl:
            exit_, exit_price, exit_was_short, exit_reason = True, self._active_sl, False, "SL"
            self._is_short = None
            self._active_sl = None

        # ---- 1b. Exit B (opt-in): the 15m Supertrend itself has a FRESH
        #          flip AGAINST the still-open position -- a pure trend-
        #          change exit (no price level, so exit_price is
        #          candle.close). Only checked if the SL above didn't
        #          already close this bar. ----
        if (self.exit_on_15m_flip_against and not exit_ and self._is_short is not None
                and dir15_flipped):
            against = ((self._is_short is True and is_green15)
                       or (self._is_short is False and is_red15))
            if against:
                exit_, exit_price, exit_was_short, exit_reason = True, candle.close, self._is_short, "15M_FLIP"
                self._is_short = None
                self._active_sl = None

        # ---- 2. Entry: a FRESH flip, confirmed by the 15m filter already
        #         agreeing at this exact moment. SL frozen at THIS flip
        #         candle's own Supertrend value. Blocked entirely after a
        #         target hit until the 15m itself has its own next flip. ----
        if self._is_short is None and dir_flipped and not self._blocked_until_15m_flip:
            if is_red and is_red15 and is_red1h:
                self._is_short = True
                self._active_sl = st_value
                entry_signal, entry_is_short = True, True
                entry_price, sl_level = candle.close, st_value
            elif is_green and is_green15 and is_green1h:
                self._is_short = False
                self._active_sl = st_value
                entry_signal, entry_is_short = True, False
                entry_price, sl_level = candle.close, st_value

        if not (exit_ or entry_signal):
            return None
        return Supertrend15mFilterFixedSlDecision(
            candle=candle, exit=exit_, exit_price=exit_price, exit_was_short=exit_was_short,
            exit_reason=exit_reason, entry_signal=entry_signal, entry_is_short=entry_is_short,
            entry_price=entry_price, sl_level=sl_level,
        )
