"""Dchannel Strategy (2026-07-10 rewrite) -- Williams %R + Donchian-touch +
open=low/open=high confirmation + EMA(1000) trend filter, on 1-minute
synthetic Heikin Ashi candles. Option BUYING (the original 2026-07-09 version
of this file sold options on real 5m candles with a Donchian-touch reversal
pattern; that version is fully superseded).

Setup detection runs on SYNTHETIC HEIKIN ASHI candles computed internally from
the real candles fed to :meth:`update` -- same convention as HeikinAshiStrategy.
Trigger/SL LEVELS come from the HA candles; whether price actually CROSSES
those levels (entry fill, pre-entry invalidation, SL) is always checked
against the REAL candle's high/low, since that is what a live order actually
fills against (matches HeikinAshiStrategy exactly).

Rules (one position at a time; long/CALL-buy side described below, the
short/PUT-buy side is the exact mirror):

  * **Williams %R(14)** on HA candles crosses below -80 (oversold) -> ARMS a
    bullish hunt. This codebase's %R convention ranges 0 (most overbought) to
    -100 (most oversold), so the mirror "overbought" zone is symmetric around
    0: wr > -20 arms a bearish hunt. One-time: once armed, further %R
    movement does NOT cancel it -- only a subsequent OPPOSITE-direction %R
    cross switches the hunt to the other side instead (does not stack), or
    completing/finding the signal ends it.
  * Once armed, watch subsequent closed HA candles for the first one whose
    low touches/breaches the **Donchian(20) LOWER band** (computed from the
    prior 20 CLOSED HA candles, excluding the current one). That is the
    starting reference candle -- the **signal range** (highest-high /
    lowest-low) begins here.
  * **Ratchet (2026-07-10 refinement)**: on each subsequent candle while
    still hunting, if that candle is BOTH more oversold (%R lower than the
    current reference's %R) AND has a lower high than the current reference
    -> it becomes the NEW reference, and the range RESETS to start fresh
    from it (an earlier/tighter, more favorable entry trigger than sticking
    with the original touch candle). A candle that doesn't beat both
    conditions just widens the existing range (max high / min low) instead.
  * The range keeps ratcheting/widening until a candle is found whose
    **HA open == HA low** (tolerant equality -- no lower wick) AND whose
    **HA close is ABOVE the EMA(1000)** -- the confirming candle. The
    signal range's final high/low includes this candle too.
  * **Entry**: ASAP the instant REAL price breaks ABOVE the signal range's
    high -- enter immediately.
  * **Pre-entry invalidation**: if REAL price breaks BELOW the signal
    range's low before the entry trigger, the whole setup is discarded
    untraded.
  * **Execution**: BUY a CALL nearest ``--target-premium`` (default ~125,
    the 100-150 range's midpoint).
  * **SL**: REAL BTC price touching the signal range's low (the entry
    trigger to SL distance IS the risk).
  * **TP** (2026-07-10 update, 1:2 risk:reward on BTC price): fully internal
    and BTC-price-driven, exactly like the SL -- computed once at entry as
    ``entry + rr_multiple * (entry - sl)`` for a long (mirror for short),
    default ``rr_multiple=2.0``. No longer premium-based/external -- the
    option is purely the execution vehicle; both SL and TP are decided by
    BTC price alone.
  * Forced **EOD square-off** at ``square_off_hour:minute`` (default 17:25
    IST); a settlement gap up to ``day_start_hour:minute`` (default 17:30)
    blocks new hunts/pending setups and cancels any pending setup outright.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..enums import PositionState
from ..models import Candle

__all__ = ["DchannelStrategy", "DchannelDecision"]

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


class _Sma:
    """Simple moving average over the last ``length`` values."""

    def __init__(self, length: int) -> None:
        self._buf: deque[float] = deque(maxlen=length)

    def update(self, x: float) -> float:
        self._buf.append(x)
        return sum(self._buf) / len(self._buf)

    @property
    def value(self) -> float | None:
        return sum(self._buf) / len(self._buf) if self._buf else None

    @property
    def ready(self) -> bool:
        return len(self._buf) == self._buf.maxlen


class _WrWindow:
    """Williams %R rolling window: push THEN read (current bar included)."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._highs: deque[float] = deque(maxlen=period)
        self._lows: deque[float] = deque(maxlen=period)

    def update(self, high: float, low: float) -> tuple[float, float]:
        self._highs.append(high)
        self._lows.append(low)
        return max(self._highs), min(self._lows)

    @property
    def ready(self) -> bool:
        return len(self._highs) >= self.period


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
class DchannelDecision:
    candle: Candle
    long_exit: bool
    short_exit: bool
    long_exit_price: float
    short_exit_price: float
    buy_signal: bool         # bullish confirmation -> BUY a CALL
    sell_signal: bool        # bearish confirmation -> BUY a PUT
    entry_price: float
    exit_reason: str | None  # "SL" | "EOD" | None (TP is externally notified)
    sl_level: float | None   # the SL level active right after a just-opened position

    @property
    def has_exit(self) -> bool:
        return self.long_exit or self.short_exit

    @property
    def has_entry(self) -> bool:
        return self.buy_signal or self.sell_signal


class DchannelStrategy:
    def __init__(
        self,
        *,
        dc_period: int = 20,
        wr_period: int = 14,
        wr_level: float = 80.0,
        ema_length: int = 1000,
        ma_length: int = 0,
        wr_enabled: bool = True,
        anchor_mode: str = "ratchet",
        rr_multiple: float = 2.0,
        tp_pct: float | None = None,
        day_tz: str = "Asia/Kolkata",
        day_start_hour: int = 17,
        day_start_minute: int = 30,
        square_off_hour: int = 17,
        square_off_minute: int = 25,
    ) -> None:
        self.dc_period = dc_period
        self.wr_period = wr_period
        self.wr_level = wr_level
        self.ema_length = ema_length
        # ma_length > 0 switches the trend filter from price-vs-EMA to an
        # EMA-vs-SMA cross (bull: EMA > SMA, bear: EMA < SMA). 0 keeps the
        # original price-vs-EMA behavior. wr_enabled=False removes the Williams
        # %R arming gate entirely (both hunts always active, ratchet drops its
        # %R condition).
        self.ma_length = ma_length
        self.wr_enabled = wr_enabled
        # anchor_mode = "ratchet" (legacy) OR "highest_touch" (2026-07-12 rewrite):
        #   the signal range is anchored at the MOST-EXTREME band-touching candle
        #   (bear: highest high touching the DC upper; bull: lowest low touching
        #   the DC lower), then spans highest-high/lowest-low through the
        #   completing candle. No ratchet reset.
        self.anchor_mode = anchor_mode
        self.rr_multiple = rr_multiple  # TP = entry +/- rr_multiple * (entry-to-SL risk), BTC-price-driven
        # If set, TP is a flat +/- pct of the entry BTC price instead of an
        # RR multiple of the entry-to-SL risk (e.g. tp_pct=0.005 -> 0.5%).
        self.tp_pct = tp_pct
        self._tz = ZoneInfo(day_tz)
        self._sess_mins = day_start_hour * 60 + day_start_minute
        self._sq_mins = square_off_hour * 60 + square_off_minute
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._dc = _Donchian(self.dc_period)
        self._wr = _WrWindow(self.wr_period)
        self._ema = _Ema(self.ema_length)
        self._ma = _Sma(self.ma_length) if self.ma_length > 0 else None
        self._warmup_bars = 0

        # Running Heikin Ashi state.
        self._ha_open: float | None = None
        self._ha_close: float | None = None

        self._prev_now_mins: int | None = None

        # Hunt state machine (bull = looking to BUY a CALL, bear = PUT).
        self._hunt_bull = self._hunt_bear = False   # %R-armed
        self._touched_bull = self._touched_bear = False  # DC band touched
        self._range_hi_bull = self._range_lo_bull = None
        self._range_hi_bear = self._range_lo_bear = None
        # Ratchet reference: the current "starting" candle's own %R, used to
        # detect a LATER candle that's both more extreme AND has a tighter
        # extreme (lower high for bull / higher low for bear) -- when found,
        # the range RESETS to start fresh from that new candle (earlier/
        # better entry trigger than sticking with the original touch candle).
        self._ref_wr_bull: float | None = None
        self._ref_wr_bear: float | None = None
        # highest_touch anchor: the most-extreme DC-touching HA extreme so far
        # (bull = lowest ha_low touching the lower band, bear = highest ha_high
        # touching the upper band). A later, more-extreme touch restarts the range.
        self._anchor_lo_bull: float | None = None
        self._anchor_hi_bear: float | None = None

        # Pending breakout + open position (shared final stage).
        self._pending_long = self._pending_short = False
        self._pending_trigger: float | None = None
        self._pending_sl: float | None = None
        self._in_long = self._in_short = False
        self._sl_level: float | None = None
        self._tp_level: float | None = None  # entry +/- rr_multiple * risk, BTC-price-driven

    @property
    def position_state(self) -> PositionState:
        if self._in_long:
            return PositionState.LONG
        if self._in_short:
            return PositionState.SHORT
        return PositionState.FLAT

    @property
    def ready(self) -> bool:
        return (self._dc.ready
                and (self._wr.ready or not self.wr_enabled)
                and self._warmup_bars >= self.ema_length
                and (self._ma is None or self._ma.ready))

    def _bull_trend_ok(self, ha_close: float, ema_val: float | None, ma_val: float | None) -> bool:
        if self.ma_length > 0:
            return ema_val is not None and ma_val is not None and ema_val > ma_val
        return ema_val is not None and ha_close > ema_val

    def _bear_trend_ok(self, ha_close: float, ema_val: float | None, ma_val: float | None) -> bool:
        if self.ma_length > 0:
            return ema_val is not None and ma_val is not None and ema_val < ma_val
        return ema_val is not None and ha_close < ema_val

    @property
    def has_pending(self) -> bool:
        return self._pending_long or self._pending_short

    @property
    def sl_level(self) -> float | None:
        return self._sl_level

    def force_flat(self) -> None:
        self._in_long = self._in_short = False
        self._sl_level = None
        self._tp_level = None
        self._clear_pending()
        self._clear_hunts()

    def notify_exit(self, reason: str) -> None:
        """Backtest/engine tells us a trade closed for a reason this strategy
        couldn't detect itself (e.g. the option contract couldn't be priced at
        entry). SL/TP/EOD are all internal (BTC-price-driven) and already
        flatten the position inside :meth:`update` -- this is only a fallback."""
        self._in_long = self._in_short = False
        self._sl_level = None
        self._tp_level = None

    def _compute_tp(self, trig: float, sl: float, is_long: bool) -> float:
        """TP level in BTC price: a flat +/- ``tp_pct`` of the entry when set,
        else ``rr_multiple`` * the entry-to-SL risk (both fully BTC-driven)."""
        if self.tp_pct is not None:
            return trig * (1.0 + self.tp_pct) if is_long else trig * (1.0 - self.tp_pct)
        return trig + self.rr_multiple * (trig - sl) if is_long else trig - self.rr_multiple * (sl - trig)

    def check_intracandle_sl(self, price: float) -> tuple[bool, bool, float | None]:
        long_sl = self._in_long and self._sl_level is not None and price <= self._sl_level
        short_sl = self._in_short and self._sl_level is not None and price >= self._sl_level
        return bool(long_sl), bool(short_sl), self._sl_level

    def check_intracandle_tp(self, price: float) -> tuple[bool, bool, float | None]:
        long_tp = self._in_long and self._tp_level is not None and price >= self._tp_level
        short_tp = self._in_short and self._tp_level is not None and price <= self._tp_level
        return bool(long_tp), bool(short_tp), self._tp_level

    def apply_intracandle_pending(self, candle: Candle) -> tuple[bool, bool, float]:
        """ASAP entry/invalidation against REAL price -- SL checked BEFORE the
        trigger (conservative). Returns ``(confirmed, invalidated, entry_price)``."""
        if self._pending_long and self._pending_trigger is not None and self._pending_sl is not None:
            trig, sl = self._pending_trigger, self._pending_sl
            if candle.open <= sl or candle.low <= sl:
                self._clear_pending()
                return False, True, 0.0
            if candle.open >= trig or candle.high >= trig:
                self._in_long, self._in_short = True, False
                self._sl_level = sl
                self._tp_level = self._compute_tp(trig, sl, True)
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
                self._tp_level = self._compute_tp(trig, sl, False)
                self._clear_pending()
                return True, False, trig
        return False, False, 0.0

    # ------------------------------------------------------------------ #
    def update(self, candle: Candle) -> DchannelDecision | None:
        # --- Advance the running Heikin Ashi candle ---
        if self._ha_open is None:
            ha_open = (candle.open + candle.close) / 2.0
        else:
            ha_open = (self._ha_open + self._ha_close) / 2.0
        ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0
        ha_high = max(candle.high, ha_open, ha_close)
        ha_low = min(candle.low, ha_open, ha_close)
        self._ha_open, self._ha_close = ha_open, ha_close

        wr_hh, wr_ll = self._wr.update(ha_high, ha_low)
        wr = (wr_hh - ha_close) / (wr_hh - wr_ll) * -100.0 if wr_hh != wr_ll and self._wr.ready else None
        ema_val = self._ema.update(ha_close)
        ma_val = self._ma.update(ha_close) if self._ma is not None else None
        dc_upper, dc_lower = self._dc.upper, self._dc.lower
        self._warmup_bars += 1

        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute
        square_off = (self._prev_now_mins is not None
                      and now_mins >= self._sq_mins and self._prev_now_mins < self._sq_mins)
        in_gap = self._sq_mins <= now_mins < self._sess_mins

        if square_off or in_gap:
            if self._pending_long or self._pending_short:
                self._clear_pending()
            self._clear_hunts()

        long_exit = short_exit = False
        long_exit_price = short_exit_price = candle.close
        exit_reason: str | None = None
        buy_signal = sell_signal = False
        entry_price = candle.close
        new_sl: float | None = None

        # --- Exits: fixed SL (REAL price), then TP (1:2 RR, REAL price), then EOD ---
        if self._in_long:
            if self._sl_level is not None and candle.low <= self._sl_level:
                long_exit, long_exit_price, exit_reason = True, self._sl_level, "SL"
            elif self._tp_level is not None and candle.high >= self._tp_level:
                long_exit, long_exit_price, exit_reason = True, self._tp_level, "TP"
            elif square_off:
                long_exit, long_exit_price, exit_reason = True, candle.close, "EOD"
        elif self._in_short:
            if self._sl_level is not None and candle.high >= self._sl_level:
                short_exit, short_exit_price, exit_reason = True, self._sl_level, "SL"
            elif self._tp_level is not None and candle.low <= self._tp_level:
                short_exit, short_exit_price, exit_reason = True, self._tp_level, "TP"
            elif square_off:
                short_exit, short_exit_price, exit_reason = True, candle.close, "EOD"
        if long_exit or short_exit:
            self._in_long = self._in_short = False
            self._sl_level = None
            self._tp_level = None

        # --- Breakout trigger for a PRIOR-armed pending setup (closed-bar
        #     fallback; ASAP path is apply_intracandle_pending), REAL price. ---
        flat = not self._in_long and not self._in_short
        if flat and not square_off and not in_gap and self._pending_long:
            trig, sl = self._pending_trigger, self._pending_sl
            if trig is not None and candle.high >= trig:
                buy_signal, entry_price, new_sl = True, trig, sl
                self._in_long, self._sl_level = True, sl
                self._tp_level = self._compute_tp(trig, sl, True)
                self._clear_pending()
            elif sl is not None and candle.low <= sl:
                self._clear_pending()
        elif flat and not square_off and not in_gap and self._pending_short:
            trig, sl = self._pending_trigger, self._pending_sl
            if trig is not None and candle.low <= trig:
                sell_signal, entry_price, new_sl = True, trig, sl
                self._in_short, self._sl_level = True, sl
                self._tp_level = self._compute_tp(trig, sl, False)
                self._clear_pending()
            elif sl is not None and candle.high >= sl:
                self._clear_pending()

        # --- Hunt state machine (HA-based), only while flat/no pending/gates ok ---
        flat_now = not self._in_long and not self._in_short
        if (flat_now and not self.has_pending and not square_off and not in_gap
                and self.ready and (wr is not None or not self.wr_enabled)):
            if self.wr_enabled:
                # %R crossing arms (or switches) a hunt.
                # %R ranges 0 (most overbought) to -100 (most oversold) in this
                # codebase's convention -- it can NEVER exceed 0, so "overbought"
                # is the symmetric zone near 0: wr > -(100 - wr_level).
                crossed_bull_wr = wr < -self.wr_level               # oversold -> bullish hunt
                crossed_bear_wr = wr > -(100.0 - self.wr_level)     # overbought -> bearish hunt
                if crossed_bull_wr and not self._hunt_bull:
                    self._hunt_bull = True
                    self._hunt_bear = False
                    self._touched_bear = False
                    self._range_hi_bear = self._range_lo_bear = None
                elif crossed_bear_wr and not self._hunt_bear:
                    self._hunt_bear = True
                    self._hunt_bull = False
                    self._touched_bull = False
                    self._range_hi_bull = self._range_lo_bull = None
            else:
                # %R disabled: both directions are always hunting (the DC touch
                # alone starts a range). Never resets touched/range state.
                self._hunt_bull = self._hunt_bear = True

            # Bullish hunt progression.
            if self._hunt_bull and dc_lower is not None:
                if not self._touched_bull:
                    if candle.low <= dc_lower:
                        self._touched_bull = True
                        self._range_hi_bull, self._range_lo_bull = ha_high, ha_low
                        self._ref_wr_bull = wr
                        self._anchor_lo_bull = ha_low
                else:
                    if self.anchor_mode == "highest_touch":
                        # Anchor = the LOWEST band-touching candle: a later candle
                        # that touches the DC lower band at a NEW low restarts the
                        # range from it; everything else just widens hi/lo.
                        if (candle.low <= dc_lower and self._anchor_lo_bull is not None
                                and ha_low < self._anchor_lo_bull):
                            self._range_hi_bull, self._range_lo_bull = ha_high, ha_low
                            self._anchor_lo_bull = ha_low
                        else:
                            self._range_hi_bull = max(self._range_hi_bull, ha_high)
                            self._range_lo_bull = min(self._range_lo_bull, ha_low)
                    else:
                        # Ratchet (legacy): a LATER candle with a lower high (and,
                        # when %R is on, also more oversold) RESETS the range.
                        more_oversold = (not self.wr_enabled) or (
                            wr is not None and self._ref_wr_bull is not None and wr < self._ref_wr_bull)
                        if more_oversold and ha_high < self._range_hi_bull:
                            self._range_hi_bull, self._range_lo_bull = ha_high, ha_low
                            self._ref_wr_bull = wr
                        else:
                            self._range_hi_bull = max(self._range_hi_bull, ha_high)
                            self._range_lo_bull = min(self._range_lo_bull, ha_low)
                    if (_close_enough(ha_open, ha_low, candle.close)
                            and self._bull_trend_ok(ha_close, ema_val, ma_val)):
                        self._pending_long = True
                        self._pending_trigger = self._range_hi_bull
                        self._pending_sl = self._range_lo_bull
                        self._hunt_bull = self._touched_bull = False
                        self._range_hi_bull = self._range_lo_bull = None
                        self._ref_wr_bull = self._anchor_lo_bull = None

            # Bearish hunt progression (mirror: more OVERBOUGHT -- wr closer to
            # 0 -- AND a HIGHER low than the current reference ratchets it).
            if self._hunt_bear and dc_upper is not None:
                if not self._touched_bear:
                    if candle.high >= dc_upper:
                        self._touched_bear = True
                        self._range_hi_bear, self._range_lo_bear = ha_high, ha_low
                        self._ref_wr_bear = wr
                        self._anchor_hi_bear = ha_high
                else:
                    if self.anchor_mode == "highest_touch":
                        # Anchor = the HIGHEST band-touching candle: a later candle
                        # touching the DC upper band at a NEW high restarts the range.
                        if (candle.high >= dc_upper and self._anchor_hi_bear is not None
                                and ha_high > self._anchor_hi_bear):
                            self._range_hi_bear, self._range_lo_bear = ha_high, ha_low
                            self._anchor_hi_bear = ha_high
                        else:
                            self._range_hi_bear = max(self._range_hi_bear, ha_high)
                            self._range_lo_bear = min(self._range_lo_bear, ha_low)
                    else:
                        more_overbought = (not self.wr_enabled) or (
                            wr is not None and self._ref_wr_bear is not None and wr > self._ref_wr_bear)
                        if more_overbought and ha_low > self._range_lo_bear:
                            self._range_hi_bear, self._range_lo_bear = ha_high, ha_low
                            self._ref_wr_bear = wr
                        else:
                            self._range_hi_bear = max(self._range_hi_bear, ha_high)
                            self._range_lo_bear = min(self._range_lo_bear, ha_low)
                    if (_close_enough(ha_open, ha_high, candle.close)
                            and self._bear_trend_ok(ha_close, ema_val, ma_val)):
                        self._pending_short = True
                        self._pending_trigger = self._range_lo_bear
                        self._pending_sl = self._range_hi_bear
                        self._hunt_bear = self._touched_bear = False
                        self._range_hi_bear = self._range_lo_bear = None
                        self._ref_wr_bear = self._anchor_hi_bear = None

        self._dc.push(ha_high, ha_low)
        self._prev_now_mins = now_mins

        if not (long_exit or short_exit or buy_signal or sell_signal):
            return None
        return DchannelDecision(
            candle=candle, long_exit=long_exit, short_exit=short_exit,
            long_exit_price=long_exit_price, short_exit_price=short_exit_price,
            buy_signal=buy_signal, sell_signal=sell_signal, entry_price=entry_price,
            exit_reason=exit_reason, sl_level=new_sl,
        )

    # ------------------------------------------------------------------ #
    def _clear_pending(self) -> None:
        self._pending_long = self._pending_short = False
        self._pending_trigger = self._pending_sl = None

    def _clear_hunts(self) -> None:
        self._hunt_bull = self._hunt_bear = False
        self._touched_bull = self._touched_bear = False
        self._range_hi_bull = self._range_lo_bull = None
        self._range_hi_bear = self._range_lo_bear = None
        self._ref_wr_bull = self._ref_wr_bear = None
        self._anchor_lo_bull = self._anchor_hi_bear = None
