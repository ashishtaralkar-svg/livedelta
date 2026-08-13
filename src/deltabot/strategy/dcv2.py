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
      **Exception (2026-07-15, two-stage trail)**: if at BUY trigger time
      BOTH EMA(50) and EMA(200) sit ABOVE the trigger price (price broke out
      from below the EMAs), the EMA-relationship exit is replaced by a
      two-stage trailing exit for that trade:
        1. initially only the fixed signal-range SL is live;
        2. once a candle CLOSES above BOTH EMAs, the trail ARMS;
        3. after arming, a candle CLOSING below BOTH EMAs exits the trade at
           that close (reason "TRAIL"). The fixed SL stays live throughout.
      Mirror for a SELL triggering above both EMAs (arms on a close below
      both, exits on a close back above both). Close checks use the REAL
      candle close against the HA-based EMA levels, per this codebase's
      levels-from-HA / crossings-from-real-price convention.
  * **After an SL or TRAIL exit**, that exit bar never counts as the
    re-touch -- the new-signal search begins on the immediate next bar, and
    a FRESH band touch from there starts the new range (Pine's justClosedSL).
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


class _Sma:
    """Simple moving average over the last ``length`` values."""

    def __init__(self, length: int) -> None:
        self._window: deque[float] = deque(maxlen=length)

    def update(self, x: float) -> float:
        self._window.append(x)
        return sum(self._window) / len(self._window)

    @property
    def value(self) -> float | None:
        return sum(self._window) / len(self._window) if self._window else None


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
        long_ma_kind: str = "ema",
        ema_filter_fast: int = 0,
        ema_filter_slow: int = 0,
        ema_filter_gate: bool = True,
        tp_cooldown_until_cross: bool = False,
        entry_above_ema_long: bool = False,
        use_heikin_ashi: bool = True,
        confirm_mode: str = "open_extreme",
        direction_gate: str = "ema",
        no_settlement_gap: bool = False,
        ema_exit: bool = True,
        disable_trail: bool = False,
        target_rr: float = 0.0,
        target_pct: float = 0.0,
        skip_weekdays: frozenset[int] = frozenset({5, 6}),  # Mon=0 .. Sat=5, Sun=6 (IST)
        day_tz: str = "Asia/Kolkata",
        day_start_hour: int = 17,
        day_start_minute: int = 30,
        square_off_hour: int = 17,
        square_off_minute: int = 25,
        entry_start_hour: int = 0,
        entry_end_hour: int = 24,
        require_ema_direction_at_trigger: bool = False,
    ) -> None:
        self.dc_period = dc_period
        self.ema_trend_length = ema_trend_length
        self.ema_long_length = ema_long_length
        # long_ma_kind: which MA the "long"/reference line uses for the hunt
        # direction gate + EMA-reversal/trail exits. "ema" (default) = the
        # classic EMA(ema_long_length) (e.g. 200). "sma" = a Simple MA of the
        # same length -- e.g. ema_long_length=50 gives the EMA(50)-vs-SMA(50)
        # gate: BUY hunts when EMA(50) > SMA(50), SELL when EMA(50) < SMA(50).
        self.long_ma_kind = long_ma_kind
        # Optional second EMA trend filter (0/0 = off). When both > 0, an entry
        # is gated: BUY needs EMA(fast) > EMA(slow), SELL needs EMA(fast) <
        # EMA(slow), on HA close. Layered ON TOP of the existing EMA(50)/(200)
        # logic -- everything else unchanged.
        self.ema_filter_fast = ema_filter_fast
        self.ema_filter_slow = ema_filter_slow
        self._ema_filter_on = ema_filter_fast > 0 and ema_filter_slow > 0
        # ema_filter_gate: True (default) -> the EMA(fast)/(slow) pair GATES
        # entries. False -> the pair is still COMPUTED (for ema_filter_relation,
        # e.g. trend-based position sizing) but does NOT block any entry.
        self.ema_filter_gate = ema_filter_gate
        # tp_cooldown_until_cross: after a 70% TP is booked (signalled from the
        # runner/engine via enter_tp_cooldown()), STOP hunting/entering new
        # trades until the NEXT EMA crossover -- either EMA(50)/(200) OR
        # EMA(fast)/(slow) crossing in EITHER direction. First crossover lifts
        # the pause; hunting then resumes normally. Reduces repeated re-entries
        # in the same trend leg after a win. Only a TP arms this (SL/EMA-cross/
        # TRAIL exits do not). Off by default -- existing behavior unchanged.
        self.tp_cooldown_until_cross = tp_cooldown_until_cross
        # entry_above_ema_long: at the breakout trigger, require the entry price
        # to sit on the trend side of EMA(200) -- a BUY only fires if trig >
        # EMA(200), a SELL only if trig < EMA(200). Blocks counter-trend
        # breakouts that cross back through the 200 EMA. Consumes the setup
        # untraded when it disagrees (same as the weekend/EMA-filter gates).
        self.entry_above_ema_long = entry_above_ema_long
        # ema_exit=False: disable BOTH EMA-based exits (the "cross" EMA-reversal
        # exit AND the two-stage "trail" exit) -- a position then closes ONLY on
        # its fixed range SL (or an external administrative close: square-off/
        # rollover/weekend/TP, handled outside the strategy).
        self.ema_exit = ema_exit
        # disable_trail=True: force EVERY trade to use the plain "cross" EMA
        # exit (close when EMA(50) crosses to the unfavorable side of
        # EMA(200)), even the subset that would normally get the two-stage
        # "trail" exit (entries that triggered from the wrong side of BOTH
        # EMAs -- see _exit_mode_for). No-op if ema_exit=False (no EMA-based
        # exit at all either way).
        self.disable_trail = disable_trail
        # target_rr > 0: on entry, set a fixed profit target at
        # entry +/- risk*target_rr, where risk = |entry - sl_level| (the SAME
        # distance as the fixed SL, just on the profitable side). 1.0 = a 1:1
        # risk:reward target. Checked independently of ema_exit -- it can run
        # alongside EMA_CROSS/TRAIL (whichever hits first) or, with
        # ema_exit=False, be the ONLY profit exit next to the fixed SL.
        self.target_rr = target_rr
        # target_pct > 0: a FIXED-percent profit target at entry +/-
        # target_pct%% of the entry price (independent of the SL distance).
        # Used for directional BTC trades (--mode btc): e.g. 1.0 = exit at a
        # 1%% move in favour. Takes precedence over target_rr when both set.
        self.target_pct = target_pct
        # use_heikin_ashi=False -> run every indicator/pattern on the RAW candle
        # OHLC (normal candlestick chart) instead of synthetic Heikin Ashi.
        self.use_heikin_ashi = use_heikin_ashi
        # confirm_mode:
        #   "open_extreme" (default, deployed): the confirming candle has
        #     open==low (bull) / open==high (bear), highest_touch re-anchor.
        #   "next_green": a 2-candle range -- a candle touches the DC band and
        #     the IMMEDIATE NEXT candle is green (bull) / red (bear); the range
        #     spans those two candles. No open==extreme, no re-anchor.
        self.confirm_mode = confirm_mode
        # direction_gate:
        #   "ema" (default, deployed): EMA(50) vs EMA(200) decides which side
        #     hunts, plus the EMA-reversal / two-stage-trail exits.
        #   "session_line": EMAs are IGNORED entirely. A horizontal session line
        #     is drawn at each 17:30 (the session-open price). Both directions
        #     hunt; a BUY range is taken only if it sits ENTIRELY ABOVE the line
        #     (range low >= line), a SELL range only if ENTIRELY BELOW it (range
        #     high <= line). No EMA-reversal/trail exit; the trade exits on its
        #     fixed range SL or the 17:25 square-off (which FLATTENS it). No
        #     rollover -- hunting restarts fresh at 17:30 with the new line.
        self.direction_gate = direction_gate
        # no_settlement_gap: remove the 17:25-17:30 no-trade window (continuous).
        self.no_settlement_gap = no_settlement_gap
        self.skip_weekdays = skip_weekdays
        self._tz = ZoneInfo(day_tz)
        self._sess_mins = day_start_hour * 60 + day_start_minute
        self._sq_mins = square_off_hour * 60 + square_off_minute
        # entry_start_hour/entry_end_hour: gate the TRADE itself to an IST hour
        # window [start, end) -- e.g. 7/17 = only 07:00-16:59 entries. Same
        # convention as skip_weekdays: pattern hunting/arming continues outside
        # the window regardless; a trigger hit outside it just consumes the
        # pending setup untraded. Default 0/24 = no restriction (unchanged).
        self.entry_start_hour = entry_start_hour
        self.entry_end_hour = entry_end_hour
        # require_ema_direction_at_trigger: a pending setup that fully arms
        # (touched + confirmed -> _pending_long/_pending_short) is NOT cleared
        # by an EMA flip in the meantime -- _reset_ranges() (called on every
        # hunt-direction change) only clears in-progress range-forming state,
        # never an already-armed pending trigger/SL. So a setup armed during a
        # bullish EMA phase can still fire its entry later even if EMA(50) has
        # since crossed back below EMA(200) by the time price hits the
        # trigger. On: re-verify the EMA(trend)-vs-EMA(long) relationship
        # still agrees with the trade direction AT THE TRIGGER MOMENT itself
        # (BUY needs ema_trend > ema_long, SELL needs ema_trend < ema_long) --
        # checked in BOTH the closed-bar trigger (update()) and the ASAP
        # intracandle trigger (apply_intracandle_pending(), what live actually
        # uses by default); disagreeing consumes the setup untraded, same
        # convention as the other trigger-time filters. Off (default) =
        # unchanged behavior.
        self.require_ema_direction_at_trigger = require_ema_direction_at_trigger
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._dc = _Donchian(self.dc_period)
        self._ema_trend = _Ema(self.ema_trend_length)
        self._ema_long = (_Sma(self.ema_long_length) if self.long_ma_kind == "sma"
                          else _Ema(self.ema_long_length))
        self._ema_filt_fast = _Ema(self.ema_filter_fast) if self._ema_filter_on else None
        self._ema_filt_slow = _Ema(self.ema_filter_slow) if self._ema_filter_on else None
        self._warmup_bars = 0
        self._dbg: dict = {}   # per-bar diagnostic snapshot (see update / debug_state)

        # Running Heikin Ashi state.
        self._ha_open: float | None = None
        self._ha_close: float | None = None

        self._prev_now_mins: int | None = None
        self._session_line: float | None = None   # price at the last 17:30 (session_line mode)

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
        self._target_level: float | None = None   # set on entry when target_rr > 0
        # "cross" (default): exit when the EMA50/200 relationship flips against
        # the position. "trail": the trade triggered from the wrong side of
        # BOTH EMAs (buy below both / sell above both) -> fixed SL until a
        # close beyond both EMAs ARMS the trail, then a close back on the
        # entry side of both EMAs exits ("TRAIL").
        self._exit_mode = "cross"
        self._trail_armed = False

        # TP cooldown: True after a booked TP, cleared on the next EMA crossover.
        # _prev_sign_*: last bar's sign of (fast - slow) for each EMA pair, used
        # to detect a crossover (sign flip). 0 = not yet determined.
        self._tp_cooldown = False
        self._prev_sign_5020 = 0
        self._prev_sign_150600 = 0

    @property
    def position_state(self) -> PositionState:
        if self._in_long:
            return PositionState.LONG
        if self._in_short:
            return PositionState.SHORT
        return PositionState.FLAT

    @property
    def ready(self) -> bool:
        need = max(self.ema_long_length, self.ema_filter_slow if self._ema_filter_on else 0)
        return self._dc.ready and self._warmup_bars >= need

    @property
    def has_pending(self) -> bool:
        return self._pending_long or self._pending_short

    @property
    def sl_level(self) -> float | None:
        return self._sl_level

    def debug_state(self) -> dict:
        """Full internal-state snapshot for the engine's optional per-candle
        diagnostic log (no side effects). Combines this bar's computed values
        (``_dbg``) with the persistent hunt/pending/position flags, so a missed
        or unexpected signal can be diagnosed minute-by-minute after the fact."""
        def r(x):
            return round(x, 2) if isinstance(x, (int, float)) else None
        return {
            **self._dbg,
            "ready": self.ready,
            "hunt_bull": self._hunt_bull, "hunt_bear": self._hunt_bear,
            "touched_bull": self._touched_bull, "touched_bear": self._touched_bear,
            "range_hi_bull": r(self._range_hi_bull), "range_lo_bull": r(self._range_lo_bull),
            "range_hi_bear": r(self._range_hi_bear), "range_lo_bear": r(self._range_lo_bear),
            "pending_long": self._pending_long, "pending_short": self._pending_short,
            "pending_trig": r(self._pending_trigger), "pending_sl": r(self._pending_sl),
            "pos": self.position_state.name, "sl_level": r(self._sl_level),
            "exit_mode": self._exit_mode,
        }

    def force_flat(self) -> None:
        self._in_long = self._in_short = False
        self._sl_level = None
        self._target_level = None
        self._exit_mode = "cross"
        self._trail_armed = False
        self._clear_pending()
        self._clear_hunts()

    def enter_tp_cooldown(self) -> None:
        """Called by the runner/engine right after a 70% TP is booked. Arms the
        wait-for-crossover pause (no-op unless tp_cooldown_until_cross is on)."""
        if self.tp_cooldown_until_cross:
            self._tp_cooldown = True

    def _ema_filter_ok(self, is_long: bool) -> bool:
        """Optional EMA(fast)/(slow) trend filter. Off, or gate disabled -> always
        True. On+gated: a BUY needs EMA(fast) > EMA(slow), a SELL needs <."""
        if not self._ema_filter_on or not self.ema_filter_gate:
            return True
        f, s = self._ema_filt_fast.value, self._ema_filt_slow.value
        if f is None or s is None:
            return False   # not warm yet -> don't enter
        return f > s if is_long else f < s

    @property
    def ema_filter_relation(self) -> int:
        """+1 if EMA(fast) > EMA(slow), -1 if <, 0 if off/not-warm. Read for
        trend-based position sizing (not gating)."""
        if not self._ema_filter_on:
            return 0
        f, s = self._ema_filt_fast.value, self._ema_filt_slow.value
        if f is None or s is None:
            return 0
        return 1 if f > s else -1 if f < s else 0

    def _entry_ema_long_ok(self, is_long: bool, price: float) -> bool:
        """Optional EMA(200)-side entry filter. Off -> always True. On: a BUY
        needs the trigger price ABOVE EMA(200), a SELL needs it BELOW."""
        if not self.entry_above_ema_long:
            return True
        el = self._ema_long.value
        if el is None:
            return False   # not warm yet -> don't enter
        return price > el if is_long else price < el

    def _ema_direction_ok(self, is_long: bool) -> bool:
        """require_ema_direction_at_trigger gate: re-verify EMA(trend) vs
        EMA(long) still agrees with the trade direction AT THE TRIGGER MOMENT
        -- shared by the closed-bar trigger (update()) and the ASAP intracandle
        trigger (apply_intracandle_pending()), so a setup armed while e.g.
        EMA(50) > EMA(200) but triggering after EMA(50) has since crossed
        below EMA(200) gets cancelled either way. Off (default) -> always True."""
        if not self.require_ema_direction_at_trigger:
            return True
        et, el = self._ema_trend.value, self._ema_long.value
        if et is None or el is None:
            return False   # not warm yet -> don't enter
        return et > el if is_long else et < el

    def _exit_mode_for(self, trig: float, is_long: bool) -> str:
        """session_line mode: no EMA/trail exit at all ("none" -> SL / 17:25
        only). EMA mode: a trade triggering from the wrong side of BOTH EMAs
        (buy below both, sell above both) uses the two-stage "trail" exit;
        otherwise the normal EMA-relationship "cross" exit."""
        if self.direction_gate == "session_line":
            return "none"
        if self.disable_trail:
            return "cross"
        et, el = self._ema_trend.value, self._ema_long.value
        if et is None or el is None:
            return "cross"
        if is_long:
            return "trail" if (et > trig and el > trig) else "cross"
        return "trail" if (et < trig and el < trig) else "cross"

    def _target_for(self, entry_price: float, sl_level: float | None, is_long: bool) -> float | None:
        """Fixed profit target. target_pct (fixed %% of entry) takes precedence;
        else target_rr (multiple of the SL distance). None if both disabled."""
        if self.target_pct > 0:
            dist = self.target_pct / 100.0 * entry_price
            return entry_price + dist if is_long else entry_price - dist
        if self.target_rr <= 0 or sl_level is None:
            return None
        risk = abs(entry_price - sl_level)
        return entry_price + risk * self.target_rr if is_long else entry_price - risk * self.target_rr

    def _session_ok_long(self, range_lo: float | None) -> bool:
        """session_line mode: a BUY range is valid only if it sits ENTIRELY
        ABOVE the session line. No-op (always True) in EMA mode."""
        if self.direction_gate != "session_line":
            return True
        return (self._session_line is not None and range_lo is not None
                and range_lo >= self._session_line)

    def _session_ok_short(self, range_hi: float | None) -> bool:
        if self.direction_gate != "session_line":
            return True
        return (self._session_line is not None and range_hi is not None
                and range_hi <= self._session_line)

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
                if (not self._ema_filter_ok(is_long=True) or not self._entry_ema_long_ok(True, trig)
                        or not self._ema_direction_ok(is_long=True)):
                    self._clear_pending()   # trend/EMA200/direction filter blocks -> consume, no entry
                    return False, False, 0.0
                self._in_long, self._in_short = True, False
                self._sl_level = sl
                self._target_level = self._target_for(trig, sl, is_long=True)
                self._exit_mode = self._exit_mode_for(trig, is_long=True)
                self._trail_armed = False
                self._clear_pending()
                return True, False, trig
        elif self._pending_short and self._pending_trigger is not None and self._pending_sl is not None:
            trig, sl = self._pending_trigger, self._pending_sl
            if candle.open >= sl or candle.high >= sl:
                self._clear_pending()
                return False, True, 0.0
            if candle.open <= trig or candle.low <= trig:
                if (not self._ema_filter_ok(is_long=False) or not self._entry_ema_long_ok(False, trig)
                        or not self._ema_direction_ok(is_long=False)):
                    self._clear_pending()
                    return False, False, 0.0
                self._in_short, self._in_long = True, False
                self._sl_level = sl
                self._target_level = self._target_for(trig, sl, is_long=False)
                self._exit_mode = self._exit_mode_for(trig, is_long=False)
                self._trail_armed = False
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

        # Bar OHLC that indicators/patterns run on: Heikin Ashi, or the RAW
        # candle when use_heikin_ashi is False (normal candlestick chart).
        if self.use_heikin_ashi:
            bar_open, bar_high, bar_low, bar_close = ha_open, ha_high, ha_low, ha_close
        else:
            bar_open, bar_high, bar_low, bar_close = candle.open, candle.high, candle.low, candle.close

        ema_trend = self._ema_trend.update(bar_close)
        ema_long = self._ema_long.update(bar_close)
        if self._ema_filter_on:
            self._ema_filt_fast.update(bar_close)
            self._ema_filt_slow.update(bar_close)
        dc_upper, dc_lower = self._dc.upper, self._dc.lower
        self._warmup_bars += 1

        # TP cooldown: after a booked TP, wait for the NEXT crossover of either
        # EMA pair -- EMA(50)/(200) OR EMA(fast)/(slow) -- in EITHER direction,
        # then resume hunting. Track each pair's sign and clear on a flip.
        if self.tp_cooldown_until_cross:
            crossed = False
            s1 = (1 if ema_trend > ema_long else -1 if ema_trend < ema_long else 0)
            if s1 != 0 and self._prev_sign_5020 != 0 and s1 != self._prev_sign_5020:
                crossed = True
            if s1 != 0:
                self._prev_sign_5020 = s1
            if self._ema_filter_on:
                ff, fs = self._ema_filt_fast.value, self._ema_filt_slow.value
                if ff is not None and fs is not None:
                    s2 = (1 if ff > fs else -1 if ff < fs else 0)
                    if s2 != 0 and self._prev_sign_150600 != 0 and s2 != self._prev_sign_150600:
                        crossed = True
                    if s2 != 0:
                        self._prev_sign_150600 = s2
            if crossed and self._tp_cooldown:
                self._tp_cooldown = False

        # Diagnostic snapshot of the values the strategy USED this bar (DC bands
        # are the prior-20 exclusion, before this bar is pushed). Read by
        # debug_state() for the engine's optional per-candle state log.
        self._dbg = {
            "ha_o": round(bar_open, 2), "ha_h": round(bar_high, 2),
            "ha_l": round(bar_low, 2), "ha_c": round(bar_close, 2),
            "ema50": round(ema_trend, 2), "ema200": round(ema_long, 2),
            "dc_lo": round(dc_lower, 2) if dc_lower is not None else None,
            "dc_hi": round(dc_upper, 2) if dc_upper is not None else None,
            "open_eq_low": _close_enough(bar_open, bar_low, candle.close),
            "open_eq_high": _close_enough(bar_open, bar_high, candle.close),
        }

        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        now_mins = local.hour * 60 + local.minute
        square_off = (self._prev_now_mins is not None
                      and now_mins >= self._sq_mins and self._prev_now_mins < self._sq_mins)
        in_gap = self._sq_mins <= now_mins < self._sess_mins
        # no_settlement_gap: trade continuously across 17:25-17:30 -- no entry
        # block and no pending/hunt clearing at the boundary (the option roll is
        # handled by the executor/runner; the 17:00 expiry cutoff already keeps
        # near-settlement entries on the next-day option, so holding is safe).
        if self.no_settlement_gap:
            square_off = in_gap = False
        day_blocked = local.weekday() in self.skip_weekdays
        hour_blocked = not (self.entry_start_hour <= local.hour < self.entry_end_hour)
        # session_line mode: (re)draw the session line at 17:30 each day.
        session_start = (self._prev_now_mins is not None
                         and now_mins >= self._sess_mins and self._prev_now_mins < self._sess_mins)
        if self.direction_gate == "session_line" and session_start:
            self._session_line = candle.open

        # --- 0. Settlement gap OR TP cooldown: cancel pending + clear hunts
        #     (never closes a position). During cooldown we stay fully idle until
        #     the next crossover clears it above. ---
        if square_off or in_gap or self._tp_cooldown:
            self._clear_pending()
            self._clear_hunts()

        # --- 0.5. Hunt arming. EMA mode: STATE-based on EMA(50) vs EMA(200)
        #     (the "not already armed" guard fires the reset only on the
        #     transition). session_line mode: EMAs ignored -> BOTH directions
        #     always hunt; the session-line filter decides which range is taken
        #     (in _arm_long/_arm_short). Suppressed while a TP cooldown is active. ---
        if not square_off and not in_gap and not self._tp_cooldown and self.ready:
            if self.direction_gate == "session_line":
                if not self._hunt_bull:
                    self._hunt_bull = True
                if not self._hunt_bear:
                    self._hunt_bear = True
            elif ema_trend > ema_long and not self._hunt_bull:
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
        #     relationship flipping against the position. session_line mode adds
        #     a 17:25 square-off that FLATTENS the trade (its only time exit;
        #     exit_mode is "none" so no EMA/trail exit fires there). ---
        if self._in_long:
            if self.direction_gate == "session_line" and square_off:
                long_exit, long_exit_price, exit_reason = True, candle.close, "EOD"
            elif self._sl_level is not None and candle.low <= self._sl_level:
                long_exit, long_exit_price, exit_reason = True, self._sl_level, "SL"
                just_closed_sl = True
            elif self._target_level is not None and candle.high >= self._target_level:
                long_exit, long_exit_price, exit_reason = True, self._target_level, "TARGET"
                just_closed_sl = True
            elif self.ema_exit and self._exit_mode == "cross" and ema_trend < ema_long:
                long_exit, long_exit_price, exit_reason = True, candle.close, "EMA_CROSS"
            elif self.ema_exit and self._exit_mode == "trail":
                # Two-stage trail (entry was below both EMAs): a REAL close
                # above BOTH EMAs arms it; once armed, a REAL close back below
                # BOTH EMAs exits. A bar can't do both, so if/elif is safe.
                if not self._trail_armed and candle.close > ema_trend and candle.close > ema_long:
                    self._trail_armed = True
                elif self._trail_armed and candle.close < ema_trend and candle.close < ema_long:
                    long_exit, long_exit_price, exit_reason = True, candle.close, "TRAIL"
                    just_closed_sl = True   # like SL: skip this bar, hunt anew from next
        elif self._in_short:
            if self.direction_gate == "session_line" and square_off:
                short_exit, short_exit_price, exit_reason = True, candle.close, "EOD"
            elif self._sl_level is not None and candle.high >= self._sl_level:
                short_exit, short_exit_price, exit_reason = True, self._sl_level, "SL"
                just_closed_sl = True
            elif self._target_level is not None and candle.low <= self._target_level:
                short_exit, short_exit_price, exit_reason = True, self._target_level, "TARGET"
                just_closed_sl = True
            elif self.ema_exit and self._exit_mode == "cross" and ema_trend > ema_long:
                short_exit, short_exit_price, exit_reason = True, candle.close, "EMA_CROSS"
            elif self.ema_exit and self._exit_mode == "trail":
                if not self._trail_armed and candle.close < ema_trend and candle.close < ema_long:
                    self._trail_armed = True
                elif self._trail_armed and candle.close > ema_trend and candle.close > ema_long:
                    short_exit, short_exit_price, exit_reason = True, candle.close, "TRAIL"
                    just_closed_sl = True   # like SL: skip this bar, hunt anew from next
        if long_exit or short_exit:
            self._in_long = self._in_short = False
            self._sl_level = None
            self._target_level = None
            self._exit_mode = "cross"
            self._trail_armed = False

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
                ema_dir_ok = self._ema_direction_ok(is_long=True)
                if (not day_blocked and not hour_blocked and ema_dir_ok and self._ema_filter_ok(is_long=True)
                        and self._entry_ema_long_ok(is_long=True, price=trig)):
                    buy_signal, entry_price, new_sl = True, trig, sl
                    self._in_long, self._sl_level = True, sl
                    self._target_level = self._target_for(trig, sl, is_long=True)
                    self._exit_mode = self._exit_mode_for(trig, is_long=True)
                    self._trail_armed = False
                self._clear_pending()   # consumed (traded, blocked, or filtered out)
        elif flat and not square_off and not in_gap and self._pending_short:
            trig, sl = self._pending_trigger, self._pending_sl
            if sl is not None and candle.high >= sl:
                self._clear_pending()
            elif trig is not None and candle.low <= trig:
                ema_dir_ok = self._ema_direction_ok(is_long=False)
                if (not day_blocked and not hour_blocked and ema_dir_ok and self._ema_filter_ok(is_long=False)
                        and self._entry_ema_long_ok(is_long=False, price=trig)):
                    sell_signal, entry_price, new_sl = True, trig, sl
                    self._in_short, self._sl_level = True, sl
                    self._target_level = self._target_for(trig, sl, is_long=False)
                    self._exit_mode = self._exit_mode_for(trig, is_long=False)
                    self._trail_armed = False
                self._clear_pending()

        # --- 4. Hunt progression (HA-based), only while flat/no pending/gates ok.
        #     just_closed_sl: after an SL OR TRAIL exit, that exit bar never
        #     counts -- the new-signal search begins on the IMMEDIATE NEXT bar. ---
        flat_now = not self._in_long and not self._in_short
        if (flat_now and not self.has_pending and not square_off and not in_gap
                and not just_closed_sl and self.ready):
            # Bullish hunt progression.
            if self._hunt_bull and dc_lower is not None:
                if self.confirm_mode == "next_green":
                    if not self._touched_bull:
                        if candle.low <= dc_lower:
                            self._touched_bull = True
                            self._range_hi_bull, self._range_lo_bull = bar_high, bar_low
                    elif bar_close > bar_open:   # immediate NEXT candle is GREEN -> 2-candle range
                        self._arm_long(max(self._range_hi_bull, bar_high),
                                       min(self._range_lo_bull, bar_low))
                    elif candle.low <= dc_lower:  # not green but re-touches -> new touch candle
                        self._range_hi_bull, self._range_lo_bull = bar_high, bar_low
                    else:                          # not green, no touch -> discard, keep hunting
                        self._touched_bull = False
                        self._range_hi_bull = self._range_lo_bull = None
                else:
                    if not self._touched_bull:
                        if candle.low <= dc_lower:
                            self._touched_bull = True
                            self._range_hi_bull, self._range_lo_bull = bar_high, bar_low
                            self._anchor_lo_bull = bar_low
                    else:
                        if (candle.low <= dc_lower and self._anchor_lo_bull is not None
                                and bar_low < self._anchor_lo_bull):
                            self._range_hi_bull, self._range_lo_bull = bar_high, bar_low
                            self._anchor_lo_bull = bar_low
                        else:
                            self._range_hi_bull = max(self._range_hi_bull, bar_high)
                            self._range_lo_bull = min(self._range_lo_bull, bar_low)
                        # CONFIRM: open==low shape -- direction came from the state gate.
                        if _close_enough(bar_open, bar_low, candle.close):
                            self._arm_long(self._range_hi_bull, self._range_lo_bull)

            # Bearish hunt progression (mirror).
            if self._hunt_bear and dc_upper is not None:
                if self.confirm_mode == "next_green":
                    if not self._touched_bear:
                        if candle.high >= dc_upper:
                            self._touched_bear = True
                            self._range_hi_bear, self._range_lo_bear = bar_high, bar_low
                    elif bar_close < bar_open:   # immediate NEXT candle is RED -> 2-candle range
                        self._arm_short(max(self._range_hi_bear, bar_high),
                                        min(self._range_lo_bear, bar_low))
                    elif candle.high >= dc_upper:
                        self._range_hi_bear, self._range_lo_bear = bar_high, bar_low
                    else:
                        self._touched_bear = False
                        self._range_hi_bear = self._range_lo_bear = None
                else:
                    if not self._touched_bear:
                        if candle.high >= dc_upper:
                            self._touched_bear = True
                            self._range_hi_bear, self._range_lo_bear = bar_high, bar_low
                            self._anchor_hi_bear = bar_high
                    else:
                        if (candle.high >= dc_upper and self._anchor_hi_bear is not None
                                and bar_high > self._anchor_hi_bear):
                            self._range_hi_bear, self._range_lo_bear = bar_high, bar_low
                            self._anchor_hi_bear = bar_high
                        else:
                            self._range_hi_bear = max(self._range_hi_bear, bar_high)
                            self._range_lo_bear = min(self._range_lo_bear, bar_low)
                        if _close_enough(bar_open, bar_high, candle.close):
                            self._arm_short(self._range_hi_bear, self._range_lo_bear)

        self._dc.push(bar_high, bar_low)
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
    def _arm_long(self, rng_hi: float, rng_lo: float) -> None:
        if not self._session_ok_long(rng_lo):
            # session_line mode: range not entirely above the line -> discard the
            # completed range but keep hunting (huntBull stays on).
            self._touched_bull = False
            self._range_hi_bull = self._range_lo_bull = None
            self._anchor_lo_bull = None
            return
        self._pending_long = True
        self._pending_trigger = rng_hi
        self._pending_sl = rng_lo
        self._hunt_bull = self._touched_bull = False
        self._range_hi_bull = self._range_lo_bull = None
        self._anchor_lo_bull = None

    def _arm_short(self, rng_hi: float, rng_lo: float) -> None:
        if not self._session_ok_short(rng_hi):
            self._touched_bear = False
            self._range_hi_bear = self._range_lo_bear = None
            self._anchor_hi_bear = None
            return
        self._pending_short = True
        self._pending_trigger = rng_lo
        self._pending_sl = rng_hi
        self._hunt_bear = self._touched_bear = False
        self._range_hi_bear = self._range_lo_bear = None
        self._anchor_hi_bear = None

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
