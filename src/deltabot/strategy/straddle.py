"""Straddle4pm Strategy -- a fixed-TIME daily long straddle, no BTC-price
signal at all (the only strategy in this repo that isn't indicator-driven).

At a configured time each day (default 16:00 IST) this fires an entry
trigger exactly once. The caller (live trader / backtest script) then BUYS
a CALL and a PUT near ``target_premium`` (e.g. ~100 each) as two
independent legs -- a classic long straddle: no directional bias, a bet
that BTC makes a big enough move (either way) before the daily square-off
for one side to pay for both.

EXIT (entirely a caller-side concern, same separation as
supertrend_fixed_sl.py's own TP -- this class only sees BTC candles, never
option premiums):
  * Whichever leg's premium first reaches the exit target (a FIXED ABSOLUTE
    value, e.g. 250 -- NOT a percentage of entry like every other TP in
    this repo, because entry premium is already normalized to ~100 by the
    target_premium selection) is closed at target, and the OTHER leg is
    closed alongside it immediately at whatever it's currently worth. Once
    one side has paid off, the straddle's whole thesis is resolved -- there
    is no reason to keep holding the other leg hoping for a double payout.
  * If NEITHER leg reaches target, the daily square-off (17:25 IST,
    matching every other bot) closes both at market.
  * NO stop-loss on either leg individually -- max loss per leg is
    naturally capped at the premium paid (buying, not selling), same
    buy-side risk profile as ema21bot.

See core/straddle_trader.py for the live engine, scripts/backtest_straddle.py
for the backtest. Both call this class's ``update()`` every closed candle
and act on a ``True`` return by buying both legs.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from ..models import Candle


class StraddleStrategy:
    """Fires an entry trigger at most once per calendar day, the first
    closed candle whose local time falls in
    [entry_hour:entry_minute, entry_hour:entry_minute + entry_grace_minutes).

    The grace window exists so a bot that was down through the exact entry
    bar (restart, gap) doesn't fire hours late later that same day -- it
    just skips that day's entry instead, same spirit as every other
    strategy's warmup/gap handling.
    """

    def __init__(
        self,
        entry_hour: int = 16,
        entry_minute: int = 0,
        entry_grace_minutes: int = 15,
        day_tz: str = "Asia/Kolkata",
    ) -> None:
        self.entry_hour = entry_hour
        self.entry_minute = entry_minute
        self.entry_grace_minutes = entry_grace_minutes
        self._tz = ZoneInfo(day_tz)
        self._last_fired_date: date | None = None

    @property
    def entry_start_mins(self) -> int:
        return self.entry_hour * 60 + self.entry_minute

    def update(self, candle: Candle) -> bool:
        """Return True on this candle if today's entry should fire now."""
        local = datetime.fromtimestamp(candle.start_time, tz=self._tz)
        if local.date() == self._last_fired_date:
            return False
        mins = local.hour * 60 + local.minute
        start = self.entry_start_mins
        if start <= mins < start + self.entry_grace_minutes:
            self._last_fired_date = local.date()
            return True
        return False

    def unfire_today(self) -> None:
        """Undo today's fire (e.g. the caller's attempt to buy both legs
        failed) so a LATER candle still inside today's grace window can
        retry. A no-op once the grace window has already closed."""
        self._last_fired_date = None
