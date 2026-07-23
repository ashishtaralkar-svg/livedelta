"""DCv2Engine: signal -> executor-call translation, 70% decay TP, strategy-exit
close, intracandle SL, 17:25 square-off / Friday-flat, 17:30 rollover, self-heal,
and weekend entry block. Fakes isolate the engine's own logic from REST/executor
internals (same style as test_dchannel_trader.py)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from deltabot.config import Settings
from deltabot.core.dcv2_trader import DCv2Engine
from deltabot.enums import NotifyEvent, PositionState, SignalDir
from deltabot.models import Candle

_ist = ZoneInfo("Asia/Kolkata")


class FakeExecutor:
    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.underlying = "BTC"
        self.open_calls: list[tuple[int, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (900.0, "P-BTC-60000-070726")
        self._close_result: float | None = 270.0

    async def open_option_by_premium(self, signal_dir: int, target_premium: float):
        self.open_calls.append((signal_dir, target_premium))
        fill, symbol = self._open_result
        if fill is not None:
            self.has_open_position = True
            self.tracked_symbol = symbol
            self.tracked_product_id = 123
        return fill, symbol

    async def close_option(self):
        self.close_calls += 1
        self.has_open_position = False
        self.tracked_symbol = None
        return self._close_result

    def clear(self) -> None:
        self.has_open_position = False
        self.tracked_symbol = None
        self.tracked_product_id = None

    def adopt(self, product_id, size, option_type, symbol=None) -> None:
        self.has_open_position = True
        self.tracked_product_id = product_id
        self.tracked_symbol = symbol


class FakeRest:
    def __init__(self, positions=None, mark=None) -> None:
        self._positions = positions or []
        self._mark = mark

    def get_option_positions(self, underlying):
        return self._positions

    def get_mark_price(self, symbol):
        return self._mark


def _make_engine(**kw) -> DCv2Engine:
    base = dict(strategy="dcv2", target_premium=900.0, take_profit_pct=70.0,
                option_contracts=25, state_file="")
    base.update(kw)
    settings = Settings(_env_file=None, **base)
    engine = DCv2Engine(settings, rest=FakeRest(), notifier=AsyncMock())
    engine.executor = FakeExecutor()
    return engine


def _c(start: int, o=100.0, h=101.0, low=99.0, cl=100.0) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _exit_calls(notifier):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == NotifyEvent.EXIT]


# ---------------------------------------------------------------------- #
# Entry: signal -> option side + 70% TP level
# ---------------------------------------------------------------------- #
async def test_buy_signal_sells_put_and_sets_70pct_tp() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, sl_level=59000.0, btc_price=60000.0, tag="ENTRY")
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 900.0)]
    assert engine._entry_premium == 900.0
    assert engine._tp_price == pytest.approx(270.0)   # 900 * 0.30
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_LONG and ev.kwargs["direction"] == "PUT"


async def test_sell_signal_sells_call() -> None:
    engine = _make_engine()
    engine.executor._open_result = (900.0, "C-BTC-60000-070726")
    await engine._open_entry(SignalDir.SHORT.value, 61000.0, 60000.0, tag="ENTRY")
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_SHORT and ev.kwargs["direction"] == "CALL"


async def test_open_entry_guarded_when_already_open() -> None:
    engine = _make_engine()
    engine.executor.has_open_position = True
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    assert engine.executor.open_calls == []


async def test_no_fill_flattens_strategy() -> None:
    engine = _make_engine()
    engine.executor._open_result = (None, None)
    engine.strategy._in_long = True   # pretend the strategy went long
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    assert engine._entry_premium is None
    assert engine.strategy.position_state == PositionState.FLAT


# ---------------------------------------------------------------------- #
# Exits
# ---------------------------------------------------------------------- #
async def test_close_leg_clears_tracking_and_notifies_reason() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.executor._close_result = 1100.0
    await engine._close_leg("SL", btc_exit_price=59000.0)
    assert engine.executor.close_calls == 1
    assert engine._entry_premium is None and engine._tp_price is None
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "SL"


async def test_close_tp_flattens_strategy_so_it_waits_for_new_signal() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    engine.executor._close_result = 270.0
    await engine._close_tp(270.0)
    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.FLAT   # force_flat -> hunt fresh
    assert engine._entry_premium is None and engine._tp_price is None
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "TP"


async def test_double_close_guard() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine._closing = True
    await engine._close_leg("SL", 59000.0)
    assert engine.executor.close_calls == 0


# ---------------------------------------------------------------------- #
# Intracandle SL (ASAP): touching the fixed range level closes the leg NOW
# ---------------------------------------------------------------------- #
async def test_intracandle_sl_closes_leg_and_flattens() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    # Pretend the strategy is long with SL at 59000; feed a sub-SL forming candle.
    engine.strategy._in_long = True
    engine.strategy._sl_level = 59000.0
    engine.strategy._warmup_bars = 10_000
    for _ in range(engine.strategy.dc_period):
        engine.strategy._dc.push(60000.0, 60000.0)
    await engine._handle_forming_candle(_c(0, 58950.0, 58990.0, 58900.0, 58950.0))
    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.FLAT
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "SL"


# ---------------------------------------------------------------------- #
# 17:25 square-off + Friday-flat
# ---------------------------------------------------------------------- #
def _fake_now(monkeypatch, dt):
    import deltabot.core.dcv2_trader as mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return dt
    monkeypatch.setattr(mod, "datetime", _FakeDatetime)


async def test_square_off_eod_closes_leg_but_keeps_direction(monkeypatch) -> None:
    """Mon-Thu 17:25: close the option, but leave the directional trade open so
    the 17:30 rollover re-sells it."""
    engine = _make_engine(skip_weekdays="Sat,Sun")
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    _fake_now(monkeypatch, datetime(2026, 7, 8, 17, 25, tzinfo=_ist))  # Wed
    await engine._square_off()
    assert engine.executor.close_calls == 1                      # option closed
    assert engine.strategy.position_state == PositionState.LONG  # trade NOT flattened
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "EOD"


async def test_square_off_friday_flattens_the_trade(monkeypatch) -> None:
    engine = _make_engine(skip_weekdays="Sat,Sun")
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    _fake_now(monkeypatch, datetime(2026, 7, 10, 17, 25, tzinfo=_ist))  # Fri (Sat tomorrow)
    await engine._square_off()
    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.FLAT   # weekend-flat
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "WEEKEND"


async def test_square_off_continuous_roll_immediately_resells(monkeypatch) -> None:
    """dcv2_continuous_roll=True, Mon-Thu: close AND re-sell in the SAME
    _square_off() call -- no waiting for the 17:30 gap to clear."""
    engine = _make_engine(skip_weekdays="Sat,Sun", dcv2_continuous_roll=True)
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    _fake_now(monkeypatch, datetime(2026, 7, 8, 17, 25, tzinfo=_ist))  # Wed
    await engine._square_off()
    assert engine.executor.close_calls == 1
    assert engine.executor.has_open_position   # re-sold immediately
    assert engine.strategy.position_state == PositionState.LONG   # directional trade untouched
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "ROLL"   # not "EOD"
    entries = [c for c in engine.notifier.notify.await_args_list
               if c.args and c.args[0] == NotifyEvent.ENTRY_LONG]
    assert entries and entries[-1].kwargs["tag"] == "ROLL"


async def test_square_off_continuous_roll_friday_still_flattens(monkeypatch) -> None:
    """Friday-flat takes priority over continuous-roll: no immediate re-sell."""
    engine = _make_engine(skip_weekdays="Sat,Sun", dcv2_continuous_roll=True)
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    _fake_now(monkeypatch, datetime(2026, 7, 10, 17, 25, tzinfo=_ist))  # Fri (Sat tomorrow)
    await engine._square_off()
    assert engine.executor.close_calls == 1
    assert not engine.executor.has_open_position
    assert engine.strategy.position_state == PositionState.FLAT
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "WEEKEND"


async def test_entries_blocked_ignores_gap_when_continuous_roll(monkeypatch) -> None:
    """With continuous_roll on, entries are never blocked by the 17:25-17:30
    same-day gap (only weekday blocking still applies)."""
    engine = _make_engine(skip_weekdays="Sat,Sun", dcv2_continuous_roll=True)
    engine._sq_off_date = datetime(2026, 7, 8, tzinfo=_ist).date()
    _fake_now(monkeypatch, datetime(2026, 7, 8, 17, 26, tzinfo=_ist))  # 1 min after square-off
    assert engine._entries_blocked() is False

    _fake_now(monkeypatch, datetime(2026, 7, 11, 12, 0, tzinfo=_ist))  # Sat -> still blocked
    assert engine._entries_blocked() is True


async def test_entries_blocked_respects_gap_by_default(monkeypatch) -> None:
    """Default (continuous_roll=False): the same-day 17:25-17:30 gap still
    blocks entries, matching today's live behavior."""
    engine = _make_engine(skip_weekdays="Sat,Sun")
    engine._sq_off_date = datetime(2026, 7, 8, tzinfo=_ist).date()
    _fake_now(monkeypatch, datetime(2026, 7, 8, 17, 26, tzinfo=_ist))  # inside the gap
    assert engine._entries_blocked() is True


# ---------------------------------------------------------------------- #
# 17:30 rollover: trade still open + option flat -> re-sell
# ---------------------------------------------------------------------- #
async def test_rollover_resells_when_trade_open_but_option_flat() -> None:
    engine = _make_engine(skip_weekdays="")   # no weekday block -> entries not blocked
    engine._sq_off_date = None
    # Mock the strategy: no closed-bar decision, but still LONG with an SL.
    strat = MagicMock()
    strat.update.return_value = None
    strat.position_state = PositionState.LONG
    strat.sl_level = 59000.0
    engine.strategy = strat
    await engine._handle_closed_candle(_c(300))
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 900.0)]   # rolled


async def test_no_rollover_when_flat() -> None:
    engine = _make_engine(skip_weekdays="")
    engine._sq_off_date = None
    strat = MagicMock()
    strat.update.return_value = None
    strat.position_state = PositionState.FLAT
    engine.strategy = strat
    await engine._handle_closed_candle(_c(300))
    assert engine.executor.open_calls == []


# ---------------------------------------------------------------------- #
# Self-heal + weekend entry block
# ---------------------------------------------------------------------- #
class VerifyRest(FakeRest):
    def __init__(self, positions=None, raises=False):
        super().__init__(positions=positions)
        self.raises = raises

    def get_option_positions(self, underlying):
        if self.raises:
            raise RuntimeError("flaky api")
        return self._positions


async def _open_and_arm(engine):
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.executor.tracked_product_id = 123
    engine._last_verify = 0.0


async def test_selfheal_flattens_after_two_misses() -> None:
    engine = _make_engine(position_verify_seconds=0.0001)
    await _open_and_arm(engine)
    engine.rest = VerifyRest(positions=[])
    await engine._maybe_verify_position()
    assert engine.executor.has_open_position          # 1st miss: warn only
    engine._last_verify = 0.0
    await engine._maybe_verify_position()
    assert not engine.executor.has_open_position       # 2nd miss: self-heal
    assert engine._entry_premium is None


async def test_selfheal_never_drops_on_fetch_error() -> None:
    engine = _make_engine(position_verify_seconds=0.0001)
    await _open_and_arm(engine)
    engine.rest = VerifyRest(raises=True)
    for _ in range(5):
        engine._last_verify = 0.0
        await engine._maybe_verify_position()
    assert engine.executor.has_open_position           # flaky API never flattens


def test_entries_blocked_on_weekend(monkeypatch) -> None:
    engine = _make_engine(skip_weekdays="Sat,Sun")
    _fake_now(monkeypatch, datetime(2026, 7, 11, 12, 0, tzinfo=_ist))  # Saturday
    assert engine._entries_blocked() is True


def test_entries_not_blocked_on_weekday(monkeypatch) -> None:
    engine = _make_engine(skip_weekdays="Sat,Sun")
    _fake_now(monkeypatch, datetime(2026, 7, 8, 12, 0, tzinfo=_ist))  # Wednesday
    assert engine._entries_blocked() is False
