"""RangeEngulfingFadeSellEngine: the live sell-only, INTRACANDLE engine for
RangeEngulfingFadeSellStrategy. Mirrors test_ema21_trader.py/test_supertrend_
trader.py's structure, but exercises the genuinely new part of this engine:
_on_forming_candle firing entry/exit off REAL-TIME ticks (not just at candle
close), and ATM execution via open_option() instead of a premium target."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from deltabot.config import Settings
from deltabot.core.range_engulfing_fade_sell_trader import RangeEngulfingFadeSellEngine
from deltabot.enums import NotifyEvent, SignalDir
from deltabot.models import Candle

_ist = ZoneInfo("Asia/Kolkata")


class FakeExecutor:
    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.tracked_size = 0
        self.underlying = "BTC"
        self.is_buy_side = False
        self.open_calls: list[tuple[int, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (200.0, "C-BTC-64000-070826")
        self._close_result: float | None = 100.0   # decayed, for profit tests

    async def open_option(self, signal_dir: int, btc_price: float):
        self.open_calls.append((signal_dir, btc_price))
        fill, symbol = self._open_result
        if fill is not None:
            self.has_open_position = True
            self.tracked_symbol = symbol
            self.tracked_product_id = 123
            self.tracked_size = 10
        return fill

    async def close_option(self):
        self.close_calls += 1
        self.has_open_position = False
        self.tracked_symbol = None
        return self._close_result

    def clear(self) -> None:
        self.has_open_position = False
        self.tracked_symbol = None
        self.tracked_product_id = None
        self.tracked_size = 0

    def adopt(self, product_id, size, option_type, symbol=None) -> None:
        self.has_open_position = True
        self.tracked_product_id = product_id
        self.tracked_symbol = symbol
        self.tracked_size = abs(size)


class FakeRest:
    def __init__(self, positions=None) -> None:
        self._positions = positions or []

    def get_option_positions(self, underlying):
        return self._positions


def _make_engine(**kw) -> RangeEngulfingFadeSellEngine:
    base = dict(strategy="range_fade", option_contracts=10, state_file="", skip_weekdays="")
    base.update(kw)
    settings = Settings(_env_file=None, **base)
    engine = RangeEngulfingFadeSellEngine(settings, rest=FakeRest(), notifier=AsyncMock())
    engine.executor = FakeExecutor()
    return engine


def _c(start: int, o=100.0, h=101.0, low=99.0, cl=100.0) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _exit_calls(notifier):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == NotifyEvent.EXIT]


async def _drain() -> None:
    """Let any asyncio.create_task()-scheduled coroutine actually run."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)


# ---------------------------------------------------------------------- #
# Closed candle: arming ONLY -- no entry/exit here (see module docstring).
# ---------------------------------------------------------------------- #
def test_closed_candle_arms_a_pattern_synchronously() -> None:
    engine = _make_engine()
    engine._on_closed_candle(_c(0, 100.0, 101.0, 98.0, 99.0))       # red
    engine._on_closed_candle(_c(900, 97.0, 105.0, 96.0, 103.0))     # green, engulfs
    st = engine.strategy.debug_state()
    assert st["pending_trigger"] == 105.0
    assert engine.executor.open_calls == []   # arming never opens anything


def test_closed_candle_does_not_spawn_a_task_on_the_happy_path() -> None:
    engine = _make_engine()
    engine._on_closed_candle(_c(0))
    assert engine._tasks == set()   # only a candle-gap re-seed would create one


# ---------------------------------------------------------------------- #
# Forming candle: the REAL-TIME reaction -- the whole point of this engine.
# ---------------------------------------------------------------------- #
async def test_forming_candle_fires_entry_the_instant_high_touches_trigger() -> None:
    engine = _make_engine()
    engine.strategy._pending_trigger = 105.0
    engine.strategy._pending_sl = 114.0
    engine.strategy._pending_target = 96.0

    engine._on_forming_candle(_c(900, 104.0, 106.0, 103.0, 104.5))   # high crosses 105 mid-candle
    assert engine.strategy.in_short   # mutated SYNCHRONOUSLY, before the order task even runs
    await _drain()

    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 105.0)]
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_SHORT and ev.kwargs["direction"] == "CALL"


async def test_forming_candle_does_not_fire_when_high_has_not_reached_trigger() -> None:
    engine = _make_engine()
    engine.strategy._pending_trigger = 105.0
    engine.strategy._pending_sl = 114.0
    engine.strategy._pending_target = 96.0

    engine._on_forming_candle(_c(900, 100.0, 103.0, 99.0, 102.0))   # high(103) < trigger(105)
    await _drain()
    assert not engine.strategy.in_short
    assert engine.executor.open_calls == []


async def test_forming_candle_fires_sl_exit_immediately() -> None:
    engine = _make_engine()
    await engine._open_entry(entry_trigger=64000.0, sl_level=65000.0)
    # _open_entry() itself never flips in_short -- that's check_intracandle_
    # entry()'s job in the real flow (it fires BEFORE _open_entry is even
    # scheduled). Set it directly here to simulate the post-fill state.
    engine.strategy._in_short = True
    engine.strategy._active_sl = 65000.0
    engine.strategy._active_target = 63000.0

    engine._on_forming_candle(_c(900, 64500.0, 65200.0, 64000.0, 65000.0))   # high touches 65000
    assert not engine.strategy.in_short   # mutated synchronously
    await _drain()

    assert engine.executor.close_calls == 1
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "SL"


async def test_forming_candle_ignores_entries_while_already_short() -> None:
    """A second forming tick after entry must not also check for a NEW
    entry (the strategy is no longer flat, so check_intracandle_exit --
    not check_intracandle_entry -- is what runs)."""
    engine = _make_engine()
    await engine._open_entry(entry_trigger=64000.0, sl_level=65000.0)
    engine.strategy._in_short = True   # see comment above
    engine.strategy._active_sl = 65000.0
    engine.strategy._active_target = 63000.0
    engine.strategy._pending_trigger = 70000.0   # some unrelated future setup, must stay untouched

    engine._on_forming_candle(_c(900, 64200.0, 64300.0, 64100.0, 64200.0))   # no SL/target touch
    await _drain()
    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 64000.0)]   # only the setup call
    assert engine.strategy._pending_trigger == 70000.0


async def test_forming_candle_respects_entries_blocked(monkeypatch) -> None:
    engine = _make_engine()
    engine.strategy._pending_trigger = 105.0
    engine.strategy._pending_sl = 114.0
    engine.strategy._pending_target = 96.0
    monkeypatch.setattr(engine, "_entries_blocked", lambda: True)

    engine._on_forming_candle(_c(900, 104.0, 106.0, 103.0, 104.5))
    await _drain()
    assert engine.executor.open_calls == []
    # pending trigger is untouched -- check_intracandle_entry was never called
    assert engine.strategy._pending_trigger == 105.0


# ---------------------------------------------------------------------- #
# Entry / exit P&L (SELL side: profit = entry - exit)
# ---------------------------------------------------------------------- #
async def test_open_entry_sells_atm_call() -> None:
    engine = _make_engine()
    await engine._open_entry(entry_trigger=64000.0, sl_level=65000.0)
    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 64000.0)]
    assert engine._entry_premium == 200.0
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_SHORT and ev.kwargs["direction"] == "CALL"


async def test_open_entry_guarded_when_already_open() -> None:
    engine = _make_engine()
    engine.executor.has_open_position = True
    await engine._open_entry(entry_trigger=64000.0, sl_level=65000.0)
    assert engine.executor.open_calls == []


async def test_no_fill_flattens_strategy() -> None:
    engine = _make_engine()
    engine.executor._open_result = (None, None)
    engine.strategy._in_short = True
    await engine._open_entry(entry_trigger=64000.0, sl_level=65000.0)
    assert engine._entry_premium is None
    assert not engine.strategy.in_short


async def test_close_leg_pnl_sign_profits_when_premium_decayed() -> None:
    engine = _make_engine()
    await engine._open_entry(entry_trigger=64000.0, sl_level=65000.0)
    engine.executor._close_result = 50.0   # decayed from 200 -> 50: a PROFIT
    await engine._close_leg("SL", 65000.0)
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["pnl"] > 0
    assert exits[-1].kwargs["reason"] == "SL"


async def test_double_close_guard() -> None:
    engine = _make_engine()
    await engine._open_entry(entry_trigger=64000.0, sl_level=65000.0)
    engine._closing = True
    await engine._close_leg("SL", 65000.0)
    assert engine.executor.close_calls == 0


# ---------------------------------------------------------------------- #
# Reconcile: single leg, sell-only, size < 0 (short)
# ---------------------------------------------------------------------- #
async def test_reconcile_adopts_a_short_position() -> None:
    engine = _make_engine()
    engine.rest = FakeRest([{"size": -10, "product_id": 55, "symbol": "C-BTC-64000-070826"}])
    await engine._sync_options_to_exchange()
    assert engine.executor.has_open_position
    assert engine.executor.tracked_product_id == 55


async def test_reconcile_flat_when_no_short_position() -> None:
    engine = _make_engine()
    engine.rest = FakeRest([])
    await engine._sync_options_to_exchange()
    assert not engine.executor.has_open_position
    assert not engine.strategy.in_short


# ---------------------------------------------------------------------- #
# 17:25 square-off
# ---------------------------------------------------------------------- #
def _fake_now(monkeypatch, dt):
    import deltabot.core.range_engulfing_fade_sell_trader as mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return dt
    monkeypatch.setattr(mod, "datetime", _FakeDatetime)


async def test_square_off_closes_open_position(monkeypatch) -> None:
    engine = _make_engine()
    await engine._open_entry(entry_trigger=64000.0, sl_level=65000.0)
    _fake_now(monkeypatch, datetime(2026, 7, 10, 17, 25, tzinfo=_ist))
    await engine._square_off()
    assert engine.executor.close_calls == 1
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "EOD"


async def test_square_off_noop_when_already_flat(monkeypatch) -> None:
    engine = _make_engine()
    _fake_now(monkeypatch, datetime(2026, 7, 10, 17, 25, tzinfo=_ist))
    await engine._square_off()
    assert engine.executor.close_calls == 0
