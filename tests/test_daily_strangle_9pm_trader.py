"""DailyStrangle9pmEngine: the live time-based SELL CE+PE strangle engine.
Both legs always open/close TOGETHER as one combined position (unlike
supertrend_trader.py's genuinely independent legs) -- see the module
docstring in src/deltabot/core/daily_strangle_9pm_trader.py."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deltabot.config import Settings
from deltabot.core.daily_strangle_9pm_trader import DailyStrangle9pmEngine
from deltabot.enums import NotifyEvent, SignalDir
from deltabot.models import Candle


class FakeExecutor:
    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.underlying = "BTC"
        self.is_buy_side = False
        self.open_calls: list[tuple[int, float, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (100.0, "C-BTC-64000-070826")
        self._close_result: float | None = 60.0   # decayed, for profit tests

    async def open_option_by_otm_pct(self, signal_dir: int, btc_price: float, otm_pct: float):
        self.open_calls.append((signal_dir, btc_price, otm_pct))
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
    def __init__(self, positions=None, marks=None, candles=None) -> None:
        self._positions = positions or []
        self._marks = marks or {}
        self._candles = candles if candles is not None else [Candle(0, 60000, 60100, 59900, 60000, 1.0)]

    def get_option_positions(self, underlying):
        return self._positions

    def get_mark_price(self, symbol):
        return self._marks.get(symbol)

    def get_candles(self, symbol, resolution, start, end):
        return self._candles


def _make_engine(**kw) -> DailyStrangle9pmEngine:
    base = dict(strategy="strangle9pm", option_contracts=10, state_file="",
                strangle9pm_otm_pct=2.0, strangle9pm_target_pct=70.0, strangle9pm_sl_pct=50.0)
    base.update(kw)
    settings = Settings(_env_file=None, **base)
    rest = FakeRest()
    engine = DailyStrangle9pmEngine(settings, rest=rest, notifier=AsyncMock())
    engine.executor_ce = FakeExecutor()
    engine.executor_pe = FakeExecutor()
    return engine


def _entry_calls(notifier, event):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == event]


# ---------------------------------------------------------------------- #
# Entry: sells both CE and PE at the same spot/otm_pct, records combined entry
# ---------------------------------------------------------------------- #
async def test_maybe_enter_opens_both_legs_with_correct_signal_dirs() -> None:
    engine = _make_engine()
    engine.executor_pe._open_result = (80.0, "P-BTC-59200-070826")
    await engine._maybe_enter()
    assert engine.executor_ce.open_calls == [(SignalDir.SHORT.value, 60000.0, 2.0)]
    assert engine.executor_pe.open_calls == [(SignalDir.LONG.value, 60000.0, 2.0)]
    assert engine.entry_premium_ce == 100.0
    assert engine.entry_premium_pe == 80.0
    assert engine.combined_entry_premium == 180.0


async def test_maybe_enter_sends_both_entry_notifications() -> None:
    engine = _make_engine()
    await engine._maybe_enter()
    shorts = _entry_calls(engine.notifier, NotifyEvent.ENTRY_SHORT)
    longs = _entry_calls(engine.notifier, NotifyEvent.ENTRY_LONG)
    assert len(shorts) == 1 and shorts[0].kwargs["direction"] == "CALL"
    assert len(longs) == 1 and longs[0].kwargs["direction"] == "PUT"


async def test_maybe_enter_skipped_when_strangle_already_open() -> None:
    engine = _make_engine()
    engine.executor_ce.has_open_position = True
    await engine._maybe_enter()
    assert engine.executor_ce.open_calls == []
    assert engine.executor_pe.open_calls == []


async def test_maybe_enter_skipped_when_no_spot_price() -> None:
    engine = _make_engine()
    engine.rest._candles = []
    await engine._maybe_enter()
    assert engine.executor_ce.open_calls == []


async def test_ce_fill_failure_aborts_entry_without_opening_pe() -> None:
    engine = _make_engine()
    engine.executor_ce._open_result = (None, None)
    await engine._maybe_enter()
    assert engine.executor_pe.open_calls == []
    assert engine.entry_premium_ce is None
    assert engine.entry_premium_pe is None


async def test_pe_fill_failure_unwinds_the_now_unhedged_ce_leg() -> None:
    engine = _make_engine()
    engine.executor_pe._open_result = (None, None)
    await engine._maybe_enter()
    assert engine.executor_ce.close_calls == 1
    assert engine.entry_premium_ce is None
    assert engine.entry_premium_pe is None
    exits = _entry_calls(engine.notifier, NotifyEvent.EXIT)
    assert any(c.kwargs.get("reason") == "PE_FAILED" for c in exits)


# ---------------------------------------------------------------------- #
# Combined premium TARGET / SL
# ---------------------------------------------------------------------- #
async def _open_strangle(engine, ce_entry=100.0, pe_entry=100.0):
    engine.executor_ce.has_open_position = True
    engine.executor_ce.tracked_symbol = "C-BTC-64000-070826"
    engine.executor_pe.has_open_position = True
    engine.executor_pe.tracked_symbol = "P-BTC-59200-070826"
    engine.entry_premium_ce = ce_entry
    engine.entry_premium_pe = pe_entry


async def test_check_target_sl_closes_both_legs_on_combined_target_hit() -> None:
    engine = _make_engine(strangle9pm_target_pct=70.0)
    await _open_strangle(engine, ce_entry=100.0, pe_entry=100.0)   # combined entry 200
    engine.rest._marks = {"C-BTC-64000-070826": 30.0, "P-BTC-59200-070826": 29.0}   # combined 59 <= 60 (30% of 200)
    await engine._check_target_sl()
    assert engine.executor_ce.close_calls == 1
    assert engine.executor_pe.close_calls == 1
    exits = _entry_calls(engine.notifier, NotifyEvent.EXIT)
    assert all(c.kwargs.get("reason") == "TARGET" for c in exits)


async def test_check_target_sl_closes_both_legs_on_combined_sl_hit() -> None:
    engine = _make_engine(strangle9pm_sl_pct=50.0)
    await _open_strangle(engine, ce_entry=100.0, pe_entry=100.0)   # combined entry 200, SL at 300
    engine.rest._marks = {"C-BTC-64000-070826": 160.0, "P-BTC-59200-070826": 145.0}   # combined 305 >= 300
    await engine._check_target_sl()
    assert engine.executor_ce.close_calls == 1
    assert engine.executor_pe.close_calls == 1
    exits = _entry_calls(engine.notifier, NotifyEvent.EXIT)
    assert all(c.kwargs.get("reason") == "SL" for c in exits)


async def test_check_target_sl_no_op_when_neither_threshold_crossed() -> None:
    engine = _make_engine()
    await _open_strangle(engine, ce_entry=100.0, pe_entry=100.0)
    engine.rest._marks = {"C-BTC-64000-070826": 95.0, "P-BTC-59200-070826": 95.0}   # combined 190, between levels
    await engine._check_target_sl()
    assert engine.executor_ce.close_calls == 0
    assert engine.executor_pe.close_calls == 0


async def test_check_target_sl_no_op_when_flat() -> None:
    engine = _make_engine()
    await engine._check_target_sl()
    assert engine.executor_ce.close_calls == 0


async def test_check_target_sl_no_op_on_missing_mark_price() -> None:
    engine = _make_engine()
    await _open_strangle(engine)
    engine.rest._marks = {}   # both marks missing
    await engine._check_target_sl()
    assert engine.executor_ce.close_calls == 0


# ---------------------------------------------------------------------- #
# _close_strangle P&L sign (SELL side: profit = entry - exit)
# ---------------------------------------------------------------------- #
async def test_close_strangle_computes_sell_side_pnl_correctly() -> None:
    engine = _make_engine(option_contracts=10)
    await _open_strangle(engine, ce_entry=100.0, pe_entry=90.0)
    engine.executor_ce._close_result = 60.0   # decayed -- profit
    engine.executor_pe._close_result = 50.0
    await engine._close_strangle("TARGET")
    exits = _entry_calls(engine.notifier, NotifyEvent.EXIT)
    ce_exit = next(c for c in exits if c.kwargs["side"] == "sell CE")
    pe_exit = next(c for c in exits if c.kwargs["side"] == "sell PE")
    assert ce_exit.kwargs["pnl"] == pytest.approx((100.0 - 60.0) * 10 * 0.001)
    assert pe_exit.kwargs["pnl"] == pytest.approx((90.0 - 50.0) * 10 * 0.001)
    assert engine.entry_premium_ce is None and engine.entry_premium_pe is None


async def test_close_strangle_is_reentrant_safe_while_already_closing() -> None:
    engine = _make_engine()
    await _open_strangle(engine)
    engine.closing = True
    await engine._close_strangle("TARGET")
    assert engine.executor_ce.close_calls == 0   # guarded, no double-close


# ---------------------------------------------------------------------- #
# Reconcile
# ---------------------------------------------------------------------- #
async def test_reconcile_adopts_both_legs_by_symbol_prefix() -> None:
    engine = _make_engine()
    engine.rest._positions = [
        {"size": -10, "symbol": "C-BTC-64000-070826", "product_id": 1},
        {"size": -10, "symbol": "P-BTC-59200-070826", "product_id": 2},
    ]
    await engine._sync_options_to_exchange()
    assert engine.executor_ce.has_open_position and engine.executor_ce.tracked_symbol == "C-BTC-64000-070826"
    assert engine.executor_pe.has_open_position and engine.executor_pe.tracked_symbol == "P-BTC-59200-070826"


async def test_reconcile_leaves_both_flat_when_exchange_empty_and_no_state() -> None:
    engine = _make_engine()
    engine.rest._positions = []
    await engine._sync_options_to_exchange()
    assert not engine.executor_ce.has_open_position
    assert not engine.executor_pe.has_open_position
    assert engine.entry_premium_ce is None and engine.entry_premium_pe is None


async def test_has_open_strangle_true_if_either_leg_open() -> None:
    engine = _make_engine()
    assert not engine.has_open_strangle
    engine.executor_ce.has_open_position = True
    assert engine.has_open_strangle
