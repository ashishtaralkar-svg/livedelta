"""DailyStrangle9pmLadderedEngine: the live 6-leg (3 OTM%% tiers x CE+PE)
SELL strangle engine -- generalization of test_daily_strangle_9pm_trader.py
to N legs, all opened/closed together as ONE combined position. See the
module docstring in src/deltabot/core/daily_strangle_9pm_laddered_trader.py."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from deltabot.config import Settings
from deltabot.core.daily_strangle_9pm_laddered_trader import DailyStrangle9pmLadderedEngine
from deltabot.enums import NotifyEvent, SignalDir
from deltabot.models import Candle


class FakeExecutor:
    def __init__(self, symbol: str = "C-BTC-64000-070826", premium: float = 100.0) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.underlying = "BTC"
        self.is_buy_side = False
        self.open_calls: list[tuple[int, float, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (premium, symbol)
        self._close_result: float | None = premium * 0.3   # decayed, profit by default

    async def open_option_by_otm_pct(self, signal_dir: int, btc_price: float, otm_pct: float):
        self.open_calls.append((signal_dir, btc_price, otm_pct))
        fill, symbol = self._open_result
        if fill is not None:
            self.has_open_position = True
            self.tracked_symbol = symbol
            self.tracked_product_id = id(self) % 100000
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
    def __init__(self, positions=None, marks=None, candles=None, balance=1000.0) -> None:
        self._positions = positions or []
        self._marks = marks or {}
        self._candles = candles if candles is not None else [Candle(0, 60000, 60100, 59900, 60000, 1.0)]
        self._candles_by_symbol: dict[str, list] = {}
        self._balance = balance

    def get_option_positions(self, underlying):
        return self._positions

    def get_mark_price(self, symbol):
        return self._marks.get(symbol)

    def get_candles(self, symbol, resolution, start, end):
        return self._candles_by_symbol.get(symbol, self._candles)

    def get_available_balance(self, asset_symbol=None):
        return self._balance


_SYMS = {
    "ce_1": "C-BTC-65000-070826", "pe_1": "P-BTC-63200-070826",
    "ce_1_5": "C-BTC-65400-070826", "pe_1_5": "P-BTC-62800-070826",
    "ce_2": "C-BTC-65800-070826", "pe_2": "P-BTC-62400-070826",
}


def _make_engine(**kw) -> DailyStrangle9pmLadderedEngine:
    base = dict(strategy="strangle9pmladder", option_contracts=10, state_file="",
                strangle9pm_laddered_otm_pcts="1,1.5,2", strangle9pm_target_pct=70.0, strangle9pm_sl_pct=105.0)
    base.update(kw)
    settings = Settings(_env_file=None, **base)
    rest = FakeRest()
    engine = DailyStrangle9pmLadderedEngine(settings, rest=rest, notifier=AsyncMock())
    for leg in engine.legs:
        prefix = "C" if leg.is_call else "P"
        sym = _SYMS.get(leg.name, f"{prefix}-BTC-99999-070826-{leg.name}")
        leg.executor = FakeExecutor(symbol=sym, premium=100.0)
    return engine


def _entry_calls(notifier, event):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == event]


async def _open_ladder(engine, entry_each: float = 100.0) -> None:
    for leg in engine.legs:
        leg.executor.has_open_position = True
        leg.executor.tracked_symbol = _SYMS[leg.name]
        leg.entry_premium = entry_each


# ---------------------------------------------------------------------- #
# Construction
# ---------------------------------------------------------------------- #
def test_engine_builds_six_legs_for_default_three_tiers() -> None:
    engine = _make_engine()
    assert len(engine.legs) == 6
    names = {leg.name for leg in engine.legs}
    assert names == {"ce_1", "pe_1", "ce_1_5", "pe_1_5", "ce_2", "pe_2"}
    assert all(leg.state_file == "" for leg in engine.legs)   # state persistence off (empty base)


def test_engine_builds_custom_tier_count() -> None:
    engine = _make_engine(strangle9pm_laddered_otm_pcts="1,2,3,4")
    assert len(engine.legs) == 8


# ---------------------------------------------------------------------- #
# Entry
# ---------------------------------------------------------------------- #
async def test_maybe_enter_opens_all_six_legs_with_correct_otm_and_side() -> None:
    engine = _make_engine()
    await engine._maybe_enter()
    by_name = {leg.name: leg for leg in engine.legs}
    assert by_name["ce_1"].executor.open_calls == [(SignalDir.SHORT.value, 60000.0, 1.0)]
    assert by_name["pe_1"].executor.open_calls == [(SignalDir.LONG.value, 60000.0, 1.0)]
    assert by_name["ce_1_5"].executor.open_calls == [(SignalDir.SHORT.value, 60000.0, 1.5)]
    assert by_name["pe_2"].executor.open_calls == [(SignalDir.LONG.value, 60000.0, 2.0)]
    assert engine.combined_entry_premium == 600.0   # 6 legs x 100


async def test_maybe_enter_skipped_when_strangle_already_open() -> None:
    engine = _make_engine()
    engine.legs[0].executor.has_open_position = True
    await engine._maybe_enter()
    assert all(not leg.executor.open_calls for leg in engine.legs)


async def test_maybe_enter_skipped_when_no_spot_price() -> None:
    engine = _make_engine()
    engine.rest._candles = []
    await engine._maybe_enter()
    assert all(not leg.executor.open_calls for leg in engine.legs)


async def test_leg_fill_failure_unwinds_all_previously_opened_legs() -> None:
    engine = _make_engine()
    # Fail the 3rd leg (ce_1_5) -- the first two (ce_1, pe_1) already filled.
    engine.legs[2].executor._open_result = (None, None)
    await engine._maybe_enter()
    assert engine.legs[0].executor.close_calls == 1
    assert engine.legs[1].executor.close_calls == 1
    assert engine.legs[0].entry_premium is None
    assert engine.legs[1].entry_premium is None
    exits = _entry_calls(engine.notifier, NotifyEvent.EXIT)
    assert any(c.kwargs.get("reason") == "LEG_FAILED" for c in exits)


# ---------------------------------------------------------------------- #
# Combined premium TARGET / SL
# ---------------------------------------------------------------------- #
async def test_check_target_sl_closes_all_legs_on_combined_target_hit() -> None:
    engine = _make_engine(strangle9pm_target_pct=70.0)
    await _open_ladder(engine, entry_each=100.0)   # combined entry 600, target <= 180
    for leg in engine.legs:
        engine.rest._marks[_SYMS[leg.name]] = 25.0   # combined 150 <= 180
    await engine._check_target_sl()
    assert all(leg.executor.close_calls == 1 for leg in engine.legs)
    exits = _entry_calls(engine.notifier, NotifyEvent.EXIT)
    assert len(exits) == 6
    assert all(c.kwargs.get("reason") == "TARGET" for c in exits)


async def test_check_target_sl_closes_all_legs_on_combined_sl_hit() -> None:
    engine = _make_engine(strangle9pm_sl_pct=105.0)
    await _open_ladder(engine, entry_each=100.0)   # combined entry 600, SL >= 1230
    for leg in engine.legs:
        engine.rest._marks[_SYMS[leg.name]] = 210.0   # combined 1260 >= 1230
    await engine._check_target_sl()
    assert all(leg.executor.close_calls == 1 for leg in engine.legs)


async def test_check_target_sl_no_op_when_neither_threshold_crossed() -> None:
    engine = _make_engine()
    await _open_ladder(engine, entry_each=100.0)
    for leg in engine.legs:
        engine.rest._marks[_SYMS[leg.name]] = 95.0   # combined 570, between levels
    await engine._check_target_sl()
    assert all(leg.executor.close_calls == 0 for leg in engine.legs)


async def test_check_target_sl_no_op_when_flat() -> None:
    engine = _make_engine()
    await engine._check_target_sl()
    assert all(leg.executor.close_calls == 0 for leg in engine.legs)


# ---------------------------------------------------------------------- #
# Dynamic lot sizing: divide by number of TIERS (not legs), per request
# ---------------------------------------------------------------------- #
async def test_dynamic_sizing_off_by_default_keeps_static_lots() -> None:
    engine = _make_engine(option_contracts=10)
    engine.rest._balance = 500.0
    lots = await engine._maybe_recompute_lots()
    assert lots == 10
    assert engine.settings.option_contracts == 10


async def test_dynamic_sizing_divides_by_number_of_tiers() -> None:
    engine = _make_engine(option_contracts=10, strangle9pm_compound_capital=True,
                          strangle9pm_capital_per_lot=1.0, strangle9pm_max_lots=1000)
    engine.rest._balance = 300.0   # base_lots = 300, 3 tiers -> 100 per tier
    lots = await engine._maybe_recompute_lots()
    assert lots == 100
    assert engine.settings.option_contracts == 100


async def test_dynamic_sizing_floors_at_1_lot_per_tier() -> None:
    engine = _make_engine(option_contracts=10, strangle9pm_compound_capital=True,
                          strangle9pm_capital_per_lot=1.0, strangle9pm_max_lots=1000)
    engine.rest._balance = 2.0   # base_lots = 2, // 3 tiers = 0 -> floored to 1
    lots = await engine._maybe_recompute_lots()
    assert lots == 1


async def test_dynamic_sizing_respects_max_lots_before_dividing() -> None:
    engine = _make_engine(option_contracts=10, strangle9pm_compound_capital=True,
                          strangle9pm_capital_per_lot=1.0, strangle9pm_max_lots=90)
    engine.rest._balance = 5000.0   # raw_lots huge, capped to 90, then // 3 = 30
    lots = await engine._maybe_recompute_lots()
    assert lots == 30


# ---------------------------------------------------------------------- #
# Reconcile
# ---------------------------------------------------------------------- #
async def test_reconcile_adopts_all_six_legs_by_symbol_prefix_when_no_state() -> None:
    engine = _make_engine()
    engine.rest._positions = [
        {"size": -10, "symbol": sym, "product_id": i}
        for i, sym in enumerate(_SYMS.values())
    ]
    await engine._sync_options_to_exchange()
    assert all(leg.executor.has_open_position for leg in engine.legs)


async def test_reconcile_leaves_all_flat_when_exchange_empty_and_no_state() -> None:
    engine = _make_engine()
    engine.rest._positions = []
    await engine._sync_options_to_exchange()
    assert all(not leg.executor.has_open_position for leg in engine.legs)
    assert all(leg.entry_premium is None for leg in engine.legs)


async def test_has_open_strangle_true_if_any_leg_open() -> None:
    engine = _make_engine()
    assert not engine.has_open_strangle
    engine.legs[3].executor.has_open_position = True
    assert engine.has_open_strangle
