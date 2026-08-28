"""Ema21BreakdownEngine: the live buy-side engine for Ema21BreakdownStrategy.
Mirrors test_dcv3_trader.py's structure (same FakeExecutor/FakeRest shape,
same P&L-sign/reconcile/self-heal conventions -- this engine is a
closed-bar-only, no-rollover simplification of DCv3Engine, see the module
docstring in src/deltabot/core/ema21_trader.py)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from deltabot.config import Settings
from deltabot.core.ema21_trader import Ema21BreakdownEngine
from deltabot.enums import NotifyEvent, PositionState, SignalDir
from deltabot.models import Candle

_ist = ZoneInfo("Asia/Kolkata")


class FakeExecutor:
    """Mirrors test_dcv3_trader's FakeExecutor -- a LONG (bought) position."""

    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.tracked_size: int = 0
        self.underlying = "BTC"
        self.is_buy_side = True
        self.open_calls: list[tuple[int, float]] = []
        self.balance_fraction_calls: list[tuple[int, float, float, str | None]] = []
        self.trade_price_calls: list[tuple[int, float]] = []
        self.trade_price_balance_calls: list[tuple[int, float, float, str | None]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (100.0, "C-BTC-64000-070726")
        self._open_size = 25   # lots for the static open_option_by_premium path
        self._balance_fraction_result: tuple[float | None, str | None, int] = (100.0, "C-BTC-64000-070726", 7)
        self._trade_price_result: tuple[float | None, str | None] = (100.0, "C-BTC-64000-070726")
        self._trade_price_balance_result: tuple[float | None, str | None, int] = (100.0, "C-BTC-64000-070726", 9)
        self._close_result: float | None = 400.0   # 4x, for TP tests

    async def open_option_by_premium(self, signal_dir: int, target_premium: float):
        self.open_calls.append((signal_dir, target_premium))
        fill, symbol = self._open_result
        if fill is not None:
            self.has_open_position = True
            self.tracked_symbol = symbol
            self.tracked_product_id = 123
            self.tracked_size = self._open_size
        return fill, symbol

    async def open_option_by_balance_fraction(self, signal_dir, target_premium, balance_fraction, margin_asset):
        self.balance_fraction_calls.append((signal_dir, target_premium, balance_fraction, margin_asset))
        fill, symbol, lots = self._balance_fraction_result
        if fill is not None and lots > 0:
            self.has_open_position = True
            self.tracked_symbol = symbol
            self.tracked_product_id = 123
            self.tracked_size = lots
        return fill, symbol, lots

    async def open_option_by_trade_price(self, signal_dir, target_premium):
        self.trade_price_calls.append((signal_dir, target_premium))
        fill, symbol = self._trade_price_result
        if fill is not None:
            self.has_open_position = True
            self.tracked_symbol = symbol
            self.tracked_product_id = 123
            self.tracked_size = self._open_size
        return fill, symbol

    async def open_option_by_trade_price_and_balance_fraction(self, signal_dir, target_premium,
                                                                balance_fraction, margin_asset):
        self.trade_price_balance_calls.append((signal_dir, target_premium, balance_fraction, margin_asset))
        fill, symbol, lots = self._trade_price_balance_result
        if fill is not None and lots > 0:
            self.has_open_position = True
            self.tracked_symbol = symbol
            self.tracked_product_id = 123
            self.tracked_size = lots
        return fill, symbol, lots

    async def close_option(self):
        self.close_calls += 1
        self.has_open_position = False
        self.tracked_symbol = None
        self.tracked_size = 0
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
        self.tracked_size = size


class FakeRest:
    def __init__(self, positions=None, mark=None) -> None:
        self._positions = positions or []
        self._mark = mark

    def get_option_positions(self, underlying):
        return self._positions

    def get_mark_price(self, symbol):
        return self._mark


def _make_engine(**kw) -> Ema21BreakdownEngine:
    base = dict(strategy="ema21", target_premium=100.0, take_profit_pct=300.0,
                option_contracts=25, option_side="buy", state_file="", skip_weekdays="")
    base.update(kw)
    settings = Settings(_env_file=None, **base)
    engine = Ema21BreakdownEngine(settings, rest=FakeRest(), notifier=AsyncMock())
    engine.executor = FakeExecutor()
    return engine


def _c(start: int, o=100.0, h=101.0, low=99.0, cl=100.0) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _exit_calls(notifier):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == NotifyEvent.EXIT]


# ---------------------------------------------------------------------- #
# Entry: bullish -> buy CALL, bearish -> buy PUT; 300% RALLY TP
# ---------------------------------------------------------------------- #
async def test_bullish_signal_buys_call_and_sets_rally_tp() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, sl_level=63000.0, btc_price=64000.0)
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 100.0)]
    assert engine._entry_premium == 100.0
    assert engine._tp_price == pytest.approx(400.0)   # 100 * 4.0 (300% rally)
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_LONG and ev.kwargs["direction"] == "CALL"


async def test_take_profit_pct_zero_disables_the_rally_target() -> None:
    """take_profit_pct<=0 means NO premium target (matches the backtest's
    own --tp-gain-pct 0 convention) -- _tp_price must be None, not the entry
    fill itself (which _tp_mult=1.0 would otherwise produce, closing on the
    very next mark tick at or above entry)."""
    engine = _make_engine(take_profit_pct=0.0)
    await engine._open_entry(SignalDir.LONG.value, sl_level=63000.0, btc_price=64000.0)
    assert engine._tp_price is None
    ev = engine.notifier.notify.await_args
    assert ev.kwargs["tp_price"] is None


async def test_ema21_balance_pct_zero_uses_the_static_lot_path() -> None:
    """Default (0) -- unchanged behavior: open_option_by_premium is used,
    open_option_by_balance_fraction is never called."""
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, sl_level=63000.0, btc_price=64000.0)
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 100.0)]
    assert engine.executor.balance_fraction_calls == []
    assert engine.executor.tracked_size == 25


async def test_ema21_balance_pct_nonzero_uses_the_dynamic_lot_path() -> None:
    engine = _make_engine(ema21_balance_pct=10.0)
    await engine._open_entry(SignalDir.LONG.value, sl_level=63000.0, btc_price=64000.0)
    assert engine.executor.open_calls == []   # static path never called
    assert engine.executor.balance_fraction_calls == [(SignalDir.LONG.value, 100.0, 0.10, None)]
    assert engine.executor.tracked_size == 7   # FakeExecutor's _balance_fraction_result lots
    assert engine._entry_premium == 100.0


async def test_ema21_balance_pct_zero_lots_flattens_strategy() -> None:
    """open_option_by_balance_fraction returning (None, None, 0) (balance too
    small for even 1 lot) must be treated exactly like a failed static entry."""
    engine = _make_engine(ema21_balance_pct=10.0)
    engine.executor._balance_fraction_result = (None, None, 0)
    await engine._open_entry(SignalDir.LONG.value, sl_level=63000.0, btc_price=64000.0)
    assert engine._entry_premium is None
    assert not engine.executor.has_open_position


async def test_ema21_use_trade_price_uses_the_trade_price_path() -> None:
    engine = _make_engine(ema21_use_trade_price=True)
    await engine._open_entry(SignalDir.LONG.value, sl_level=63000.0, btc_price=64000.0)
    assert engine.executor.trade_price_calls == [(SignalDir.LONG.value, 100.0)]
    assert engine.executor.open_calls == []             # static path never called
    assert engine.executor.balance_fraction_calls == []  # balance path never called


async def test_ema21_use_trade_price_and_balance_pct_combine() -> None:
    """Both set: strike SELECTION and lot SIZING are independent toggles --
    the combined open_option_by_trade_price_and_balance_fraction path is
    used, and neither of the two single-purpose paths fires."""
    engine = _make_engine(ema21_use_trade_price=True, ema21_balance_pct=10.0)
    await engine._open_entry(SignalDir.LONG.value, sl_level=63000.0, btc_price=64000.0)
    assert engine.executor.trade_price_balance_calls == [
        (SignalDir.LONG.value, 100.0, 0.10, None)]
    assert engine.executor.trade_price_calls == []
    assert engine.executor.balance_fraction_calls == []
    assert engine.executor.open_calls == []
    assert engine.executor.tracked_size == 9   # FakeExecutor's _trade_price_balance_result lots


async def test_dynamic_lots_are_recorded_correctly_in_saved_state() -> None:
    """position_state must save the ACTUAL tracked size (7), not the static
    settings.option_contracts (25) -- this was a real bug: size was hardcoded
    to settings.option_contracts even on the dynamic-sizing path."""
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        state_file = str(Path(d) / "pos.json")
        engine = _make_engine(ema21_balance_pct=10.0, state_file=state_file)
        await engine._open_entry(SignalDir.LONG.value, sl_level=63000.0, btc_price=64000.0)
        saved = json.loads(Path(state_file).read_text())
        assert saved["size"] == 7


async def test_close_leg_uses_actual_tracked_size_not_static_config() -> None:
    """_close_leg's P&L/notify 'size' must reflect the REAL lots that were
    open (captured before close_option() clears tracked state), not
    settings.option_contracts -- same bug class as the state-save one above."""
    engine = _make_engine(ema21_balance_pct=10.0)
    await engine._open_entry(SignalDir.LONG.value, sl_level=63000.0, btc_price=64000.0)
    assert engine.executor.tracked_size == 7
    await engine._close_leg("SL", btc_exit_price=64500.0)
    ev = _exit_calls(engine.notifier)[-1]
    assert ev.kwargs["size"] == 7


async def test_bearish_signal_buys_put() -> None:
    engine = _make_engine()
    engine.executor._open_result = (100.0, "P-BTC-64000-070726")
    await engine._open_entry(SignalDir.SHORT.value, 65000.0, 64000.0)
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_SHORT and ev.kwargs["direction"] == "PUT"


async def test_open_entry_guarded_when_already_open() -> None:
    engine = _make_engine()
    engine.executor.has_open_position = True
    await engine._open_entry(SignalDir.LONG.value, 63000.0, 64000.0)
    assert engine.executor.open_calls == []


async def test_no_fill_flattens_strategy() -> None:
    engine = _make_engine()
    engine.executor._open_result = (None, None)
    engine.strategy._in_long = True
    await engine._open_entry(SignalDir.LONG.value, 63000.0, 64000.0)
    assert engine._entry_premium is None
    assert engine.strategy.position_state == PositionState.FLAT


# ---------------------------------------------------------------------- #
# Exits + P&L sign (BUY side: profit = exit - entry)
# ---------------------------------------------------------------------- #
async def test_close_leg_pnl_sign_profits_when_premium_rose() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 63000.0, 64000.0)
    engine.executor._close_result = 250.0   # rose from 100 -> 250: a PROFIT
    await engine._close_leg("SL", btc_exit_price=63000.0)
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["pnl"] > 0
    assert exits[-1].kwargs["reason"] == "SL"


async def test_close_tp_flattens_strategy_so_it_waits_for_new_signal() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 63000.0, 64000.0)
    engine.strategy._in_long = True
    engine.executor._close_result = 400.0   # 4x
    await engine._close_tp(400.0)
    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.FLAT
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "TP" and exits[-1].kwargs["pnl"] > 0


async def test_double_close_guard() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 63000.0, 64000.0)
    engine._closing = True
    await engine._close_leg("SL", 63000.0)
    assert engine.executor.close_calls == 0


# ---------------------------------------------------------------------- #
# Closed-bar entry (no intracandle path exists in this engine at all)
# ---------------------------------------------------------------------- #
async def test_closed_bar_entry_fires_on_bullish_decision() -> None:
    engine = _make_engine()

    class _Dec:
        has_exit = False
        has_entry = True
        buy_signal = True
        sell_signal = False
        sl_level = 63000.0

    engine.strategy.update = lambda candle: _Dec()
    await engine._handle_closed_candle(_c(900))
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 100.0)]


async def test_closed_bar_no_entry_when_no_decision() -> None:
    engine = _make_engine()
    engine.strategy.update = lambda candle: None
    await engine._handle_closed_candle(_c(900))
    assert engine.executor.open_calls == []


# ---------------------------------------------------------------------- #
# 17:25 square-off: plain close, no rollover, always force_flat()
# ---------------------------------------------------------------------- #
def _fake_now(monkeypatch, dt):
    import deltabot.core.ema21_trader as mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return dt
    monkeypatch.setattr(mod, "datetime", _FakeDatetime)


async def test_square_off_closes_open_position_with_eod_reason(monkeypatch) -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 63000.0, 64000.0)
    engine.strategy._in_long = True
    _fake_now(monkeypatch, datetime(2026, 7, 10, 17, 25, tzinfo=_ist))
    await engine._square_off()
    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.FLAT   # always flattened, no rollover
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "EOD"


async def test_square_off_noop_when_already_flat(monkeypatch) -> None:
    engine = _make_engine()
    _fake_now(monkeypatch, datetime(2026, 7, 10, 17, 25, tzinfo=_ist))
    await engine._square_off()
    assert engine.executor.close_calls == 0
    assert engine.strategy.position_state == PositionState.FLAT


async def test_entries_blocked_only_checks_weekday(monkeypatch) -> None:
    """Unlike DCv2/DCv3, no resume-hour tracking -- the strategy's own
    entry_start_hour/entry_end_hour already gates the trade internally."""
    engine = _make_engine(skip_weekdays="Sat,Sun")
    _fake_now(monkeypatch, datetime(2026, 7, 8, 12, 0, tzinfo=_ist))   # Wed
    assert engine._entries_blocked() is False
    _fake_now(monkeypatch, datetime(2026, 7, 11, 12, 0, tzinfo=_ist))  # Sat
    assert engine._entries_blocked() is True


# ---------------------------------------------------------------------- #
# Self-heal: looks for a LONG (size > 0)
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
    await engine._open_entry(SignalDir.LONG.value, 63000.0, 64000.0)
    engine.executor.tracked_product_id = 123
    engine._last_verify = 0.0


async def test_selfheal_flattens_after_two_misses() -> None:
    engine = _make_engine(position_verify_seconds=0.0001)
    await _open_and_arm(engine)
    engine.rest = VerifyRest(positions=[])
    await engine._maybe_verify_position()
    assert engine.executor.has_open_position
    engine._last_verify = 0.0
    await engine._maybe_verify_position()
    assert not engine.executor.has_open_position
    assert engine._entry_premium is None


async def test_selfheal_never_drops_on_fetch_error() -> None:
    engine = _make_engine(position_verify_seconds=0.0001)
    await _open_and_arm(engine)
    engine.rest = VerifyRest(raises=True)
    for _ in range(5):
        engine._last_verify = 0.0
        await engine._maybe_verify_position()
    assert engine.executor.has_open_position


# ---------------------------------------------------------------------- #
# Reconcile: adopts a LONG (size > 0); preserves tracked state when the
# exchange returns nothing but the state file claims ownership.
# ---------------------------------------------------------------------- #
async def test_reconcile_adopts_open_long() -> None:
    engine = _make_engine()
    engine.rest = FakeRest(positions=[
        {"symbol": "C-BTC-64000-070726", "product_id": 999, "size": 25},
    ])
    await engine._sync_options_to_exchange()
    assert engine.executor.has_open_position
    assert engine.executor.tracked_symbol == "C-BTC-64000-070726"


async def test_reconcile_ignores_a_short_position() -> None:
    engine = _make_engine()
    engine.rest = FakeRest(positions=[
        {"symbol": "C-BTC-64000-070726", "product_id": 999, "size": -25},
    ])
    await engine._sync_options_to_exchange()
    assert not engine.executor.has_open_position


async def test_reconcile_preserves_state_when_exchange_empty_but_owned(monkeypatch, tmp_path) -> None:
    """Exchange fetch returns nothing but the state file claims ownership --
    must preserve the tracked/state position (not silently go flat and risk
    double-opening) and refuse new trades until manually cleared."""
    state_path = tmp_path / "ema21_pos.json"
    state_path.write_text(
        '{"symbol": "C-BTC-64000-070726", "product_id": 999, "size": 25, '
        '"entry_premium": 100.0, "tp_price": 400.0, "direction": 1}'
    )
    engine = _make_engine(state_file=str(state_path))
    engine.rest = FakeRest(positions=[])
    await engine._sync_options_to_exchange()
    assert engine.executor.has_open_position
    assert engine.executor.tracked_symbol == "C-BTC-64000-070726"
    assert engine._entry_premium == 100.0
