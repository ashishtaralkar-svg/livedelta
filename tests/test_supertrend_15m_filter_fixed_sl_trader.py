"""Supertrend15mFilterFixedSlEngine: the live sell-side engine for
Supertrend15mFilterFixedSlStrategy. Mirrors test_ema21_trader.py's /
test_supertrend_sar_trader.py's FakeExecutor/FakeRest shape, adapted for a
SELL-side, single-position, closed-bar-only, no-square-off, no-rollover
engine with a per-candle (not polled) premium target."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from deltabot.config import Settings
from deltabot.core.supertrend_15m_filter_fixed_sl_trader import Supertrend15mFilterFixedSlEngine
from deltabot.enums import NotifyEvent, SignalDir
from deltabot.models import Candle

_ist = ZoneInfo("Asia/Kolkata")


class FakeExecutor:
    """A SHORT (sold) position -- mirrors test_supertrend_sar_trader's FakeExecutor."""

    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.tracked_size: int = 0
        self.underlying = "BTC"
        self.is_buy_side = False
        self.last_leverage_ok: bool | None = None
        self.open_calls: list[tuple[int, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (100.0, "C-BTC-64000-070726")
        self._open_size = 25
        self._close_result: float | None = 30.0   # decayed from 100 -> 30

    async def open_option_by_premium(self, signal_dir: int, target_premium: float):
        self.open_calls.append((signal_dir, target_premium))
        fill, symbol = self._open_result
        if fill is not None:
            self.has_open_position = True
            self.tracked_symbol = symbol
            self.tracked_product_id = 123
            self.tracked_size = self._open_size
        return fill, symbol

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


def _make_engine(**kw) -> Supertrend15mFilterFixedSlEngine:
    base = dict(strategy="st15f", target_premium=1400.0, st15f_target_pct=70.0,
                option_contracts=25, option_side="sell", state_file="", skip_weekdays="")
    base.update(kw)
    settings = Settings(_env_file=None, **base)
    engine = Supertrend15mFilterFixedSlEngine(settings, rest=FakeRest(), notifier=AsyncMock())
    engine.executor = FakeExecutor()
    return engine


def _c(start: int, o=100.0, h=101.0, low=99.0, cl=100.0) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _exit_calls(notifier):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == NotifyEvent.EXIT]


# ---------------------------------------------------------------------- #
# Entry: entry_is_short=True -> sell CALL, False -> sell PUT
# ---------------------------------------------------------------------- #
async def test_short_entry_sells_call() -> None:
    engine = _make_engine()
    await engine._open_entry(True, sl_level=64500.0, btc_price=64000.0)
    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 1400.0)]
    assert engine._entry_premium == 100.0
    assert engine._current_is_short is True
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_SHORT and ev.kwargs["direction"] == "CALL"


async def test_long_entry_sells_put() -> None:
    engine = _make_engine()
    engine.executor._open_result = (100.0, "P-BTC-64000-070726")
    await engine._open_entry(False, sl_level=63500.0, btc_price=64000.0)
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 1400.0)]
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_LONG and ev.kwargs["direction"] == "PUT"


async def test_open_entry_guarded_when_already_open() -> None:
    engine = _make_engine()
    engine.executor.has_open_position = True
    await engine._open_entry(True, 64500.0, 64000.0)
    assert engine.executor.open_calls == []


async def test_no_fill_flattens_strategy() -> None:
    engine = _make_engine()
    engine.executor._open_result = (None, None)
    engine.strategy._is_short = True
    await engine._open_entry(True, 64500.0, 64000.0)
    assert engine._entry_premium is None
    assert not engine.strategy.in_position


# ---------------------------------------------------------------------- #
# Exits + P&L sign (SELL side: profit = entry - exit)
# ---------------------------------------------------------------------- #
async def test_close_leg_pnl_sign_profits_when_premium_decayed() -> None:
    engine = _make_engine()
    await engine._open_entry(True, 64500.0, 64000.0)
    engine.executor._close_result = 30.0   # decayed from 100 -> 30: a PROFIT
    await engine._close_leg("SL", btc_exit_price=64200.0)
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["pnl"] > 0
    assert exits[-1].kwargs["reason"] == "SL"


async def test_double_close_guard() -> None:
    engine = _make_engine()
    await engine._open_entry(True, 64500.0, 64000.0)
    engine._closing = True
    await engine._close_leg("SL", 64200.0)
    assert engine.executor.close_calls == 0


# ---------------------------------------------------------------------- #
# Profit target: checked once per closed candle, not a poll -- reduction
# convention (target = entry * (1 - target_pct/100)).
# ---------------------------------------------------------------------- #
async def test_target_hit_closes_and_calls_notify_target_hit_only() -> None:
    """Deliberately verifies the engine relies on notify_target_hit() being
    self-sufficient (no separate force_flat() call at this call site) --
    the exact fix for the real bug found in the backtest."""
    engine = _make_engine(st15f_target_pct=70.0)   # target = entry * 0.30
    await engine._open_entry(True, 64500.0, 64000.0)
    assert engine._entry_premium == 100.0

    class _Dec:
        has_exit = False
        exit_price = None
        has_entry = False
        entry_is_short = None
        sl_level = None

    engine.strategy.update = lambda candle: _Dec()
    engine.rest = FakeRest(mark=29.0)   # <= 100 * 0.30 -> target hit
    engine.executor._close_result = 29.0
    await engine._handle_closed_candle(_c(900))

    assert engine.executor.close_calls == 1
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "TARGET" and exits[-1].kwargs["pnl"] > 0
    assert not engine.strategy.in_position
    assert engine.strategy._blocked_until_15m_flip is True


async def test_target_not_hit_leaves_position_open() -> None:
    engine = _make_engine(st15f_target_pct=70.0)
    await engine._open_entry(True, 64500.0, 64000.0)

    class _Dec:
        has_exit = False
        exit_price = None
        has_entry = False
        entry_is_short = None
        sl_level = None

    engine.strategy.update = lambda candle: _Dec()
    engine.rest = FakeRest(mark=50.0)   # above the 30.0 target -- no hit
    await engine._handle_closed_candle(_c(900))
    assert engine.executor.close_calls == 0
    assert engine.executor.has_open_position


async def test_target_pct_zero_disables_target_checking() -> None:
    engine = _make_engine(st15f_target_pct=0.0)
    assert engine._target_frac is None
    await engine._open_entry(True, 64500.0, 64000.0)

    class _Dec:
        has_exit = False
        exit_price = None
        has_entry = False
        entry_is_short = None
        sl_level = None

    engine.strategy.update = lambda candle: _Dec()
    engine.rest = FakeRest(mark=0.01)   # would trivially "hit" any target
    await engine._handle_closed_candle(_c(900))
    assert engine.executor.close_calls == 0


# ---------------------------------------------------------------------- #
# Closed-bar exit + entry (no intracandle path exists in this engine)
# ---------------------------------------------------------------------- #
async def test_closed_bar_exit_fires_on_decision_without_extra_force_flat() -> None:
    engine = _make_engine()
    await engine._open_entry(True, 64500.0, 64000.0)

    class _Dec:
        has_exit = True
        exit_price = 64500.0
        has_entry = False
        entry_is_short = None
        sl_level = None

    engine.strategy.update = lambda candle: _Dec()
    engine.rest = FakeRest(mark=None)
    await engine._handle_closed_candle(_c(900))
    assert engine.executor.close_calls == 1
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "SL"


async def test_closed_bar_entry_fires_on_short_decision() -> None:
    engine = _make_engine()

    class _Dec:
        has_exit = False
        exit_price = None
        has_entry = True
        entry_is_short = True
        sl_level = 64500.0

    engine.strategy.update = lambda candle: _Dec()
    await engine._handle_closed_candle(_c(900))
    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 1400.0)]


async def test_closed_bar_no_entry_when_no_decision() -> None:
    engine = _make_engine()
    engine.strategy.update = lambda candle: None
    await engine._handle_closed_candle(_c(900))
    assert engine.executor.open_calls == []


# ---------------------------------------------------------------------- #
# Entries blocked: weekday skip only -- no square-off/restart concept here
# ---------------------------------------------------------------------- #
async def test_entries_blocked_only_checks_weekday(monkeypatch) -> None:
    import deltabot.core.supertrend_15m_filter_fixed_sl_trader as mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return dt

    engine = _make_engine(skip_weekdays="Sat,Sun")
    dt = datetime(2026, 7, 8, 12, 0, tzinfo=_ist)   # Wed
    monkeypatch.setattr(mod, "datetime", _FakeDatetime)
    assert engine._entries_blocked() is False
    dt = datetime(2026, 7, 11, 12, 0, tzinfo=_ist)  # Sat
    assert engine._entries_blocked() is True


# ---------------------------------------------------------------------- #
# Self-heal: looks for a SHORT (size < 0)
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
    await engine._open_entry(True, 64500.0, 64000.0)
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
# Reconcile: adopts a SHORT (size < 0); preserves tracked state when the
# exchange returns nothing but the state file claims ownership.
# ---------------------------------------------------------------------- #
async def test_reconcile_adopts_open_short() -> None:
    engine = _make_engine()
    engine.rest = FakeRest(positions=[
        {"symbol": "C-BTC-64000-070726", "product_id": 999, "size": -25},
    ])
    await engine._sync_options_to_exchange()
    assert engine.executor.has_open_position
    assert engine.executor.tracked_symbol == "C-BTC-64000-070726"


async def test_reconcile_ignores_a_long_position() -> None:
    engine = _make_engine()
    engine.rest = FakeRest(positions=[
        {"symbol": "C-BTC-64000-070726", "product_id": 999, "size": 25},
    ])
    await engine._sync_options_to_exchange()
    assert not engine.executor.has_open_position


async def test_reconcile_preserves_state_when_exchange_empty_but_owned(tmp_path) -> None:
    state_path = tmp_path / "st15f_pos.json"
    state_path.write_text(
        '{"symbol": "C-BTC-64000-070726", "product_id": 999, "size": -25, '
        '"entry_premium": 100.0, "is_short": true}'
    )
    engine = _make_engine(state_file=str(state_path))
    engine.rest = FakeRest(positions=[])
    await engine._sync_options_to_exchange()
    assert engine.executor.has_open_position
    assert engine.executor.tracked_symbol == "C-BTC-64000-070726"
    assert engine._entry_premium == 100.0


# ---------------------------------------------------------------------- #
# Minute-precise expiry cutoff -- the whole reason for the executor subclass
# ---------------------------------------------------------------------- #
def test_minute_precise_expiry_rolls_at_the_exact_configured_minute(monkeypatch) -> None:
    from deltabot.core.supertrend_15m_filter_fixed_sl_trader import _MinutePreciseOptionsExecutor
    import deltabot.core.supertrend_15m_filter_fixed_sl_trader as mod

    engine = _make_engine(st15f_expiry_cutoff_hour=17, st15f_expiry_cutoff_minute=26)
    real_executor = _MinutePreciseOptionsExecutor(engine.rest, engine.settings, 17, 26)

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return dt

    monkeypatch.setattr(mod, "datetime", _FakeDatetime)

    dt = datetime(2026, 7, 8, 17, 25, tzinfo=_ist)   # before cutoff -- today's expiry
    assert real_executor._select_expiry() == dt.date()

    dt = datetime(2026, 7, 8, 17, 26, tzinfo=_ist)   # AT cutoff -- next day's expiry
    assert real_executor._select_expiry() == datetime(2026, 7, 9, tzinfo=_ist).date()

    dt = datetime(2026, 7, 8, 18, 0, tzinfo=_ist)    # well past cutoff -- next day's expiry
    assert real_executor._select_expiry() == datetime(2026, 7, 9, tzinfo=_ist).date()
