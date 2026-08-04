"""ThreeCandleEngine: the option-BUY live engine wired to ThreeCandleStrategy.
Same structure/fakes as test_dcv3_trader.py (also a BUY-side engine): rally TP
(not decay), reconcile looks for a LONG position, P&L sign, PUT/CALL direction
labeling, weekend-flat/entries-blocked, self-heal, reconcile."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from deltabot.config import Settings
from deltabot.core.three_candle_trader import ThreeCandleEngine
from deltabot.enums import NotifyEvent, PositionState, SignalDir
from deltabot.models import Candle

_ist = ZoneInfo("Asia/Kolkata")


class FakeExecutor:
    """Mirrors test_dcv3_trader's FakeExecutor (LONG/buy-side)."""

    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.underlying = "BTC"
        self.is_buy_side = True
        self.open_calls: list[tuple[int, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (400.0, "C-BTC-64000-070726")
        self._close_result: float | None = 600.0   # +50%, for TP tests

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


def _make_engine(**kw) -> ThreeCandleEngine:
    base = dict(strategy="three_candle", target_premium=400.0, take_profit_pct=50.0,
                option_contracts=25, option_side="buy", state_file="", skip_weekdays="Sun")
    base.update(kw)
    settings = Settings(_env_file=None, **base)
    engine = ThreeCandleEngine(settings, rest=FakeRest(), notifier=AsyncMock())
    engine.executor = FakeExecutor()
    return engine


def _c(start: int, o=100.0, h=101.0, low=99.0, cl=100.0) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _exit_calls(notifier):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == NotifyEvent.EXIT]


# ---------------------------------------------------------------------- #
# Entry: signal -> option side + 50% RALLY TP
# ---------------------------------------------------------------------- #
async def test_buy_signal_buys_call_and_sets_rally_tp() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, sl_level=59000.0, btc_price=60000.0, tag="ENTRY")
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 400.0)]
    assert engine._entry_premium == 400.0
    assert engine._tp_price == pytest.approx(600.0)   # 400 * 1.5 (50% rally)
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_LONG and ev.kwargs["direction"] == "CALL"


async def test_sell_signal_buys_put() -> None:
    engine = _make_engine()
    engine.executor._open_result = (400.0, "P-BTC-60000-070726")
    await engine._open_entry(SignalDir.SHORT.value, 61000.0, 60000.0, tag="ENTRY")
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_SHORT and ev.kwargs["direction"] == "PUT"


async def test_open_entry_guarded_when_already_open() -> None:
    engine = _make_engine()
    engine.executor.has_open_position = True
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    assert engine.executor.open_calls == []


async def test_no_fill_flattens_strategy() -> None:
    engine = _make_engine()
    engine.executor._open_result = (None, None)
    engine.strategy._in_long = True
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    assert engine._entry_premium is None
    assert engine.strategy.position_state == PositionState.FLAT


# ---------------------------------------------------------------------- #
# Exits + P&L sign (BUY side: profit = exit - entry)
# ---------------------------------------------------------------------- #
async def test_close_leg_pnl_sign_profits_when_premium_rose() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.executor._close_result = 500.0   # rose from 400 -> 500: PROFIT
    await engine._close_leg("SL", btc_exit_price=61000.0)
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["pnl"] > 0
    assert exits[-1].kwargs["reason"] == "SL"


async def test_close_tp_flattens_strategy_so_it_waits_for_new_signal() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    engine.executor._close_result = 600.0
    await engine._close_tp(600.0)
    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.FLAT
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "TP" and exits[-1].kwargs["pnl"] > 0


async def test_double_close_guard() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine._closing = True
    await engine._close_leg("SL", 59000.0)
    assert engine.executor.close_calls == 0


# ---------------------------------------------------------------------- #
# Intracandle SL (ASAP)
# ---------------------------------------------------------------------- #
async def test_intracandle_sl_closes_leg_and_flattens() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
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
# 17:25 square-off: Mon-Sat config -> Saturday flattens (Sunday is the skip day)
# ---------------------------------------------------------------------- #
def _fake_now(monkeypatch, dt):
    import deltabot.core.three_candle_trader as mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return dt
    monkeypatch.setattr(mod, "datetime", _FakeDatetime)


async def test_square_off_saturday_flattens_mon_sat_config(monkeypatch) -> None:
    """Mon-Sat config (skip_weekdays=Sun, weekend_flat=True): Saturday's
    square-off flattens (tomorrow=Sunday is a skip day)."""
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    _fake_now(monkeypatch, datetime(2026, 7, 11, 17, 25, tzinfo=_ist))  # Saturday
    await engine._square_off()
    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.FLAT
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "WEEKEND"


async def test_square_off_wednesday_keeps_direction_for_rollover(monkeypatch) -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    _fake_now(monkeypatch, datetime(2026, 7, 8, 17, 25, tzinfo=_ist))  # Wednesday
    await engine._square_off()
    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.LONG   # NOT flattened
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "EOD"


async def test_entries_blocked_on_sunday() -> None:
    engine = _make_engine()   # skip_weekdays="Sun"
    import deltabot.core.three_candle_trader as mod
    orig = mod.datetime

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 7, 12, 12, 0, tzinfo=_ist)  # Sunday
    mod.datetime = _FakeDatetime
    try:
        assert engine._entries_blocked() is True
    finally:
        mod.datetime = orig


async def test_square_off_continuous_roll_immediately_rebuys(monkeypatch) -> None:
    engine = _make_engine(dcv2_continuous_roll=True)
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    _fake_now(monkeypatch, datetime(2026, 7, 8, 17, 25, tzinfo=_ist))  # Wed
    await engine._square_off()
    assert engine.executor.close_calls == 1
    assert engine.executor.has_open_position   # re-bought immediately
    assert engine.strategy.position_state == PositionState.LONG
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "ROLL"
    entries = [c for c in engine.notifier.notify.await_args_list
               if c.args and c.args[0] == NotifyEvent.ENTRY_LONG]
    assert entries and entries[-1].kwargs["tag"] == "ROLL"


async def test_square_off_continuous_roll_skips_when_flattened_by_race(monkeypatch) -> None:
    """Same race-condition fix as dcv2/dcv3: a same-timestamp candle-close
    flattening the strategy while close_option() awaits must skip the roll,
    not default to a phantom entry."""
    engine = _make_engine(dcv2_continuous_roll=True)
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    engine.strategy._sl_level = 59000.0

    real_close = engine.executor.close_option

    async def _close_and_race_flatten():
        fill = await real_close()
        engine.strategy.force_flat()
        return fill
    engine.executor.close_option = _close_and_race_flatten

    calls_before = len(engine.notifier.notify.await_args_list)
    _fake_now(monkeypatch, datetime(2026, 7, 8, 17, 25, tzinfo=_ist))  # Wed
    await engine._square_off()

    assert engine.executor.close_calls == 1
    assert not engine.executor.has_open_position
    assert engine.strategy.position_state == PositionState.FLAT
    new_calls = engine.notifier.notify.await_args_list[calls_before:]
    entries = [c for c in new_calls
               if c.args and c.args[0] in (NotifyEvent.ENTRY_LONG, NotifyEvent.ENTRY_SHORT)]
    assert not entries


# ---------------------------------------------------------------------- #
# 17:30 rollover: trade still open + option flat -> re-BUY
# ---------------------------------------------------------------------- #
async def test_rollover_rebuys_when_trade_open_but_option_flat() -> None:
    engine = _make_engine()
    engine._sq_off_date = None
    strat = MagicMock()
    strat.update.return_value = None
    strat.position_state = PositionState.LONG
    strat.sl_level = 59000.0
    engine.strategy = strat
    await engine._handle_closed_candle(_c(300))
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 400.0)]


async def test_no_rollover_when_flat() -> None:
    engine = _make_engine()
    engine._sq_off_date = None
    strat = MagicMock()
    strat.update.return_value = None
    strat.position_state = PositionState.FLAT
    engine.strategy = strat
    await engine._handle_closed_candle(_c(300))
    assert engine.executor.open_calls == []


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
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
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
# Reconcile: adopts a LONG (size > 0)
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


async def test_reconnect_while_flat_preserves_hunt() -> None:
    """Same reconnect fix as dcv2/dcv3: a reconcile while already flat must
    NOT wipe an in-progress pattern hunt."""
    engine = _make_engine()
    engine.strategy._sell_state = "got_ab"
    engine.strategy._sell_a = {"high": 101.0, "low": 95.0}
    engine.strategy._sell_b = {"high": 100.0, "low": 93.0}
    assert engine.strategy.position_state == PositionState.FLAT
    await engine._sync_options_to_exchange()
    assert engine.strategy._sell_state == "got_ab"
    assert engine.strategy._sell_a == {"high": 101.0, "low": 95.0}
