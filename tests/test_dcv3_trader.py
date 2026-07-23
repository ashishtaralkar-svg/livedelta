"""DCv3Engine: the BUY-side mirror of DCv2Engine (test_dcv2_trader.py). Same
structure; asserts the flips are correct -- rally TP (not decay), reconcile
looks for a LONG position, P&L sign, PUT/CALL direction labeling, and that
weekend-flat/entries-blocked behave per the 24/7 config (skip_weekdays empty,
weekend_flat off)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from deltabot.config import Settings
from deltabot.core.dcv3_trader import DCv3Engine
from deltabot.enums import NotifyEvent, PositionState, SignalDir
from deltabot.models import Candle

_ist = ZoneInfo("Asia/Kolkata")


class FakeExecutor:
    """Mirrors test_dcv2_trader's FakeExecutor but as a LONG position (buy side)."""

    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.underlying = "BTC"
        self.is_buy_side = True
        self.open_calls: list[tuple[int, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (500.0, "C-BTC-64000-070726")
        self._close_result: float | None = 1000.0   # doubled, for TP tests

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


def _make_engine(**kw) -> DCv3Engine:
    base = dict(strategy="dcv3", target_premium=500.0, take_profit_pct=100.0,
                option_contracts=25, option_side="buy", state_file="", skip_weekdays="")
    base.update(kw)
    settings = Settings(_env_file=None, **base)
    engine = DCv3Engine(settings, rest=FakeRest(), notifier=AsyncMock())
    engine.executor = FakeExecutor()
    return engine


def _c(start: int, o=100.0, h=101.0, low=99.0, cl=100.0) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _exit_calls(notifier):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == NotifyEvent.EXIT]


# ---------------------------------------------------------------------- #
# Entry: signal -> option side + 100% RALLY TP (mirror of DCv2's 70% decay)
# ---------------------------------------------------------------------- #
async def test_buy_signal_buys_call_and_sets_rally_tp() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, sl_level=59000.0, btc_price=60000.0, tag="ENTRY")
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 500.0)]
    assert engine._entry_premium == 500.0
    assert engine._tp_price == pytest.approx(1000.0)   # 500 * 2.0 (100% rally -> doubles)
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_LONG and ev.kwargs["direction"] == "CALL"   # mirror of DCv2's PUT


async def test_sell_signal_buys_put() -> None:
    engine = _make_engine()
    engine.executor._open_result = (500.0, "P-BTC-60000-070726")
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
# Exits + P&L sign (BUY side: profit = exit - entry, the mirror of SELL)
# ---------------------------------------------------------------------- #
async def test_close_leg_pnl_sign_profits_when_premium_rose() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.executor._close_result = 750.0   # rose from 500 -> 750: should be a PROFIT
    await engine._close_leg("EMA_CROSS", btc_exit_price=61000.0)
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["pnl"] > 0
    assert exits[-1].kwargs["reason"] == "EMA_CROSS"


async def test_close_tp_flattens_strategy_so_it_waits_for_new_signal() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    engine.executor._close_result = 1000.0   # doubled
    await engine._close_tp(1000.0)
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
# 17:25 square-off: 24/7 config -> weekend_flat OFF, always EOD (never WEEKEND)
# ---------------------------------------------------------------------- #
def _fake_now(monkeypatch, dt):
    import deltabot.core.dcv3_trader as mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return dt
    monkeypatch.setattr(mod, "datetime", _FakeDatetime)


async def test_square_off_friday_does_not_flatten_when_weekend_flat_off(monkeypatch) -> None:
    """24/7 config: dcv2_weekend_flat=False -> even on Friday, the trade is
    NOT flattened (only the option leg closes; rollover re-buys at 17:30)."""
    engine = _make_engine(dcv2_weekend_flat=False)
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine.strategy._in_long = True
    _fake_now(monkeypatch, datetime(2026, 7, 10, 17, 25, tzinfo=_ist))  # Friday
    await engine._square_off()
    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.LONG   # NOT flattened
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "EOD"   # not WEEKEND


async def test_entries_never_blocked_on_saturday_24x7(monkeypatch) -> None:
    """24/7 config: DELTA_SKIP_WEEKDAYS is empty -> Saturday entries allowed."""
    engine = _make_engine()   # skip_weekdays="" by default in _make_engine
    _fake_now(monkeypatch, datetime(2026, 7, 11, 12, 0, tzinfo=_ist))  # Saturday
    assert engine._entries_blocked() is False


async def test_square_off_continuous_roll_immediately_rebuys(monkeypatch) -> None:
    """dcv2_continuous_roll=True: close AND re-buy in the SAME _square_off()
    call -- no waiting for the 17:30 gap to clear."""
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


async def test_entries_blocked_ignores_gap_when_continuous_roll(monkeypatch) -> None:
    engine = _make_engine(skip_weekdays="Sat,Sun", dcv2_continuous_roll=True)
    engine._sq_off_date = datetime(2026, 7, 8, tzinfo=_ist).date()
    _fake_now(monkeypatch, datetime(2026, 7, 8, 17, 26, tzinfo=_ist))  # 1 min after square-off
    assert engine._entries_blocked() is False

    _fake_now(monkeypatch, datetime(2026, 7, 11, 12, 0, tzinfo=_ist))  # Sat -> still blocked
    assert engine._entries_blocked() is True


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
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 500.0)]


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
# Self-heal: looks for a LONG (size > 0), the mirror of DCv2's short check
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


async def test_selfheal_recognizes_long_position_size_positive() -> None:
    """A LONG (size > 0) position at the tracked product_id must NOT be
    treated as a miss -- the mirror of DCv2's size < 0 check."""
    engine = _make_engine(position_verify_seconds=0.0001)
    await _open_and_arm(engine)
    engine.rest = VerifyRest(positions=[{"symbol": "C-BTC-64000-070726", "product_id": 123, "size": 25}])
    for _ in range(3):
        engine._last_verify = 0.0
        await engine._maybe_verify_position()
    assert engine.executor.has_open_position
    assert engine._verify_misses == 0


async def test_selfheal_never_drops_on_fetch_error() -> None:
    engine = _make_engine(position_verify_seconds=0.0001)
    await _open_and_arm(engine)
    engine.rest = VerifyRest(raises=True)
    for _ in range(5):
        engine._last_verify = 0.0
        await engine._maybe_verify_position()
    assert engine.executor.has_open_position


# ---------------------------------------------------------------------- #
# Reconcile: adopts a LONG (size > 0), the mirror of DCv2's short reconcile
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
    """A short (size < 0) on this sub-account would be someone else's -- dcv3
    only ever adopts longs. With none found and nothing believed-owned, it
    stays flat (does not mistakenly adopt the short)."""
    engine = _make_engine()
    engine.rest = FakeRest(positions=[
        {"symbol": "C-BTC-64000-070726", "product_id": 999, "size": -25},
    ])
    await engine._sync_options_to_exchange()
    assert not engine.executor.has_open_position
