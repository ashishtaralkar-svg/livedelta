"""DchannelEngine: decision -> executor-call translation, premium-decay TP,
BTC-exit close, and double-open/double-close guards.

Uses a lightweight FakeExecutor/FakeRest so these tests isolate the engine's own
logic (not OptionsExecutor's REST internals).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from deltabot.config import Settings
from deltabot.core.dchannel_trader import DchannelEngine
from deltabot.enums import NotifyEvent, PositionState, SignalDir
from deltabot.models import Candle

_dch_ist = ZoneInfo("Asia/Kolkata")


class FakeExecutor:
    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.underlying = "BTC"
        self.open_calls: list[tuple[int, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (1000.0, "P-BTC-60000-070726")
        self._close_result: float | None = 300.0

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


def _make_engine(**settings_kwargs) -> DchannelEngine:
    base = dict(strategy="dchannel", target_premium=1000.0, take_profit_pct=70.0,
                option_contracts=1)
    base.update(settings_kwargs)
    settings = Settings(_env_file=None, state_file="", **base)
    engine = DchannelEngine(settings, rest=FakeRest(), notifier=AsyncMock())
    engine.executor = FakeExecutor()
    return engine


def _c(start: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _exit_calls(notifier):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == NotifyEvent.EXIT]


# ---------------------------------------------------------------------- #
# Entry: signal -> option side + TP level
# ---------------------------------------------------------------------- #
async def test_buy_signal_sells_put_and_sets_70pct_decay_tp() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, sl_level=59000.0, btc_price=60000.0)

    assert engine.executor.open_calls == [(SignalDir.LONG.value, 1000.0)]
    assert engine._entry_premium == 1000.0
    assert engine._tp_price == pytest.approx(300.0)  # 1000 * (1 - 70/100)
    event = engine.notifier.notify.await_args.args[0]
    assert event == NotifyEvent.ENTRY_LONG
    assert engine.notifier.notify.await_args.kwargs["direction"] == "PUT"


async def test_sell_signal_sells_call() -> None:
    engine = _make_engine()
    engine.executor._open_result = (1000.0, "C-BTC-60000-070726")
    await engine._open_entry(SignalDir.SHORT.value, sl_level=61000.0, btc_price=60000.0)

    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 1000.0)]
    event = engine.notifier.notify.await_args.args[0]
    assert event == NotifyEvent.ENTRY_SHORT
    assert engine.notifier.notify.await_args.kwargs["direction"] == "CALL"


async def test_open_entry_guarded_when_already_open() -> None:
    engine = _make_engine()
    engine.executor.has_open_position = True
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0)
    assert engine.executor.open_calls == []


async def test_no_fill_flattens_strategy_and_stays_flat() -> None:
    engine = _make_engine()
    engine.executor._open_result = (None, None)
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0)
    assert engine._entry_premium is None
    assert engine.strategy.position_state == PositionState.FLAT


# ---------------------------------------------------------------------- #
# Exits
# ---------------------------------------------------------------------- #
async def test_tp_close_clears_tracking_and_flattens_strategy() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0)
    engine.executor._close_result = 300.0

    await engine._close_tp(300.0)

    assert engine.executor.close_calls == 1
    assert engine._entry_premium is None and engine._tp_price is None
    assert not engine.executor.has_open_position
    assert engine.strategy.position_state == PositionState.FLAT
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "TP"


async def test_btc_exit_close_notifies_reason() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.SHORT.value, 61000.0, 60000.0)
    engine.executor._close_result = 1100.0

    await engine._close_btc_exit("SL", 61000.0)

    assert engine.executor.close_calls == 1
    assert engine._entry_premium is None
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "SL"


async def test_double_close_guard() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0)
    engine._closing = True
    await engine._close_tp(300.0)
    assert engine.executor.close_calls == 0  # guard blocks the second close


async def test_tp_close_noop_when_flat() -> None:
    engine = _make_engine()
    await engine._close_tp(300.0)
    assert engine.executor.close_calls == 0


# ---------------------------------------------------------------------- #
# Reconcile
# ---------------------------------------------------------------------- #
async def test_reconcile_flat_when_no_positions() -> None:
    engine = _make_engine()
    engine.rest = FakeRest(positions=[])
    await engine._sync_options_to_exchange()
    assert not engine.executor.has_open_position
    assert engine._entry_premium is None


async def test_reconcile_adopts_open_short() -> None:
    engine = _make_engine()
    engine.rest = FakeRest(positions=[
        {"symbol": "C-BTC-63000-070726", "product_id": 999, "size": -1},
    ])
    await engine._sync_options_to_exchange()
    assert engine.executor.has_open_position
    assert engine.executor.tracked_symbol == "C-BTC-63000-070726"


# ---------------------------------------------------------------------- #
# EOD square-off
# ---------------------------------------------------------------------- #
async def test_eod_square_off_closes_and_flattens() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0)
    await engine._square_off_all()
    assert engine.executor.close_calls == 1
    assert engine._entry_premium is None
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["reason"] == "EOD"


# ---------------------------------------------------------------------- #
# Self-heal: position closed OUTSIDE the bot
# ---------------------------------------------------------------------- #
class VerifyRest(FakeRest):
    """get_option_positions returns whatever we set; can also raise."""
    def __init__(self, positions=None, raises=False):
        super().__init__(positions=positions)
        self.raises = raises
        self.calls = 0

    def get_option_positions(self, underlying):
        self.calls += 1
        if self.raises:
            raise RuntimeError("flaky api")
        return self._positions


async def _open_and_arm(engine):
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0)
    engine.executor.tracked_product_id = 123
    engine._last_verify = 0.0


async def test_selfheal_flattens_when_position_gone_after_two_misses() -> None:
    engine = _make_engine(position_verify_seconds=0.0001)
    await _open_and_arm(engine)
    engine.rest = VerifyRest(positions=[])          # exchange shows NOTHING

    await engine._maybe_verify_position()           # 1st miss -> only warns
    assert engine.executor.has_open_position        # still tracked, not dropped yet

    engine._last_verify = 0.0
    await engine._maybe_verify_position()           # 2nd miss -> self-heal
    assert not engine.executor.has_open_position
    assert engine._entry_premium is None and engine._tp_price is None
    assert engine.strategy.position_state == PositionState.FLAT


async def test_selfheal_does_not_drop_position_still_on_exchange() -> None:
    engine = _make_engine(position_verify_seconds=0.0001)
    await _open_and_arm(engine)
    engine.rest = VerifyRest(positions=[{"symbol": "P-BTC-60000-070726", "product_id": 123, "size": -1}])

    for _ in range(3):
        engine._last_verify = 0.0
        await engine._maybe_verify_position()
    assert engine.executor.has_open_position        # never dropped
    assert engine._verify_misses == 0


async def test_selfheal_never_drops_position_on_fetch_error() -> None:
    engine = _make_engine(position_verify_seconds=0.0001)
    await _open_and_arm(engine)
    engine.rest = VerifyRest(raises=True)           # API keeps failing

    for _ in range(5):
        engine._last_verify = 0.0
        await engine._maybe_verify_position()
    assert engine.executor.has_open_position        # a flaky API must NEVER flatten us
    assert engine._verify_misses == 0


# ---------------------------------------------------------------------- #
# 2026-07-14: DELTA_SKIP_WEEKDAYS now actually blocks new entries
# ---------------------------------------------------------------------- #
def test_entries_blocked_on_skip_weekday(monkeypatch) -> None:
    engine = _make_engine(skip_weekdays="Sat,Sun")
    saturday = datetime(2026, 7, 11, 12, 0, tzinfo=_dch_ist)  # 2026-07-11 is a Saturday

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return saturday

    import deltabot.core.dchannel_trader as mod
    monkeypatch.setattr(mod, "datetime", _FakeDatetime)
    assert engine._entries_blocked() is True


def test_entries_not_blocked_on_non_skip_weekday(monkeypatch) -> None:
    engine = _make_engine(skip_weekdays="Sat,Sun")
    wednesday = datetime(2026, 7, 8, 12, 0, tzinfo=_dch_ist)  # 2026-07-08 is a Wednesday

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return wednesday

    import deltabot.core.dchannel_trader as mod
    monkeypatch.setattr(mod, "datetime", _FakeDatetime)
    assert engine._entries_blocked() is False
