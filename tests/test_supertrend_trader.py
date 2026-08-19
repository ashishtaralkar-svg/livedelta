"""SupertrendFixedSlEngine: the live dual-leg SELL engine for
SupertrendFixedSlStrategy. Mirrors test_ema21_trader.py's structure, but
every entry/exit/reconcile/self-heal case is doubled for the two
INDEPENDENT legs (short/CE, long/PE) this engine tracks simultaneously --
see the module docstring in src/deltabot/core/supertrend_trader.py."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from deltabot.config import Settings
from deltabot.core.supertrend_trader import SupertrendFixedSlEngine
from deltabot.enums import NotifyEvent, SignalDir
from deltabot.models import Candle

_ist = ZoneInfo("Asia/Kolkata")


class FakeExecutor:
    """Mirrors test_ema21_trader's FakeExecutor -- a SHORT (sold) position
    (both legs of this strategy sell)."""

    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.underlying = "BTC"
        self.is_buy_side = False
        self.open_calls: list[tuple[int, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (1400.0, "C-BTC-64000-070826")
        self._close_result: float | None = 1000.0   # decayed, for profit tests

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
    def __init__(self, positions=None) -> None:
        self._positions = positions or []

    def get_option_positions(self, underlying):
        return self._positions


def _make_engine(**kw) -> SupertrendFixedSlEngine:
    base = dict(strategy="supertrend", target_premium=1400.0,
                option_contracts=10, state_file="", skip_weekdays="")
    base.update(kw)
    settings = Settings(_env_file=None, **base)
    engine = SupertrendFixedSlEngine(settings, rest=FakeRest(), notifier=AsyncMock())
    engine.executor_short = FakeExecutor()
    engine.executor_long = FakeExecutor()
    engine.short.executor = engine.executor_short
    engine.long.executor = engine.executor_long
    return engine


def _c(start: int, o=100.0, h=101.0, low=99.0, cl=100.0) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _exit_calls(notifier):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == NotifyEvent.EXIT]


class _Dec:
    """Minimal stand-in for SupertrendFixedSlDecision -- lets tests drive
    _handle_closed_candle without reverse-engineering real Supertrend math."""
    def __init__(self, *, short_exit=False, long_exit=False, short_exit_price=0.0,
                 long_exit_price=0.0, sell_signal=False, buy_signal=False,
                 short_sl=None, long_sl=None):
        self.short_exit = short_exit
        self.long_exit = long_exit
        self.short_exit_price = short_exit_price
        self.long_exit_price = long_exit_price
        self.sell_signal = sell_signal
        self.buy_signal = buy_signal
        self.short_sl = short_sl
        self.long_sl = long_sl


# ---------------------------------------------------------------------- #
# Entry: bearish flip -> sell CALL (short leg), bullish flip -> sell PUT (long leg)
# ---------------------------------------------------------------------- #
async def test_bearish_flip_opens_short_leg_sells_call() -> None:
    engine = _make_engine()
    await engine._open_leg(engine.short, SignalDir.SHORT.value, sl_level=65000.0, btc_price=64000.0)
    assert engine.executor_short.open_calls == [(SignalDir.SHORT.value, 1400.0)]
    assert engine.short.entry_premium == 1400.0
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_SHORT and ev.kwargs["direction"] == "CALL"


async def test_bullish_flip_opens_long_leg_sells_put() -> None:
    engine = _make_engine()
    engine.executor_long._open_result = (1400.0, "P-BTC-64000-070726")
    await engine._open_leg(engine.long, SignalDir.LONG.value, sl_level=63000.0, btc_price=64000.0)
    assert engine.executor_long.open_calls == [(SignalDir.LONG.value, 1400.0)]
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_LONG and ev.kwargs["direction"] == "PUT"


async def test_both_legs_can_be_open_simultaneously() -> None:
    """The whole point of this strategy -- unlike every strict-single-trade
    engine, a bearish AND a bullish flip in sequence must open BOTH legs."""
    engine = _make_engine()
    engine.strategy.update = lambda candle: _Dec(sell_signal=True, short_sl=65000.0)
    await engine._handle_closed_candle(_c(300))
    engine.strategy.update = lambda candle: _Dec(buy_signal=True, long_sl=63000.0)
    await engine._handle_closed_candle(_c(600))
    assert engine.executor_short.has_open_position
    assert engine.executor_long.has_open_position


async def test_open_leg_guarded_when_already_open() -> None:
    engine = _make_engine()
    engine.executor_short.has_open_position = True
    await engine._open_leg(engine.short, SignalDir.SHORT.value, 65000.0, 64000.0)
    assert engine.executor_short.open_calls == []


async def test_no_fill_flattens_only_that_leg() -> None:
    engine = _make_engine()
    engine.executor_short._open_result = (None, None)
    engine.strategy._in_short = True
    await engine._open_leg(engine.short, SignalDir.SHORT.value, 65000.0, 64000.0)
    assert engine.short.entry_premium is None
    assert not engine.strategy.in_short


# ---------------------------------------------------------------------- #
# Exits + P&L sign (SELL side: profit = entry - exit, opposite of ema21bot)
# ---------------------------------------------------------------------- #
async def test_close_leg_pnl_sign_profits_when_premium_decayed() -> None:
    engine = _make_engine()
    await engine._open_leg(engine.short, SignalDir.SHORT.value, 65000.0, 64000.0)
    engine.executor_short._close_result = 900.0   # decayed from 1400 -> 900: a PROFIT
    await engine._close_leg(engine.short, "SL", 65000.0)
    exits = _exit_calls(engine.notifier)
    assert exits and exits[-1].kwargs["pnl"] > 0
    assert exits[-1].kwargs["reason"] == "SL"


async def test_sl_exit_on_one_leg_does_not_touch_the_other() -> None:
    engine = _make_engine()
    await engine._open_leg(engine.short, SignalDir.SHORT.value, 65000.0, 64000.0)
    await engine._open_leg(engine.long, SignalDir.LONG.value, 63000.0, 64000.0)
    engine.strategy.update = lambda candle: _Dec(short_exit=True, short_exit_price=65000.0)
    await engine._handle_closed_candle(_c(900))
    assert not engine.executor_short.has_open_position
    assert engine.executor_long.has_open_position   # untouched
    assert engine.long.entry_premium == 1400.0       # long leg's tracking intact


async def test_double_close_guard() -> None:
    engine = _make_engine()
    await engine._open_leg(engine.short, SignalDir.SHORT.value, 65000.0, 64000.0)
    engine.short.closing = True
    await engine._close_leg(engine.short, "SL", 65000.0)
    assert engine.executor_short.close_calls == 0


# ---------------------------------------------------------------------- #
# Closed-bar dispatch: exit + entry can BOTH fire in the same bar, for
# DIFFERENT legs (e.g. CE stops out while PE enters).
# ---------------------------------------------------------------------- #
async def test_exit_and_entry_fire_same_bar_for_different_legs() -> None:
    engine = _make_engine()
    await engine._open_leg(engine.short, SignalDir.SHORT.value, 65000.0, 64000.0)
    engine.strategy.update = lambda candle: _Dec(
        short_exit=True, short_exit_price=65000.0, buy_signal=True, long_sl=63000.0)
    await engine._handle_closed_candle(_c(900))
    assert not engine.executor_short.has_open_position   # closed
    assert engine.executor_long.has_open_position         # opened


async def test_closed_bar_no_action_when_no_decision() -> None:
    engine = _make_engine()
    engine.strategy.update = lambda candle: None
    await engine._handle_closed_candle(_c(900))
    assert engine.executor_short.open_calls == []
    assert engine.executor_long.open_calls == []


# ---------------------------------------------------------------------- #
# 17:25 square-off: closes whichever legs are open, always force_flat()
# ---------------------------------------------------------------------- #
def _fake_now(monkeypatch, dt):
    import deltabot.core.supertrend_trader as mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return dt
    monkeypatch.setattr(mod, "datetime", _FakeDatetime)


async def test_square_off_closes_both_open_legs(monkeypatch) -> None:
    engine = _make_engine()
    await engine._open_leg(engine.short, SignalDir.SHORT.value, 65000.0, 64000.0)
    await engine._open_leg(engine.long, SignalDir.LONG.value, 63000.0, 64000.0)
    _fake_now(monkeypatch, datetime(2026, 7, 10, 17, 25, tzinfo=_ist))
    await engine._square_off()
    assert engine.executor_short.close_calls == 1
    assert engine.executor_long.close_calls == 1
    assert not engine.strategy.in_short and not engine.strategy.in_long
    exits = _exit_calls(engine.notifier)
    assert len(exits) == 2 and all(e.kwargs["reason"] == "EOD" for e in exits)


async def test_square_off_noop_when_both_flat(monkeypatch) -> None:
    engine = _make_engine()
    _fake_now(monkeypatch, datetime(2026, 7, 10, 17, 25, tzinfo=_ist))
    await engine._square_off()
    assert engine.executor_short.close_calls == 0
    assert engine.executor_long.close_calls == 0


async def test_square_off_closes_only_the_open_leg(monkeypatch) -> None:
    """Only the short leg is open -- square-off must not touch the long
    executor at all."""
    engine = _make_engine()
    await engine._open_leg(engine.short, SignalDir.SHORT.value, 65000.0, 64000.0)
    _fake_now(monkeypatch, datetime(2026, 7, 10, 17, 25, tzinfo=_ist))
    await engine._square_off()
    assert engine.executor_short.close_calls == 1
    assert engine.executor_long.close_calls == 0


async def test_entries_blocked_only_checks_weekday(monkeypatch) -> None:
    engine = _make_engine(skip_weekdays="Sat,Sun")
    _fake_now(monkeypatch, datetime(2026, 7, 8, 12, 0, tzinfo=_ist))   # Wed
    assert engine._entries_blocked() is False
    _fake_now(monkeypatch, datetime(2026, 7, 11, 12, 0, tzinfo=_ist))  # Sat
    assert engine._entries_blocked() is True


# ---------------------------------------------------------------------- #
# Self-heal: verifies each leg independently
# ---------------------------------------------------------------------- #
class VerifyRest(FakeRest):
    def __init__(self, positions=None, raises=False):
        super().__init__(positions=positions)
        self.raises = raises

    def get_option_positions(self, underlying):
        if self.raises:
            raise RuntimeError("flaky api")
        return self._positions


async def test_selfheal_flattens_after_two_misses_short_leg_only() -> None:
    engine = _make_engine(position_verify_seconds=0.0001)
    await engine._open_leg(engine.short, SignalDir.SHORT.value, 65000.0, 64000.0)
    await engine._open_leg(engine.long, SignalDir.LONG.value, 63000.0, 64000.0)
    engine.short.executor.tracked_product_id = 123
    engine.rest = VerifyRest(positions=[])   # neither leg found on exchange

    engine.short.last_verify = 0.0
    engine.long.last_verify = 0.0
    await engine._maybe_verify_leg(engine.short)
    await engine._maybe_verify_leg(engine.long)
    assert engine.executor_short.has_open_position   # still 1st miss
    assert engine.executor_long.has_open_position

    engine.short.last_verify = 0.0
    engine.long.last_verify = 0.0
    await engine._maybe_verify_leg(engine.short)
    await engine._maybe_verify_leg(engine.long)
    assert not engine.executor_short.has_open_position   # 2nd miss -> self-healed
    assert not engine.executor_long.has_open_position


async def test_selfheal_never_drops_on_fetch_error() -> None:
    engine = _make_engine(position_verify_seconds=0.0001)
    await engine._open_leg(engine.short, SignalDir.SHORT.value, 65000.0, 64000.0)
    engine.short.executor.tracked_product_id = 123
    engine.rest = VerifyRest(raises=True)
    for _ in range(5):
        engine.short.last_verify = 0.0
        await engine._maybe_verify_leg(engine.short)
    assert engine.executor_short.has_open_position


# ---------------------------------------------------------------------- #
# Reconcile: disambiguates CE vs PE by SYMBOL PREFIX, not sign (both legs
# are short on the exchange).
# ---------------------------------------------------------------------- #
async def test_reconcile_adopts_both_legs_by_symbol_prefix() -> None:
    engine = _make_engine()
    engine.rest = FakeRest(positions=[
        {"symbol": "C-BTC-64000-070726", "product_id": 111, "size": -10},
        {"symbol": "P-BTC-62000-070726", "product_id": 222, "size": -10},
    ])
    await engine._sync_options_to_exchange()
    assert engine.executor_short.has_open_position
    assert engine.executor_short.tracked_symbol == "C-BTC-64000-070726"
    assert engine.executor_long.has_open_position
    assert engine.executor_long.tracked_symbol == "P-BTC-62000-070726"


async def test_reconcile_lone_put_does_not_leak_into_short_leg() -> None:
    engine = _make_engine()
    engine.rest = FakeRest(positions=[
        {"symbol": "P-BTC-62000-070726", "product_id": 222, "size": -10},
    ])
    await engine._sync_options_to_exchange()
    assert not engine.executor_short.has_open_position   # no CE found -- stays flat
    assert engine.executor_long.has_open_position
    assert engine.executor_long.tracked_symbol == "P-BTC-62000-070726"


async def test_reconcile_preserves_state_when_exchange_empty_but_owned(tmp_path) -> None:
    state_path = tmp_path / "supertrend_pos.json"
    short_path = tmp_path / "supertrend_pos_short.json"
    short_path.write_text(
        '{"symbol": "C-BTC-64000-070726", "product_id": 999, "size": 10, "entry_premium": 1400.0}'
    )
    engine = _make_engine(state_file=str(state_path))
    engine.short.state_file = str(short_path)
    engine.rest = FakeRest(positions=[])
    engine._reconcile_leg(engine.short, [], __import__("deltabot.enums", fromlist=["OptionType"]).OptionType.CALL)
    assert engine.executor_short.has_open_position
    assert engine.executor_short.tracked_symbol == "C-BTC-64000-070726"
    assert engine.short.entry_premium == 1400.0


def test_leg_path_derivation() -> None:
    from pathlib import Path

    from deltabot.core.supertrend_trader import _leg_path
    assert Path(_leg_path("state/supertrend_pos.json", "short")) == Path("state/supertrend_pos_short.json")
    assert Path(_leg_path("state/supertrend_pos.json", "long")) == Path("state/supertrend_pos_long.json")
    assert _leg_path("", "short") == ""
