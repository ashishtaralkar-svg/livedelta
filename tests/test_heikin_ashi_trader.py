"""HeikinAshiEngine: decision -> executor-call translation, ASAP intracandle
entry/SL/TRAIL, and double-open/double-close guards.

Uses a lightweight FakeExecutor test double instead of a real OptionsExecutor
so these tests isolate the engine's own logic (not OptionsExecutor's REST
internals, which are out of scope here).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from deltabot.config import Settings
from deltabot.core.heikin_ashi_trader import HeikinAshiEngine
from deltabot.enums import NotifyEvent, PositionState, SignalDir
from deltabot.models import Candle


class FakeExecutor:
    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.underlying = "BTC"
        self.open_calls: list[tuple[int, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (900.0, "P-BTC-60000-070726")
        self._close_result: float | None = 700.0

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


def _make_engine(**settings_kwargs) -> HeikinAshiEngine:
    settings = Settings(_env_file=None, state_file="", **settings_kwargs)
    engine = HeikinAshiEngine(settings, rest=object(), notifier=AsyncMock())
    engine.executor = FakeExecutor()
    return engine


def _c(start: int, o: float, h: float, low: float, cl: float) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


# ---------------------------------------------------------------------- #
# Entry: decision -> executor-call translation
# ---------------------------------------------------------------------- #
async def test_open_entry_sells_put_on_buy_signal() -> None:
    engine = _make_engine(target_premium=900.0)
    await engine._open_entry(SignalDir.LONG.value, sl_level=59000.0, btc_price=60000.0)

    assert engine.executor.open_calls == [(SignalDir.LONG.value, 900.0)]
    assert engine._entry_premium == 900.0
    engine.notifier.notify.assert_awaited_once()
    event = engine.notifier.notify.await_args.args[0]
    kwargs = engine.notifier.notify.await_args.kwargs
    assert event == NotifyEvent.ENTRY_LONG
    assert kwargs["direction"] == "PUT"
    assert kwargs["contract"] == "P-BTC-60000-070726"


async def test_open_entry_sells_call_on_sell_signal() -> None:
    engine = _make_engine(target_premium=900.0)
    await engine._open_entry(SignalDir.SHORT.value, sl_level=61000.0, btc_price=60000.0)

    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 900.0)]
    event = engine.notifier.notify.await_args.args[0]
    kwargs = engine.notifier.notify.await_args.kwargs
    assert event == NotifyEvent.ENTRY_SHORT
    assert kwargs["direction"] == "CALL"


async def test_open_entry_guarded_when_already_open() -> None:
    engine = _make_engine()
    engine.executor.has_open_position = True
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0)
    assert engine.executor.open_calls == []  # never called -- already open


async def test_open_entry_skips_and_flattens_on_no_fill() -> None:
    engine = _make_engine()
    engine.executor._open_result = (None, None)
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0)
    assert engine._entry_premium is None
    assert engine.strategy.position_state == PositionState.FLAT


# ---------------------------------------------------------------------- #
# Exit: decision -> executor-call translation
# ---------------------------------------------------------------------- #
async def test_close_buys_back_and_notifies_exit() -> None:
    engine = _make_engine()
    engine.executor.has_open_position = True
    engine.executor.tracked_symbol = "P-BTC-60000-070726"
    engine._entry_premium = 900.0

    await engine._close("SL")

    assert engine.executor.close_calls == 1
    assert engine._entry_premium is None
    event = engine.notifier.notify.await_args.args[0]
    kwargs = engine.notifier.notify.await_args.kwargs
    assert event == NotifyEvent.EXIT
    assert kwargs["reason"] == "SL"
    assert kwargs["exit_premium"] == 700.0


async def test_close_guarded_when_nothing_open() -> None:
    engine = _make_engine()
    await engine._close("SL")
    assert engine.executor.close_calls == 0


# ---------------------------------------------------------------------- #
# ASAP intracandle: entry trigger, fixed SL, and TRAIL all fire on real price
# crossing a level, without waiting for the 1-minute candle to close.
# ---------------------------------------------------------------------- #
async def test_forming_candle_asap_sl_closes_immediately() -> None:
    engine = _make_engine()
    engine.strategy._warmup_bars = 999
    engine.strategy._st._ready = True
    engine.executor.has_open_position = True
    engine.executor.tracked_symbol = "P-BTC-60000-070726"
    engine._entry_premium = 900.0
    engine.strategy._in_long, engine.strategy._in_short = True, False
    engine.strategy._sl_level = 59000.0

    forming = _c(1_700_000_000, 59100.0, 59150.0, 58990.0, 59050.0)  # low breaches SL
    await engine._handle_forming_candle(forming)

    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.FLAT


async def test_forming_candle_asap_trail_closes_immediately() -> None:
    engine = _make_engine()
    engine.strategy._warmup_bars = 999
    engine.strategy._st._ready = True
    engine.executor.has_open_position = True
    engine.executor.tracked_symbol = "P-BTC-60000-070726"
    engine._entry_premium = 900.0
    engine.strategy._in_long, engine.strategy._in_short = True, False
    engine.strategy._sl_level = 50000.0  # far away -- SL must not be the cause
    engine.strategy._trail_sl_level = 59000.0  # simulate an already-armed trail stop

    forming = _c(1_700_000_000, 59100.0, 59150.0, 58990.0, 59050.0)  # low drops below trail
    await engine._handle_forming_candle(forming)

    assert engine.executor.close_calls == 1
    assert engine.strategy.position_state == PositionState.FLAT
    kwargs = engine.notifier.notify.await_args.kwargs
    assert kwargs["reason"] == "TRAIL"


async def test_forming_candle_sl_takes_priority_over_trail() -> None:
    """Same convention as the closed-candle path: SL wins over TRAIL on a
    same-tick cross."""
    engine = _make_engine()
    engine.strategy._warmup_bars = 999
    engine.strategy._st._ready = True
    engine.executor.has_open_position = True
    engine.executor.tracked_symbol = "P-BTC-60000-070726"
    engine._entry_premium = 900.0
    engine.strategy._in_long, engine.strategy._in_short = True, False
    engine.strategy._sl_level = 59000.0
    engine.strategy._trail_sl_level = 59500.0  # would ALSO fire TRAIL on this bar's low

    forming = _c(1_700_000_000, 59100.0, 59150.0, 58990.0, 59050.0)
    await engine._handle_forming_candle(forming)

    kwargs = engine.notifier.notify.await_args.kwargs
    assert kwargs["reason"] == "SL"


async def test_forming_candle_asap_entry_confirms_breakout() -> None:
    engine = _make_engine(target_premium=900.0)
    engine.strategy._warmup_bars = 999
    engine.strategy._st._ready = True
    engine.strategy._pending_long = True
    engine.strategy._pending_short = False
    engine.strategy._pending_trigger = 60_300.0
    engine.strategy._pending_sl = 60_000.0

    forming = _c(1, 60_050.0, 60_310.0, 60_020.0, 60_290.0)  # crosses the trigger
    await engine._handle_forming_candle(forming)

    assert engine.executor.open_calls == [(SignalDir.LONG.value, 900.0)]
    assert engine.strategy.position_state == PositionState.LONG


async def test_forming_candle_no_checks_before_strategy_ready() -> None:
    """Guards against acting on an immature Supertrend/EMA200 -- mirrors the
    strategy's own `ready` gate."""
    engine = _make_engine()
    engine.strategy._pending_long = True
    engine.strategy._pending_trigger = 60_300.0
    engine.strategy._pending_sl = 60_000.0
    # strategy.ready is False by default (fresh strategy, no warmup).

    forming = _c(1, 60_050.0, 60_310.0, 60_020.0, 60_290.0)
    await engine._handle_forming_candle(forming)

    assert engine.executor.open_calls == []
