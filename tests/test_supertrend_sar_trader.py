"""SupertrendSarEngine: the live sell-side engine for SupertrendSarStrategy.
Mirrors test_ema21_trader.py's FakeExecutor/FakeRest shape and P&L-sign/
reconcile/self-heal conventions, but SELL-side (profit on decay, reconcile
looks for size<0) and with the extra TP-roll (_close_and_roll) mechanic
this engine adds on top of the strategy's own SL/entry Decisions."""

from __future__ import annotations

from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from deltabot.config import Settings
from deltabot.core.supertrend_sar_trader import SupertrendSarEngine
from deltabot.enums import NotifyEvent, SignalDir
from deltabot.models import Candle
from deltabot.strategy.supertrend_sar import SupertrendSarDecision

_ist = ZoneInfo("Asia/Kolkata")


class FakeExecutor:
    """A SHORT (sold) position -- mirrors test_ema21_trader's FakeExecutor
    but is_buy_side=False and reconcile/self-heal check size<0."""

    def __init__(self) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.tracked_size: int = 0
        self.underlying = "BTC"
        self.is_buy_side = False
        self.open_calls: list[tuple[int, float]] = []
        self.close_calls = 0
        self._open_result: tuple[float | None, str | None] = (1400.0, "C-BTC-76000-050926")
        self._open_size = 10
        self._close_result: float | None = 700.0   # 50% decay, for TP-roll tests

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
        self.tracked_size = abs(size)


class FakeRest:
    def __init__(self, positions=None, mark=None) -> None:
        self._positions = positions or []
        self._mark = mark

    def get_option_positions(self, underlying):
        return self._positions

    def get_mark_price(self, symbol):
        return self._mark


def _make_engine(**kw) -> SupertrendSarEngine:
    base = dict(strategy="sar", target_premium=1400.0, sar_tp_pct=50.0,
                option_contracts=10, option_side="sell", state_file="", skip_weekdays="")
    base.update(kw)
    settings = Settings(_env_file=None, **base)
    engine = SupertrendSarEngine(settings, rest=FakeRest(), notifier=AsyncMock())
    engine.executor = FakeExecutor()
    return engine


def _c(start: int, o=79000.0, h=79100.0, low=78900.0, cl=79000.0) -> Candle:
    return Candle(start_time=start, open=o, high=h, low=low, close=cl, volume=1.0)


def _dec(*, exit=False, exit_price=0.0, exit_was_short=False,
         entry=False, entry_is_short=False, entry_price=79000.0, sl_level=None,
         candle=None) -> SupertrendSarDecision:
    return SupertrendSarDecision(
        candle=candle or _c(1000), exit=exit, exit_price=exit_price, exit_was_short=exit_was_short,
        entry_signal=entry, entry_is_short=entry_is_short, entry_price=entry_price, sl_level=sl_level,
    )


def _exit_calls(notifier):
    return [c for c in notifier.notify.await_args_list if c.args and c.args[0] == NotifyEvent.EXIT]


# ---------------------------------------------------------------------- #
# Entry: entry_is_short=True -> sell CALL (SignalDir.SHORT); False -> sell
# PUT (SignalDir.LONG). TP price is a DECAY target (below entry).
# ---------------------------------------------------------------------- #
async def test_entry_is_short_true_sells_call_via_signal_dir_short() -> None:
    engine = _make_engine()
    await engine._open_entry(True, sl_level=79500.0, btc_price=79000.0)
    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 1400.0)]
    assert engine._entry_premium == 1400.0
    assert engine._tp_price == pytest.approx(700.0)   # 1400 * 0.50
    assert engine._current_is_short is True
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_SHORT and ev.kwargs["direction"] == "CALL"
    assert ev.kwargs["side"] == "sell"


async def test_entry_is_short_false_sells_put_via_signal_dir_long() -> None:
    engine = _make_engine()
    await engine._open_entry(False, sl_level=78500.0, btc_price=79000.0)
    assert engine.executor.open_calls == [(SignalDir.LONG.value, 1400.0)]
    assert engine._current_is_short is False
    ev = engine.notifier.notify.await_args
    assert ev.args[0] == NotifyEvent.ENTRY_LONG and ev.kwargs["direction"] == "PUT"


async def test_sar_tp_pct_zero_disables_the_decay_target() -> None:
    engine = _make_engine(sar_tp_pct=0.0)
    await engine._open_entry(True, sl_level=79500.0, btc_price=79000.0)
    assert engine._tp_price is None
    ev = engine.notifier.notify.await_args
    assert ev.kwargs["tp_price"] is None


async def test_open_entry_noop_when_already_in_progress_or_open() -> None:
    engine = _make_engine()
    engine.executor.has_open_position = True
    await engine._open_entry(True, sl_level=None, btc_price=79000.0)
    assert engine.executor.open_calls == []   # guarded, never even tries


# ---------------------------------------------------------------------- #
# P&L sign: SELL side, profit = entry - exit (opposite of a buy-side engine).
# ---------------------------------------------------------------------- #
async def test_close_leg_pnl_is_entry_minus_exit() -> None:
    engine = _make_engine()
    await engine._open_entry(True, sl_level=None, btc_price=79000.0)   # entry 1400.0
    engine.executor._close_result = 900.0   # decayed further, still a win
    await engine._close_leg("SL", btc_exit_price=79200.0)
    assert engine.executor.close_calls == 1
    ev = _exit_calls(engine.notifier)[-1]
    assert ev.kwargs["pnl"] == pytest.approx((1400.0 - 900.0) * 10 * 0.001)
    assert ev.kwargs["side"] == "sell"
    assert engine._entry_premium is None   # cleared


async def test_close_leg_pnl_negative_when_premium_rises_against_the_seller() -> None:
    engine = _make_engine()
    await engine._open_entry(True, sl_level=None, btc_price=79000.0)   # entry 1400.0
    engine.executor._close_result = 2000.0   # moved against the seller
    await engine._close_leg("SL", btc_exit_price=79800.0)
    ev = _exit_calls(engine.notifier)[-1]
    assert ev.kwargs["pnl"] == pytest.approx((1400.0 - 2000.0) * 10 * 0.001)
    assert ev.kwargs["pnl"] < 0


# ---------------------------------------------------------------------- #
# TP-roll: closes, then IMMEDIATELY resells the SAME direction -- purely
# option-level, strategy.update() never called for this.
# ---------------------------------------------------------------------- #
async def test_tp_roll_reopens_the_same_direction_immediately() -> None:
    engine = _make_engine()
    await engine._open_entry(True, sl_level=None, btc_price=79000.0)   # short/CE, entry 1400.0
    engine.executor._close_result = 700.0   # 50% decay -- TP level
    engine.executor._open_result = (1350.0, "C-BTC-77000-050926")   # the freshly-rolled contract
    await engine._close_and_roll(700.0)
    assert engine.executor.close_calls == 1
    # Rolled back into ANOTHER short/CE at target_premium -- SAME signal_dir.
    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 1400.0), (SignalDir.SHORT.value, 1400.0)]
    assert engine.executor.has_open_position   # the new leg is open
    assert engine._entry_premium == 1350.0     # the NEW leg's entry, not the old one
    assert engine._current_is_short is True
    tp_call = _exit_calls(engine.notifier)[-1]
    assert tp_call.kwargs["reason"] == "TP"
    assert tp_call.kwargs["pnl"] == pytest.approx((1400.0 - 700.0) * 10 * 0.001)
    entry_call = engine.notifier.notify.await_args_list[-1]
    assert entry_call.args[0] == NotifyEvent.ENTRY_SHORT


async def test_tp_roll_reopens_long_direction_too() -> None:
    engine = _make_engine()
    engine.executor._open_result = (1400.0, "P-BTC-82000-050926")
    await engine._open_entry(False, sl_level=None, btc_price=79000.0)   # long/PE
    engine.executor._close_result = 700.0
    engine.executor._open_result = (1380.0, "P-BTC-81000-050926")
    await engine._close_and_roll(700.0)
    assert engine.executor.open_calls[-1] == (SignalDir.LONG.value, 1400.0)
    assert engine._current_is_short is False


async def test_tp_roll_noop_when_not_in_position() -> None:
    engine = _make_engine()
    await engine._close_and_roll(700.0)
    assert engine.executor.close_calls == 0
    assert engine.executor.open_calls == []


# ---------------------------------------------------------------------- #
# Closed-candle handling: SL-hit + same-bar reversal (one Decision carries
# both an exit and an entry -- executor must be flat again by the time the
# entry half runs).
# ---------------------------------------------------------------------- #
async def test_sl_hit_and_reversal_close_then_immediately_reopen_opposite_side() -> None:
    engine = _make_engine()
    await engine._open_entry(True, sl_level=None, btc_price=79000.0)   # short/CE open
    engine.executor._close_result = 1600.0   # SL loss
    engine.executor._open_result = (1420.0, "P-BTC-82000-050926")   # reversal's fresh PE
    dec = _dec(exit=True, exit_price=79500.0, exit_was_short=True,
               entry=True, entry_is_short=False, sl_level=78000.0)
    engine.strategy.update = lambda candle: dec
    await engine._handle_closed_candle(_c(2000))
    assert engine.executor.close_calls == 1
    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 1400.0), (SignalDir.LONG.value, 1400.0)]
    assert engine.executor.has_open_position
    assert engine._current_is_short is False
    reasons = [c.kwargs["reason"] for c in _exit_calls(engine.notifier)]
    assert reasons[-1] == "SL"


async def test_day_first_entry_decision_with_no_prior_position_opens_directly() -> None:
    engine = _make_engine()
    dec = _dec(entry=True, entry_is_short=True, sl_level=79800.0)
    engine.strategy.update = lambda candle: dec
    await engine._handle_closed_candle(_c(3000))
    assert engine.executor.open_calls == [(SignalDir.SHORT.value, 1400.0)]


async def test_none_decision_does_nothing() -> None:
    engine = _make_engine()
    engine.strategy.update = lambda candle: None
    await engine._handle_closed_candle(_c(4000))
    assert engine.executor.open_calls == []
    assert engine.executor.close_calls == 0


async def test_weekday_block_does_not_suppress_a_reversals_reentry() -> None:
    """A skipped weekday blocks fresh entries, but a same-bar reversal must
    still reopen -- otherwise the strategy's own internal state (now
    reversed) would desync from the executor (still flat) and never
    self-correct until a manual restart."""
    engine = _make_engine(skip_weekdays="Mon,Tue,Wed,Thu,Fri,Sat,Sun")  # every day blocked
    await engine._open_entry(True, sl_level=None, btc_price=79000.0)
    engine.executor._close_result = 1600.0
    engine.executor._open_result = (1420.0, "P-BTC-82000-050926")
    dec = _dec(exit=True, exit_price=79500.0, exit_was_short=True,
               entry=True, entry_is_short=False, sl_level=78000.0)
    engine.strategy.update = lambda candle: dec
    await engine._handle_closed_candle(_c(5000))
    assert engine.executor.has_open_position   # the reversal DID reopen despite the block


async def test_weekday_block_suppresses_a_fresh_day_first_entry() -> None:
    engine = _make_engine(skip_weekdays="Mon,Tue,Wed,Thu,Fri,Sat,Sun")
    dec = _dec(entry=True, entry_is_short=True, sl_level=79800.0)
    engine.strategy.update = lambda candle: dec
    await engine._handle_closed_candle(_c(6000))
    assert engine.executor.open_calls == []


# ---------------------------------------------------------------------- #
# Square-off: EOD close, sell-side P&L sign, force_flat() called.
# ---------------------------------------------------------------------- #
async def test_square_off_closes_open_position_with_eod_reason() -> None:
    engine = _make_engine()
    await engine._open_entry(True, sl_level=None, btc_price=79000.0)
    engine.executor._close_result = 1100.0
    await engine._square_off()
    assert engine.executor.close_calls == 1
    ev = _exit_calls(engine.notifier)[-1]
    assert ev.kwargs["reason"] == "EOD"
    assert ev.kwargs["pnl"] == pytest.approx((1400.0 - 1100.0) * 10 * 0.001)
    assert not engine.strategy.in_position


async def test_square_off_noop_when_already_flat() -> None:
    engine = _make_engine()
    await engine._square_off()
    assert engine.executor.close_calls == 0
    assert _exit_calls(engine.notifier) == []


# ---------------------------------------------------------------------- #
# Reconcile: looks for size<0 (SHORT), the sell-side mirror of ema21's size>0.
# ---------------------------------------------------------------------- #
async def test_reconcile_adopts_an_existing_short() -> None:
    engine = _make_engine()
    engine.rest = FakeRest(positions=[
        {"size": -10, "product_id": 55, "symbol": "C-BTC-77000-050926"},
    ])
    await engine._sync_options_to_exchange()
    assert engine.executor.has_open_position
    assert engine.executor.tracked_symbol == "C-BTC-77000-050926"
    assert engine.executor.tracked_size == 10


async def test_reconcile_ignores_a_long_position_from_a_different_bot() -> None:
    """size>0 belongs to some OTHER (buy-mode) bot on the same sub-account
    -- this sell-mode engine must never adopt it."""
    engine = _make_engine()
    engine.rest = FakeRest(positions=[
        {"size": 5, "product_id": 99, "symbol": "P-BTC-80000-050926"},
    ])
    await engine._sync_options_to_exchange()
    assert not engine.executor.has_open_position


async def test_reconcile_flattens_strategy_when_exchange_is_genuinely_empty() -> None:
    engine = _make_engine()
    engine.strategy._in_position = True
    engine.strategy._is_short = True
    engine.rest = FakeRest(positions=[])
    await engine._sync_options_to_exchange()
    assert not engine.executor.has_open_position
    assert not engine.strategy.in_position


# ---------------------------------------------------------------------- #
# Self-heal: only a size<0 match on the tracked product_id counts as "still open".
# ---------------------------------------------------------------------- #
async def test_self_heal_ignores_positive_size_positions() -> None:
    engine = _make_engine(position_verify_seconds=1.0)
    await engine._open_entry(True, sl_level=None, btc_price=79000.0)
    engine._last_verify = 0.0
    engine.rest = FakeRest(positions=[
        {"size": 10, "product_id": 123},   # wrong sign -- must NOT count as a match
    ])
    await engine._maybe_verify_position()
    assert engine._verify_misses == 1   # treated as a miss, not a match
