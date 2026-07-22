"""Paper (shadow) mode: PaperExecutor places NO orders but tracks realistic
contracts/marks, and DCv2Engine(paper_mode=True) skips reconcile + self-heal
so a paper leg is never flattened by the exchange-verify logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deltabot.config import Settings
from deltabot.core.dcv2_trader import DCv2Engine
from deltabot.core.options_executor import PaperExecutor
from deltabot.enums import OptionType, SignalDir
from deltabot.models import Candle


def _settings(**kw) -> Settings:
    base = dict(strategy="dcv2", paper_mode=True, state_file="", option_contracts=25,
                target_premium=900.0, take_profit_pct=70.0, dcv2_intracandle_enabled=False)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _rest_with_chain(mark=900.0):
    rest = MagicMock()
    rest.get_option_chain = MagicMock(side_effect=lambda underlying, expiry, otype: (
        [{"symbol": "P-BTC-64000-180726", "strike": 64000, "product_id": 222, "mark_price": mark}]
        if otype == OptionType.PUT else
        [{"symbol": "C-BTC-64000-180726", "strike": 64000, "product_id": 111, "mark_price": mark}]))
    rest.get_mark_price = MagicMock(return_value=270.0)   # paper close reads this
    return rest


# ---------------------------------------------------------------------- #
# PaperExecutor: selects real contracts, reads real marks, NEVER orders
# ---------------------------------------------------------------------- #
async def test_paper_open_places_no_order_but_tracks() -> None:
    rest = _rest_with_chain(mark=905.0)
    ex = PaperExecutor(rest, _settings())
    fill, symbol = await ex.open_option_by_premium(SignalDir.LONG.value, 900.0)
    assert symbol == "P-BTC-64000-180726"     # bullish -> PUT (sell side)
    assert fill == 905.0                       # "fill" = current mark
    assert ex.has_open_position
    rest.place_market_order.assert_not_called()   # THE key assertion: no real order


async def test_paper_close_reads_mark_no_order() -> None:
    rest = _rest_with_chain()
    ex = PaperExecutor(rest, _settings())
    await ex.open_option_by_premium(SignalDir.LONG.value, 900.0)
    fill = await ex.close_option()
    assert fill == 270.0                       # from get_mark_price
    assert not ex.has_open_position
    rest.place_market_order.assert_not_called()


async def test_paper_guarded_against_double_open() -> None:
    ex = PaperExecutor(_rest_with_chain(), _settings())
    await ex.open_option_by_premium(SignalDir.LONG.value, 900.0)
    fill, sym = await ex.open_option_by_premium(SignalDir.LONG.value, 900.0)
    assert fill is None and sym is None


# ---------------------------------------------------------------------- #
# Engine paper-mode: uses PaperExecutor, skips reconcile + self-heal
# ---------------------------------------------------------------------- #
def _make_engine() -> DCv2Engine:
    engine = DCv2Engine(_settings(), rest=_rest_with_chain(), notifier=AsyncMock())
    return engine


async def test_engine_uses_paper_executor() -> None:
    assert isinstance(_make_engine().executor, PaperExecutor)


async def test_reconcile_paper_starts_flat_without_exchange_call() -> None:
    engine = _make_engine()
    await engine._sync_options_to_exchange()
    engine.rest.get_option_positions.assert_not_called()   # no signed call in paper mode
    assert not engine.executor.has_open_position


async def test_selfheal_is_noop_in_paper_mode() -> None:
    """Critically: a paper position (not on the exchange) must NOT be flattened
    by self-heal -- otherwise every shadow trade would vanish after 2 polls."""
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    engine._last_verify = 0.0
    for _ in range(5):
        engine._last_verify = 0.0
        await engine._maybe_verify_position()
    assert engine.executor.has_open_position          # never dropped
    engine.rest.get_option_positions.assert_not_called()


async def test_paper_entry_notifies_and_holds() -> None:
    engine = _make_engine()
    await engine._open_entry(SignalDir.LONG.value, 59000.0, 60000.0, tag="ENTRY")
    assert engine.executor.has_open_position
    assert engine._entry_premium is not None
    assert engine.rest.place_market_order.call_count == 0   # never trades for real
