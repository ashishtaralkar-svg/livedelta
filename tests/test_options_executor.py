"""OptionsExecutor: side-aware (sell/buy) direction mapping, order side on open
and close, and that default (sell) behavior is byte-for-byte unchanged."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deltabot.config import Settings
from deltabot.core.options_executor import OptionsExecutor
from deltabot.enums import OptionType, Side, SignalDir
from deltabot.models import OrderResult


def _settings(**kw) -> Settings:
    base = dict(option_contracts=25, option_min_available_balance=0.0, option_leverage=0)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _fake_rest(fill_price=900.0):
    chain = {
        OptionType.CALL: [{"symbol": "C-BTC-64000-180726", "strike": 64000, "product_id": 111, "mark_price": fill_price}],
        OptionType.PUT: [{"symbol": "P-BTC-64000-180726", "strike": 64000, "product_id": 222, "mark_price": fill_price}],
    }
    rest = MagicMock()
    rest.get_option_chain = MagicMock(side_effect=lambda underlying, expiry, otype: chain[otype])
    rest.place_market_order = MagicMock(return_value=OrderResult(
        order_id=1, product_id=111, side=Side.SELL, size=25, average_fill_price=fill_price,
    ))
    return rest


async def test_sell_side_bull_signal_sells_put_default_unchanged() -> None:
    ex = OptionsExecutor(_fake_rest(), _settings())   # option_side defaults to "sell"
    fill, symbol = await ex.open_option_by_premium(SignalDir.LONG.value, 900.0)
    assert symbol == "P-BTC-64000-180726"
    order_side = ex._rest.place_market_order.call_args.args[2]
    assert order_side == Side.SELL


async def test_sell_side_bear_signal_sells_call() -> None:
    ex = OptionsExecutor(_fake_rest(), _settings(option_side="sell"))
    fill, symbol = await ex.open_option_by_premium(SignalDir.SHORT.value, 900.0)
    assert symbol == "C-BTC-64000-180726"
    assert ex._rest.place_market_order.call_args.args[2] == Side.SELL


async def test_buy_side_bull_signal_buys_call() -> None:
    ex = OptionsExecutor(_fake_rest(), _settings(option_side="buy"))
    fill, symbol = await ex.open_option_by_premium(SignalDir.LONG.value, 500.0)
    assert symbol == "C-BTC-64000-180726"          # bullish -> CALL (mirror of sell side)
    assert ex._rest.place_market_order.call_args.args[2] == Side.BUY


async def test_buy_side_bear_signal_buys_put() -> None:
    ex = OptionsExecutor(_fake_rest(), _settings(option_side="buy"))
    fill, symbol = await ex.open_option_by_premium(SignalDir.SHORT.value, 500.0)
    assert symbol == "P-BTC-64000-180726"
    assert ex._rest.place_market_order.call_args.args[2] == Side.BUY


async def test_sell_side_close_buys_back_reduce_only() -> None:
    ex = OptionsExecutor(_fake_rest(), _settings(option_side="sell"))
    await ex.open_option_by_premium(SignalDir.LONG.value, 900.0)
    ex._rest.place_market_order.reset_mock()
    await ex.close_option()
    args = ex._rest.place_market_order.call_args
    assert args.args[2] == Side.BUY and args.args[3] is True   # reduce_only


async def test_buy_side_close_sells_reduce_only() -> None:
    ex = OptionsExecutor(_fake_rest(), _settings(option_side="buy"))
    await ex.open_option_by_premium(SignalDir.LONG.value, 500.0)
    ex._rest.place_market_order.reset_mock()
    await ex.close_option()
    args = ex._rest.place_market_order.call_args
    assert args.args[2] == Side.SELL and args.args[3] is True


async def test_buy_side_skips_leverage_call() -> None:
    ex = OptionsExecutor(_fake_rest(), _settings(option_side="buy", option_leverage=5))
    await ex.open_option_by_premium(SignalDir.LONG.value, 500.0)
    ex._rest.set_leverage.assert_not_called()


async def test_sell_side_still_sets_leverage_when_configured() -> None:
    ex = OptionsExecutor(_fake_rest(), _settings(option_side="sell", option_leverage=5))
    await ex.open_option_by_premium(SignalDir.LONG.value, 900.0)
    ex._rest.set_leverage.assert_called_once()
