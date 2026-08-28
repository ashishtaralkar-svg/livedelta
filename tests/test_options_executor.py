"""OptionsExecutor: side-aware (sell/buy) direction mapping, order side on open
and close, and that default (sell) behavior is byte-for-byte unchanged."""

from __future__ import annotations

import time
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


# --------------------------------------------------------------------- #
# open_option_by_balance_fraction: lot count computed from a FRACTION of
# available balance instead of the static settings.option_contracts.
# mark_price=900.0 (from _fake_rest's default), BTC lot_size=0.001 ->
# cost_per_lot = 0.9. balance=100.0, fraction=0.10 -> target=$10 ->
# lots = int(10.0 // 0.9) = 11.
# --------------------------------------------------------------------- #
def _fake_rest_with_balance(fill_price=900.0, balance=100.0):
    rest = _fake_rest(fill_price)
    rest.get_available_balance = MagicMock(return_value=balance)
    return rest


async def test_balance_fraction_computes_lots_from_real_mark_price_not_settings() -> None:
    ex = OptionsExecutor(_fake_rest_with_balance(fill_price=900.0, balance=100.0),
                         _settings(option_side="buy", option_contracts=25))
    fill, symbol, lots = await ex.open_option_by_balance_fraction(
        SignalDir.LONG.value, 500.0, 0.10, None)
    assert lots == 11   # NOT settings.option_contracts (25)
    assert symbol == "C-BTC-64000-180726"
    assert ex.tracked_size == 11
    order_size = ex._rest.place_market_order.call_args.args[1]
    assert order_size == 11


async def test_balance_fraction_zero_lots_places_no_order_and_stays_flat() -> None:
    """Balance too small for even 1 lot -- must skip the trade entirely, not
    force a minimum size."""
    ex = OptionsExecutor(_fake_rest_with_balance(fill_price=900.0, balance=0.05),
                         _settings(option_side="buy"))
    fill, symbol, lots = await ex.open_option_by_balance_fraction(
        SignalDir.LONG.value, 500.0, 0.10, None)
    assert (fill, symbol, lots) == (None, None, 0)
    ex._rest.place_market_order.assert_not_called()
    assert not ex.has_open_position


async def test_balance_fraction_passes_margin_asset_through() -> None:
    ex = OptionsExecutor(_fake_rest_with_balance(), _settings(option_side="buy"))
    await ex.open_option_by_balance_fraction(SignalDir.LONG.value, 500.0, 0.10, "USDT")
    ex._rest.get_available_balance.assert_called_once_with("USDT")


async def test_balance_fraction_guarded_when_already_open() -> None:
    ex = OptionsExecutor(_fake_rest_with_balance(), _settings(option_side="buy"))
    await ex.open_option_by_balance_fraction(SignalDir.LONG.value, 500.0, 0.10, None)
    ex._rest.place_market_order.reset_mock()
    fill, symbol, lots = await ex.open_option_by_balance_fraction(
        SignalDir.LONG.value, 500.0, 0.10, None)
    assert (fill, symbol, lots) == (None, None, 0)
    ex._rest.place_market_order.assert_not_called()


# --------------------------------------------------------------------- #
# select_by_trade_price / open_option_by_trade_price: prefer each nearby
# candidate's recent HISTORICAL TRADE-PRICE candle (matching backtest's own
# data source) over mark_price, falling back to mark_price when stale.
# --------------------------------------------------------------------- #
def _fake_rest_multi_strike(candles_by_symbol: dict[str, list] | None = None,
                            candles_side_effect=None):
    chain = [
        {"symbol": "C-BTC-64000-180726", "strike": 64000, "product_id": 111, "mark_price": 100.0},
        {"symbol": "C-BTC-64200-180726", "strike": 64200, "product_id": 112, "mark_price": 105.0},
    ]
    rest = MagicMock()
    rest.get_option_chain = MagicMock(return_value=chain)
    rest.place_market_order = MagicMock(return_value=OrderResult(
        order_id=1, product_id=111, side=Side.BUY, size=25, average_fill_price=100.0,
    ))
    if candles_side_effect is not None:
        rest.get_candles = MagicMock(side_effect=candles_side_effect)
    else:
        rest.get_candles = MagicMock(side_effect=lambda sym, res, start, end:
                                     (candles_by_symbol or {}).get(sym, []))
    return rest


def _candle(ts, close):
    from deltabot.models import Candle
    return Candle(start_time=ts, open=close, high=close, low=close, close=close, volume=1.0)


async def test_trade_price_overrides_mark_price_ranking() -> None:
    """mark_price alone would pick strike 64000 (|100-102|=2 < |105-102|=3),
    but its recent trade price is actually further from target -- trade
    price must win the ranking."""
    now = int(time.time())
    candles = {
        "C-BTC-64000-180726": [_candle(now - 10, 90.0)],    # trade price 90 -- |90-102|=12
        "C-BTC-64200-180726": [_candle(now - 10, 103.0)],   # trade price 103 -- |103-102|=1, closer
    }
    ex = OptionsExecutor(_fake_rest_multi_strike(candles), _settings(option_side="buy"))
    best = await ex.select_by_trade_price(SignalDir.LONG.value, 102.0)
    assert best["symbol"] == "C-BTC-64200-180726"


async def test_trade_price_falls_back_to_mark_price_when_candle_missing() -> None:
    now = int(time.time())
    candles = {
        "C-BTC-64000-180726": [],   # no recent trade -- falls back to mark_price=100.0
        "C-BTC-64200-180726": [_candle(now - 10, 50.0)],   # far from target
    }
    ex = OptionsExecutor(_fake_rest_multi_strike(candles), _settings(option_side="buy"))
    best = await ex.select_by_trade_price(SignalDir.LONG.value, 102.0)
    assert best["symbol"] == "C-BTC-64000-180726"   # mark_price(100) closer to 102 than trade(50)


async def test_trade_price_falls_back_to_mark_price_when_candle_stale() -> None:
    now = int(time.time())
    candles = {
        # candle is 300s old, beyond the default 120s max_age -- stale, falls back to mark_price
        "C-BTC-64000-180726": [_candle(now - 300, 200.0)],
        "C-BTC-64200-180726": [_candle(now - 10, 999.0)],
    }
    ex = OptionsExecutor(_fake_rest_multi_strike(candles), _settings(option_side="buy"))
    best = await ex.select_by_trade_price(SignalDir.LONG.value, 102.0)
    assert best["symbol"] == "C-BTC-64000-180726"   # falls back to its mark_price(100), closest


async def test_trade_price_falls_back_to_mark_price_on_fetch_error() -> None:
    def side_effect(sym, res, start, end):
        if sym == "C-BTC-64000-180726":
            raise RuntimeError("boom")
        return [_candle(int(time.time()) - 10, 999.0)]
    ex = OptionsExecutor(_fake_rest_multi_strike(candles_side_effect=side_effect), _settings(option_side="buy"))
    best = await ex.select_by_trade_price(SignalDir.LONG.value, 102.0)
    assert best["symbol"] == "C-BTC-64000-180726"   # error -> falls back to mark_price(100), closest


async def test_open_by_trade_price_uses_static_lot_count() -> None:
    now = int(time.time())
    candles = {
        "C-BTC-64000-180726": [_candle(now - 10, 100.0)],
        "C-BTC-64200-180726": [_candle(now - 10, 999.0)],
    }
    ex = OptionsExecutor(_fake_rest_multi_strike(candles), _settings(option_side="buy", option_contracts=25))
    fill, symbol = await ex.open_option_by_trade_price(SignalDir.LONG.value, 102.0)
    assert symbol == "C-BTC-64000-180726"
    assert ex.tracked_size == 25   # static, unaffected by trade-price mode
    order_size = ex._rest.place_market_order.call_args.args[1]
    assert order_size == 25


async def test_open_by_trade_price_guarded_when_already_open() -> None:
    now = int(time.time())
    candles = {"C-BTC-64000-180726": [_candle(now - 10, 100.0)],
               "C-BTC-64200-180726": [_candle(now - 10, 999.0)]}
    ex = OptionsExecutor(_fake_rest_multi_strike(candles), _settings(option_side="buy"))
    await ex.open_option_by_trade_price(SignalDir.LONG.value, 102.0)
    ex._rest.place_market_order.reset_mock()
    fill, symbol = await ex.open_option_by_trade_price(SignalDir.LONG.value, 102.0)
    assert (fill, symbol) == (None, None)
    ex._rest.place_market_order.assert_not_called()
