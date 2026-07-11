"""OptionsExecutor.option_leverage: set the product leverage before selling,
best-effort (never blocks the trade), and skip entirely when unset.
"""

from __future__ import annotations

from types import SimpleNamespace

from deltabot.config import Settings
from deltabot.core.options_executor import OptionsExecutor
from deltabot.enums import SignalDir


class FakeRest:
    def __init__(self, leverage_raises: bool = False) -> None:
        self.leverage_calls: list[tuple[int, int]] = []
        self.orders: list[tuple] = []
        self._leverage_raises = leverage_raises

    # chain lookup used by select_by_premium
    def get_option_chain(self, underlying, expiry, option_type):
        return [{"symbol": "P-BTC-60000-070726", "strike": 60000,
                 "product_id": 777, "mark_price": 1000.0}]

    def set_leverage(self, product_id, leverage):
        if self._leverage_raises:
            raise RuntimeError("leverage not supported for options")
        self.leverage_calls.append((product_id, leverage))
        return {"ok": True}

    def place_market_order(self, product_id, size, side, reduce_only=False):
        self.orders.append((product_id, size, side, reduce_only))
        return SimpleNamespace(average_fill_price=1000.0, order_id="o1")

    def get_available_balance(self, asset=None):
        return 1e9


def _executor(rest, **settings_kwargs) -> OptionsExecutor:
    s = Settings(_env_file=None, options_mode=True, option_contracts=1, **settings_kwargs)
    return OptionsExecutor(rest, s)


async def test_leverage_set_before_sell_when_configured() -> None:
    rest = FakeRest()
    ex = _executor(rest, option_leverage=50)
    fill, symbol = await ex.open_option_by_premium(SignalDir.LONG.value, 1000.0)
    assert fill == 1000.0
    assert rest.leverage_calls == [(777, 50)]        # leverage set on the product...
    assert rest.orders and rest.orders[0][0] == 777  # ...before the SELL order


async def test_leverage_not_touched_when_zero() -> None:
    rest = FakeRest()
    ex = _executor(rest, option_leverage=0)
    await ex.open_option_by_premium(SignalDir.LONG.value, 1000.0)
    assert rest.leverage_calls == []  # never called
    assert rest.orders  # order still placed


async def test_leverage_failure_does_not_block_trade() -> None:
    rest = FakeRest(leverage_raises=True)
    ex = _executor(rest, option_leverage=50)
    fill, _ = await ex.open_option_by_premium(SignalDir.LONG.value, 1000.0)
    assert fill == 1000.0     # trade still went through
    assert rest.orders        # order placed despite set_leverage raising
