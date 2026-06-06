"""Order execution engine.

Executes a reversal (close-then-open) as a single serialized sequence guarded by
an ``asyncio.Lock`` so two signals can never interleave into duplicate orders.
Each leg is placed as a market order and confirmed by polling the live position
via REST. The synchronous :class:`RestClient` calls are offloaded to threads.
"""

from __future__ import annotations

import asyncio

from ..config import Settings
from ..enums import ActionKind, PositionState, Side
from ..exchange.rest_client import DeltaRestError, RestClient
from ..logging_setup import get_logger
from ..models import Action, OrderResult, Position

log = get_logger(__name__)


class OrderExecutionError(RuntimeError):
    pass


class OrderEngine:
    def __init__(self, rest: RestClient, settings: Settings) -> None:
        self.rest = rest
        self.settings = settings
        self.product_id = settings.product_id  # set by trader after resolution
        self._lock = asyncio.Lock()

    def set_product_id(self, product_id: int) -> None:
        self.product_id = product_id

    async def execute_plan(self, actions: list[Action], current_pos: Position) -> list[OrderResult]:
        """Run an ordered list of CLOSE/OPEN actions atomically (under the lock)."""
        if not actions:
            return []
        results: list[OrderResult] = []
        async with self._lock:
            for action in actions:
                if action.kind == ActionKind.CLOSE:
                    size = current_pos.abs_size or self.settings.contracts
                    res = await self._place(action.side, size, reduce_only=True)
                    await self._confirm_flat()
                else:  # OPEN
                    res = await self._place(action.side, self.settings.contracts, reduce_only=False)
                    expected = PositionState.LONG if action.side == Side.BUY else PositionState.SHORT
                    await self._confirm_state(expected)
                results.append(res)
        return results

    # ------------------------------------------------------------------ #
    async def _place(self, side: Side, size: int, reduce_only: bool) -> OrderResult:
        log.info(
            "Placing market order",
            extra={"extra": {"side": side.value, "size": size, "reduce_only": reduce_only}},
        )
        try:
            return await asyncio.to_thread(
                self.rest.place_market_order, self.product_id, size, side, reduce_only
            )
        except DeltaRestError as exc:
            raise OrderExecutionError(f"order failed: {exc}") from exc

    async def _confirm_flat(self) -> None:
        await self._poll_until(lambda pos: pos.size == 0, "flat")

    async def _confirm_state(self, expected: PositionState) -> None:
        await self._poll_until(lambda pos: pos.state == expected, expected.value)

    async def _poll_until(self, predicate, label: str) -> Position:
        deadline = asyncio.get_running_loop().time() + self.settings.fill_confirm_timeout_s
        last: Position | None = None
        while asyncio.get_running_loop().time() < deadline:
            last = await asyncio.to_thread(self.rest.get_position, self.product_id)
            if predicate(last):
                return last
            await asyncio.sleep(0.5)
        raise OrderExecutionError(
            f"fill confirmation timed out (wanted {label}, got size="
            f"{last.size if last else 'unknown'})"
        )

    async def current_position(self) -> Position:
        return await asyncio.to_thread(self.rest.get_position, self.product_id)
