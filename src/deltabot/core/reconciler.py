"""Reconcile the bot's in-memory state with the exchange's actual position.

``GET /v2/positions`` is the single source of truth. We reconcile on startup and
after every WebSocket reconnect. Reconciliation NEVER places an order — it only
reads the exchange and aligns the in-memory state machine, preventing blind
double-fires after a reconnect.
"""

from __future__ import annotations

import asyncio

from ..exchange.rest_client import RestClient
from ..logging_setup import get_logger
from .state_machine import PositionStateMachine

log = get_logger(__name__)


async def fetch_state(rest: RestClient, product_id: int):
    return await asyncio.to_thread(rest.get_position, product_id)


async def reconcile(rest: RestClient, product_id: int, sm: PositionStateMachine, *, context: str) -> None:
    """Read the live position and set the state machine to match it."""
    pos = await fetch_state(rest, product_id)
    previous = sm.state
    sm.set_state(pos.state)
    log.info(
        "Reconciled state with exchange",
        extra={
            "extra": {
                "context": context,
                "previous": previous.value,
                "exchange": pos.state.value,
                "size": pos.size,
            }
        },
    )
