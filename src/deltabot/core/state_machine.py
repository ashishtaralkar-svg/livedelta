"""Pure position state machine and reversal decision logic.

This module contains NO exchange/networking code so it can be reused identically
by the live engine and the backtester. Given the current position state and a
desired Supertrend direction, :meth:`decide` returns the ordered list of
primitive actions (close then open) needed to reach the target — or an empty
list when already aligned (idempotent no-op).
"""

from __future__ import annotations

from ..enums import ActionKind, PositionState, Side, SignalDir
from ..models import Action


def decide(state: PositionState, signal_dir: int, contracts: int) -> list[Action]:
    """Return the actions to move from ``state`` to the side implied by ``signal_dir``.

    ``signal_dir``: +1 -> want LONG, -1 -> want SHORT.
    ``contracts``: size used when opening (closing uses the existing position size).

    Rules:
      * FLAT  + long  -> [open buy]
      * FLAT  + short -> [open sell]
      * LONG  + short -> [close (reduce_only sell), open sell]   (reversal)
      * SHORT + long  -> [close (reduce_only buy),  open buy]    (reversal)
      * already aligned -> [] (no duplicate orders)
    """
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    want_long = signal_dir == SignalDir.LONG.value

    if state == PositionState.FLAT:
        return [Action(ActionKind.OPEN, Side.BUY if want_long else Side.SELL)]

    if state == PositionState.LONG:
        if want_long:
            return []  # already long
        return [
            Action(ActionKind.CLOSE, Side.SELL, reduce_only=True),
            Action(ActionKind.OPEN, Side.SELL),
        ]

    # state == SHORT
    if not want_long:
        return []  # already short
    return [
        Action(ActionKind.CLOSE, Side.BUY, reduce_only=True),
        Action(ActionKind.OPEN, Side.BUY),
    ]


def plan_actions(
    *, long_exit: bool, short_exit: bool, buy_signal: bool, sell_signal: bool
) -> list[Action]:
    """Translate a :class:`~deltabot.strategy.pine_strategy.StrategyDecision` into an
    ordered list of primitive CLOSE/OPEN actions.

    Exits are emitted before entries so a same-bar reversal closes the old position
    first. At most one of (long_exit, short_exit) and one of (buy_signal, sell_signal)
    can be set, since the strategy holds a single position at a time.

      * long_exit  -> CLOSE (reduce-only SELL)   close a long
      * short_exit -> CLOSE (reduce-only BUY)    close a short
      * buy_signal  -> OPEN BUY
      * sell_signal -> OPEN SELL
    """
    actions: list[Action] = []
    if long_exit:
        actions.append(Action(ActionKind.CLOSE, Side.SELL, reduce_only=True))
    if short_exit:
        actions.append(Action(ActionKind.CLOSE, Side.BUY, reduce_only=True))
    if buy_signal:
        actions.append(Action(ActionKind.OPEN, Side.BUY))
    if sell_signal:
        actions.append(Action(ActionKind.OPEN, Side.SELL))
    return actions


class PositionStateMachine:
    """Tracks the bot's current position state and computes reversal plans."""

    def __init__(self, contracts: int, state: PositionState = PositionState.FLAT) -> None:
        self.contracts = contracts
        self.state = state

    def set_state(self, state: PositionState) -> None:
        """Set the state explicitly (used by reconciliation against the exchange)."""
        self.state = state

    def decide(self, signal_dir: int) -> list[Action]:
        """Plan actions for the given desired direction from the current state."""
        return decide(self.state, signal_dir, self.contracts)

    def apply_signal(self, signal_dir: int) -> PositionState:
        """Advance the in-memory state to reflect a completed reversal."""
        self.state = PositionState.LONG if signal_dir == SignalDir.LONG.value else PositionState.SHORT
        return self.state
