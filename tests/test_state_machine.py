"""Truth-table tests for the position state machine."""

from __future__ import annotations

from deltabot.core.state_machine import PositionStateMachine, decide, plan_actions
from deltabot.enums import ActionKind, PositionState, Side, SignalDir


def test_flat_long_opens_buy():
    actions = decide(PositionState.FLAT, SignalDir.LONG.value, 1)
    assert len(actions) == 1
    assert actions[0].kind == ActionKind.OPEN and actions[0].side == Side.BUY


def test_flat_short_opens_sell():
    actions = decide(PositionState.FLAT, SignalDir.SHORT.value, 1)
    assert len(actions) == 1
    assert actions[0].kind == ActionKind.OPEN and actions[0].side == Side.SELL


def test_long_to_short_reverses():
    actions = decide(PositionState.LONG, SignalDir.SHORT.value, 1)
    assert [a.kind for a in actions] == [ActionKind.CLOSE, ActionKind.OPEN]
    assert actions[0].side == Side.SELL and actions[0].reduce_only is True
    assert actions[1].side == Side.SELL and actions[1].reduce_only is False


def test_short_to_long_reverses():
    actions = decide(PositionState.SHORT, SignalDir.LONG.value, 1)
    assert [a.kind for a in actions] == [ActionKind.CLOSE, ActionKind.OPEN]
    assert actions[0].side == Side.BUY and actions[0].reduce_only is True
    assert actions[1].side == Side.BUY and actions[1].reduce_only is False


def test_aligned_signals_are_noops():
    assert decide(PositionState.LONG, SignalDir.LONG.value, 1) == []
    assert decide(PositionState.SHORT, SignalDir.SHORT.value, 1) == []


def test_idempotent_repeat_signal():
    sm = PositionStateMachine(contracts=1)
    assert len(sm.decide(SignalDir.LONG.value)) == 1  # FLAT -> open
    sm.apply_signal(SignalDir.LONG.value)
    assert sm.state == PositionState.LONG
    assert sm.decide(SignalDir.LONG.value) == []  # repeat -> no duplicate order


def test_invalid_contracts_raises():
    import pytest

    with pytest.raises(ValueError):
        decide(PositionState.FLAT, SignalDir.LONG.value, 0)


# --------------------------------------------------------------------------- #
# plan_actions (Pine strategy decision -> orders)
# --------------------------------------------------------------------------- #
def test_plan_actions_pure_entry():
    actions = plan_actions(long_exit=False, short_exit=False, buy_signal=True, sell_signal=False)
    assert [(a.kind, a.side, a.reduce_only) for a in actions] == [
        (ActionKind.OPEN, Side.BUY, False)
    ]


def test_plan_actions_pure_exit():
    actions = plan_actions(long_exit=True, short_exit=False, buy_signal=False, sell_signal=False)
    assert [(a.kind, a.side, a.reduce_only) for a in actions] == [
        (ActionKind.CLOSE, Side.SELL, True)
    ]


def test_plan_actions_same_bar_reversal_closes_before_opening():
    # Short exits (SL) and a fresh BUY fires on the same bar -> close then open.
    actions = plan_actions(long_exit=False, short_exit=True, buy_signal=True, sell_signal=False)
    assert [(a.kind, a.side) for a in actions] == [
        (ActionKind.CLOSE, Side.BUY),
        (ActionKind.OPEN, Side.BUY),
    ]
    assert actions[0].reduce_only is True


def test_plan_actions_noop_when_nothing_fires():
    assert plan_actions(long_exit=False, short_exit=False, buy_signal=False, sell_signal=False) == []
