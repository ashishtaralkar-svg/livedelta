"""Enumerations shared across the bot."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Side(StrEnum):
    """Order side as understood by the Delta REST API."""

    BUY = "buy"
    SELL = "sell"


class PositionState(StrEnum):
    """The bot's view of the current net position."""

    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(StrEnum):
    """Delta order types."""

    MARKET = "market_order"
    LIMIT = "limit_order"


class SignalDir(IntEnum):
    """Supertrend direction. +1 == bullish/long, -1 == bearish/short."""

    LONG = 1
    SHORT = -1


class ActionKind(StrEnum):
    """A single primitive action emitted by the position state machine."""

    CLOSE = "CLOSE"
    OPEN = "OPEN"


class OptionType(StrEnum):
    """BTC option contract type."""

    CALL = "C"
    PUT = "P"


class NotifyEvent(StrEnum):
    """Telegram notification event types."""

    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT = "EXIT"
    SKIPPED = "SKIPPED"
    PAPER_ENTRY = "PAPER_ENTRY"
    PAPER_EXIT = "PAPER_EXIT"
    REVERSAL = "REVERSAL"
    API_ERROR = "API_ERROR"
    RESTART = "RESTART"
    DAILY_PNL = "DAILY_PNL"
