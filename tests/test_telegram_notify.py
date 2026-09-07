"""TelegramNotifier._format(): the leverage line on an options ENTRY message
is opt-in per-call (only shown when the caller passes leverage_ok, i.e. only
when OptionsExecutor actually attempted DELTA_OPTION_LEVERAGE) -- every
existing bot that never passes it must render byte-for-byte unchanged."""

from __future__ import annotations

from deltabot.enums import NotifyEvent
from deltabot.notify.telegram import _format


def _entry_ctx(**kw) -> dict:
    base = dict(direction="CALL", contract="C-BTC-76000-050926", premium=1400.0, btc_price=79000.0,
                sl_level=79500.0, tp_price=980.0, side="sell")
    base.update(kw)
    return base


def test_entry_message_has_no_leverage_line_when_not_configured() -> None:
    """leverage_ok absent entirely (every existing bot's call site) -- must
    match pre-leverage-notify behavior exactly, no stray line/None text."""
    msg = _format(NotifyEvent.ENTRY_SHORT, _entry_ctx())
    assert "Leverage" not in msg


def test_entry_message_has_no_leverage_line_when_explicitly_none() -> None:
    """leverage_ok=None (option_leverage<=0 or buy-side -- never attempted)."""
    msg = _format(NotifyEvent.ENTRY_SHORT, _entry_ctx(leverage=0, leverage_ok=None))
    assert "Leverage" not in msg


def test_entry_message_shows_leverage_applied() -> None:
    msg = _format(NotifyEvent.ENTRY_SHORT, _entry_ctx(leverage=50, leverage_ok=True))
    assert "Leverage: 50x" in msg
    assert "✅" in msg


def test_entry_message_flags_a_failed_leverage_call() -> None:
    msg = _format(NotifyEvent.ENTRY_SHORT, _entry_ctx(leverage=50, leverage_ok=False))
    assert "Leverage: 50x" in msg
    assert "failed" in msg.lower()
