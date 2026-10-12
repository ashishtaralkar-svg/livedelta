"""Lightweight JSON state file — persists the currently-open option position across
container restarts so each bot only reconciles its OWN position.

Each bot is configured with a unique ``DELTA_STATE_FILE`` path (e.g.
``state/pine_pos.json`` vs ``state/revbreak_pos.json``). On startup reconcile
the bot loads this file and adopts only the matching symbol; any other open
short position is ignored (it belongs to the other bot).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger

log = get_logger(__name__)


def save(path: str, **fields: Any) -> bool:
    """Write ``fields`` to the state file at ``path`` (creates parent dirs).
    Returns False if the write failed (e.g. the mounted dir is not writable)."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(fields))
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to write position state file", extra={"extra": {"path": path, "error": str(exc)}})
        return False


def check_writable(path: str) -> str | None:
    """Probe that the state file's directory accepts writes. Returns an error
    string if not (a root-owned bind mount after a fresh clone is the usual cause)."""
    try:
        d = Path(path).parent
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f".{Path(path).name}.probe"
        probe.write_text("ok")
        probe.unlink()
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def load(path: str) -> dict | None:
    """Return the saved state dict, or ``None`` if the file is missing/corrupt."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def clear(path: str) -> None:
    """Delete the state file (position closed)."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to clear position state file", extra={"extra": {"path": path, "error": str(exc)}})
