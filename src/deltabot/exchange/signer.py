"""HMAC-SHA256 request signing for the Delta Exchange API.

REST signing prehash (exact concatenation order, per Delta docs / official SDK):

    signature_data = method + timestamp + requestPath + query_string + body

WebSocket private-channel auth signs the fixed prehash ``'GET' + timestamp + '/live'``.

The signature is valid for 5 seconds, so callers should sign immediately before
sending the request.
"""

from __future__ import annotations

import hashlib
import hmac
import time


def epoch_seconds() -> str:
    """Current Unix time in seconds as a string (the value sent in headers)."""
    return str(int(time.time()))


def _sign(secret: str, message: str) -> str:
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def rest_signature(
    secret: str,
    method: str,
    timestamp: str,
    path: str,
    query_string: str = "",
    body: str = "",
) -> str:
    """Sign a REST request.

    ``method`` is the uppercase HTTP verb. ``query_string`` must include the
    leading ``?`` when present, else be empty. ``body`` is the exact JSON string
    sent (empty for GET).
    """
    message = method.upper() + timestamp + path + query_string + body
    return _sign(secret, message)


def ws_auth_signature(secret: str, timestamp: str) -> str:
    """Sign the WebSocket private-channel auth payload."""
    message = "GET" + timestamp + "/live"
    return _sign(secret, message)
