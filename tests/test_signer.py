"""Known-vector tests for HMAC-SHA256 request signing."""

from __future__ import annotations

import hashlib
import hmac

from deltabot.exchange.signer import rest_signature, ws_auth_signature


def _expected(secret: str, message: str) -> str:
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def test_rest_signature_get_no_body():
    secret = "test_secret"
    ts = "1700000000"
    sig = rest_signature(secret, "GET", ts, "/v2/positions", "?product_ids=27", "")
    assert sig == _expected(secret, "GET" + ts + "/v2/positions" + "?product_ids=27")


def test_rest_signature_post_with_body():
    secret = "test_secret"
    ts = "1700000001"
    body = '{"product_id": 27, "size": 1, "side": "buy", "order_type": "market_order"}'
    sig = rest_signature(secret, "POST", ts, "/v2/orders", "", body)
    assert sig == _expected(secret, "POST" + ts + "/v2/orders" + body)


def test_rest_method_uppercased():
    secret = "s"
    ts = "1"
    lower = rest_signature(secret, "post", ts, "/v2/orders", "", "")
    upper = rest_signature(secret, "POST", ts, "/v2/orders", "", "")
    assert lower == upper


def test_ws_auth_signature_fixed_prehash():
    secret = "test_secret"
    ts = "1700000002"
    sig = ws_auth_signature(secret, ts)
    assert sig == _expected(secret, "GET" + ts + "/live")


def test_signature_is_deterministic():
    assert rest_signature("k", "GET", "1", "/p", "", "") == rest_signature("k", "GET", "1", "/p", "", "")
