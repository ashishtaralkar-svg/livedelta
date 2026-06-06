"""Resilient Delta Exchange REST client.

Implemented directly on ``httpx`` with our own HMAC signing (see ``signer.py``)
so behaviour is fully under our control and unit-testable, rather than coupling
to a specific ``delta-rest-client`` version. The endpoints and the signing
scheme match the official API and SDK.
"""

from __future__ import annotations

import json
import time
from datetime import date

import httpx

from .. import __version__
from ..enums import OptionType, OrderType, Side
from ..logging_setup import get_logger
from ..models import Candle, OrderResult, Position
from . import signer

log = get_logger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class DeltaRestError(RuntimeError):
    """Raised when the API returns an error that is not worth retrying."""


class RestClient:
    """Synchronous REST client. Methods are blocking and intended to be run via
    ``asyncio.to_thread`` from the async engine, keeping signing/timing simple
    (the 5-second signature window is respected by signing per attempt)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        timeout: float = 15.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.max_retries = max_retries
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._user_agent = f"deltabot-v{__version__}"

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ #
    # Core request with signing + retry/backoff
    # ------------------------------------------------------------------ #
    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        body: dict | None = None,
        signed: bool = False,
    ) -> dict:
        query_string = ""
        if params:
            # httpx-style encoding; include leading '?' in the signed string.
            query_string = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        body_str = json.dumps(body, separators=(",", ":")) if body is not None else ""

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            headers = {"User-Agent": self._user_agent, "Content-Type": "application/json"}
            if signed:
                ts = signer.epoch_seconds()
                sig = signer.rest_signature(
                    self.api_secret, method, ts, path, query_string, body_str
                )
                headers.update({"api-key": self.api_key, "timestamp": ts, "signature": sig})
            try:
                resp = self._client.request(
                    method,
                    path,
                    params=params,
                    content=body_str if body_str else None,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                self._backoff(attempt, reason=str(exc))
                continue

            if resp.status_code in _RETRYABLE_STATUS:
                last_exc = DeltaRestError(f"{resp.status_code}: {resp.text[:200]}")
                self._backoff(attempt, reason=f"status {resp.status_code}")
                continue

            if resp.status_code >= 400:
                raise DeltaRestError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            if isinstance(data, dict) and data.get("success") is False:
                raise DeltaRestError(f"{method} {path} -> {data}")
            return data

        raise DeltaRestError(f"{method} {path} failed after {self.max_retries} attempts: {last_exc}")

    def _backoff(self, attempt: int, reason: str) -> None:
        delay = min(2 ** (attempt - 1), 30)
        log.warning(
            "REST retry backoff",
            extra={"extra": {"attempt": attempt, "delay_s": delay, "reason": reason}},
        )
        time.sleep(delay)

    # ------------------------------------------------------------------ #
    # Public endpoints
    # ------------------------------------------------------------------ #
    def resolve_product_id(self, symbol: str) -> int:
        """Look up the numeric product id for a symbol (e.g. BTCUSD -> 27)."""
        data = self._request("GET", f"/v2/tickers/{symbol}")
        result = data.get("result") or {}
        pid = result.get("product_id") or result.get("id")
        if pid is None:
            raise DeltaRestError(f"Could not resolve product_id for {symbol}: {data}")
        return int(pid)

    def resolve_option_product_id(
        self,
        underlying: str,
        expiry: date,
        strike: int,
        option_type: OptionType,
    ) -> int:
        """Look up the product_id for a specific options contract on Delta Exchange.

        Delta symbol format: ``{C|P}-{UNDERLYING}-{STRIKE}-{ddmmyy}``
        Example: ``P-BTC-100400-060625`` (a 100400-strike BTC put expiring 6 Jun 2025).
        The option-type prefix comes FIRST and the expiry is numeric ``ddmmyy``.
        """
        symbol = f"{option_type.value}-{underlying}-{strike}-{expiry.strftime('%d%m%y')}"
        log.debug("Resolving option symbol", extra={"extra": {"symbol": symbol}})
        return self.resolve_product_id(symbol)

    def get_option_chain(
        self, underlying: str, expiry: date, option_type: OptionType
    ) -> list[dict]:
        """Return the LIVE option chain for ``(underlying, expiry, option_type)``.

        Queries ``GET /v2/tickers`` filtered by contract type, underlying and expiry
        so we only ever see strikes Delta is actually listing (which avoids hitting
        an unlisted strike). The ``expiry_date`` query param uses ``DD-MM-YYYY``
        (distinct from the ``ddmmyy`` embedded in the contract symbol).

        Each entry: ``{product_id, symbol, strike, mark_price}``.
        """
        contract_type = "call_options" if option_type == OptionType.CALL else "put_options"
        data = self._request(
            "GET",
            "/v2/tickers",
            params={
                "contract_types": contract_type,
                "underlying_asset_symbols": underlying,
                "expiry_date": expiry.strftime("%d-%m-%Y"),
            },
        )
        result = data.get("result") or []
        out: list[dict] = []
        for r in result:
            strike = r.get("strike_price")
            pid = r.get("product_id") or r.get("id")
            if strike in (None, "") or pid is None:
                continue
            mark = r.get("mark_price")
            out.append(
                {
                    "product_id": int(pid),
                    "symbol": str(r.get("symbol") or ""),
                    "strike": float(strike),
                    "mark_price": float(mark) if mark not in (None, "") else None,
                }
            )
        return out

    def get_candles(self, symbol: str, resolution: str, start: int, end: int) -> list[Candle]:
        data = self._request(
            "GET",
            "/v2/history/candles",
            params={"symbol": symbol, "resolution": resolution, "start": start, "end": end},
        )
        rows = data.get("result") or []
        candles = [Candle.from_rest(r) for r in rows]
        candles.sort(key=lambda c: c.start_time)
        return candles

    # ------------------------------------------------------------------ #
    # Authenticated endpoints
    # ------------------------------------------------------------------ #
    def set_leverage(self, product_id: int, leverage: int) -> dict:
        return self._request(
            "POST",
            f"/v2/products/{product_id}/orders/leverage",
            body={"leverage": str(leverage)},
            signed=True,
        )

    def get_position(self, product_id: int) -> Position:
        data = self._request(
            "GET", "/v2/positions", params={"product_id": product_id}, signed=True
        )
        result = data.get("result")
        # /v2/positions returns a single object when product_id is given (may be null).
        if not result:
            return Position(product_id=product_id, size=0)
        if isinstance(result, list):
            result = next((r for r in result if int(r.get("product_id", 0)) == product_id), None)
            if not result:
                return Position(product_id=product_id, size=0)
        size = int(result.get("size") or 0)
        entry = result.get("entry_price")
        return Position(
            product_id=product_id,
            size=size,
            entry_price=float(entry) if entry not in (None, "") else None,
        )

    def get_available_balance(self, asset_symbol: str | None = None) -> float:
        """Return available wallet balance for margin checks.

        If ``asset_symbol`` is given, return that asset's available balance;
        otherwise return the maximum available balance across all wallets.
        """
        data = self._request("GET", "/v2/wallet/balances", signed=True)
        result = data.get("result") or []
        balances: list[tuple[str, float]] = []
        for r in result:
            avail = r.get("available_balance")
            if avail in (None, ""):
                continue
            sym = r.get("asset_symbol") or (r.get("asset") or {}).get("symbol") or ""
            balances.append((str(sym), float(avail)))
        if not balances:
            return 0.0
        if asset_symbol:
            for sym, val in balances:
                if sym == asset_symbol:
                    return val
            return 0.0
        return max(val for _, val in balances)

    def get_option_positions(self, underlying: str) -> list[dict]:
        """Return open option positions (size != 0) for ``underlying``.

        Uses ``GET /v2/positions/margined`` (all margined positions in one call)
        and filters to option contracts — whose symbols are prefixed ``C-``/``P-``
        — for the given underlying. Each entry is normalised to a dict with
        ``product_id``, ``size`` (signed; <0 short), ``symbol`` and ``entry_price``.
        """
        data = self._request("GET", "/v2/positions/margined", signed=True)
        result = data.get("result") or []
        out: list[dict] = []
        for r in result:
            size = int(r.get("size") or 0)
            if size == 0:
                continue
            sym = str(r.get("product_symbol") or (r.get("product") or {}).get("symbol") or "")
            if not (sym.startswith("C-") or sym.startswith("P-")):
                continue
            if f"-{underlying}-" not in sym:
                continue
            entry = r.get("entry_price")
            out.append(
                {
                    "product_id": int(r.get("product_id") or (r.get("product") or {}).get("id") or 0),
                    "size": size,
                    "symbol": sym,
                    "entry_price": float(entry) if entry not in (None, "") else None,
                }
            )
        return out

    def place_market_order(
        self, product_id: int, size: int, side: Side, reduce_only: bool = False
    ) -> OrderResult:
        body: dict = {
            "product_id": product_id,
            "size": int(size),
            "side": side.value,
            "order_type": OrderType.MARKET.value,
        }
        if reduce_only:
            body["reduce_only"] = True
        data = self._request("POST", "/v2/orders", body=body, signed=True)
        result = data.get("result") or {}
        avg = result.get("average_fill_price")
        return OrderResult(
            order_id=result.get("id"),
            product_id=product_id,
            side=side,
            size=int(size),
            state=result.get("state"),
            average_fill_price=float(avg) if avg not in (None, "") else None,
            raw=result,
        )

    def cancel_all_orders(self, product_id: int) -> dict:
        return self._request(
            "DELETE", "/v2/orders/all", body={"product_id": product_id}, signed=True
        )
