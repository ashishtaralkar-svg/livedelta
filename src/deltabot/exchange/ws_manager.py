"""Delta Exchange WebSocket manager: connect, (optional) auth, subscribe to the
1-minute candlestick channel, enforce a heartbeat watchdog, and auto-reconnect
with exponential backoff. On every (re)connection an ``on_reconnect`` callback
fires so the engine can re-sync state from the REST API (positions = truth).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import websockets

from ..logging_setup import get_logger
from . import signer

log = get_logger(__name__)

CandleCallback = Callable[[dict], None]
ReconnectCallback = Callable[[], Awaitable[None]]


class WebSocketManager:
    def __init__(
        self,
        ws_url: str,
        symbol: str,
        resolution: str = "1m",
        api_key: str | None = None,
        api_secret: str | None = None,
        on_candle: CandleCallback | None = None,
        on_reconnect: ReconnectCallback | None = None,
        heartbeat_timeout_s: float = 35.0,
    ) -> None:
        self.ws_url = ws_url
        self.symbol = symbol
        self.channel = f"candlestick_{resolution}"
        self.api_key = api_key
        self.api_secret = api_secret
        self.on_candle = on_candle
        self.on_reconnect = on_reconnect
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self._stop = asyncio.Event()
        self._last_msg_at = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """Connect/serve loop that never exits until :meth:`stop` is called."""
        self._loop = asyncio.get_running_loop()
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._connect_once()
                backoff = 1.0  # reset after a clean session
            except Exception as exc:  # noqa: BLE001 — reconnect on any failure
                log.warning("WebSocket session ended", extra={"extra": {"error": str(exc)}})
            if self._stop.is_set():
                break
            delay = min(backoff, 30.0)
            log.info("Reconnecting WebSocket", extra={"extra": {"delay_s": delay}})
            await asyncio.sleep(delay)
            backoff *= 2

    async def _connect_once(self) -> None:
        async with websockets.connect(
            self.ws_url, ping_interval=20, ping_timeout=10, close_timeout=5
        ) as ws:
            self._last_msg_at = self._now()
            if self.api_key and self.api_secret:
                await self._authenticate(ws)
            await self._subscribe(ws)
            await self._send(ws, {"type": "enable_heartbeat"})

            if self.on_reconnect is not None:
                # Re-sync state from REST before trusting the new stream.
                await self.on_reconnect()

            watchdog = asyncio.create_task(self._watchdog(ws))
            try:
                async for raw in ws:
                    self._last_msg_at = self._now()
                    self._handle(raw)
                    if self._stop.is_set():
                        break
            finally:
                watchdog.cancel()

    async def _authenticate(self, ws) -> None:
        ts = signer.epoch_seconds()
        sig = signer.ws_auth_signature(self.api_secret, ts)
        await self._send(
            ws,
            {"type": "auth", "payload": {"api-key": self.api_key, "signature": sig, "timestamp": ts}},
        )
        log.info("WebSocket auth sent")

    async def _subscribe(self, ws) -> None:
        await self._send(
            ws,
            {
                "type": "subscribe",
                "payload": {"channels": [{"name": self.channel, "symbols": [self.symbol]}]},
            },
        )
        log.info(
            "WebSocket subscribed",
            extra={"extra": {"channel": self.channel, "symbol": self.symbol}},
        )

    async def _watchdog(self, ws) -> None:
        """Force a reconnect if no message arrives within the heartbeat timeout."""
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self.heartbeat_timeout_s / 2)
                if self._now() - self._last_msg_at > self.heartbeat_timeout_s:
                    log.warning("Heartbeat timeout — closing socket to reconnect")
                    await ws.close()
                    return
        except asyncio.CancelledError:
            pass

    def _handle(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        mtype = msg.get("type")
        if mtype in ("heartbeat", "subscriptions", "success", "auth"):
            return
        if mtype == self.channel or "candle_start_time" in msg:
            if self.on_candle is not None:
                self.on_candle(msg)

    async def _send(self, ws, payload: dict) -> None:
        await ws.send(json.dumps(payload))

    def _now(self) -> float:
        # Monotonic-ish wall clock; loop.time() is monotonic and async-safe.
        return self._loop.time() if self._loop else 0.0
