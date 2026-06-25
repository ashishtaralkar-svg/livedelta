"""Options execution engine — translates strategy signals into C/P long trades.

When OPTIONS_MODE is enabled the underlying BTC price and the existing
Supertrend/EMA/Prev-Day strategy are still used for signal generation.
Execution is replaced:

  BUY  signal  -> Buy ITM CALL (C)  (strike = round(btc_close - offset, interval))
  SELL signal  -> Buy ITM PUT  (P)  (strike = round(btc_close + offset, interval))
  Exit (SL/EOD)-> Sell (reduce-only) the tracked long option at market price

Delta Exchange uses single-letter C/P designators (NOT the NSE CE/PE convention).
The contract symbol is ``{C|P}-{UNDERLYING}-{STRIKE}-{ddmmyy}`` (e.g.
``P-BTC-100400-060625``); the construction lives in
:meth:`RestClient.resolve_option_product_id`.

Expiry: nearest daily expiry in IST. Delta options settle at 17:30 IST (12:00
UTC); if the current IST hour is at/after ``option_expiry_cutoff_hour`` the next
day's expiry is used so we never enter a contract that expires imminently.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from ..config import Settings
from ..enums import OptionType, Side, SignalDir
from ..exchange.rest_client import DeltaRestError, RestClient
from ..logging_setup import get_logger

log = get_logger(__name__)

_IST = ZoneInfo("Asia/Kolkata")


class OptionsMarginError(RuntimeError):
    """Raised when available balance is below the configured floor to sell options."""


class OptionsExecutor:
    """Manages entry and exit for a single short-option leg."""

    def __init__(self, rest: RestClient, settings: Settings) -> None:
        self._rest = rest
        self._settings = settings
        # Tracked open position (cleared on close / reconcile-flat).
        self._product_id: int | None = None
        self._size: int = 0
        self._option_type: OptionType | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def underlying(self) -> str:
        """Bare underlying asset token for option symbols (e.g. ``BTC``).

        Strips ``USDT`` before ``USD`` so ``BTCUSDT`` -> ``BTC`` (not ``BTCT``).
        """
        return self._settings.symbol.replace("USDT", "").replace("USD", "")

    @property
    def has_open_position(self) -> bool:
        return self._product_id is not None

    @property
    def tracked_product_id(self) -> int | None:
        return self._product_id

    def adopt(self, product_id: int, size: int, option_type: OptionType) -> None:
        """Adopt an existing short option position discovered during reconciliation."""
        self._product_id = product_id
        self._size = abs(size)
        self._option_type = option_type
        log.info(
            "Adopted existing option position",
            extra={"extra": {"product_id": product_id, "size": self._size, "type": option_type.value}},
        )

    def clear(self) -> None:
        """Forget any tracked position (used when the exchange shows flat)."""
        self._product_id = None
        self._size = 0
        self._option_type = None

    async def open_option(self, signal_dir: int, btc_price: float) -> float | None:
        """Open a long option position from the strategy signal direction.

        ``signal_dir``: ``SignalDir.LONG`` (+1) -> buy CALL; ``SignalDir.SHORT``
        (-1) -> buy PUT. ``btc_price``: close of the just-closed BTC candle used
        for the ATM reference. Returns the average fill price (or ``None``).

        State is recorded ONLY after the BUY is accepted, so a failure before the
        fill leaves us cleanly flat. Raises :class:`OptionsMarginError` if the
        balance pre-check fails, or :class:`DeltaRestError` if the contract cannot
        be resolved — both are handled by the caller without crashing the engine.
        """
        if self._product_id is not None:
            log.warning(
                "open_option called while a position is already tracked — skipping",
                extra={"extra": {"existing_product_id": self._product_id}},
            )
            return None

        option_type = OptionType.CALL if signal_dir == SignalDir.LONG else OptionType.PUT
        target_strike = self._calc_strike(btc_price, option_type)
        expiry = self._select_expiry()
        underlying = self.underlying

        log.info(
            "Options entry",
            extra={
                "extra": {
                    "signal": "BUY" if signal_dir == SignalDir.LONG else "SELL",
                    "option_type": option_type.value,
                    "target_strike": target_strike,
                    "expiry": expiry.isoformat(),
                    "underlying": underlying,
                }
            },
        )

        # Margin pre-check (best-effort balance floor; the exchange is the final authority).
        await self._check_balance()

        product_id, strike = await self._select_contract(
            underlying, expiry, target_strike, option_type
        )

        size = self._settings.option_contracts
        result = await asyncio.to_thread(
            self._rest.place_market_order, product_id, size, Side.BUY
        )

        # Record state only AFTER the BUY has been accepted by the exchange.
        self._product_id = product_id
        self._size = size
        self._option_type = option_type

        log.info(
            "Option BUY order placed",
            extra={
                "extra": {
                    "product_id": product_id,
                    "size": size,
                    "fill_price": result.average_fill_price,
                    "order_id": result.order_id,
                }
            },
        )
        return result.average_fill_price

    async def close_option(self) -> float | None:
        """Sell (reduce-only) the tracked long option to close the position.

        Returns the average fill price (or ``None`` if nothing was tracked). State
        is cleared ONLY after the sell is accepted, so a failure leaves the
        position tracked for a retry rather than orphaning it.
        """
        if self._product_id is None:
            log.warning("close_option called but no option position is tracked — skipping")
            return None

        product_id = self._product_id
        size = self._size

        log.info("Options exit", extra={"extra": {"product_id": product_id, "size": size}})

        result = await asyncio.to_thread(
            self._rest.place_market_order, product_id, size, Side.SELL, True  # reduce_only=True
        )

        # Clear state only AFTER the sell has been accepted.
        self._product_id = None
        self._size = 0
        self._option_type = None

        log.info(
            "Option SELL (close) order placed",
            extra={
                "extra": {
                    "product_id": product_id,
                    "size": size,
                    "fill_price": result.average_fill_price,
                    "order_id": result.order_id,
                }
            },
        )
        return result.average_fill_price

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _select_contract(
        self, underlying: str, expiry: date, target_strike: int, option_type: OptionType
    ) -> tuple[int, float]:
        """Pick the best (nearest-to-target) LISTED option contract.

        Queries the live chain and snaps to the listed strike closest to
        ``target_strike`` (= price ± offset). This works for any strike grid
        (100/200/250/1000) and never builds an unlisted symbol. Falls back to
        direct symbol resolution if the chain query returns nothing.

        Returns ``(product_id, chosen_strike)``.
        """
        try:
            chain = await asyncio.to_thread(
                self._rest.get_option_chain, underlying, expiry, option_type
            )
        except DeltaRestError as exc:
            log.warning(
                "Option chain query failed — falling back to direct symbol resolution",
                extra={"extra": {"error": str(exc)}},
            )
            chain = []

        if chain:
            best = min(chain, key=lambda c: abs(c["strike"] - target_strike))
            log.info(
                "Selected option contract from chain",
                extra={
                    "extra": {
                        "symbol": best["symbol"],
                        "strike": best["strike"],
                        "target": target_strike,
                        "mark_price": best["mark_price"],
                        "chain_size": len(chain),
                    }
                },
            )
            return best["product_id"], best["strike"]

        # Fallback: build the symbol directly with the rounded target strike.
        log.warning(
            "Option chain empty — resolving target strike directly",
            extra={"extra": {"target_strike": target_strike, "expiry": expiry.isoformat()}},
        )
        product_id = await asyncio.to_thread(
            self._rest.resolve_option_product_id, underlying, expiry, target_strike, option_type
        )
        return product_id, float(target_strike)

    async def _check_balance(self) -> None:
        """Best-effort margin gate: skip the sell if available balance is below the
        configured floor. The exchange's own margin check remains the authority."""
        floor = self._settings.option_min_available_balance
        if floor <= 0:
            return
        avail = await asyncio.to_thread(
            self._rest.get_available_balance, self._settings.option_margin_asset or None
        )
        if avail < floor:
            raise OptionsMarginError(
                f"Available balance {avail} below required floor {floor} — skipping option sell"
            )
        log.debug("Margin pre-check passed", extra={"extra": {"available": avail, "floor": floor}})

    def _select_expiry(self) -> date:
        """Return the nearest daily expiry date in IST.

        If the current IST hour is at/past ``option_expiry_cutoff_hour``, use
        tomorrow to avoid entering a position that expires today (Delta options
        settle at 17:30 IST / 12:00 UTC).
        """
        now_ist = datetime.now(tz=_IST)
        if now_ist.hour >= self._settings.option_expiry_cutoff_hour:
            return (now_ist + timedelta(days=1)).date()
        return now_ist.date()

    def _calc_strike(self, btc_price: float, option_type: OptionType) -> int:
        """Compute the TARGET strike (the bot then snaps to the nearest listed one).

        CALL (bought on a BUY signal): strike = price - offset  (ITM call)
        PUT  (bought on a SELL signal): strike = price + offset  (ITM put)

        Rounded to ``option_strike_interval`` for a tidy target; the actual traded
        strike is the nearest LISTED strike chosen by :meth:`_select_contract`, so
        the interval is only a hint, not a hard requirement.
        """
        interval = self._settings.option_strike_interval
        offset = self._settings.option_offset
        raw = btc_price + offset if option_type == OptionType.PUT else btc_price - offset
        return int(round(raw / interval) * interval)
