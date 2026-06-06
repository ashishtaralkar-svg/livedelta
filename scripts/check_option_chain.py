"""Read-only smoke test for options resolution — places NO orders.

Validates the exact option-selection path the live bot uses, end to end:
  1. fetch the latest BTC price from REST history
  2. compute the target strike (price +/- offset) for each signal direction
  3. query the LIVE Delta option chain for the nearest daily expiry
  4. snap to the nearest listed strike and print the contract that WOULD be sold

It only reads public ticker/chain/candle endpoints, so placeholder API keys are
fine. Run it after filling in .env (region / testnet / symbol / option_* settings):

    python scripts/check_option_chain.py

A healthy run prints a non-empty chain and a SELECTED product_id + real symbol
(e.g. ``P-BTC-100400-060625``) for both BUY and SELL. An empty chain or a resolve
error tells you the symbol/expiry/region needs fixing BEFORE you trade live.
"""

from __future__ import annotations

import asyncio
import time

from deltabot.config import load_settings
from deltabot.core.options_executor import OptionsExecutor
from deltabot.enums import OptionType, SignalDir
from deltabot.exchange.rest_client import RestClient
from deltabot.logging_setup import setup_logging


def _latest_btc_price(rest: RestClient, settings) -> float:
    now = int(time.time())
    candles = rest.get_candles(settings.symbol, settings.resolution, now - 3600, now)
    if not candles:
        raise SystemExit(
            f"No candles returned for {settings.symbol} — check region/symbol/network."
        )
    return candles[-1].close


async def _probe(rest: RestClient, exec_: OptionsExecutor, underlying: str, price: float, signal_dir) -> None:
    option_type = OptionType.PUT if signal_dir == SignalDir.LONG else OptionType.CALL
    target = exec_._calc_strike(price, option_type)  # noqa: SLF001 (diagnostic)
    expiry = exec_._select_expiry()  # noqa: SLF001
    label = "BUY  -> sell PUT " if signal_dir == SignalDir.LONG else "SELL -> sell CALL"

    print(f"\n=== {label} ===")
    print(f"  BTC ~ {price:.1f} | offset={exec_._settings.option_offset} -> target strike={target}")
    print(f"  expiry={expiry.isoformat()}  type={option_type.value}")

    chain = await asyncio.to_thread(rest.get_option_chain, underlying, expiry, option_type)
    print(f"  live chain size: {len(chain)}")
    for c in sorted(chain, key=lambda c: abs(c["strike"] - target))[:5]:
        print(f"    {c['symbol']:<30} strike={c['strike']:<10} mark={c['mark_price']}")

    try:
        pid, strike = await exec_._select_contract(underlying, expiry, target, option_type)  # noqa: SLF001
        print(f"  -> SELECTED product_id={pid}  strike={strike}")
    except Exception as exc:  # noqa: BLE001
        print(f"  !! resolution FAILED: {exc}")


async def _run_all(rest: RestClient, exec_: OptionsExecutor, underlying: str, price: float) -> None:
    await _probe(rest, exec_, underlying, price, SignalDir.LONG)
    await _probe(rest, exec_, underlying, price, SignalDir.SHORT)


def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    print(f"region={settings.region}  testnet={settings.testnet}  symbol={settings.symbol}")
    print(f"REST base: {settings.rest_base_url}")

    rest = RestClient(
        base_url=settings.rest_base_url,
        api_key=settings.api_key.get_secret_value(),
        api_secret=settings.api_secret.get_secret_value(),
    )
    try:
        exec_ = OptionsExecutor(rest, settings)
        underlying = exec_.underlying
        print(f"underlying token: {underlying!r}")
        price = _latest_btc_price(rest, settings)
        asyncio.run(_run_all(rest, exec_, underlying, price))
    finally:
        rest.close()


if __name__ == "__main__":
    main()
