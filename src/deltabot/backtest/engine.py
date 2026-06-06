"""Backtest engine: replays closed candles through the SAME strategy used live,
so backtest and live cannot diverge.

The strategy is the Pine port (prev-day OHLC + Supertrend + 50-EMA high/low with
a previous-candle stop-loss and a timed square-off). Entries fill at the bar
close; stop-loss exits fill at the stop level; square-off exits fill at the bar
close — matching the live closed-candle approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings, load_settings
from ..enums import SignalDir
from ..models import Candle
from ..pnl import RoundTrip, TradeLedger
from ..strategy.pine_strategy import PineStrategy


@dataclass
class BacktestResult:
    trips: list[RoundTrip]
    final_state: str
    candles_processed: int
    contract_value: float
    contracts: int
    extra: dict = field(default_factory=dict)


class BacktestEngine:
    """Replays the Pine strategy over historical candles into a trade ledger."""

    def __init__(
        self,
        period: int = 10,
        multiplier: float = 3.0,
        contracts: int = 1,
        contract_value: float = 0.001,
        settings: Settings | None = None,
    ) -> None:
        self.period = period
        self.multiplier = multiplier
        self.contracts = contracts
        self.contract_value = contract_value
        # Strategy-shape parameters (EMA length, day boundary, square-off, etc.)
        # come from settings so backtest and live share one configuration source.
        self.settings = settings or load_settings()

    def _make_strategy(self) -> PineStrategy:
        s = self.settings
        return PineStrategy(
            atr_period=self.period,
            st_multiplier=self.multiplier,
            ema_length=s.ema_length,
            day_tz=s.day_tz,
            day_start_hour=s.day_start_hour,
            day_start_minute=s.day_start_minute,
            square_off_hour=s.square_off_hour,
            square_off_minute=s.square_off_minute,
            use_close=s.use_close,
        )

    def run(self, candles: list[Candle]) -> BacktestResult:
        strategy = self._make_strategy()
        ledger = TradeLedger(contract_value=self.contract_value, contracts=self.contracts)
        processed = 0

        for candle in candles:
            decision = strategy.update(candle)
            processed += 1
            if decision is None:
                continue

            # Exits first (so a same-bar reversal closes the old position before
            # the new one opens), then entries.
            if decision.long_exit or decision.short_exit:
                price = decision.long_exit_price if decision.long_exit else decision.short_exit_price
                ledger.close(price, candle.start_time)
            if decision.buy_signal:
                ledger.open(SignalDir.LONG.value, decision.entry_price, candle.start_time)
            elif decision.sell_signal:
                ledger.open(SignalDir.SHORT.value, decision.entry_price, candle.start_time)

        # Close any residual open position at the last candle for accounting.
        if ledger.has_open and candles:
            ledger.close(candles[-1].close, candles[-1].start_time)

        return BacktestResult(
            trips=ledger.trips,
            final_state=strategy.position_state.value,
            candles_processed=processed,
            contract_value=self.contract_value,
            contracts=self.contracts,
        )
