from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock

from deltabot.config import Settings
from deltabot.core import position_state
from deltabot.core.weekly_strangle_hedge_trader import (
    WeeklyStrangleHedgeEngine,
    _IST,
    expiry_from_symbol,
    next_friday_expiry,
)
from deltabot.enums import OptionType
from deltabot.models import Candle


class FakeExecutor:
    def __init__(self, fill: float = 1000.0, close_fill: float = 300.0) -> None:
        self.has_open_position = False
        self.tracked_symbol: str | None = None
        self.tracked_product_id: int | None = None
        self.tracked_strike: float | None = None
        self.tracked_size = 0
        self.underlying = "BTC"
        self.fill = fill
        self.close_fill = close_fill
        self.open_calls: list[tuple] = []
        self.close_calls = 0

    async def open_option_at_strike(self, option_type, expiry, target_strike):
        self.open_calls.append((option_type, expiry, target_strike))
        if self.fill is None:
            return None, None
        strike = round(target_strike / 1000) * 1000
        self.has_open_position = True
        self.tracked_strike = float(strike)
        self.tracked_symbol = f"{option_type.value}-BTC-{strike}-{expiry.strftime('%d%m%y')}"
        self.tracked_product_id = strike
        self.tracked_size = 15
        return self.fill, self.tracked_symbol

    async def close_option(self):
        self.close_calls += 1
        self.has_open_position = False
        self.tracked_symbol = None
        return self.close_fill

    def clear(self) -> None:
        self.has_open_position = False
        self.tracked_symbol = None
        self.tracked_product_id = None

    def adopt(self, product_id, size, option_type, symbol=None) -> None:
        self.has_open_position = True
        self.tracked_product_id = product_id
        self.tracked_symbol = symbol
        self.tracked_size = abs(size)


class FakeRest:
    def __init__(self, btc: float = 80000.0, positions=None) -> None:
        self.btc = btc
        self.positions = positions or []

    def get_candles(self, symbol, resolution, start, end):
        t = int(time.time()) - 120
        return [Candle(t, self.btc, self.btc, self.btc, self.btc, 1.0)]

    def get_option_positions(self, underlying):
        return self.positions


def _engine(state_file: str = "", **kw):
    s = Settings(_env_file=None, strategy="weeklyhedge", option_contracts=15, state_file=state_file, **kw)
    rest = FakeRest()
    notifier = AsyncMock()
    eng = WeeklyStrangleHedgeEngine(s, rest, notifier)
    for leg in eng.legs.values():
        leg.executor = FakeExecutor()
    return eng, rest, notifier


def test_next_friday_expiry():
    fri = datetime(2026, 10, 2, 21, 0, tzinfo=_IST)
    assert next_friday_expiry(fri) == date(2026, 10, 9)
    assert next_friday_expiry(datetime(2026, 10, 5, 9, 0, tzinfo=_IST)) == date(2026, 10, 9)
    assert expiry_from_symbol("C-BTC-87000-091026") == date(2026, 10, 9)


async def test_entry_sells_both_legs_and_sets_breakevens():
    eng, rest, _ = _engine()
    await eng._maybe_enter()
    ce, pe = eng.legs["ce"].executor, eng.legs["pe"].executor
    assert ce.open_calls[0][0] == OptionType.CALL and ce.open_calls[0][2] == 81600.0
    assert pe.open_calls[0][0] == OptionType.PUT and pe.open_calls[0][2] == 78400.0
    assert eng.meta["combined"] == 2000.0
    assert eng.meta["upper"] == 82000.0 + 2000.0 and eng.meta["lower"] == 78000.0 - 2000.0
    assert not eng.legs["hce"].executor.open_calls and not eng.legs["hpe"].executor.open_calls


async def test_entry_refused_with_untracked_positions_on_account():
    eng, rest, notifier = _engine()
    rest.positions = [{"symbol": "P-BTC-85400-021026", "size": -2, "product_id": 1}]
    await eng._maybe_enter()
    assert not eng.legs["ce"].executor.open_calls
    assert not eng.has_position
    notifier.notify.assert_awaited()


async def test_pe_failure_unwinds_ce():
    eng, _, _ = _engine()
    eng.legs["pe"].executor.fill = None
    await eng._maybe_enter()
    assert eng.legs["ce"].executor.close_calls == 1
    assert not eng.has_position and eng.meta is None


async def test_breakeven_cross_buys_hedge_once():
    eng, rest, _ = _engine()
    await eng._maybe_enter()
    rest.btc = eng.meta["upper"] + 100
    await eng._tick()
    await eng._tick()
    h = eng.legs["hce"].executor
    assert len(h.open_calls) == 1
    assert h.open_calls[0][2] == eng.meta["upper"] + 2000
    assert eng.meta["hedged"] == ["CE"]
    assert not eng.legs["hpe"].executor.open_calls


async def test_inside_breakevens_no_hedge():
    eng, rest, _ = _engine()
    await eng._maybe_enter()
    rest.btc = 81000.0
    await eng._tick()
    assert not eng.legs["hce"].executor.open_calls and not eng.legs["hpe"].executor.open_calls


async def test_closes_everything_at_expiry_time():
    eng, rest, _ = _engine()
    await eng._maybe_enter()
    rest.btc = eng.meta["lower"] - 100
    await eng._tick()
    eng.meta["expiry"] = (datetime.now(_IST) - timedelta(days=1)).date().isoformat()
    await eng._tick()
    assert not eng.has_position and eng.meta is None
    for name in ("ce", "pe", "hpe"):
        assert eng.legs[name].executor.close_calls == 1


async def test_reconcile_adopts_only_own_state_symbols(tmp_path):
    base = str(tmp_path / "wk_pos.json")
    eng, rest, _ = _engine(state_file=base)
    position_state.save(eng.legs["ce"].state_file, symbol="C-BTC-87000-091026", product_id=11, size=15, entry_premium=770)
    position_state.save(eng.legs["pe"].state_file, symbol="P-BTC-84000-091026", product_id=12, size=15, entry_premium=752)
    position_state.save(eng.meta_file, expiry="2026-10-09", combined=1522, upper=88522, lower=82478, hedged=[])
    rest.positions = [
        {"symbol": "C-BTC-87000-091026", "size": -15, "product_id": 11},
        {"symbol": "P-BTC-99999-091026", "size": -2, "product_id": 99},
    ]
    await eng._reconcile()
    assert eng.legs["ce"].executor.has_open_position
    assert eng.legs["ce"].entry_premium == 770
    assert not eng.legs["pe"].executor.has_open_position
    assert position_state.load(eng.legs["pe"].state_file) is None
    assert eng.meta["upper"] == 88522


def test_daily_cycle_next_entry_is_every_day():
    eng, _, _ = _engine(weekly_cycle="daily")
    mon_10pm = datetime(2026, 10, 5, 22, 0, tzinfo=_IST)
    assert eng._next_entry(mon_10pm) == datetime(2026, 10, 6, 21, 0, tzinfo=_IST)
    mon_8pm = datetime(2026, 10, 5, 20, 0, tzinfo=_IST)
    assert eng._next_entry(mon_8pm) == datetime(2026, 10, 5, 21, 0, tzinfo=_IST)


async def test_daily_cycle_sells_next_day_expiry():
    eng, _, _ = _engine(weekly_cycle="daily")
    await eng._maybe_enter()
    expiry = eng.legs["ce"].executor.open_calls[0][1]
    assert expiry == (datetime.now(_IST) + timedelta(days=1)).date()


async def test_hedge_offset_pct_of_entry_spot():
    eng, rest, _ = _engine(weekly_hedge_offset_pct=1.0)
    await eng._maybe_enter()
    rest.btc = eng.meta["lower"] - 100
    await eng._tick()
    h = eng.legs["hpe"].executor
    assert h.open_calls[0][2] == eng.meta["lower"] - 800.0   # 1% of 80000 entry spot
