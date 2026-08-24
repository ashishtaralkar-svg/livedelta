"""resolve_by_premium: strike selection must match OptionsExecutor.select_by_premium's
GLOBAL minimum over the (here, faked) chain -- not just the first strike a
directional walk happens to cross. fetch_option_candles is monkeypatched so
these tests exercise the SEARCH LOGIC only, no real network calls."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from deltabot.backtest import option_pricing as op
from deltabot.enums import OptionType
from deltabot.models import Candle

_EXPIRY = datetime(2026, 7, 8)


def _fake_fetch(premium_by_strike):
    def _fetch(client, symbol, start, end, resolution):
        strike = int(symbol.split("-")[2])
        if strike not in premium_by_strike:
            return {}
        p = premium_by_strike[strike]
        return {1000: Candle(start_time=1000, open=p, high=p, low=p, close=p, volume=1.0)}
    return _fetch


def _resolve(monkeypatch, premiums, target, entry_btc=60000, option_type=OptionType.CALL,
             interval=200):
    monkeypatch.setattr(op, "fetch_option_candles", _fake_fetch(premiums))
    return op.resolve_by_premium(
        client=MagicMock(), underlying="BTC", option_type=option_type,
        entry_btc=entry_btc, expiry=_EXPIRY, interval=interval,
        target_premium=target, entry_ts=1000, exit_ts=1000,
        win_start=0, win_end=2000, resolution="1m", step=60, cache={},
    )


def test_overshoots_past_the_first_crossing_to_find_the_true_closest_strike(monkeypatch) -> None:
    """A strike ONE STEP PAST the first crossing is genuinely closer to
    target than the crossing strike itself -- the old "stop at first
    crossing" search would have missed it entirely (confirmed: it returns
    61200 on this exact data); the fixed search must reach 61400."""
    premiums = {
        60000: 500, 60200: 400, 60400: 300, 60600: 200, 60800: 150,
        61000: 102, 61200: 90, 61400: 96, 61600: 80, 61800: 70,
    }
    symbol, strike, candles = _resolve(monkeypatch, premiums, target=95)
    assert strike == 61400   # diff=1, only reachable by overshooting past 61200 (diff=5)
    assert symbol == "C-BTC-61400-080726"


def test_simple_monotonic_case_still_picks_the_crossing_strike(monkeypatch) -> None:
    """No overshoot needed when the chain decays cleanly monotonically --
    the crossing strike genuinely IS the closest. Regression guard: the fix
    must not change the common case."""
    premiums = {60000: 500, 60200: 400, 60400: 300, 60600: 200, 60800: 100}
    symbol, strike, candles = _resolve(monkeypatch, premiums, target=210)
    assert strike == 60600   # diff=10, clearly better than 60400's diff=90 or 60800's diff=110


def test_itm_search_widened_beyond_the_old_fixed_6_steps(monkeypatch) -> None:
    """target_premium sits 10 strikes ITM (CALL: ITM = below ATM) -- the old
    code only ever checked 6 ITM steps and would have missed this entirely
    (fallen back to whatever the OTM side found, a much worse match)."""
    itm_strikes = {60000 - i * 200: 500 + i * 80 for i in range(15)}   # premium rises going ITM
    otm_strikes = {60000 + i * 200: max(10, 500 - i * 80) for i in range(1, 15)}
    premiums = {**itm_strikes, **otm_strikes}
    target = itm_strikes[60000 - 10 * 200]   # exactly the premium 10 strikes ITM
    symbol, strike, candles = _resolve(monkeypatch, premiums, target=target)
    assert strike == 60000 - 10 * 200


def test_put_side_mirrors_direction(monkeypatch) -> None:
    """PUT: OTM = strikes BELOW atm (otm = -interval); confirm the search
    still finds the true closest with the direction mirrored."""
    premiums = {
        60000: 500, 59800: 400, 59600: 300, 59400: 200, 59200: 150,
        59000: 102, 58800: 90, 58600: 96, 58400: 80, 58200: 70,
    }
    symbol, strike, candles = _resolve(monkeypatch, premiums, target=95, option_type=OptionType.PUT)
    assert strike == 58600
    assert symbol == "P-BTC-58600-080726"


def test_returns_none_when_no_strike_has_valid_data(monkeypatch) -> None:
    result = _resolve(monkeypatch, {}, target=100)
    assert result is None
