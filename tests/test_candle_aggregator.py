"""Tests for closed-candle detection."""

from __future__ import annotations

from deltabot.core.candle_aggregator import CandleAggregator


def _msg(start_time: int, close: float) -> dict:
    return {
        "candle_start_time": start_time,
        "open": close - 1,
        "high": close + 2,
        "low": close - 2,
        "close": close,
        "volume": 1.0,
    }


def test_emits_closed_candle_on_rollover():
    emitted = []
    agg = CandleAggregator(on_closed=lambda c: emitted.append(c))
    agg.ingest(_msg(60, 100))  # first candle, nothing closed yet
    assert emitted == []
    agg.ingest(_msg(60, 101))  # still forming -> update
    assert emitted == []
    agg.ingest(_msg(120, 102))  # rollover -> candle@60 closed
    assert len(emitted) == 1
    assert emitted[0].start_time == 60
    assert emitted[0].close == 101  # last snapshot of the forming bar wins


def test_no_duplicate_emission():
    emitted = []
    agg = CandleAggregator(on_closed=lambda c: emitted.append(c))
    agg.ingest(_msg(60, 100))
    agg.ingest(_msg(120, 101))
    agg.ingest(_msg(120, 101))  # duplicate forming update for the new bar
    agg.ingest(_msg(180, 102))
    starts = [c.start_time for c in emitted]
    assert starts == [60, 120]


def test_out_of_order_ignored():
    emitted = []
    agg = CandleAggregator(on_closed=lambda c: emitted.append(c))
    agg.ingest(_msg(120, 100))
    agg.ingest(_msg(180, 101))  # closes 120
    agg.ingest(_msg(60, 99))  # stale, older than current -> ignored
    agg.ingest(_msg(240, 102))  # closes 180
    starts = [c.start_time for c in emitted]
    assert starts == [120, 180]


def test_forming_candle_never_emitted():
    emitted = []
    agg = CandleAggregator(on_closed=lambda c: emitted.append(c))
    for px in (100, 101, 102, 103):
        agg.ingest(_msg(60, px))  # only ever the forming candle
    assert emitted == []  # nothing closes until a newer start_time arrives
