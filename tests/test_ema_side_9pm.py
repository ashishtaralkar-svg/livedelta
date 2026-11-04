from datetime import datetime
from zoneinfo import ZoneInfo

from deltabot.core.ema_side_9pm_trader import ema, session_window, stacked_signal
from deltabot.enums import SignalDir

IST = ZoneInfo("Asia/Kolkata")


def test_ema_seeds_with_sma():
    out = ema([1, 2, 3, 4], 2)
    assert out[0] is None
    assert out[1] == 1.5
    assert abs(out[2] - (2 / 3 * 3 + 1 / 3 * 1.5)) < 1e-9


def test_stacked_signal_directions():
    up = [100 + i for i in range(300)]
    assert stacked_signal(up, 50, 200)[0] == SignalDir.LONG.value      # sell PUT
    down = [400 - i for i in range(300)]
    assert stacked_signal(down, 50, 200)[0] == SignalDir.SHORT.value   # sell CALL
    # rally back above EMA50 while EMA50 is still below EMA200 -> no signal
    mixed = down[:-1] + [down[-2] + 60]
    assert stacked_signal(mixed, 50, 200)[0] is None
    assert stacked_signal(up[:100], 50, 200)[0] is None                # not enough history


def test_session_window():
    def at(d, h, m):
        return datetime(2026, 11, d, h, m, tzinfo=IST)

    start, end = session_window(at(4, 21, 0), 21, 0, 17, 0)
    assert start == at(4, 21, 0) and end == at(5, 17, 0)
    start, end = session_window(at(5, 9, 30), 21, 0, 17, 0)
    assert start == at(4, 21, 0)                                       # still last night's session
    assert session_window(at(5, 17, 0), 21, 0, 17, 0) is None
    assert session_window(at(5, 20, 59), 21, 0, 17, 0) is None
