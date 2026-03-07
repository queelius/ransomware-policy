"""Tests for SimClock."""

from datetime import datetime, timedelta

from simulator.clock import SimClock


class TestSimClock:
    def test_now_returns_start(self):
        start = datetime(2025, 6, 15, 10, 0, 0)
        clock = SimClock(start)
        assert clock.now() == start

    def test_tick_advances_time(self):
        start = datetime(2025, 6, 15, 10, 0, 0)
        clock = SimClock(start)
        new_time = clock.tick(30)
        assert new_time == start + timedelta(seconds=30)
        assert clock.now() == new_time

    def test_multiple_ticks(self):
        start = datetime(2025, 6, 15, 10, 0, 0)
        clock = SimClock(start)
        clock.tick(10)
        clock.tick(20)
        assert clock.now() == start + timedelta(seconds=30)

    def test_elapsed(self):
        start = datetime(2025, 6, 15, 10, 0, 0)
        clock = SimClock(start)
        assert clock.elapsed() == 0.0
        clock.tick(45.5)
        assert clock.elapsed() == 45.5

    def test_reset(self):
        start = datetime(2025, 6, 15, 10, 0, 0)
        clock = SimClock(start)
        clock.tick(100)

        new_start = datetime(2025, 7, 1, 0, 0, 0)
        clock.reset(new_start)
        assert clock.now() == new_start
        assert clock.elapsed() == 0.0
