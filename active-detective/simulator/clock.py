"""Simulation clock for coordinating time across sub-registries.

Provides a monotonic clock that generators advance via tick().
All registries read from the same clock to stay consistent.
"""

from __future__ import annotations

from datetime import datetime, timedelta


class SimClock:
    """Monotonic simulation clock.

    Generators call ``tick(seconds)`` to advance time.
    All sub-registries call ``now()`` for the current timestamp.
    """

    def __init__(self, start: datetime) -> None:
        self._start = start
        self._current = start

    def now(self) -> datetime:
        """Current simulation time."""
        return self._current

    def tick(self, seconds: float) -> datetime:
        """Advance the clock and return the new time."""
        self._current += timedelta(seconds=seconds)
        return self._current

    def elapsed(self) -> float:
        """Seconds elapsed since simulation start."""
        return (self._current - self._start).total_seconds()

    def reset(self, start: datetime) -> None:
        """Reset the clock to a new start time."""
        self._start = start
        self._current = start
