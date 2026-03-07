"""EventLog: mutable Windows Event Log state.

Stores event log entries the agent can query via query_event_log.
Attack generators log Defender disabled, suspicious service starts, etc.
"""

from __future__ import annotations

from datetime import datetime

from simulator.models import EventLogEntry


class EventLog:
    """Mutable event log tracking Windows Event Log entries.

    List-based (append-only, like a real log).
    """

    def __init__(self) -> None:
        self._entries: list[EventLogEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def log_event(
        self,
        source: str,
        event_id: int,
        message: str,
        level: str = "Information",
        timestamp: datetime | None = None,
    ) -> EventLogEntry:
        """Append an event log entry."""
        entry = EventLogEntry(
            timestamp=timestamp or datetime(2025, 1, 1),
            source=source,
            event_id=event_id,
            level=level,
            message=message,
        )
        self._entries.append(entry)
        return entry

    def query_events(
        self,
        source: str | None = None,
        event_id: int | None = None,
        since: datetime | None = None,
    ) -> list[EventLogEntry]:
        """Query events with optional filters."""
        results = self._entries
        if source is not None:
            results = [e for e in results if e.source == source]
        if event_id is not None:
            results = [e for e in results if e.event_id == event_id]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        return results

    def clear_log(self, timestamp: datetime | None = None) -> None:
        """Clear the log (auto-logs event 1102 per Windows behavior)."""
        self._entries.clear()
        self.log_event(
            source="Security",
            event_id=1102,
            message="The audit log was cleared.",
            level="Warning",
            timestamp=timestamp,
        )

    def count_events(
        self,
        source: str | None = None,
        event_id: int | None = None,
    ) -> int:
        """Count matching events."""
        return len(self.query_events(source=source, event_id=event_id))

    def seed_events(self, now: datetime) -> None:
        """Populate with typical baseline events."""
        self.log_event("System", 6005, "The Event log service was started.",
                       "Information", now)
        self.log_event("System", 6013, "The system uptime is 86400 seconds.",
                       "Information", now)
        self.log_event("Service Control Manager", 7036,
                       "The Windows Defender service entered the running state.",
                       "Information", now)
        self.log_event("Service Control Manager", 7036,
                       "The Windows Update service entered the running state.",
                       "Information", now)
        self.log_event("Security", 4624,
                       "An account was successfully logged on. Subject: SYSTEM",
                       "Information", now)
