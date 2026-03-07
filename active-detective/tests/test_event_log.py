"""Tests for EventLog."""

from datetime import datetime, timedelta

import pytest

from simulator.event_log import EventLog


class TestEventLog:
    @pytest.fixture
    def log(self):
        return EventLog()

    def test_log_event(self, log):
        now = datetime(2025, 1, 1)
        entry = log.log_event("System", 6005, "Service started.", "Information", now)
        assert entry.source == "System"
        assert entry.event_id == 6005
        assert len(log) == 1

    def test_query_all(self, log):
        now = datetime(2025, 1, 1)
        log.log_event("System", 6005, "Start", timestamp=now)
        log.log_event("Security", 4624, "Logon", timestamp=now)
        assert len(log.query_events()) == 2

    def test_query_by_source(self, log):
        now = datetime(2025, 1, 1)
        log.log_event("System", 6005, "Start", timestamp=now)
        log.log_event("Security", 4624, "Logon", timestamp=now)
        log.log_event("System", 6013, "Uptime", timestamp=now)
        results = log.query_events(source="System")
        assert len(results) == 2

    def test_query_by_event_id(self, log):
        now = datetime(2025, 1, 1)
        log.log_event("System", 6005, "Start", timestamp=now)
        log.log_event("System", 6005, "Start again", timestamp=now)
        log.log_event("System", 6013, "Uptime", timestamp=now)
        results = log.query_events(event_id=6005)
        assert len(results) == 2

    def test_query_since(self, log):
        t1 = datetime(2025, 1, 1, 10, 0, 0)
        t2 = datetime(2025, 1, 1, 11, 0, 0)
        t3 = datetime(2025, 1, 1, 12, 0, 0)
        log.log_event("System", 1, "A", timestamp=t1)
        log.log_event("System", 2, "B", timestamp=t2)
        log.log_event("System", 3, "C", timestamp=t3)
        results = log.query_events(since=t2)
        assert len(results) == 2
        assert all(e.timestamp >= t2 for e in results)

    def test_query_combined_filters(self, log):
        now = datetime(2025, 1, 1)
        log.log_event("System", 6005, "Start", timestamp=now)
        log.log_event("Security", 6005, "Other", timestamp=now)
        log.log_event("System", 6013, "Uptime", timestamp=now)
        results = log.query_events(source="System", event_id=6005)
        assert len(results) == 1

    def test_clear_log(self, log):
        now = datetime(2025, 1, 1)
        log.log_event("System", 6005, "Start", timestamp=now)
        log.log_event("System", 6013, "Uptime", timestamp=now)
        assert len(log) == 2

        log.clear_log(now)
        # Should only have the audit clear event
        assert len(log) == 1
        assert log.query_events()[0].event_id == 1102

    def test_count_events(self, log):
        now = datetime(2025, 1, 1)
        log.log_event("System", 6005, "Start", timestamp=now)
        log.log_event("System", 6005, "Start again", timestamp=now)
        log.log_event("Security", 4624, "Logon", timestamp=now)
        assert log.count_events(source="System") == 2
        assert log.count_events(event_id=4624) == 1
        assert log.count_events() == 3

    def test_seed_events(self, log):
        now = datetime(2025, 1, 1)
        log.seed_events(now)
        assert len(log) == 5
        # Should have System and Security events
        sources = {e.source for e in log.query_events()}
        assert "System" in sources
        assert "Security" in sources

    def test_len(self, log):
        assert len(log) == 0
        log.log_event("System", 1, "Test")
        assert len(log) == 1
