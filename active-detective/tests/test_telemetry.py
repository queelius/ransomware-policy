"""Tests for telemetry window builder and observability filter."""

from datetime import datetime

import numpy as np
import pytest

from simulator.models import (
    FileEvent,
    NetEvent,
    ProcessEvent,
    ScenarioType,
)
from simulator.telemetry import (
    Episode,
    apply_observability_filter,
    format_telemetry_window,
    generate_episode,
)


class TestObservabilityFilter:
    def _make_events(self, n: int, now: datetime) -> list:
        return [
            FileEvent(ts=now, path=f"C:/file_{i}.txt", size_delta=0,
                      entropy_delta=0.0, extension_change=None)
            for i in range(n)
        ]

    def test_full_observability_keeps_all(self):
        now = datetime(2025, 1, 1)
        events = self._make_events(100, now)
        rng = np.random.RandomState(42)
        surviving = apply_observability_filter(events, 1.0, rng)
        assert len(surviving) == 100

    def test_zero_observability_drops_all(self):
        now = datetime(2025, 1, 1)
        events = self._make_events(100, now)
        rng = np.random.RandomState(42)
        surviving = apply_observability_filter(events, 0.0, rng)
        assert len(surviving) == 0

    def test_partial_observability_statistical(self):
        """At 50% observability, ~50% of events should survive."""
        now = datetime(2025, 1, 1)
        events = self._make_events(1000, now)
        rng = np.random.RandomState(42)
        surviving = apply_observability_filter(events, 0.5, rng)
        # Allow ±5% tolerance
        ratio = len(surviving) / 1000
        assert 0.45 <= ratio <= 0.55

    def test_custom_drop_probability(self):
        """Events with custom drop_probability use that instead of global."""
        now = datetime(2025, 1, 1)
        event = FileEvent(
            ts=now, path="C:/test.txt", size_delta=0,
            entropy_delta=0.0, extension_change=None,
            drop_probability=1.0,  # always drop
        )
        rng = np.random.RandomState(42)
        surviving = apply_observability_filter([event], 1.0, rng)
        assert len(surviving) == 0

    def test_reproducible_with_seed(self):
        now = datetime(2025, 1, 1)
        events = self._make_events(100, now)
        r1 = apply_observability_filter(events, 0.5, np.random.RandomState(42))
        r2 = apply_observability_filter(events, 0.5, np.random.RandomState(42))
        assert len(r1) == len(r2)


class TestFormatTelemetryWindow:
    def test_empty_events(self):
        now = datetime(2025, 1, 1)
        text = format_telemetry_window([], now)
        assert "no events" in text.lower()

    def test_process_event_format(self):
        now = datetime(2025, 1, 1)
        event = ProcessEvent(
            ts=now, pid=100, name="test.exe",
            parent_pid=4, command_line="test.exe --flag",
        )
        text = format_telemetry_window([event], now)
        assert "PROC" in text
        assert "pid=100" in text
        assert "test.exe" in text

    def test_file_event_format(self):
        now = datetime(2025, 1, 1)
        event = FileEvent(
            ts=now, path="C:/test.docx", size_delta=1024,
            entropy_delta=3.5, extension_change=".docx -> .docx.locked",
        )
        text = format_telemetry_window([event], now)
        assert "FILE" in text
        assert "C:/test.docx" in text
        assert "ext_change" in text

    def test_net_event_format(self):
        now = datetime(2025, 1, 1)
        event = NetEvent(
            ts=now, destination="10.0.0.1", bytes_transferred=1024,
            protocol="tcp", direction="outbound",
        )
        text = format_telemetry_window([event], now)
        assert "NET" in text
        assert "10.0.0.1" in text
        assert "outbound" in text

    def test_sorted_by_timestamp(self):
        from datetime import timedelta
        now = datetime(2025, 1, 1)
        e1 = FileEvent(ts=now + timedelta(seconds=30), path="C:/late.txt",
                        size_delta=0, entropy_delta=0.0, extension_change=None)
        e2 = FileEvent(ts=now + timedelta(seconds=5), path="C:/early.txt",
                        size_delta=0, entropy_delta=0.0, extension_change=None)
        text = format_telemetry_window([e1, e2], now)
        early_pos = text.index("early.txt")
        late_pos = text.index("late.txt")
        assert early_pos < late_pos


class TestGenerateEpisode:
    def test_benign_episode(self):
        rng = np.random.RandomState(42)
        ep = generate_episode(ScenarioType.BENIGN, 0.5, rng)
        assert ep.ground_truth.is_ransomware is False
        assert ep.ground_truth.label == "benign"
        assert ep.observability == 0.5
        assert len(ep.input_text) > 0

    def test_blitz_episode(self):
        rng = np.random.RandomState(42)
        ep = generate_episode(ScenarioType.BLITZ, 0.7, rng, attack_progress=0.8)
        assert ep.ground_truth.is_ransomware is True
        assert "blitz" in ep.ground_truth.label

    def test_exfil_episode(self):
        rng = np.random.RandomState(42)
        ep = generate_episode(ScenarioType.EXFIL_FIRST, 0.9, rng,
                              attack_progress=0.3)
        assert ep.ground_truth.attack_phase == "exfiltration"

    def test_observability_affects_event_count(self):
        rng1 = np.random.RandomState(42)
        ep_high = generate_episode(ScenarioType.BLITZ, 0.9, rng1,
                                    attack_progress=0.5)
        rng2 = np.random.RandomState(42)
        ep_low = generate_episode(ScenarioType.BLITZ, 0.1, rng2,
                                   attack_progress=0.5)
        # Same raw events, but low obs should see fewer
        assert ep_low.visible_event_count <= ep_high.visible_event_count

    def test_reproducible(self):
        ep1 = generate_episode(ScenarioType.BENIGN, 0.5, np.random.RandomState(42))
        ep2 = generate_episode(ScenarioType.BENIGN, 0.5, np.random.RandomState(42))
        assert ep1.input_text == ep2.input_text
