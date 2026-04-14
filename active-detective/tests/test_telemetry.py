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
        assert "entropy_delta" not in text

    def test_file_event_excludes_entropy_delta(self):
        """entropy_delta must not appear in passive telemetry — agent must actively investigate."""
        events = [
            FileEvent(
                ts=datetime(2025, 6, 15, 10, 0, 5),
                path="C:/test.docx",
                size_delta=1024,
                entropy_delta=3.5,
                extension_change=".docx->.locked",
            )
        ]
        text = format_telemetry_window(events, datetime(2025, 6, 15, 10, 0, 0))
        assert "entropy_delta" not in text
        assert "FILE" in text
        assert "size_delta" in text
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


class TestMultiWindowGeneration:
    def test_returns_history_windows(self):
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BLITZ, 0.8, rng, attack_progress=0.6, n_history=2)
        assert len(episode.history_windows) == 2
        assert isinstance(episode.history_windows[0], str)
        assert len(episode.history_windows[0]) > 0

    def test_benign_has_history(self):
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BENIGN, 0.8, rng, n_history=2)
        assert len(episode.history_windows) == 2

    def test_zero_history(self):
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BLITZ, 0.8, rng, attack_progress=0.6, n_history=0)
        assert episode.history_windows == []

    def test_default_history_is_two(self):
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BLITZ, 0.8, rng, attack_progress=0.6)
        assert len(episode.history_windows) == 2

    def test_history_progress_increases(self):
        """History windows should represent earlier attack stages."""
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BLITZ, 0.8, rng, attack_progress=0.6, n_history=2)
        # History windows exist and are different from each other
        assert episode.history_windows[0] != episode.history_windows[1]

    def test_reproducible_with_seed(self):
        ep1 = generate_episode(
            ScenarioType.BLITZ, 0.8, np.random.RandomState(42),
            attack_progress=0.6, n_history=2)
        ep2 = generate_episode(
            ScenarioType.BLITZ, 0.8, np.random.RandomState(42),
            attack_progress=0.6, n_history=2)
        assert ep1.history_windows == ep2.history_windows
        assert ep1.input_text == ep2.input_text


class TestEpisodeHost:
    def test_episode_has_host(self):
        rng = np.random.RandomState(42)
        episode = generate_episode(ScenarioType.BENIGN, 0.8, rng)
        assert episode.host is not None
        # Host should have files seeded
        assert len(episode.host.files.all_paths()) > 0


class TestHistoryTemporalCoupling:
    """Phase 3: history windows must replay on a single shared HostState.

    Before Phase 3, each history window generated an independent
    HostState.create(), so 'temporal context' was actually noise from
    different seed draws. A file encrypted in t-2 wasn't encrypted in
    the current window. These tests pin the new semantics.
    """

    def test_files_encrypted_in_history_stay_encrypted(self):
        """BLITZ encryptions in t-2 must persist to current window."""
        rng = np.random.RandomState(42)
        episode = generate_episode(
            ScenarioType.BLITZ, observability=1.0, rng=rng,
            attack_progress=0.9, n_history=2,
        )
        # At attack_progress=0.9, many files should be encrypted across
        # all 3 windows (t-2, t-1, current). The final host should have
        # encrypted files present.
        encrypted = [
            p for p in episode.host.files.all_paths()
            if episode.host.files.get_file(p).is_encrypted
        ]
        assert len(encrypted) > 0

    def test_ground_truth_invariant_to_n_history(self):
        """Ground truth (attack phase, is_ransomware) must not depend on n_history.

        Under the broken implementation, history generation consumed parent-
        rng draws before the current window, shifting scenario sampling so
        drastically that attack_phase could flip (reconnaissance vs encryption).
        The fix draws current-window seeds before history seeds.
        """
        for progress in (0.1, 0.5, 0.9):
            ep0 = generate_episode(
                ScenarioType.BLITZ, observability=1.0,
                rng=np.random.RandomState(42),
                attack_progress=progress, n_history=0,
            )
            ep2 = generate_episode(
                ScenarioType.BLITZ, observability=1.0,
                rng=np.random.RandomState(42),
                attack_progress=progress, n_history=2,
            )
            ep5 = generate_episode(
                ScenarioType.BLITZ, observability=1.0,
                rng=np.random.RandomState(42),
                attack_progress=progress, n_history=5,
            )
            assert ep0.ground_truth.attack_phase == ep2.ground_truth.attack_phase == ep5.ground_truth.attack_phase
            assert ep0.ground_truth.is_ransomware == ep2.ground_truth.is_ransomware

    def test_episode_host_is_final_state(self):
        """The returned host should reflect all window mutations, not just current."""
        rng = np.random.RandomState(42)
        # SLEEPER encrypts very few files per window. With history,
        # total encrypted files should be higher than a single window.
        ep_hist = generate_episode(
            ScenarioType.SLEEPER, observability=1.0,
            rng=np.random.RandomState(42),
            attack_progress=0.8, n_history=2,
        )
        ep_nohist = generate_episode(
            ScenarioType.SLEEPER, observability=1.0,
            rng=np.random.RandomState(42),
            attack_progress=0.8, n_history=0,
        )
        hist_encrypted = sum(
            1 for p in ep_hist.host.files.all_paths()
            if ep_hist.host.files.get_file(p).is_encrypted
        )
        nohist_encrypted = sum(
            1 for p in ep_nohist.host.files.all_paths()
            if ep_nohist.host.files.get_file(p).is_encrypted
        )
        # With n_history=2, sleeper encrypts 1-2 per window over 3 windows,
        # so final host should have strictly more encrypted files than the
        # single-window case. Under the broken implementation, each window
        # has its own disconnected host and the final host only reflects
        # current-window encryptions.
        assert hist_encrypted > nohist_encrypted, (
            f"Expected history to accumulate encryptions, got "
            f"hist={hist_encrypted} vs nohist={nohist_encrypted}"
        )

    def test_episode_host_matches_telemetry_files(self):
        """Files mentioned in telemetry should exist in episode.host."""
        rng = np.random.RandomState(42)
        episode = generate_episode(ScenarioType.BLITZ, 0.9, rng, attack_progress=0.5)
        # Extract file paths from telemetry text
        import re
        paths = re.findall(r'path=(C:/[^\s]+)', episode.input_text)
        for path in paths:
            # File should exist in the host (possibly renamed by attack)
            # At minimum, the directory should exist
            dir_path = path.rsplit('/', 1)[0]
            assert dir_path in episode.host.files.directories()
