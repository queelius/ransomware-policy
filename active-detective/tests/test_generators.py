"""Tests for benign and attack scenario generators."""

from datetime import datetime

import numpy as np
import pytest

from simulator.generators import (
    office_edits,
    browser_downloads,
    backup_operations,
    av_scan,
    system_maintenance,
    blitz_encryptor,
    slow_sleeper,
    exfil_first,
    semantic_shuffle,
)
from simulator.host import HostState
from simulator.models import (
    EventLogEvent,
    FileEvent,
    NetEvent,
    ProcessEvent,
    RegistryEvent,
)


@pytest.fixture
def host():
    """Return a seeded HostState."""
    rng = np.random.RandomState(42)
    now = datetime(2025, 6, 15, 10, 0, 0)
    return HostState.create(rng, now)


# ── Benign generators ────────────────────────────────────────────────


class TestOfficeEdits:
    def test_produces_events(self, host):
        rng = np.random.RandomState(123)
        events = office_edits(host, rng)

        assert len(events) >= 3  # at least process + 2 file events
        assert any(isinstance(e, ProcessEvent) for e in events)
        assert any(isinstance(e, FileEvent) for e in events)

    def test_spawns_office_process(self, host):
        rng = np.random.RandomState(123)
        events = office_edits(host, rng)

        proc_events = [e for e in events if isinstance(e, ProcessEvent)]
        assert len(proc_events) == 1
        assert proc_events[0].name in ("WINWORD.EXE", "EXCEL.EXE", "POWERPNT.EXE")

    def test_no_extension_changes(self, host):
        rng = np.random.RandomState(123)
        events = office_edits(host, rng)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        for fe in file_events:
            assert fe.extension_change is None


class TestBrowserDownloads:
    def test_produces_file_and_net_events(self, host):
        rng = np.random.RandomState(456)
        events = browser_downloads(host, rng)

        assert any(isinstance(e, FileEvent) for e in events)
        assert any(isinstance(e, NetEvent) for e in events)

    def test_downloads_go_to_downloads_dir(self, host):
        rng = np.random.RandomState(456)
        events = browser_downloads(host, rng)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        for fe in file_events:
            assert "Downloads" in fe.path

    def test_opens_download_connections(self, host):
        rng = np.random.RandomState(456)
        events = browser_downloads(host, rng)

        # Should have opened connections for downloads
        conns = host.connections.all_connections()
        # Includes seeded connections (3) plus download connections
        assert len(conns) >= 4

    def test_network_is_inbound(self, host):
        rng = np.random.RandomState(456)
        events = browser_downloads(host, rng)

        net_events = [e for e in events if isinstance(e, NetEvent)]
        for ne in net_events:
            assert ne.direction == "inbound"


class TestBackupOperations:
    def test_touches_modest_number_of_files(self, host):
        """Post-Phase 6: backup touches 3-8 files (was 8-20) to avoid
        teaching an inverted 'many events = benign' heuristic."""
        rng = np.random.RandomState(789)
        events = backup_operations(host, rng)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert 3 <= len(file_events) <= 9  # 3-8 reads + optional 1 archive write

    def test_zero_delta_reads(self, host):
        rng = np.random.RandomState(789)
        events = backup_operations(host, rng)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        zero_delta = [fe for fe in file_events
                      if fe.size_delta == 0 and fe.entropy_delta == 0.0]
        assert len(zero_delta) >= 3


class TestAvScan:
    def test_moderate_read_count(self, host):
        """Post-Phase 6: av_scan touches 4-9 files (was 10-30)."""
        rng = np.random.RandomState(101)
        events = av_scan(host, rng)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert 4 <= len(file_events) <= 10

    def test_no_modifications(self, host):
        rng = np.random.RandomState(101)
        events = av_scan(host, rng)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        for fe in file_events:
            assert fe.size_delta == 0
            assert fe.entropy_delta == 0.0

    def test_logs_scan_complete_event(self, host):
        rng = np.random.RandomState(101)
        events = av_scan(host, rng)

        evtlog_events = [e for e in events if isinstance(e, EventLogEvent)]
        assert any(e.event_id == 1001 for e in evtlog_events)

    def test_spawns_defender(self, host):
        rng = np.random.RandomState(101)
        events = av_scan(host, rng)

        proc_events = [e for e in events if isinstance(e, ProcessEvent)]
        assert any(pe.name == "MsMpEng.exe" for pe in proc_events)


class TestSystemMaintenance:
    def test_creates_temp_files(self, host):
        initial_count = len(host.files)
        rng = np.random.RandomState(202)
        events = system_maintenance(host, rng)

        assert len(host.files) > initial_count
        file_events = [e for e in events if isinstance(e, FileEvent)]
        for fe in file_events:
            assert "Temp" in fe.path or "tmp" in fe.path.lower()


# ── Attack generators ────────────────────────────────────────────────


class TestBlitzEncryptor:
    def test_encrypts_many_files(self, host):
        initial_unencrypted = len(host.files.unencrypted_files())
        rng = np.random.RandomState(42)

        events, phase = blitz_encryptor(host, rng, progress=0.8)

        assert phase == "encryption"
        remaining = len(host.files.unencrypted_files())
        encrypted_count = initial_unencrypted - remaining
        assert encrypted_count >= 5  # should encrypt many files

    def test_high_entropy_in_events(self, host):
        rng = np.random.RandomState(42)
        events, _ = blitz_encryptor(host, rng, progress=0.8)

        file_events = [e for e in events if isinstance(e, FileEvent)
                       and e.extension_change is not None]
        # Encryption events should show positive entropy delta
        for fe in file_events:
            assert fe.entropy_delta > 0.0

    def test_extension_changes(self, host):
        rng = np.random.RandomState(42)
        events, _ = blitz_encryptor(host, rng, progress=0.8)

        file_events = [e for e in events if isinstance(e, FileEvent)
                       and e.extension_change is not None]
        assert len(file_events) > 0
        for fe in file_events:
            assert ".locked" in fe.extension_change or \
                   ".encrypted" in fe.extension_change or \
                   ".enc" in fe.extension_change or \
                   ".cry" in fe.extension_change

    def test_ransom_note_at_high_progress(self, host):
        rng = np.random.RandomState(42)
        events, _ = blitz_encryptor(host, rng, progress=0.9)

        paths = [e.path for e in events if isinstance(e, FileEvent)]
        assert any("README_DECRYPT" in p for p in paths)

    def test_early_progress_is_recon(self, host):
        rng = np.random.RandomState(42)
        events, phase = blitz_encryptor(host, rng, progress=0.05)

        assert phase == "reconnaissance"
        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert len(file_events) == 0

    def test_disables_defender_via_registry(self, host):
        rng = np.random.RandomState(42)
        events, _ = blitz_encryptor(host, rng, progress=0.8)

        reg_events = [e for e in events if isinstance(e, RegistryEvent)]
        assert len(reg_events) >= 1
        assert any("DisableAntiSpyware" in e.value_name for e in reg_events)

    def test_logs_defender_disabled_event(self, host):
        rng = np.random.RandomState(42)
        events, _ = blitz_encryptor(host, rng, progress=0.8)

        evtlog_events = [e for e in events if isinstance(e, EventLogEvent)]
        assert any(e.event_id == 5001 for e in evtlog_events)

    def test_opens_file_handles(self, host):
        rng = np.random.RandomState(42)
        events, _ = blitz_encryptor(host, rng, progress=0.5)

        proc_events = [e for e in events if isinstance(e, ProcessEvent)]
        mal_pid = proc_events[0].pid
        proc = host.processes.get_process(mal_pid)
        assert len(proc.open_file_handles) > 0

    def test_blitz_more_than_sleeper(self):
        now = datetime(2025, 6, 15, 10, 0, 0)
        host1 = HostState.create(np.random.RandomState(42), now)
        rng1 = np.random.RandomState(42)
        blitz_events, _ = blitz_encryptor(host1, rng1, progress=0.5)

        host2 = HostState.create(np.random.RandomState(42), now)
        rng2 = np.random.RandomState(42)
        sleeper_events, _ = slow_sleeper(host2, rng2, progress=0.5)

        blitz_file_events = [e for e in blitz_events if isinstance(e, FileEvent)
                             and e.extension_change is not None]
        sleeper_file_events = [e for e in sleeper_events if isinstance(e, FileEvent)
                               and e.extension_change is not None]
        assert len(blitz_file_events) > len(sleeper_file_events)


class TestSlowSleeper:
    def test_encrypts_few_files(self, host):
        rng = np.random.RandomState(42)
        events, phase = slow_sleeper(host, rng, progress=0.5)

        assert phase == "encryption"
        file_events = [e for e in events if isinstance(e, FileEvent)
                       and e.extension_change is not None]
        assert 1 <= len(file_events) <= 2

    def test_loads_suspicious_module(self, host):
        rng = np.random.RandomState(42)
        events, _ = slow_sleeper(host, rng, progress=0.5)

        proc_events = [e for e in events if isinstance(e, ProcessEvent)]
        mal_pid = proc_events[0].pid
        proc = host.processes.get_process(mal_pid)
        assert "beacon_x64.dll" in proc.loaded_modules

    def test_uses_legit_process_names(self, host):
        rng = np.random.RandomState(42)
        events, _ = slow_sleeper(host, rng, progress=0.5)

        proc_events = [e for e in events if isinstance(e, ProcessEvent)]
        legit_names = {"SearchIndexer.exe", "RuntimeBroker.exe", "taskhostw.exe"}
        assert all(pe.name in legit_names for pe in proc_events)


class TestExfilFirst:
    def test_exfil_phase_no_encryption(self, host):
        initial_unencrypted = len(host.files.unencrypted_files())
        rng = np.random.RandomState(42)

        events, phase = exfil_first(host, rng, progress=0.3)

        assert phase == "exfiltration"
        assert len(host.files.unencrypted_files()) == initial_unencrypted

    def test_exfil_opens_connections(self, host):
        rng = np.random.RandomState(42)
        events, phase = exfil_first(host, rng, progress=0.3)

        # Should have opened connections for C2 traffic
        conns = host.connections.all_connections()
        c2_conns = [c for c in conns if c.remote_port == 443
                    and c.remote_address in ("185.142.98.12", "91.215.85.47",
                                              "45.153.241.81")]
        assert len(c2_conns) >= 2

    def test_exfil_phase_has_network_events(self, host):
        rng = np.random.RandomState(42)
        events, phase = exfil_first(host, rng, progress=0.3)

        net_events = [e for e in events if isinstance(e, NetEvent)]
        assert len(net_events) >= 2
        for ne in net_events:
            assert ne.direction == "outbound"
            assert ne.bytes_transferred >= 100_000

    def test_encryption_phase(self, host):
        rng = np.random.RandomState(42)
        events, phase = exfil_first(host, rng, progress=0.8)

        assert phase == "encryption"
        file_events = [e for e in events if isinstance(e, FileEvent)
                       and e.extension_change is not None]
        assert len(file_events) > 0


class TestSemanticShuffle:
    def test_no_extension_changes(self, host):
        rng = np.random.RandomState(42)
        events, phase = semantic_shuffle(host, rng, progress=0.5)

        assert phase == "manipulation"
        file_events = [e for e in events if isinstance(e, FileEvent)]
        for fe in file_events:
            assert fe.extension_change is None

    def test_moderate_entropy(self, host):
        rng = np.random.RandomState(42)
        events, _ = semantic_shuffle(host, rng, progress=0.5)

        for path in host.files.all_paths():
            f = host.files.get_file(path)
            if f.is_encrypted:
                assert f.entropy <= 6.5

    def test_marks_ground_truth_encrypted(self, host):
        rng = np.random.RandomState(42)
        events, _ = semantic_shuffle(host, rng, progress=0.5)

        all_files_list = [host.files.get_file(p) for p in host.files.all_paths()]
        encrypted = [f for f in all_files_list if f.is_encrypted]
        assert len(encrypted) > 0
