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
from simulator.models import FileEvent, NetEvent, ProcessEvent
from simulator.registry import FileRegistry, ProcessTable


@pytest.fixture
def populated_env():
    """Return a (registry, ptable) with seeded filesystem and processes."""
    rng = np.random.RandomState(42)
    now = datetime(2025, 6, 15, 10, 0, 0)
    registry = FileRegistry()
    registry.seed_filesystem(rng, now)
    ptable = ProcessTable()
    ptable.seed_processes(now)
    return registry, ptable, now


# ── Benign generators ────────────────────────────────────────────────


class TestOfficeEdits:
    def test_produces_events(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(123)
        events = office_edits(registry, ptable, rng, now)

        assert len(events) >= 3  # at least process + 2 file events
        assert any(isinstance(e, ProcessEvent) for e in events)
        assert any(isinstance(e, FileEvent) for e in events)

    def test_spawns_office_process(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(123)
        events = office_edits(registry, ptable, rng, now)

        proc_events = [e for e in events if isinstance(e, ProcessEvent)]
        assert len(proc_events) == 1
        assert proc_events[0].name in ("WINWORD.EXE", "EXCEL.EXE", "POWERPNT.EXE")

    def test_no_extension_changes(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(123)
        events = office_edits(registry, ptable, rng, now)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        for fe in file_events:
            assert fe.extension_change is None


class TestBrowserDownloads:
    def test_produces_file_and_net_events(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(456)
        events = browser_downloads(registry, ptable, rng, now)

        assert any(isinstance(e, FileEvent) for e in events)
        assert any(isinstance(e, NetEvent) for e in events)

    def test_downloads_go_to_downloads_dir(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(456)
        events = browser_downloads(registry, ptable, rng, now)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        for fe in file_events:
            assert "Downloads" in fe.path

    def test_network_is_inbound(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(456)
        events = browser_downloads(registry, ptable, rng, now)

        net_events = [e for e in events if isinstance(e, NetEvent)]
        for ne in net_events:
            assert ne.direction == "inbound"


class TestBackupOperations:
    def test_touches_many_files(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(789)
        events = backup_operations(registry, ptable, rng, now)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert len(file_events) >= 8  # high file count

    def test_zero_delta_reads(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(789)
        events = backup_operations(registry, ptable, rng, now)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        # Most should be zero-delta reads (backup archive may have non-zero)
        zero_delta = [fe for fe in file_events
                      if fe.size_delta == 0 and fe.entropy_delta == 0.0]
        assert len(zero_delta) >= 8


class TestAvScan:
    def test_high_read_count(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(101)
        events = av_scan(registry, ptable, rng, now)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert len(file_events) >= 10

    def test_no_modifications(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(101)
        events = av_scan(registry, ptable, rng, now)

        file_events = [e for e in events if isinstance(e, FileEvent)]
        for fe in file_events:
            assert fe.size_delta == 0
            assert fe.entropy_delta == 0.0

    def test_spawns_defender(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(101)
        events = av_scan(registry, ptable, rng, now)

        proc_events = [e for e in events if isinstance(e, ProcessEvent)]
        assert any(pe.name == "MsMpEng.exe" for pe in proc_events)


class TestSystemMaintenance:
    def test_creates_temp_files(self, populated_env):
        registry, ptable, now = populated_env
        initial_count = len(registry)
        rng = np.random.RandomState(202)
        events = system_maintenance(registry, ptable, rng, now)

        assert len(registry) > initial_count
        file_events = [e for e in events if isinstance(e, FileEvent)]
        for fe in file_events:
            assert "Temp" in fe.path or "tmp" in fe.path.lower()


# ── Attack generators ────────────────────────────────────────────────


class TestBlitzEncryptor:
    def test_encrypts_many_files(self, populated_env):
        registry, ptable, now = populated_env
        initial_unencrypted = len(registry.unencrypted_files())
        rng = np.random.RandomState(42)

        events, phase = blitz_encryptor(registry, ptable, rng, now, progress=0.8)

        assert phase == "encryption"
        remaining = len(registry.unencrypted_files())
        encrypted_count = initial_unencrypted - remaining
        assert encrypted_count >= 5  # should encrypt many files

    def test_high_entropy_in_events(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, _ = blitz_encryptor(registry, ptable, rng, now, progress=0.8)

        file_events = [e for e in events if isinstance(e, FileEvent)
                       and e.extension_change is not None]
        # Encryption events should show positive entropy delta
        # (images already have high entropy ~6.5-7.2, so delta may be small)
        for fe in file_events:
            assert fe.entropy_delta > 0.0  # entropy increases on encryption

    def test_extension_changes(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, _ = blitz_encryptor(registry, ptable, rng, now, progress=0.8)

        file_events = [e for e in events if isinstance(e, FileEvent)
                       and e.extension_change is not None]
        assert len(file_events) > 0
        for fe in file_events:
            assert ".locked" in fe.extension_change or \
                   ".encrypted" in fe.extension_change or \
                   ".enc" in fe.extension_change or \
                   ".cry" in fe.extension_change

    def test_ransom_note_at_high_progress(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, _ = blitz_encryptor(registry, ptable, rng, now, progress=0.9)

        paths = [e.path for e in events if isinstance(e, FileEvent)]
        assert any("README_DECRYPT" in p for p in paths)

    def test_early_progress_is_recon(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, phase = blitz_encryptor(registry, ptable, rng, now, progress=0.05)

        assert phase == "reconnaissance"
        # Only process event, no encryption
        file_events = [e for e in events if isinstance(e, FileEvent)]
        assert len(file_events) == 0

    def test_blitz_more_than_sleeper(self, populated_env):
        registry1, ptable1, now = populated_env
        rng1 = np.random.RandomState(42)
        blitz_events, _ = blitz_encryptor(registry1, ptable1, rng1, now, progress=0.5)

        # Fresh env for sleeper
        rng2 = np.random.RandomState(42)
        registry2 = FileRegistry()
        registry2.seed_filesystem(rng2, now)
        ptable2 = ProcessTable()
        ptable2.seed_processes(now)
        rng2 = np.random.RandomState(42)
        sleeper_events, _ = slow_sleeper(registry2, ptable2, rng2, now, progress=0.5)

        blitz_file_events = [e for e in blitz_events if isinstance(e, FileEvent)
                             and e.extension_change is not None]
        sleeper_file_events = [e for e in sleeper_events if isinstance(e, FileEvent)
                               and e.extension_change is not None]
        assert len(blitz_file_events) > len(sleeper_file_events)


class TestSlowSleeper:
    def test_encrypts_few_files(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, phase = slow_sleeper(registry, ptable, rng, now, progress=0.5)

        assert phase == "encryption"
        file_events = [e for e in events if isinstance(e, FileEvent)
                       and e.extension_change is not None]
        assert 1 <= len(file_events) <= 2  # 1-2 files per window

    def test_uses_legit_process_names(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, _ = slow_sleeper(registry, ptable, rng, now, progress=0.5)

        proc_events = [e for e in events if isinstance(e, ProcessEvent)]
        legit_names = {"SearchIndexer.exe", "RuntimeBroker.exe", "taskhostw.exe"}
        assert all(pe.name in legit_names for pe in proc_events)


class TestExfilFirst:
    def test_exfil_phase_no_encryption(self, populated_env):
        registry, ptable, now = populated_env
        initial_unencrypted = len(registry.unencrypted_files())
        rng = np.random.RandomState(42)

        events, phase = exfil_first(registry, ptable, rng, now, progress=0.3)

        assert phase == "exfiltration"
        # No files encrypted during exfil phase
        assert len(registry.unencrypted_files()) == initial_unencrypted

    def test_exfil_phase_has_network_events(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, phase = exfil_first(registry, ptable, rng, now, progress=0.3)

        net_events = [e for e in events if isinstance(e, NetEvent)]
        assert len(net_events) >= 2
        for ne in net_events:
            assert ne.direction == "outbound"
            assert ne.bytes_transferred >= 100_000

    def test_encryption_phase(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, phase = exfil_first(registry, ptable, rng, now, progress=0.8)

        assert phase == "encryption"
        file_events = [e for e in events if isinstance(e, FileEvent)
                       and e.extension_change is not None]
        assert len(file_events) > 0


class TestSemanticShuffle:
    def test_no_extension_changes(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, phase = semantic_shuffle(registry, ptable, rng, now, progress=0.5)

        assert phase == "manipulation"
        file_events = [e for e in events if isinstance(e, FileEvent)]
        for fe in file_events:
            assert fe.extension_change is None

    def test_moderate_entropy(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, _ = semantic_shuffle(registry, ptable, rng, now, progress=0.5)

        # All modified files should stay below encryption threshold
        for path in registry.all_paths():
            f = registry.get_file(path)
            if f.is_encrypted:  # semantically "encrypted" but low entropy
                assert f.entropy <= 6.5

    def test_marks_ground_truth_encrypted(self, populated_env):
        registry, ptable, now = populated_env
        rng = np.random.RandomState(42)
        events, _ = semantic_shuffle(registry, ptable, rng, now, progress=0.5)

        # Some files should be marked as encrypted in ground truth
        encrypted = [f for f in registry.unencrypted_files()
                     if False]  # unencrypted only returns non-encrypted
        # Check that is_encrypted was set
        all_files_list = [registry.get_file(p) for p in registry.all_paths()]
        encrypted = [f for f in all_files_list if f.is_encrypted]
        assert len(encrypted) > 0
