"""Tests for FileRegistry and ProcessTable."""

from datetime import datetime

import numpy as np
import pytest

from simulator.models import ContentType
from simulator.registry import FileRegistry, ProcessTable


class TestFileRegistry:
    @pytest.fixture
    def registry(self):
        r = FileRegistry()
        now = datetime(2025, 1, 1)
        r.add_file("C:/Users/A/Documents/report.docx", 1024, 4.2, ".docx",
                    ContentType.DOC, now)
        r.add_file("C:/Users/A/Documents/budget.xlsx", 2048, 3.8, ".xlsx",
                    ContentType.DOC, now)
        r.add_file("C:/Users/A/Desktop/photo.jpg", 500_000, 6.5, ".jpg",
                    ContentType.IMAGE, now)
        return r

    def test_len(self, registry):
        assert len(registry) == 3

    def test_contains(self, registry):
        assert "C:/Users/A/Documents/report.docx" in registry
        assert "C:/nonexistent.txt" not in registry

    def test_get_file(self, registry):
        f = registry.get_file("C:/Users/A/Documents/report.docx")
        assert f is not None
        assert f.size == 1024
        assert f.entropy == 4.2
        assert f.is_encrypted is False

    def test_get_file_missing(self, registry):
        assert registry.get_file("C:/nope.txt") is None

    def test_list_directory(self, registry):
        docs = registry.list_directory("C:/Users/A/Documents")
        assert len(docs) == 2
        paths = {f.path for f in docs}
        assert "C:/Users/A/Documents/report.docx" in paths
        assert "C:/Users/A/Documents/budget.xlsx" in paths

    def test_list_directory_empty(self, registry):
        assert registry.list_directory("C:/Nonexistent") == []

    def test_directories(self, registry):
        dirs = registry.directories()
        assert "C:/Users/A/Documents" in dirs
        assert "C:/Users/A/Desktop" in dirs

    def test_encrypt_file(self, registry):
        now = datetime(2025, 1, 1, 0, 5)
        rng = np.random.RandomState(42)
        result = registry.encrypt_file(
            "C:/Users/A/Documents/report.docx", now, rng=rng,
        )
        assert result is not None
        assert result.is_encrypted is True
        assert result.entropy >= 7.5
        assert result.extension.endswith(".locked")
        assert result.modified_at == now

        # Old path removed, new path exists
        assert "C:/Users/A/Documents/report.docx" not in registry
        assert "C:/Users/A/Documents/report.docx.locked" in registry

    def test_encrypt_already_encrypted(self, registry):
        now = datetime(2025, 1, 1, 0, 5)
        rng = np.random.RandomState(42)
        registry.encrypt_file("C:/Users/A/Documents/report.docx", now, rng=rng)
        # Try encrypting the new path again
        result = registry.encrypt_file(
            "C:/Users/A/Documents/report.docx.locked", now, rng=rng,
        )
        assert result is None  # already encrypted

    def test_encrypt_missing_file(self, registry):
        now = datetime(2025, 1, 1)
        result = registry.encrypt_file("C:/nope.txt", now)
        assert result is None

    def test_modify_file(self, registry):
        now = datetime(2025, 1, 1, 0, 5)
        result = registry.modify_file(
            "C:/Users/A/Documents/report.docx", now,
            size_delta=512, entropy_delta=0.3,
        )
        assert result is not None
        assert result.size == 1024 + 512
        assert abs(result.entropy - 4.5) < 0.01

    def test_modify_clamps_entropy(self, registry):
        now = datetime(2025, 1, 1, 0, 5)
        result = registry.modify_file(
            "C:/Users/A/Documents/report.docx", now,
            entropy_delta=10.0,  # would exceed 8.0
        )
        assert result.entropy == 8.0

    def test_rename_file(self, registry):
        result = registry.rename_file(
            "C:/Users/A/Desktop/photo.jpg",
            "C:/Users/A/Desktop/vacation.jpg",
        )
        assert result is not None
        assert result.path == "C:/Users/A/Desktop/vacation.jpg"
        assert "C:/Users/A/Desktop/photo.jpg" not in registry
        assert "C:/Users/A/Desktop/vacation.jpg" in registry

    def test_rename_missing(self, registry):
        assert registry.rename_file("C:/nope.txt", "C:/still_nope.txt") is None

    def test_unencrypted_files(self, registry):
        assert len(registry.unencrypted_files()) == 3
        rng = np.random.RandomState(42)
        registry.encrypt_file(
            "C:/Users/A/Documents/report.docx",
            datetime(2025, 1, 1), rng=rng,
        )
        assert len(registry.unencrypted_files()) == 2

    def test_seed_filesystem(self):
        registry = FileRegistry()
        rng = np.random.RandomState(42)
        now = datetime(2025, 1, 1)
        registry.seed_filesystem(rng, now)

        # Should have a reasonable number of files
        assert len(registry) >= 24   # sum of all min_counts
        assert len(registry) <= 64   # sum of all max_counts

        # All files should have valid entropy and size
        for path in registry.all_paths():
            f = registry.get_file(path)
            assert 0.0 <= f.entropy <= 8.0
            assert f.size > 0
            assert f.content_type in ContentType

    def test_seed_filesystem_reproducible(self):
        r1 = FileRegistry()
        r2 = FileRegistry()
        now = datetime(2025, 1, 1)
        r1.seed_filesystem(np.random.RandomState(42), now)
        r2.seed_filesystem(np.random.RandomState(42), now)
        assert r1.all_paths() == r2.all_paths()


class TestProcessTable:
    @pytest.fixture
    def ptable(self):
        pt = ProcessTable()
        pt.seed_processes(datetime(2025, 1, 1))
        return pt

    def test_seed_processes(self, ptable):
        # System process + 9 base processes
        assert len(ptable) == 10
        assert ptable.get_process(4) is not None  # System
        assert ptable.get_process(4).name == "System"

    def test_spawn_process(self, ptable):
        now = datetime(2025, 1, 1, 0, 5)
        p = ptable.spawn_process(
            "malware.exe", parent_pid=4,
            command_line="malware.exe --encrypt", start_time=now,
        )
        assert p.pid >= 1000
        assert p.name == "malware.exe"
        assert ptable.get_process(p.pid) is not None

        # Should be in parent's child list
        assert p.pid in ptable.get_children(4)

    def test_kill_process(self, ptable):
        now = datetime(2025, 1, 1, 0, 5)
        p = ptable.spawn_process("test.exe", 4, "test.exe", now)
        pid = p.pid

        killed = ptable.kill_process(pid)
        assert killed is not None
        assert killed.name == "test.exe"
        assert ptable.get_process(pid) is None
        assert pid not in ptable.get_children(4)

    def test_kill_missing(self, ptable):
        assert ptable.kill_process(99999) is None

    def test_get_children(self, ptable):
        now = datetime(2025, 1, 1, 0, 5)
        parent = ptable.spawn_process("parent.exe", 4, "parent.exe", now)
        child1 = ptable.spawn_process("child1.exe", parent.pid, "child1.exe", now)
        child2 = ptable.spawn_process("child2.exe", parent.pid, "child2.exe", now)

        children = ptable.get_children(parent.pid)
        assert child1.pid in children
        assert child2.pid in children
        assert len(children) == 2

    def test_get_children_missing(self, ptable):
        assert ptable.get_children(99999) == []

    def test_all_pids(self, ptable):
        pids = ptable.all_pids()
        assert 4 in pids  # System
        assert len(pids) == 10
