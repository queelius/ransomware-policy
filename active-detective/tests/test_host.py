"""Tests for HostState composition."""

from datetime import datetime

import numpy as np
import pytest

from simulator.host import HostState


class TestHostState:
    def test_create_seeds_all_registries(self):
        rng = np.random.RandomState(42)
        host = HostState.create(rng)

        assert len(host.files) > 20
        assert len(host.processes) == 10
        assert len(host.connections) == 3
        assert len(host.registry) == 5
        assert len(host.event_log) == 5
        assert host.clock.elapsed() == 0.0

    def test_create_reproducible(self):
        h1 = HostState.create(np.random.RandomState(42))
        h2 = HostState.create(np.random.RandomState(42))
        assert h1.files.all_paths() == h2.files.all_paths()
        assert h1.processes.all_pids() == h2.processes.all_pids()

    def test_clock_is_consistent(self):
        now = datetime(2025, 7, 1, 12, 0, 0)
        host = HostState.create(np.random.RandomState(42), now=now)
        assert host.clock.now() == now

    def test_custom_now(self):
        now = datetime(2025, 3, 1, 8, 0, 0)
        host = HostState.create(np.random.RandomState(42), now=now)
        # Files should have modified_at <= now
        for path in host.files.all_paths():
            f = host.files.get_file(path)
            assert f.modified_at <= now

    def test_sub_registries_are_mutable(self):
        host = HostState.create(np.random.RandomState(42))
        initial_files = len(host.files)
        host.files.add_file("C:/new.txt", 100, 3.0, ".txt",
                           __import__("simulator.models", fromlist=["ContentType"]).ContentType.DOC,
                           host.clock.now())
        assert len(host.files) == initial_files + 1
