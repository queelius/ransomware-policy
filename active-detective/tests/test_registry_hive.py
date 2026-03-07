"""Tests for RegistryHive."""

from datetime import datetime

import pytest

from simulator.registry_hive import RegistryHive


class TestRegistryHive:
    @pytest.fixture
    def hive(self):
        return RegistryHive()

    def test_set_and_get_value(self, hive):
        now = datetime(2025, 1, 1)
        hive.set_value(r"HKLM\SOFTWARE\Test", "Name", "Value", "REG_SZ", now)
        val = hive.get_value(r"HKLM\SOFTWARE\Test", "Name")
        assert val is not None
        assert val.data == "Value"
        assert val.value_type == "REG_SZ"

    def test_get_missing_key(self, hive):
        assert hive.get_value(r"HKLM\MISSING", "Name") is None

    def test_get_missing_value(self, hive):
        now = datetime(2025, 1, 1)
        hive.set_value(r"HKLM\SOFTWARE\Test", "Name", "Value", now=now)
        assert hive.get_value(r"HKLM\SOFTWARE\Test", "Missing") is None

    def test_overwrite_value(self, hive):
        now = datetime(2025, 1, 1)
        hive.set_value(r"HKLM\SOFTWARE\Test", "Name", "Old", now=now)
        hive.set_value(r"HKLM\SOFTWARE\Test", "Name", "New", now=now)
        val = hive.get_value(r"HKLM\SOFTWARE\Test", "Name")
        assert val.data == "New"

    def test_delete_key(self, hive):
        now = datetime(2025, 1, 1)
        hive.set_value(r"HKLM\SOFTWARE\Test", "Name", "Value", now=now)
        deleted = hive.delete_key(r"HKLM\SOFTWARE\Test")
        assert deleted is not None
        assert len(hive) == 0

    def test_delete_missing_key(self, hive):
        assert hive.delete_key(r"HKLM\MISSING") is None

    def test_enumerate_keys(self, hive):
        now = datetime(2025, 1, 1)
        hive.set_value(r"HKLM\SOFTWARE\Test\A", "V", "1", now=now)
        hive.set_value(r"HKLM\SOFTWARE\Test\B", "V", "2", now=now)
        hive.set_value(r"HKLM\SOFTWARE\Other", "V", "3", now=now)
        keys = hive.enumerate_keys(r"HKLM\SOFTWARE\Test")
        assert len(keys) == 2
        assert r"HKLM\SOFTWARE\Test\A" in keys
        assert r"HKLM\SOFTWARE\Test\B" in keys

    def test_get_key(self, hive):
        now = datetime(2025, 1, 1)
        hive.set_value(r"HKLM\SOFTWARE\Test", "A", "1", now=now)
        hive.set_value(r"HKLM\SOFTWARE\Test", "B", "2", now=now)
        key = hive.get_key(r"HKLM\SOFTWARE\Test")
        assert key is not None
        assert len(key.values) == 2

    def test_add_run_key(self, hive):
        now = datetime(2025, 1, 1)
        hive.add_run_key("Malware", "C:\\malware.exe", now)
        val = hive.get_value(
            r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
            "Malware")
        assert val is not None
        assert val.data == "C:\\malware.exe"

    def test_disable_defender(self, hive):
        now = datetime(2025, 1, 1)
        hive.disable_defender(now)
        val = hive.get_value(
            r"HKLM\SOFTWARE\Policies\Microsoft\Windows Defender\Real-Time Protection",
            "DisableRealtimeMonitoring")
        assert val is not None
        assert val.data == "1"

    def test_disable_vss(self, hive):
        now = datetime(2025, 1, 1)
        hive.disable_vss(now)
        val = hive.get_value(
            r"HKLM\SYSTEM\CurrentControlSet\Services\VSS",
            "Start")
        assert val.data == "4"

    def test_seed_registry(self, hive):
        now = datetime(2025, 1, 1)
        hive.seed_registry(now)
        assert len(hive) == 5
        # Defender should be enabled by default
        val = hive.get_value(
            r"HKLM\SOFTWARE\Policies\Microsoft\Windows Defender\Real-Time Protection",
            "DisableRealtimeMonitoring")
        assert val.data == "0"

    def test_len(self, hive):
        assert len(hive) == 0
        now = datetime(2025, 1, 1)
        hive.set_value(r"HKLM\A", "V", "1", now=now)
        hive.set_value(r"HKLM\B", "V", "2", now=now)
        assert len(hive) == 2
