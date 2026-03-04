"""Tests for simulator data models."""

from datetime import datetime

from simulator.models import (
    ContentType,
    FileRecord,
    ProcessRecord,
    ProcessEvent,
    FileEvent,
    NetEvent,
    ScenarioType,
    Verdict,
    GroundTruth,
)


class TestFileRecord:
    def test_basic_creation(self):
        now = datetime(2025, 1, 1)
        r = FileRecord(
            path="C:/test.docx", size=1024, entropy=4.2,
            extension=".docx", modified_at=now, content_type=ContentType.DOC,
        )
        assert r.path == "C:/test.docx"
        assert r.is_encrypted is False
        assert r.entropy_delta == 0.0
        assert r.size_delta == 0
        assert r.extension_changed is False

    def test_deltas_after_mutation(self):
        now = datetime(2025, 1, 1)
        r = FileRecord(
            path="C:/test.docx", size=1024, entropy=4.2,
            extension=".docx", modified_at=now, content_type=ContentType.DOC,
        )
        r.entropy = 7.9
        r.size = 2048
        r.extension = ".docx.locked"

        assert abs(r.entropy_delta - 3.7) < 0.01
        assert r.size_delta == 1024
        assert r.extension_changed is True


class TestProcessRecord:
    def test_basic_creation(self):
        now = datetime(2025, 1, 1)
        p = ProcessRecord(
            pid=1234, name="test.exe", parent_pid=4,
            command_line="test.exe --flag", start_time=now,
        )
        assert p.pid == 1234
        assert p.child_pids == []

    def test_child_tracking(self):
        now = datetime(2025, 1, 1)
        p = ProcessRecord(
            pid=100, name="parent.exe", parent_pid=4,
            command_line="parent.exe", start_time=now,
            child_pids=[200, 300],
        )
        assert 200 in p.child_pids
        assert len(p.child_pids) == 2


class TestTelemetryEvents:
    def test_process_event_type(self):
        e = ProcessEvent(
            ts=datetime(2025, 1, 1), pid=100, name="test.exe",
            parent_pid=4, command_line="test.exe",
        )
        assert e.event_type == "process"
        assert e.drop_probability == 0.0

    def test_file_event_type(self):
        e = FileEvent(
            ts=datetime(2025, 1, 1), path="C:/test.docx",
            size_delta=100, entropy_delta=0.5, extension_change=None,
        )
        assert e.event_type == "file"

    def test_net_event_type(self):
        e = NetEvent(
            ts=datetime(2025, 1, 1), destination="10.0.0.1",
            bytes_transferred=1024, protocol="tcp", direction="outbound",
        )
        assert e.event_type == "network"

    def test_custom_drop_probability(self):
        e = FileEvent(
            ts=datetime(2025, 1, 1), path="C:/test.docx",
            size_delta=0, entropy_delta=0.0, extension_change=None,
            drop_probability=0.5,
        )
        assert e.drop_probability == 0.5


class TestVerdict:
    def test_ransomware_predictions(self):
        assert Verdict.ALERT.is_ransomware_prediction is True
        assert Verdict.QUARANTINE.is_ransomware_prediction is True
        assert Verdict.BLOCK.is_ransomware_prediction is True

    def test_benign_predictions(self):
        assert Verdict.IGNORE.is_ransomware_prediction is False
        assert Verdict.MONITOR.is_ransomware_prediction is False


class TestGroundTruth:
    def test_benign_label(self):
        gt = GroundTruth(
            scenario_type=ScenarioType.BENIGN, is_ransomware=False,
        )
        assert gt.label == "benign"

    def test_attack_label_with_phase(self):
        gt = GroundTruth(
            scenario_type=ScenarioType.EXFIL_FIRST,
            is_ransomware=True,
            attack_phase="exfiltration",
        )
        assert gt.label == "exfil_first:exfiltration"

    def test_attack_label_without_phase(self):
        gt = GroundTruth(
            scenario_type=ScenarioType.BLITZ, is_ransomware=True,
        )
        assert gt.label == "blitz"
