"""Tests for MCP DetectiveSession state management."""
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path
from simulator.models import ScenarioType
from mcp_session import DetectiveSession


class TestSessionInit:
    def test_creates_episode_from_config(self):
        session = DetectiveSession(seed=42, scenario_type="blitz", observability=0.7, attack_progress=0.5)
        assert session.episode is not None
        assert session.episode.scenario_type == ScenarioType.BLITZ
        assert session.steps == 0
        assert session.cumulative_cost == 0.0
        assert session.verdict is None
        assert not session.is_done

    def test_benign_scenario(self):
        session = DetectiveSession(seed=99, scenario_type="benign", observability=0.9)
        assert session.episode.ground_truth.is_ransomware is False

    def test_get_telemetry_returns_formatted_text(self):
        session = DetectiveSession(seed=42, scenario_type="blitz", observability=0.7, attack_progress=0.5)
        telemetry = session.get_telemetry()
        assert isinstance(telemetry, str)
        assert len(telemetry) > 0
        assert "Window t-" in telemetry or "---" in telemetry or "[t+" in telemetry

    def test_get_telemetry_includes_history_windows(self):
        session = DetectiveSession(seed=42, scenario_type="sleeper", observability=0.7, attack_progress=0.5, n_history=2)
        telemetry = session.get_telemetry()
        assert "Window t-2" in telemetry
        assert "Window t-1" in telemetry


class TestSessionToolExecution:
    def test_execute_investigation_tool(self):
        session = DetectiveSession(seed=42, scenario_type="blitz", observability=0.7, attack_progress=0.5)
        result = session.execute_tool("scan_directory", {"path": "C:/Users/A/Documents"})
        assert "files" in result or "error" in result
        assert session.steps == 1
        assert session.cumulative_cost == pytest.approx(-0.05)
        assert session.tools_used == ["scan_directory"]

    def test_execute_multiple_tools(self):
        session = DetectiveSession(seed=42, scenario_type="blitz", observability=0.7, attack_progress=0.5)
        session.execute_tool("scan_directory", {"path": "C:/Users/A/Documents"})
        session.execute_tool("list_connections", {})
        assert session.steps == 2
        assert len(session.tools_used) == 2

    def test_budget_exceeded_returns_error(self):
        session = DetectiveSession(seed=42, scenario_type="blitz", observability=0.7, attack_progress=0.5, k_max=1)
        session.execute_tool("scan_directory", {"path": "C:/Users/A/Documents"})
        result = session.execute_tool("inspect_file", {"path": "C:/Users/A/Documents/x.docx"})
        assert "budget" in str(result).lower() or "error" in result
        assert session.steps == 1  # second call not counted

    def test_tool_after_decide_returns_error(self):
        session = DetectiveSession(seed=42, scenario_type="benign", observability=0.9)
        session.decide("ignore", "Looks clean")
        result = session.execute_tool("scan_directory", {"path": "C:/Users/A/Documents"})
        assert "error" in result


class TestSessionDecide:
    def test_decide_sets_verdict(self):
        session = DetectiveSession(seed=42, scenario_type="benign", observability=0.9)
        result = session.decide("ignore", "No indicators found")
        assert session.verdict == "ignore"
        assert session.is_done
        assert "correct" in result

    def test_decide_computes_reward(self):
        session = DetectiveSession(seed=42, scenario_type="blitz", observability=0.7, attack_progress=0.5)
        session.execute_tool("scan_directory", {"path": "C:/Users/A/Documents"})
        result = session.decide("quarantine", "Encryption detected")
        assert "reward" in result
        assert isinstance(result["reward"], float)

    def test_decide_invalid_verdict(self):
        session = DetectiveSession(seed=42, scenario_type="benign", observability=0.9)
        result = session.decide("nuke_from_orbit", "Just to be safe")
        assert "error" in result
        assert session.verdict is None

    def test_decide_writes_results_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        session = DetectiveSession(seed=42, scenario_type="blitz", observability=0.7, attack_progress=0.5, results_file=path)
        session.decide("quarantine", "Encryption detected")
        with open(path) as f:
            line = json.loads(f.readline())
        assert line["seed"] == 42
        assert line["scenario_type"] == "blitz"
        assert line["verdict"] == "quarantine"
        assert "reward" in line
        assert "tools_used" in line
        assert "steps_taken" in line
        Path(path).unlink()

    def test_double_decide_errors(self):
        session = DetectiveSession(seed=42, scenario_type="benign", observability=0.9)
        session.decide("ignore", "Clean")
        result = session.decide("alert", "Wait actually...")
        assert "error" in result
