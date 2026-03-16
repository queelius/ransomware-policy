"""Tests for compute_metrics script."""
import json
import tempfile
from pathlib import Path

from scripts.compute_metrics import compute_all_metrics, load_results


def _make_results_file(records):
    """Helper to create a temporary JSONL file with the given records."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


SAMPLE_RESULTS = [
    {
        "seed": 1,
        "scenario_type": "blitz",
        "verdict": "quarantine",
        "correct": True,
        "is_ransomware_gt": True,
        "attack_phase": "encryption",
        "steps_taken": 3,
        "cumulative_cost": -0.09,
        "tools_used": ["inspect_file", "scan_directory", "read_file_sample"],
        "reward": 0.86,
        "k_max": 5,
        "observability": 0.7,
    },
    {
        "seed": 2,
        "scenario_type": "benign",
        "verdict": "ignore",
        "correct": True,
        "is_ransomware_gt": False,
        "attack_phase": None,
        "steps_taken": 1,
        "cumulative_cost": -0.02,
        "tools_used": ["inspect_file"],
        "reward": 1.18,
        "k_max": 5,
        "observability": 0.9,
    },
    {
        "seed": 3,
        "scenario_type": "sleeper",
        "verdict": "ignore",
        "correct": False,
        "is_ransomware_gt": True,
        "attack_phase": "encryption",
        "steps_taken": 2,
        "cumulative_cost": -0.07,
        "tools_used": ["inspect_file", "check_process"],
        "reward": -1.93,
        "k_max": 5,
        "observability": 0.3,
    },
]


class TestLoadResults:
    """Tests for load_results function."""

    def test_loads_jsonl(self):
        """Test that load_results reads a JSONL file correctly."""
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        assert len(results) == 3
        Path(path).unlink()

    def test_returns_eval_results(self):
        """Test that loaded results have correct fields."""
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        assert results[0].verdict == "quarantine"
        assert results[0].is_ransomware_gt is True
        assert results[0].scenario_type == "blitz"
        assert results[0].steps_taken == 3
        Path(path).unlink()

    def test_handles_missing_tools_used(self):
        """Test that load_results defaults tools_used to empty list if missing."""
        records = [
            {
                "verdict": "ignore",
                "is_ransomware_gt": False,
                "scenario_type": "benign",
                "attack_phase": None,
                "steps_taken": 1,
                "cumulative_cost": -0.02,
                "k_max": 5,
            }
        ]
        path = _make_results_file(records)
        results = load_results(path)
        assert results[0].tools_used == []
        Path(path).unlink()


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_returns_detection_and_efficiency(self):
        """Test that compute_all_metrics returns expected top-level keys."""
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        metrics = compute_all_metrics(results)
        assert "detection" in metrics
        assert "efficiency" in metrics
        assert "by_scenario_type" in metrics
        assert "phase_accuracy" in metrics
        assert "tool_usage" in metrics
        assert "n_episodes" in metrics
        Path(path).unlink()

    def test_detection_metrics_correct(self):
        """Test that detection metrics are computed correctly."""
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        metrics = compute_all_metrics(results)
        d = metrics["detection"]
        # Verdict "quarantine" -> predicted ransomware, gt True -> TP
        # Verdict "ignore" on benign -> predicted benign, gt False -> TN
        # Verdict "ignore" on ransomware -> predicted benign, gt True -> FN
        assert d["tp"] == 1  # quarantine + ransomware
        assert d["fp"] == 0  # no false positives
        assert d["tn"] == 1  # ignore + benign
        assert d["fn"] == 1  # ignore + ransomware
        Path(path).unlink()

    def test_by_scenario_type_computed(self):
        """Test that metrics are computed per scenario type."""
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        metrics = compute_all_metrics(results)
        by_type = metrics["by_scenario_type"]
        assert "blitz" in by_type
        assert "benign" in by_type
        assert "sleeper" in by_type
        # Each should have detection fields
        assert "tp" in by_type["blitz"]
        assert "precision" in by_type["blitz"]
        Path(path).unlink()

    def test_tool_usage_aggregated(self):
        """Test that tool usage is aggregated across episodes."""
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        metrics = compute_all_metrics(results)
        tool_use = metrics["tool_usage"]
        # inspect_file used 3 times (in all 3 episodes)
        assert tool_use["inspect_file"] == 3
        # scan_directory used 1 time
        assert tool_use["scan_directory"] == 1
        # check_process used 1 time
        assert tool_use["check_process"] == 1
        Path(path).unlink()

    def test_phase_accuracy_computed(self):
        """Test that phase accuracy is computed."""
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        metrics = compute_all_metrics(results)
        phase_acc = metrics["phase_accuracy"]
        assert "encryption" in phase_acc
        assert "benign" in phase_acc
        Path(path).unlink()

    def test_efficiency_metrics_computed(self):
        """Test that efficiency metrics are computed."""
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        metrics = compute_all_metrics(results)
        e = metrics["efficiency"]
        assert "mean_steps" in e
        assert "mean_cost" in e
        assert "early_stop_rate" in e
        Path(path).unlink()

    def test_n_episodes_counts(self):
        """Test that n_episodes is set correctly."""
        path = _make_results_file(SAMPLE_RESULTS)
        results = load_results(path)
        metrics = compute_all_metrics(results)
        assert metrics["n_episodes"] == 3
        Path(path).unlink()
