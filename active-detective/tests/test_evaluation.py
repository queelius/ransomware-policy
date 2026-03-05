"""Tests for evaluation framework: metrics, baselines, and ablation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from evaluation.ablation import (
    ABLATION_VARIANTS,
    AblationResult,
    format_ablation_table,
    run_ablation_sweep,
    save_ablation_results,
)
from evaluation.baselines import (
    ExhaustiveAgent,
    HeuristicAgent,
    RandomAgent,
    evaluate_baseline,
    _extract_entropy_deltas,
    _count_extension_changes,
)
from evaluation.metrics import (
    DetectionMetrics,
    EfficiencyMetrics,
    EvalResult,
    ParetoPoint,
    compute_detection_metrics,
    compute_efficiency_metrics,
    compute_pareto_frontier,
    compute_phase_accuracy,
    compute_tool_usage,
)
from simulator.models import ScenarioType
from training.scenarios import generate_training_scenarios


# ── Metrics tests ────────────────────────────────────────────────────


class TestDetectionMetrics:
    def test_perfect_detection(self):
        results = [
            EvalResult("alert", True, "blitz", "encryption", 2, -0.04, 5),
            EvalResult("quarantine", True, "sleeper", "encryption", 3, -0.06, 5),
            EvalResult("ignore", False, "benign", None, 1, 0.0, 5),
            EvalResult("monitor", False, "benign", None, 1, 0.0, 5),
        ]
        m = compute_detection_metrics(results)
        assert m.tp == 2
        assert m.tn == 2
        assert m.fp == 0
        assert m.fn == 0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.accuracy == 1.0

    def test_all_false_negatives(self):
        results = [
            EvalResult("ignore", True, "blitz", "encryption", 1, 0.0, 5),
            EvalResult("monitor", True, "sleeper", "encryption", 1, 0.0, 5),
        ]
        m = compute_detection_metrics(results)
        assert m.fn == 2
        assert m.tp == 0
        assert m.recall == 0.0

    def test_all_false_positives(self):
        results = [
            EvalResult("alert", False, "benign", None, 1, 0.0, 5),
            EvalResult("quarantine", False, "benign", None, 1, 0.0, 5),
        ]
        m = compute_detection_metrics(results)
        assert m.fp == 2
        assert m.tn == 0
        assert m.precision == 0.0

    def test_empty_results(self):
        m = compute_detection_metrics([])
        assert m.accuracy == 0.0
        assert m.f1 == 0.0

    def test_invalid_verdict(self):
        results = [
            EvalResult("invalid", True, "blitz", "encryption", 1, 0.0, 5),
        ]
        m = compute_detection_metrics(results)
        assert m.fn == 1  # invalid treated as benign prediction

    def test_to_dict(self):
        m = DetectionMetrics(tp=5, fp=2, tn=8, fn=1)
        d = m.to_dict()
        assert d["tp"] == 5
        assert "precision" in d
        assert "f1" in d


class TestEfficiencyMetrics:
    def test_basic_efficiency(self):
        results = [
            EvalResult("ignore", False, "benign", None, 1, 0.0, 5),
            EvalResult("alert", True, "blitz", "encryption", 3, -0.06, 5),
            EvalResult("ignore", False, "benign", None, 5, -0.10, 5),
        ]
        em = compute_efficiency_metrics(results)
        assert em.total_episodes == 3
        assert em.mean_steps == 3.0  # (1+3+5)/3
        assert em.early_stops == 2  # steps < 5

    def test_all_early_stops(self):
        results = [
            EvalResult("ignore", False, "benign", None, 1, 0.0, 5),
            EvalResult("ignore", False, "benign", None, 2, 0.0, 5),
        ]
        em = compute_efficiency_metrics(results)
        assert em.early_stop_rate == 1.0

    def test_no_early_stops(self):
        results = [
            EvalResult("ignore", False, "benign", None, 5, -0.10, 5),
        ]
        em = compute_efficiency_metrics(results)
        assert em.early_stop_rate == 0.0

    def test_to_dict(self):
        em = EfficiencyMetrics(total_episodes=10, total_steps=30,
                               total_cost=-0.6, early_stops=7)
        d = em.to_dict()
        assert d["mean_steps"] == 3.0
        assert d["early_stop_rate"] == 0.7


class TestPhaseAccuracy:
    def test_per_phase(self):
        results = [
            EvalResult("alert", True, "blitz", "encryption", 2, -0.04, 5),
            EvalResult("ignore", True, "blitz", "encryption", 1, 0.0, 5),
            EvalResult("ignore", False, "benign", None, 1, 0.0, 5),
            EvalResult("alert", True, "exfil_first", "exfiltration", 2, -0.04, 5),
        ]
        pa = compute_phase_accuracy(results)
        assert pa["encryption"] == 0.5  # 1 of 2 correct
        assert pa["benign"] == 1.0
        assert pa["exfiltration"] == 1.0


class TestToolUsage:
    def test_counts_all_tools(self):
        results = [
            EvalResult("ignore", False, "benign", None, 2, -0.04, 5,
                       tools_used=["check_process", "DECIDE"]),
            EvalResult("alert", True, "blitz", "encryption", 3, -0.06, 5,
                       tools_used=["scan_directory", "inspect_file", "DECIDE"]),
        ]
        usage = compute_tool_usage(results)
        assert usage["check_process"] == 1
        assert usage["scan_directory"] == 1
        assert usage["DECIDE"] == 2


class TestParetoFrontier:
    def test_simple_pareto(self):
        named = {
            "cheap_bad": [
                EvalResult("ignore", False, "benign", None, 1, 0.0, 5),
                EvalResult("ignore", True, "blitz", None, 1, 0.0, 5),
            ],
            "expensive_good": [
                EvalResult("ignore", False, "benign", None, 3, -0.06, 5),
                EvalResult("alert", True, "blitz", None, 3, -0.06, 5),
            ],
        }
        pareto = compute_pareto_frontier(named)
        assert len(pareto) >= 1
        # The expensive_good config should be on the frontier
        names = [p.config_name for p in pareto]
        assert "expensive_good" in names


# ── Baseline tests ───────────────────────────────────────────────────


@pytest.fixture
def small_scenarios():
    batch = generate_training_scenarios(n_episodes=10, seed=42)
    return [
        {
            "scenario_type": ep.scenario_type.value,
            "observability": ep.observability,
        }
        for ep in batch.episodes
    ]


class TestRandomAgent:
    def test_produces_results(self, small_scenarios):
        agent = RandomAgent(k_max=3)
        results = evaluate_baseline(agent, small_scenarios, k_max=3, seed=42)
        assert len(results) == 10
        for r in results:
            assert r.verdict in {"ignore", "monitor", "alert",
                                  "quarantine", "block"}

    def test_reproducible(self, small_scenarios):
        agent = RandomAgent(k_max=3)
        r1 = evaluate_baseline(agent, small_scenarios, k_max=3, seed=42)
        r2 = evaluate_baseline(agent, small_scenarios, k_max=3, seed=42)
        for a, b in zip(r1, r2):
            assert a.verdict == b.verdict


class TestExhaustiveAgent:
    def test_uses_max_steps(self, small_scenarios):
        agent = ExhaustiveAgent(k_max=5)
        results = evaluate_baseline(agent, small_scenarios, k_max=5, seed=42)
        assert len(results) == 10
        for r in results:
            # Should use k_max steps (k-1 investigation + 1 DECIDE)
            assert r.steps_taken == 5

    def test_produces_valid_verdicts(self, small_scenarios):
        agent = ExhaustiveAgent(k_max=3)
        results = evaluate_baseline(agent, small_scenarios, k_max=3, seed=42)
        for r in results:
            assert r.verdict in ABLATION_VARIANTS  or r.verdict in {
                "ignore", "monitor", "alert", "quarantine", "block"}


class TestHeuristicAgent:
    def test_decides_immediately(self, small_scenarios):
        agent = HeuristicAgent()
        results = evaluate_baseline(agent, small_scenarios, k_max=5, seed=42)
        for r in results:
            assert r.steps_taken == 1  # only DECIDE

    def test_uses_no_investigation_tools(self, small_scenarios):
        agent = HeuristicAgent()
        results = evaluate_baseline(agent, small_scenarios, k_max=5, seed=42)
        for r in results:
            assert r.tools_used == ["DECIDE"]


class TestHelperFunctions:
    def test_extract_entropy_deltas(self):
        text = "[t+5s] FILE path=C:/test.docx entropy_delta=+3.50 size_delta=+1024"
        deltas = _extract_entropy_deltas(text)
        assert len(deltas) == 1
        assert abs(deltas[0] - 3.5) < 0.01

    def test_extract_multiple_entropy_deltas(self):
        text = ("entropy_delta=+2.10 something "
                "entropy_delta=-0.30 more "
                "entropy_delta=+5.00")
        deltas = _extract_entropy_deltas(text)
        assert len(deltas) == 3

    def test_count_extension_changes(self):
        text = "ext_change=.docx->.locked ext_change=.xlsx->.encrypted"
        assert _count_extension_changes(text) == 2

    def test_no_extension_changes(self):
        assert _count_extension_changes("normal text") == 0


# ── Ablation tests ───────────────────────────────────────────────────


class TestAblationVariants:
    def test_all_variants_defined(self):
        assert "full" in ABLATION_VARIANTS
        assert "no_recall_memory" in ABLATION_VARIANTS
        assert "no_scan_directory" in ABLATION_VARIANTS
        assert "no_check_process" in ABLATION_VARIANTS
        assert "inspect_only" in ABLATION_VARIANTS

    def test_full_has_all_tools(self):
        assert set(ABLATION_VARIANTS["full"]) == {
            "inspect_file", "check_process", "scan_directory", "recall_memory"
        }

    def test_no_recall_excludes_recall(self):
        assert "recall_memory" not in ABLATION_VARIANTS["no_recall_memory"]
        assert "inspect_file" in ABLATION_VARIANTS["no_recall_memory"]

    def test_inspect_only(self):
        assert ABLATION_VARIANTS["inspect_only"] == ["inspect_file"]


class TestRunAblationSweep:
    def test_sweep_runs(self):
        def mock_eval_fn(scenarios, available_tools):
            return [
                EvalResult(
                    verdict="ignore",
                    is_ransomware_gt=False,
                    scenario_type="benign",
                    attack_phase=None,
                    steps_taken=1,
                    cumulative_cost=0.0,
                    k_max=5,
                    tools_used=["DECIDE"],
                )
                for _ in scenarios
            ]

        scenarios = [{"scenario_type": "benign", "observability": 0.5}] * 5
        results = run_ablation_sweep(mock_eval_fn, scenarios)
        assert len(results) == len(ABLATION_VARIANTS)

    def test_tracks_removed_tool_attempts(self):
        def mock_eval_fn(scenarios, available_tools):
            # Simulate agent trying to use recall_memory even when removed
            return [
                EvalResult(
                    verdict="ignore",
                    is_ransomware_gt=False,
                    scenario_type="benign",
                    attack_phase=None,
                    steps_taken=2,
                    cumulative_cost=-0.03,
                    k_max=5,
                    tools_used=["recall_memory", "DECIDE"],
                )
                for _ in scenarios
            ]

        scenarios = [{"scenario_type": "benign"}] * 3
        results = run_ablation_sweep(mock_eval_fn, scenarios)

        # The "no_recall_memory" variant should detect attempted use
        no_recall = next(r for r in results if r.variant_name == "no_recall_memory")
        assert "recall_memory" in no_recall.attempted_removed_tools
        assert no_recall.attempted_removed_tools["recall_memory"] == 3


class TestAblationSerialization:
    def test_save_and_load(self):
        result = AblationResult(
            variant_name="test",
            available_tools=["inspect_file"],
            detection=DetectionMetrics(tp=5, fp=1, tn=8, fn=2),
            efficiency=EfficiencyMetrics(16, 32, -0.64, 12),
            tool_usage={"inspect_file": 10, "DECIDE": 16},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save_ablation_results([result], path)

        with open(path) as f:
            loaded = json.load(f)

        assert len(loaded) == 1
        assert loaded[0]["variant"] == "test"
        assert loaded[0]["detection"]["tp"] == 5
        assert loaded[0]["detection"]["f1"] > 0

        Path(path).unlink()


class TestFormatAblationTable:
    def test_produces_markdown_table(self):
        result = AblationResult(
            variant_name="full",
            available_tools=["inspect_file", "check_process"],
            detection=DetectionMetrics(tp=5, fp=1, tn=8, fn=2),
            efficiency=EfficiencyMetrics(16, 32, -0.64, 12),
            tool_usage={"inspect_file": 10},
        )
        table = format_ablation_table([result])
        assert "| Variant |" in table
        assert "full" in table
        assert "|" in table
