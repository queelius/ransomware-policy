"""Tests for batch scenario generator."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from simulator.models import ScenarioType
from training.scenarios import (
    DEFAULT_OBSERVABILITY_LEVELS,
    DEFAULT_SCENARIO_MIX,
    ScenarioBatch,
    generate_training_scenarios,
    load_scenarios,
    save_scenarios,
)


class TestGenerateTrainingScenarios:
    def test_generates_correct_count(self):
        batch = generate_training_scenarios(n_episodes=50, seed=42)
        assert len(batch.episodes) == 50

    def test_reproducible(self):
        b1 = generate_training_scenarios(n_episodes=20, seed=42)
        b2 = generate_training_scenarios(n_episodes=20, seed=42)
        for e1, e2 in zip(b1.episodes, b2.episodes):
            assert e1.input_text == e2.input_text
            assert e1.scenario_type == e2.scenario_type
            assert e1.observability == e2.observability

    def test_different_seeds_differ(self):
        b1 = generate_training_scenarios(n_episodes=10, seed=42)
        b2 = generate_training_scenarios(n_episodes=10, seed=99)
        # Very unlikely all 10 match
        texts_match = sum(
            e1.input_text == e2.input_text
            for e1, e2 in zip(b1.episodes, b2.episodes)
        )
        assert texts_match < 10

    def test_scenario_mix_proportions(self):
        """Generated mix should be within tolerance of target proportions."""
        batch = generate_training_scenarios(n_episodes=500, seed=42)
        counts = batch.scenario_counts
        total = len(batch.episodes)

        for scenario_type, target_prop in DEFAULT_SCENARIO_MIX.items():
            actual_prop = counts.get(scenario_type.value, 0) / total
            # Allow ±8% tolerance for 500 episodes
            assert abs(actual_prop - target_prop) < 0.08, (
                f"{scenario_type.value}: expected ~{target_prop:.2f}, "
                f"got {actual_prop:.2f}"
            )

    def test_observability_distribution(self):
        """All configured observability levels should appear."""
        batch = generate_training_scenarios(n_episodes=100, seed=42)
        obs_values = {ep.observability for ep in batch.episodes}
        for level in DEFAULT_OBSERVABILITY_LEVELS:
            assert level in obs_values

    def test_custom_scenario_mix(self):
        mix = {
            ScenarioType.BENIGN: 0.5,
            ScenarioType.BLITZ: 0.5,
        }
        batch = generate_training_scenarios(
            n_episodes=100, scenario_mix=mix, seed=42)
        counts = batch.scenario_counts
        assert set(counts.keys()) == {"benign", "blitz"}

    def test_custom_observability_levels(self):
        batch = generate_training_scenarios(
            n_episodes=50,
            observability_levels=[0.1, 0.9],
            seed=42,
        )
        obs_values = {ep.observability for ep in batch.episodes}
        assert obs_values <= {0.1, 0.9}

    def test_episodes_have_valid_ground_truth(self):
        batch = generate_training_scenarios(n_episodes=20, seed=42)
        for ep in batch.episodes:
            gt = ep.ground_truth
            if ep.scenario_type == ScenarioType.BENIGN:
                assert not gt.is_ransomware
            else:
                assert gt.is_ransomware

    def test_episodes_have_nonempty_text(self):
        batch = generate_training_scenarios(n_episodes=20, seed=42)
        for ep in batch.episodes:
            assert len(ep.input_text) > 0


class TestScenarioBatch:
    def test_scenario_counts(self):
        batch = generate_training_scenarios(n_episodes=50, seed=42)
        counts = batch.scenario_counts
        assert sum(counts.values()) == 50

    def test_observability_stats(self):
        batch = generate_training_scenarios(n_episodes=50, seed=42)
        stats = batch.observability_stats
        assert 0.0 < stats["min"] <= stats["max"] <= 1.0
        assert stats["min"] <= stats["mean"] <= stats["max"]


class TestSaveLoad:
    def test_save_and_load_roundtrip(self):
        batch = generate_training_scenarios(n_episodes=10, seed=42)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name

        save_scenarios(batch, path)
        records = load_scenarios(path)

        assert len(records) == 10
        for rec, ep in zip(records, batch.episodes):
            assert rec["scenario_type"] == ep.scenario_type.value
            assert rec["is_ransomware"] == ep.ground_truth.is_ransomware
            assert rec["observability"] == ep.observability
            assert rec["input_text"] == ep.input_text

        Path(path).unlink()

    def test_save_creates_parent_dirs(self):
        batch = generate_training_scenarios(n_episodes=3, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "data.jsonl"
            save_scenarios(batch, path)
            assert path.exists()

    def test_jsonl_format(self):
        batch = generate_training_scenarios(n_episodes=5, seed=42)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name

        save_scenarios(batch, path)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 5
        for line in lines:
            record = json.loads(line)
            assert "input_text" in record
            assert "scenario_type" in record
            assert "is_ransomware" in record

        Path(path).unlink()
