"""Tests for evaluation manifest generation."""
import json
import tempfile
from pathlib import Path
from scripts.generate_eval_manifest import generate_manifest


class TestManifestGeneration:
    def test_generates_correct_count(self):
        manifest = generate_manifest(n=50, seed=42)
        assert len(manifest) == 50

    def test_scenario_distribution_roughly_matches_mix(self):
        manifest = generate_manifest(n=200, seed=42)
        types = [m["scenario_type"] for m in manifest]
        benign_frac = types.count("benign") / len(types)
        assert 0.3 <= benign_frac <= 0.5

    def test_each_entry_has_required_fields(self):
        manifest = generate_manifest(n=10, seed=42)
        for entry in manifest:
            assert "seed" in entry
            assert "scenario_type" in entry
            assert "observability" in entry
            assert "attack_progress" in entry

    def test_benign_has_zero_progress(self):
        manifest = generate_manifest(n=100, seed=42)
        for entry in manifest:
            if entry["scenario_type"] == "benign":
                assert entry["attack_progress"] == 0.0

    def test_seeds_are_unique(self):
        manifest = generate_manifest(n=100, seed=42)
        seeds = [m["seed"] for m in manifest]
        assert len(seeds) == len(set(seeds))

    def test_deterministic_with_same_seed(self):
        m1 = generate_manifest(n=50, seed=42)
        m2 = generate_manifest(n=50, seed=42)
        assert m1 == m2
