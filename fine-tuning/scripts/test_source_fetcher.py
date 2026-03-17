#!/usr/bin/env python3
"""Tests for source_fetcher.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from source_fetcher import (
    SourceConfig,
    compute_content_hash,
    load_sources,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SOURCES = {
    "sources": [
        {
            "key": "train_only",
            "name": "Training Source",
            "url": "https://example.com/train",
            "type": "knowledge_base",
            "use": "training",
            "license": "MIT",
            "topics": ["ransomware_ttps"],
            "format": "json",
            "notes": "training source notes",
        },
        {
            "key": "eval_only",
            "name": "Evaluation Source",
            "url": "https://example.com/eval",
            "type": "real_traces",
            "use": "evaluation",
            "license": "MIT",
            "topics": ["malware_samples"],
            "format": "json",
            "notes": "eval only",
        },
        {
            "key": "both_use",
            "name": "Both Source",
            "url": "active-detective/simulator/",
            "type": "synthetic",
            "use": "both",
            "license": "Project-internal",
            "topics": ["forensic_analysis"],
            "format": "json",
            "notes": "used for both",
        },
    ]
}


@pytest.fixture()
def sources_yaml(tmp_path: Path) -> Path:
    """Write a minimal sources.yaml to a temp dir and return its path."""
    p = tmp_path / "sources.yaml"
    with open(p, "w") as fh:
        yaml.dump(SAMPLE_SOURCES, fh)
    return p


# ---------------------------------------------------------------------------
# Tests: load_sources
# ---------------------------------------------------------------------------


class TestLoadSources:
    def test_filter_training(self, sources_yaml: Path) -> None:
        """filter_use='training' returns entries with use=training or use=both."""
        result = load_sources(sources_yaml, filter_use="training")
        keys = {s.key for s in result}
        assert keys == {"train_only", "both_use"}

    def test_filter_evaluation(self, sources_yaml: Path) -> None:
        """filter_use='evaluation' returns entries with use=evaluation or use=both."""
        result = load_sources(sources_yaml, filter_use="evaluation")
        keys = {s.key for s in result}
        assert keys == {"eval_only", "both_use"}

    def test_returns_source_config_instances(self, sources_yaml: Path) -> None:
        result = load_sources(sources_yaml, filter_use="training")
        for s in result:
            assert isinstance(s, SourceConfig)

    def test_fields_populated(self, sources_yaml: Path) -> None:
        result = load_sources(sources_yaml, filter_use="training")
        train = [s for s in result if s.key == "train_only"][0]
        assert train.name == "Training Source"
        assert train.url == "https://example.com/train"
        assert train.source_type == "knowledge_base"
        assert train.use == "training"
        assert train.license == "MIT"
        assert train.topics == ["ransomware_ttps"]
        assert train.fmt == "json"
        assert "training source notes" in train.notes

    def test_empty_sources(self, tmp_path: Path) -> None:
        """An empty sources list returns an empty list."""
        p = tmp_path / "empty.yaml"
        with open(p, "w") as fh:
            yaml.dump({"sources": []}, fh)
        assert load_sources(p) == []

    def test_no_matching_use(self, sources_yaml: Path) -> None:
        """A filter_use with no direct match still returns 'both' entries."""
        result = load_sources(sources_yaml, filter_use="nonexistent")
        keys = {s.key for s in result}
        # Only the 'both' entry matches; no direct 'nonexistent' use exists.
        assert keys == {"both_use"}

    def test_truly_empty_match(self, tmp_path: Path) -> None:
        """When no entry has use=both or the requested use, returns empty."""
        data = {
            "sources": [
                {
                    "key": "train_only",
                    "name": "Train",
                    "url": "https://example.com",
                    "type": "knowledge_base",
                    "use": "training",
                    "license": "MIT",
                    "format": "json",
                }
            ]
        }
        p = tmp_path / "src.yaml"
        with open(p, "w") as fh:
            yaml.dump(data, fh)
        result = load_sources(p, filter_use="evaluation")
        assert result == []


# ---------------------------------------------------------------------------
# Tests: compute_content_hash
# ---------------------------------------------------------------------------


class TestComputeContentHash:
    def test_deterministic(self) -> None:
        """Same input always produces the same hash."""
        data = b"ransomware detection telemetry"
        h1 = compute_content_hash(data)
        h2 = compute_content_hash(data)
        assert h1 == h2

    def test_hex_string(self) -> None:
        """Returns a 64-character hex string (SHA-256)."""
        h = compute_content_hash(b"test data")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_inputs_different_hashes(self) -> None:
        h1 = compute_content_hash(b"input A")
        h2 = compute_content_hash(b"input B")
        assert h1 != h2

    def test_known_value(self) -> None:
        """Verify against a known SHA-256."""
        import hashlib

        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert compute_content_hash(data) == expected

    def test_empty_bytes(self) -> None:
        """Empty bytes still produces a valid hash."""
        h = compute_content_hash(b"")
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Tests: write_manifest
# ---------------------------------------------------------------------------


class TestWriteManifest:
    def test_produces_valid_json(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        entries = [
            {
                "key": "test_source",
                "name": "Test",
                "fetched_at": "2026-03-17T00:00:00+00:00",
                "files": [
                    {"file": "a.json", "sha256": "abc123", "bytes": 42}
                ],
            }
        ]
        write_manifest(manifest_path, entries)

        with open(manifest_path) as fh:
            data = json.load(fh)

        assert "generated_at" in data
        assert "sources" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["key"] == "test_source"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "nested" / "dir" / "manifest.json"
        write_manifest(manifest_path, [])
        assert manifest_path.exists()

    def test_empty_entries(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        write_manifest(manifest_path, [])
        with open(manifest_path) as fh:
            data = json.load(fh)
        assert data["sources"] == []

    def test_multiple_entries(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        entries = [
            {"key": f"src_{i}", "name": f"Source {i}", "fetched_at": "now", "files": []}
            for i in range(5)
        ]
        write_manifest(manifest_path, entries)
        with open(manifest_path) as fh:
            data = json.load(fh)
        assert len(data["sources"]) == 5

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        write_manifest(manifest_path, [{"key": "old"}])
        write_manifest(manifest_path, [{"key": "new"}])
        with open(manifest_path) as fh:
            data = json.load(fh)
        assert data["sources"] == [{"key": "new"}]
