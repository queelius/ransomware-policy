"""Tests for the embedding-similarity memory store."""

import pytest

from tools.memory import MemoryStore, recall_memory, _cosine_similarity

import numpy as np


class TestMemoryStore:
    @pytest.fixture
    def store(self):
        # Uses fallback embedder (no sentence-transformers needed)
        return MemoryStore(top_k=2)

    def test_empty_store_returns_empty(self, store):
        assert store.recall("anything") == []
        assert len(store) == 0

    def test_add_and_recall(self, store):
        store.add_window("File entropy spike on report.docx",
                         {"window_id": "t-1"})
        store.add_window("Normal office activity, chrome downloads",
                         {"window_id": "t-2"})
        store.add_window("Process svchost.exe spawned with encoded command",
                         {"window_id": "t-3"})

        assert len(store) == 3
        results = store.recall("entropy spike encryption", top_k=2)
        assert len(results) == 2
        assert all("similarity" in r for r in results)
        assert all("summary" in r for r in results)

    def test_similarity_ordering(self, store):
        store.add_window("entropy spike detected on multiple docx files",
                         {"window_id": "t-1"})
        store.add_window("completely unrelated network traffic to google",
                         {"window_id": "t-2"})

        results = store.recall("entropy spike encryption")
        # First result should have higher similarity
        if len(results) == 2:
            assert results[0]["similarity"] >= results[1]["similarity"]

    def test_clear(self, store):
        store.add_window("test window 1", {"window_id": "t-1"})
        store.add_window("test window 2", {"window_id": "t-2"})
        assert len(store) == 2

        store.clear()
        assert len(store) == 0
        assert store.recall("anything") == []

    def test_summary_truncation(self):
        store = MemoryStore(max_summary_chars=20)
        long_text = "A" * 100
        store.add_window(long_text, {"window_id": "t-1"})

        results = store.recall("A" * 50, top_k=1)
        assert len(results) == 1
        assert len(results[0]["summary"]) <= 23  # 20 + "..."
        assert results[0]["summary"].endswith("...")

    def test_top_k_limits(self, store):
        for i in range(10):
            store.add_window(f"window {i}", {"window_id": f"t-{i}"})

        results = store.recall("window", top_k=3)
        assert len(results) == 3


class TestRecallMemory:
    def test_tool_interface(self):
        store = MemoryStore()
        store.add_window("test window content", {"window_id": "t-1"})
        result = recall_memory(store, "test")
        assert "matches" in result
        assert isinstance(result["matches"], list)

    def test_empty_store(self):
        store = MemoryStore()
        result = recall_memory(store, "anything")
        assert result == {"matches": []}


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert _cosine_similarity(a, b) == 0.0
