"""Embedding-similarity memory store and recall_memory tool.

Stores past telemetry windows as embeddings for retrieval.
Uses cosine similarity over numpy arrays (no FAISS dependency).
The embedding model is lazy-loaded on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MemoryEntry:
    """A stored telemetry window with its embedding."""

    window_text: str
    summary: str
    metadata: dict
    embedding: np.ndarray


class MemoryStore:
    """Embedding-similarity store over past telemetry windows.

    Exposed to the agent as the ``recall_memory`` tool.
    The RL training discovers *when* historical context is worth the cost.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        max_summary_chars: int = 300,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self.max_summary_chars = max_summary_chars
        self._entries: list[MemoryEntry] = []
        self._model = None  # lazy-loaded

    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                # Fallback: random embeddings for testing without GPU
                self._model = _FallbackEmbedder()
        return self._model

    def _embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        if isinstance(model, _FallbackEmbedder):
            return model.encode(text)
        return model.encode(text, convert_to_numpy=True)

    def add_window(self, window_text: str, metadata: dict | None = None) -> None:
        """Embed and store a telemetry window."""
        metadata = metadata or {}
        summary = window_text[:self.max_summary_chars]
        if len(window_text) > self.max_summary_chars:
            summary += "..."
        embedding = self._embed(window_text)
        self._entries.append(MemoryEntry(
            window_text=window_text,
            summary=summary,
            metadata=metadata,
            embedding=embedding,
        ))

    def recall(self, query: str, top_k: int | None = None) -> list[dict]:
        """Find top-k most similar past windows to the query.

        Returns list of {"window": str, "summary": str, "similarity": float}.
        """
        if not self._entries:
            return []

        top_k = top_k or self.top_k
        query_emb = self._embed(query)

        # Cosine similarity against all stored embeddings
        similarities = []
        for entry in self._entries:
            sim = _cosine_similarity(query_emb, entry.embedding)
            similarities.append((sim, entry))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, entry in similarities[:top_k]:
            results.append({
                "window": entry.metadata.get("window_id", "unknown"),
                "summary": entry.summary,
                "similarity": round(float(sim), 3),
            })

        return results

    def clear(self) -> None:
        """Reset the store (called on env reset between episodes)."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def recall_memory(memory_store: MemoryStore, query: str) -> dict:
    """The recall_memory tool function. Cost: -0.03.

    Returns top-k relevant historical windows.
    """
    matches = memory_store.recall(query)
    return {"matches": matches}


# ── Helpers ──────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class _FallbackEmbedder:
    """Deterministic hash-based embedder for testing without sentence-transformers."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, text: str, **kwargs) -> np.ndarray:
        # Use hash of text as seed for reproducible pseudo-embeddings
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
