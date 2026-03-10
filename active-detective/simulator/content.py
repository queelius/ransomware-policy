"""Synthetic file content generation for simulated host filesystem.

Generates plausible ~1KB byte contents with realistic magic bytes headers
and controlled Shannon entropy. This enables the read_file_sample tool to
return meaningful forensic data (hex dumps, entropy, magic bytes) instead
of "File contents not available."
"""

from __future__ import annotations

import numpy as np


# ── Magic bytes by file extension ────────────────────────────────────

_MAGIC_BYTES: dict[str, bytes] = {
    # OOXML / ZIP container
    ".docx": b"PK\x03\x04",
    ".xlsx": b"PK\x03\x04",
    ".pptx": b"PK\x03\x04",
    ".zip":  b"PK\x03\x04",
    # PDF
    ".pdf":  b"%PDF-1.5\n",
    # Images
    ".jpg":  b"\xff\xd8\xff\xe0\x00\x10JFIF",
    ".png":  b"\x89PNG\r\n\x1a\n",
    ".bmp":  b"BM\x00\x00\x00\x00\x00\x00",
    # Executables
    ".exe":  b"MZ\x90\x00\x03\x00\x00\x00",
    # Database
    ".db":   b"SQLite format 3\x00",
    # Text formats (structural headers, not magic bytes per se)
    ".xml":  b"<?xml version=\"1.0\"?>\n",
    ".ini":  b"[Settings]\n",
}

# Extensions that get no header — pure body content
_NO_HEADER_EXTS = frozenset({".txt", ".tmp", ".log", ".csv"})


def generate_content(
    extension: str,
    target_entropy: float,
    rng: np.random.RandomState,
    size: int = 1024,
) -> bytes:
    """Generate synthetic file content with realistic header and target entropy.

    Parameters
    ----------
    extension : str
        File extension (e.g. ".pdf") to determine magic bytes header.
    target_entropy : float
        Desired Shannon entropy in bits/byte (0-8).
    rng : np.random.RandomState
        Random state for reproducibility.
    size : int
        Total content size in bytes.

    Returns
    -------
    bytes
        Synthetic file content.
    """
    header = _MAGIC_BYTES.get(extension, b"")
    if extension in _NO_HEADER_EXTS:
        header = b""

    body_size = max(0, size - len(header))
    body = _generate_entropy_targeted_bytes(target_entropy, body_size, rng)
    return header + body


def encrypt_content(
    size: int,
    rng: np.random.RandomState,
) -> bytes:
    """Generate encrypted (near-max entropy) content — no recognizable headers.

    Returns uniformly random bytes (~8.0 bits/byte entropy).
    """
    return bytes(rng.randint(0, 256, size, dtype=np.uint8))


def corrupt_content(
    original: bytes,
    rng: np.random.RandomState,
    preserve_header: int = 64,
    target_entropy: float = 5.5,
) -> bytes:
    """Corrupt file content while partially preserving the header.

    Used by semantic_shuffle to simulate content manipulation that
    doesn't look like encryption (moderate entropy, header may survive).

    Parameters
    ----------
    original : bytes
        Original file content.
    rng : np.random.RandomState
        Random state for reproducibility.
    preserve_header : int
        Number of leading bytes to preserve.
    target_entropy : float
        Target entropy for the corrupted body (default 5.0-6.0 range).
    """
    preserved = original[:preserve_header]
    body_size = max(0, len(original) - preserve_header)
    # Moderate entropy — between normal text and encrypted
    body = _generate_entropy_targeted_bytes(target_entropy, body_size, rng)
    return preserved + body


def _generate_entropy_targeted_bytes(
    target_entropy: float,
    size: int,
    rng: np.random.RandomState,
) -> bytes:
    """Generate random bytes with approximately the target Shannon entropy.

    Uses uniform sampling over k = clamp(round(2^H), 2, 256) symbols.
    log2(k) ≈ H bits/byte, exact for integer H.
    """
    if size == 0:
        return b""

    target_entropy = max(0.0, min(8.0, target_entropy))
    k = int(round(2 ** target_entropy))
    k = max(2, min(256, k))

    # Sample uniformly from k distinct byte values
    symbols = rng.randint(0, k, size, dtype=np.uint8)
    return bytes(symbols)
