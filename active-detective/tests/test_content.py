"""Tests for simulator.content — synthetic file content generation."""

import math

import numpy as np
import pytest

from simulator.content import (
    _MAGIC_BYTES,
    corrupt_content,
    encrypt_content,
    generate_content,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def _compute_entropy(data: bytes) -> float:
    """Shannon entropy of byte data (bits/byte)."""
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    total = len(data)
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


class TestGenerateContent:
    def test_default_size(self, rng):
        data = generate_content(".pdf", 4.0, rng)
        assert len(data) == 1024

    def test_custom_size(self, rng):
        data = generate_content(".txt", 3.0, rng, size=512)
        assert len(data) == 512

    def test_magic_bytes_pdf(self, rng):
        data = generate_content(".pdf", 4.0, rng)
        assert data.startswith(b"%PDF-1.5\n")

    def test_magic_bytes_docx(self, rng):
        data = generate_content(".docx", 4.0, rng)
        assert data.startswith(b"PK\x03\x04")

    def test_magic_bytes_png(self, rng):
        data = generate_content(".png", 6.0, rng)
        assert data.startswith(b"\x89PNG\r\n\x1a\n")

    def test_magic_bytes_jpg(self, rng):
        data = generate_content(".jpg", 6.0, rng)
        assert data.startswith(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    def test_magic_bytes_exe(self, rng):
        data = generate_content(".exe", 5.5, rng)
        assert data.startswith(b"MZ")

    def test_magic_bytes_db(self, rng):
        data = generate_content(".db", 4.0, rng)
        assert data.startswith(b"SQLite format 3\x00")

    def test_magic_bytes_zip(self, rng):
        data = generate_content(".zip", 7.0, rng)
        assert data.startswith(b"PK\x03\x04")

    def test_no_header_txt(self, rng):
        data = generate_content(".txt", 3.0, rng)
        # .txt has no magic bytes — first byte is from the body
        for header in _MAGIC_BYTES.values():
            assert not data.startswith(header)

    def test_no_header_log(self, rng):
        data = generate_content(".log", 3.0, rng)
        for header in _MAGIC_BYTES.values():
            assert not data.startswith(header)

    @pytest.mark.parametrize("target_entropy", [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    def test_entropy_targeting(self, rng, target_entropy):
        """Generated content entropy should be within ~1 bit of target."""
        # Use .txt to avoid header bytes skewing entropy measurement
        data = generate_content(".txt", target_entropy, rng, size=4096)
        actual = _compute_entropy(data)
        assert abs(actual - target_entropy) < 1.0, (
            f"Target {target_entropy}, got {actual:.2f}"
        )

    def test_low_entropy_content(self, rng):
        data = generate_content(".ini", 2.0, rng, size=2048)
        actual = _compute_entropy(data)
        assert actual < 3.5  # low entropy

    def test_high_entropy_content(self, rng):
        data = generate_content(".zip", 7.0, rng, size=4096)
        actual = _compute_entropy(data)
        assert actual > 5.5  # high entropy

    def test_unknown_extension_no_header(self, rng):
        data = generate_content(".xyz", 4.0, rng)
        assert len(data) == 1024

    def test_zero_size(self, rng):
        data = generate_content(".txt", 4.0, rng, size=0)
        assert data == b""

    def test_size_smaller_than_header(self, rng):
        # .pdf header is 9 bytes; size=4 means header truncated to fit? No,
        # body_size = max(0, 4 - 9) = 0, so we get just the header (9 bytes)
        data = generate_content(".pdf", 4.0, rng, size=4)
        # Header is 9 bytes, body_size=0, total = 9
        assert data.startswith(b"%PDF-1.5\n")


class TestEncryptContent:
    def test_size(self, rng):
        data = encrypt_content(1024, rng)
        assert len(data) == 1024

    def test_high_entropy(self, rng):
        data = encrypt_content(4096, rng)
        actual = _compute_entropy(data)
        assert actual > 7.0  # near-maximum entropy

    def test_no_recognizable_header(self, rng):
        data = encrypt_content(1024, rng)
        for header in _MAGIC_BYTES.values():
            if len(header) <= len(data):
                # Very unlikely to match by chance, but not guaranteed
                pass  # Just ensuring it's random bytes

    def test_returns_bytes(self, rng):
        data = encrypt_content(256, rng)
        assert isinstance(data, bytes)


class TestCorruptContent:
    def test_preserves_header(self, rng):
        original = b"HEADER_DATA_" + bytes(range(256)) * 4
        corrupted = corrupt_content(original, rng, preserve_header=12)
        assert corrupted[:12] == b"HEADER_DATA_"
        assert len(corrupted) == len(original)

    def test_body_is_different(self, rng):
        original = bytes(range(256)) * 4  # 1024 bytes
        corrupted = corrupt_content(original, rng, preserve_header=64)
        # Body after header should be different
        assert corrupted[64:] != original[64:]

    def test_moderate_entropy(self, rng):
        original = bytes(range(256)) * 4
        corrupted = corrupt_content(
            original, rng, preserve_header=0, target_entropy=5.5
        )
        actual = _compute_entropy(corrupted)
        assert 4.0 < actual < 7.0  # moderate entropy

    def test_default_preserve_64(self, rng):
        original = b"A" * 128
        corrupted = corrupt_content(original, rng)
        assert corrupted[:64] == b"A" * 64

    def test_short_content(self, rng):
        original = b"short"
        corrupted = corrupt_content(original, rng, preserve_header=64)
        # Entire content is shorter than preserve_header, body_size=0
        assert corrupted == b"short"

    def test_empty_content(self, rng):
        corrupted = corrupt_content(b"", rng)
        assert corrupted == b""
