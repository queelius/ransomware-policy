"""Shared test fixtures for the active-detective test suite."""

import sys
from pathlib import Path

import pytest

# Ensure the active-detective package is importable from tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
