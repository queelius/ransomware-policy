"""Smoke tests for MCP server module imports and tool registration."""
import os
import pytest

try:
    import mcp as _mcp
    _has_mcp = True
except ImportError:
    _has_mcp = False

pytestmark = pytest.mark.skipif(not _has_mcp, reason="mcp package not installed")


def test_mcp_server_imports():
    os.environ["SCENARIO_SEED"] = "42"
    os.environ["SCENARIO_TYPE"] = "blitz"
    os.environ["OBSERVABILITY"] = "0.7"
    os.environ["ATTACK_PROGRESS"] = "0.5"
    try:
        import mcp_server
        assert hasattr(mcp_server, "server")
        assert hasattr(mcp_server, "_session")
    finally:
        for key in ["SCENARIO_SEED", "SCENARIO_TYPE", "OBSERVABILITY", "ATTACK_PROGRESS"]:
            os.environ.pop(key, None)
