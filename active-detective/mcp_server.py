"""MCP server exposing active-detective investigation tools.

Usage:
    SCENARIO_SEED=42 SCENARIO_TYPE=blitz python mcp_server.py

Environment variables:
    SCENARIO_SEED (required): Integer seed for reproducible episode generation
    SCENARIO_TYPE (required): benign, blitz, sleeper, exfil_first, semantic_shuffle
    OBSERVABILITY (optional): Float 0.0-1.0, default 0.7
    ATTACK_PROGRESS (optional): Float 0.0-1.0, default 0.5
    K_MAX (optional): Max investigation steps, default 5
    N_HISTORY (optional): Number of prior telemetry windows, default 2
    RESULTS_FILE (optional): Path to append JSONL results
    TOOL_SET (optional): Ablation variant name (default: full)
"""
from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from mcp_session import DetectiveSession
from evaluation.ablation import ABLATION_VARIANTS, ALL_TOOLS

# Read configuration from environment
_seed = int(os.environ.get("SCENARIO_SEED", "0"))
_scenario_type = os.environ.get("SCENARIO_TYPE", "benign")
_observability = float(os.environ.get("OBSERVABILITY", "0.7"))
_attack_progress = float(os.environ.get("ATTACK_PROGRESS", "0.5"))
_k_max = int(os.environ.get("K_MAX", "5"))
_n_history = int(os.environ.get("N_HISTORY", "2"))
_results_file = os.environ.get("RESULTS_FILE")
_tool_set = os.environ.get("TOOL_SET", "full")

# Resolve available tools from ablation variant
_no_tools = _tool_set == "no_tools"
if _no_tools:
    _available_tools: set[str] = set()
elif _tool_set in ABLATION_VARIANTS:
    _available_tools = set(ABLATION_VARIANTS[_tool_set])
else:
    _available_tools = set(ALL_TOOLS)

# Create session
_session = DetectiveSession(
    seed=_seed, scenario_type=_scenario_type, observability=_observability,
    attack_progress=_attack_progress, k_max=_k_max, n_history=_n_history,
    results_file=_results_file,
)

server = FastMCP("active-detective")


def _check_available(tool_name: str) -> str | None:
    if tool_name not in _available_tools:
        return (
            f"Tool '{tool_name}' is not available in this evaluation. "
            f"Available tools: {', '.join(sorted(_available_tools)) or 'none'}. "
            "Use DECIDE to render your verdict."
        )
    return None


@server.tool()
def get_telemetry() -> str:
    """Retrieve the host telemetry windows for analysis.
    Call this first to see the telemetry data before investigating.
    Returns current window plus prior history windows."""
    return _session.get_telemetry()


@server.tool()
def inspect_file(path: str) -> str:
    """Inspect a file's metadata: entropy, size, extension, content_type, modified timestamp.
    Cost: 1 investigation step."""
    err = _check_available("inspect_file")
    if err: return err
    return json.dumps(_session.execute_tool("inspect_file", {"path": path}), indent=2)


@server.tool()
def check_process(pid: int) -> str:
    """Check a running process: name, command line, parent, children.
    Cost: 1 investigation step."""
    err = _check_available("check_process")
    if err: return err
    return json.dumps(_session.execute_tool("check_process", {"pid": pid}), indent=2)


@server.tool()
def scan_directory(path: str) -> str:
    """List all files in a directory with metadata summaries. Most expensive tool.
    Cost: 1 investigation step."""
    err = _check_available("scan_directory")
    if err: return err
    return json.dumps(_session.execute_tool("scan_directory", {"path": path}), indent=2)


@server.tool()
def list_connections(filter_state: str | None = None) -> str:
    """List active network connections with optional state filter.
    filter_state: established, listening, closed, syn_sent (or omit for all).
    Cost: 1 investigation step."""
    err = _check_available("list_connections")
    if err: return err
    args = {}
    if filter_state: args["filter_state"] = filter_state
    return json.dumps(_session.execute_tool("list_connections", args), indent=2)


@server.tool()
def inspect_connection(conn_id: int) -> str:
    """Inspect a specific network connection in detail.
    Cost: 1 investigation step."""
    err = _check_available("inspect_connection")
    if err: return err
    return json.dumps(_session.execute_tool("inspect_connection", {"conn_id": conn_id}), indent=2)


@server.tool()
def query_registry(key_path: str) -> str:
    """Query a Windows registry key and its values.
    Useful for detecting persistence, Defender disabled, VSS deleted.
    Cost: 1 investigation step."""
    err = _check_available("query_registry")
    if err: return err
    return json.dumps(_session.execute_tool("query_registry", {"key_path": key_path}), indent=2)


@server.tool()
def list_process_handles(pid: int) -> str:
    """List a process's open file handles, loaded modules, and security context.
    Cost: 1 investigation step."""
    err = _check_available("list_process_handles")
    if err: return err
    return json.dumps(_session.execute_tool("list_process_handles", {"pid": pid}), indent=2)


@server.tool()
def query_event_log(source: str | None = None, event_id: int | None = None, since: str | None = None) -> str:
    """Query Windows Event Log entries.
    Filter by source (e.g. 'Windows Defender'), event_id, and/or timestamp.
    Cost: 1 investigation step."""
    err = _check_available("query_event_log")
    if err: return err
    args: dict = {}
    if source: args["source"] = source
    if event_id is not None: args["event_id"] = event_id
    if since: args["since"] = since
    return json.dumps(_session.execute_tool("query_event_log", args), indent=2)


@server.tool()
def read_file_sample(path: str, offset: int = 0, length: int = 256) -> str:
    """Read raw bytes from a file for forensic analysis.
    Returns hex dump, computed entropy, and magic bytes.
    Cost: 1 investigation step."""
    err = _check_available("read_file_sample")
    if err: return err
    return json.dumps(
        _session.execute_tool("read_file_sample", {"path": path, "offset": offset, "length": length}),
        indent=2,
    )


@server.tool()
def DECIDE(verdict: str, explanation: str) -> str:
    """Submit your final verdict. This ends the investigation.
    verdict: ignore, monitor, alert, quarantine, or block.
    explanation: Brief reasoning for your verdict."""
    return json.dumps(_session.decide(verdict, explanation), indent=2)


if __name__ == "__main__":
    server.run(transport="stdio")
