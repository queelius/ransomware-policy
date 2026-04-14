"""Agent investigation tools: inspect_file, check_process, scan_directory,
plus dispatch for all tools via execute_tool.

Each tool takes simulator state and returns a JSON-serializable dict.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

from simulator.host import HostState
from simulator.models import Verdict
from simulator.registry import FileRegistry, ProcessTable


class ToolName(str, Enum):
    """Canonical tool identifiers. String values match wire format."""

    INSPECT_FILE = "inspect_file"
    CHECK_PROCESS = "check_process"
    SCAN_DIRECTORY = "scan_directory"
    LIST_CONNECTIONS = "list_connections"
    INSPECT_CONNECTION = "inspect_connection"
    QUERY_REGISTRY = "query_registry"
    LIST_PROCESS_HANDLES = "list_process_handles"
    QUERY_EVENT_LOG = "query_event_log"
    READ_FILE_SAMPLE = "read_file_sample"
    DECIDE = "DECIDE"


# Tool name → cost (negative values, subtracted from reward)
TOOL_COSTS: dict[str, float] = {
    ToolName.INSPECT_FILE.value: -0.02,
    ToolName.CHECK_PROCESS.value: -0.02,
    ToolName.SCAN_DIRECTORY.value: -0.05,
    ToolName.LIST_CONNECTIONS.value: -0.03,
    ToolName.INSPECT_CONNECTION.value: -0.03,
    ToolName.QUERY_REGISTRY.value: -0.03,
    ToolName.LIST_PROCESS_HANDLES.value: -0.03,
    ToolName.QUERY_EVENT_LOG.value: -0.04,
    ToolName.READ_FILE_SAMPLE.value: -0.04,
    ToolName.DECIDE.value: 0.0,
}


# Parameter names by positional order (for parser.py fallback)
TOOL_PARAMS: dict[str, tuple[str, ...]] = {
    ToolName.INSPECT_FILE.value: ("path",),
    ToolName.CHECK_PROCESS.value: ("pid",),
    ToolName.SCAN_DIRECTORY.value: ("path",),
    ToolName.LIST_CONNECTIONS.value: ("filter",),
    ToolName.INSPECT_CONNECTION.value: ("conn_id",),
    ToolName.QUERY_REGISTRY.value: ("key_path",),
    ToolName.LIST_PROCESS_HANDLES.value: ("pid",),
    ToolName.QUERY_EVENT_LOG.value: ("source", "event_id", "since"),
    ToolName.READ_FILE_SAMPLE.value: ("path", "offset", "length"),
    ToolName.DECIDE.value: ("verdict", "explanation"),
}


# Derived from Verdict enum, single source of truth
VALID_VERDICTS: set[str] = {v.value for v in Verdict}


def inspect_file(registry: FileRegistry, path: str) -> dict:
    """Inspect a file's metadata. Cost: -0.02.

    Returns entropy, size, extension, modified_ts, content_type.
    """
    record = registry.get_file(path)
    if record is None:
        return {"error": f"File not found: {path}"}

    return {
        "entropy": round(record.entropy, 2),
        "size": record.size,
        "extension": record.extension,
        "modified_ts": record.modified_at.isoformat(),
        "content_type": record.content_type.value,
    }


def check_process(ptable: ProcessTable, pid: int) -> dict:
    """Check a process's details. Cost: -0.02.

    Returns name, command_line, parent process name, child PIDs.
    """
    record = ptable.get_process(pid)
    if record is None:
        return {"error": f"Process not found: pid={pid}"}

    parent = ptable.get_process(record.parent_pid)
    parent_name = parent.name if parent else "unknown"

    return {
        "name": record.name,
        "command_line": record.command_line,
        "parent": parent_name,
        "child_pids": record.child_pids,
    }


def scan_directory(registry: FileRegistry, path: str) -> dict:
    """List files in a directory with metadata summaries. Cost: -0.05.

    More expensive because it reveals more information.
    """
    files = registry.list_directory(path)
    if not files:
        return {"files": [], "note": f"No files found in {path}"}

    return {
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "entropy": round(f.entropy, 2),
                "extension": f.extension,
            }
            for f in files
        ]
    }


# ── Dispatch table ───────────────────────────────────────────────────
#
# Each entry maps a ToolName to a callable that takes (host, args) and
# returns the result dict. Cost is looked up via TOOL_COSTS separately.


def _do_inspect_file(host: HostState, args: dict) -> dict:
    return inspect_file(host.files, args.get("path", ""))


def _do_check_process(host: HostState, args: dict) -> dict:
    pid = args.get("pid")
    if pid is None:
        return {"error": "Missing 'pid' argument"}
    return check_process(host.processes, int(pid))


def _do_scan_directory(host: HostState, args: dict) -> dict:
    return scan_directory(host.files, args.get("path", ""))


def _do_list_connections(host: HostState, args: dict) -> dict:
    from tools.network_tools import list_connections
    filter_state = args.get("filter", args.get("filter_state"))
    return list_connections(host.connections, filter_state)


def _do_inspect_connection(host: HostState, args: dict) -> dict:
    from tools.network_tools import inspect_connection
    conn_id = args.get("conn_id")
    if conn_id is None:
        return {"error": "Missing 'conn_id' argument"}
    return inspect_connection(host.connections, int(conn_id))


def _do_query_registry(host: HostState, args: dict) -> dict:
    from tools.forensic_tools import query_registry
    return query_registry(host.registry, args.get("key_path", ""))


def _do_list_process_handles(host: HostState, args: dict) -> dict:
    from tools.forensic_tools import list_process_handles
    pid = args.get("pid")
    if pid is None:
        return {"error": "Missing 'pid' argument"}
    return list_process_handles(host.processes, int(pid))


def _do_query_event_log(host: HostState, args: dict) -> dict:
    from tools.forensic_tools import query_event_log
    return query_event_log(
        host.event_log,
        source=args.get("source"),
        event_id=args.get("event_id"),
        since=args.get("since"),
    )


def _do_read_file_sample(host: HostState, args: dict) -> dict:
    from tools.forensic_tools import read_file_sample
    return read_file_sample(
        host.files,
        args.get("path", ""),
        int(args.get("offset", 0)),
        int(args.get("length", 256)),
    )


def _do_decide(host: HostState, args: dict) -> dict:
    verdict = args.get("verdict", "")
    explanation = args.get("explanation", "")
    if verdict not in VALID_VERDICTS:
        return {
            "error": f"Invalid verdict '{verdict}'. "
                     f"Must be one of: {', '.join(sorted(VALID_VERDICTS))}"
        }
    return {"verdict": verdict, "explanation": explanation}


_DISPATCH: dict[str, Callable[[HostState, dict], dict]] = {
    ToolName.INSPECT_FILE.value: _do_inspect_file,
    ToolName.CHECK_PROCESS.value: _do_check_process,
    ToolName.SCAN_DIRECTORY.value: _do_scan_directory,
    ToolName.LIST_CONNECTIONS.value: _do_list_connections,
    ToolName.INSPECT_CONNECTION.value: _do_inspect_connection,
    ToolName.QUERY_REGISTRY.value: _do_query_registry,
    ToolName.LIST_PROCESS_HANDLES.value: _do_list_process_handles,
    ToolName.QUERY_EVENT_LOG.value: _do_query_event_log,
    ToolName.READ_FILE_SAMPLE.value: _do_read_file_sample,
    ToolName.DECIDE.value: _do_decide,
}


def execute_tool(
    tool_name: str,
    args: dict,
    host: HostState,
) -> tuple[dict, float]:
    """Execute a tool by name and return (result_dict, cost).

    Unknown tools return an error with zero cost. Tools that error
    (missing arg, invalid verdict, etc.) still charge the tool cost
    except DECIDE, which is always free.
    """
    handler = _DISPATCH.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool: {tool_name}"}, 0.0

    result = handler(host, args)

    # DECIDE charges no cost regardless of validity
    if tool_name == ToolName.DECIDE.value:
        return result, 0.0

    return result, TOOL_COSTS.get(tool_name, 0.0)
